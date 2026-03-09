# start zipvoice/utils/common.py
"""Common utility classes and functions for ZipVoice training and inference."""

import collections
import os
import socket
import subprocess
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson
import structlog
import torch
from packaging import version
from torch import distributed as dist
from torch import nn
from torch.amp import GradScaler  # noqa: F401  # re-exported for checkpoint.py
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

Pathlike = str | Path

log = structlog.get_logger()


class AttributeDict(dict):
    """A dict subclass that allows attribute-style access to its keys."""

    def __getattr__(self, key):
        """Return the value for key, or raise AttributeError if missing."""
        if key in self:
            return self[key]
        msg = f"No such attribute '{key}'"
        raise AttributeError(msg)

    def __setattr__(self, key, value):
        """Set key to value in the underlying dict."""
        self[key] = value

    def __delattr__(self, key):
        """Delete key from the underlying dict, or raise AttributeError."""
        if key in self:
            del self[key]
            return
        msg = f"No such attribute '{key}'"
        raise AttributeError(msg)

    def __str__(self, indent: int = 2):
        """Return a JSON-formatted string of the dict."""
        tmp = {}
        for k, v in self.items():
            # PosixPath is ont JSON serializable
            if isinstance(v, (Path, torch.device, torch.dtype)):
                v = str(v)
            tmp[k] = v
        return orjson.dumps(tmp, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode()


class MetricsTracker(collections.defaultdict):
    """A defaultdict subclass for tracking training metrics keyed by name."""

    def __init__(self):
        """Initialize with int default factory so missing keys return 0."""
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super().__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        """Return element-wise sum of two MetricsTrackers."""
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            if v - v == 0:
                ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        """Return a new MetricsTracker with all values scaled by alpha."""
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        """Return a human-readable summary string of normalized metrics."""
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = f"{v:.4g}"
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    msg = f"Unexpected key: {k}"
                    raise ValueError(msg)
        frames = "{:.2f}".format(self["frames"])
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "{:.2f}".format(self["utterances"])
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> list[tuple[str, float]]:
        """Return a list of pairs like [('ctc_loss', 0.1), ('att_loss', 0.07)].

        Values are normalized by frame or utterance count.
        """
        num_frames = self.get("frames", 1)
        num_utterances = self.get("utterances", 1)
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """Reduce metrics across all distributed processes using all-reduce.

        Args:
            device: The device to use for the reduction tensor.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist(), strict=False):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)


@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    """Context manager that wraps torch.amp.autocast with version compatibility.

    Fixes FutureWarning: ``torch.cuda.amp.autocast(args...)`` is deprecated.
    Please use ``torch.amp.autocast('cuda', args...)`` instead.
    """
    if version.parse(torch.__version__) >= version.parse("2.3.0"):
        # Use new unified API
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:
        # Suppress deprecation warning and use old CUDA-specific autocast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(**kwargs):
                yield


def create_grad_scaler(device="cuda", **kwargs):
    """Create a GradScaler compatible with both torch < 2.3.0 and >= 2.3.0.

    Accepts all kwargs like: enabled, init_scale, growth_factor, etc.
    Suppresses the FutureWarning about deprecated ``torch.cuda.amp.GradScaler``.
    """
    if version.parse(torch.__version__) >= version.parse("2.3.0"):
        from torch.amp import GradScaler

        return GradScaler(device=device, **kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return torch.cuda.amp.GradScaler(**kwargs)


def setup_dist(
    rank=None,
    world_size=None,
    master_port=None,
    use_ddp_launch=False,
    master_addr=None,
):
    """Rank and world_size are used only if use_ddp_launch is False."""
    if "MASTER_ADDR" not in os.environ:
        # NOTE: Required for PyTorch DDP; env vars are the only interface
        os.environ["MASTER_ADDR"] = "localhost" if master_addr is None else str(master_addr)

    if "MASTER_PORT" not in os.environ:
        # NOTE: Required for PyTorch DDP; env vars are the only interface
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")


def cleanup_dist():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def prepare_input(  # noqa: PLR0913
    params: AttributeDict,
    batch: dict,
    device: torch.device,
    return_tokens: bool = True,
    return_feature: bool = True,
    return_audio: bool = False,
):
    """Parse the features and targets of the current batch.

    Args:
      params:
        It is returned by :func:`get_params`.
      batch:
        It is the return value from iterating
        ``lhotse.dataset.K2SpeechRecognitionDataset``.
      device:
        The device of Tensor.
      return_tokens:
        Whether to include token tensors in the output.
      return_feature:
        Whether to include feature tensors in the output.
      return_audio:
        Whether to include raw audio tensors in the output.
    """
    return_list = []

    if return_tokens:
        return_list += [batch["tokens"]]

    if return_feature:
        features = batch["features"].to(device)
        features_lens = batch["features_lens"].to(device)
        return_list += [features * params.feat_scale, features_lens]

    if return_audio:
        return_list += [batch["audio"], batch["audio_lens"]]

    return return_list


def prepare_avg_tokens_durations(features_lens, tokens_lens):
    """Compute average token durations for each utterance.

    Args:
        features_lens: Tensor of feature (frame) lengths per utterance.
        tokens_lens: Tensor of token counts per utterance.

    Returns:
        List of per-token duration lists.
    """
    tokens_durations = []
    for i in range(len(features_lens)):
        utt_duration = features_lens[i]
        avg_token_duration = utt_duration // tokens_lens[i]
        tokens_durations.append([avg_token_duration] * tokens_lens[i])
    return tokens_durations


def pad_labels(y: list[list[int]], pad_id: int, device: torch.device):
    """Pad the transcripts to the same length with a padding id.

    Args:
      y: The transcripts, which is a list of a list of token ids.
      pad_id: The padding token id to fill with.
      device: The device to place the output tensor on.

    Returns:
      Return a Tensor of padded transcripts.
    """
    y = [token_ids + [pad_id] for token_ids in y]
    length = max([len(token_ids) for token_ids in y])
    y = [token_ids + [pad_id] * (length - len(token_ids)) for token_ids in y]
    return torch.tensor(y, dtype=torch.int64, device=device)


def get_tokens_index(durations: list[list[int]], num_frames: int) -> torch.Tensor:
    """Get the position in the transcript for each frame.

    Returns the position in the symbol-sequence to look up.

    Args:
      durations:
        Duration of each token in transcripts.
      num_frames:
        The maximum frame length of the current batch.

    Returns:
      Return a Tensor of shape (batch_size, num_frames)
    """
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        if cur_frame != num_frames:
            msg = f"cur_frame {cur_frame} != num_frames {num_frames}"
            raise RuntimeError(msg)
    return ans


def to_int_tuple(s: str | int):
    """Convert a string of comma-separated ints or a single int to a tuple.

    Args:
        s: A single integer or a comma-separated string of integers.

    Returns:
        A tuple of integers.
    """
    if isinstance(s, int):
        return (s,)
    return tuple(map(int, s.split(",")))


def get_adjusted_batch_count(params: AttributeDict) -> float:
    """Return equivalent batch count if the reference duration had been used.

    Args:
        params: Training parameters containing batch_idx_train, max_duration,
            world_size, and ref_duration.

    Returns:
        Adjusted batch count for use with set_batch_count().
    """
    return params.batch_idx_train * (params.max_duration * params.world_size) / params.ref_duration


def set_batch_count(model: nn.Module | DDP, batch_count: float) -> None:
    """Set the batch_count attribute on all submodules that have it.

    Args:
        model: The model whose submodules should be updated.
        batch_count: The current batch count to set.
    """
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def condition_time_mask(
    features_lens: torch.Tensor,
    mask_percent: tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """Apply random time masking.

    Args:
        features_lens:
            input tensor of shape ``(B)``
        mask_percent:
            A (min, max) tuple for the fraction of frames to mask.
        max_len:
            the maximum length of the mask.

    Returns:
        Return a 2-D bool tensor (B, T), where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.
    """
    mask_size = (torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent) * features_lens).to(
        torch.int64
    )
    mask_starts = (torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (seq_range[None, :] < mask_ends[:, None])
    return mask


def condition_time_mask_suffix(
    features_lens: torch.Tensor,
    mask_percent: tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """Apply time masking from the end time index.

    Args:
        features_lens:
            input tensor of shape ``(B)``
        mask_percent:
            A (min, max) tuple for the fraction of frames to mask.
        max_len:
            the maximum length of the mask.

    Returns:
        Return a 2-D bool tensor (B, T), where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.
    """
    mask_size = (torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent) * features_lens).to(
        torch.int64
    )
    mask_starts = (torch.ones_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (seq_range[None, :] < mask_ends[:, None])
    return mask


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Create a padding mask from a lengths tensor.

    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.

    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    if lengths.ndim != 1:
        msg = f"Expected 1-D tensor, got ndim={lengths.ndim}"
        raise ValueError(msg)
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def str2bool(v: str | bool) -> bool:
    """Convert a string representation of a boolean to an actual bool.

    Accepts: yes/true/t/y/1 for True; no/false/f/n/0 for False.

    Args:
        v: String or bool value to convert.

    Returns:
        Boolean result.

    Raises:
        ValueError: If the string is not a recognised boolean representation.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    msg = f"Boolean value expected, got: {v!r}"
    raise ValueError(msg)


def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    import logging  # stdlib backend for structlog — kept local to avoid module-level ban

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


def get_git_sha1():
    """Return the short git SHA1 of the current commit, or None on error.

    Returns:
        Short git commit hash string like '1a2b3c4-clean', or None if unavailable.
    """
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                subprocess.run(
                    ["git", "diff", "--shortstat"],  # noqa: S607
                    check=True,
                    stdout=subprocess.PIPE,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + "-dirty" if dirty_commit else git_commit + "-clean"
    except subprocess.CalledProcessError:
        return None

    return git_commit


def get_git_date():
    """Return the git date of the last commit, or None on error.

    Returns:
        Git date string, or None if unavailable.
    """
    try:
        git_date = (
            subprocess.run(
                ["git", "log", "-1", "--format=%ad", "--date=local"],  # noqa: S607
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except subprocess.CalledProcessError:
        return None

    return git_date


def get_git_branch_name():
    """Return the current git branch name, or None on error.

    Returns:
        Branch name string, or None if unavailable.
    """
    try:
        git_date = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except subprocess.CalledProcessError:
        return None

    return git_date


def get_env_info() -> dict[str, Any]:
    """Get the environment information."""
    return {
        "torch-version": str(torch.__version__),
        "torch-cuda-available": torch.cuda.is_available(),
        "torch-cuda-version": torch.version.cuda,
        "python-version": sys.version[:4],
        "zipvoice-git-branch": get_git_branch_name(),
        "zipvoice-git-sha1": get_git_sha1(),
        "zipvoice-git-date": get_git_date(),
        "zipvoice-path": str(Path(__file__).resolve().parent.parent),
        "hostname": socket.gethostname(),
        "IP address": socket.gethostbyname(socket.gethostname()),
    }


def get_parameter_groups_with_lrs(  # noqa: C901, PLR0912
    model: nn.Module,
    lr: float,
    include_names: bool = False,
    freeze_modules: list[str] = None,
    unfreeze_modules: list[str] = None,
) -> list[dict]:
    """Build parameter groups with per-module learning-rate scaling for ScaledAdam.

    This is for use with the ScaledAdam optimizers (more recent versions that accept
    lists of named-parameters; we can, if needed, create a version without the names).

    It provides a way to specify learning-rate scales inside the module, so that if
    any nn.Module in the hierarchy has a floating-point parameter 'lr_scale', it will
    scale the LR of any parameters inside that module or its submodules.  Note: you
    can set module parameters outside the __init__ function, e.g.:
      >>> a = nn.Linear(10, 10)
      >>> a.lr_scale = 0.5

    Returns: a list of dicts, of the following form:
      if include_names == False:
        [  { 'params': [ tensor1, tensor2, ... ], 'lr': 0.01 },
           { 'params': [ tensor3, tensor4, ... ], 'lr': 0.005 },
         ...   ]
      if include_names == true:
        [  { 'named_params': [ (name1, tensor1, (name2, tensor2), ... ], 'lr': 0.01 },
           { 'named_params': [ (name3, tensor3), (name4, tensor4), ... ], 'lr': 0.005 },
         ...   ]

    """
    # Use freeze_modules or unfreeze_modules to freeze or unfreeze modules
    if unfreeze_modules is None:
        unfreeze_modules = []
    if freeze_modules is None:
        freeze_modules = []
    if len(freeze_modules) and len(unfreeze_modules):
        msg = "freeze_modules and unfreeze_modules are mutually exclusive; specify only one."
        raise ValueError(msg)

    # flat_lr_scale just contains the lr_scale explicitly specified
    # for each prefix of the name, e.g. 'encoder.layers.3', these need
    # to be multiplied for all prefix of the name of any given parameter.
    flat_lr_scale = defaultdict(lambda: 1.0)
    names = []
    for name, m in model.named_modules():
        names.append(name)
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale

    # lr_to_parames is a dict from learning rate (floating point) to: if
    # include_names == true, a list of (name, parameter) for that learning rate;
    # otherwise a list of parameters for that learning rate.
    lr_to_params = defaultdict(list)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            log.info("remove_param_no_grad", name=name)
            continue
        split_name = name.split(".")
        # caution: as a special case, if the name is '', split_name will be [ '' ].
        prefix = split_name[0]
        if len(freeze_modules) > 0:
            if prefix == "module":  # DDP
                module_name = split_name[1]
                if module_name in freeze_modules:
                    log.info("remove_frozen_param", name=name)
                    continue
            else:
                if prefix in freeze_modules:
                    log.info("remove_frozen_param", name=name)
                    continue
        elif len(unfreeze_modules) > 0:
            if prefix == "module":  # DDP
                module_name = split_name[1]
                if module_name not in unfreeze_modules:
                    log.info("remove_non_unfrozen_param", name=name)
                    continue
            else:
                if prefix not in unfreeze_modules:
                    log.info("remove_non_unfrozen_param", name=name)
                    continue
        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

    if include_names:
        return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
    else:
        return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]


# end zipvoice/utils/common.py
