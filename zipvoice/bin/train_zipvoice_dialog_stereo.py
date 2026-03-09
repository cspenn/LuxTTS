# start zipvoice/bin/train_zipvoice_dialog_stereo.py
#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script trains a ZipVoice-Dialog model.

Usage:

python3 -m zipvoice.bin.train_zipvoice_dialog_stereo \
    --world-size 8 \
    --use-fp16 1 \
    --base-lr 0.002 \
    --max-duration 500 \
    --model-config conf/zipvoice_base.json \
    --token-file "data/tokens_dialog.txt" \
    --manifest-dir data/fbank \
    --exp-dir exp/zipvoice_dialog_stereo
"""

import copy
import os
from functools import partial
from pathlib import Path
from shutil import copyfile

import orjson
import structlog
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import typer
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from zipvoice.dataset.datamodule import TtsDataModule

import zipvoice.utils.diagnostics as diagnostics
from zipvoice.bin.train_zipvoice import (
    display_and_save_batch,
    get_params,
    tokenize_text,
)
from zipvoice.models.zipvoice_dialog import ZipVoiceDialogStereo
from zipvoice.tokenizer.tokenizer import DialogTokenizer
from zipvoice.utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_copy_proj_three_channel_alter,
    remove_checkpoints,
    resume_checkpoint,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from zipvoice.utils.common import (
    AttributeDict,
    GradScaler,
    MetricsTracker,
    cleanup_dist,
    create_grad_scaler,
    get_adjusted_batch_count,
    get_parameter_groups_with_lrs,
    prepare_input,
    set_batch_count,
    setup_dist,
    setup_logger,
    torch_autocast,
)
from zipvoice.utils.hooks import register_inf_check_hooks
from zipvoice.utils.lr_scheduler import FixedLRScheduler, LRScheduler
from zipvoice.utils.optim import ScaledAdam

log = structlog.get_logger()

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler | LRScheduler


app = typer.Typer(
    help="Train a ZipVoice-Dialog stereo model.",
    add_completion=False,
)


def compute_fbank_loss(  # noqa: PLR0913
    params: AttributeDict,
    model: nn.Module | DDP,
    features: Tensor,
    features_lens: Tensor,
    tokens: list[list[int]],
    is_training: bool,
    use_two_channel: bool,
) -> tuple[Tensor, MetricsTracker]:
    """Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      features:
        The target acoustic feature.
      features_lens:
        The number of frames of each utterance.
      tokens:
        Input tokens that representing the transcripts.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      use_two_channel:
        True for using two channel features, False for using one channel features.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape

    if features.size(2) != 3 * params.feat_dim:
        msg = "we assume three channel features, the last channel is the mixed-channel feature"
        raise ValueError(msg)
    features = features[:, :, : params.feat_dim * 2] if use_two_channel else features[:, :, params.feat_dim * 2 :]

    noise = torch.randn_like(features)  # (B, T, F)

    # Sampling t from uniform distribution
    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (torch.arange(batch_size, device=device) / batch_size).unsqueeze(1).unsqueeze(2)
    with torch.set_grad_enabled(is_training):
        loss = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            condition_drop_ratio=params.condition_drop_ratio,
            se_weight=1 if use_two_channel else 0,
        )

    if loss.requires_grad != is_training:
        msg = f"loss.requires_grad ({loss.requires_grad}) != is_training ({is_training})"
        raise RuntimeError(msg)
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames

    return loss, info


def train_one_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
    params: AttributeDict,
    model: nn.Module | DDP,
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: nn.Module | None = None,
    tb_writer: SummaryWriter | None = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer.
      scheduler:
        The learning rate scheduler, we call step() every epoch.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      model_avg:
        The averaged model, used for EMA model averaging. Only used with rank 0.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params) + 100000)

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            log.info("computing_validation_loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            log.info(
                "validation_loss",
                epoch=params.cur_epoch,
                global_batch_idx=params.batch_idx_train,
                validation=str(valid_info),
            )
            log.info(
                "max_memory_allocated",
                mb=torch.cuda.max_memory_allocated() // 1000000,
            )
            if tb_writer is not None:
                valid_info.write_summary(tb_writer, "train/valid_", params.batch_idx_train)

        params.batch_idx_train += 1

        batch_size = len(batch["text"])

        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )

        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                    use_two_channel=(batch_idx % 2 == 1),
                )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            log.info("caught_exception", error=str(e))
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if rank == 0 and params.batch_idx_train > 0 and params.batch_idx_train % params.average_period == 0:
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if params.batch_idx_train > 0 and params.batch_idx_train % params.save_every_n == 0:
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it. The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have
            # different behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 1024.0 or (cur_grad_scale < 4096.0 and params.batch_idx_train % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                log.warning("grad_scale_small", cur_grad_scale=cur_grad_scale)
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                msg = f"grad_scale is too small, exiting: {cur_grad_scale}"
                raise RuntimeError(msg)

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            log.info(
                "train_step",
                epoch=params.cur_epoch,
                batch=batch_idx,
                global_batch_idx=params.batch_idx_train,
                batch_size=batch_size,
                loss=str(loss_info),
                tot_loss=str(tot_loss),
                cur_lr=f"{cur_lr:.2e}",
                grad_scale=scaler._scale.item() if params.use_fp16 else None,
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
                loss_info.write_summary(tb_writer, "train/current_", params.batch_idx_train)
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module | DDP,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for _batch_idx, batch in enumerate(valid_dl):
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )

        loss, loss_info = compute_fbank_loss(
            params=params,
            model=model,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            is_training=False,
            use_two_channel=True,
        )
        if loss.requires_grad is not False:
            msg = "loss.requires_grad should be False during validation"
            raise RuntimeError(msg)
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def scan_pessimistic_batches_for_oom(
    model: nn.Module | DDP,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    """Scan pessimistic batches for OOM errors before training starts."""
    from lhotse.dataset import find_pessimistic_batches

    log.info("sanity_check_oom")
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )
        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                    use_two_channel=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                log.exception(
                    "cuda_out_of_memory",
                    criterion=str(criterion),
                    crit_value=str(crit_values[criterion]),
                )
            display_and_save_batch(batch, params=params)
            raise
        log.info("max_memory_allocated", mb=torch.cuda.max_memory_allocated() // 1000000)


def run(rank, world_size, args):  # noqa: C901, PLR0912, PLR0915
    """Train the model in a single process.

    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        An argparse.Namespace containing all training arguments.
    """
    params = get_params()
    params.update(vars(args))
    params.valid_interval = params.save_every_n
    # Set epoch to a large number to ignore it.
    if params.num_iters > 0:
        params.num_epochs = 1000000
    with open(params.model_config, "rb") as f:
        model_config = orjson.loads(f.read())
    params.update(model_config["model"])
    params.update(model_config["feature"])

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

    tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard") if args.tensorboard and rank == 0 else None

    if torch.cuda.is_available():
        params.device = torch.device("cuda", rank)
    else:
        params.device = torch.device("cpu")
    log.info("device", device=str(params.device))

    tokenizer = DialogTokenizer(token_file=params.token_file)
    tokenizer_config = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "spk_a_id": tokenizer.spk_a_id,
        "spk_b_id": tokenizer.spk_b_id,
    }
    params.update(tokenizer_config)

    log.info("params", params=str(params))

    log.info("about_to_create_model")

    model = ZipVoiceDialogStereo(
        **model_config["model"],
        **tokenizer_config,
    )

    if params.checkpoint is None:
        msg = "A pre-trained checkpoint is required"
        raise ValueError(msg)
    log.info("loading_pretrained_model", checkpoint=str(params.checkpoint))

    if params.finetune:
        # load a pre-trained ZipVoice-Dialog-Stereo model
        _ = load_checkpoint(filename=params.checkpoint, model=model, strict=True)
    else:
        # load a pre-trained ZipVoice-Dialog model, duplicate the proj layers
        load_checkpoint_copy_proj_three_channel_alter(
            filename=params.checkpoint,
            in_proj_key="fm_decoder.in_proj",
            out_proj_key="fm_decoder.out_proj",
            dim=params.feat_dim,
            model=model,
        )
    num_param = sum([p.numel() for p in model.parameters()])
    log.info("num_parameters", num_param=num_param)

    model_avg: nn.Module | None = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    if params.start_epoch <= 0:
        msg = f"start_epoch must be positive, got {params.start_epoch}"
        raise ValueError(msg)
    if params.start_epoch > 1:
        checkpoints = resume_checkpoint(params=params, model=model, model_avg=model_avg)

    model = model.to(params.device)
    if world_size > 1:
        log.info("using_ddp")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = FixedLRScheduler(optimizer)

    scaler = create_grad_scaler(enabled=params.use_fp16)

    if params.start_epoch > 1 and checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            log.info("loading_optimizer_state_dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            log.info("loading_scheduler_state_dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

        if "grad_scaler" in checkpoints:
            log.info("loading_grad_scaler_state_dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(512)  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    def remove_short_and_long_utt(c: Cut, min_len: float, max_len: float):
        return not (c.duration < min_len or c.duration > max_len)

    _remove_short_and_long_utt = partial(remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len)

    datamodule = TtsDataModule(args)
    train_cuts = datamodule.train_custom_cuts(params.train_manifest)
    train_cuts = train_cuts.filter(_remove_short_and_long_utt)
    dev_cuts = datamodule.dev_custom_cuts(params.dev_manifest)
    # To avoid OOM issues due to too long dev cuts
    dev_cuts = dev_cuts.filter(_remove_short_and_long_utt)

    if not hasattr(train_cuts[0].supervisions[0], "tokens") or not hasattr(dev_cuts[0].supervisions[0], "tokens"):
        log.warning(
            "tokens_not_prepared",
            detail="will tokenize on-the-fly, which can slow down training significantly",
        )
    _tokenize_text = partial(tokenize_text, tokenizer=tokenizer)
    train_cuts = train_cuts.map(_tokenize_text)
    dev_cuts = dev_cuts.map(_tokenize_text)

    train_dl = datamodule.train_dataloaders(train_cuts)

    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    if params.scan_oom:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    log.info("training_started")

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        log.info("start_epoch", epoch=epoch)
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)

    log.info("done")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


@app.command()
def main(  # noqa: PLR0913
    world_size: int = typer.Option(1, "--world-size", help="Number of GPUs for DDP training."),  # noqa: B008
    master_port: int = typer.Option(12356, "--master-port", help="Master port to use for DDP training."),  # noqa: B008
    tensorboard: bool = typer.Option(  # noqa: B008
        True, "--tensorboard", help="Should various information be logged in tensorboard."
    ),
    num_epochs: int = typer.Option(8, "--num-epochs", help="Number of epochs to train."),  # noqa: B008
    num_iters: int = typer.Option(25000, "--num-iters", help="Number of iter to train, will ignore num_epochs if > 0."),  # noqa: B008
    start_epoch: int = typer.Option(  # noqa: B008
        1,
        "--start-epoch",
        help="Resume training from this epoch. It should be positive. If larger than 1, "
        "it will load checkpoint from exp-dir/epoch-{start_epoch-1}.pt",
    ),
    checkpoint: str = typer.Option(  # noqa: B008
        ...,
        "--checkpoint",
        help="Checkpoints of pre-trained models, either a ZipVoice model or a ZipVoice-Dialog model.",
    ),
    exp_dir: str = typer.Option(  # noqa: B008
        "exp/zipvoice_dialog",
        "--exp-dir",
        help="The experiment dir. It specifies the directory where all training related files, "
        "e.g., checkpoints, log, etc, are saved",
    ),
    base_lr: float = typer.Option(0.002, "--base-lr", help="The base learning rate."),  # noqa: B008
    ref_duration: float = typer.Option(  # noqa: B008
        50,
        "--ref-duration",
        help="Reference batch duration for purposes of adjusting batch counts for "
        "setting various schedules inside the model.",
    ),
    finetune: bool = typer.Option(  # noqa: B008
        False,
        "--finetune",
        help="Whether to fine-tune from our pre-trained ZipVoice-Dialog model. "
        "False means to fine-tune from a pre-trained ZipVoice model.",
    ),
    seed: int = typer.Option(42, "--seed", help="The seed for random generators intended for reproducibility"),  # noqa: B008
    print_diagnostics: bool = typer.Option(  # noqa: B008
        False, "--print-diagnostics", help="Accumulate stats on activations, print them and exit."
    ),
    scan_oom: bool = typer.Option(False, "--scan-oom", help="Scan pessimistic batches to see whether they cause OOMs."),  # noqa: B008
    inf_check: bool = typer.Option(  # noqa: B008
        False, "--inf-check", help="Add hooks to check for infinite module outputs and gradients."
    ),
    save_every_n: int = typer.Option(  # noqa: B008
        5000,
        "--save-every-n",
        help="Save checkpoint after processing this number of batches periodically.",
    ),
    keep_last_k: int = typer.Option(30, "--keep-last-k", help="Only keep this number of checkpoints on disk."),  # noqa: B008
    average_period: int = typer.Option(  # noqa: B008
        200, "--average-period", help="Update the averaged model after processing this number of batches."
    ),
    use_fp16: bool = typer.Option(True, "--use-fp16", help="Whether to use half precision training."),  # noqa: B008
    feat_scale: float = typer.Option(0.1, "--feat-scale", help="The scale factor of fbank feature"),  # noqa: B008
    condition_drop_ratio: float = typer.Option(  # noqa: B008
        0.2, "--condition-drop-ratio", help="The drop rate of text condition during training."
    ),
    train_manifest: str | None = typer.Option(None, "--train-manifest", help="Path of the training manifest"),  # noqa: B008
    dev_manifest: str | None = typer.Option(None, "--dev-manifest", help="Path of the validation manifest"),  # noqa: B008
    min_len: float = typer.Option(1.0, "--min-len", help="The minimum audio length used for training"),  # noqa: B008
    max_len: float = typer.Option(60.0, "--max-len", help="The maximum audio length used for training"),  # noqa: B008
    model_config: str = typer.Option("zipvoice_base.json", "--model-config", help="The model configuration file."),  # noqa: B008
    token_file: str = typer.Option(  # noqa: B008
        "data/tokens_dialog.txt",
        "--token-file",
        help="The file that contains information that maps tokens to ids, "
        "which is a text file with '{token}\\t{token_id}' per line.",
    ),
    manifest_dir: str = typer.Option(  # noqa: B008
        "data/fbank", "--manifest-dir", help="Path to the directory containing manifests."
    ),
    max_duration: float = typer.Option(200.0, "--max-duration", help="Maximum batch duration in seconds."),  # noqa: B008
) -> None:
    """Train a ZipVoice-Dialog stereo model using distributed data parallel training."""
    args = AttributeDict(
        {
            "world_size": world_size,
            "master_port": master_port,
            "tensorboard": tensorboard,
            "num_epochs": num_epochs,
            "num_iters": num_iters,
            "start_epoch": start_epoch,
            "checkpoint": checkpoint,
            "exp_dir": Path(exp_dir),
            "base_lr": base_lr,
            "ref_duration": ref_duration,
            "finetune": finetune,
            "seed": seed,
            "print_diagnostics": print_diagnostics,
            "scan_oom": scan_oom,
            "inf_check": inf_check,
            "save_every_n": save_every_n,
            "keep_last_k": keep_last_k,
            "average_period": average_period,
            "use_fp16": use_fp16,
            "feat_scale": feat_scale,
            "condition_drop_ratio": condition_drop_ratio,
            "train_manifest": train_manifest,
            "dev_manifest": dev_manifest,
            "min_len": min_len,
            "max_len": max_len,
            "model_config": model_config,
            "token_file": token_file,
            "manifest_dir": manifest_dir,
            "max_duration": max_duration,
        }
    )

    if world_size < 1:
        msg = f"world_size must be >= 1, got {world_size}"
        raise ValueError(msg)
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app()

# end zipvoice/bin/train_zipvoice_dialog_stereo.py
