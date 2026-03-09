# start zipvoice/bin/generate_averaged_model.py
#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation
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
"""Load checkpoints and average them.

This script loads checkpoints and averages them.

python3 -m zipvoice.bin.generate_averaged_model  \
    --epoch 11 \
    --avg 4 \
    --model-name zipvoice \
    --exp-dir exp/zipvoice

It will generate a file `epoch-11-avg-14.pt` in the given `exp_dir`.
You can later load it by `torch.load("epoch-11-avg-4.pt")`.
"""

from pathlib import Path

import orjson
import structlog
import torch
import typer

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_dialog import ZipVoiceDialog, ZipVoiceDialogStereo
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import (
    average_checkpoints_with_averaged_model,
    find_checkpoints,
)
from zipvoice.utils.common import AttributeDict

log = structlog.get_logger()


app = typer.Typer(help="Load checkpoints and average them.", add_completion=False)


@app.command()
@torch.no_grad()
def main(  # noqa: PLR0912, PLR0915, C901
    epoch: int = typer.Option(  # noqa: B008
        11,
        "--epoch",
        help="It specifies the checkpoint to use for decoding. Note: Epoch counts "
        "from 1. You can specify --avg to use more checkpoints for model averaging.",
    ),
    iter: int = typer.Option(  # noqa: B008
        0,
        "--iter",
        help="If positive, --epoch is ignored and it will use the checkpoint "
        "exp_dir/checkpoint-iter.pt. You can specify --avg to use more checkpoints "
        "for model averaging.",
    ),
    avg: int = typer.Option(  # noqa: B008
        4,
        "--avg",
        help="Number of checkpoints to average. Automatically select consecutive "
        "checkpoints before the checkpoint specified by '--epoch' or --iter",
    ),
    exp_dir: str = typer.Option(  # noqa: B008
        "exp/zipvoice", "--exp-dir", help="The experiment dir"
    ),
    model_name: str = typer.Option(  # noqa: B008
        "zipvoice", "--model-name", help="The model type to be averaged."
    ),
) -> None:
    """Load model checkpoints, average them, and save the result."""
    params = AttributeDict()
    params.update({"epoch": epoch, "iter": iter, "avg": avg, "exp_dir": exp_dir, "model_name": model_name})
    params.exp_dir = Path(params.exp_dir)

    with open(params.exp_dir / "model.json", "rb") as f:
        model_config = orjson.loads(f.read())

    # Any tokenizer can be used here.
    # Use SimpleTokenizer for simplicity.
    tokenizer = SimpleTokenizer(token_file=params.exp_dir / "tokens.txt")
    if params.model_name in ["zipvoice", "zipvoice_distill"]:
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        }
    elif params.model_name in ["zipvoice_dialog", "zipvoice_dialog_stereo"]:
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "spk_a_id": tokenizer.spk_a_id,
            "spk_b_id": tokenizer.spk_b_id,
        }

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    log.info("script_started")

    params.device = torch.device("cpu")
    log.info("device", device=str(params.device))

    log.info("about_to_create_model")
    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_distill":
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_dialog":
        model = ZipVoiceDialog(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_dialog_stereo":
        model = ZipVoiceDialogStereo(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        msg = f"Unknown model name: {params.model_name}"
        raise ValueError(msg)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[: params.avg + 1]
        if len(filenames) == 0:
            msg = f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            raise ValueError(msg)
        elif len(filenames) < params.avg + 1:
            msg = f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
            raise ValueError(msg)
        filename_start = filenames[-1]
        filename_end = filenames[0]
        log.info(
            "averaging_iteration_checkpoints",
            filename_start=str(filename_start),
            filename_end=str(filename_end),
        )
        model.to(params.device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=params.device,
            ),
            strict=True,
        )
    else:
        if params.avg <= 0:
            msg = f"avg must be > 0, got {params.avg}"
            raise ValueError(msg)
        start = params.epoch - params.avg
        if start < 1:
            msg = f"start epoch must be >= 1, got {start}"
            raise ValueError(msg)
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        log.info(
            "averaging_epoch_checkpoints",
            start=start,
            epoch=params.epoch,
        )
        model.to(params.device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=params.device,
            ),
            strict=True,
        )
    if params.iter > 0:
        filename = params.exp_dir / f"iter-{params.iter}-avg-{params.avg}.pt"
    else:
        filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"

    log.info("saving_averaged_checkpoint", filename=str(filename))
    torch.save({"model": model.state_dict()}, filename)

    num_param = sum([p.numel() for p in model.parameters()])
    log.info("model_parameters", num_param=num_param)

    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/generate_averaged_model.py
