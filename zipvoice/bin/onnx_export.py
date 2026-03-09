# start zipvoice/bin/onnx_export.py
#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
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

"""Export a pre-trained ZipVoice or ZipVoice-Distill model from PyTorch to ONNX.

This script exports the model from PyTorch to ONNX format.

Usage:

python3 -m zipvoice.bin.onnx_export \
    --model-name zipvoice \
    --model-dir exp/zipvoice \
    --checkpoint-name epoch-11-avg-4.pt \
    --onnx-model-dir exp/zipvoice

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.
"""

from pathlib import Path

import onnx
import orjson
import safetensors.torch
import structlog
import torch
import typer
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled

log = structlog.get_logger()


app = typer.Typer(help="Export a pre-trained ZipVoice model from PyTorch to ONNX.", add_completion=False)


def add_meta_data(filename: str, meta_data: dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


class OnnxTextModel(nn.Module):
    """A wrapper for ZipVoice text encoder for ONNX export."""

    def __init__(self, model: nn.Module):
        """Initialize with the ZipVoice text encoder components.

        Args:
            model: The ZipVoice model to extract the text encoder from.
        """
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder
        self.pad_id = model.pad_id

    def forward(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tensor:
        """Run the text encoder forward pass.

        Args:
            tokens: Text token tensor.
            prompt_tokens: Prompt text token tensor.
            prompt_features_len: Length of prompt features.
            speed: Speed control tensor.

        Returns:
            Text condition tensor.
        """
        cat_tokens = torch.cat([prompt_tokens, tokens], dim=1)
        cat_tokens = nn.functional.pad(cat_tokens, (0, 1), value=self.pad_id)
        tokens_len = cat_tokens.shape[1] - 1
        padding_mask = (torch.arange(tokens_len + 1) == tokens_len).unsqueeze(0)

        embed = self.embed(cat_tokens)
        embed = self.text_encoder(x=embed, t=None, padding_mask=padding_mask)

        features_len = torch.ceil(prompt_features_len / prompt_tokens.shape[1] * tokens_len / speed).to(
            dtype=torch.int64
        )

        token_dur = torch.div(features_len, tokens_len, rounding_mode="floor").to(dtype=torch.int64)

        # If you pass a scalar tensor, ONNX may infer the shape as [1] (rank-1 tensor),
        # but sometimes expects an actual scalar (rank-0).
        # When exporting, ONNX may generate a model where Concat expects inputs of the
        # same rank, but receives [1] and [].
        # In PyTorch, this is usually fine. In ONNX Runtime (C++), this causes the error like
        # "Ranks of input data are different, cannot concatenate them. expected rank: 1 got: 2"
        # If you use x.item(), ONNX loses the dynamic link and the input mismatch error can happen at inference.
        # use reshape(()) to convert a rank-1 tensor to a rank-0 tensor.

        token_dur = token_dur.reshape(())
        features_len = features_len.reshape(())

        text_condition = embed[:, :-1, :].unsqueeze(2).expand(-1, -1, token_dur, -1)
        text_condition = text_condition.reshape(embed.shape[0], -1, embed.shape[2])

        text_condition = torch.cat(
            [
                text_condition,
                embed[:, -1:, :].expand(-1, features_len - text_condition.shape[1], -1),
            ],
            dim=1,
        )

        return text_condition


class OnnxFlowMatchingModel(nn.Module):
    """A wrapper for ZipVoice flow-matching decoder for ONNX export."""

    def __init__(self, model: nn.Module, distill: bool = False):
        """Initialize with the ZipVoice flow-matching decoder components.

        Args:
            model: The ZipVoice model to extract the FM decoder from.
            distill: Whether the model is a distilled model.
        """
        super().__init__()
        self.distill = distill
        self.fm_decoder = model.fm_decoder
        self.model_func = model.forward_fm_decoder
        self.feat_dim = model.feat_dim

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        """Run the flow-matching decoder forward pass.

        Args:
            t: Current timestep tensor.
            x: Current noisy features tensor.
            text_condition: Text conditioning tensor.
            speech_condition: Speech conditioning tensor.
            guidance_scale: Classifier-free guidance scale tensor.

        Returns:
            Predicted velocity tensor.
        """
        if self.distill:
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                guidance_scale=guidance_scale,
            )
        else:
            x = x.repeat(2, 1, 1)
            text_condition = torch.cat([torch.zeros_like(text_condition), text_condition], dim=0)
            speech_condition = torch.cat(
                [
                    torch.where(t > 0.5, torch.zeros_like(speech_condition), speech_condition),
                    speech_condition,
                ],
                dim=0,
            )
            guidance_scale = torch.where(t > 0.5, guidance_scale, guidance_scale * 2.0)
            data_uncond, data_cond = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
            ).chunk(2, dim=0)
            v = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return v


def export_text_encoder(
    model: OnnxTextModel,
    filename: str,
    opset_version: int = 13,
) -> None:
    """Export the text encoder model to ONNX format.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
    prompt_tokens = torch.tensor([[0, 1]], dtype=torch.int64)
    prompt_features_len = torch.tensor(10, dtype=torch.int64)
    speed = torch.tensor(1.0, dtype=torch.float32)

    model = torch.jit.trace(model, (tokens, prompt_tokens, prompt_features_len, speed))

    torch.onnx.export(
        model,
        (tokens, prompt_tokens, prompt_features_len, speed),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["tokens", "prompt_tokens", "prompt_features_len", "speed"],
        output_names=["text_condition"],
        dynamic_axes={
            "tokens": {0: "N", 1: "T"},
            "prompt_tokens": {0: "N", 1: "T"},
            "text_condition": {0: "N", 1: "T"},
        },
    )

    meta_data = {
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ZipVoice text encoder",
        "use_espeak": "1",
        "use_pinyin": "1",
    }
    log.info("meta_data", meta_data=meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)

    log.info("exported", filename=str(filename))


def export_fm_decoder(
    model: OnnxFlowMatchingModel,
    filename: str,
    opset_version: int = 13,
) -> None:
    """Export the flow matching decoder model to ONNX format.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    feat_dim = model.feat_dim
    seq_len = 200
    t = torch.tensor(0.5, dtype=torch.float32)
    x = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    text_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    speech_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    guidance_scale = torch.tensor(1.0, dtype=torch.float32)

    model = torch.jit.trace(model, (t, x, text_condition, speech_condition, guidance_scale))

    torch.onnx.export(
        model,
        (t, x, text_condition, speech_condition, guidance_scale),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["t", "x", "text_condition", "speech_condition", "guidance_scale"],
        output_names=["v"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "text_condition": {0: "N", 1: "T"},
            "speech_condition": {0: "N", 1: "T"},
            "v": {0: "N", 1: "T"},
        },
    )

    meta_data = {
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ZipVoice flow-matching decoder",
        "feat_dim": str(feat_dim),
        "sample_rate": "24000",
        "n_fft": "1024",
        "hop_length": "256",
        "window_length": "1024",
        "num_mels": "100",
    }
    log.info("meta_data", meta_data=meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)

    log.info("exported", filename=str(filename))


@app.command()
@torch.no_grad()
def main(  # noqa: PLR0915
    onnx_model_dir: str = typer.Option(  # noqa: B008
        "exp", "--onnx-model-dir", help="Dir to the exported models"
    ),
    model_name: str = typer.Option(  # noqa: B008
        "zipvoice", "--model-name", help="The model used for inference"
    ),
    model_dir: str = typer.Option(  # noqa: B008
        None,
        "--model-dir",
        help="The model directory that contains model checkpoint, configuration "
        "file model.json, and tokens file tokens.txt. Will download pre-trained "
        "checkpoint from huggingface if not specified.",
    ),
    checkpoint_name: str = typer.Option(  # noqa: B008
        "model.pt", "--checkpoint-name", help="The name of model checkpoint."
    ),
) -> None:
    """Export a ZipVoice model to ONNX format with int8 quantization."""
    params = AttributeDict()
    params.update(
        {
            "onnx_model_dir": onnx_model_dir,
            "model_name": model_name,
            "model_dir": model_dir,
            "checkpoint_name": checkpoint_name,
        }
    )

    params.model_dir = Path(params.model_dir)
    if not params.model_dir.is_dir():
        msg = f"{params.model_dir} does not exist"
        raise FileNotFoundError(msg)
    for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
        if not (params.model_dir / filename).is_file():
            msg = f"{params.model_dir / filename} does not exist"
            raise FileNotFoundError(msg)
    model_ckpt = params.model_dir / params.checkpoint_name
    model_config = params.model_dir / "model.json"
    token_file = params.model_dir / "tokens.txt"

    log.info("loading_model", model_dir=str(params.model_dir))

    tokenizer = SimpleTokenizer(token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "rb") as f:
        model_config = orjson.loads(f.read())

    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
        distill = False
    else:
        if params.model_name != "zipvoice_distill":
            msg = f"Unsupported model name: {params.model_name}"
            raise ValueError(msg)
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )
        distill = True

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=True)
    else:
        msg = f"Unsupported model checkpoint format: {model_ckpt}"
        raise ValueError(msg)

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    log.info("exporting_model")
    onnx_model_dir = Path(params.onnx_model_dir)
    onnx_model_dir.mkdir(parents=True, exist_ok=True)
    opset_version = 13

    text_encoder = OnnxTextModel(model=model)
    text_encoder_file = onnx_model_dir / "text_encoder.onnx"
    export_text_encoder(
        model=text_encoder,
        filename=text_encoder_file,
        opset_version=opset_version,
    )

    fm_decoder = OnnxFlowMatchingModel(model=model, distill=distill)
    fm_decoder_file = onnx_model_dir / "fm_decoder.onnx"
    export_fm_decoder(
        model=fm_decoder,
        filename=fm_decoder_file,
        opset_version=opset_version,
    )

    log.info("generating_int8_quantization_models")

    text_encoder_int8_file = onnx_model_dir / "text_encoder_int8.onnx"
    quantize_dynamic(
        model_input=text_encoder_file,
        model_output=text_encoder_int8_file,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    fm_decoder_int8_file = onnx_model_dir / "fm_decoder_int8.onnx"
    quantize_dynamic(
        model_input=fm_decoder_file,
        model_output=fm_decoder_int8_file,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/onnx_export.py
