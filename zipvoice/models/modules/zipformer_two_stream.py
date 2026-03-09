"""TTSZipformerTwoStream: dual-stream Zipformer encoder for dialog TTS."""

# start zipvoice/models/modules/zipformer_two_stream.py
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

import math

import torch
from torch import Tensor, nn

from zipvoice.models.modules.scaling import FloatLike, ScheduledFloat, SwooshR
from zipvoice.models.modules.zipformer import (
    DownsampledZipformer2Encoder,
    TTSZipformer,
    Zipformer2Encoder,
    Zipformer2EncoderLayer,
)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: shape of (N) or (N, T)
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an Tensor of positional embeddings. shape of (N, dim) or (T, N, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )

    if timesteps.dim() == 2:
        timesteps = timesteps.transpose(0, 1)  # (N, T) -> (T, N)

    args = timesteps[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


class TTSZipformerTwoStream(TTSZipformer):
    """Dual-stream TTS Zipformer for stereo/dialog synthesis.

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same
    length as downsampling_factor if they are single ints or one-element tuples.
    The length of downsampling_factor defines the number of stacks.

        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks,
            one per encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection
            per attention head
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to
            projection, e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        use_time_embed: (bool): if True, do not take time embedding as additional input.
        time_embed_dim: (int): the dimension of the time embedding.
    """

    def __init__(  # noqa: PLR0913, C901, PLR0915
        self,
        in_dim: tuple[int, ...],
        out_dim: tuple[int, ...],
        downsampling_factor: tuple[int, ...] = (2, 4),
        num_encoder_layers: int | tuple[int, ...] = 4,
        cnn_module_kernel: int | tuple[int, ...] = 31,
        encoder_dim: int = 384,
        query_head_dim: int = 24,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        num_heads: int = 8,
        feedforward_dim: int = 1536,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        warmup_batches: float = 4000.0,
        use_time_embed: bool = True,
        time_embed_dim: int = 192,
        use_conv: bool = True,
    ) -> None:
        """Initialize TTSZipformerTwoStream."""
        nn.Module.__init__(self)

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))
        if isinstance(downsampling_factor, int):
            downsampling_factor = (downsampling_factor,)

        def _to_tuple(x):
            """Converts a single int or 1-tuple to a tuple matching downsampling_factor length."""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                if not (len(x) == len(downsampling_factor) and isinstance(x[0], int)):
                    msg = (
                        f"Expected len(x) == len(downsampling_factor) and isinstance(x[0], int), "
                        f"got len(x)={len(x)}, len(downsampling_factor)={len(downsampling_factor)}"
                    )
                    raise ValueError(msg)
            return x

        def _assert_downsampling_factor(factors):
            """Assert downsampling_factor follows u-net style."""
            if not (factors[0] == 1 and factors[-1] == 1):
                msg = f"downsampling_factor must start and end with 1, got {factors}"
                raise ValueError(msg)

            for i in range(1, len(factors) // 2 + 1):
                if factors[i] != factors[i - 1] * 2:
                    msg = (
                        f"downsampling_factor must double in first half: "
                        f"factors[{i}]={factors[i]} != factors[{i - 1}]*2={factors[i - 1] * 2}"
                    )
                    raise ValueError(msg)

            for i in range(len(factors) // 2 + 1, len(factors)):
                if factors[i] * 2 != factors[i - 1]:
                    msg = (
                        f"downsampling_factor must halve in second half: "
                        f"factors[{i}]*2={factors[i] * 2} != factors[{i - 1}]={factors[i - 1]}"
                    )
                    raise ValueError(msg)

        _assert_downsampling_factor(downsampling_factor)
        self.downsampling_factor = downsampling_factor  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)
        self.encoder_dim = encoder_dim
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim
        self.num_heads = num_heads

        self.use_time_embed = use_time_embed

        self.time_embed_dim = time_embed_dim
        if self.use_time_embed:
            if time_embed_dim == -1:
                msg = "time_embed_dim must not be -1 when use_time_embed is True"
                raise ValueError(msg)
        else:
            time_embed_dim = -1

        if not (len(in_dim) == len(out_dim) == 2):
            msg = (
                f"Expected len(in_dim) == len(out_dim) == 2, got len(in_dim)={len(in_dim)}, len(out_dim)={len(out_dim)}"
            )
            raise ValueError(msg)

        self.in_dim = in_dim
        self.in_proj = nn.ModuleList([nn.Linear(in_dim[0], encoder_dim), nn.Linear(in_dim[1], encoder_dim)])
        self.out_dim = out_dim
        self.out_proj = nn.ModuleList([nn.Linear(encoder_dim, out_dim[0]), nn.Linear(encoder_dim, out_dim[1])])

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim,
                pos_dim=pos_dim,
                num_heads=num_heads,
                query_head_dim=query_head_dim,
                pos_head_dim=pos_head_dim,
                value_head_dim=value_head_dim,
                feedforward_dim=feedforward_dim,
                use_conv=use_conv,
                cnn_module_kernel=cnn_module_kernel[i],
                dropout=dropout,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                embed_dim=encoder_dim,
                time_embed_dim=time_embed_dim,
                pos_dim=pos_dim,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim,
                    downsample=downsampling_factor[i],
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        if self.use_time_embed:
            self.time_embed = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim * 2),
                SwooshR(),
                nn.Linear(time_embed_dim * 2, time_embed_dim),
            )
        else:
            self.time_embed = None

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the two-stream forward pass.

        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          t:
            A t tensor of shape (batch_size,) or (batch_size, seq_len)
          padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.

        Returns:
          Return the output embeddings. its shape is
            (batch_size, output_seq_len, encoder_dim).
        """
        if x.size(2) not in self.in_dim:
            msg = f"{x.size(2)} not in {self.in_dim}"
            raise ValueError(msg)
        index = 0 if x.size(2) == self.in_dim[0] else 1
        x = x.permute(1, 0, 2)
        x = self.in_proj[index](x)

        if t is not None:
            if t.dim() != 1 and t.dim() != 2:
                msg = f"Expected t.dim() == 1 or 2, got {t.dim()}, shape={t.shape}"
                raise ValueError(msg)
            time_emb = timestep_embedding(t, self.time_embed_dim)
            time_emb = self.time_embed(time_emb)
        else:
            time_emb = None

        attn_mask = None

        for _i, module in enumerate(self.encoders):
            x = module(
                x,
                time_emb=time_emb,
                src_key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )
        x = self.out_proj[index](x)
        x = x.permute(1, 0, 2)
        return x


# end zipvoice/models/modules/zipformer_two_stream.py
