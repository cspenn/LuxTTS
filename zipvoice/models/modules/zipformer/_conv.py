# start zipvoice/models/modules/zipformer/_conv.py
# Copyright    2022-2024  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao,
#                                                       Wei Kang
#                                                       Han Zhu)
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

"""Convolution and downsampling modules: ConvolutionModule, SimpleDownsample, SimpleUpsample."""

import math

import torch
from torch import Tensor, nn

from zipvoice.models.modules.scaling import (
    ActivationDropoutAndLinear,
    Balancer,
    ScheduledFloat,
    Whiten,
)
from zipvoice.models.modules.zipformer._attention import (
    Identity,
    _whitening_schedule,
)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: shape of (N) or (N, T)
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        an Tensor of positional embeddings. shape of (N, dim) or (T, N, dim)
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


class SimpleDownsample(torch.nn.Module):
    """Does downsampling with attention, by weighted sum."""

    def __init__(self, downsample: int):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None  # will be set from training code

        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """Downsample the input tensor using attention-weighted sum.

        Args:
            src: Input tensor of shape (seq_len, batch_size, in_channels).

        Returns:
            Tensor of shape ((seq_len+downsample-1)//downsample, batch_size, channels).
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        if src.shape[0] != d_seq_len * ds:
            msg = f"Expected src.shape[0] == {d_seq_len * ds}, got {src.shape[0]}"
            raise ValueError(msg)

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    """A very simple form of upsampling that just repeats the input."""

    def __init__(self, upsample: int):
        super().__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        """Upsample the input tensor by repeating frames.

        Args:
            src: Input tensor of shape (seq_len, batch_size, num_channels).

        Returns:
            Tensor of shape (seq_len*upsample, batch_size, num_channels).
        """
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super().__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        if (kernel_size - 1) % 2 != 0:
            msg = f"kernel_size - 1 must be even (kernel_size must be odd), got {kernel_size}"
            raise ValueError(msg)

        bottleneck_dim = channels

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu). For
        # most layers the normal rms value of channels of x seems to be in the range 1
        # to 4, but sometimes, for some reason, for layer 0 the rms ends up being very
        # large, between 50 and 100 for different channels.  This will cause very peaky
        # and sparse derivatives for the sigmoid gating function, which will tend to
        # make the loss function not learn effectively.  (for most layers the average
        # absolute values are in the range 0.5..9.0, and the average p(x>0), i.e.
        # positive proportion, at the output of pointwise_conv1.output is around 0.35 to
        # 0.45 for different layers, which likely breaks down as 0.5 for the "linear"
        # half and 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that
        # if we constrain the rms values to a reasonable range via a constraint of
        # max_abs=10.0, it will be in a better position to start learning something,
        # i.e. to latch onto the correct range.
        self.balancer1 = Balancer(
            bottleneck_dim,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        if kernel_size % 2 != 1:
            msg = f"kernel_size must be odd, got {kernel_size}"
            raise ValueError(msg)

        self.depthwise_conv = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            groups=bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.balancer2 = Balancer(
            bottleneck_dim,
            channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwooshR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for the src keys per batch (optional):
                (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)
        return x


# end zipvoice/models/modules/zipformer/_conv.py
