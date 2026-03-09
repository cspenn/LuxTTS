# start zipvoice/models/modules/zipformer/_encoder.py
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

"""Encoder modules: TTSZipformer, Zipformer2Encoder, Zipformer2EncoderLayer, BypassModule, etc."""

import copy
import random

import torch
from torch import Tensor, nn

from zipvoice.models.modules.scaling import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
    FloatLike,
    Identity,
    ScaledLinear,
    ScheduledFloat,
    SwooshR,
    Whiten,
    limit_param_value,
)
from zipvoice.models.modules.zipformer._attention import (
    CompactRelPositionalEncoding,
    RelPositionMultiheadAttentionWeights,
    SelfAttention,
    _whitening_schedule,
)
from zipvoice.models.modules.zipformer._conv import (
    ConvolutionModule,
    SimpleDownsample,
    SimpleUpsample,
    timestep_embedding,
)


class BypassModule(nn.Module):
    """Implements a learnable bypass scale with randomized per-sequence layer-skipping.

    The bypass is limited during early stages of training
    to be close to "straight-through", i.e. to not do the bypass operation much
    initially, in order to force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),  # noqa: B008
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 corresponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(
                self.bypass_scale,
                min=float(self.scale_min),
                max=float(self.scale_max),
            )
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for
                # sequences on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor):
        """Compute bypass-scaled combination of original and processed tensor.

        Args:
            src_orig: tensor of shape (seq_len, batch_size, num_channels)
            src: tensor of shape (seq_len, batch_size, num_channels)

        Returns:
            Tensor with the same shape as src and src_orig.
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig) * bypass_scale


class FeedforwardModule(nn.Module):
    """Feedforward module in TTSZipformer model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """Like ConvolutionModule, but uses attention-weight multiplication instead of convolution.

    Refactored so that we use multiplication
       by attention weights (borrowed from the attention module) in place of actual
       convolution.  We also took out the second nonlinearity, the one after the
       attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at
        # 2.0, because we noticed that well-trained instances of this module have
        # abs-value before the sigmoid starting from about 3, and poorly-trained
        # instances of the module have smaller abs values before the sigmoid.
        self.balancer = Balancer(
            hidden_channels,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels, bias=True, initial_scale=0.05)

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """.

        Args:
            x: a Tensor of shape (seq_len, batch_size, num_channels)
            attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)

        Returns:
            a Tensor with the same shape as x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        if attn_weights.shape != (num_heads, batch_size, seq_len, seq_len):
            expected = (num_heads, batch_size, seq_len, seq_len)
            msg = f"Expected attn_weights.shape == {expected}, got {attn_weights.shape}"
            raise ValueError(msg)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x


class Zipformer2EncoderLayer(nn.Module):
    """A single encoder layer in the Zipformer2 architecture.

    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module (default=31).

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        use_conv: bool = True,
        attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),  # noqa: B008
        conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),  # noqa: B008
        const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),  # noqa: B008
        ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),  # noqa: B008
        ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),  # noqa: B008
        bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),  # noqa: B008
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, dropout)

        self.nonlin_attention = NonlinAttention(embed_dim, hidden_channels=3 * embed_dim // 4)

        self.use_conv = use_conv

        if self.use_conv:
            self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel)

            self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel)

        self.norm = BiasNorm(embed_dim)

        self.balancer1 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.2,
            max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(4.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.balancer2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.1,
            max_abs=4.0,
        )

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Tensor | None:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """Apply sequence-level dropout to x.

        x shape: (seq_len, batch_size, embed_dim).
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(  # noqa: C901, PLR0912, PLR0915
        self,
        src: Tensor,
        pos_emb: Tensor,
        time_emb: Tensor | None = None,
        attn_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
          src: the sequence to the encoder (required):
            shape (seq_len, batch_size, embedding_dim).
          pos_emb: (1, 2*seq_len-1, pos_emb_dim) or
            (batch_size, 2*seq_len-1, pos_emb_dim)
          time_emb: the embedding representing the current timestep
            shape (batch_size, embedding_dim) or (seq_len, batch_size, embedding_dim).
          attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len)
            or (seq_len, seq_len), interpreted as (batch_size, tgt_seq_len, src_seq_len)
            or (tgt_seq_len, src_seq_len). True means masked position. May be None.
          src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len);
            True means masked position.  May be None.

        Returns:
           A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )
        if time_emb is not None:
            src = src + time_emb

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        selected_attn_weights = attn_weights[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < float(self.const_attention_rate):  # noqa: S311
            # Make attention weights constant.  The intention is to
            # encourage these modules to do something similar to an
            # averaging-over-time operation.
            # only need the mask, can just use the 1st one and expand later
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
            selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))

        na = self.balancer_na(self.nonlin_attention(src, selected_attn_weights))

        src = src + (na if self_attn_dropout_mask is None else na * self_attn_dropout_mask)

        self_attn = self.self_attn1(src, attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if self.use_conv:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                conv_skip_rate = 0.0
            else:
                conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0

            if time_emb is not None:
                src = src + time_emb

            src = src + self.sequence_dropout(
                self.conv_module1(
                    src,
                    src_key_padding_mask=src_key_padding_mask,
                ),
                conv_skip_rate,
            )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)), ff2_skip_rate)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if self.use_conv:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                conv_skip_rate = 0.0
            else:
                conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0

            if time_emb is not None:
                src = src + time_emb

            src = src + self.sequence_dropout(
                self.conv_module2(
                    src,
                    src_key_padding_mask=src_key_padding_mask,
                ),
                conv_skip_rate,
            )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.balancer_ff3(self.feed_forward3(src)), ff3_skip_rate)

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers with positional encoding.

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        embed_dim: int,
        time_embed_dim: int,
        pos_dim: int,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15, length_factor=1.0)
        if time_embed_dim != -1:
            self.time_emb = nn.Sequential(
                SwooshR(),
                nn.Linear(time_embed_dim, embed_dim),
            )
        else:
            self.time_emb = None

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        if not (0 <= warmup_begin <= warmup_end):
            msg = f"Expected 0 <= warmup_begin <= warmup_end, got warmup_begin={warmup_begin}, warmup_end={warmup_end}"
            raise ValueError(msg)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        time_emb: Tensor | None = None,
        attn_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required):
                shape (seq_len, batch_size, embedding_dim).
            time_emb: the embedding representing the current timestep:
                shape  (batch_size, embedding_dim)
                or (seq_len, batch_size, embedding_dim) .
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len)
                or (seq_len, seq_len), interpreted as
                (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len);
                True means masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)
        if self.time_emb is not None:
            if time_emb is None:
                msg = "time_emb must not be None when self.time_emb is set"
                raise ValueError(msg)
            time_emb = self.time_emb(time_emb)
        else:
            if time_emb is not None:
                msg = "time_emb must be None when self.time_emb is not set"
                raise ValueError(msg)

        output = src

        for _i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                time_emb=time_emb,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        return output


class DownsampledZipformer2Encoder(nn.Module):
    r"""DownsampledZipformer2Encoder is a zipformer encoder at a reduced frame rate.

    Evaluated after convolutional downsampling, and then upsampled again at the output,
    combined with the origin input, so that the output has the same shape as the input.
    """

    def __init__(self, encoder: nn.Module, dim: int, downsample: int):
        super().__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(downsample)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        time_emb: Tensor | None = None,
        attn_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required):
                shape (seq_len, batch_size, embedding_dim).
            time_emb: the embedding representing the current timestep:
                shape  (batch_size, embedding_dim)
                or (seq_len, batch_size, embedding_dim) .
            feature_mask: something that broadcasts with src, that we'll multiply `src`
                by at every layer: if a Tensor, likely of shape
                (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len)
                or (seq_len, seq_len), interpreted as
                (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len);
                True means masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if time_emb is not None and time_emb.dim() == 3:
            time_emb = time_emb[::ds]
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[..., ::ds]

        src = self.encoder(
            src,
            time_emb=time_emb,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)


class TTSZipformer(nn.Module):
    """TTS-oriented Zipformer encoder with configurable stacks and downsampling.

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
        use_time_embed: (bool): if True, take time embedding as an additional input.
        time_embed_dim: (int): the dimension of the time embedding.
        use_guidance_scale_embed (bool): if True, take guidance scale embedding as
            an additional input.
        guidance_scale_embed_dim: (int): the dimension of the guidance scale embedding.
    """

    def __init__(  # noqa: PLR0913, C901, PLR0915
        self,
        in_dim: int,
        out_dim: int,
        downsampling_factor: int | tuple[int, ...] = (2, 4),
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
        use_guidance_scale_embed: bool = False,
        guidance_scale_embed_dim: int = 192,
        use_conv: bool = True,
    ) -> None:
        super().__init__()

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
        self.use_guidance_scale_embed = use_guidance_scale_embed

        self.time_embed_dim = time_embed_dim
        if self.use_time_embed:
            if time_embed_dim == -1:
                msg = "time_embed_dim must not be -1 when use_time_embed is True"
                raise ValueError(msg)
        else:
            time_embed_dim = -1
        self.guidance_scale_embed_dim = guidance_scale_embed_dim

        self.in_proj = nn.Linear(in_dim, encoder_dim)
        self.out_proj = nn.Linear(encoder_dim, out_dim)

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

        if self.use_guidance_scale_embed:
            self.guidance_scale_embed = ScaledLinear(
                guidance_scale_embed_dim,
                time_embed_dim,
                bias=False,
                initial_scale=0.1,
            )
        else:
            self.guidance_scale_embed = None

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        padding_mask: Tensor | None = None,
        guidance_scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the TTS Zipformer forward pass.

        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          t:
            A t tensor of shape (batch_size,) or (batch_size, seq_len)
          padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
          guidance_scale:
            The guidance scale in classifier-free guidance of distillation model.

        Returns:
          Return the output embeddings. its shape is
            (batch_size, output_seq_len, encoder_dim)
        """
        x = x.permute(1, 0, 2)
        x = self.in_proj(x)

        if t is not None:
            if t.dim() != 1 and t.dim() != 2:
                msg = f"Expected t.dim() == 1 or 2, got {t.dim()}, shape={t.shape}"
                raise ValueError(msg)
            time_emb = timestep_embedding(t, self.time_embed_dim)
            if guidance_scale is not None:
                if guidance_scale.dim() != 1 and guidance_scale.dim() != 2:
                    msg = (
                        f"Expected guidance_scale.dim() == 1 or 2, "
                        f"got {guidance_scale.dim()}, shape={guidance_scale.shape}"
                    )
                    raise ValueError(msg)
                guidance_scale_emb = self.guidance_scale_embed(
                    timestep_embedding(guidance_scale, self.guidance_scale_embed_dim)
                )
                time_emb = time_emb + guidance_scale_emb
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
        x = self.out_proj(x)
        x = x.permute(1, 0, 2)
        return x


# end zipvoice/models/modules/zipformer/_encoder.py
