# start zipvoice/models/modules/zipformer/_attention.py
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

"""Attention modules: CompactRelPositionalEncoding, RelPositionMultiheadAttentionWeights, SelfAttention."""

import copy
import math
import random

import structlog
import torch
from torch import Tensor, nn

if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
else:
    DEVICE_TYPE = "cpu"

log = structlog.get_logger()

from zipvoice.models.modules.scaling import (  # noqa: E402
    Balancer,
    Dropout2,
    FloatLike,
    Identity,
    ScaledLinear,
    ScheduledFloat,
    Whiten,
    penalize_abs_values_gt,
    softmax,
)


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


class CompactRelPositionalEncoding(torch.nn.Module):
    """Compact relative positional encoding that compresses large offsets logarithmically.

    This version is "compact" meaning it is able
    to encode the important information about the relative position in a relatively
    small number of dimensions. The goal is to make it so that small differences between
    large relative offsets (e.g. 1000 vs. 1001) make very little difference to the
    embedding.   Such differences were potentially important when encoding absolute
    position, but not important when encoding relative position because there is now no
    need to compare two large offsets with each other.

    Our embedding works by projecting the interval [-infinity,infinity] to a finite
    interval using the atan() function, before doing the Fourier transform of that fixed
    interval.  The atan() function would compress the "long tails" too small, making it
    hard to distinguish between different magnitudes of large offsets, so we use a
    logarithmic function to compress large offsets to a smaller range before applying
    atan(). Scalings are chosen in such a way that the embedding can clearly distinguish
    individual offsets as long as they are quite close to the origin, e.g. abs(offset)
    <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim % 2 != 0:
            msg = f"embed_dim must be even, got {embed_dim}"
            raise ValueError(msg)
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        if length_factor < 1.0:
            msg = f"length_factor must be >= 1.0, got {length_factor}"
            raise ValueError(msg)
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        """Reset the positional encodings."""
        T = x.size(0) + left_context_len

        if self.pe is not None and self.pe.size(0) >= T * 2 - 1:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return

        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more
        # resolution for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity
        # to infinity; but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which is
        # important.
        x_compressed = (
            compression_length * x.sign() * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 ,
        #  atan(x))
        x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        """Create positional encoding.

        Args:
            x (Tensor): Input tensor (time, batch, `*`).
            left_context_len: (int): Length of cached left context.

        Returns:
            positional embedding, of shape (batch, left_context_len + 2*time-1, `*`).
        """
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        # length of positive side: x.size(0) + left_context_len
        # length of negative side: x.size(0)
        pos_emb = self.pe[
            self.pe.size(0) // 2 - x_size_left + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :,
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Computes multi-head attention weights with relative position encoding.

    Various other modules consume the resulting attention weights:
    see, for example, the SimpleAttention module which allows you to compute
    conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language
        Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),  # noqa: B008
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(
            embed_dim,
            in_proj_dim,
            bias=True,
            initial_scale=query_head_dim**-0.25,
        )

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be sufficient to fix the problem.
        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=0.0,
            max_abs=100.0,
            prob=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05)

        # the following are for diagnostics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(  # noqa: C901, PLR0912, PLR0915
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Compute relative-position multi-head attention weights.

        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 1, pos_dim)
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).
                Positions that are True in this mask will be ignored as sources in the
                attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or
                (batch_size, seq_len, seq_len), interpreted as
                ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.

        Returns:
           a tensor of attention weights, of
            shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        if p.shape[-1] != num_heads * pos_head_dim:
            msg = (
                f"Expected p.shape[-1] == num_heads * pos_head_dim, "
                f"got p.shape[-1]={p.shape[-1]}, num_heads={num_heads}, pos_head_dim={pos_head_dim}"
            )
            raise ValueError(msg)

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # We can't put random.random() in the same line
            use_pos_scores = True
        elif not self.training or random.random() >= float(self.pos_emb_skip_rate):  # noqa: S311
            use_pos_scores = True

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head,
            #  batch, time1, seq_len2) [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of
            # pos_scores from relative to absolute position.  I don't know whether I
            # might have got the time-offsets backwards or not, but let this code define
            # which way round it is supposed to be.
            if torch.jit.is_tracing():
                (num_heads, batch_size, time1, n) = pos_scores.shape
                rows = torch.arange(start=time1 - 1, end=-1, step=-1)
                cols = torch.arange(seq_len)
                rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads, batch_size, time1, seq_len)
            else:
                pos_scores = pos_scores.as_strided(
                    (num_heads, batch_size, seq_len, seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (seq_len - 1),
                )

            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:  # noqa: S311
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores, limit=25.0, penalty=1.0e-04, name=self.name)

        if attn_scores.shape != (num_heads, batch_size, seq_len, seq_len):
            expected = (num_heads, batch_size, seq_len, seq_len)
            msg = f"Expected attn_scores.shape == {expected}, got {attn_scores.shape}"
            raise ValueError(msg)

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                msg = f"Expected attn_mask.dtype == torch.bool, got {attn_mask.dtype}"
                raise ValueError(msg)
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, seq_len):
                msg = f"Expected key_padding_mask.shape == {(batch_size, seq_len)}, got {key_padding_mask.shape}"
                raise ValueError(msg)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:  # noqa: S311
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        return attn_weights

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad(), torch.amp.autocast(DEVICE_TYPE, enabled=False):
            attn_weights = attn_weights.to(torch.float32)
            attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(dim=-1).mean(dim=(1, 2))
            log.debug(
                "attn_weights_entropy",
                name=self.name,
                entropy=str(attn_weights_entropy),
            )


class SelfAttention(nn.Module):
    """Simplest possible attention module using already-computed attention weights.

    Works with already-computed
    attention weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim,
            embed_dim,
            bias=True,
            initial_scale=0.05,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """Apply self-attention using pre-computed attention weights.

        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
          attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
            with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
            attn_weights.sum(dim=-1) == 1.

        Returns:
           a tensor with the same shape as x.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        if attn_weights.shape != (num_heads, batch_size, seq_len, seq_len):
            expected = (num_heads, batch_size, seq_len, seq_len)
            msg = f"Expected attn_weights.shape == {expected}, got {attn_weights.shape}"
            raise ValueError(msg)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x


# end zipvoice/models/modules/zipformer/_attention.py
