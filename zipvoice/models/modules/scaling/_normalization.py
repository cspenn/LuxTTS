# start zipvoice/models/modules/scaling/_normalization.py
# Copyright    2022-2025  Xiaomi Corp.        (authors: Daniel Povey
#                                                       Wei Kang)
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

"""Gradient-based normalization and whitening modules: Balancer and Whiten."""

import math
import random

import structlog
import torch
import torch.nn as nn
from torch import Tensor

from zipvoice.models.modules.scaling._activations import (
    DEVICE_TYPE,
    _no_op,
    get_memory_allocated,
    with_loss,
)
from zipvoice.models.modules.scaling._piecewise import (
    CutoffEstimator,
    FloatLike,
    ScheduledFloat,
)

log = structlog.get_logger()


class BalancerFunction(torch.autograd.Function):
    """Autograd function that applies gradient-based balancing to channels."""

    @staticmethod
    def forward(  # noqa: PLR0913
        ctx,
        x: Tensor,
        min_mean: float,
        max_mean: float,
        min_rms: float,
        max_rms: float,
        grad_scale: float,
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        ctx.save_for_backward(x)
        ctx.config = (
            min_mean,
            max_mean,
            min_rms,
            max_rms,
            grad_scale,
            channel_dim,
        )
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> tuple[Tensor, None, None, None, None, None]:
        (x,) = ctx.saved_tensors
        (
            min_mean,
            max_mean,
            min_rms,
            max_rms,
            grad_scale,
            channel_dim,
        ) = ctx.config

        try:
            with torch.enable_grad(), torch.amp.autocast(DEVICE_TYPE, enabled=False):
                x = x.to(torch.float32)
                x = x.detach()
                x.requires_grad = True
                mean_dims = [i for i in range(x.ndim) if i != channel_dim]
                uncentered_var = (x**2).mean(dim=mean_dims, keepdim=True)
                mean = x.mean(dim=mean_dims, keepdim=True)
                stddev = (uncentered_var - (mean * mean)).clamp(min=1.0e-20).sqrt()
                rms = uncentered_var.clamp(min=1.0e-20).sqrt()

                m = mean / stddev
                # part of loss that relates to mean / stddev
                m_loss = (m - m.clamp(min=min_mean, max=max_mean)).abs()

                # put a much larger scale on the RMS-max-limit loss, so that if both
                # it and the m_loss are violated we fix the RMS loss first.
                rms_clamped = rms.clamp(min=min_rms, max=max_rms)
                r_loss = (rms_clamped / rms).log().abs()

                loss = m_loss + r_loss

                loss.backward(gradient=torch.ones_like(loss))
                loss_grad = x.grad
                loss_grad_rms = (loss_grad**2).mean(dim=mean_dims, keepdim=True).sqrt().clamp(min=1.0e-20)

                loss_grad = loss_grad * (grad_scale / loss_grad_rms)

                x_grad_float = x_grad.to(torch.float32)
                # scale each element of loss_grad by the absolute value of the
                # corresponding element of x_grad, which we view as a noisy estimate
                # of its magnitude for that (frame and dimension).  later we can
                # consider factored versions.
                x_grad_mod = x_grad_float + (x_grad_float.abs() * loss_grad)
                x_grad = x_grad_mod.to(x_grad.dtype)
        except Exception as e:  # P8: intentional — training loop must survive per-batch backward errors
            log.info(
                "balancer_backward_exception",
                error=str(e),
                size=list(x_grad.shape),
            )

        return x_grad, None, None, None, None, None, None


class Balancer(torch.nn.Module):
    """Modifies backpropped derivatives to encourage positive channel activations.

    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
         prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: FloatLike = 0.05,
        max_positive: FloatLike = 0.95,
        min_abs: FloatLike = 0.2,
        max_abs: FloatLike = 100.0,
        grad_scale: FloatLike = 0.04,
        prob: FloatLike | None = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        # 5% of the time we will return and do nothing because memory usage is
        # too high.
        self.mem_cutoff = CutoffEstimator(0.05)

        # actually self.num_channels is no longer needed except for an assertion.
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        if (
            torch.jit.is_scripting()
            or not x.requires_grad
            or ((x.is_cuda or x.device.type == "mps") and self.mem_cutoff(get_memory_allocated()))
        ):
            return _no_op(x)

        prob = float(self.prob)
        if random.random() < prob:  # noqa: S311
            # The following inner-functions convert from the way we historically
            # specified these limitations, as limits on the absolute value and the
            # proportion of positive values, to limits on the RMS value and
            # the (mean / stddev).
            def _abs_to_rms(x):
                # for normally distributed data, if the expected absolute value is x,
                # the expected rms value will be sqrt(pi/2) * x.
                return 1.25331413732 * x

            def _proportion_positive_to_mean(x):
                def _atanh(x):
                    eps = 1.0e-10
                    # eps is to prevent crashes if x is exactly 0 or 1.
                    # we'll just end up returning a fairly large value.
                    return (math.log(1 + x + eps) - math.log(1 - x + eps)) / 2.0

                def _approx_inverse_erf(x):
                    # 1 / (sqrt(pi) * ln(2)),
                    # see https://math.stackexchange.com/questions/321569/
                    # approximating-the-error-function-erf-by-analytical-functions
                    # this approximation is extremely crude and gets progressively worse
                    # for x very close to -1 or +1, but we mostly care about the
                    # "middle" region
                    # e.g. _approx_inverse_erf(0.05) = 0.0407316414078772,
                    # and math.erf(0.0407316414078772) = 0.045935330944660666,
                    # which is pretty close to 0.05.
                    return 0.8139535143 * _atanh(x)

                # first convert x from the range 0..1 to the range -1..1 which the error
                # function returns
                x = -1 + (2 * x)
                return _approx_inverse_erf(x)

            min_mean = _proportion_positive_to_mean(float(self.min_positive))
            max_mean = _proportion_positive_to_mean(float(self.max_positive))
            min_rms = _abs_to_rms(float(self.min_abs))
            max_rms = _abs_to_rms(float(self.max_abs))
            grad_scale = float(self.grad_scale)

            if x.shape[self.channel_dim] != self.num_channels:
                msg = f"Expected x.shape[{self.channel_dim}] == {self.num_channels}, got {x.shape[self.channel_dim]}"
                raise ValueError(msg)

            return BalancerFunction.apply(
                x,
                min_mean,
                max_mean,
                min_rms,
                max_rms,
                grad_scale,
                self.channel_dim,
            )
        else:
            return _no_op(x)


def penalize_abs_values_gt(x: Tensor, limit: float, penalty: float, name: str = None) -> Tensor:
    """Returns x unmodified, but adds a gradient penalty for absolute values exceeding limit.

    E.g. if limit == 10.0, then if x has any values over 10 it will get a penalty.

    Caution: the value of this penalty will be affected by grad scaling used
    in automatic mixed precision training.  For this reasons we use this,
    it shouldn't really matter, or may even be helpful; we just use this
    to disallow really implausible values of scores to be given to softmax.

    The name is for randomly printed debug info.
    """
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    # The following is a memory efficient way to penalize the absolute values of
    # x that's over the limit.  (The memory efficiency comes when you think
    # about which items torch needs to cache for the autograd, and which ones it
    # can throw away).  The numerical value of aux_loss as computed here will
    # actually be larger than it should be, by limit * over_limit.sum(), but it
    # has the same derivative as the real aux_loss which is penalty * (x.abs() -
    # limit).relu().
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    # note: we don't do sum() here on aux)_loss, but it's as if we had done
    # sum() due to how with_loss() works.
    x = with_loss(x, aux_loss, name)
    # you must use x for something, or this will be ineffective.
    return x


def _diag(x: Tensor):  # like .diag(), but works for tensors with 3 dims.
    """Extract diagonal, supporting 2D and 3D tensors."""
    if x.ndim == 2:
        return x.diag()
    else:
        (batch, dim, dim) = x.shape
        x = x.reshape(batch, dim * dim)
        x = x[:, :: dim + 1]
        if x.shape != (batch, dim):
            msg = f"Expected x.shape == {(batch, dim)}, got {x.shape}"
            raise ValueError(msg)
        return x


def _whitening_metric(x: Tensor, num_groups: int):
    """Compute the "whitening metric" measuring deviation from ideal whiteness.

    A value which will be 1.0 if all the eigenvalues of the centered feature
    covariance are the same within each group's covariance matrix and also between
    groups.

    Args:
        x: a Tensor of shape (*, num_channels)
        num_groups: the number of groups of channels, a number >=1 that divides
            num_channels

    Returns:
        Returns a scalar Tensor that will be 1.0 if the data is "perfectly white" and
        greater than 1.0 otherwise.
    """
    if x.dtype == torch.float16:
        msg = "x.dtype must not be torch.float16"
        raise ValueError(msg)
    x = x.reshape(-1, x.shape[-1])
    (num_frames, num_channels) = x.shape
    if num_channels % num_groups != 0:
        msg = f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        raise ValueError(msg)
    channels_per_group = num_channels // num_groups
    x = x.reshape(num_frames, num_groups, channels_per_group).transpose(0, 1)
    # x now has shape (num_groups, num_frames, channels_per_group)
    # subtract the mean so we use the centered, not uncentered, covariance.
    # My experience has been that when we "mess with the gradients" like this,
    # it's better not do anything that tries to move the mean around, because
    # that can easily cause instability.
    x = x - x.mean(dim=1, keepdim=True)
    # x_covar: (num_groups, channels_per_group, channels_per_group)
    x_covar = torch.matmul(x.transpose(1, 2), x)
    x_covar_mean_diag = _diag(x_covar).mean()
    # the following expression is what we'd get if we took the matrix product
    # of each covariance and measured the mean of its trace, i.e.
    # the same as _diag(torch.matmul(x_covar, x_covar)).mean().
    x_covarsq_mean_diag = (x_covar**2).sum() / (num_groups * channels_per_group)
    # this metric will be >= 1.0; the larger it is, the less 'white' the data was.
    metric = x_covarsq_mean_diag / (x_covar_mean_diag**2 + 1.0e-20)
    return metric


class WhiteningPenaltyFunction(torch.autograd.Function):
    """Autograd function that applies a whitening penalty in the backward pass."""

    @staticmethod
    def forward(ctx, x: Tensor, module: nn.Module) -> Tensor:
        ctx.save_for_backward(x)
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x_orig,) = ctx.saved_tensors
        w = ctx.module

        try:
            with torch.enable_grad(), torch.amp.autocast(DEVICE_TYPE, enabled=False):
                x_detached = x_orig.to(torch.float32).detach()
                x_detached.requires_grad = True

                metric = _whitening_metric(x_detached, w.num_groups)

                if random.random() < 0.005:  # noqa: S311
                    log.debug(
                        "whitening",
                        name=w.name,
                        num_groups=w.num_groups,
                        num_channels=x_orig.shape[-1],
                        metric=f"{metric.item():.2f}",
                        limit=float(w.whitening_limit),
                    )

                if metric < float(w.whitening_limit):
                    w.prob = w.min_prob
                    return x_grad, None
                else:
                    w.prob = w.max_prob
                    metric.backward()
                    penalty_grad = x_detached.grad
                    scale = w.grad_scale * (x_grad.to(torch.float32).norm() / (penalty_grad.norm() + 1.0e-20))
                    penalty_grad = penalty_grad * scale
                    return x_grad + penalty_grad.to(x_grad.dtype), None
        except Exception as e:  # P8: intentional — training loop must survive per-batch backward errors
            log.info(
                "whiten_backward_exception",
                error=str(e),
                size=list(x_grad.shape),
            )
        return x_grad, None


class Whiten(nn.Module):
    """Applies a whitening penalty in the backward pass to encourage decorrelated features."""

    def __init__(
        self,
        num_groups: int,
        whitening_limit: FloatLike,
        prob: float | tuple[float, float],
        grad_scale: FloatLike,
    ):
        """Initialize the Whiten module.

        Args:
            num_groups: the number of groups to divide the channel dim into before
              whitening.  We will attempt to make the feature covariance
              within each group, after mean subtraction, as "white" as possible,
              while having the same trace across all groups.
            whitening_limit: a value greater than 1.0, that dictates how much
              freedom we have to violate the constraints.  1.0 would mean perfectly
              white, with exactly the same trace across groups; larger values
              give more freedom.  E.g. 2.0.
            prob: the probability with which we apply the gradient modification
              (also affects the grad scale).  May be supplied as a float,
              or as a pair (min_prob, max_prob).
            grad_scale: determines the scale on the gradient term from this object,
              relative to the rest of the gradient on the attention weights.
              E.g. 0.02 (you may want to use smaller values than this if prob is large)
        """
        super().__init__()
        if num_groups < 1:
            msg = f"num_groups must be >= 1, got {num_groups}"
            raise ValueError(msg)
        if float(whitening_limit) < 1:
            msg = f"whitening_limit must be >= 1, got {whitening_limit}"
            raise ValueError(msg)
        if grad_scale < 0:
            msg = f"grad_scale must be >= 0, got {grad_scale}"
            raise ValueError(msg)
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.grad_scale = grad_scale

        if isinstance(prob, float):
            prob = (prob, prob)
        (self.min_prob, self.max_prob) = prob
        if not (0 < self.min_prob <= self.max_prob <= 1):
            msg = f"Expected 0 < min_prob <= max_prob <= 1, got min_prob={self.min_prob}, max_prob={self.max_prob}"
            raise ValueError(msg)
        self.prob = self.max_prob
        self.name = None  # will be set in training loop

    def forward(self, x: Tensor) -> Tensor:
        """In the forward pass, returns the input unmodified.

        In the backward pass, it will modify the gradients to ensure that the
        distribution in each group has close to (lambda times I) as the covariance
        after mean subtraction, with the same lambda across groups.
        For whitening_limit > 1, there will be more freedom to violate this
        constraint.

        Args:
           x: the input of shape (*, num_channels)

        Returns:
            x, unmodified.   You should make sure
        you use the returned value, or the graph will be freed
        and nothing will happen in backprop.
        """
        grad_scale = float(self.grad_scale)
        if not x.requires_grad or random.random() > self.prob or grad_scale == 0:  # noqa: S311
            return _no_op(x)
        else:
            return WhiteningPenaltyFunction.apply(x, self)


# end zipvoice/models/modules/scaling/_normalization.py
