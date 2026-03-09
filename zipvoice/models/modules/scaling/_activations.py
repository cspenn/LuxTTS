# start zipvoice/models/modules/scaling/_activations.py
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

"""Activation functions, dropout utilities, and low-level scaling helpers."""

import random
import sys

import structlog

log = structlog.get_logger()

try:
    import k2
except Exception as e:  # P8: intentional — k2 is optional; code falls back to PyTorch implementation
    log.warning(
        "k2_import_failed",
        error=str(e),
        note=(
            "Swoosh functions will fallback to PyTorch implementation, "
            "leading to slower speed and higher memory consumption."
        ),
    )
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch import Tensor  # noqa: E402

from zipvoice.models.modules.scaling._piecewise import FloatLike  # noqa: E402

if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
else:
    DEVICE_TYPE = "cpu"


def get_memory_allocated():
    """Return currently allocated device memory in bytes (0 for CPU)."""
    if DEVICE_TYPE == "cuda":
        return torch.cuda.memory_allocated()
    elif DEVICE_TYPE == "mps":
        return torch.mps.current_allocated_memory()
    else:
        return 0


def custom_amp_decorator(dec, cuda_amp_deprecated):
    """Wrap an AMP decorator to handle the deprecated cuda.amp vs amp API."""

    def decorator(func):
        return dec(func) if not cuda_amp_deprecated else dec(device_type=DEVICE_TYPE)(func)

    return decorator


if hasattr(torch.amp, "custom_fwd"):
    _deprecated = True
    from torch.amp import custom_bwd, custom_fwd
else:
    _deprecated = False
    from torch.cuda.amp import custom_bwd, custom_fwd

custom_fwd = custom_amp_decorator(custom_fwd, _deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, _deprecated)


def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    """ONNX-compatible logaddexp."""
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


# RuntimeError: Exporting the operator logaddexp to ONNX opset version
# 14 is not supported. Please feel free to request support or submit
# a pull request on PyTorch GitHub.
#
# The following function is to solve the above error when exporting
# models to ONNX via torch.jit.trace()
def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    """Logaddexp that is safe for ONNX export and torch.jit scripting/tracing."""
    # Caution(fangjun): Put torch.jit.is_scripting() before
    # torch.onnx.is_in_onnx_export();
    # otherwise, it will cause errors for torch.jit.script().
    #
    # torch.logaddexp() works for both torch.jit.script() and
    # torch.jit.trace() but it causes errors for ONNX export.
    #
    if torch.jit.is_scripting():
        # Note: We cannot use torch.jit.is_tracing() here as it also
        # matches torch.onnx.export().
        return torch.logaddexp(x, y)
    elif torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        # for torch.jit.trace()
        return torch.logaddexp(x, y)


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


class SoftmaxFunction(torch.autograd.Function):
    """Tries to handle half-precision derivatives in a randomized way.

    Should be more accurate for training than the default behavior.
    """

    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        # if x dtype is float16, x.softmax() returns a float32 because
        # (presumably) that op does not support float16, and autocast
        # is enabled.
        if torch.is_autocast_enabled():
            ans = ans.to(torch.float16)
        ctx.save_for_backward(ans)
        ctx.x_dtype = x.dtype
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        with torch.amp.autocast(DEVICE_TYPE, enabled=False):
            ans_grad = ans_grad.to(torch.float32)
            ans = ans.to(torch.float32)
            x_grad = ans_grad * ans
            x_grad = x_grad - ans * x_grad.sum(dim=ctx.dim, keepdim=True)
            return x_grad, None


def softmax(x: Tensor, dim: int):
    """Memory-efficient softmax that handles half-precision derivatives better."""
    if not x.requires_grad or torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim=dim)

    return SoftmaxFunction.apply(x, dim)


class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(  # noqa: PLR0913
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        if bias.ndim != 1:
            msg = f"Expected bias.ndim == 1, got {bias.ndim}"
            raise ValueError(msg)
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        x = ans_or_x / scales if ctx.store_output_for_backprop else ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            scales = (torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """A simpler, and hopefully cheaper, replacement for LayerNorm.

    The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.


    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[self.channel_dim] != self.num_channels:
            msg = f"Expected x.shape[{self.channel_dim}] == {self.num_channels}, got {x.shape[self.channel_dim]}"
            raise ValueError(msg)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(
            x,
            self.bias,
            log_scale,
            self.channel_dim,
            self.store_output_for_backprop,
        )


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:  # noqa: PLR0913
    """Behaves like a constructor of a modified version of nn.Linear.

    Gives an easy way to set the default initial parameter scale.

    Args:
        *args: Accepts the standard args that nn.Linear accepts
            e.g. in_features, out_features, bias=False.
        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
        **kwargs: Additional keyword arguments passed to nn.Linear.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


class WithLoss(torch.autograd.Function):
    """Adds an auxiliary loss y to the graph without affecting the forward value of x."""

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        if random.random() < 0.002 and name is not None:  # noqa: S311
            loss_sum = y.sum().item()
            log.debug("with_loss", name=name, loss_sum=f"{loss_sum:.3e}")
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    """Returns x but adds y.sum() to the loss function."""
    return WithLoss.apply(x, y, name)


class LimitParamValue(torch.autograd.Function):
    """Autograd function that limits a parameter's range by reversing gradients."""

    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        if max < min:
            msg = f"max ({max}) must be >= min ({min})"
            raise ValueError(msg)
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0)
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:  # noqa: S311
        return LimitParamValue.apply(x, min, max)
    else:
        return x


# Identity more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
class Identity(torch.nn.Module):
    """Identity module that is friendlier to backward hooks than nn.Identity."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _no_op(x)


# Dropout2 is just like normal dropout, except it supports schedules
# on the dropout rates.
class Dropout2(nn.Module):
    """Dropout that supports scheduled dropout rates."""

    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)


class MulForDropout3(torch.autograd.Function):
    """Returns (x * y * alpha) where alpha is a float and y doesn't require grad and is zero-or-one."""

    @staticmethod
    @custom_fwd
    def forward(ctx, x, y, alpha):
        if y.requires_grad:
            msg = "y must not require grad"
            raise ValueError(msg)
        ans = x * y * alpha
        ctx.save_for_backward(ans)
        ctx.alpha = alpha
        return ans

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        (ans,) = ctx.saved_tensors
        x_grad = ctx.alpha * ans_grad * (ans != 0)
        return x_grad, None, None


# Dropout3 is just like normal dropout, except it supports schedules on the dropout
# rates, and it lets you choose one dimension to share the dropout mask over
class Dropout3(nn.Module):
    """Dropout with shared mask across a dimension and scheduled dropout rate support."""

    def __init__(self, p: FloatLike, shared_dim: int):
        super().__init__()
        self.p = p
        self.shared_dim = shared_dim

    def forward(self, x: Tensor) -> Tensor:
        p = float(self.p)
        if not self.training or p == 0:
            return _no_op(x)
        scale = 1.0 / (1 - p)
        rand_shape = list(x.shape)
        rand_shape[self.shared_dim] = 1
        mask = torch.rand(*rand_shape, device=x.device) > p
        ans = MulForDropout3.apply(x, mask, scale)
        return ans


class SwooshLFunction(torch.autograd.Function):
    """swoosh_l(x) =  log(1 + exp(x-4)) - 0.08*x - 0.035."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        coeff = -0.08

        with torch.amp.autocast(DEVICE_TYPE, enabled=False), torch.enable_grad():
            x = x.detach()
            x.requires_grad = True
            y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035

            if not requires_grad:
                return y

            y.backward(gradient=torch.ones_like(y))

            grad = x.grad
            floor = coeff
            ceil = 1.0 + coeff + 0.005

            d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(grad)

            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
            if x.dtype == torch.float16 or torch.is_autocast_enabled():
                y = y.to(torch.float16)
            return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        coeff = -0.08
        floor = coeff
        ceil = 1.0 + coeff + 0.005
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshL(torch.nn.Module):
    """SwooshL activation: log(1 + exp(x-4)) - 0.08*x - 0.035."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 4.0) - 0.08 * x - 0.035
        elif "k2" not in sys.modules:
            return SwooshLFunction.apply(x)
        else:
            if not x.requires_grad:
                return k2.swoosh_l_forward(x)
            else:
                return k2.swoosh_l(x)


class SwooshLOnnx(torch.nn.Module):
    """ONNX-safe SwooshL activation."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 4.0) - 0.08 * x - 0.035


class SwooshRFunction(torch.autograd.Function):
    """swoosh_r(x) =  log(1 + exp(x-1)) - 0.08*x - 0.313261687.

    derivatives are between -0.08 and 0.92.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad

        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        with torch.amp.autocast(DEVICE_TYPE, enabled=False), torch.enable_grad():
            x = x.detach()
            x.requires_grad = True
            y = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

            if not requires_grad:
                return y
            y.backward(gradient=torch.ones_like(y))

            grad = x.grad
            floor = -0.08
            ceil = 0.925

            d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(grad)

            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
            if x.dtype == torch.float16 or torch.is_autocast_enabled():
                y = y.to(torch.float16)
            return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.08
        ceil = 0.925
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshR(torch.nn.Module):
    """SwooshR activation: log(1 + exp(x-1)) - 0.08*x - 0.313261687."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687
        elif "k2" not in sys.modules:
            return SwooshRFunction.apply(x)
        else:
            if not x.requires_grad:
                return k2.swoosh_r_forward(x)
            else:
                return k2.swoosh_r(x)


class SwooshROnnx(torch.nn.Module):
    """ONNX-safe SwooshR activation."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 1.0) - 0.08 * x - 0.313261687


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: Tensor):
    """Simple SwooshL without custom backprop (for use inside ActivationDropoutAndLinear)."""
    with torch.amp.autocast(DEVICE_TYPE, enabled=False):
        x = x.to(torch.float32)
        x_offset = x - 4.0
        log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
        log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
        return log_sum - 0.08 * x - 0.035


# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: Tensor):
    """Simple SwooshR without custom backprop (for use inside ActivationDropoutAndLinear)."""
    with torch.amp.autocast(DEVICE_TYPE, enabled=False):
        x = x.to(torch.float32)
        x_offset = x - 1.0
        log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
        log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
        return log_sum - 0.08 * x - 0.313261687


class ActivationDropoutAndLinearFunction(torch.autograd.Function):
    """Memory-efficient fused activation + dropout + linear (uses k2 kernels)."""

    @staticmethod
    @custom_fwd
    def forward(  # noqa: PLR0913
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        activation: str,
        dropout_p: float,
        dropout_shared_dim: int | None,
    ):
        if dropout_p != 0.0:
            dropout_shape = list(x.shape)
            if dropout_shared_dim is not None:
                dropout_shape[dropout_shared_dim] = 1
            # else it won't be very memory efficient.
            dropout_mask = (1.0 / (1.0 - dropout_p)) * (
                torch.rand(*dropout_shape, device=x.device, dtype=x.dtype) > dropout_p
            )
        else:
            dropout_mask = None

        ctx.save_for_backward(x, weight, bias, dropout_mask)

        ctx.activation = activation

        forward_activation_dict = {
            "SwooshL": k2.swoosh_l_forward,
            "SwooshR": k2.swoosh_r_forward,
        }
        # it will raise a KeyError if this fails.  This will be an error.  We let it
        # propagate to the user.
        activation_func = forward_activation_dict[activation]
        x = activation_func(x)
        if dropout_mask is not None:
            x = x * dropout_mask
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias, dropout_mask) = saved

        forward_and_deriv_activation_dict = {
            "SwooshL": k2.swoosh_l_forward_and_deriv,
            "SwooshR": k2.swoosh_r_forward_and_deriv,
        }
        # the following lines a KeyError if the activation is unrecognized.
        # This will be an error.  We let it propagate to the user.
        func = forward_and_deriv_activation_dict[ctx.activation]

        y, func_deriv = func(x)
        if dropout_mask is not None:
            y = y * dropout_mask
        # now compute derivative of y w.r.t. weight and bias..
        # y: (..., in_channels), ans_grad: (..., out_channels),
        (out_channels, in_channels) = weight.shape

        in_channels = y.shape[-1]
        g = ans_grad.reshape(-1, out_channels)
        weight_deriv = torch.matmul(g.t(), y.reshape(-1, in_channels))
        y_deriv = torch.matmul(ans_grad, weight)
        bias_deriv = None if bias is None else g.sum(dim=0)
        x_deriv = y_deriv * func_deriv
        if dropout_mask is not None:
            # order versus func_deriv does not matter
            x_deriv = x_deriv * dropout_mask

        return x_deriv, weight_deriv, bias_deriv, None, None, None


class ActivationDropoutAndLinear(torch.nn.Module):
    """Merges activation, dropout, and nn.Linear in a memory-efficient way.

    This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

    Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: int | None = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        linear = ScaledLinear(in_channels, out_channels, bias=bias, initial_scale=initial_scale)

        self.weight = linear.weight
        # register_parameter properly handles making it a parameter when linear.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter("bias", linear.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if torch.jit.is_scripting() or torch.jit.is_tracing() or "k2" not in sys.modules:
            if self.activation == "SwooshL":
                x = SwooshLForward(x)
            elif self.activation == "SwooshR":
                x = SwooshRForward(x)
            else:
                msg = f"Unknown activation: {self.activation}"
                raise ValueError(msg)
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return ActivationDropoutAndLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.activation,
            float(self.dropout_p),
            self.dropout_shared_dim,
        )


# end zipvoice/models/modules/scaling/_activations.py
