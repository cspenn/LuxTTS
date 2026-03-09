# start zipvoice/models/modules/scaling/__init__.py
"""Scaling utilities for ZipVoice — backward-compatible subpackage.

This package replaces the monolithic scaling.py module.  All names that were
previously importable from ``zipvoice.models.modules.scaling`` continue to
work unchanged.
"""

from zipvoice.models.modules.scaling._activations import (
    DEVICE_TYPE,
    ActivationDropoutAndLinear,
    ActivationDropoutAndLinearFunction,
    BiasNorm,
    BiasNormFunction,
    Dropout2,
    Dropout3,
    Identity,
    LimitParamValue,
    MulForDropout3,
    ScaledLinear,
    SoftmaxFunction,
    SwooshL,
    SwooshLForward,
    SwooshLFunction,
    SwooshLOnnx,
    SwooshR,
    SwooshRForward,
    SwooshRFunction,
    SwooshROnnx,
    WithLoss,
    _no_op,
    custom_amp_decorator,
    custom_bwd,
    custom_fwd,
    get_memory_allocated,
    limit_param_value,
    logaddexp,
    logaddexp_onnx,
    softmax,
    with_loss,
)
from zipvoice.models.modules.scaling._normalization import (
    Balancer,
    BalancerFunction,
    Whiten,
    WhiteningPenaltyFunction,
    _diag,
    _whitening_metric,
    penalize_abs_values_gt,
)
from zipvoice.models.modules.scaling._piecewise import (
    CutoffEstimator,
    FloatLike,
    PiecewiseLinear,
    ScheduledFloat,
)

__all__ = [
    # _piecewise
    "CutoffEstimator",
    "FloatLike",
    "PiecewiseLinear",
    "ScheduledFloat",
    # _activations
    "DEVICE_TYPE",
    "ActivationDropoutAndLinear",
    "ActivationDropoutAndLinearFunction",
    "BiasNorm",
    "BiasNormFunction",
    "Dropout2",
    "Dropout3",
    "Identity",
    "LimitParamValue",
    "MulForDropout3",
    "ScaledLinear",
    "SoftmaxFunction",
    "SwooshL",
    "SwooshLForward",
    "SwooshLFunction",
    "SwooshLOnnx",
    "SwooshR",
    "SwooshRForward",
    "SwooshRFunction",
    "SwooshROnnx",
    "WithLoss",
    "_no_op",
    "custom_amp_decorator",
    "custom_bwd",
    "custom_fwd",
    "get_memory_allocated",
    "limit_param_value",
    "logaddexp",
    "logaddexp_onnx",
    "softmax",
    "with_loss",
    # _normalization
    "Balancer",
    "BalancerFunction",
    "Whiten",
    "WhiteningPenaltyFunction",
    "_diag",
    "_whitening_metric",
    "penalize_abs_values_gt",
]
# end zipvoice/models/modules/scaling/__init__.py
