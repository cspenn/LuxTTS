# start zipvoice/models/modules/zipformer/__init__.py
"""Zipformer modules for ZipVoice — backward-compatible subpackage.

This package replaces the monolithic zipformer.py module.  All names that were
previously importable from ``zipvoice.models.modules.zipformer`` continue to
work unchanged.
"""

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
from zipvoice.models.modules.zipformer._encoder import (
    BypassModule,
    DownsampledZipformer2Encoder,
    FeedforwardModule,
    NonlinAttention,
    TTSZipformer,
    Zipformer2Encoder,
    Zipformer2EncoderLayer,
)

__all__ = [
    # _attention
    "CompactRelPositionalEncoding",
    "RelPositionMultiheadAttentionWeights",
    "SelfAttention",
    "_whitening_schedule",
    # _conv
    "ConvolutionModule",
    "SimpleDownsample",
    "SimpleUpsample",
    "timestep_embedding",
    # _encoder
    "BypassModule",
    "DownsampledZipformer2Encoder",
    "FeedforwardModule",
    "NonlinAttention",
    "TTSZipformer",
    "Zipformer2Encoder",
    "Zipformer2EncoderLayer",
]
# end zipvoice/models/modules/zipformer/__init__.py
