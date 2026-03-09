# start zipvoice/exceptions.py
"""Custom exceptions for LuxTTS.

All project-specific exceptions inherit from LuxTTSError,
enabling callers to catch the full project exception hierarchy
with a single except clause.
"""


class LuxTTSError(Exception):
    """Base exception for all LuxTTS errors."""


class ModelLoadError(LuxTTSError):
    """Raised when model weights or config cannot be loaded.

    This covers network failures during download, missing checkpoint
    files, and invalid config.json structures.
    """


class TokenizerError(LuxTTSError):
    """Raised when tokenization fails.

    Examples include missing token files, unknown language codes,
    and unsupported text encodings.
    """


class AudioProcessingError(LuxTTSError):
    """Raised when audio encoding or decoding fails.

    Examples include invalid sample rates, corrupt audio data,
    and unsupported audio formats.
    """


# end zipvoice/exceptions.py
