"""ZipVoice text-to-speech package."""

# start zipvoice/__init__.py
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*",
)

from zipvoice.config import LuxTTSSettings, load_settings  # noqa: E402

__all__ = ["LuxTTSSettings", "load_settings"]
# end zipvoice/__init__.py
