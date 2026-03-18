from . import _version
from . import wavelengthrange


try:
    __version__ = _version.version
except Exception:
    __version__ = "dev"
