"""Persistent feedback plumbing; no fitting, training or implicit label promotion."""
from .core import Store, ValidationError, normalize_source, source_key

__all__ = ["Store", "ValidationError", "normalize_source", "source_key"]
__version__ = "0.1.0"
