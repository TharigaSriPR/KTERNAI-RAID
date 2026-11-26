"""
Backwards-compatible extraction models wrapper forwarding to schemas.py.
"""
from .schemas import RAIDExtractionResult, RAIDCategory

# Backwards compatibility alias
RAIDItemExtraction = RAIDExtractionResult

__all__ = ["RAIDItemExtraction", "RAIDExtractionResult", "RAIDCategory"]
