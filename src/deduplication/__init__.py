"""
Semantic Deduplication package.
Provides TF-IDF and Cosine Similarity evaluation, contextual match classification,
and duplicate detection for RAID items.
"""

from .schemas import MatchStatus, DeduplicationResult
from .similarity import calculate_text_similarity, evaluate_similarity, canonicalize_text
from .deduplicator import SemanticDeduplicator

__all__ = [
    "MatchStatus",
    "DeduplicationResult",
    "calculate_text_similarity",
    "evaluate_similarity",
    "canonicalize_text",
    "SemanticDeduplicator",
]
