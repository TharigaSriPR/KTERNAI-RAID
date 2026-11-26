"""
Deduplication and Duplicate Detection module.
Compares newly drafted RAID items against existing drafts and approved items
using token similarity / embeddings to prevent duplicate log entries.
"""

from typing import List, Dict, Any, Optional, Tuple
from config import DEDUPE_SIMILARITY_THRESHOLD


def calculate_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculates word-token Jaccard similarity between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection) / float(union)


class DeduplicationMatcher:
    def __init__(self, threshold: float = DEDUPE_SIMILARITY_THRESHOLD):
        self.threshold = threshold

    def find_duplicate(
        self, 
        candidate: Dict[str, Any], 
        existing_items: List[Dict[str, Any]]
    ) -> Optional[Tuple[str, float]]:
        """
        Checks candidate against existing items of the same RAID type.
        Returns (existing_id, similarity_score) if matched above threshold, else None.
        """
        candidate_text = f"{candidate.get('title', '')} {candidate.get('description', '')}"
        candidate_type = candidate.get("type")
        candidate_workstream = candidate.get("workstream")

        for item in existing_items:
            # Same RAID category
            if item.get("type") != candidate_type:
                continue

            # Prioritize matching in the same workstream
            if candidate_workstream and item.get("workstream") and candidate_workstream != item.get("workstream"):
                continue

            target_text = f"{item.get('title', '')} {item.get('description', '')}"
            score = calculate_jaccard_similarity(candidate_text, target_text)

            if score >= self.threshold:
                return (item.get("id"), score)

        return None
