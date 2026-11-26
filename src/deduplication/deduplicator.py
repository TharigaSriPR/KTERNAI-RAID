"""
Semantic Deduplicator module.
Compares newly drafted or predicted RAID items against existing records (drafts & approved)
using TF-IDF cosine similarity, category alignment, and workstream context.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from src.deduplication.schemas import DeduplicationResult, MatchStatus
from src.deduplication.similarity import evaluate_similarity, calculate_text_similarity
from src.classification.schemas import RAIDExtractionResult
from config import DEDUPE_SIMILARITY_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_dict(item: Union[Dict[str, Any], RAIDExtractionResult]) -> Dict[str, Any]:
    """Helper to convert RAIDExtractionResult or BaseModel into dictionary."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return dict(item)


class SemanticDeduplicator:
    """
    Evaluates new RAID candidates against existing registers to identify duplicate
    and related items without modifying or overwriting existing records.
    """

    def __init__(
        self,
        duplicate_threshold: float = 0.75,
        review_threshold: float = 0.50
    ):
        self.duplicate_threshold = duplicate_threshold
        self.review_threshold = review_threshold
        self._recorded_matches: Set[Tuple[str, str]] = set()

    def check_duplicate(
        self,
        candidate: Union[Dict[str, Any], RAIDExtractionResult],
        existing_items: List[Union[Dict[str, Any], RAIDExtractionResult]]
    ) -> DeduplicationResult:
        """
        Finds the highest-scoring match for a single candidate against existing items.
        Returns the top DeduplicationResult, or a NO_MATCH result if no items or no match.
        """
        cand_dict = _to_dict(candidate)
        cand_id = str(cand_dict.get("id") or cand_dict.get("draft_id") or "candidate")

        best_result = DeduplicationResult(
            new_raid_id=cand_id,
            similarity_score=0.0,
            match_status=MatchStatus.NO_MATCH,
            match_reason="No existing records to compare against.",
            same_category=False,
            same_workstream=False,
            requires_human_review=False
        )

        for item in existing_items:
            item_dict = _to_dict(item)
            item_id = str(item_dict.get("id") or item_dict.get("draft_id") or "")
            if cand_id and item_id and cand_id == item_id:
                # Do not compare an item against itself
                continue

            result = evaluate_similarity(
                cand_dict,
                item_dict,
                duplicate_threshold=self.duplicate_threshold,
                review_threshold=self.review_threshold
            )

            if result.similarity_score > best_result.similarity_score:
                best_result = result

        return best_result

    def find_all_matches(
        self,
        candidate: Union[Dict[str, Any], RAIDExtractionResult],
        existing_items: List[Union[Dict[str, Any], RAIDExtractionResult]]
    ) -> List[DeduplicationResult]:
        """
        Returns all matches above review threshold for a given candidate,
        ordered by descending similarity score.
        """
        cand_dict = _to_dict(candidate)
        cand_id = str(cand_dict.get("id") or cand_dict.get("draft_id") or "candidate")
        matches: List[DeduplicationResult] = []

        for item in existing_items:
            item_dict = _to_dict(item)
            item_id = str(item_dict.get("id") or item_dict.get("draft_id") or "")
            if cand_id and item_id and cand_id == item_id:
                continue

            result = evaluate_similarity(
                cand_dict,
                item_dict,
                duplicate_threshold=self.duplicate_threshold,
                review_threshold=self.review_threshold
            )

            if result.match_status != MatchStatus.NO_MATCH:
                pair_key = (cand_id, item_id)
                if pair_key not in self._recorded_matches:
                    self._recorded_matches.add(pair_key)
                    matches.append(result)

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    # Compatibility method matching original DeduplicationMatcher signature
    def find_duplicate(
        self,
        candidate: Dict[str, Any],
        existing_items: List[Dict[str, Any]]
    ) -> Optional[Tuple[str, float]]:
        """
        Legacy adapter method returning (existing_id, similarity_score) if matched.
        """
        res = self.check_duplicate(candidate, existing_items)
        if res.match_status == MatchStatus.LIKELY_DUPLICATE:
            return (res.matched_raid_id, res.similarity_score)
        return None
