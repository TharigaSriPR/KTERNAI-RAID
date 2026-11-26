"""
Semantic similarity and comparison module using TF-IDF and Cosine Similarity.
Provides deterministic, offline text vectorisation, domain-aware canonicalisation,
and multi-factor contextual evaluation (category, workstream, phase).
"""

from typing import Dict, Any, Optional, Tuple
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.deduplication.schemas import DeduplicationResult, MatchStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Preserve domain-specific IT words that sklearn treats as general stopwords
DOMAIN_PRESERVED_WORDS = {
    "system", "bill", "part", "side", "call", "describe",
    "find", "interest", "fire", "fill", "show", "detail"
}
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - DOMAIN_PRESERVED_WORDS

# Domain phrase canonicalisation mappings for SAP transformation contexts
DOMAIN_CANONICALISATIONS = [
    (r"\bsit\b", "integration testing"),
    (r"\buat\b", "acceptance testing"),
    (r"\bcustomer master\b", "client master data"),
    (r"\bclient master\b", "client master data"),
    (r"\bmaster data records\b", "master data"),
    (r"\bpo approval\b", "purchase order approval"),
    (r"\bpo\b", "purchase order"),
    (r"\bpr\b", "purchase requisition"),
    (r"\bgr\b", "goods receipt"),
]


def stem_word(w: str) -> str:
    """Lightweight rule-based suffix normalization."""
    w = w.lower()
    if w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def canonicalize_and_preprocess(text: str) -> str:
    """Normalises text, standardises SAP domain acronyms, and applies light stemming."""
    if not text:
        return ""
    normalized = text.lower()
    for pattern, replacement in DOMAIN_CANONICALISATIONS:
        normalized = re.sub(pattern, replacement, normalized)

    words = re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", normalized)
    processed = [stem_word(w) for w in words if w not in CUSTOM_STOP_WORDS]
    return " ".join(processed)


def canonicalize_text(text: str) -> str:
    """Backward compatibility helper for canonicalize_text."""
    return canonicalize_and_preprocess(text)


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two text strings using TF-IDF vectorisation.
    
    Safe guarantees:
    - Returns 0.0 if either string is empty or contains only whitespace.
    - Handles arbitrary length and punctuation without raising exceptions.
    - Does not call external APIs or paid embeddings.
    """
    if not text1 or not text2:
        return 0.0

    t1_clean = canonicalize_and_preprocess(text1.strip())
    t2_clean = canonicalize_and_preprocess(text2.strip())

    if not t1_clean or not t2_clean:
        return 0.0

    if t1_clean == t2_clean:
        return 1.0

    try:
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\./]+\b"
        )
        tfidf = vectorizer.fit_transform([t1_clean, t2_clean])
        if tfidf.shape[1] == 0:
            return 0.0
        score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return max(0.0, min(1.0, round(score, 4)))
    except Exception as exc:
        logger.warning("Error computing TF-IDF cosine similarity: %s", exc)
        return 0.0


def evaluate_similarity(
    candidate: Dict[str, Any],
    existing_item: Dict[str, Any],
    duplicate_threshold: float = 0.75,
    review_threshold: float = 0.50
) -> DeduplicationResult:
    """
    Compares a candidate RAID item against an existing RAID record using text similarity
    and contextual attributes (RAID category, workstream, phase).
    
    Decision Logic:
    1. Score >= duplicate_threshold:
       - Same Category + Same Workstream: LIKELY_DUPLICATE
       - Different Category: RELATED_NOT_DUPLICATE (captures risk-to-issue progression)
       - Different Workstream: RELATED_NOT_DUPLICATE (captures separate functional scope)
    2. review_threshold <= Score < duplicate_threshold:
       - Same Category: POSSIBLE_DUPLICATE
       - Different Category: RELATED_NOT_DUPLICATE
    3. Score < review_threshold:
       - NO_MATCH
    """
    cand_title = candidate.get("title") or ""
    cand_desc = candidate.get("description") or ""
    cand_text = f"{cand_title} {cand_desc}".strip()

    exist_title = existing_item.get("title") or ""
    exist_desc = existing_item.get("description") or ""
    exist_text = f"{exist_title} {exist_desc}".strip()

    cand_id = candidate.get("id") or candidate.get("draft_id")
    cand_signal_id = candidate.get("evidence_source_id") or candidate.get("source_signal_id")
    exist_id = existing_item.get("id") or existing_item.get("draft_id")

    cand_cat = candidate.get("type") or candidate.get("category") or "Unknown"
    exist_cat = existing_item.get("type") or existing_item.get("category") or "Unknown"

    cand_ws = candidate.get("workstream") or "General"
    exist_ws = existing_item.get("workstream") or "General"

    same_cat = str(cand_cat).strip().lower() == str(exist_cat).strip().lower()

    # Workstream context evaluation:
    # Differing workstreams occur when both items declare explicit, non-general modules that disagree (e.g. FI vs MM)
    cand_ws_clean = str(cand_ws).strip().lower()
    exist_ws_clean = str(exist_ws).strip().lower()
    has_explicit_ws = (
        cand_ws_clean not in ("", "general", "unknown", "pmo", "all") and
        exist_ws_clean not in ("", "general", "unknown", "pmo", "all")
    )
    is_different_workstream = has_explicit_ws and (cand_ws_clean != exist_ws_clean)
    same_ws = (cand_ws_clean == exist_ws_clean) or (not is_different_workstream)

    score_full = calculate_text_similarity(cand_text, exist_text)
    score_title = calculate_text_similarity(cand_title, exist_title)
    score = max(score_full, score_title)

    # Determine status and justification
    if score >= duplicate_threshold:
        if is_different_workstream:
            status = MatchStatus.RELATED_NOT_DUPLICATE
            reason = (
                f"High text similarity ({score:.2f}), but differing workstreams ('{cand_ws}' vs '{exist_ws}') "
                f"indicate separate functional scopes."
            )
            requires_review = True
        elif not same_cat:
            status = MatchStatus.RELATED_NOT_DUPLICATE
            reason = (
                f"High text similarity ({score:.2f}), but differing categories ('{cand_cat}' vs '{exist_cat}') "
                f"indicate risk/issue evolution rather than duplicate records."
            )
            requires_review = True
        else:
            status = MatchStatus.LIKELY_DUPLICATE
            reason = (
                f"High text similarity ({score:.2f}) within identical category '{cand_cat}' "
                f"and workstream '{cand_ws}'. Flagged as likely duplicate."
            )
            requires_review = True
    elif score >= review_threshold:
        if is_different_workstream:
            status = MatchStatus.RELATED_NOT_DUPLICATE
            reason = (
                f"Moderate text similarity ({score:.2f}) across different workstreams ('{cand_ws}' vs '{exist_ws}')."
            )
            requires_review = True
        elif not same_cat:
            status = MatchStatus.RELATED_NOT_DUPLICATE
            reason = (
                f"Moderate text similarity ({score:.2f}) across differing categories ('{cand_cat}' vs '{exist_cat}')."
            )
            requires_review = True
        else:
            status = MatchStatus.POSSIBLE_DUPLICATE
            reason = (
                f"Moderate text similarity ({score:.2f}) within same category '{cand_cat}'. "
                f"Recommended for PM review."
            )
            requires_review = True
    else:
        status = MatchStatus.NO_MATCH
        reason = f"Similarity score ({score:.2f}) is below review threshold ({review_threshold:.2f}). Distinct item."
        requires_review = False

    return DeduplicationResult(
        new_raid_id=str(cand_id) if cand_id else None,
        new_signal_id=str(cand_signal_id) if cand_signal_id else None,
        matched_raid_id=str(exist_id) if (exist_id and status != MatchStatus.NO_MATCH) else None,
        similarity_score=score,
        match_status=status,
        match_reason=reason,
        same_category=same_cat,
        same_workstream=same_ws,
        requires_human_review=requires_review,
        metadata={
            "candidate_category": str(cand_cat),
            "matched_category": str(exist_cat),
            "candidate_workstream": str(cand_ws),
            "matched_workstream": str(exist_ws),
            "duplicate_threshold": duplicate_threshold,
            "review_threshold": review_threshold
        }
    )
