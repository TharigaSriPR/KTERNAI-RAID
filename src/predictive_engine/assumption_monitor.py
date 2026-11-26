"""
Assumption Contradiction Monitor.
Monitors previously recorded RAID Assumptions and detects when subsequent communications
or project signals contradict them.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import re
from datetime import datetime

from src.normalisation.schemas import ProjectSignal
from src.predictive_engine.schemas import (
    PredictionItem,
    PredictionType,
    PredictionSeverity,
    PredictionStatus,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "of", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "is", "am", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "shall",
    "could", "might", "must", "that", "this", "these", "those", "i", "we",
    "you", "he", "she", "it", "they", "our", "their", "my", "your", "his", "her"
}

CONTRADICTION_INDICATORS = [
    "incomplete", "delayed", "delay", "postponed", "missed", "not available",
    "unavailable", "cancelled", "canceled", "will not be", "cannot proceed",
    "blocked", "not ready", "behind schedule", "failed", "failure", "rejected",
    "cannot meet", "not completed", "slip", "slippage", "insufficient",
    "pending approval", "refused", "no access", "cannot deliver"
]

ACTIVE_BLOCKER_INDICATORS = [
    "blocked", "cannot proceed", "failed", "failure", "cancelled",
    "canceled", "missed", "stopped", "critical blocker"
]


class AssumptionMonitor:
    """
    Monitors existing assumptions and evaluates new signals to identify contradictions.
    
    Guarantees:
    - Never flags an assumption as contradicted merely because a new signal is negative.
    - Never flags an assumption as contradicted when the new signal does not mention the topic.
    - Preserves both original assumption text and contradictory evidence snippet.
    - Captures source IDs for both the assumption and the contradicting signal.
    - Prevents generating duplicate predictions for the same pair of signals.
    """

    def __init__(self):
        self._evaluated_pairs: Set[Tuple[str, str]] = set()

    def _extract_topic_keywords(self, text: str) -> Set[str]:
        """Extracts significant domain keywords from assumption text, excluding stopwords."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text.lower())
        meaningful = {w for w in words if w not in STOPWORDS and len(w) > 2}
        return meaningful

    def _extract_contradicting_snippet(self, content: str, topic_words: Set[str]) -> str:
        """Finds the specific sentence or fragment containing topic overlap and contradiction."""
        sentences = re.split(r"[.\n;!?]+", content)
        best_sentence = ""
        best_score = 0
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            topic_hits = sum(1 for tw in topic_words if tw in s_lower)
            has_indicator = any(ind in s_lower for ind in CONTRADICTION_INDICATORS)
            if has_indicator and topic_hits > best_score:
                best_score = topic_hits
                best_sentence = s_clean

        return best_sentence if best_sentence else content[:200].strip()

    def check_contradictions(
        self,
        assumptions: List[Dict[str, Any]],
        signals: List[ProjectSignal]
    ) -> List[PredictionItem]:
        """
        Evaluates a list of assumptions against subsequent project signals.
        Returns newly detected ASSUMPTION_CONTRADICTED predictions.
        """
        predictions: List[PredictionItem] = []

        for asm in assumptions:
            asm_id = asm.get("id") or asm.get("draft_id") or "asm-unknown"
            asm_title = asm.get("title", "Untitled Assumption")
            asm_desc = asm.get("description", "")
            asm_evidence = asm.get("evidence") or asm.get("evidence_text") or asm_desc
            asm_source_id = asm.get("evidence_source_id") or asm.get("source_signal_id") or ""
            asm_workstream = asm.get("workstream", "General")

            # Extract topic keywords from assumption
            full_asm_text = f"{asm_title} {asm_desc} {asm_evidence}"
            topic_keywords = self._extract_topic_keywords(full_asm_text)
            if not topic_keywords:
                continue

            asm_time_raw = asm.get("timestamp") or asm.get("detected_at") or asm.get("created_at")

            for sig in signals:
                # Do not compare assumption against its own source signal
                if asm_source_id and sig.id == asm_source_id:
                    continue

                # Chronological check: contradictory signal must be later than the assumption
                if asm_time_raw and sig.timestamp:
                    try:
                        from dateutil import parser as dt_parser
                        from datetime import timezone
                        t_asm = dt_parser.parse(str(asm_time_raw))
                        t_sig = dt_parser.parse(str(sig.timestamp))
                        # Normalize to naive UTC datetimes for safe comparison
                        if t_asm.tzinfo is not None:
                            t_asm = t_asm.astimezone(timezone.utc).replace(tzinfo=None)
                        if t_sig.tzinfo is not None:
                            t_sig = t_sig.astimezone(timezone.utc).replace(tzinfo=None)
                        if t_sig < t_asm:
                            logger.debug("Signal %s (%s) is older than assumption %s (%s); skipping.", sig.id, t_sig, asm_id, t_asm)
                            continue
                    except Exception as exc:
                        logger.debug("Error comparing assumption timestamps: %s", exc)

                pair_key = (str(asm_id), str(sig.id))
                if pair_key in self._evaluated_pairs:
                    continue

                sig_content_lower = sig.content.lower()

                # Requirement: Must share substantive domain topic keywords, not just generic terms
                matched_keywords = {tw for tw in topic_keywords if tw in sig_content_lower}
                min_match = 2 if len(topic_keywords) >= 3 else 1
                if len(matched_keywords) < min_match:
                    continue

                # Verify that matched keywords are not solely generic fillers
                generic_project_words = {"data", "test", "testing", "project", "team", "system", "work", "status", "general"}
                specific_matches = matched_keywords - generic_project_words
                if not specific_matches and len(topic_keywords) > len(generic_project_words):
                    continue

                # Check for contradiction / negation indicators
                matched_indicators = [ind for ind in CONTRADICTION_INDICATORS if ind in sig_content_lower]
                if not matched_indicators:
                    continue

                # Mark pair as evaluated to avoid duplicate generation
                self._evaluated_pairs.add(pair_key)

                contradicting_snippet = self._extract_contradicting_snippet(sig.content, topic_keywords)

                # Determine suggested category: Issue if active blocker, else Risk
                is_active_blocker = any(b in sig_content_lower for b in ACTIVE_BLOCKER_INDICATORS)
                suggested_cat = "Issue" if is_active_blocker else "Risk"
                severity = PredictionSeverity.HIGH.value if is_active_blocker else PredictionSeverity.MEDIUM.value

                combined_evidence = (
                    f"Original Assumption: \"{asm_evidence}\" (Ref: {asm_source_id or asm_id}) | "
                    f"Contradicting Evidence: \"{contradicting_snippet}\" (Source: {sig.source_type} by {sig.author})"
                )

                pred = PredictionItem(
                    prediction_type=PredictionType.ASSUMPTION_CONTRADICTED,
                    title=f"Predictive Warning: Contradicted Assumption - {asm_title}",
                    description=(
                        f"Previously recorded assumption '{asm_title}' appears contradicted by recent "
                        f"{sig.source_type} communication from {sig.author}: '{contradicting_snippet}'."
                    ),
                    suggested_raid_category=suggested_cat,
                    confidence=0.88,
                    severity=severity,
                    evidence=combined_evidence,
                    source_signal_ids=[s for s in [asm_source_id, sig.id] if s],
                    related_raid_ids=[asm_id],
                    workstream=sig.workstream or asm_workstream or "General",
                    phase=sig.phase or "Realize",
                    metadata={
                        "assumption_id": asm_id,
                        "assumption_title": asm_title,
                        "contradicting_signal_id": sig.id,
                        "matched_topic_words": list(matched_keywords),
                        "matched_indicators": matched_indicators
                    }
                )
                predictions.append(pred)
                logger.info(
                    "Detected contradicted assumption '%s' via signal %s (source_type=%s)",
                    asm_title, sig.id, sig.source_type
                )

        return predictions
