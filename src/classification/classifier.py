"""
RAID Classifier Module.
Coordinates prompt construction, LLM generation, response parsing,
Pydantic validation, and evidence verification.
"""

import json
import re
from typing import Optional, Dict, Any, Union, List

from config import (
    CONFIDENCE_THRESHOLD,
    LLM_MAX_INPUT_LENGTH
)
from src.normalisation.schemas import ProjectSignal
from .schemas import (
    RAIDCategory,
    RAIDExtractionResult,
    ClassificationStatus,
    ClassificationErrorCategory,
    EvidenceStatus,
    ClassificationOutcome
)
from .prompt_templates import build_classification_prompt, SYSTEM_INSTRUCTION
from .providers.base_provider import BaseLLMProvider
from .llm_client import get_llm_provider
from src.utils.logger import get_logger

logger = get_logger(__name__)

STOP_WORDS = {
    "this", "that", "with", "from", "have", "were", "been", "will", "would",
    "could", "should", "what", "when", "where", "which", "there", "their",
    "about", "into", "more", "some", "such", "than", "them", "then", "these",
    "they", "your", "also", "only", "must", "very", "just", "done", "does",
    "please", "team", "project"
}

DOMAIN_TERMS = {
    "delay", "blocked", "slip", "slippage", "overdue", "transport", "failure",
    "failed", "dump", "defect", "bug", "patch", "fix", "agreed", "approve",
    "approved", "decision", "dependency", "prerequisite", "vendor", "budget",
    "scope", "deadline", "cutover", "migration", "testing", "uat", "sit",
    "basis", "refresh", "outage", "downtime", "bottleneck", "remediation",
    "unmapped", "exception", "workaround", "deliverable"
}

CATEGORY_INDICATORS = {
    "risk": {"may", "might", "risk", "could", "potential", "slip", "delay", "uncertainty"},
    "issue": {"blocked", "fail", "failed", "error", "dump", "critical", "broken", "issue", "problem", "stalled"},
    "assumption": {"assume", "assuming", "assumption", "presume", "belief", "baseline"},
    "decision": {"agree", "agreed", "decide", "decided", "approved", "chosen", "concluded", "agreement"},
    "dependency": {"depend", "depends", "dependency", "prerequisite", "after", "only after", "waiting", "refresh"}
}

GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|dear|good\s+(?:morning|afternoon|evening))\b",
    re.IGNORECASE
)

SIGNOFF_PATTERN = re.compile(
    r"^(thanks|thank\s+you|regards|best\s+regards|cheers|sincerely|warm\s+regards)\b",
    re.IGNORECASE
)

DISCLAIMER_PATTERN = re.compile(
    r"confidentiality\s+notice|this\s+(?:email|message)\s+(?:is|was)\s+(?:confidential|scanned)|disclaimer:|all\s+rights\s+reserved|scanned\s+by",
    re.IGNORECASE
)


def sanitize_error_message(msg: str) -> str:
    """Masks potential credentials or keys from error strings and truncates to safe length."""
    if not msg:
        return ""
    cleaned = str(msg)
    cleaned = re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", cleaned)
    cleaned = re.sub(r"sk-[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", cleaned)
    cleaned = re.sub(
        r"(?:api[_-]?key|token|bearer|secret|password|auth)\s*[=:]\s*['\"]?[0-9A-Za-z-_.]{6,}['\"]?",
        "[REDACTED_CREDENTIAL]",
        cleaned,
        flags=re.IGNORECASE
    )
    return cleaned[:500].strip()


def find_best_fallback_sentence(
    content: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None
) -> Optional[str]:
    """
    Identifies the most relevant authentic complete sentence from source content
    when Gemini's extracted evidence quote is hallucinated or missing.
    Filters out greetings, sign-offs, and disclaimers.
    Returns None if no sentence meets the relevance threshold.
    """
    if not content or not content.strip():
        return None

    # Sentence boundary splitting (punctuation followed by whitespace or linebreaks)
    raw_sentences = re.split(r"(?<=[.!?\n])\s+", content)

    # Extract substantive keywords from title and description
    keywords = set()
    for text in (title, description):
        if text:
            for w in re.findall(r"[A-Za-z0-9_-]{4,}", text.lower()):
                if w not in STOP_WORDS:
                    keywords.add(w)

    cat_key = (category or "").strip().lower()
    cat_terms = CATEGORY_INDICATORS.get(cat_key, set())

    best_sentence: Optional[str] = None
    best_score: float = 0.0

    for sent in raw_sentences:
        clean_sent = sent.strip()
        # Discard fragments that are too short to be substantive statements
        if len(clean_sent) < 15:
            continue

        # Discard greetings, sign-offs, and disclaimers
        if GREETING_PATTERN.match(clean_sent):
            continue
        if SIGNOFF_PATTERN.match(clean_sent):
            continue
        if DISCLAIMER_PATTERN.search(clean_sent):
            continue

        sent_words = set(re.findall(r"[A-Za-z0-9_-]{3,}", clean_sent.lower()))

        score = 0.0
        # Title/Description keyword overlap
        for kw in keywords:
            if kw in sent_words:
                score += 2.0

        # Domain project terms
        for dt in DOMAIN_TERMS:
            if dt in sent_words:
                score += 1.5

        # Category indicator terms
        for ct in cat_terms:
            if ct in sent_words:
                score += 1.5

        # Administrative sentence penalty
        if re.search(r"meeting\s+minutes\s+attached|please\s+find\s+(?:the\s+)?attached", clean_sent, re.IGNORECASE):
            score -= 2.0

        if score > best_score:
            best_score = score
            best_sentence = clean_sent

    # Require minimum relevance threshold of 2.0 (at least one strong keyword or domain match)
    if best_score >= 2.0 and best_sentence:
        return best_sentence

    return None


class RAIDClassifier:
    """
    Core engine that classifies ProjectSignal instances into RAID categories
    using an LLM provider and validates extracted entities using Pydantic v2.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        confidence_threshold: Optional[float] = None,
        max_input_length: int = LLM_MAX_INPUT_LENGTH,
        db_manager: Optional[Any] = None
    ):
        self.provider = provider or get_llm_provider()
        self.confidence_threshold = (
            CONFIDENCE_THRESHOLD if confidence_threshold is None else float(confidence_threshold)
        )
        self.max_input_length = max_input_length
        self.db = db_manager
        self.last_outcome: Optional[ClassificationOutcome] = None

    @staticmethod
    def _strip_markdown_code_fences(raw_text: str) -> str:
        """Strips ```json ... ``` markdown wrappers if present."""
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def classify_signal_detailed(self, signal: ProjectSignal) -> ClassificationOutcome:
        """
        Executes classification and returns a detailed ClassificationOutcome
        capturing status, error category, failure reason, and the extracted result.
        """
        prov_name = getattr(self.provider, "provider_name", "unknown")
        mod_name = getattr(self.provider, "model_name", "unknown")

        # If pre-filtered as noise, skip LLM call entirely
        if signal.is_noise:
            logger.debug("Signal %s is pre-filtered noise; skipping LLM classification.", signal.id)
            return ClassificationOutcome(
                status=ClassificationStatus.NOT_RAID,
                extraction=None,
                error_category=None,
                error_message="Pre-filtered as noise",
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider="noise_filter",
                llm_model="rule_based"
            )

        content_to_analyze = signal.content or ""
        if len(content_to_analyze) > self.max_input_length:
            logger.warning(
                "Signal %s content exceeds max input length (%d > %d); truncating safely.",
                signal.id, len(content_to_analyze), self.max_input_length
            )
            content_to_analyze = content_to_analyze[:self.max_input_length] + " [TRUNCATED]"

        prompt = build_classification_prompt(
            signal_id=signal.id,
            source_type=signal.source_type,
            author=signal.author,
            timestamp=signal.timestamp,
            subject=signal.subject,
            content=content_to_analyze,
            workstream=signal.workstream,
            phase=signal.phase
        )

        try:
            logger.debug("Requesting classification for signal %s via %s", signal.id, self.provider.provider_name)
            raw_response = self.provider.generate_text(prompt, system_instruction=SYSTEM_INSTRUCTION)
        except Exception as exc:
            err_str = str(exc)
            safe_msg = sanitize_error_message(err_str)
            err_lower = err_str.lower()

            if "timeout" in err_lower or "timed out" in err_lower:
                cat = ClassificationErrorCategory.API_TIMEOUT
            elif "429" in err_lower or "rate" in err_lower or "resource_exhausted" in err_lower:
                cat = ClassificationErrorCategory.RATE_LIMIT
            elif any(k in err_lower for k in ("401", "403", "api_key", "api key", "permission_denied", "authentication", "not configured", "invalid api key")):
                cat = ClassificationErrorCategory.AUTHENTICATION_FAILURE
            else:
                cat = ClassificationErrorCategory.API_ERROR

            logger.error("LLM Provider failed to classify signal %s: %s", signal.id, safe_msg)
            return ClassificationOutcome(
                status=ClassificationStatus.FAILED,
                extraction=None,
                error_category=cat,
                error_message=safe_msg,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        if not raw_response or not raw_response.strip():
            safe_msg = "Empty response received from LLM provider."
            logger.error("Empty response for signal %s", signal.id)
            return ClassificationOutcome(
                status=ClassificationStatus.FAILED,
                extraction=None,
                error_category=ClassificationErrorCategory.EMPTY_RESPONSE,
                error_message=safe_msg,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        # Clean JSON text
        cleaned_json = self._strip_markdown_code_fences(raw_response)

        try:
            parsed_data = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            safe_msg = sanitize_error_message(f"Malformed JSON: {exc}")
            logger.error("Failed to parse LLM JSON response for signal %s: %s", signal.id, safe_msg)
            return ClassificationOutcome(
                status=ClassificationStatus.FAILED,
                extraction=None,
                error_category=ClassificationErrorCategory.INVALID_JSON,
                error_message=safe_msg,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        if not isinstance(parsed_data, dict):
            safe_msg = f"LLM returned non-dictionary JSON response of type {type(parsed_data).__name__}"
            logger.error("LLM returned non-dictionary JSON response for signal %s: %s", signal.id, safe_msg)
            return ClassificationOutcome(
                status=ClassificationStatus.FAILED,
                extraction=None,
                error_category=ClassificationErrorCategory.INVALID_JSON,
                error_message=safe_msg,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        # Ensure source_signal_id is attached
        if not parsed_data.get("source_signal_id"):
            parsed_data["source_signal_id"] = signal.id

        # Preserve workstream and phase from source signal if omitted by model
        if not parsed_data.get("workstream"):
            parsed_data["workstream"] = signal.workstream
        if not parsed_data.get("phase"):
            parsed_data["phase"] = signal.phase

        # Validate with Pydantic v2
        try:
            result = RAIDExtractionResult.model_validate(parsed_data)
        except Exception as exc:
            safe_msg = sanitize_error_message(f"Pydantic validation error: {exc}")
            logger.error("Pydantic validation error for signal %s: %s", signal.id, safe_msg)
            return ClassificationOutcome(
                status=ClassificationStatus.FAILED,
                extraction=None,
                error_category=ClassificationErrorCategory.SCHEMA_VALIDATION_ERROR,
                error_message=safe_msg,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        object.__setattr__(result, "llm_provider", prov_name)
        object.__setattr__(result, "llm_model", mod_name)

        # Evidence validation: verify quote against un-truncated original source text
        if result.is_raid:
            original_full_text = f"{signal.subject} {signal.content}".strip()
            clean_original = " ".join(original_full_text.split())

            evidence_verified = False
            if result.evidence and result.evidence.strip():
                clean_evidence = " ".join(result.evidence.split())
                if clean_evidence.lower() in clean_original.lower():
                    evidence_verified = True
                    object.__setattr__(result, "evidence_status", EvidenceStatus.VERIFIED)

            if not evidence_verified:
                # Attempt to find authentic supporting sentence from content
                cat_name = result.category.value if hasattr(result.category, "value") else str(result.category)
                fallback_sentence = find_best_fallback_sentence(
                    content=signal.content,
                    title=result.title,
                    description=result.description,
                    category=cat_name
                )
                if fallback_sentence:
                    logger.info(
                        "Signal %s: Quote was invalid or missing. Selected authentic fallback sentence from source.",
                        signal.id
                    )
                    object.__setattr__(result, "evidence", fallback_sentence)
                    object.__setattr__(result, "evidence_status", EvidenceStatus.FALLBACK)
                else:
                    logger.info(
                        "Signal %s: No reliable supporting sentence identified. Marked evidence as unavailable.",
                        signal.id
                    )
                    object.__setattr__(result, "evidence", "")
                    object.__setattr__(result, "evidence_status", EvidenceStatus.UNAVAILABLE)
        else:
            object.__setattr__(result, "evidence_status", EvidenceStatus.UNAVAILABLE)

        category_str = result.category.value if hasattr(result.category, "value") else str(result.category)
        logger.info(
            "Signal %s classified as '%s' with confidence %.2f (title: %s, evidence_status: %s)",
            signal.id, category_str, result.confidence, result.title or "N/A", result.evidence_status
        )

        if not result.is_raid:
            return ClassificationOutcome(
                status=ClassificationStatus.NOT_RAID,
                extraction=result,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        if result.confidence < self.confidence_threshold:
            return ClassificationOutcome(
                status=ClassificationStatus.BELOW_CONFIDENCE_THRESHOLD,
                extraction=result,
                signal_id=signal.id,
                source_type=str(signal.source_type),
                llm_provider=prov_name,
                llm_model=mod_name
            )

        return ClassificationOutcome(
            status=ClassificationStatus.COMPLETED,
            extraction=result,
            signal_id=signal.id,
            source_type=str(signal.source_type),
            llm_provider=prov_name,
            llm_model=mod_name
        )

    def classify_signal(self, signal: ProjectSignal) -> Optional[RAIDExtractionResult]:
        """
        Classifies a ProjectSignal into a validated RAIDExtractionResult.
        Returns None if classification failed or if signal is noise.
        Maintains complete backward compatibility with existing callers.
        """
        outcome = self.classify_signal_detailed(signal)
        self.last_outcome = outcome

        # Optional direct DB status recording if db_manager is attached
        if self.db and hasattr(self.db, "update_signal_classification_status"):
            try:
                failure_reason = outcome.error_message
                if outcome.status == ClassificationStatus.BELOW_CONFIDENCE_THRESHOLD and outcome.extraction:
                    try:
                        failure_reason = outcome.extraction.model_dump_json()
                    except Exception:
                        failure_reason = outcome.error_message
                self.db.update_signal_classification_status(
                    signal_id=signal.id,
                    status=outcome.status.value,
                    failure_reason=failure_reason,
                    error_category=outcome.error_category.value if outcome.error_category else None,
                    user_id=getattr(signal, "user_id", "default_user"),
                    increment_attempts=True,
                    llm_provider=outcome.llm_provider,
                    llm_model=outcome.llm_model
                )
            except Exception as exc:
                logger.warning("Could not persist classification status for %s: %s", signal.id, exc)

        return outcome.extraction

    # Alias for method name consistency
    classify_and_extract = classify_signal

    def analyze_signal(self, signal: ProjectSignal) -> Optional[Dict[str, Any]]:
        """
        Backwards-compatible adapter for pipeline integration.
        Translates RAIDExtractionResult to the pipeline dictionary structure.
        """
        outcome = self.classify_signal_detailed(signal)
        self.last_outcome = outcome

        if outcome.status != ClassificationStatus.COMPLETED or not outcome.extraction:
            return None

        extraction = outcome.extraction
        cat_str = extraction.category.value if hasattr(extraction.category, "value") else str(extraction.category)

        evidence_text = extraction.evidence or "[Evidence unavailable in source content]"
        evidence_status_val = (
            extraction.evidence_status.value
            if hasattr(extraction.evidence_status, "value")
            else str(extraction.evidence_status)
        )

        return {
            "type": cat_str,
            "title": extraction.title or "Untitled RAID Item",
            "description": extraction.description or extraction.title,
            "workstream": extraction.workstream or signal.workstream or "General",
            "phase": extraction.phase or signal.phase or "Realize",
            "probability": extraction.probability,
            "impact": extraction.impact,
            "severity": extraction.severity,
            "scoring_reason": extraction.reasoning_summary or f"Identified {cat_str} from {signal.source_type}",
            "likely_owner": extraction.owner,
            "due_date": extraction.due_date or "",
            "mitigation_action": extraction.mitigation_action or "",
            "confidence_score": extraction.confidence,
            "evidence_quote": evidence_text,
            "evidence_status": evidence_status_val
        }
