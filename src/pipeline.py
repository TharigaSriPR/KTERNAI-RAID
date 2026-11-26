"""
RAID Agent Pipeline Orchestrator.
KTern.AI / KaarTech Architectural Dataflow:
  Sources -> Ingestion -> Normalisation -> Noise Filtering
  -> [LLM Classification & Extraction + Rule-Based Predictive Engine]
  -> Duplicate Detection & Matching -> Validation & Evidence Linking
  -> Human Review Queue & SQLite Database
"""

import uuid
import threading
from typing import List, Dict, Any, Optional, Sequence, Union, Tuple
from pathlib import Path

from config import CONFIDENCE_THRESHOLD, MAX_RETRIES

from src.ingestion.email_ingestor import EmailIngestor
from src.ingestion.chat_ingestor import ChatIngestor
from src.ingestion.meeting_ingestor import MeetingIngestor
from src.ingestion.plan_ingestor import PlanIngestor
from src.ingestion.defect_ingestor import DefectIngestor

from src.normalisation.normaliser import SignalNormaliser
from src.normalisation.schemas import ProjectSignal, SourceType
from src.noise_filter.filter_rules import NoiseFilter
from src.classification.classifier import RAIDClassifier
from src.classification.schemas import ClassificationStatus
from src.predictive_engine import PredictiveEngine, PredictionItem
from src.deduplication import SemanticDeduplicator, MatchStatus
from src.evidence.linker import EvidenceLinker
from src.database.db_manager import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAIDPipeline:
    def __init__(self, db_manager: Optional[DatabaseManager] = None, max_retries: int = MAX_RETRIES):
        self.db = db_manager or DatabaseManager()
        self.max_retries = max_retries
        self._in_flight_lock = threading.Lock()
        self._in_flight_signals: set = set()

        self.email_ingestor = EmailIngestor()
        self.chat_ingestor = ChatIngestor()
        self.meeting_ingestor = MeetingIngestor()
        self.plan_ingestor = PlanIngestor()
        self.defect_ingestor = DefectIngestor()

        self.normaliser = SignalNormaliser()
        self.noise_filter = NoiseFilter()
        self.classifier = RAIDClassifier(db_manager=self.db)
        self.predictive_engine = PredictiveEngine()
        self.deduplicator = SemanticDeduplicator()
        self.evidence_linker = EvidenceLinker()

    def check_provider_readiness(self) -> Tuple[bool, str]:
        """
        Validates provider configuration and required credentials without exposing keys
        and without consuming API tokens or making premature network calls.
        """
        provider = self.classifier.provider
        if provider is None:
            return False, "No LLM provider configured."

        prov_name = getattr(provider, "provider_name", "").lower()
        if prov_name == "gemini":
            api_key = getattr(provider, "api_key", None)
            if not api_key or not str(api_key).strip():
                return False, "Gemini API key is not configured. Classification cannot proceed."
            if len(str(api_key).strip()) < 8:
                return False, "Gemini API key appears malformed or placeholder."
            return True, "Gemini provider configuration validated."
        elif prov_name == "mock":
            return True, "Mock provider is ready for local execution."
        elif prov_name == "openrouter":
            api_key = getattr(provider, "api_key", None)
            if not api_key or not str(api_key).strip():
                return False, "OpenRouter API key is not configured."
            return True, "OpenRouter provider configuration validated."

        if hasattr(provider, "is_configured"):
            ok = provider.is_configured()
            return (True, f"Provider '{prov_name}' configuration validated.") if ok else (False, f"Provider '{prov_name}' not configured.")

        return True, f"Provider '{prov_name}' configuration validated."

    def process_signals(
        self,
        raw_signals: Sequence[Union[Dict[str, Any], ProjectSignal]],
        reference_date: Optional[Any] = None,
        user_id: str = "default_user",
        run_predictive_engine: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end pipeline on raw ingested signals for a specific user.
        Accepts both dictionary inputs and pre-instantiated ProjectSignal instances.

        :param raw_signals: Sequence of input signals (dicts or ProjectSignal).
        :param reference_date: Optional reference date for schedule slippage evaluation.
        :param user_id: Tenant user ID.
        :param run_predictive_engine: If True (default), executes rule-based foresight
                                      predictions (schedule slippage, defect clustering,
                                      dependency blockages, contradicted assumptions) and
                                      generates foresight draft items. If False, skips
                                      predictive foresight, running in isolated classification mode.
        """
        results = {
            "total_ingested": len(raw_signals),
            "normalized": 0,
            "noise_filtered": 0,
            "classified_drafts": 0,
            "predictive_drafts": 0,
            "duplicates_detected": 0,
            "drafts_created": 0,
            "already_processed": 0,
            "drafts": []
        }

        # Step 1: Normalisation
        normalized_signals: List[ProjectSignal] = []
        for index, raw in enumerate(raw_signals):
            try:
                sig = self.normaliser.normalise(raw)
                if sig:
                    sig.user_id = user_id
                    # Step 2: Noise Filtering
                    self.noise_filter.apply_filter(sig)
                    # Save raw signal in DB
                    is_new = self.db.save_raw_signal(sig, user_id=user_id)

                    if sig.is_noise:
                        results["noise_filtered"] += 1
                        self.db.update_signal_classification_status(
                            sig.id,
                            status="NOT_RAID",
                            failure_reason="Pre-filtered as noise",
                            user_id=user_id,
                            increment_attempts=False
                        )
                    else:
                        if not is_new:
                            existing_sig = self.db.get_raw_signal_by_id(sig.id, user_id=user_id)
                            has_draft = self.db.has_draft_for_signal(sig.id, user_id=user_id)
                            curr_status = existing_sig.get("classification_status") if existing_sig else None
                            if has_draft or curr_status in ("COMPLETED", "NOT_RAID", "BELOW_CONFIDENCE_THRESHOLD"):
                                results["already_processed"] += 1
                                continue
                        normalized_signals.append(sig)
            except Exception as exc:
                rec_id = raw.id if isinstance(raw, ProjectSignal) else (raw.get("id") or raw.get("source_id") if isinstance(raw, dict) else f"index-{index}")
                logger.warning("Pipeline skipping malformed signal at index %d (%s): %s", index, rec_id, exc)

        results["normalized"] = len(normalized_signals)

        # Existing items for deduplication and assumption monitoring scoped to user
        existing_drafts = self.db.get_pending_drafts(user_id=user_id)
        approved_items = self.db.get_approved_items(user_id=user_id)
        all_existing = existing_drafts + approved_items
        existing_assumptions = self.db.get_assumptions(user_id=user_id) if run_predictive_engine else []

        new_drafts: List[Dict[str, Any]] = []

        # Step 3: LLM Classification & Extraction
        newly_classified_assumptions: List[Dict[str, Any]] = []
        effective_threshold = self.classifier.confidence_threshold

        for sig in normalized_signals:
            # Mark status as PROCESSING
            self.db.update_signal_classification_status(
                sig.id,
                status="PROCESSING",
                user_id=user_id,
                increment_attempts=True
            )

            outcome = self.classifier.classify_signal_detailed(sig)

            if outcome.status == ClassificationStatus.FAILED:
                err_cat = outcome.error_category.value if outcome.error_category else None
                self.db.update_signal_classification_status(
                    sig.id,
                    status="FAILED",
                    failure_reason=outcome.error_message,
                    error_category=err_cat,
                    user_id=user_id,
                    increment_attempts=False,
                    llm_provider=outcome.llm_provider,
                    llm_model=outcome.llm_model
                )
                logger.warning("Signal %s classification failed: %s (%s)", sig.id, outcome.error_message, err_cat)
                continue

            if outcome.status == ClassificationStatus.NOT_RAID:
                self.db.update_signal_classification_status(
                    sig.id,
                    status="NOT_RAID",
                    user_id=user_id,
                    increment_attempts=False,
                    llm_provider=outcome.llm_provider,
                    llm_model=outcome.llm_model
                )
                continue

            if outcome.status == ClassificationStatus.BELOW_CONFIDENCE_THRESHOLD:
                extracted_json = None
                if outcome.extraction:
                    try:
                        extracted_json = outcome.extraction.model_dump_json()
                    except Exception:
                        try:
                            extracted_json = json.dumps(outcome.extraction.model_dump(mode="json"))
                        except Exception:
                            extracted_json = None
                self.db.update_signal_classification_status(
                    sig.id,
                    status="BELOW_CONFIDENCE_THRESHOLD",
                    failure_reason=extracted_json or f"Confidence {outcome.extraction.confidence if outcome.extraction else 0.0:.2f} below threshold",
                    user_id=user_id,
                    increment_attempts=False,
                    llm_provider=outcome.llm_provider,
                    llm_model=outcome.llm_model
                )
                continue

            extracted = outcome.extraction
            if extracted and extracted.is_raid and extracted.confidence >= effective_threshold:
                self.db.update_signal_classification_status(
                    sig.id,
                    status="COMPLETED",
                    user_id=user_id,
                    increment_attempts=False,
                    llm_provider=outcome.llm_provider,
                    llm_model=outcome.llm_model
                )
                cat_val = extracted.category.value if hasattr(extracted.category, "value") else str(extracted.category)
                draft_id = f"raid-{uuid.uuid4().hex[:8]}"
                evidence_status_str = (
                    extracted.evidence_status.value
                    if hasattr(extracted.evidence_status, "value")
                    else str(extracted.evidence_status or "unavailable")
                )
                draft_record = {
                    "id": draft_id,
                    "user_id": user_id,
                    "type": cat_val,
                    "title": extracted.title or sig.subject,
                    "description": extracted.description or sig.content,
                    "workstream": extracted.workstream or sig.workstream or "General",
                    "phase": extracted.phase or sig.phase or "Realize",
                    "probability": extracted.probability,
                    "impact": extracted.impact,
                    "severity": extracted.severity,
                    "scoring_reason": extracted.reasoning_summary or f"Classified by LLM with confidence {extracted.confidence:.2f}",
                    "likely_owner": extracted.owner,
                    "due_date": extracted.due_date or "",
                    "mitigation_action": extracted.mitigation_action or "",
                    "confidence_score": extracted.confidence,
                    "evidence_text": extracted.evidence or "[Evidence unavailable in source content]",
                    "evidence_source_id": sig.id,
                    "evidence_source_type": sig.source_type,
                    "detection_source": "llm_extraction",
                    "duplicate_of_id": None,
                    "status": "Draft",
                    "source_file": sig.raw_meta.get("source_file", "meeting_notes") if isinstance(sig.raw_meta, dict) else "meeting_notes",
                    "meeting_date": (sig.raw_meta.get("meeting_date") if isinstance(sig.raw_meta, dict) else None) or sig.timestamp,
                    "evidence_status": evidence_status_str,
                    "llm_provider": outcome.llm_provider or getattr(self.classifier.provider, "provider_name", "unknown"),
                    "llm_model": outcome.llm_model or getattr(self.classifier.provider, "model_name", "unknown")
                }
                new_drafts.append(draft_record)
                results["classified_drafts"] += 1

                if cat_val == "Assumption":
                    newly_classified_assumptions.append(draft_record)

        # Step 4: Rule-Based Predictive Engine (Foresight)
        if run_predictive_engine:
            all_assumptions = existing_assumptions + newly_classified_assumptions
            predictions = self.predictive_engine.run_foresight(
                signals=normalized_signals,
                existing_assumptions=all_assumptions,
                reference_date=reference_date
            )

            for pred in predictions:
                # Store in predictions table
                self.db.save_prediction(pred, user_id=user_id)

                # Convert to draft RAID item for human review
                pred_draft_id = f"raid-{uuid.uuid4().hex[:8]}"
                primary_source = pred.source_signal_ids[0] if pred.source_signal_ids else ""
                pred_record = {
                    "id": pred_draft_id,
                    "user_id": user_id,
                    "type": pred.suggested_raid_category,
                    "title": pred.title,
                    "description": pred.description,
                    "workstream": pred.workstream,
                    "phase": pred.phase,
                    "probability": "High" if pred.suggested_raid_category == "Risk" else None,
                    "impact": "High" if pred.severity == "High" else "Medium",
                    "severity": pred.severity if pred.suggested_raid_category == "Issue" else None,
                    "scoring_reason": f"Rule-based foresight ({pred.prediction_type}) with confidence {pred.confidence:.2f}",
                    "likely_owner": None,
                    "due_date": "Immediate",
                    "mitigation_action": "Review schedule or quality logs with workstream lead",
                    "confidence_score": pred.confidence,
                    "evidence_text": pred.evidence,
                    "evidence_source_id": primary_source,
                    "evidence_source_type": "project_plan" if pred.prediction_type.startswith("TASK") else "predictive_engine",
                    "detection_source": "predictive_engine",
                    "duplicate_of_id": None,
                    "status": "Draft",
                    "evidence_status": "verified",
                    "llm_provider": "rule_based_foresight",
                    "llm_model": "predictive_engine"
                }
                new_drafts.append(pred_record)
                results["predictive_drafts"] += 1

        # Step 5: Semantic Deduplication & Validation Linking
        for draft in new_drafts:
            # Check against existing drafts + approved records
            dedupe_result = self.deduplicator.check_duplicate(draft, all_existing)
            if dedupe_result.match_status == MatchStatus.LIKELY_DUPLICATE:
                draft["duplicate_of_id"] = dedupe_result.matched_raid_id
                results["duplicates_detected"] += 1

            # Log deduplication result in database
            if dedupe_result.matched_raid_id:
                self.db.save_deduplication_match(dedupe_result, user_id=user_id)

            # Step 6: Evidence linking validation & SQLite persistence
            if not self.db.draft_exists_for_signal(draft["evidence_source_id"], draft["type"], draft["title"], user_id=user_id):
                self.db.save_raid_draft(draft, user_id=user_id)
                results["drafts_created"] += 1

        results["drafts"] = new_drafts

        logger.info(
            "Pipeline processing complete: %d signals ingested, %d drafts created (%d classified, %d predictive, %d duplicates flagged)",
            results["total_ingested"], results["drafts_created"],
            results["classified_drafts"], results["predictive_drafts"],
            results["duplicates_detected"]
        )
        return results

    def retry_failed_signals(
        self,
        user_id: str = "default_user",
        signal_ids: Optional[List[str]] = None,
        include_terminal: bool = False,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retries classification for signals marked as FAILED.
        Bounded by MAX_RETRIES, validates provider readiness before dispatching,
        and skips terminal errors unless explicitly requested.
        Guarantees that existing drafts are not duplicated and records attempt counts.
        """
        if signal_ids:
            failed_records = []
            for sid in signal_ids:
                rec = self.db.get_raw_signal_by_id(sid, user_id=user_id)
                if rec and rec.get("classification_status") == "FAILED":
                    failed_records.append(rec)
        else:
            failed_records = self.db.get_failed_signals(user_id=user_id)

        eff_max_retries = int(max_retries if max_retries is not None else self.max_retries)
        terminal_categories = {"INVALID_JSON", "SCHEMA_VALIDATION_ERROR", "EMPTY_RESPONSE"}

        results: Dict[str, Any] = {
            "status": "success",
            "retried_count": 0,
            "recovered_drafts": 0,
            "still_failed": 0,
            "not_raid": 0,
            "below_confidence": 0,
            "skipped_count": 0,
            "skipped": [],
            "drafts": []
        }

        # Stage 6: Provider Pre-Flight Check
        is_ready, preflight_msg = self.check_provider_readiness()
        if not is_ready:
            logger.warning("Provider pre-flight check failed before retry: %s", preflight_msg)
            results["status"] = "error"
            results["error"] = preflight_msg
            results["skipped_count"] = len(failed_records)
            results["skipped"] = [
                {"id": r["id"], "reason": f"Provider pre-flight failed: {preflight_msg}"}
                for r in failed_records
            ]
            return results

        effective_threshold = self.classifier.confidence_threshold

        for rec in failed_records:
            sig_id = rec["id"]
            attempts = int(rec.get("classification_attempts") or 0)
            err_cat = rec.get("classification_error_category")

            # Check bounded retry limit
            if attempts >= eff_max_retries:
                logger.info("Signal %s exceeded max retries (%d >= %d); skipping.", sig_id, attempts, eff_max_retries)
                results["skipped_count"] += 1
                results["skipped"].append({
                    "id": sig_id,
                    "reason": f"Max retries reached ({attempts} >= {eff_max_retries})"
                })
                continue

            # Check terminal error filtering
            if not include_terminal and err_cat in terminal_categories:
                logger.info("Signal %s has terminal error category '%s'; skipping automatic retry.", sig_id, err_cat)
                results["skipped_count"] += 1
                results["skipped"].append({
                    "id": sig_id,
                    "reason": f"Terminal error category '{err_cat}' skipped by default"
                })
                continue

            # Concurrency protection
            with self._in_flight_lock:
                if sig_id in self._in_flight_signals:
                    logger.info("Signal %s is already in-flight in a concurrent thread; skipping.", sig_id)
                    results["skipped_count"] += 1
                    results["skipped"].append({
                        "id": sig_id,
                        "reason": "Signal is currently in-flight in another thread"
                    })
                    continue
                self._in_flight_signals.add(sig_id)

            try:
                results["retried_count"] += 1
                sig = ProjectSignal(
                    id=rec["id"],
                    source_type=rec["source_type"],
                    source_id=rec.get("source_id", ""),
                    author=rec.get("author", "Unknown"),
                    timestamp=rec.get("timestamp", ""),
                    subject=rec.get("subject", ""),
                    content=rec["content"],
                    workstream=rec.get("workstream", "General"),
                    phase=rec.get("phase", "Realize"),
                    user_id=user_id
                )

                self.db.update_signal_classification_status(
                    sig.id,
                    status="PROCESSING",
                    user_id=user_id,
                    increment_attempts=True
                )

                outcome = self.classifier.classify_signal_detailed(sig)

                if outcome.status == ClassificationStatus.FAILED:
                    outcome_err_cat = outcome.error_category.value if outcome.error_category else None
                    self.db.update_signal_classification_status(
                        sig.id,
                        status="FAILED",
                        failure_reason=outcome.error_message,
                        error_category=outcome_err_cat,
                        user_id=user_id,
                        increment_attempts=False,
                        llm_provider=outcome.llm_provider,
                        llm_model=outcome.llm_model
                    )
                    results["still_failed"] += 1
                elif outcome.status == ClassificationStatus.NOT_RAID:
                    self.db.update_signal_classification_status(
                        sig.id,
                        status="NOT_RAID",
                        user_id=user_id,
                        increment_attempts=False,
                        llm_provider=outcome.llm_provider,
                        llm_model=outcome.llm_model
                    )
                    results["not_raid"] += 1
                elif outcome.status == ClassificationStatus.BELOW_CONFIDENCE_THRESHOLD:
                    self.db.update_signal_classification_status(
                        sig.id,
                        status="BELOW_CONFIDENCE_THRESHOLD",
                        user_id=user_id,
                        increment_attempts=False,
                        llm_provider=outcome.llm_provider,
                        llm_model=outcome.llm_model
                    )
                    results["below_confidence"] += 1
                elif outcome.status == ClassificationStatus.COMPLETED and outcome.extraction:
                    extracted = outcome.extraction
                    self.db.update_signal_classification_status(
                        sig.id,
                        status="COMPLETED",
                        user_id=user_id,
                        increment_attempts=False,
                        llm_provider=outcome.llm_provider,
                        llm_model=outcome.llm_model
                    )

                    cat_val = extracted.category.value if hasattr(extracted.category, "value") else str(extracted.category)
                    title = extracted.title or sig.subject
                    evidence_status_str = (
                        extracted.evidence_status.value
                        if hasattr(extracted.evidence_status, "value")
                        else str(extracted.evidence_status or "unavailable")
                    )

                    # Check if draft already exists before creating
                    if not self.db.draft_exists_for_signal(sig.id, cat_val, title, user_id=user_id):
                        draft_id = f"raid-{uuid.uuid4().hex[:8]}"
                        draft_record = {
                            "id": draft_id,
                            "user_id": user_id,
                            "type": cat_val,
                            "title": title,
                            "description": extracted.description or sig.content,
                            "workstream": extracted.workstream or sig.workstream or "General",
                            "phase": extracted.phase or sig.phase or "Realize",
                            "probability": extracted.probability,
                            "impact": extracted.impact,
                            "severity": extracted.severity,
                            "scoring_reason": extracted.reasoning_summary or f"Classified by LLM with confidence {extracted.confidence:.2f} (Retried)",
                            "likely_owner": extracted.owner,
                            "due_date": extracted.due_date or "",
                            "mitigation_action": extracted.mitigation_action or "",
                            "confidence_score": extracted.confidence,
                            "evidence_text": extracted.evidence or "[Evidence unavailable in source content]",
                            "evidence_source_id": sig.id,
                            "evidence_source_type": sig.source_type,
                            "detection_source": "llm_extraction",
                            "duplicate_of_id": None,
                            "status": "Draft",
                            "evidence_status": evidence_status_str,
                            "llm_provider": outcome.llm_provider or getattr(self.classifier.provider, "provider_name", "unknown"),
                            "llm_model": outcome.llm_model or getattr(self.classifier.provider, "model_name", "unknown")
                        }
                        self.db.save_raid_draft(draft_record, user_id=user_id)
                        results["recovered_drafts"] += 1
                        results["drafts"].append(draft_record)
            finally:
                with self._in_flight_lock:
                    self._in_flight_signals.discard(sig_id)

        logger.info(
            "Retry failed signals complete for user '%s': %d retried, %d recovered drafts, %d still failed, %d skipped",
            user_id, results["retried_count"], results["recovered_drafts"], results["still_failed"], results["skipped_count"]
        )
        return results
