"""
RaidService Layer.
Provides high-level business logic, status lifecycle management, approval, rejection,
modification, and transactional audit trails for the RAID Identification Agent POC.

Separates business rules from SQLite persistence and future Streamlit UI logic.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
import sqlite3
import tempfile
import os

from src.database.db_manager import DatabaseManager
from src.normalisation.schemas import ProjectSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Service-Level Exceptions
# ---------------------------------------------------------------------------

class RaidServiceError(Exception):
    """Base exception for all RAID Service business errors."""
    pass


class RaidNotFoundError(RaidServiceError):
    """Raised when a referenced RAID draft or approved item does not exist."""
    pass


class InvalidRaidStatusError(RaidServiceError):
    """Raised when an operation violates permitted status transitions."""
    pass


class RaidValidationError(RaidServiceError):
    """Raised when field values or updates violate domain validation constraints."""
    pass


class DuplicateApprovalError(RaidServiceError):
    """Raised when attempting to approve an already approved RAID item."""
    pass


class AuditPersistenceError(RaidServiceError):
    """Raised when an audit record or transactional operation fails."""
    pass


# ---------------------------------------------------------------------------
# Lifecycle Status Definitions
# ---------------------------------------------------------------------------

class RaidStatus(str, Enum):
    """Supported RAID draft lifecycle statuses."""
    DRAFT = "Draft"
    PENDING_REVIEW = "Pending_Review"
    APPROVED = "Accepted"  # Maps to database CHECK constraint 'Accepted'
    REJECTED = "Rejected"
    MODIFIED = "Modified"


IMMUTABLE_FIELDS = {
    "id", "draft_id", "evidence_source_id", "evidence_source_type",
    "detection_source", "created_at", "detected_at"
}

PERMITTED_UPDATE_FIELDS = {
    "title", "description", "type", "workstream", "phase",
    "probability", "impact", "severity", "scoring_reason",
    "likely_owner", "due_date", "mitigation_action",
    "confidence_score", "evidence_text", "status"
}

VALID_RAID_TYPES = {"Risk", "Assumption", "Issue", "Decision", "Dependency"}


# ---------------------------------------------------------------------------
# RaidService
# ---------------------------------------------------------------------------

class RaidService:
    """
    Business service layer managing the full RAID lifecycle.
    Enforces status transitions, atomic approvals, rejections, partial modifications,
    and append-only audit histories.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        pipeline: Optional[Any] = None
    ):
        self.db = db_manager or DatabaseManager()
        self._pipeline = pipeline

    @property
    def pipeline(self):
        """Lazy-loaded RAIDPipeline instance linked to this service's DatabaseManager."""
        if self._pipeline is None:
            from src.pipeline import RAIDPipeline
            self._pipeline = RAIDPipeline(self.db)
        return self._pipeline

    # -----------------------------------------------------------------------
    # Retrieval Methods
    # -----------------------------------------------------------------------

    def get_pending_drafts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves all RAID drafts awaiting human PM review."""
        return self.db.get_pending_drafts(user_id=user_id)

    def get_low_confidence_candidates(
        self,
        user_id: str = "default_user",
        include_deferred: bool = True,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieves signals marked BELOW_CONFIDENCE_THRESHOLD (and optionally DEFERRED)
        for human reviewer inspection. Reconstructs extraction metadata where available.
        Excludes signals that have already been confirmed into drafts.
        Reuses existing DatabaseManager.get_signals_by_classification_status.
        """
        eff_user_id = user_id or "default_user"
        raw_signals = self.db.get_signals_by_classification_status(
            status="BELOW_CONFIDENCE_THRESHOLD",
            user_id=eff_user_id,
            limit=limit
        )
        if include_deferred:
            deferred_signals = self.db.get_signals_by_classification_status(
                status="DEFERRED",
                user_id=eff_user_id,
                limit=limit
            )
            raw_signals = raw_signals + deferred_signals
            raw_signals.sort(key=lambda x: str(x.get("timestamp") or ""), reverse=True)
            if limit:
                raw_signals = raw_signals[:limit]

        candidates: List[Dict[str, Any]] = []
        for row in raw_signals:
            sig_id = row.get("id")
            # If draft already exists, skip from pending low-confidence queue
            if self.db.has_draft_for_signal(sig_id, user_id=eff_user_id):
                continue

            extracted: Dict[str, Any] = {}
            raw_reason = row.get("classification_failure_reason") or ""
            if raw_reason.startswith("{") and raw_reason.endswith("}"):
                try:
                    parsed = json.loads(raw_reason)
                    if isinstance(parsed, dict):
                        extracted = parsed
                except Exception:
                    extracted = {}

            conf_val = extracted.get("confidence")
            if conf_val is None:
                import re
                m = re.search(r"confidence\s+([0-9\.]+)", raw_reason, re.IGNORECASE)
                if m:
                    try:
                        conf_val = float(m.group(1))
                    except (ValueError, TypeError):
                        conf_val = 0.0
                else:
                    conf_val = 0.0
            try:
                conf_float = float(conf_val)
            except (ValueError, TypeError):
                conf_float = 0.0

            cand = {
                "signal_id": sig_id,
                "source_type": row.get("source_type") or "signal",
                "source_id": row.get("source_id") or "",
                "author": row.get("author") or "Unknown",
                "timestamp": str(row.get("timestamp") or "")[:19],
                "subject": row.get("subject") or "",
                "content": row.get("content") or "",
                "workstream": extracted.get("workstream") or row.get("workstream") or "General",
                "phase": extracted.get("phase") or row.get("phase") or "Realize",
                "predicted_category": extracted.get("category") or "Risk",
                "confidence_score": conf_float,
                "title": extracted.get("title") or row.get("subject") or f"Low-Confidence Candidate {sig_id}",
                "description": extracted.get("description") or row.get("content") or "",
                "evidence_text": extracted.get("evidence") or row.get("content") or "",
                "evidence_status": extracted.get("evidence_status") or "unavailable",
                "reasoning_summary": extracted.get("reasoning_summary") or raw_reason or "",
                "likely_owner": extracted.get("owner"),
                "due_date": extracted.get("due_date") or "",
                "probability": extracted.get("probability"),
                "impact": extracted.get("impact"),
                "severity": extracted.get("severity"),
                "mitigation_action": extracted.get("mitigation_action") or "",
                "llm_provider": row.get("llm_provider") or "unknown",
                "llm_model": row.get("llm_model") or "unknown",
                "status": row.get("classification_status") or "BELOW_CONFIDENCE_THRESHOLD"
            }
            candidates.append(cand)

        return candidates

    def get_approved_items(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves all items from the official approved RAID register."""
        return self.db.get_approved_items(user_id=user_id)

    # -----------------------------------------------------------------------
    # Data Folder Discovery, Deduplication & Pipeline Synchronization (Stage 7 & 7C)
    # -----------------------------------------------------------------------

    def get_mcp_client(self):
        """Returns or initializes the session MCP client instance."""
        if not hasattr(self, "_mcp_client") or self._mcp_client is None:
            from src.mcp.client import MCPClient
            self._mcp_client = MCPClient()
        return self._mcp_client

    def configure_mcp_folder(self, folder_path: str) -> Dict[str, Any]:
        """Configures and validates target folder via MCP Client."""
        client = self.get_mcp_client()
        return client.configure_folder(folder_path)

    def close_mcp_client(self) -> None:
        """Cleanly closes the active MCP client connection."""
        if hasattr(self, "_mcp_client") and self._mcp_client is not None:
            try:
                self._mcp_client.close()
            except Exception as exc:
                logger.warning("Error closing MCP client: %s", exc)
            finally:
                self._mcp_client = None

    def scan_data_folder(
        self,
        data_dir: Optional[Any] = None,
        user_id: str = "default_user",
        use_mcp: bool = False
    ) -> Dict[str, Any]:
        """
        Discovers all supported project source files under the data folder,
        computes SHA-256 hashes, checks against database sync records,
        and categorizes files as NEW, PROCESSED, MODIFIED, or FAILED.
        Can use MCP Client over STDIO when use_mcp is True.
        """
        eff_user_id = user_id or "default_user"
        dir_str = str(data_dir) if data_dir else None

        if use_mcp:
            try:
                client = self.get_mcp_client()
                return client.scan_folder(folder_path=dir_str, user_id=eff_user_id)
            except Exception as exc:
                logger.warning("MCP scan_folder failed (%s), falling back to direct scanner.", exc)

        from src.ingestion.folder_scanner import DataFolderScanner
        scanner = DataFolderScanner(db_manager=self.db, data_dir=Path(data_dir) if data_dir else None)
        return scanner.scan_data_folder(target_dir=Path(data_dir) if data_dir else None, user_id=eff_user_id)

    def process_discovered_files(
        self,
        file_paths: Optional[List[str]] = None,
        data_dir: Optional[Any] = None,
        user_id: str = "default_user",
        run_predictive_engine: bool = True,
        force_reprocess: bool = False,
        stage_callback: Optional[Callable[[str, str], None]] = None,
        use_mcp: bool = False
    ) -> Dict[str, Any]:
        """
        Processes discovered project data files through the complete architectural pipeline:
        File -> Ingestion Adapter -> ProjectSignal -> Normalisation -> Noise Filter ->
        LLM Classification -> RAID Drafts -> SQLite DB.

        When use_mcp=True, retrieves file contents and metadata through the MCP Client tool 'read_file'.
        Skips unchanged files unless force_reprocess is True.
        Prevents duplicate ProjectSignal creation and redundant LLM API calls.
        Updates source_file_sync records with processing counts and metrics.
        """
        from src.ingestion.folder_scanner import DataFolderScanner
        from src.normalisation.schemas import SourceType
        from config import BASE_DIR

        eff_user_id = user_id or "default_user"
        dir_path = Path(data_dir) if data_dir else None
        scanner = DataFolderScanner(db_manager=self.db, data_dir=dir_path)

        if use_mcp:
            try:
                client = self.get_mcp_client()
                scan_summary = client.scan_folder(folder_path=str(dir_path) if dir_path else None, user_id=eff_user_id)
            except Exception as exc:
                logger.warning("MCP scan failed during processing (%s), falling back to direct scanner.", exc)
                scan_summary = scanner.scan_data_folder(target_dir=dir_path, user_id=eff_user_id)
        else:
            scan_summary = scanner.scan_data_folder(target_dir=dir_path, user_id=eff_user_id)

        # Determine target files
        if file_paths:
            target_files = [f for f in scan_summary["files"] if f["file_path"] in file_paths]
        elif force_reprocess:
            target_files = scan_summary["files"]
        else:
            # Process NEW (DISCOVERED), MODIFIED, or previously FAILED files
            target_files = [
                f for f in scan_summary["files"]
                if f.get("status") in ("DISCOVERED", "MODIFIED", "FAILED")
            ]

        results = {
            "total_files_scanned": scan_summary["total_supported"],
            "files_targeted": len(target_files),
            "files_processed_successfully": 0,
            "files_failed": 0,
            "total_signals_ingested": 0,
            "total_signals_normalized": 0,
            "total_noise_filtered": 0,
            "total_drafts_created": 0,
            "total_duplicates_skipped": 0,
            "file_results": []
        }

        if stage_callback:
            stage_callback("DISCOVERY", f"Targeting {len(target_files)} project data file(s) for ingestion.")

        for f_info in target_files:
            rel_path_str = f_info["file_path"]
            file_name = f_info["file_name"]
            source_type_str = f_info["source_type"]
            full_path = BASE_DIR / rel_path_str if not Path(rel_path_str).is_absolute() else Path(rel_path_str)

            # Initialize / save tracking record in database for this file
            self.db.save_file_sync_record(f_info, user_id=eff_user_id)

            if use_mcp:
                try:
                    client = self.get_mcp_client()
                    mcp_read = client.read_file(rel_path_str)
                    if not mcp_read.get("success", True):
                        err_msg = mcp_read.get("error", "MCP read_file failed")
                        logger.error("MCP read_file rejected %s: %s", rel_path_str, err_msg)
                        self.db.update_file_sync_status(
                            file_path=rel_path_str,
                            status="FAILED",
                            error_message=f"MCP Read Error: {err_msg}",
                            user_id=eff_user_id
                        )
                        results["files_failed"] += 1
                        results["file_results"].append({
                            "file_path": rel_path_str,
                            "file_name": file_name,
                            "status": "FAILED",
                            "error": err_msg,
                            "signals_count": 0,
                            "drafts_created": 0
                        })
                        continue
                except Exception as mcp_exc:
                    logger.warning("MCP read call failed for %s (%s), proceeding with direct file ingestion.", rel_path_str, mcp_exc)

            if stage_callback:
                stage_callback("INGESTING", f"Reading source records from {file_name} ({source_type_str})...")

            try:
                st_enum = SourceType(source_type_str)
                ingestor = scanner.get_ingestor_for_source_type(st_enum)
                signals = ingestor.ingest(full_path)
            except Exception as exc:
                logger.error("Failed to ingest source file %s: %s", rel_path_str, exc)
                self.db.update_file_sync_status(
                    file_path=rel_path_str,
                    status="FAILED",
                    error_message=f"Ingestion error: {exc}",
                    user_id=eff_user_id
                )
                results["files_failed"] += 1
                results["file_results"].append({
                    "file_path": rel_path_str,
                    "file_name": file_name,
                    "status": "FAILED",
                    "error": str(exc),
                    "signals_count": 0,
                    "drafts_created": 0
                })
                continue

            # Attach source file lineage in raw_meta
            for sig in signals:
                if isinstance(sig.raw_meta, dict):
                    sig.raw_meta["source_file"] = rel_path_str
                sig.user_id = eff_user_id

            if stage_callback:
                stage_callback("PROCESSING", f"Executing pipeline on {len(signals)} signals from {file_name}...")

            try:
                pipe_res = self.pipeline.process_signals(
                    signals,
                    user_id=eff_user_id,
                    run_predictive_engine=run_predictive_engine
                )

                created_drafts_count = pipe_res.get("drafts_created", 0)
                norm_count = pipe_res.get("normalized", 0)
                noise_count = pipe_res.get("noise_filtered", 0)
                dups_count = pipe_res.get("duplicates_detected", 0)
                already_proc = pipe_res.get("already_processed", 0)

                # Update database sync record
                self.db.update_file_sync_status(
                    file_path=rel_path_str,
                    status="PROCESSED",
                    record_count=len(signals),
                    signals_created=norm_count,
                    signals_classified=pipe_res.get("classified_drafts", 0),
                    drafts_created=created_drafts_count,
                    error_message=None,
                    user_id=eff_user_id
                )

                results["files_processed_successfully"] += 1
                results["total_signals_ingested"] += len(signals)
                results["total_signals_normalized"] += norm_count
                results["total_noise_filtered"] += noise_count
                results["total_drafts_created"] += created_drafts_count
                results["total_duplicates_skipped"] += dups_count + already_proc

                results["file_results"].append({
                    "file_path": rel_path_str,
                    "file_name": file_name,
                    "source_type": source_type_str,
                    "status": "PROCESSED",
                    "signals_count": len(signals),
                    "drafts_created": created_drafts_count,
                    "noise_filtered": noise_count,
                    "already_processed": already_proc
                })

            except Exception as exc:
                logger.error("Pipeline failure while processing %s: %s", rel_path_str, exc)
                self.db.update_file_sync_status(
                    file_path=rel_path_str,
                    status="FAILED",
                    error_message=f"Pipeline error: {exc}",
                    user_id=eff_user_id
                )
                results["files_failed"] += 1
                results["file_results"].append({
                    "file_path": rel_path_str,
                    "file_name": file_name,
                    "status": "FAILED",
                    "error": str(exc),
                    "signals_count": len(signals),
                    "drafts_created": 0
                })

        if stage_callback:
            stage_callback("COMPLETED", f"Synchronisation complete. Ingested {results['total_signals_ingested']} signals, generated {results['total_drafts_created']} drafts.")

        return results

    def get_file_sync_history(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves history of all tracked project data files and sync states."""
        eff_user_id = user_id or "default_user"
        return self.db.get_all_file_sync_records(user_id=eff_user_id)


    def get_approved_raid(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Alias for get_approved_items."""
        return self.get_approved_items(user_id=user_id)

    def get_rejected_drafts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves rejected drafts for historical review."""
        return self.db.get_rejected_drafts(user_id=user_id)

    def get_raid_by_id(self, raid_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves a RAID item by ID, checking both drafts and approved registers.
        """
        draft = self.db.get_raid_draft_by_id(raid_id, user_id=user_id)
        if draft:
            return draft
        return self.db.get_approved_item_by_id(raid_id, user_id=user_id)

    def get_audit_history(self, raid_id: Optional[str] = None, user_id: Optional[str] = "default_user") -> List[Dict[str, Any]]:
        """Retrieves the append-only audit history, optionally filtered by item ID."""
        return self.db.get_audit_history(item_id=raid_id, user_id=user_id)

    def get_predictions(self, status: Optional[str] = None, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves predictive foresight findings."""
        return self.db.get_predictions(status=status, user_id=user_id)

    def get_deduplication_matches(self, new_raid_id: Optional[str] = None, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves deduplication match records."""
        return self.db.get_deduplication_matches(new_raid_id=new_raid_id, user_id=user_id)

    def get_all_drafts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves all RAID drafts across all lifecycle statuses."""
        return self.db.get_all_drafts(user_id=user_id)

    def get_kpis(self, user_id: str = "default_user") -> Dict[str, int]:
        """Calculates KPI summary metrics directly from database."""
        return self.db.get_kpis(user_id=user_id)

    # -----------------------------------------------------------------------
    # Approval Functionality
    # -----------------------------------------------------------------------

    def approve_raid(
        self,
        raid_id: str,
        approved_by: str,
        comments: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Atomically approves a pending RAID draft and moves it into the approved register.
        
        Business Rules:
        1. Draft must exist and belong to user_id (if specified).
        2. Draft cannot already be approved (DuplicateApprovalError).
        3. Draft cannot be in 'Rejected' state (InvalidRaidStatusError).
        4. Required fields (type, title, description) must be validated.
        5. Duplicate warnings are recorded in audit trail.
        6. Entire operation runs inside an atomic SQLite transaction with rollback on failure.
        """
        if not approved_by or not str(approved_by).strip():
            raise RaidValidationError("A valid approver name or identity is required.")

        draft = self.db.get_raid_draft_by_id(raid_id, user_id=user_id)
        if not draft:
            raise RaidNotFoundError(f"RAID draft with ID '{raid_id}' not found.")

        eff_user_id = user_id or draft.get("user_id") or "default_user"

        current_status = draft.get("status")
        if current_status in (RaidStatus.APPROVED.value, "Approved"):
            raise DuplicateApprovalError(f"RAID item '{raid_id}' has already been approved.")
        if current_status == RaidStatus.REJECTED.value:
            raise InvalidRaidStatusError(f"Cannot approve RAID item '{raid_id}' because it has been rejected.")
        if current_status not in (RaidStatus.DRAFT.value, RaidStatus.PENDING_REVIEW.value, RaidStatus.MODIFIED.value):
            raise InvalidRaidStatusError(f"Cannot approve RAID item with status '{current_status}'.")

        # Check if already in approved register
        existing_approved = self.db.get_approved_item_by_draft_id(raid_id, user_id=eff_user_id)
        if existing_approved:
            raise DuplicateApprovalError(f"Approved record already exists for draft '{raid_id}'.")

        # Validate required fields
        raid_type = draft.get("type")
        if not raid_type or raid_type not in VALID_RAID_TYPES:
            raise RaidValidationError(f"Invalid or missing RAID type: '{raid_type}'.")
        title = draft.get("title")
        if not title or not str(title).strip():
            raise RaidValidationError("RAID title is required and cannot be empty.")
        desc = draft.get("description")
        if not desc or not str(desc).strip():
            raise RaidValidationError("RAID description is required and cannot be empty.")

        # Check for duplicate warnings
        dup_warning = ""
        if draft.get("duplicate_of_id"):
            dup_warning = f" [Duplicate Warning: flagged against item {draft['duplicate_of_id']}]"

        approved_id = f"appr-{uuid.uuid4().hex[:8]}"
        now_utc = datetime.now(timezone.utc).isoformat()

        approved_record = {
            "id": approved_id,
            "user_id": eff_user_id,
            "draft_id": raid_id,
            "type": raid_type,
            "title": title.strip(),
            "description": desc.strip(),
            "workstream": draft.get("workstream") or "General",
            "phase": draft.get("phase") or "Realize",
            "severity_or_impact": draft.get("severity") or draft.get("impact") or draft.get("probability") or "Medium",
            "owner": draft.get("likely_owner"),
            "due_date": draft.get("due_date") or "",
            "mitigation_action": draft.get("mitigation_action") or "",
            "status": "Open",
            "evidence_links": draft.get("evidence_text") or "",
            "approved_by": approved_by.strip(),
            "approved_at": now_utc,
            "last_updated_at": now_utc
        }

        action_note = f"Approved by {approved_by.strip()}{dup_warning}"
        if comments:
            action_note += f". Comments: {comments.strip()}"

        audit_record = {
            "user_id": eff_user_id,
            "item_id": raid_id,
            "item_table": "raid_drafts",
            "action": "APPROVED",
            "proposed_data": json.dumps(dict(draft)),
            "final_data": json.dumps(approved_record),
            "action_by": approved_by.strip(),
            "action_notes": action_note,
            "timestamp": now_utc
        }

        try:
            self.db.execute_approval_transaction(
                draft_id=raid_id,
                approved_record=approved_record,
                audit_record=audit_record,
                approved_status=RaidStatus.APPROVED.value,
                user_id=eff_user_id
            )
            logger.info("Successfully approved RAID draft %s as approved item %s by %s", raid_id, approved_id, approved_by)
            return approved_record
        except sqlite3.IntegrityError as exc:
            logger.warning("Duplicate approval or integrity constraint conflict for %s: %s", raid_id, exc)
            raise DuplicateApprovalError(f"RAID item '{raid_id}' has already been approved.") from exc
        except Exception as exc:
            if "already approved" in str(exc).lower() or "unique constraint" in str(exc).lower():
                logger.warning("Duplicate approval detected during transaction for %s: %s", raid_id, exc)
                raise DuplicateApprovalError(f"RAID item '{raid_id}' has already been approved.") from exc
            logger.error("Failed to execute approval transaction for %s: %s", raid_id, exc)
            raise AuditPersistenceError(f"Database error during approval transaction: {exc}") from exc

    # -----------------------------------------------------------------------
    # Rejection Functionality
    # -----------------------------------------------------------------------

    def reject_raid(
        self,
        raid_id: str,
        rejected_by: str,
        reason: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rejects a RAID draft, recording the mandatory reason and logging in audit trail.
        
        Business Rules:
        1. Draft must exist and belong to user_id (if specified).
        2. Cannot reject an already approved RAID item.
        3. Cannot reject an already rejected item.
        4. Non-empty rejection reason is required.
        5. Rejected item is preserved in database for historical review.
        """
        if not rejected_by or not str(rejected_by).strip():
            raise RaidValidationError("A valid user name or identity is required.")
        if not reason or not str(reason).strip():
            raise RaidValidationError("A non-empty rejection reason is required.")

        draft = self.db.get_raid_draft_by_id(raid_id, user_id=user_id)
        if not draft:
            raise RaidNotFoundError(f"RAID draft with ID '{raid_id}' not found.")

        eff_user_id = user_id or draft.get("user_id") or "default_user"

        current_status = draft.get("status")
        if current_status in (RaidStatus.APPROVED.value, "Approved"):
            raise InvalidRaidStatusError(f"Cannot reject RAID item '{raid_id}' because it has already been approved.")
        if current_status == RaidStatus.REJECTED.value:
            raise InvalidRaidStatusError(f"RAID item '{raid_id}' has already been rejected.")
        if current_status not in (RaidStatus.DRAFT.value, RaidStatus.PENDING_REVIEW.value, RaidStatus.MODIFIED.value):
            raise InvalidRaidStatusError(f"Cannot reject RAID item with status '{current_status}'.")

        now_utc = datetime.now(timezone.utc).isoformat()
        reason_clean = reason.strip()

        audit_record = {
            "user_id": eff_user_id,
            "item_id": raid_id,
            "item_table": "raid_drafts",
            "action": "REJECTED",
            "proposed_data": json.dumps(dict(draft)),
            "final_data": json.dumps({"status": "Rejected", "reason": reason_clean}),
            "action_by": rejected_by.strip(),
            "action_notes": f"Rejection reason: {reason_clean}",
            "timestamp": now_utc
        }

        try:
            self.db.execute_rejection_transaction(
                draft_id=raid_id,
                rejection_reason=reason_clean,
                audit_record=audit_record,
                user_id=eff_user_id
            )
            logger.info("Successfully rejected RAID draft %s by %s. Reason: %s", raid_id, rejected_by, reason_clean)
            return self.db.get_raid_draft_by_id(raid_id, user_id=eff_user_id)
        except Exception as exc:
            logger.error("Failed to execute rejection transaction for %s: %s", raid_id, exc)
            raise AuditPersistenceError(f"Database error during rejection transaction: {exc}") from exc

    # -----------------------------------------------------------------------
    # Modification Functionality
    # -----------------------------------------------------------------------

    def modify_raid(
        self,
        raid_id: str,
        modified_by: str,
        updates: Dict[str, Any],
        comments: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Modifies specified fields of a pending RAID draft.
        
        Business Rules:
        1. Draft must exist and belong to user_id (if specified).
        2. Cannot modify an already approved RAID item.
        3. Cannot modify a rejected RAID item.
        4. Immutable fields (id, evidence_source_id, etc.) cannot be modified.
        5. Unknown or invalid field names are rejected with RaidValidationError.
        6. Field values are strictly validated (title/description non-empty, valid type, etc.).
        7. Only the specified fields are changed; unmodified fields are preserved.
        8. Before and after values are captured in audit history.
        """
        if not modified_by or not str(modified_by).strip():
            raise RaidValidationError("A valid user name or identity is required.")
        if not updates or not isinstance(updates, dict):
            raise RaidValidationError("Updates dictionary cannot be empty.")

        draft = self.db.get_raid_draft_by_id(raid_id, user_id=user_id)
        if not draft:
            raise RaidNotFoundError(f"RAID draft with ID '{raid_id}' not found.")

        eff_user_id = user_id or draft.get("user_id") or "default_user"

        current_status = draft.get("status")
        if current_status in (RaidStatus.APPROVED.value, "Approved"):
            raise InvalidRaidStatusError(f"Cannot modify already approved RAID item '{raid_id}' through draft modification.")
        if current_status == RaidStatus.REJECTED.value:
            raise InvalidRaidStatusError(f"Cannot modify rejected RAID item '{raid_id}'.")

        # 1. Check for immutable fields
        for key in updates.keys():
            if key in IMMUTABLE_FIELDS:
                raise RaidValidationError(f"Cannot modify immutable field '{key}'.")
            if key not in PERMITTED_UPDATE_FIELDS:
                raise RaidValidationError(f"Invalid field name '{key}'. Permitted fields: {sorted(list(PERMITTED_UPDATE_FIELDS))}")

        # 2. Validate field values
        clean_updates: Dict[str, Any] = {}
        for key, val in updates.items():
            if key == "title":
                if not val or not str(val).strip():
                    raise RaidValidationError("Title cannot be empty.")
                clean_updates[key] = str(val).strip()
            elif key == "description":
                if not val or not str(val).strip():
                    raise RaidValidationError("Description cannot be empty.")
                clean_updates[key] = str(val).strip()
            elif key == "type":
                if val not in VALID_RAID_TYPES:
                    raise RaidValidationError(f"Invalid RAID type '{val}'. Must be one of {sorted(list(VALID_RAID_TYPES))}")
                clean_updates[key] = val
            elif key == "confidence_score":
                try:
                    conf = float(val)
                    if not (0.0 <= conf <= 1.0):
                        raise ValueError()
                    clean_updates[key] = conf
                except Exception:
                    raise RaidValidationError("confidence_score must be a float between 0.0 and 1.0.")
            else:
                clean_updates[key] = val

        # Mark draft as Modified if not already
        if "status" not in clean_updates:
            clean_updates["status"] = RaidStatus.MODIFIED.value

        old_values = {k: draft.get(k) for k in clean_updates.keys()}
        now_utc = datetime.now(timezone.utc).isoformat()

        audit_record = {
            "user_id": eff_user_id,
            "item_id": raid_id,
            "item_table": "raid_drafts",
            "action": "MODIFIED",
            "proposed_data": json.dumps(old_values),
            "final_data": json.dumps(clean_updates),
            "action_by": modified_by.strip(),
            "action_notes": comments or f"Modified fields: {', '.join(clean_updates.keys())}",
            "timestamp": now_utc
        }

        try:
            self.db.execute_modification_transaction(
                draft_id=raid_id,
                updates=clean_updates,
                audit_record=audit_record,
                user_id=eff_user_id
            )
            logger.info("Successfully modified RAID draft %s by %s: %s", raid_id, modified_by, list(clean_updates.keys()))
            return self.db.get_raid_draft_by_id(raid_id, user_id=eff_user_id)
        except Exception as exc:
            logger.error("Failed to execute modification transaction for %s: %s", raid_id, exc)
            raise AuditPersistenceError(f"Database error during modification transaction: {exc}") from exc

    # -----------------------------------------------------------------------
    # Meeting Minutes Ingestion & Traceability Methods
    # -----------------------------------------------------------------------

    def is_source_already_ingested(self, source_name_or_id: str, user_id: str = "default_user") -> bool:
        """
        Checks if a source file or meeting has already been ingested for this user.
        Prevents duplicate ingestion of the same source file.
        """
        return self.db.is_source_already_ingested(source_name_or_id, user_id=user_id)

    def get_raw_signal_by_id(self, signal_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a raw signal record by ID, optionally scoped to a user."""
        return self.db.get_raw_signal_by_id(signal_id, user_id=user_id)

    def get_drafts_by_source(self, source_name: str, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves all drafts originating from a specific source file or meeting for a user."""
        return self.db.get_drafts_by_source_file(source_name, user_id=user_id)

    def get_source_metadata_for_draft(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves original source metadata (source file, meeting date, title, author, evidence)
        for a RAID draft from the underlying raw_signal record.
        """
        sig_id = draft.get("evidence_source_id")
        raw_sig = self.get_raw_signal_by_id(sig_id) if sig_id else None

        source_file = draft.get("source_file")
        meeting_date = draft.get("meeting_date")
        meeting_title = draft.get("title")
        source_id = sig_id or "N/A"
        author = "Meeting Scribe"

        if raw_sig:
            source_id = raw_sig.get("source_id") or sig_id
            author = raw_sig.get("author") or author
            meeting_title = raw_sig.get("subject") or meeting_title
            if not meeting_date:
                meeting_date = str(raw_sig.get("timestamp", ""))[:10]
            if not source_file:
                # Extract filename from source_id or raw_sig
                sid = str(raw_sig.get("source_id", ""))
                if "-para-" in sid or "-chunk-" in sid or "-mom-" in sid:
                    source_file = sid.split("-para-")[0].split("-chunk-")[0].split("-mom-")[0]
                else:
                    source_file = sid or raw_sig.get("subject")

        return {
            "source_file": source_file or "meeting_notes",
            "meeting_date": meeting_date or "—",
            "meeting_title": meeting_title,
            "source_id": source_id,
            "author": author,
            "evidence_text": draft.get("evidence_text") or (raw_sig.get("content")[:300] if raw_sig else "")
        }

    def process_meeting_minutes(
        self,
        files: Optional[List[Any]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        manual_entry: Optional[Dict[str, Any]] = None,
        stage_callback: Optional[Callable[[str, str], None]] = None,
        user_id: str = "default_user",
        run_predictive_engine: bool = True
    ) -> Dict[str, Any]:
        """
        Processes meeting minutes from uploaded files (.docx, .txt, .csv) and/or manual text entry
        through the complete end-to-end RAID Agent pipeline.

        Stages:
        1. Uploaded
        2. Reading meeting minutes
        3. Normalising content
        4. Filtering noise
        5. Extracting RAID items (Risks, Assumptions, Issues, Decisions, Dependencies)
        6. Checking duplicates
        7. Saving results (status strictly 'Draft', never automatically approved)
        8. Completed / Failed
        """
        def _notify(stage: str, details: str):
            if stage_callback:
                try:
                    stage_callback(stage, details)
                except Exception as cb_exc:
                    logger.debug("stage_callback raised: %s", cb_exc)

        # Stage 1: Uploaded
        _notify("Uploaded", "Validating uploaded documents and input payloads...")

        has_files = bool((files and len(files) > 0) or (file_paths and len(file_paths) > 0))
        has_manual = bool(manual_entry and str(manual_entry.get("content", "")).strip())

        if not has_files and not has_manual:
            _notify("Failed", "No files or manual content provided.")
            raise RaidValidationError("Please provide at least one meeting minutes file (.docx, .txt, .csv) or paste meeting minutes text.")

        # Stage 2: Reading meeting minutes
        _notify("Reading meeting minutes", "Reading meeting minutes documents...")

        all_signals: List[Any] = []
        skipped_sources: List[Dict[str, str]] = []
        errors: List[str] = []

        # 2a. Process file_paths
        if file_paths:
            for fp in file_paths:
                p = Path(fp)
                if not p.exists():
                    errors.append(f"File not found: {p.name}")
                    continue
                if p.suffix.lower() not in (".docx", ".txt", ".csv"):
                    errors.append(f"File '{p.name}' has unsupported extension '{p.suffix}'. Only .docx, .txt, .csv are supported.")
                    continue
                if p.stat().st_size == 0:
                    errors.append(f"File '{p.name}' is empty (0 bytes).")
                    continue
                if self.is_source_already_ingested(p.name, user_id=user_id):
                    skipped_sources.append({"source": p.name, "reason": "Already ingested; skipped to prevent duplicates."})
                    continue

                try:
                    sigs = self.pipeline.meeting_ingestor.ingest(p)
                    if sigs:
                        all_signals.extend(sigs)
                    else:
                        errors.append(f"No discussion items found in file '{p.name}'.")
                except Exception as exc:
                    errors.append(f"Failed to read file '{p.name}': {exc}")

        # 2b. Process uploaded file objects (Streamlit UploadedFile or dicts)
        if files:
            for f in files:
                fname = getattr(f, "name", None) or (f.get("name") if isinstance(f, dict) else "unknown_file")
                fext = Path(fname).suffix.lower()

                if fext not in (".docx", ".txt", ".csv"):
                    errors.append(f"File '{fname}' has unsupported format '{fext}'. Supported formats: .docx, .txt, .csv")
                    continue

                # Get bytes
                f_bytes = None
                if hasattr(f, "getvalue"):
                    f_bytes = f.getvalue()
                elif hasattr(f, "read"):
                    f_bytes = f.read()
                elif isinstance(f, dict) and "content" in f:
                    f_bytes = f["content"]
                    if isinstance(f_bytes, str):
                        f_bytes = f_bytes.encode("utf-8")

                if f_bytes is None or len(f_bytes) == 0:
                    errors.append(f"File '{fname}' is empty (0 bytes).")
                    continue

                if self.is_source_already_ingested(fname, user_id=user_id):
                    skipped_sources.append({"source": fname, "reason": "Already ingested; skipped to prevent duplicates."})
                    continue

                # Write to temp file for ingestion adapter
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=fext, delete=False) as tf:
                        tf.write(f_bytes)
                        temp_file = Path(tf.name)

                    sigs = self.pipeline.meeting_ingestor.ingest(temp_file, original_filename=fname)
                    if sigs:
                        all_signals.extend(sigs)
                    else:
                        errors.append(f"No readable discussion items found in file '{fname}'.")
                except Exception as exc:
                    errors.append(f"Failed to read file '{fname}': {exc}")
                finally:
                    if temp_file and temp_file.exists():
                        try:
                            temp_file.unlink()
                        except Exception:
                            pass

        # 2c. Process manual entry
        if has_manual:
            m_title = str(manual_entry.get("title", "")).strip() or "Meeting Minutes"
            m_content = str(manual_entry.get("content", "")).strip()
            m_date = str(manual_entry.get("date", "")).strip() or None
            m_proj = str(manual_entry.get("project_name", "")).strip() or None

            manual_source_id = f"manual-{m_title}"
            if self.is_source_already_ingested(manual_source_id, user_id=user_id):
                skipped_sources.append({"source": m_title, "reason": "Already ingested; skipped to prevent duplicates."})
            else:
                try:
                    m_sigs = self.pipeline.meeting_ingestor.ingest_text(
                        content=m_content,
                        title=m_title,
                        meeting_date=m_date,
                        project_name=m_proj
                    )
                    if m_sigs:
                        all_signals.extend(m_sigs)
                    else:
                        errors.append("No discussion points could be parsed from manual text input.")
                except Exception as exc:
                    errors.append(f"Failed to parse manual text input: {exc}")

        # If no valid signals extracted:
        if not all_signals:
            if skipped_sources and not errors:
                # All provided files were previously ingested duplicates
                existing_drafts = []
                for s in skipped_sources:
                    existing_drafts.extend(self.get_drafts_by_source(s["source"], user_id=user_id))
                _notify("Completed", "Source was previously ingested. Retrieved existing drafts without creating duplicates.")
                return {
                    "status": "skipped",
                    "total_signals": 0,
                    "noise_filtered": 0,
                    "normalized_count": 0,
                    "drafts_created": 0,
                    "duplicates_detected": 0,
                    "drafts": existing_drafts,
                    "skipped_sources": skipped_sources,
                    "errors": errors
                }
            err_msg = "; ".join(errors) if errors else "No readable meeting minutes content found."
            _notify("Failed", err_msg)
            raise RaidValidationError(f"Processing failed: {err_msg}")

        # Stage 3: Normalising content
        _notify("Normalising content", f"Normalising {len(all_signals)} extracted discussion point(s)...")
        normalized_signals: List[ProjectSignal] = []
        for raw in all_signals:
            try:
                sig = self.pipeline.normaliser.normalise(raw)
                if sig:
                    sig.user_id = user_id
                    normalized_signals.append(sig)
            except Exception as exc:
                logger.warning("Normalisation error on meeting signal: %s", exc)

        # Stage 4: Filtering noise
        _notify("Filtering noise", "Applying heuristic zero-token noise filter...")
        actionable_signals: List[ProjectSignal] = []
        noise_count = 0
        for sig in normalized_signals:
            self.pipeline.noise_filter.apply_filter(sig)
            self.db.save_raw_signal(sig, overwrite=False, user_id=user_id)
            if sig.is_noise:
                noise_count += 1
            else:
                actionable_signals.append(sig)

        # Stage 5: Extracting RAID items
        _notify("Extracting RAID items", f"Running classification on {len(actionable_signals)} actionable discussion point(s)...")
        new_drafts: List[Dict[str, Any]] = []
        classified_assumptions: List[Dict[str, Any]] = []
        effective_threshold = self.pipeline.classifier.confidence_threshold

        for sig in actionable_signals:
            try:
                self.db.update_signal_classification_status(
                    sig.id,
                    status="PROCESSING",
                    user_id=user_id,
                    increment_attempts=True
                )
                outcome = self.pipeline.classifier.classify_signal_detailed(sig)

                if outcome.status.value == "FAILED":
                    err_cat = outcome.error_category.value if outcome.error_category else None
                    self.db.update_signal_classification_status(
                        sig.id,
                        status="FAILED",
                        failure_reason=outcome.error_message,
                        error_category=err_cat,
                        user_id=user_id,
                        increment_attempts=False
                    )
                    logger.warning("Meeting signal %s classification failed: %s (%s)", sig.id, outcome.error_message, err_cat)
                    continue

                if outcome.status.value == "NOT_RAID":
                    self.db.update_signal_classification_status(
                        sig.id,
                        status="NOT_RAID",
                        user_id=user_id,
                        increment_attempts=False
                    )
                    continue

                if outcome.status.value == "BELOW_CONFIDENCE_THRESHOLD":
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
                        increment_attempts=False
                    )
                    cat_val = extracted.category.value if hasattr(extracted.category, "value") else str(extracted.category)
                    # Policy: Dependency is treated as a valid actionable review category
                    if cat_val in ("Risk", "Assumption", "Issue", "Decision", "Dependency"):
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
                            "llm_provider": outcome.llm_provider or getattr(self.pipeline.classifier.provider, "provider_name", "unknown"),
                            "llm_model": outcome.llm_model or getattr(self.pipeline.classifier.provider, "model_name", "unknown")
                        }
                        new_drafts.append(draft_record)
                        if cat_val == "Assumption":
                            classified_assumptions.append(draft_record)
            except Exception as exc:
                logger.error("Error classifying meeting signal %s: %s", sig.id, exc)
                self.db.update_signal_classification_status(
                    sig.id,
                    status="FAILED",
                    failure_reason=str(exc),
                    error_category="PROCESSING_ERROR",
                    user_id=user_id,
                    increment_attempts=False
                )

        # Predictive Foresight
        if run_predictive_engine:
            try:
                existing_assumptions = self.db.get_assumptions(user_id=user_id)
                all_assumptions = existing_assumptions + classified_assumptions
                predictions = self.pipeline.predictive_engine.run_foresight(
                    signals=actionable_signals,
                    existing_assumptions=all_assumptions
                )
                for pred in predictions:
                    self.db.save_prediction(pred, user_id=user_id)
                    if pred.suggested_raid_category in ("Risk", "Issue"):
                        pred_record = {
                            "id": f"raid-{uuid.uuid4().hex[:8]}",
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
                            "evidence_source_id": pred.source_signal_ids[0] if pred.source_signal_ids else "",
                            "evidence_source_type": "predictive_engine",
                            "detection_source": "predictive_engine",
                            "duplicate_of_id": None,
                            "status": "Draft",
                            "source_file": "predictive_engine",
                            "meeting_date": datetime.now(timezone.utc).isoformat(),
                            "evidence_status": "verified",
                            "llm_provider": "rule_based_foresight",
                            "llm_model": "predictive_engine"
                        }
                        new_drafts.append(pred_record)
            except Exception as exc:
                logger.warning("Predictive foresight evaluation encountered non-fatal error: %s", exc)

        # Stage 6: Checking duplicates
        _notify("Checking duplicates", "Checking semantic similarity against existing register...")
        from src.deduplication.schemas import MatchStatus
        all_existing = self.db.get_pending_drafts(user_id=user_id) + self.db.get_approved_items(user_id=user_id)
        duplicates_count = 0

        for draft in new_drafts:
            try:
                dedupe_result = self.pipeline.deduplicator.check_duplicate(draft, all_existing)
                if dedupe_result.match_status == MatchStatus.LIKELY_DUPLICATE:
                    draft["duplicate_of_id"] = dedupe_result.matched_raid_id
                    duplicates_count += 1
                if dedupe_result.matched_raid_id:
                    self.db.save_deduplication_match(dedupe_result, user_id=user_id)
            except Exception as exc:
                logger.warning("Deduplication check error for draft %s: %s", draft.get("id"), exc)

        # Stage 7: Saving results
        _notify("Saving results", f"Persisting {len(new_drafts)} draft(s) into database...")
        saved_drafts: List[Dict[str, Any]] = []

        for draft in new_drafts:
            # Enforce Draft status: never automatically approved
            draft["status"] = "Draft"
            try:
                if not self.db.draft_exists_for_signal(draft["evidence_source_id"], draft["type"], draft["title"], user_id=user_id):
                    self.db.save_raid_draft(draft, user_id=user_id)
                    saved_drafts.append(draft)
                else:
                    existing = self.db.get_raid_draft_by_id(draft["id"], user_id=user_id) or draft
                    saved_drafts.append(existing)
            except Exception as exc:
                logger.error("Failed to persist RAID draft %s: %s", draft.get("id"), exc)

        # Stage 8: Completed / Failed
        _notify("Completed", f"Processing complete: {len(saved_drafts)} draft(s) created in review queue.")

        return {
            "status": "success",
            "total_signals": len(all_signals),
            "noise_filtered": noise_count,
            "normalized_count": len(normalized_signals),
            "drafts_created": len(saved_drafts),
            "duplicates_detected": duplicates_count,
            "drafts": saved_drafts,
            "skipped_sources": skipped_sources,
            "errors": errors
        }

    def retry_failed_signals(
        self,
        user_id: str = "default_user",
        signal_ids: Optional[List[str]] = None,
        include_terminal: bool = False,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retries classification for failed signals via the pipeline.
        """
        return self.pipeline.retry_failed_signals(
            user_id=user_id,
            signal_ids=signal_ids,
            include_terminal=include_terminal,
            max_retries=max_retries
        )

    def get_failed_signals(self, user_id: str = "default_user", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves raw signals whose classification failed.
        """
        return self.db.get_failed_signals(user_id=user_id, limit=limit)

    # -----------------------------------------------------------------------
    # Low-Confidence RAID Review Actions (Stage 6C)
    # -----------------------------------------------------------------------

    def confirm_low_confidence_candidate(
        self,
        signal_id: str,
        reviewer_name: str,
        updates: Optional[Dict[str, Any]] = None,
        comments: Optional[str] = None,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Explicitly confirms a low-confidence candidate as a RAID draft item.
        Preserves original signal, provenance, and original model confidence.
        Creates a 'Draft' record in raid_drafts (never automatically approved).
        Records human confirmation in the audit trail.
        """
        if not reviewer_name or not str(reviewer_name).strip():
            raise RaidValidationError("Reviewer Identity is required to confirm a low-confidence candidate.")

        eff_user_id = user_id or "default_user"
        signal = self.db.get_raw_signal_by_id(signal_id, user_id=eff_user_id)
        if not signal:
            raise RaidNotFoundError(f"Signal '{signal_id}' not found for user '{eff_user_id}'.")

        # Duplicate draft prevention
        if self.db.has_draft_for_signal(signal_id, user_id=eff_user_id):
            raise DuplicateApprovalError(f"Candidate signal '{signal_id}' has already been confirmed as a RAID draft.")

        # Parse model's original extraction payload if present
        extracted: Dict[str, Any] = {}
        raw_reason = signal.get("classification_failure_reason") or ""
        if raw_reason.startswith("{") and raw_reason.endswith("}"):
            try:
                parsed = json.loads(raw_reason)
                if isinstance(parsed, dict):
                    extracted = parsed
            except Exception:
                extracted = {}

        clean_updates = dict(updates or {})

        orig_conf = extracted.get("confidence")
        try:
            orig_conf_float = float(orig_conf) if orig_conf is not None else 0.0
        except (ValueError, TypeError):
            orig_conf_float = 0.0

        raid_type = clean_updates.get("type") or extracted.get("category") or "Risk"
        if raid_type not in VALID_RAID_TYPES:
            raise RaidValidationError(f"Invalid RAID type '{raid_type}'. Must be one of {sorted(list(VALID_RAID_TYPES))}.")

        title = clean_updates.get("title") or extracted.get("title") or signal.get("subject") or "Confirmed RAID Item"
        description = clean_updates.get("description") or extracted.get("description") or signal.get("content") or ""

        if not str(title).strip():
            raise RaidValidationError("RAID draft title cannot be empty.")
        if not str(description).strip():
            raise RaidValidationError("RAID draft description cannot be empty.")

        workstream = clean_updates.get("workstream") or extracted.get("workstream") or signal.get("workstream") or "General"
        phase = clean_updates.get("phase") or extracted.get("phase") or signal.get("phase") or "Realize"

        # Owner attribution: preserve explicit owner or None (never force author)
        owner = clean_updates.get("likely_owner") if "likely_owner" in clean_updates else extracted.get("owner")
        due_date = clean_updates.get("due_date", "") if "due_date" in clean_updates else (extracted.get("due_date") or "")
        mitigation = clean_updates.get("mitigation_action", "") if "mitigation_action" in clean_updates else (extracted.get("mitigation_action") or "")

        probability = clean_updates.get("probability") if "probability" in clean_updates else extracted.get("probability")
        impact = clean_updates.get("impact") if "impact" in clean_updates else extracted.get("impact")
        severity = clean_updates.get("severity") if "severity" in clean_updates else extracted.get("severity")

        reasoning = extracted.get("reasoning_summary")
        scoring_reason = clean_updates.get("scoring_reason") or reasoning or f"Confirmed by {reviewer_name.strip()} (original confidence {orig_conf_float:.2f})"

        draft_id = f"raid-{uuid.uuid4().hex[:8]}"
        evidence_text = clean_updates.get("evidence_text") or extracted.get("evidence") or signal.get("content") or ""
        evidence_status_str = clean_updates.get("evidence_status") or extracted.get("evidence_status") or "verified"

        draft_record = {
            "id": draft_id,
            "user_id": eff_user_id,
            "type": raid_type,
            "title": str(title).strip(),
            "description": str(description).strip(),
            "workstream": workstream,
            "phase": phase,
            "probability": probability,
            "impact": impact,
            "severity": severity,
            "scoring_reason": scoring_reason,
            "likely_owner": owner,
            "due_date": due_date,
            "mitigation_action": mitigation,
            "confidence_score": orig_conf_float,
            "evidence_text": evidence_text,
            "evidence_source_id": signal_id,
            "evidence_source_type": signal.get("source_type") or "signal",
            "detection_source": "low_confidence_human_confirmed",
            "duplicate_of_id": None,
            "status": "Draft",
            "evidence_status": evidence_status_str,
            "llm_provider": signal.get("llm_provider") or "unknown",
            "llm_model": signal.get("llm_model") or "unknown"
        }

        # Save to raid_drafts (status=Draft, NEVER approved)
        self.db.save_raid_draft(draft_record, user_id=eff_user_id)

        # Update signal status to CONFIRMED_RAID
        self.db.update_signal_classification_status(
            signal_id=signal_id,
            status="CONFIRMED_RAID",
            user_id=eff_user_id,
            increment_attempts=False
        )

        # Record audit history entry
        now_utc = datetime.now(timezone.utc).isoformat()
        audit_entry = {
            "item_id": draft_id,
            "item_table": "raid_drafts",
            "action": "CONFIRMED_LOW_CONFIDENCE",
            "proposed_data": json.dumps({"source_signal_id": signal_id, "original_confidence": orig_conf_float}),
            "final_data": json.dumps(draft_record),
            "action_by": reviewer_name.strip(),
            "action_notes": f"Confirmed low-confidence signal as {raid_type} Draft. {comments or ''}".strip(),
            "user_id": eff_user_id,
            "timestamp": now_utc
        }
        self.db.insert_audit_entry(audit_entry, user_id=eff_user_id)

        logger.info(
            "Reviewer %s confirmed low-confidence signal %s as RAID draft %s (%s)",
            reviewer_name, signal_id, draft_id, raid_type
        )
        return draft_record

    def reject_low_confidence_candidate(
        self,
        signal_id: str,
        reviewer_name: str,
        rejection_reason: str = "Marked as NOT_RAID by reviewer",
        user_id: str = "default_user"
    ) -> bool:
        """
        Marks a low-confidence signal as NOT_RAID.
        Preserves original signal, does not create a draft, and logs human decision in audit history.
        """
        if not reviewer_name or not str(reviewer_name).strip():
            raise RaidValidationError("Reviewer Identity is required to mark a candidate as NOT_RAID.")

        eff_user_id = user_id or "default_user"
        signal = self.db.get_raw_signal_by_id(signal_id, user_id=eff_user_id)
        if not signal:
            raise RaidNotFoundError(f"Signal '{signal_id}' not found for user '{eff_user_id}'.")

        # Update classification status
        self.db.update_signal_classification_status(
            signal_id=signal_id,
            status="NOT_RAID",
            user_id=eff_user_id,
            increment_attempts=False
        )

        now_utc = datetime.now(timezone.utc).isoformat()
        audit_entry = {
            "item_id": signal_id,
            "item_table": "raw_signals",
            "action": "REJECTED_LOW_CONFIDENCE",
            "proposed_data": json.dumps({"original_status": signal.get("classification_status")}),
            "final_data": json.dumps({"classification_status": "NOT_RAID"}),
            "action_by": reviewer_name.strip(),
            "action_notes": str(rejection_reason).strip(),
            "user_id": eff_user_id,
            "timestamp": now_utc
        }
        self.db.insert_audit_entry(audit_entry, user_id=eff_user_id)

        logger.info("Reviewer %s marked low-confidence signal %s as NOT_RAID", reviewer_name, signal_id)
        return True

    def defer_low_confidence_candidate(
        self,
        signal_id: str,
        reviewer_name: str,
        defer_notes: Optional[str] = None,
        deferral_notes: Optional[str] = None,
        user_id: str = "default_user"
    ) -> bool:
        """
        Defers a low-confidence signal for subsequent review.
        Preserves original signal, keeps candidate available in the review queue, and logs decision.
        """
        if not reviewer_name or not str(reviewer_name).strip():
            raise RaidValidationError("Reviewer Identity is required to defer a candidate.")

        effective_notes = deferral_notes if deferral_notes is not None else (defer_notes or "Deferred for manual review")

        eff_user_id = user_id or "default_user"
        signal = self.db.get_raw_signal_by_id(signal_id, user_id=eff_user_id)
        if not signal:
            raise RaidNotFoundError(f"Signal '{signal_id}' not found for user '{eff_user_id}'.")

        self.db.update_signal_classification_status(
            signal_id=signal_id,
            status="DEFERRED",
            user_id=eff_user_id,
            increment_attempts=False
        )

        now_utc = datetime.now(timezone.utc).isoformat()
        audit_entry = {
            "item_id": signal_id,
            "item_table": "raw_signals",
            "action": "DEFERRED_LOW_CONFIDENCE",
            "proposed_data": json.dumps({"original_status": signal.get("classification_status")}),
            "final_data": json.dumps({"classification_status": "DEFERRED"}),
            "action_by": reviewer_name.strip(),
            "action_notes": str(effective_notes).strip(),
            "user_id": eff_user_id,
            "timestamp": now_utc
        }
        self.db.insert_audit_entry(audit_entry, user_id=eff_user_id)

        logger.info("Reviewer %s deferred low-confidence signal %s for manual review", reviewer_name, signal_id)
        return True




