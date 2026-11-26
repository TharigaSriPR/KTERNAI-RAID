"""
Database Manager module for SQLite storage.
Handles persistence for raw signals, RAID drafts, approved items, audit history,
predictive findings, and deduplication match audit logs.
"""

import sqlite3
import json
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from config import DB_PATH
from src.database.models import (
    CREATE_RAW_SIGNALS_TABLE,
    CREATE_RAID_DRAFTS_TABLE,
    CREATE_APPROVED_RAID_TABLE,
    CREATE_AUDIT_HISTORY_TABLE,
    CREATE_PREDICTIONS_TABLE,
    CREATE_DEDUPLICATION_MATCHES_TABLE,
    CREATE_USERS_TABLE,
    CREATE_USER_SESSIONS_TABLE,
    CREATE_LOGIN_ATTEMPTS_TABLE,
    CREATE_USER_GMAIL_TOKENS_TABLE,
    CREATE_OAUTH_STATES_TABLE,
    CREATE_GMAIL_SYNC_CHECKPOINTS_TABLE,
    CREATE_GMAIL_SYNC_HISTORY_TABLE,
    CREATE_SOURCE_FILE_SYNC_TABLE
)
from src.normalisation.schemas import ProjectSignal
from src.predictive_engine.schemas import PredictionItem
from src.deduplication.schemas import DeduplicationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        from src.database.migrations import run_migrations
        run_migrations(self.db_path)

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initializes all database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(CREATE_RAW_SIGNALS_TABLE)
            cursor.execute(CREATE_RAID_DRAFTS_TABLE)
            cursor.execute(CREATE_APPROVED_RAID_TABLE)
            cursor.execute(CREATE_AUDIT_HISTORY_TABLE)
            cursor.executescript(CREATE_PREDICTIONS_TABLE)
            cursor.executescript(CREATE_DEDUPLICATION_MATCHES_TABLE)
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_approved_raid_draft_id ON approved_raid(draft_id);")
            cursor.executescript(CREATE_USERS_TABLE)
            cursor.executescript(CREATE_USER_SESSIONS_TABLE)
            cursor.executescript(CREATE_LOGIN_ATTEMPTS_TABLE)
            cursor.executescript(CREATE_USER_GMAIL_TOKENS_TABLE)
            cursor.executescript(CREATE_OAUTH_STATES_TABLE)
            cursor.executescript(CREATE_GMAIL_SYNC_CHECKPOINTS_TABLE)
            cursor.executescript(CREATE_GMAIL_SYNC_HISTORY_TABLE)
            cursor.executescript(CREATE_SOURCE_FILE_SYNC_TABLE)
            conn.commit()

    def signal_exists(self, source_type: str, source_id: str, user_id: str = "default_user") -> bool:
        """Checks if a signal with the same source_type, source_id, and user_id has already been stored."""
        if not source_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM raw_signals WHERE source_type = ? AND source_id = ? AND user_id = ? LIMIT 1",
                (source_type, source_id, user_id)
            )
            return cursor.fetchone() is not None

    def get_signal_id_by_source(self, source_type: str, source_id: str, user_id: str = "default_user") -> Optional[str]:
        """Returns the ID of an already stored signal matching source_type, source_id, and user_id."""
        if not source_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM raw_signals WHERE source_type = ? AND source_id = ? AND user_id = ? LIMIT 1",
                (source_type, source_id, user_id)
            )
            row = cursor.fetchone()
            return row["id"] if row else None

    def save_raw_signal(
        self, 
        signal: Union[Dict[str, Any], ProjectSignal], 
        overwrite: bool = False,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Persists a normalized raw project signal.
        If overwrite is False, skips signals where (user_id, source_type, source_id) already exists,
        preventing accidental duplicate ingestion.
        Returns True if inserted/updated, False if skipped.
        """
        data = signal.model_dump() if isinstance(signal, ProjectSignal) else dict(signal)
        eff_user_id = user_id or data.get("user_id") or "default_user"
        data["user_id"] = eff_user_id

        source_type = data.get("source_type")
        source_id = data.get("source_id")

        if not overwrite and source_id:
            existing_id = self.get_signal_id_by_source(source_type, source_id, user_id=eff_user_id)
            if existing_id:
                logger.debug("Duplicate signal skipped: user_id='%s', source_type='%s', source_id='%s'", eff_user_id, source_type, source_id)
                if isinstance(signal, ProjectSignal):
                    object.__setattr__(signal, "id", existing_id)
                elif isinstance(signal, dict):
                    signal["id"] = existing_id
                return False

        is_noise_val = int(data.get("is_noise", 0))
        default_status = "NOT_RAID" if is_noise_val == 1 else "PENDING"

        full_signal = {
            "id": data.get("id"),
            "user_id": eff_user_id,
            "source_type": data.get("source_type"),
            "source_id": data.get("source_id"),
            "author": data.get("author", "Unknown"),
            "timestamp": data.get("timestamp"),
            "subject": data.get("subject", ""),
            "content": data.get("content", ""),
            "workstream": data.get("workstream", "General"),
            "phase": data.get("phase", "Realize"),
            "is_noise": is_noise_val,
            "noise_reason": data.get("noise_reason"),
            "classification_status": data.get("classification_status") or default_status,
            "classification_failure_reason": data.get("classification_failure_reason"),
            "classification_error_category": data.get("classification_error_category"),
            "classification_attempts": int(data.get("classification_attempts", 0)),
            "last_attempted_at": data.get("last_attempted_at"),
            "last_classified_at": data.get("last_classified_at"),
            "llm_provider": data.get("llm_provider"),
            "llm_model": data.get("llm_model")
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO raw_signals 
                (id, user_id, source_type, source_id, author, timestamp, subject, content, workstream, phase, 
                 is_noise, noise_reason, classification_status, classification_failure_reason, classification_error_category, 
                 classification_attempts, last_attempted_at, last_classified_at, llm_provider, llm_model)
                VALUES 
                (:id, :user_id, :source_type, :source_id, :author, :timestamp, :subject, :content, :workstream, :phase, 
                 :is_noise, :noise_reason, :classification_status, :classification_failure_reason, :classification_error_category, 
                 :classification_attempts, :last_attempted_at, :last_classified_at, :llm_provider, :llm_model)
            """, full_signal)
            conn.commit()
        return True

    def save_raw_signals_batch(
        self,
        signals: List[Union[Dict[str, Any], ProjectSignal]],
        overwrite: bool = False,
        user_id: Optional[str] = None
    ) -> int:
        """
        Persists a batch of raw project signals, returning count of newly inserted records.
        """
        inserted_count = 0
        for sig in signals:
            if self.save_raw_signal(sig, overwrite=overwrite, user_id=user_id):
                inserted_count += 1
        logger.info("Saved %d new raw signals to database (out of %d processed)", inserted_count, len(signals))
        return inserted_count

    def save_raid_draft(self, draft: Dict[str, Any], user_id: Optional[str] = None):
        """Stores a newly generated RAID draft item."""
        draft_data = dict(draft)
        eff_user_id = user_id or draft_data.get("user_id") or "default_user"
        draft_data["user_id"] = eff_user_id

        full_draft = {
            "id": draft_data.get("id"),
            "user_id": eff_user_id,
            "type": draft_data.get("type", "Risk"),
            "title": draft_data.get("title", ""),
            "description": draft_data.get("description", ""),
            "workstream": draft_data.get("workstream", "General"),
            "phase": draft_data.get("phase", "Realize"),
            "probability": draft_data.get("probability"),
            "impact": draft_data.get("impact"),
            "severity": draft_data.get("severity"),
            "scoring_reason": draft_data.get("scoring_reason", ""),
            "likely_owner": draft_data.get("likely_owner"),
            "due_date": draft_data.get("due_date", ""),
            "mitigation_action": draft_data.get("mitigation_action", ""),
            "confidence_score": float(draft_data.get("confidence_score", 0.8)),
            "evidence_text": draft_data.get("evidence_text", ""),
            "evidence_source_id": draft_data.get("evidence_source_id", ""),
            "evidence_source_type": draft_data.get("evidence_source_type", "email"),
            "detection_source": draft_data.get("detection_source", "pipeline"),
            "duplicate_of_id": draft_data.get("duplicate_of_id"),
            "status": draft_data.get("status", "Draft"),
            "evidence_status": draft_data.get("evidence_status", "verified"),
            "llm_provider": draft_data.get("llm_provider") or "unknown",
            "llm_model": draft_data.get("llm_model") or "unknown"
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO raid_drafts 
                (id, user_id, type, title, description, workstream, phase, probability, impact, severity, 
                 scoring_reason, likely_owner, due_date, mitigation_action, confidence_score, 
                 evidence_text, evidence_source_id, evidence_source_type, detection_source, duplicate_of_id, status, evidence_status, llm_provider, llm_model)
                VALUES 
                (:id, :user_id, :type, :title, :description, :workstream, :phase, :probability, :impact, :severity, 
                 :scoring_reason, :likely_owner, :due_date, :mitigation_action, :confidence_score, 
                 :evidence_text, :evidence_source_id, :evidence_source_type, :detection_source, :duplicate_of_id, :status, :evidence_status, :llm_provider, :llm_model)
            """, full_draft)

            # Log audit trail for Agent proposal
            cursor.execute("""
                INSERT INTO audit_history (user_id, item_id, item_table, action, proposed_data, final_data, action_by, action_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eff_user_id,
                full_draft['id'],
                'raid_drafts',
                'PROPOSED',
                json.dumps(full_draft),
                None,
                f"Agent ({full_draft.get('detection_source', 'pipeline')})",
                f"Proposed {full_draft['type']} with confidence {full_draft.get('confidence_score', 0)}"
            ))
            conn.commit()

    def get_pending_drafts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves drafts waiting for PM review."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM raid_drafts WHERE user_id = ? AND status IN ('Draft', 'Pending_Review', 'Modified') ORDER BY confidence_score DESC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_raid_draft_by_id(self, draft_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a single RAID draft by ID, optionally scoped to a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM raid_drafts WHERE id = ? AND user_id = ?", (draft_id, user_id))
            else:
                cursor.execute("SELECT * FROM raid_drafts WHERE id = ?", (draft_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_signals_by_classification_status(
        self,
        status: str,
        user_id: str = "default_user",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieves raw signals filtered by classification status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM raw_signals 
                WHERE user_id = ? AND classification_status = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (user_id, status, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_signal_classification_status(
        self,
        signal_id: str,
        status: str,
        failure_reason: Optional[str] = None,
        error_category: Optional[str] = None,
        user_id: Optional[str] = None,
        increment_attempts: bool = True,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> bool:
        """
        Updates the classification status and error tracking metadata for a raw signal.
        Preserves isolation and records timestamps.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            attempts_expr = "classification_attempts + 1" if increment_attempts else "classification_attempts"

            # If terminal completed status, set last_classified_at
            if status in ("COMPLETED", "NOT_RAID", "BELOW_CONFIDENCE_THRESHOLD"):
                sql = f"""
                    UPDATE raw_signals
                    SET classification_status = ?,
                        classification_failure_reason = ?,
                        classification_error_category = ?,
                        classification_attempts = {attempts_expr},
                        last_attempted_at = ?,
                        last_classified_at = ?,
                        llm_provider = COALESCE(?, llm_provider),
                        llm_model = COALESCE(?, llm_model)
                    WHERE id = ? { "AND user_id = ?" if user_id else "" }
                """
                params = [status, failure_reason, error_category, now, now, llm_provider, llm_model, signal_id]
            else:
                sql = f"""
                    UPDATE raw_signals
                    SET classification_status = ?,
                        classification_failure_reason = ?,
                        classification_error_category = ?,
                        classification_attempts = {attempts_expr},
                        last_attempted_at = ?,
                        llm_provider = COALESCE(?, llm_provider),
                        llm_model = COALESCE(?, llm_model)
                    WHERE id = ? { "AND user_id = ?" if user_id else "" }
                """
                params = [status, failure_reason, error_category, now, llm_provider, llm_model, signal_id]

            if user_id:
                params.append(user_id)

            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0

    def get_failed_signals(self, user_id: str = "default_user", limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves raw signals whose classification failed for inspection and recovery."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM raw_signals 
                WHERE user_id = ? AND classification_status = 'FAILED'
                ORDER BY last_attempted_at DESC LIMIT ?
                """,
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]


    def get_approved_item_by_id(self, item_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves an approved RAID record by ID, optionally scoped to a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM approved_raid WHERE id = ? AND user_id = ?", (item_id, user_id))
            else:
                cursor.execute("SELECT * FROM approved_raid WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def get_approved_item_by_draft_id(self, draft_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves an approved RAID record by originating draft ID, optionally scoped to a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM approved_raid WHERE draft_id = ? AND user_id = ?", (draft_id, user_id))
            else:
                cursor.execute("SELECT * FROM approved_raid WHERE draft_id = ?", (draft_id,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def get_rejected_drafts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves rejected drafts for historical review."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM raid_drafts WHERE user_id = ? AND status = 'Rejected' ORDER BY detected_at DESC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_drafts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves all RAID drafts across all lifecycle statuses for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM raid_drafts WHERE user_id = ? ORDER BY detected_at DESC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_kpis(self, user_id: str = "default_user") -> Dict[str, int]:
        """Calculates KPI summary counts directly from SQLite tables scoped to a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raid_drafts WHERE user_id = ?", (user_id,))
            total_drafts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM raid_drafts WHERE user_id = ? AND status IN ('Draft', 'Pending_Review', 'Modified')", (user_id,))
            pending_reviews = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM approved_raid WHERE user_id = ?", (user_id,))
            approved_items = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM raid_drafts WHERE user_id = ? AND status = 'Rejected'", (user_id,))
            rejected_drafts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,))
            active_predictions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM deduplication_matches WHERE user_id = ?", (user_id,))
            potential_duplicates = cursor.fetchone()[0]

            return {
                "total_drafts": total_drafts,
                "pending_reviews": pending_reviews,
                "approved_items": approved_items,
                "rejected_drafts": rejected_drafts,
                "active_predictions": active_predictions,
                "potential_duplicates": potential_duplicates
            }

    def execute_approval_transaction(
        self,
        draft_id: str,
        approved_record: Dict[str, Any],
        audit_record: Dict[str, Any],
        approved_status: str = "Accepted",
        user_id: Optional[str] = None
    ) -> bool:
        """
        Atomically executes approval:
        1. Updates draft status to approved_status ('Accepted' / 'Approved').
        2. Inserts new record into approved_raid.
        3. Appends an entry into audit_history.
        Rolls back all operations if any step fails.
        """
        appr_data = dict(approved_record)
        eff_user_id = user_id or appr_data.get("user_id") or "default_user"
        appr_data["user_id"] = eff_user_id

        full_approved = {
            "id": appr_data.get("id"),
            "user_id": eff_user_id,
            "draft_id": appr_data.get("draft_id"),
            "type": appr_data.get("type", "Risk"),
            "title": appr_data.get("title", ""),
            "description": appr_data.get("description", ""),
            "workstream": appr_data.get("workstream", "General"),
            "phase": appr_data.get("phase", "Realize"),
            "severity_or_impact": appr_data.get("severity_or_impact", "Medium"),
            "owner": appr_data.get("owner"),
            "due_date": appr_data.get("due_date", ""),
            "mitigation_action": appr_data.get("mitigation_action", ""),
            "status": appr_data.get("status", "Open"),
            "evidence_links": appr_data.get("evidence_links", ""),
            "approved_by": appr_data.get("approved_by", "PM"),
            "approved_at": appr_data.get("approved_at"),
            "last_updated_at": appr_data.get("last_updated_at"),
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Update draft status (ensures draft is pending, belongs to user, and not already approved)
            if user_id:
                cursor.execute(
                    "UPDATE raid_drafts SET status = ? WHERE id = ? AND user_id = ? AND status NOT IN ('Accepted', 'Approved')",
                    (approved_status, draft_id, eff_user_id)
                )
            else:
                cursor.execute(
                    "UPDATE raid_drafts SET status = ? WHERE id = ? AND status NOT IN ('Accepted', 'Approved')",
                    (approved_status, draft_id)
                )
            if cursor.rowcount == 0:
                raise ValueError(f"Draft '{draft_id}' not found or already approved.")

            # 2. Insert into approved_raid
            cursor.execute("""
                INSERT INTO approved_raid 
                (id, user_id, draft_id, type, title, description, workstream, phase, severity_or_impact, 
                 owner, due_date, mitigation_action, status, evidence_links, approved_by, approved_at, last_updated_at)
                VALUES 
                (:id, :user_id, :draft_id, :type, :title, :description, :workstream, :phase, :severity_or_impact, 
                 :owner, :due_date, :mitigation_action, :status, :evidence_links, :approved_by, :approved_at, :last_updated_at)
            """, full_approved)

            # 3. Insert into audit_history
            cursor.execute("""
                INSERT INTO audit_history (user_id, item_id, item_table, action, proposed_data, final_data, action_by, action_notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eff_user_id,
                audit_record["item_id"],
                audit_record.get("item_table", "raid_drafts"),
                audit_record["action"],
                audit_record.get("proposed_data"),
                audit_record.get("final_data"),
                audit_record["action_by"],
                audit_record.get("action_notes"),
                audit_record.get("timestamp")
            ))
            conn.commit()
            return True

    def execute_rejection_transaction(
        self,
        draft_id: str,
        rejection_reason: str,
        audit_record: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Atomically executes rejection:
        1. Updates draft status to 'Rejected' and records reason.
        2. Appends an entry into audit_history.
        Rolls back all operations if any step fails.
        """
        eff_user_id = user_id or audit_record.get("user_id") or "default_user"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "UPDATE raid_drafts SET status = 'Rejected', scoring_reason = ? WHERE id = ? AND user_id = ?",
                    (f"Rejected: {rejection_reason}", draft_id, eff_user_id)
                )
            else:
                cursor.execute(
                    "UPDATE raid_drafts SET status = 'Rejected', scoring_reason = ? WHERE id = ?",
                    (f"Rejected: {rejection_reason}", draft_id)
                )
            if cursor.rowcount == 0:
                raise ValueError(f"Draft '{draft_id}' not found for rejection update.")

            cursor.execute("""
                INSERT INTO audit_history (user_id, item_id, item_table, action, proposed_data, final_data, action_by, action_notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eff_user_id,
                audit_record["item_id"],
                audit_record.get("item_table", "raid_drafts"),
                audit_record["action"],
                audit_record.get("proposed_data"),
                audit_record.get("final_data"),
                audit_record["action_by"],
                audit_record.get("action_notes"),
                audit_record.get("timestamp")
            ))
            conn.commit()
            return True

    def execute_modification_transaction(
        self,
        draft_id: str,
        updates: Dict[str, Any],
        audit_record: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Atomically updates only the specified fields in raid_drafts
        and records the before/after state in audit_history.
        """
        if not updates:
            return False

        eff_user_id = user_id or audit_record.get("user_id") or "default_user"

        set_clauses = [f"{k} = :{k}" for k in updates.keys()]
        params = dict(updates)
        params["_target_id"] = draft_id

        if user_id:
            query = f"UPDATE raid_drafts SET {', '.join(set_clauses)} WHERE id = :_target_id AND user_id = :_user_id"
            params["_user_id"] = eff_user_id
        else:
            query = f"UPDATE raid_drafts SET {', '.join(set_clauses)} WHERE id = :_target_id"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if cursor.rowcount == 0:
                raise ValueError(f"Draft '{draft_id}' not found for modification update.")

            cursor.execute("""
                INSERT INTO audit_history (user_id, item_id, item_table, action, proposed_data, final_data, action_by, action_notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eff_user_id,
                audit_record["item_id"],
                audit_record.get("item_table", "raid_drafts"),
                audit_record["action"],
                audit_record.get("proposed_data"),
                audit_record.get("final_data"),
                audit_record["action_by"],
                audit_record.get("action_notes"),
                audit_record.get("timestamp")
            ))
            conn.commit()
            return True

    def insert_audit_entry(self, audit_record: Dict[str, Any], user_id: Optional[str] = None) -> int:
        """Appends an audit entry to the audit_history table."""
        eff_user_id = user_id or audit_record.get("user_id") or "default_user"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_history (user_id, item_id, item_table, action, proposed_data, final_data, action_by, action_notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eff_user_id,
                audit_record["item_id"],
                audit_record.get("item_table", "raid_drafts"),
                audit_record["action"],
                audit_record.get("proposed_data"),
                audit_record.get("final_data"),
                audit_record["action_by"],
                audit_record.get("action_notes"),
                audit_record.get("timestamp")
            ))
            conn.commit()
            return cursor.lastrowid

    def get_approved_items(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves items from the official approved RAID register for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM approved_raid WHERE user_id = ? ORDER BY approved_at DESC", (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_assumptions(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves recorded assumptions from both drafts and approved items for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, title, description, workstream, phase, evidence_text, evidence_source_id
                FROM raid_drafts
                WHERE type = 'Assumption' AND user_id = ?
                UNION ALL
                SELECT id, type, title, description, workstream, phase, evidence_links AS evidence_text, draft_id AS evidence_source_id
                FROM approved_raid
                WHERE type = 'Assumption' AND user_id = ?
            """, (user_id, user_id))
            return [dict(row) for row in cursor.fetchall()]

    def get_audit_trail(self, item_id: Optional[str] = None, user_id: Optional[str] = "default_user") -> List[Dict[str, Any]]:
        """Retrieves the complete audit trail or history, scoped to user and optionally item ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if item_id and user_id:
                cursor.execute(
                    "SELECT * FROM audit_history WHERE item_id = ? AND user_id = ? ORDER BY timestamp DESC",
                    (item_id, user_id)
                )
            elif user_id:
                cursor.execute(
                    "SELECT * FROM audit_history WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id,)
                )
            elif item_id:
                cursor.execute(
                    "SELECT * FROM audit_history WHERE item_id = ? ORDER BY timestamp DESC",
                    (item_id,)
                )
            else:
                cursor.execute("SELECT * FROM audit_history ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    # Alias for get_audit_trail
    get_audit_history = get_audit_trail

    # Predictive Engine Database Helpers
    def save_prediction(self, prediction: Union[Dict[str, Any], PredictionItem], user_id: Optional[str] = None) -> bool:
        """
        Stores a predictive finding. Avoids duplicating already stored predictions.
        Returns True if inserted, False if already exists.
        """
        data = prediction.model_dump() if isinstance(prediction, PredictionItem) else dict(prediction)
        pred_id = str(data.get("prediction_id") or data.get("id"))
        eff_user_id = user_id or data.get("user_id") or "default_user"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM predictions WHERE id = ? AND user_id = ? LIMIT 1", (pred_id, eff_user_id))
            if cursor.fetchone() is not None:
                return False

            source_signals_json = json.dumps(data.get("source_signal_ids", []))
            related_raids_json = json.dumps(data.get("related_raid_ids", []))
            metadata_json = json.dumps(data.get("metadata", {}))

            cursor.execute("""
                INSERT INTO predictions 
                (id, user_id, prediction_type, title, description, suggested_raid_category, confidence, 
                 severity, evidence, source_signal_ids, related_raid_ids, workstream, phase, 
                 status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred_id,
                eff_user_id,
                str(data.get("prediction_type")),
                data.get("title", ""),
                data.get("description", ""),
                data.get("suggested_raid_category", "Risk"),
                float(data.get("confidence", 0.8)),
                data.get("severity", "Medium"),
                data.get("evidence", ""),
                source_signals_json,
                related_raids_json,
                data.get("workstream", "General"),
                data.get("phase", "Realize"),
                str(data.get("status", "DETECTED")),
                metadata_json
            ))
            conn.commit()
            return True

    def get_predictions(self, status: Optional[str] = None, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves stored predictions for a user, optionally filtered by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM predictions WHERE user_id = ? AND status = ? ORDER BY confidence DESC",
                    (user_id, status)
                )
            else:
                cursor.execute(
                    "SELECT * FROM predictions WHERE user_id = ? ORDER BY confidence DESC",
                    (user_id,)
                )
            rows = [dict(r) for r in cursor.fetchall()]
            for r in rows:
                try:
                    r["source_signal_ids"] = json.loads(r["source_signal_ids"])
                except Exception:
                    pass
                try:
                    r["related_raid_ids"] = json.loads(r["related_raid_ids"]) if r.get("related_raid_ids") else []
                except Exception:
                    pass
                try:
                    r["metadata"] = json.loads(r["metadata"]) if r.get("metadata") else {}
                except Exception:
                    pass
            return rows

    # Deduplication Database Helpers
    def save_deduplication_match(self, match: Union[Dict[str, Any], DeduplicationResult], user_id: Optional[str] = None) -> bool:
        """Stores a deduplication match finding for audit and PM review."""
        data = match.model_dump() if isinstance(match, DeduplicationResult) else dict(match)
        new_raid_id = data.get("new_raid_id")
        matched_raid_id = data.get("matched_raid_id")
        eff_user_id = user_id or data.get("user_id") or "default_user"

        if not new_raid_id or not matched_raid_id:
            return False

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Prevent duplicate match logging for the exact pair under the same user
            cursor.execute(
                "SELECT 1 FROM deduplication_matches WHERE user_id = ? AND new_raid_id = ? AND matched_raid_id = ? LIMIT 1",
                (eff_user_id, new_raid_id, matched_raid_id)
            )
            if cursor.fetchone() is not None:
                return False

            cursor.execute("""
                INSERT INTO deduplication_matches
                (user_id, new_raid_id, matched_raid_id, similarity_score, match_status, match_reason, 
                 same_category, same_workstream, requires_human_review)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eff_user_id,
                new_raid_id,
                matched_raid_id,
                float(data.get("similarity_score", 0.0)),
                str(data.get("match_status")),
                data.get("match_reason", ""),
                1 if data.get("same_category") else 0,
                1 if data.get("same_workstream") else 0,
                1 if data.get("requires_human_review") else 0
            ))
            conn.commit()
            return True

    def get_deduplication_matches(self, new_raid_id: Optional[str] = None, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieves deduplication match records for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if new_raid_id:
                cursor.execute(
                    "SELECT * FROM deduplication_matches WHERE user_id = ? AND new_raid_id = ? ORDER BY similarity_score DESC",
                    (user_id, new_raid_id)
                )
            else:
                cursor.execute(
                    "SELECT * FROM deduplication_matches WHERE user_id = ? ORDER BY similarity_score DESC",
                    (user_id,)
                )
            return [dict(r) for r in cursor.fetchall()]

    def is_source_already_ingested(self, source_name_or_id: str, source_type: str = "meeting_notes", user_id: str = "default_user") -> bool:
        """
        Checks if signals originating from this source file or source ID already exist in raw_signals for this user.
        Prevents duplicate ingestion of the same source file.
        """
        if not source_name_or_id or not str(source_name_or_id).strip():
            return False
        clean_name = str(source_name_or_id).strip()
        stem = Path(clean_name).stem
        name = Path(clean_name).name

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM raw_signals 
                WHERE user_id = ? AND source_type = ? 
                  AND (source_id LIKE ? OR source_id LIKE ? OR subject LIKE ? OR id = ?)
                LIMIT 1
            """, (user_id, source_type, f"{name}%", f"{stem}%", f"{stem}%", clean_name))
            return cursor.fetchone() is not None

    def get_raw_signals(self, user_id: str = "default_user", limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves raw signals belonging to a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM raw_signals WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_raw_signal_by_id(self, signal_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a single raw signal record by ID, optionally scoped to a user."""
        if not signal_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM raw_signals WHERE id = ? AND user_id = ?", (signal_id, user_id))
            else:
                cursor.execute("SELECT * FROM raw_signals WHERE id = ?", (signal_id,))
            row = cursor.fetchone()
            return dict(row) if row else None








    def get_drafts_by_source_file(self, source_name: str, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """
        Retrieves all drafts linked to raw signals originating from a specific source file or name for a user.
        """
        if not source_name:
            return []
        stem = Path(source_name).stem
        name = Path(source_name).name
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT rd.* FROM raid_drafts rd
                JOIN raw_signals rs ON rd.evidence_source_id = rs.id
                WHERE rd.user_id = ? AND rs.user_id = ?
                  AND (rs.source_id LIKE ? OR rs.source_id LIKE ? OR rs.subject LIKE ?)
                ORDER BY rd.confidence_score DESC
            """, (user_id, user_id, f"{name}%", f"{stem}%", f"{stem}%"))
            return [dict(row) for row in cursor.fetchall()]

    def draft_exists_for_signal(self, evidence_source_id: str, raid_type: str, title: str, user_id: Optional[str] = None) -> bool:
        """
        Checks if a draft already exists for this evidence signal and title.
        Ensures repeated processing does not create duplicate RAID drafts unnecessarily.
        """
        if not evidence_source_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "SELECT 1 FROM raid_drafts WHERE user_id = ? AND evidence_source_id = ? AND type = ? AND title = ? LIMIT 1",
                    (user_id, evidence_source_id, raid_type, title)
                )
            else:
                cursor.execute(
                    "SELECT 1 FROM raid_drafts WHERE evidence_source_id = ? AND type = ? AND title = ? LIMIT 1",
                    (evidence_source_id, raid_type, title)
                )
            return cursor.fetchone() is not None

    def has_draft_for_signal(self, evidence_source_id: str, user_id: Optional[str] = None) -> bool:
        """Checks if any draft already exists referencing this evidence signal ID."""
        if not evidence_source_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "SELECT 1 FROM raid_drafts WHERE user_id = ? AND evidence_source_id = ? LIMIT 1",
                    (user_id, evidence_source_id)
                )
            else:
                cursor.execute(
                    "SELECT 1 FROM raid_drafts WHERE evidence_source_id = ? LIMIT 1",
                    (evidence_source_id,)
                )
            return cursor.fetchone() is not None

    def get_low_confidence_signals(
        self,
        user_id: str = "default_user",
        include_deferred: bool = True,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieves raw signals that were marked below confidence threshold or deferred."""
        statuses = ("BELOW_CONFIDENCE_THRESHOLD", "DEFERRED") if include_deferred else ("BELOW_CONFIDENCE_THRESHOLD",)
        placeholders = ",".join("?" for _ in statuses)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM raw_signals 
                WHERE user_id = ? AND classification_status IN ({placeholders})
                ORDER BY timestamp DESC LIMIT ?
                """,
                (user_id, *statuses, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Phase 1: Authentication & User Management CRUD
    # -----------------------------------------------------------------------

    def create_user(self, user_data: Dict[str, Any]) -> str:
        """Inserts a new user record. Raises sqlite3.IntegrityError if username or email exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, username, email, password_hash, salt, algorithm, full_name, role)
                VALUES (:id, :username, :email, :password_hash, :salt, :algorithm, :full_name, :role)
            """, user_data)
            conn.commit()
            return user_data["id"]

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def get_user_by_username_or_email(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieves user by either username or email."""
        clean = identifier.strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ? LIMIT 1",
                (clean, clean)
            )
            row = cursor.fetchone()
            return dict(row) if row else None





    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Stores a new session record using SHA-256 hashed session token."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_sessions (id, user_id, session_token_hash, ip_address, user_agent, expires_at, is_active)
                VALUES (:id, :user_id, :session_token_hash, :ip_address, :user_agent, :expires_at, 1)
            """, session_data)
            conn.commit()
            return session_data["id"]

    def get_session_by_hash(self, session_token_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves active session and associated user if not expired."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, u.username, u.email, u.full_name, u.role
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token_hash = ? AND s.is_active = 1
                LIMIT 1
            """, (session_token_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def update_session_activity(self, session_id: str):
        """Updates last_active_at for a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE user_sessions SET last_active_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            conn.commit()

    def invalidate_session(self, session_token_hash: str) -> bool:
        """Invalidates an active session (logout)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE user_sessions SET is_active = 0 WHERE session_token_hash = ?",
                (session_token_hash,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def rotate_session(
        self,
        old_token_hash: str,
        new_session_id: str,
        new_token_hash: str,
        new_expires_at: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """Atomically invalidates old session and issues new session for the same user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id FROM user_sessions WHERE session_token_hash = ? AND is_active = 1",
                (old_token_hash,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            user_id = row["user_id"]
            # Invalidate old session
            cursor.execute(
                "UPDATE user_sessions SET is_active = 0 WHERE session_token_hash = ?",
                (old_token_hash,)
            )
            # Create new session
            cursor.execute("""
                INSERT INTO user_sessions (id, user_id, session_token_hash, ip_address, user_agent, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (new_session_id, user_id, new_token_hash, ip_address, user_agent, new_expires_at))
            conn.commit()
            return user_id

    def record_failed_login(
        self,
        identifier: str,
        ip_address: str,
        max_attempts: int = 5,
        lockout_minutes: int = 15,
        max_ip_attempts: int = 15,
        ip_lockout_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Increments failed login attempts for (identifier, ip) as well as global IP attempts.
        Provides multi-tier protection against distributed brute-force and credential stuffing.
        """
        clean_id = identifier.strip().lower()
        clean_ip = ip_address.strip()
        from datetime import datetime, timezone, timedelta

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Record specific identifier + IP attempt
            cursor.execute(
                "SELECT * FROM login_attempts WHERE identifier = ? AND ip_address = ?",
                (clean_id, clean_ip)
            )
            row = cursor.fetchone()
            is_locked = False
            locked_until = None

            if row:
                count = row["attempt_count"] + 1
                if count >= max_attempts:
                    locked_until = (datetime.now(timezone.utc) + timedelta(minutes=lockout_minutes)).isoformat()
                    is_locked = True
                cursor.execute("""
                    UPDATE login_attempts
                    SET attempt_count = ?, last_attempt_at = CURRENT_TIMESTAMP, locked_until = ?
                    WHERE id = ?
                """, (count, locked_until, row["id"]))
            else:
                cursor.execute("""
                    INSERT INTO login_attempts (identifier, ip_address, attempt_count, locked_until)
                    VALUES (?, ?, 1, NULL)
                """, (clean_id, clean_ip))
                count = 1

            # 2. Track global IP failures (defend against credential stuffing across multiple usernames)
            cursor.execute(
                "SELECT * FROM login_attempts WHERE identifier = '__IP__' AND ip_address = ?",
                (clean_ip,)
            )
            ip_row = cursor.fetchone()
            if ip_row:
                ip_count = ip_row["attempt_count"] + 1
                ip_locked_until = None
                if ip_count >= max_ip_attempts:
                    ip_locked_until = (datetime.now(timezone.utc) + timedelta(minutes=ip_lockout_minutes)).isoformat()
                    is_locked = True
                    locked_until = locked_until or ip_locked_until
                cursor.execute("""
                    UPDATE login_attempts
                    SET attempt_count = ?, last_attempt_at = CURRENT_TIMESTAMP, locked_until = ?
                    WHERE id = ?
                """, (ip_count, ip_locked_until, ip_row["id"]))
            else:
                cursor.execute("""
                    INSERT INTO login_attempts (identifier, ip_address, attempt_count, locked_until)
                    VALUES ('__IP__', ?, 1, NULL)
                """, (clean_ip,))

            conn.commit()
            return {"attempt_count": count, "locked_until": locked_until, "is_locked": is_locked}

    def get_login_lockout(self, identifier: str, ip_address: str) -> Optional[Dict[str, Any]]:
        """
        Returns lockout status if account/IP pair OR the global IP is currently locked out.
        Mitigates both targeted password guessing and wide-scale credential stuffing.
        """
        clean_id = identifier.strip().lower()
        clean_ip = ip_address.strip()
        from datetime import datetime, timezone

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check either the specific identifier+IP lockout or the global IP lockout
            cursor.execute("""
                SELECT * FROM login_attempts 
                WHERE ( (identifier = ? AND ip_address = ?) OR (identifier = '__IP__' AND ip_address = ?) )
                  AND locked_until IS NOT NULL
                ORDER BY locked_until DESC
                LIMIT 1
            """, (clean_id, clean_ip, clean_ip))
            row = cursor.fetchone()
            if not row or not row["locked_until"]:
                return None
            try:
                lock_dt = datetime.fromisoformat(row["locked_until"].replace("Z", "+00:00"))
                if lock_dt > datetime.now(timezone.utc):
                    return dict(row)
            except Exception:
                pass
            return None

    def reset_failed_logins(self, identifier: str, ip_address: str):
        """Clears failed login attempts for identifier+IP upon successful authentication."""
        clean_id = identifier.strip().lower()
        clean_ip = ip_address.strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM login_attempts WHERE identifier = ? AND ip_address = ?",
                (clean_id, clean_ip)
            )
            conn.commit()

    # -----------------------------------------------------------------------
    # Phase 3: Gmail OAuth 2.0 State & Token CRUD
    # -----------------------------------------------------------------------

    def save_oauth_state(
        self,
        state: str,
        user_id: str,
        session_id: str,
        code_verifier: Optional[str],
        redirect_uri: str,
        expires_at: str
    ) -> None:
        """Stores a cryptographically bound, single-use OAuth state record (hashed state and encrypted verifier)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO oauth_states (state, user_id, session_id, code_verifier, redirect_uri, expires_at, is_used)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (state, user_id, session_id, code_verifier, redirect_uri, expires_at))
            conn.commit()

    def get_oauth_state(self, state_key: str, raw_state: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves an OAuth state record by state hash or legacy raw state value."""
        if not state_key:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if raw_state and raw_state != state_key:
                cursor.execute("SELECT * FROM oauth_states WHERE state = ? OR state = ? LIMIT 1", (state_key, raw_state))
            else:
                cursor.execute("SELECT * FROM oauth_states WHERE state = ? LIMIT 1", (state_key,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def consume_oauth_state(self, state_key: str, raw_state: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Atomically claims an OAuth state in a single SQL operation to prevent concurrent replay.
        Updates is_used = 1 WHERE (state = ? OR state = ?) AND is_used = 0.
        Returns the claimed state record, or None if already used or non-existent.
        """
        if not state_key:
            return None
        target_keys = [state_key]
        if raw_state and raw_state != state_key:
            target_keys.append(raw_state)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if len(target_keys) == 2:
                cursor.execute("""
                    UPDATE oauth_states 
                    SET is_used = 1 
                    WHERE (state = ? OR state = ?) AND is_used = 0
                """, (target_keys[0], target_keys[1]))
            else:
                cursor.execute("""
                    UPDATE oauth_states 
                    SET is_used = 1 
                    WHERE state = ? AND is_used = 0
                """, (target_keys[0],))

            if cursor.rowcount == 0:
                conn.commit()
                return None
            conn.commit()

            if len(target_keys) == 2:
                cursor.execute("SELECT * FROM oauth_states WHERE state = ? OR state = ? LIMIT 1", (target_keys[0], target_keys[1]))
            else:
                cursor.execute("SELECT * FROM oauth_states WHERE state = ? LIMIT 1", (target_keys[0],))
            row = cursor.fetchone()
            return dict(row) if row else None





    def mark_oauth_state_used(self, state_key: str, raw_state: Optional[str] = None) -> bool:
        """Marks an OAuth state as used immediately."""
        if not state_key:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if raw_state and raw_state != state_key:
                cursor.execute("UPDATE oauth_states SET is_used = 1 WHERE state = ? OR state = ?", (state_key, raw_state))
            else:
                cursor.execute("UPDATE oauth_states SET is_used = 1 WHERE state = ?", (state_key,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_oauth_state_verifier(self, state_key: str, raw_state: Optional[str] = None) -> None:
        """Clears the code_verifier ciphertext from an OAuth state record after token exchange."""
        if not state_key:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if raw_state and raw_state != state_key:
                cursor.execute("UPDATE oauth_states SET code_verifier = NULL WHERE state = ? OR state = ?", (state_key, raw_state))
            else:
                cursor.execute("UPDATE oauth_states SET code_verifier = NULL WHERE state = ?", (state_key,))
            conn.commit()

    def cleanup_expired_oauth_states(self) -> int:
        """Deletes expired or used OAuth states older than 1 hour to keep state table compact."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM oauth_states 
                WHERE is_used = 1 OR expires_at < datetime('now', '-1 hour')
            """)
            conn.commit()
            return cursor.rowcount

    def save_user_gmail_token(self, token_data: Dict[str, Any]) -> None:
        """
        Stores or updates an authenticated user's encrypted Gmail credentials.
        Enforces one Gmail connection per user_id.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_gmail_tokens 
                (id, user_id, google_email, encrypted_access_token, encrypted_refresh_token, 
                 token_expiry, granted_scopes, connection_status, updated_at)
                VALUES 
                (:id, :user_id, :google_email, :encrypted_access_token, :encrypted_refresh_token, 
                 :token_expiry, :granted_scopes, :connection_status, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    google_email = excluded.google_email,
                    encrypted_access_token = excluded.encrypted_access_token,
                    encrypted_refresh_token = COALESCE(excluded.encrypted_refresh_token, user_gmail_tokens.encrypted_refresh_token),
                    token_expiry = excluded.token_expiry,
                    granted_scopes = excluded.granted_scopes,
                    connection_status = excluded.connection_status,
                    updated_at = CURRENT_TIMESTAMP
            """, token_data)
            conn.commit()

    def get_user_gmail_token(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves encrypted Gmail token record for an authenticated user."""
        if not user_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_gmail_tokens WHERE user_id = ? LIMIT 1", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def update_user_gmail_status(self, user_id: str, status: str) -> bool:
        """Updates connection status (e.g., 'connected', 'revoked', 'disconnected', 'expired')."""
        if not user_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_gmail_tokens 
                SET connection_status = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE user_id = ?
            """, (status, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def disconnect_user_gmail(self, user_id: str) -> bool:
        """
        Disconnects user's Gmail connection:
        Clears tokens and marks connection_status as 'disconnected'.
        """
        if not user_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_gmail_tokens 
                SET connection_status = 'disconnected', 
                    encrypted_access_token = '',
                    encrypted_refresh_token = NULL,
                    updated_at = CURRENT_TIMESTAMP 
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Phase 4: Gmail Ingestion, Synchronization Checkpoints & History CRUD
    # -----------------------------------------------------------------------

    def raw_signal_exists(self, user_id: str, source_type: str, source_id: str) -> bool:
        """Checks if a raw signal with the given source_id and source_type exists for the specified user."""
        return self.signal_exists(source_type=source_type, source_id=source_id, user_id=user_id)

    def get_sync_checkpoint(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the Gmail synchronization checkpoint for a specific user."""
        if not user_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gmail_sync_checkpoints WHERE user_id = ? LIMIT 1", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None





    def save_sync_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Stores or updates the Gmail synchronization checkpoint for a user."""
        import uuid
        params = {
            "id": checkpoint_data.get("id") or f"sync-chk-{uuid.uuid4().hex[:8]}",
            "user_id": checkpoint_data["user_id"],
            "last_history_id": checkpoint_data.get("last_history_id"),
            "last_sync_at": checkpoint_data.get("last_sync_at"),
            "sync_status": checkpoint_data.get("sync_status", "idle"),
            "is_syncing": int(checkpoint_data.get("is_syncing", 0)),
            "sync_lock_at": checkpoint_data.get("sync_lock_at"),
            "messages_synced_total": int(checkpoint_data.get("messages_synced_total", 0)),
            "last_error": checkpoint_data.get("last_error")
        }
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO gmail_sync_checkpoints 
                (id, user_id, last_history_id, last_sync_at, sync_status, is_syncing, sync_lock_at, messages_synced_total, last_error, updated_at)
                VALUES 
                (:id, :user_id, :last_history_id, :last_sync_at, :sync_status, :is_syncing, :sync_lock_at, :messages_synced_total, :last_error, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_history_id = COALESCE(excluded.last_history_id, gmail_sync_checkpoints.last_history_id),
                    last_sync_at = COALESCE(excluded.last_sync_at, gmail_sync_checkpoints.last_sync_at),
                    sync_status = excluded.sync_status,
                    is_syncing = excluded.is_syncing,
                    sync_lock_at = excluded.sync_lock_at,
                    messages_synced_total = excluded.messages_synced_total,
                    last_error = excluded.last_error,
                    updated_at = CURRENT_TIMESTAMP
            """, params)
            conn.commit()

    def acquire_sync_lock(self, user_id: str, timeout_seconds: int = 300) -> bool:
        """
        Safely acquires a synchronization lock for the user's Gmail connection.
        If a lock exists but is older than timeout_seconds, it is treated as stale and acquired.
        Returns True if acquired, False if another sync job is currently running.
        """
        if not user_id:
            return False
        import uuid
        from datetime import datetime, timezone, timedelta

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gmail_sync_checkpoints WHERE user_id = ? LIMIT 1", (user_id,))
            row = cursor.fetchone()

            if not row:
                # First time: create checkpoint row with lock acquired
                chk_id = f"sync-chk-{uuid.uuid4().hex[:8]}"
                cursor.execute("""
                    INSERT INTO gmail_sync_checkpoints
                    (id, user_id, sync_status, is_syncing, sync_lock_at)
                    VALUES (?, ?, 'in_progress', 1, ?)
                """, (chk_id, user_id, now_iso))
                conn.commit()
                return True

            chk = dict(row)
            if chk.get("is_syncing") == 1:
                # Check for stale lock timeout
                lock_at_str = chk.get("sync_lock_at")
                if lock_at_str:
                    try:
                        lock_dt = datetime.fromisoformat(lock_at_str.replace("Z", "+00:00"))
                        if datetime.now(timezone.utc) - lock_dt < timedelta(seconds=timeout_seconds):
                            # Active lock held by another job
                            return False
                    except Exception:
                        pass
                else:
                    return False

            # Lock is available or stale: acquire lock
            cursor.execute("""
                UPDATE gmail_sync_checkpoints
                SET is_syncing = 1, sync_status = 'in_progress', sync_lock_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (now_iso, user_id))
            conn.commit()
            return True

    def release_sync_lock(self, user_id: str, sync_status: str = "idle", error: Optional[str] = None) -> None:
        """Releases the sync lock for the user's Gmail connection."""
        if not user_id:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE gmail_sync_checkpoints
                SET is_syncing = 0, sync_status = ?, last_error = ?, sync_lock_at = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (sync_status, error, user_id))
            conn.commit()

    def save_sync_history(self, history_data: Dict[str, Any]) -> None:
        """Stores a synchronization audit history log."""
        import uuid
        from datetime import datetime, timezone

        params = {
            "id": history_data.get("id") or f"synclog-{uuid.uuid4().hex[:8]}",
            "user_id": history_data["user_id"],
            "sync_mode": history_data.get("sync_mode", "full"),
            "status": history_data.get("status", "success"),
            "messages_fetched": int(history_data.get("messages_fetched", 0)),
            "messages_processed": int(history_data.get("messages_processed", 0)),
            "duplicates_skipped": int(history_data.get("duplicates_skipped", 0)),
            "messages_failed": int(history_data.get("messages_failed", 0)),
            "drafts_created": int(history_data.get("drafts_created", 0)),
            "error_message": history_data.get("error_message"),
            "started_at": history_data.get("started_at") or datetime.now(timezone.utc).isoformat(),
            "completed_at": history_data.get("completed_at")
        }
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO gmail_sync_history
                (id, user_id, sync_mode, status, messages_fetched, messages_processed, 
                 duplicates_skipped, messages_failed, drafts_created, error_message, started_at, completed_at)
                VALUES
                (:id, :user_id, :sync_mode, :status, :messages_fetched, :messages_processed,
                 :duplicates_skipped, :messages_failed, :drafts_created, :error_message, :started_at, :completed_at)
            """, params)
            conn.commit()

    def get_sync_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent synchronization audit history logs for the user."""
        if not user_id:
            return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM gmail_sync_history
                WHERE user_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (user_id, limit))
            return [dict(r) for r in cursor.fetchall()]

    def count_user_gmail_signals(self, user_id: str) -> int:
        """Counts raw email signals imported from Gmail for the user."""
        if not user_id:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM raw_signals
                WHERE user_id = ? AND (source_type = 'email' OR source_type = 'gmail')
            """, (user_id,))
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def count_user_gmail_drafts(self, user_id: str) -> int:
        """Counts RAID drafts derived from Gmail email signals for the user."""
        if not user_id:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT d.id) as cnt
                FROM raid_drafts d
                JOIN raw_signals s ON d.evidence_source_id = s.id
                WHERE d.user_id = ? AND s.user_id = ?
                  AND (s.source_type = 'email' OR s.source_type = 'gmail')
            """, (user_id, user_id))
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    # =========================================================================
    # Data Folder File Sync & Tracking Methods (Stage 7)
    # =========================================================================

    def save_file_sync_record(self, record: Dict[str, Any], user_id: str = "default_user") -> None:
        """
        Inserts or updates a source file sync tracking record.
        """
        eff_user_id = user_id or record.get("user_id") or "default_user"
        path_hash = hashlib.md5(str(record["file_path"]).encode()).hexdigest()[:10]
        rec_id = record.get("id") or f"sync-{eff_user_id}-{path_hash}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO source_file_sync (
                    id, file_path, file_name, source_type, file_size, content_hash,
                    last_modified, record_count, signals_created, signals_classified,
                    drafts_created, status, error_message, user_id, discovered_at, processed_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE(?, CURRENT_TIMESTAMP), ?
                )
                ON CONFLICT(file_path, user_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    source_type = excluded.source_type,
                    file_size = excluded.file_size,
                    content_hash = excluded.content_hash,
                    last_modified = excluded.last_modified,
                    record_count = excluded.record_count,
                    signals_created = excluded.signals_created,
                    signals_classified = excluded.signals_classified,
                    drafts_created = excluded.drafts_created,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    processed_at = COALESCE(excluded.processed_at, source_file_sync.processed_at)
            """, (
                rec_id,
                record["file_path"],
                record["file_name"],
                record["source_type"],
                int(record.get("file_size", 0)),
                record["content_hash"],
                record.get("last_modified", ""),
                int(record.get("record_count", 0)),
                int(record.get("signals_created", 0)),
                int(record.get("signals_classified", 0)),
                int(record.get("drafts_created", 0)),
                record.get("status", "DISCOVERED"),
                record.get("error_message"),
                eff_user_id,
                record.get("discovered_at"),
                record.get("processed_at")
            ))

    def get_file_sync_record(self, file_path: str, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """
        Retrieves the sync tracking record for a given file path and tenant.
        """
        eff_user_id = user_id or "default_user"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM source_file_sync WHERE file_path = ? AND user_id = ?",
                (file_path, eff_user_id)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_file_sync_records(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """
        Retrieves all file sync records for the given user, ordered by discovery/processed time.
        """
        eff_user_id = user_id or "default_user"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM source_file_sync WHERE user_id = ? ORDER BY file_path ASC",
                (eff_user_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def update_file_sync_status(
        self,
        file_path: str,
        status: str,
        record_count: Optional[int] = None,
        signals_created: Optional[int] = None,
        signals_classified: Optional[int] = None,
        drafts_created: Optional[int] = None,
        error_message: Optional[str] = None,
        user_id: str = "default_user"
    ) -> bool:
        """
        Updates status, metrics and timestamps for a tracked source file.
        """
        eff_user_id = user_id or "default_user"
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE source_file_sync
                SET status = ?,
                    record_count = COALESCE(?, record_count),
                    signals_created = COALESCE(?, signals_created),
                    signals_classified = COALESCE(?, signals_classified),
                    drafts_created = COALESCE(?, drafts_created),
                    error_message = ?,
                    processed_at = CASE WHEN ? IN ('PROCESSED', 'FAILED') THEN ? ELSE processed_at END
                WHERE file_path = ? AND user_id = ?
            """, (
                status,
                record_count,
                signals_created,
                signals_classified,
                drafts_created,
                error_message,
                status,
                now_utc,
                file_path,
                eff_user_id
            ))
            return cursor.rowcount > 0



