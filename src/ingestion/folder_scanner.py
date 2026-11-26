"""
Data Folder Discovery & Deduplication Scanner.
KTern.AI / KaarTech Architectural Dataflow:
Recursively scans project data directories, discovers supported project communication
and management records (emails, chats, MOMs, project plans, defect registers),
computes SHA-256 hashes for source-level deduplication, detects modifications,
routes files to appropriate ingestion adapters, and filters out non-source files
(databases, backups, summary reports, and test artifacts).
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from config import DATA_DIR, BASE_DIR
ROOT_DIR = BASE_DIR
from src.normalisation.schemas import SourceType
from src.ingestion.base_ingestor import BaseIngestor
from src.ingestion.email_ingestor import EmailIngestor
from src.ingestion.chat_ingestor import ChatIngestor
from src.ingestion.meeting_ingestor import MeetingIngestor
from src.ingestion.plan_ingestor import PlanIngestor
from src.ingestion.defect_ingestor import DefectIngestor
from src.database.db_manager import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported file extensions per source type
SUPPORTED_EXTENSIONS: Dict[SourceType, Set[str]] = {
    SourceType.EMAIL: {".csv", ".eml", ".txt"},
    SourceType.CHAT: {".csv", ".json", ".txt"},
    SourceType.MEETING_NOTES: {".docx", ".txt", ".csv"},
    SourceType.PROJECT_PLAN: {".csv", ".xlsx", ".xls"},
    SourceType.DEFECT: {".csv", ".xlsx", ".xls"},
}

ALL_SUPPORTED_EXTENSIONS: Set[str] = {
    ext for exts in SUPPORTED_EXTENSIONS.values() for ext in exts
}

# Directories that must NEVER be scanned as source document inputs
EXCLUDED_DIR_NAMES: Set[str] = {
    "db", ".git", "__pycache__", "processed", ".venv", "venv", ".pytest_cache", ".system_generated"
}

# Extensions that must NEVER be ingested as project source data
EXCLUDED_EXTENSIONS: Set[str] = {
    ".db", ".sqlite", ".sqlite3", ".bak", ".py", ".pyc", ".log", ".tmp",
    ".swp", ".ds_store", ".zip", ".tar", ".gz"
}

# Substrings in filenames representing test artifacts, validation summaries or database backups
EXCLUDED_FILENAME_SUBSTRINGS: List[str] = [
    "summary", "results", "validation", ".backup", "test_out", ".bak", "export"
]


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 digest of file content for reliable change detection."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class DataFolderScanner:
    """
    Controlled discovery scanner that walks data directories, identifies source types,
    checks database sync state to prevent duplicates, and routes files to appropriate ingestors.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None, data_dir: Optional[Path] = None):
        self.db = db_manager or DatabaseManager()
        self.data_dir = (data_dir or DATA_DIR).resolve()
        self._ingestors: Dict[SourceType, BaseIngestor] = {
            SourceType.EMAIL: EmailIngestor(),
            SourceType.CHAT: ChatIngestor(),
            SourceType.MEETING_NOTES: MeetingIngestor(),
            SourceType.PROJECT_PLAN: PlanIngestor(),
            SourceType.DEFECT: DefectIngestor()
        }

    def determine_source_type(self, file_path: Path) -> Optional[SourceType]:
        """
        Determines the ProjectSignal source type based on directory path, filename, and extension.
        Returns None if the file does not match any project source category.
        """
        path_str = str(file_path).lower()
        parts = [p.lower() for p in file_path.parts]
        name = file_path.name.lower()
        suffix = file_path.suffix.lower()

        # 1. Directory-based classification (strongest signal)
        if any(d in parts for d in ("emails", "email", "outlook", "mail")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.EMAIL]:
                return SourceType.EMAIL
        if any(d in parts for d in ("chats", "chat", "teams", "slack", "transcripts")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.CHAT]:
                return SourceType.CHAT
        if any(d in parts for d in ("meeting_notes", "meetings", "mom", "minutes")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.MEETING_NOTES]:
                return SourceType.MEETING_NOTES
        if any(d in parts for d in ("project_plans", "plans", "schedule", "tasks")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.PROJECT_PLAN]:
                return SourceType.PROJECT_PLAN
        if any(d in parts for d in ("defects", "defect", "bugs", "issues", "jira")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.DEFECT]:
                return SourceType.DEFECT

        # 2. Filename keyword classification
        if any(k in name for k in ("email", "mail", "msg")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.EMAIL]:
                return SourceType.EMAIL
        if any(k in name for k in ("chat", "teams", "slack", "transcript")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.CHAT]:
                return SourceType.CHAT
        if any(k in name for k in ("mom", "meeting", "minutes")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.MEETING_NOTES]:
                return SourceType.MEETING_NOTES
        if any(k in name for k in ("plan", "task", "schedule", "wbs")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.PROJECT_PLAN]:
                return SourceType.PROJECT_PLAN
        if any(k in name for k in ("defect", "bug", "issue")):
            if suffix in SUPPORTED_EXTENSIONS[SourceType.DEFECT]:
                return SourceType.DEFECT

        # 3. Format-specific fallback
        if suffix == ".docx":
            return SourceType.MEETING_NOTES
        if suffix == ".eml":
            return SourceType.EMAIL

        return None

    def is_excluded_path(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validates whether a path should be excluded from ingestion.
        Returns (is_excluded, reason).
        """
        # Hidden files or directories
        for part in file_path.parts:
            if part.startswith(".") and part not in (".", ".."):
                return True, f"Hidden directory or file ({part})"

        # Excluded directory names
        for part in file_path.parts:
            if part.lower() in EXCLUDED_DIR_NAMES:
                return True, f"Excluded directory ({part})"

        suffix = file_path.suffix.lower()
        if suffix in EXCLUDED_EXTENSIONS:
            return True, f"Excluded file extension ({suffix})"

        name_lower = file_path.name.lower()
        for sub in EXCLUDED_FILENAME_SUBSTRINGS:
            if sub in name_lower:
                return True, f"Matches excluded pattern ('{sub}')"

        return False, ""

    def scan_data_folder(
        self,
        target_dir: Optional[Path] = None,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Recursively discovers all supported files under target_dir, extracts file metadata,
        compares content hashes against the source_file_sync database, and categorizes status.

        Returns discovery summary:
        {
            "data_dir": str,
            "total_supported": int,
            "new_count": int,
            "processed_count": int,
            "modified_count": int,
            "failed_count": int,
            "files": List[Dict[str, Any]],
            "new_files": List[Dict[str, Any]],
            "processed_files": List[Dict[str, Any]],
            "modified_files": List[Dict[str, Any]],
            "failed_files": List[Dict[str, Any]],
            "unsupported_files": List[Dict[str, Any]]
        }
        """
        base_dir = (target_dir or self.data_dir).resolve()
        if not base_dir.exists():
            logger.warning("Target data directory does not exist: %s", base_dir)
            return {
                "data_dir": str(base_dir),
                "total_supported": 0,
                "new_count": 0,
                "processed_count": 0,
                "modified_count": 0,
                "failed_count": 0,
                "files": [],
                "new_files": [],
                "processed_files": [],
                "modified_files": [],
                "failed_files": [],
                "unsupported_files": []
            }

        all_discovered: List[Dict[str, Any]] = []
        new_files: List[Dict[str, Any]] = []
        processed_files: List[Dict[str, Any]] = []
        modified_files: List[Dict[str, Any]] = []
        failed_files: List[Dict[str, Any]] = []
        unsupported_files: List[Dict[str, Any]] = []

        # Recursively scan files
        for root, dirs, files in os.walk(base_dir):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIR_NAMES and not d.startswith(".")]

            for file_name in sorted(files):
                full_path = Path(root) / file_name
                try:
                    rel_path = full_path.relative_to(ROOT_DIR)
                except ValueError:
                    rel_path = full_path

                rel_str = str(rel_path)

                # Check exclusions
                excluded, reason = self.is_excluded_path(full_path)
                if excluded:
                    continue

                # Determine source type
                source_type = self.determine_source_type(full_path)
                if not source_type:
                    unsupported_files.append({
                        "file_path": rel_str,
                        "file_name": file_name,
                        "reason": f"No matching project source category for extension {full_path.suffix}"
                    })
                    continue

                try:
                    file_stat = full_path.stat()
                    file_size = file_stat.st_size
                    mtime_iso = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc).isoformat()
                    content_hash = compute_file_hash(full_path)
                except Exception as exc:
                    logger.error("Failed to read file metadata for %s: %s", rel_str, exc)
                    failed_files.append({
                        "file_path": rel_str,
                        "file_name": file_name,
                        "source_type": source_type.value,
                        "status": "FAILED",
                        "error_message": f"Read error: {exc}"
                    })
                    continue

                # Query sync table in database
                existing_sync = self.db.get_file_sync_record(rel_str, user_id=user_id)

                file_info = {
                    "file_path": rel_str,
                    "file_name": file_name,
                    "source_type": source_type.value,
                    "file_size": file_size,
                    "content_hash": content_hash,
                    "last_modified": mtime_iso,
                    "user_id": user_id,
                    "record_count": 0,
                    "signals_created": 0,
                    "signals_classified": 0,
                    "drafts_created": 0,
                    "error_message": None
                }

                if not existing_sync:
                    # Brand-new file (discovered on disk, not yet processed)
                    file_info["status"] = "DISCOVERED"
                    file_info["discovered_at"] = datetime.now(timezone.utc).isoformat()
                    new_files.append(file_info)
                else:
                    file_info["id"] = existing_sync.get("id")
                    file_info["record_count"] = existing_sync.get("record_count", 0)
                    file_info["signals_created"] = existing_sync.get("signals_created", 0)
                    file_info["signals_classified"] = existing_sync.get("signals_classified", 0)
                    file_info["drafts_created"] = existing_sync.get("drafts_created", 0)
                    file_info["discovered_at"] = existing_sync.get("discovered_at")
                    file_info["processed_at"] = existing_sync.get("processed_at")

                    if existing_sync.get("status") == "FAILED":
                        file_info["status"] = "FAILED"
                        file_info["error_message"] = existing_sync.get("error_message")
                        failed_files.append(file_info)
                    elif content_hash == existing_sync.get("content_hash"):
                        if existing_sync.get("status") == "PROCESSED":
                            file_info["status"] = "PROCESSED"
                            processed_files.append(file_info)
                        else:
                            file_info["status"] = existing_sync.get("status", "DISCOVERED")
                            new_files.append(file_info)
                    else:
                        # Content hash differs: file was modified on disk
                        file_info["status"] = "MODIFIED"
                        file_info["error_message"] = None
                        modified_files.append(file_info)

                all_discovered.append(file_info)

        return {
            "data_dir": str(base_dir),
            "total_supported": len(all_discovered),
            "new_count": len(new_files),
            "processed_count": len(processed_files),
            "modified_count": len(modified_files),
            "failed_count": len(failed_files),
            "files": all_discovered,
            "new_files": new_files,
            "processed_files": processed_files,
            "modified_files": modified_files,
            "failed_files": failed_files,
            "unsupported_files": unsupported_files
        }

    def get_ingestor_for_source_type(self, source_type: SourceType) -> BaseIngestor:
        """Returns the existing ingestion adapter for the specified source type."""
        return self._ingestors[source_type]
