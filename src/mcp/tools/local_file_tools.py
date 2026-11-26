"""
Local File Tools for MCP Server.
Provides:
1. select_or_configure_folder: Validates and configures permitted local folder to scan.
2. scan_folder: Recursively discovers supported files and categorizes them using content hashes and DB sync records.
3. read_file: Safely reads content and metadata of a permitted file with size and path traversal checks.
4. get_file_status: Retrieves sync/processing status of a specific file from SQLite database.
"""

import os
import base64
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from config import DATA_DIR, BASE_DIR
from src.normalisation.schemas import SourceType
from src.ingestion.folder_scanner import (
    DataFolderScanner,
    compute_file_hash,
    SUPPORTED_EXTENSIONS,
    ALL_SUPPORTED_EXTENSIONS,
    EXCLUDED_DIR_NAMES,
    EXCLUDED_EXTENSIONS
)
from src.database.db_manager import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum allowed file size for reading (50 MB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# System directories that must NEVER be accessed
FORBIDDEN_ROOT_PREFIXES = [
    "/bin", "/sbin", "/etc", "/dev", "/usr", "/System", "/Library",
    "/private/etc", "/private/var/db"
]

# Text extensions that are read as UTF-8 strings
TEXT_EXTENSIONS: Set[str] = {".csv", ".eml", ".txt", ".json"}

# Binary extensions that are read and base64 encoded
BINARY_EXTENSIONS: Set[str] = {".docx", ".xlsx", ".xls"}

# Configuration persistence file path
CONFIG_STATE_FILE = BASE_DIR / "data" / ".mcp_folder_config.json"

# Active configured folder in-memory state
_configured_folder: Optional[Path] = None


def get_current_configured_folder() -> Path:
    """
    Returns the currently active configured folder path.
    Checks in-memory state first, then persistent config file, and defaults to DATA_DIR.
    """
    global _configured_folder
    if _configured_folder is not None and _configured_folder.exists() and _configured_folder.is_dir():
        return _configured_folder

    # Check persisted config file
    if CONFIG_STATE_FILE.exists():
        try:
            import json
            data = json.loads(CONFIG_STATE_FILE.read_text(encoding="utf-8"))
            saved_path = Path(data.get("configured_folder", "")).resolve()
            if saved_path.exists() and saved_path.is_dir():
                _configured_folder = saved_path
                return _configured_folder
        except Exception as exc:
            logger.warning("Could not read persistent MCP folder config: %s", exc)

    _configured_folder = DATA_DIR.resolve()
    return _configured_folder


def set_current_configured_folder(folder_path: Path) -> None:
    """
    Sets the currently active configured folder path in memory and persists to disk.
    """
    global _configured_folder
    resolved = folder_path.resolve()
    _configured_folder = resolved

    # Persist to disk so subsequent / newly spawned server processes retain the folder
    try:
        import json
        CONFIG_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_STATE_FILE.write_text(
            json.dumps({
                "configured_folder": str(resolved),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }, indent=2),
            encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Failed to persist MCP folder config to %s: %s", CONFIG_STATE_FILE, exc)


def validate_path_safety(path: Path) -> Tuple[bool, str]:
    """
    Validates that a path does not traverse into prohibited operating system directories
    or dangerous file locations.
    """
    try:
        resolved = path.resolve()
    except Exception as exc:
        return False, f"Could not resolve path: {exc}"

    # Check against root
    if str(resolved) in ("/", ""):
        return False, "Access to system root directory is prohibited."

    # Check system forbidden prefixes
    resolved_str = str(resolved)
    for prefix in FORBIDDEN_ROOT_PREFIXES:
        if resolved_str == prefix or resolved_str.startswith(prefix + "/"):
            return False, f"Access to system directory '{prefix}' is prohibited."

    return True, ""


# ---------------------------------------------------------------------------
# Tool 1: select_or_configure_folder
# ---------------------------------------------------------------------------

def select_or_configure_folder(folder_path: str) -> Dict[str, Any]:
    """
    Validates and configures the target local directory to scan.
    Ensures path exists, is a directory, and does not violate security constraints.
    """
    if not folder_path or not str(folder_path).strip():
        return {
            "success": False,
            "error": "Folder path cannot be empty."
        }

    target = Path(folder_path.strip()).expanduser()

    # If relative, resolve against BASE_DIR
    if not target.is_absolute():
        target = (BASE_DIR / target).resolve()
    else:
        target = target.resolve()

    # Security check
    safe, reason = validate_path_safety(target)
    if not safe:
        logger.warning("Rejected folder configuration for safety: %s (%s)", target, reason)
        return {
            "success": False,
            "error": f"Security restriction: {reason}"
        }

    # Existence check
    if not target.exists():
        return {
            "success": False,
            "error": f"Directory does not exist: {target}"
        }

    if not target.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {target}"
        }

    set_current_configured_folder(target)
    logger.info("MCP folder configured to: %s", target)

    return {
        "success": True,
        "folder_path": str(target),
        "message": f"Successfully configured local folder: {target}"
    }


# ---------------------------------------------------------------------------
# Tool 2: scan_folder
# ---------------------------------------------------------------------------

def scan_folder(
    folder_path: Optional[str] = None,
    user_id: str = "default_user",
    db_manager: Optional[DatabaseManager] = None
) -> Dict[str, Any]:
    """
    Recursively discovers supported project communication, plan, and defect files.
    Computes SHA-256 hashes, queries existing SQLite sync tracking, and categorizes
    status without reading file contents into the RAID pipeline or invoking the LLM.
    """
    eff_user_id = user_id or "default_user"
    db = db_manager or DatabaseManager()

    if folder_path and str(folder_path).strip():
        config_res = select_or_configure_folder(folder_path)
        if not config_res.get("success"):
            return {
                "success": False,
                "data_dir": folder_path,
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
                "unsupported_files": [],
                "error": config_res.get("error")
            }
        target_dir = Path(config_res["folder_path"])
    else:
        target_dir = get_current_configured_folder()

    scanner = DataFolderScanner(db_manager=db, data_dir=target_dir)
    scan_summary = scanner.scan_data_folder(target_dir=target_dir, user_id=eff_user_id)
    scan_summary["success"] = True
    return scan_summary


# ---------------------------------------------------------------------------
# Tool 3: read_file
# ---------------------------------------------------------------------------

def read_file(file_path: str) -> Dict[str, Any]:
    """
    Safely reads the contents and metadata of an individual permitted file.
    Enforces path traversal checks, file size limits, and supported extensions.
    Returns text for UTF-8 files or base64 string for binary files (.docx, .xlsx).
    """
    if not file_path or not str(file_path).strip():
        return {
            "success": False,
            "file_path": file_path,
            "error": "File path cannot be empty."
        }

    raw_path = Path(file_path.strip()).expanduser()

    # Resolve path
    if not raw_path.is_absolute():
        # Check against configured folder first, then BASE_DIR
        candidate1 = (get_current_configured_folder() / raw_path).resolve()
        candidate2 = (BASE_DIR / raw_path).resolve()
        target = candidate1 if candidate1.exists() else candidate2
    else:
        target = raw_path.resolve()

    # Security check: system roots
    safe, reason = validate_path_safety(target)
    if not safe:
        return {
            "success": False,
            "file_path": str(target),
            "error": f"Security restriction: {reason}"
        }

    # Verify existence and file type
    if not target.exists():
        return {
            "success": False,
            "file_path": str(target),
            "error": f"File does not exist: {target}"
        }

    if not target.is_file():
        return {
            "success": False,
            "file_path": str(target),
            "error": f"Path is not a regular file: {target}"
        }

    suffix = target.suffix.lower()
    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "file_path": str(target),
            "error": f"Unsupported file extension '{suffix}'. Supported: {sorted(list(ALL_SUPPORTED_EXTENSIONS))}"
        }

    try:
        stat_res = target.stat()
        file_size = stat_res.st_size
        mtime_iso = datetime.fromtimestamp(stat_res.st_mtime, tz=timezone.utc).isoformat()
    except Exception as exc:
        return {
            "success": False,
            "file_path": str(target),
            "error": f"Failed to access file metadata: {exc}"
        }

    # Check size limit
    if file_size > MAX_FILE_SIZE_BYTES:
        return {
            "success": False,
            "file_path": str(target),
            "error": f"File size ({file_size} bytes) exceeds maximum limit of {MAX_FILE_SIZE_BYTES} bytes (50MB)."
        }

    # Compute hash
    try:
        content_hash = compute_file_hash(target)
    except Exception as exc:
        return {
            "success": False,
            "file_path": str(target),
            "error": f"Failed to compute file hash: {exc}"
        }

    # Determine source type
    scanner = DataFolderScanner(data_dir=get_current_configured_folder())
    source_type = scanner.determine_source_type(target)

    # Read content
    try:
        if suffix in TEXT_EXTENSIONS:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content_str = f.read()
            encoding = "utf-8"
        elif suffix in BINARY_EXTENSIONS:
            with open(target, "rb") as f:
                raw_bytes = f.read()
            content_str = base64.b64encode(raw_bytes).decode("ascii")
            encoding = "base64"
        else:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content_str = f.read()
            encoding = "utf-8"
    except Exception as exc:
        logger.error("Error reading file %s: %s", target, exc)
        return {
            "success": False,
            "file_path": str(target),
            "error": f"Read error: {exc}"
        }

    return {
        "success": True,
        "file_path": str(target),
        "file_name": target.name,
        "file_size": file_size,
        "content_hash": content_hash,
        "extension": suffix,
        "source_type": source_type.value if source_type else "unknown",
        "encoding": encoding,
        "content": content_str,
        "last_modified": mtime_iso
    }


# ---------------------------------------------------------------------------
# Tool 4: get_file_status
# ---------------------------------------------------------------------------

def get_file_status(
    file_path: str,
    user_id: str = "default_user",
    db_manager: Optional[DatabaseManager] = None
) -> Dict[str, Any]:
    """
    Queries the database sync records for a file's processing status.
    Returns status: DISCOVERED, PROCESSING, PROCESSED, MODIFIED, or FAILED.
    """
    if not file_path or not str(file_path).strip():
        return {
            "success": False,
            "file_path": file_path,
            "error": "File path cannot be empty."
        }

    db = db_manager or DatabaseManager()
    eff_user_id = user_id or "default_user"
    clean_path = file_path.strip()

    record = db.get_file_sync_record(clean_path, user_id=eff_user_id)

    # Also try relative to BASE_DIR if not found
    if not record and Path(clean_path).is_absolute():
        try:
            rel = str(Path(clean_path).relative_to(BASE_DIR))
            record = db.get_file_sync_record(rel, user_id=eff_user_id)
        except ValueError:
            pass

    if not record:
        return {
            "success": True,
            "file_path": clean_path,
            "status": "DISCOVERED",
            "record_count": 0,
            "signals_created": 0,
            "signals_classified": 0,
            "drafts_created": 0,
            "processed_at": None,
            "error_message": None
        }

    return {
        "success": True,
        "file_path": clean_path,
        "status": record.get("status", "DISCOVERED"),
        "record_count": record.get("record_count", 0),
        "signals_created": record.get("signals_created", 0),
        "signals_classified": record.get("signals_classified", 0),
        "drafts_created": record.get("drafts_created", 0),
        "processed_at": record.get("processed_at"),
        "error_message": record.get("error_message")
    }
