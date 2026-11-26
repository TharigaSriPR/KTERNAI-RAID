"""
Configuration module for the RAID Identification Agent POC.
KTern.AI / KaarTech Architecture Configuration.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from local .env file if present (without external dependencies)
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.is_file():
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip("'\"")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"

# Sub-source Directories
EMAILS_DIR = RAW_DATA_DIR / "emails"
CHATS_DIR = RAW_DATA_DIR / "chats"
MEETING_NOTES_DIR = RAW_DATA_DIR / "meeting_notes"
PROJECT_PLANS_DIR = RAW_DATA_DIR / "project_plans"
DEFECTS_DIR = RAW_DATA_DIR / "defects"

# SQLite Database
DB_PATH = DB_DIR / "raid_agent.db"

# Centralised Logging Settings
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "raid_agent.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# LLM & Model Settings (Swappable layer as per doc Section 08)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()  # openrouter, gemini, or mock
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b-it:free")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
LLM_MAX_INPUT_LENGTH = int(os.getenv("LLM_MAX_INPUT_LENGTH", "4000"))

# Step 3 Classification Settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# Step 4 Predictive Engine Settings
UPCOMING_DEADLINE_THRESHOLD_DAYS = int(os.getenv("UPCOMING_DEADLINE_THRESHOLD_DAYS", "7"))
DEFECT_CLUSTER_MIN_SIZE = int(os.getenv("DEFECT_CLUSTER_MIN_SIZE", "3"))
DEFECT_CLUSTER_SIMILARITY_THRESHOLD = float(os.getenv("DEFECT_CLUSTER_SIMILARITY_THRESHOLD", "0.40"))

# Step 4 Deduplication Thresholds
DEDUPE_DUPLICATE_THRESHOLD = float(os.getenv("DEDUPE_DUPLICATE_THRESHOLD", "0.75"))
DEDUPE_REVIEW_THRESHOLD = float(os.getenv("DEDUPE_REVIEW_THRESHOLD", "0.50"))
DEDUPE_SIMILARITY_THRESHOLD = DEDUPE_DUPLICATE_THRESHOLD  # Backward compatibility

# SAP Activate Project Phases
SAP_PHASES = ["Prepare", "Explore", "Realize", "Deploy", "Run"]

# Common Workstreams
WORKSTREAMS = [
    "FI", "CO", "SD", "MM", "PP", "QM", "PM", "PP"
    "Basis", "Security", "ABAP / Custom Code", 
    "Data Migration", "Integration", "Testing", "Change Management"
]

# RAID Categories
RAID_TYPES = ["Risk", "Assumption", "Issue", "Decision", "Dependency"]

# Backend API & CORS Security
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8501,http://localhost:8502,http://127.0.0.1:8501,http://127.0.0.1:8502"
    ).split(",")
    if origin.strip()
]
