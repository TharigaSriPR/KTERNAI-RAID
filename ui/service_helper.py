"""
Service retrieval helper for Streamlit UI pages.
Provides a cached, session-persistent RaidService instance backed by SQLite DatabaseManager.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from config import DB_PATH
from src.database.db_manager import DatabaseManager
from src.service import RaidService


def apply_navigation_styles():
    """Injects styling for the 7 sidebar navigation line items (light red, slightly larger font)."""
    st.markdown(
        """
        <style>
        /* 7 Navigation Line Items Styling */
        [data-testid="stSidebarNav"] {
            padding-top: 1.25rem !important;
        }
        
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNavItems"] a,
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] li a {
            padding: 10px 14px !important;
            border-radius: 6px !important;
            margin-bottom: 6px !important;
            text-decoration: none !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        [data-testid="stSidebarNav"] a span,
        [data-testid="stSidebarNavItems"] a span,
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] li a span,
        section[data-testid="stSidebar"] a span {
            color: #C93B3B !important; /* Professional light red */
            font-size: 1.15rem !important; /* A little big */
            font-weight: 600 !important;
            letter-spacing: 0.2px !important;
        }
        
        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebarNavItems"] a:hover {
            background-color: rgba(201, 59, 59, 0.08) !important;
        }
        
        [data-testid="stSidebarNav"] a:hover span,
        [data-testid="stSidebarNavItems"] a:hover span {
            color: #9A1C1F !important;
        }
        
        /* Active / Selected Page Navigation Item */
        [data-testid="stSidebarNav"] a[aria-current="page"],
        [data-testid="stSidebarNavItems"] a[aria-current="page"] {
            background-color: rgba(154, 28, 31, 0.10) !important;
            border-left: 4px solid #9A1C1F !important;
        }
        
        [data-testid="stSidebarNav"] a[aria-current="page"] span,
        [data-testid="stSidebarNavItems"] a[aria-current="page"] span {
            color: #9A1C1F !important;
            font-weight: 700 !important;
            font-size: 1.18rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def get_raid_service() -> RaidService:
    """Retrieves or initializes the session-scoped RaidService instance and applies nav styles."""
    apply_navigation_styles()
    import importlib
    import src.service
    if not hasattr(RaidService, "scan_data_folder"):
        importlib.reload(src.service)
    from src.service import RaidService as FreshRaidService
    if "raid_service" not in st.session_state or st.session_state.raid_service is None:
        db = DatabaseManager(DB_PATH)
        st.session_state.raid_service = FreshRaidService(db)
    return st.session_state.raid_service
