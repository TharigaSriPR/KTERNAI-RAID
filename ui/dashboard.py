"""
Main Streamlit Entry Point: ui/dashboard.py
KTern.AI RAID Identification Agent POC

Provides high-level executive dashboard with KPI metric cards,
visual categorical breakdowns, system status, and navigation to sub-pages.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from config import DB_PATH, LLM_PROVIDER
from ui.service_helper import get_raid_service
from ui.components.kpi_cards import render_kpi_cards
from ui.components.charts import render_distribution_charts


def main():
    st.set_page_config(
        page_title="KTern.AI - RAID Agent POC",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Professional KaarTech Corporate Red Theme Styles
    st.markdown(
        """
        <style>
        /* KaarTech Professional Red Theme */
        :root {
            --kaartech-red: #9A1C1F;
            --kaartech-dark: #1C1C1C;
        }
        
        /* Metric Cards Styling */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-top: 3px solid #9A1C1F;
            border-radius: 6px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #4B5563 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #1C1C1C !important;
            font-weight: 700 !important;
            font-size: 1.75rem !important;
        }
        
        /* Button Accents */
        .stButton>button {
            background-color: #9A1C1F !important;
            color: #FFFFFF !important;
            border: 1px solid #9A1C1F !important;
            border-radius: 4px !important;
        }
        .stButton>button:hover {
            background-color: #7D1417 !important;
            border-color: #7D1417 !important;
        }
        
        /* Subheaders and Headings */
        h1, h2, h3, h4 {
            color: #1C1C1C;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main Header with KaarTech Logo
    logo_path = Path(__file__).resolve().parent / "assets" / "kaartech_logo.png"
    if logo_path.exists():
        head_col1, head_col2 = st.columns([1, 8])
        with head_col1:
            st.image(str(logo_path), width=85)
        with head_col2:
            st.title("KTern.AI RAID Identification Agent")
    else:
        st.title("KTern.AI RAID Identification Agent")
    st.divider()

    # Quick Action Banner for Data Synchronisation
    c_banner1, c_banner2 = st.columns([4, 1])
    with c_banner1:
        st.markdown("🔄 **Data Synchronisation:** Ingest and process project emails, chats, meeting notes, project plans, and defect logs from the `data/` folder.")
    with c_banner2:
        st.page_link("pages/0_Data_Import_Sync.py", label="📥 Data Import & Sync")

    st.divider()

    service = get_raid_service()

    # 1. Operational KPI Cards
    kpi_data = service.get_kpis()
    render_kpi_cards(kpi_data)

    st.divider()

    # 2. Visual Summaries
    st.subheader("📊 Portfolio RAID Analytics")
    all_drafts = service.get_all_drafts()
    approved_items = service.get_approved_items()

    # Combine drafts and approved items for complete picture
    combined_summary_items = []
    for d in all_drafts:
        combined_summary_items.append({
            "id": d.get("id"),
            "type": d.get("type"),
            "status": d.get("status"),
            "phase": d.get("phase"),
            "workstream": d.get("workstream")
        })

    if not combined_summary_items and not approved_items:
        st.info("ℹ️ **Database is Empty**: No RAID items recorded yet. Run `python3 app.py` or trigger data ingestion to populate records.")
    else:
        render_distribution_charts(combined_summary_items)


render_dashboard = main

if __name__ == "__main__":
    main()
