"""
Page 6: Excel Export Center
KTern.AI RAID Identification Agent POC

Centralized export suite powered by openpyxl.
Generates styled, filter-enabled, frozen-header Excel reports
for Approved Register, Review Queue, Predictive Insights, Audit History,
and an all-in-one Multi-Tab Master Report.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from ui.service_helper import get_raid_service
from src.export.excel_export import (
    export_approved_register,
    export_review_queue,
    export_predictions,
    export_audit_history,
    export_combined_report
)

MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render_page():
    st.set_page_config(page_title="Excel Export Center - KTern.AI", page_icon="📥", layout="wide")
    st.title("📥 RAID Excel Export Center")
    st.markdown("Generate formatted, audit-compliant Microsoft Excel (`.xlsx`) workbooks for SteerCo reporting, PM review, and governance archives.")
    st.divider()

    service = get_raid_service()

    # Load active data
    approved_items = service.get_approved_items()
    pending_drafts = service.get_pending_drafts()
    predictions = service.get_predictions()
    audit_history = service.get_audit_history()

    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # --- 1. Master All-In-One Export ---
    st.subheader("📑 Master Multi-Tab Project RAID Report")
    st.caption("Consolidates Approved Register, Review Queue, Predictive Insights, and Compliance Audit Trail into one multi-worksheet workbook.")

    c_summary1, c_summary2, c_summary3, c_summary4 = st.columns(4)
    with c_summary1:
        st.metric("Approved Records", len(approved_items))
    with c_summary2:
        st.metric("Pending Review Items", len(pending_drafts))
    with c_summary3:
        st.metric("Foresight Predictions", len(predictions))
    with c_summary4:
        st.metric("Audit History Records", len(audit_history))

    combined_bytes = export_combined_report(
        approved_items=approved_items,
        draft_items=pending_drafts,
        predictions=predictions,
        audit_entries=audit_history
    )

    st.download_button(
        label="📥 Download Master Comprehensive Report (.xlsx)",
        data=combined_bytes,
        file_name=f"KTern_RAID_Master_Report_{now_str}.xlsx",
        mime=MIME_XLSX,
        use_container_width=True
    )

    st.divider()

    # --- 2. Individual Specialized Exports ---
    st.subheader("📊 Individual Dataset Exports")
    st.markdown("Download targeted single-worksheet Excel workbooks formatted with custom widths, frozen headers, and auto-filters.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Approved RAID Register")
        st.write(f"Contains **{len(approved_items)}** formally approved items.")
        appr_bytes = export_approved_register(approved_items)
        st.download_button(
            label="Download Approved Register (.xlsx)",
            data=appr_bytes,
            file_name=f"Approved_RAID_Register_{now_str}.xlsx",
            mime=MIME_XLSX,
            use_container_width=True
        )

        st.markdown("---")

        st.markdown("#### 🔮 Predictive Foresight Insights")
        st.write(f"Contains **{len(predictions)}** active heuristic risk warnings.")
        pred_bytes = export_predictions(predictions)
        st.download_button(
            label="Download Predictive Insights (.xlsx)",
            data=pred_bytes,
            file_name=f"Predictive_Foresight_{now_str}.xlsx",
            mime=MIME_XLSX,
            use_container_width=True
        )

    with col2:
        st.markdown("#### 📋 Pending Review Queue")
        st.write(f"Contains **{len(pending_drafts)}** drafts awaiting PM action.")
        rev_bytes = export_review_queue(pending_drafts)
        st.download_button(
            label="Download Review Queue (.xlsx)",
            data=rev_bytes,
            file_name=f"Pending_Review_Queue_{now_str}.xlsx",
            mime=MIME_XLSX,
            use_container_width=True
        )

        st.markdown("---")

        st.markdown("#### 📜 Compliance Audit Trail")
        st.write(f"Contains **{len(audit_history)}** historical lifecycle events.")
        audit_bytes = export_audit_history(audit_history)
        st.download_button(
            label="Download Audit History (.xlsx)",
            data=audit_bytes,
            file_name=f"RAID_Audit_Trail_{now_str}.xlsx",
            mime=MIME_XLSX,
            use_container_width=True
        )

    st.divider()

    # Excel Standards & Styling Notes
    with st.expander("ℹ️ Excel Formatting & Compliance Standards"):
        st.markdown(
            """
            - **Header Rows:** Styled in Corporate Navy (`#1F4E78`) with bold white text.
            - **Navigation:** Header row is frozen (`A2`) for continuous scrolling.
            - **Filtering:** Excel auto-filters are activated across all data tables.
            - **Column Sizing:** Dynamic auto-fitting with bounded bounds (14 to 60 characters).
            - **Data Integrity:** Numeric confidence scores are exported as true numbers; source quotations and IDs are preserved without truncation.
            - **Safety:** Read-only operation against SQLite; exports do not alter backend database state.
            """
        )


if __name__ == "__main__":
    render_page()
