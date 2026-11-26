"""
Page 2: Approved RAID Register
KTern.AI RAID Identification Agent POC

Displays official, approved project RAID items.
Features filtering, search, full evidence inspection, and read-only protection
in strict accordance with service layer lifecycle constraints.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from ui.service_helper import get_raid_service
from ui.components.evidence_viewer import render_evidence_box
from src.export.excel_export import export_approved_register


def render_page():
    st.set_page_config(page_title="Approved RAID Register - KTern.AI", page_icon="✅", layout="wide")
    st.title("✅ Official Approved RAID Register")
    st.markdown("Verified, PM-approved Risk, Assumption, Issue, Decision & Dependency items governing the SAP transformation programme.")
    st.divider()

    service = get_raid_service()
    approved_items = service.get_approved_items()

    if not approved_items:
        st.info("ℹ️ **Approved Register is Empty**: No RAID items have been approved yet.")
        st.markdown("👉 Head over to **1_RAID_Review** to inspect pending drafts and approve them into the official register.")
        return

    # --- Sidebar Filters & Search ---
    st.sidebar.markdown("### 🔍 Filter Register")

    categories = sorted(list(set(item.get("type", "Risk") for item in approved_items)))
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

    workstreams = sorted(list(set(item.get("workstream") or "General" for item in approved_items)))
    selected_workstreams = st.sidebar.multiselect("Workstream", workstreams, default=workstreams)

    phases = sorted(list(set(item.get("phase") or "Realize" for item in approved_items)))
    selected_phases = st.sidebar.multiselect("SAP Phase", phases, default=phases)

    statuses = sorted(list(set(item.get("status") or "Open" for item in approved_items)))
    selected_statuses = st.sidebar.multiselect("Register Status", statuses, default=statuses)

    search_query = st.sidebar.text_input("Search Title / Description / Owner", value="").strip().lower()

    # Apply filters
    filtered = [
        item for item in approved_items
        if item.get("type") in selected_categories
        and (item.get("workstream") or "General") in selected_workstreams
        and (item.get("phase") or "Realize") in selected_phases
        and (item.get("status") or "Open") in selected_statuses
        and (
            not search_query
            or search_query in (item.get("title") or "").lower()
            or search_query in (item.get("description") or "").lower()
            or search_query in (item.get("owner") or "").lower()
        )
    ]

    col_hdr, col_dl = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"**Displaying {len(filtered)} of {len(approved_items)} approved items**")
    with col_dl:
        excel_bytes = export_approved_register(filtered)
        st.download_button(
            label="📥 Download Excel Register",
            data=excel_bytes,
            file_name="Approved_RAID_Register.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # Overview Table
    if filtered:
        display_rows = []
        for item in filtered:
            display_rows.append({
                "Approved ID": item.get("id"),
                "Draft ID": item.get("draft_id"),
                "Type": item.get("type"),
                "Title": item.get("title"),
                "Workstream": item.get("workstream") or "General",
                "Phase": item.get("phase") or "Realize",
                "Severity/Impact": item.get("severity_or_impact") or "Medium",
                "Owner": item.get("owner") or "Unassigned",
                "Due Date": item.get("due_date") or "-",
                "Status": item.get("status") or "Open",
                "Approved By": item.get("approved_by"),
                "Approved At": (item.get("approved_at") or "")[:19]
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
    else:
        st.warning("No approved items match your current filter criteria.")
        return

    st.divider()

    # --- Item Detail Inspection ---
    st.subheader("🔎 Approved Record Inspection")
    item_map = {f"[{item['type'].upper()}] {item['id']} - {item['title'][:60]}": item['id'] for item in filtered}
    selected_label = st.selectbox("Select Approved Record to Inspect", list(item_map.keys()))
    selected_id = item_map[selected_label]
    record = next((i for i in approved_items if i["id"] == selected_id), None)

    if record:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Approved ID:** `{record.get('id')}`")
            st.write(f"**Originating Draft ID:** `{record.get('draft_id')}`")
            st.write(f"**RAID Category:** `{record.get('type')}`")
        with c2:
            st.write(f"**Workstream:** `{record.get('workstream') or 'General'}`")
            st.write(f"**SAP Phase:** `{record.get('phase') or 'Realize'}`")
            st.write(f"**Severity / Impact:** `{record.get('severity_or_impact') or 'Medium'}`")
        with c3:
            st.write(f"**Assigned Owner:** `{record.get('owner') or 'Unassigned'}`")
            st.write(f"**Target Due Date:** `{record.get('due_date') or 'Unset'}`")
            st.write(f"**Approved By:** `{record.get('approved_by')}` at `{(record.get('approved_at') or '')[:19]}`")

        st.markdown("**Full Description:**")
        st.info(record.get("description"))

        if record.get("mitigation_action"):
            st.markdown("**Mitigation / Agreed Action:**")
            st.write(record.get("mitigation_action"))

        # Evidence display
        render_evidence_box(record)

        st.info("🔒 **Governance Policy:** Approved register records are immutable through standard draft editing. Any status updates (e.g. In Progress, Closed) occur via formal program governance.")


if __name__ == "__main__":
    render_page()
