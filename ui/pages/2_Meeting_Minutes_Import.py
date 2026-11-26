"""
Page 7: Meeting Minutes Import
KTern.AI RAID Identification Agent POC

Dedicated page for uploading and ingesting meeting minutes (MOMs), processing them
through the end-to-end architectural pipeline, and reviewing detected RAID items:
- Risks (R)
- Assumptions (A)
- Issues (I)
- Decisions (D)
- Dependencies (DP)
Includes all 5 RAID categories: Risk, Assumption, Issue, Decision, and Dependency.

All generated items remain drafts / pending PM review. Never automatically approved.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import date, datetime

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from ui.service_helper import get_raid_service
from src.service import RaidValidationError, RaidServiceError

# Supported review categories across all 5 RAID types
REVIEW_CATEGORIES = ["Risk", "Assumption", "Issue", "Decision", "Dependency"]


def format_file_size(size_bytes: int) -> str:
    """Formats file size into human-readable string (B, KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def render_item_card(item: Dict[str, Any], service: Any):
    """Renders a single detected RAID item in the standard template card."""
    raid_id = item.get("id", "N/A")
    title = item.get("title", "Untitled RAID Item")
    conf = float(item.get("confidence_score", 0.0))
    conf_pct = f"{conf * 100:.0f}%"
    cat_type = item.get("type", "Risk")
    desc = item.get("description", "No description provided.")
    workstream = item.get("workstream") or "General"
    phase = item.get("phase") or "Realize"
    owner = item.get("likely_owner") or "Unassigned"
    due_date = item.get("due_date") or "—"
    dup_id = item.get("duplicate_of_id")

    # Resolve original source metadata (source file, meeting date, evidence excerpt)
    meta = service.get_source_metadata_for_draft(item)
    source_file = item.get("source_file") or meta.get("source_file") or "meeting_notes"
    meeting_date = item.get("meeting_date") or meta.get("meeting_date") or "—"
    evidence = item.get("evidence_text") or meta.get("evidence_text") or "No excerpt recorded."

    # Visual border styling per category
    category_colors = {
        "Risk": "#C93B3B",
        "Assumption": "#D97706",
        "Issue": "#DC2626",
        "Decision": "#2563EB",
        "Dependency": "#7C3AED"
    }
    border_color = category_colors.get(cat_type, "#9A1C1F")

    with st.container():
        st.markdown(
            f"""
            <div style="border-left: 4px solid {border_color}; padding: 12px 16px; background-color: #FAFAFA; border-radius: 4px; margin-bottom: 12px; border-top: 1px solid #EEEEEE; border-right: 1px solid #EEEEEE; border-bottom: 1px solid #EEEEEE;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 1.05rem; font-weight: 700; color: #1C1C1C;">
                        <code>{raid_id}</code> &nbsp; {title}
                    </span>
                    <span style="background-color: rgba(154, 28, 31, 0.1); color: #9A1C1F; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">
                        {conf_pct} Confidence
                    </span>
                </div>
                <p style="color: #4B5563; font-size: 0.93rem; margin-bottom: 10px;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Item Metadata Columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption(f"**Workstream:** `{workstream}`")
        with c2:
            st.caption(f"**Project Phase:** `{phase}`")
        with c3:
            st.caption(f"**Owner:** `{owner}`")
        with c4:
            st.caption(f"**Due Date:** `{due_date}`")

        # Specific scoring fields for Risk and Issue
        if cat_type == "Risk":
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                st.caption(f"**Probability:** `{item.get('probability') or 'Medium'}`")
            with r2:
                st.caption(f"**Impact:** `{item.get('impact') or 'Medium'}`")
            with r3:
                st.caption(f"**Mitigation:** {item.get('mitigation_action') or 'Review with workstream lead'}")
        elif cat_type == "Issue":
            i1, i2 = st.columns([1, 3])
            with i1:
                st.caption(f"**Severity:** `{item.get('severity') or 'Medium'}`")
            with i2:
                st.caption(f"**Mitigation:** {item.get('mitigation_action') or 'Investigate root cause'}")

        # Evidence & Traceability Excerpt Box
        with st.expander("🔍 Source Meeting & Evidence Excerpt", expanded=False):
            st.caption(f"**Source Document / Meeting:** `{source_file}` &nbsp;|&nbsp; **Meeting Date:** `{meeting_date}` &nbsp;|&nbsp; **Source ID:** `{item.get('evidence_source_id', 'N/A')}`")
            st.info(f"❝ {evidence} ❞")
            if dup_id:
                st.warning(f"⚠️ **Potential Duplicate Alert:** Similarity detected with existing item `{dup_id}`.")
            st.caption("ℹ️ *Item is saved as Draft in review queue. Review and approve in RAID Review.*")


def render_page():
    st.set_page_config(
        page_title="Meeting Minutes Import - KTern.AI",
        page_icon="📝",
        layout="wide"
    )

    st.title("📝 Meeting Minutes Import")
    st.markdown(
        "Upload meeting minutes (MOMs) or paste notes directly to extract **Risks, Assumptions, Issues, and Decisions** "
        "into the PM review queue. All items remain drafts awaiting formal approval."
    )
    st.divider()

    service = get_raid_service()

    # Session State initialisation
    if "mom_results" not in st.session_state:
        st.session_state.mom_results = None

    # Tabs for input modes: File Upload vs Manual Text Input
    tab_upload, tab_manual = st.tabs([
        "📁 File Upload (.docx, .txt, .csv)",
        "✍️ Manual Text Input"
    ])

    files_to_process = []
    manual_entry_payload = None
    trigger_process = False

    # -----------------------------------------------------------------------
    # TAB 1: File Upload
    # -----------------------------------------------------------------------
    with tab_upload:
        st.subheader("Upload Meeting Minutes Documents")
        st.markdown("Select one or more `.docx`, `.txt`, or `.csv` meeting notes files.")

        uploaded_files = st.file_uploader(
            "Upload MOM Files",
            type=["docx", "txt", "csv"],
            accept_multiple_files=True,
            help="Supported formats: Microsoft Word (.docx), Plain Text (.txt), CSV tabular exports (.csv)"
        )

        if uploaded_files:
            st.markdown("##### 📄 Uploaded Files Summary")
            file_records = []
            for f in uploaded_files:
                f_size = format_file_size(f.size)
                f_ext = Path(f.name).suffix.lower()
                status = "Ready to Process" if f.size > 0 else "Empty File (0 Bytes)"
                file_records.append({
                    "File Name": f.name,
                    "Size": f_size,
                    "Format": f_ext,
                    "Status": status
                })
            df_files = pd.DataFrame(file_records)
            st.dataframe(df_files, use_container_width=True, hide_index=True)
            st.caption("💡 *You can remove any file before processing using the '✕' icon next to its name above.*")

            col_btn1, _ = st.columns([1, 2])
            with col_btn1:
                if st.button("🚀 Process Meeting Minutes", type="primary", use_container_width=True, key="btn_process_files"):
                    files_to_process = uploaded_files
                    trigger_process = True
        else:
            st.info("ℹ️ No files selected yet. Drag and drop or browse files above.")

    # -----------------------------------------------------------------------
    # TAB 2: Manual Text Input
    # -----------------------------------------------------------------------
    with tab_manual:
        st.subheader("Paste Meeting Minutes Content")
        st.markdown("Directly paste discussion notes, transcripts, or meeting minutes into the text area below.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            manual_title = st.text_input(
                "Meeting Title *",
                value="",
                placeholder="e.g. SteerCo Wave 1 Scope & Architecture Alignment"
            )
            project_name = st.text_input(
                "Project Name (Optional)",
                value="",
                placeholder="e.g. S/4HANA Finance Conversion"
            )
        with col_m2:
            manual_date = st.date_input(
                "Meeting Date *",
                value=date.today()
            )

        manual_content = st.text_area(
            "Meeting Minutes Content *",
            height=200,
            placeholder="Paste discussion points, decisions, concerns, and actions here...\n\nExample:\n- Agreed that SD pricing user exit refactoring is high risk and may delay end-to-end cycle test.\n- Decided to proceed with brownfield migration approach for ECC financials rather than greenfield.\n- Planning assumption: Client will complete master data prep before SIT phase."
        )

        col_btn2, _ = st.columns([1, 2])
        with col_btn2:
            if st.button("🚀 Process Pasted Meeting Minutes", type="primary", use_container_width=True, key="btn_process_manual"):
                if not manual_title.strip():
                    st.warning("⚠️ Please provide a Meeting Title before processing.")
                elif not manual_content.strip():
                    st.warning("⚠️ Please paste the Meeting Minutes Content before processing.")
                else:
                    manual_entry_payload = {
                        "title": manual_title.strip(),
                        "date": str(manual_date),
                        "project_name": project_name.strip() if project_name else None,
                        "content": manual_content.strip()
                    }
                    trigger_process = True

    # -----------------------------------------------------------------------
    # PROCESSING EXECUTION WITH LIVE STAGES
    # -----------------------------------------------------------------------
    if trigger_process:
        st.divider()
        st.subheader("⚙️ Ingestion & Processing Pipeline")

        status_container = st.status("Initializing meeting minutes processing...", expanded=True)

        stage_logs = []

        def ui_stage_callback(stage_name: str, details: str):
            stage_logs.append((stage_name, details))
            status_container.write(f"**[{stage_name}]** {details}")

        try:
            results = service.process_meeting_minutes(
                files=files_to_process if files_to_process else None,
                manual_entry=manual_entry_payload,
                stage_callback=ui_stage_callback
            )
            st.session_state.mom_results = results
            status_container.update(
                label=f"✅ Processing Completed: {results.get('drafts_created', 0)} RAID items identified",
                state="complete",
                expanded=False
            )
            st.success(
                f"🎉 **Processing Complete!** Identified **{results.get('drafts_created', 0)}** RAID draft items. "
                f"All items have been queued in Draft status."
            )
        except RaidValidationError as val_exc:
            status_container.update(label="❌ Validation Failed", state="error", expanded=True)
            st.error(f"⚠️ **Validation Error:** {val_exc}")
        except RaidServiceError as srv_exc:
            status_container.update(label="❌ Processing Error", state="error", expanded=True)
            st.error(f"🚨 **Service Error:** {srv_exc}")
        except Exception as exc:
            status_container.update(label="❌ Processing Error", state="error", expanded=True)
            st.error(f"🚨 **Unexpected Error:** {exc}")

    # -----------------------------------------------------------------------
    # RESULTS SECTION (FOUR SEPARATE CATEGORIES: R, A, I, D)
    # -----------------------------------------------------------------------
    if st.session_state.mom_results:
        results = st.session_state.mom_results
        drafts: List[Dict[str, Any]] = results.get("drafts", [])
        skipped_sources = results.get("skipped_sources", [])
        errors = results.get("errors", [])

        st.divider()

        # Duplicate source alerts if any
        if skipped_sources:
            for s in skipped_sources:
                st.warning(
                    f"⚠️ **Duplicate Source Skipped:** `{s.get('source')}` was previously processed. "
                    "Skipped re-ingestion to prevent duplicate records."
                )

        if errors:
            for err in errors:
                st.info(f"ℹ️ {err}")

        # Summary KPIs
        st.subheader("📊 Detected RAID Items Summary")
        k1, k2, k3, k4, k5 = st.columns(5)

        # Segregate into 5 categories
        risks = [d for d in drafts if d.get("type") == "Risk"]
        assumptions = [d for d in drafts if d.get("type") == "Assumption"]
        issues = [d for d in drafts if d.get("type") == "Issue"]
        decisions = [d for d in drafts if d.get("type") == "Decision"]
        dependencies = [d for d in drafts if d.get("type") == "Dependency"]

        with k1:
            st.metric("Risks Identified", len(risks))
        with k2:
            st.metric("Assumptions Identified", len(assumptions))
        with k3:
            st.metric("Issues Identified", len(issues))
        with k4:
            st.metric("Decisions Identified", len(decisions))
        with k5:
            st.metric("Dependencies Identified", len(dependencies))

        # Five Separate Display Sections
        st.markdown("### 📋 Review Detected RAID Items")

        cat_tab_r, cat_tab_a, cat_tab_i, cat_tab_d, cat_tab_dep = st.tabs([
            f"🔴 R – Risks ({len(risks)})",
            f"🟡 A – Assumptions ({len(assumptions)})",
            f"🟠 I – Issues ({len(issues)})",
            f"🔵 D – Decisions ({len(decisions)})",
            f"🟣 DEP – Dependencies ({len(dependencies)})"
        ])

        with cat_tab_r:
            if not risks:
                st.info("ℹ️ No Risks identified from the meeting minutes.")
            else:
                for item in risks:
                    render_item_card(item, service)

        with cat_tab_a:
            if not assumptions:
                st.info("ℹ️ No Assumptions identified from the meeting minutes.")
            else:
                for item in assumptions:
                    render_item_card(item, service)

        with cat_tab_i:
            if not issues:
                st.info("ℹ️ No Issues identified from the meeting minutes.")
            else:
                for item in issues:
                    render_item_card(item, service)

        with cat_tab_d:
            if not decisions:
                st.info("ℹ️ No Decisions identified from the meeting minutes.")
            else:
                for item in decisions:
                    render_item_card(item, service)

        with cat_tab_dep:
            if not dependencies:
                st.info("ℹ️ No Dependencies identified from the meeting minutes.")
            else:
                for item in dependencies:
                    render_item_card(item, service)

        # Callout and Navigation Link to RAID Review Queue
        st.markdown("---")
        nav_col1, nav_col2 = st.columns([3, 2])
        with nav_col1:
            st.success(
                "🛡️ **Draft Safety Guarantee:** All generated items remain **Drafts / Pending Review**. "
                "Never automatically approved. Human PM review is mandatory before items enter the official register."
            )
        with nav_col2:
            st.markdown(
                """
                <div style="padding: 10px; background-color: #F8F9FA; border: 1px solid #E5E7EB; border-radius: 6px; text-align: center;">
                    <a href="1_RAID_Review" target="_self" style="font-size: 1.05rem; font-weight: 700; color: #9A1C1F; text-decoration: none;">
                        👉 Navigate to RAID Review Queue &rarr;
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Streamlit page_link if supported in multi-page environment
            try:
                st.page_link("pages/1_RAID_Review.py", label="Open RAID Review Page", icon="📋")
            except Exception:
                pass


render_meeting_import = render_page

if __name__ == "__main__":
    render_page()
