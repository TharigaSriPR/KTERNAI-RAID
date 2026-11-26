"""
Page 1: RAID Review Queue
KTern.AI RAID Identification Agent POC

Human-in-the-loop review page strictly aligned with the Suggested RAID Item Template
from the RAID Identification Agent Problem Statement PDF:
1. RAID Overview: 5 separate tables (Risk, Assumption, Issue, Decision, Dependency).
   - Includes all 5 RAID categories: Risk, Assumption, Issue, Decision, and Dependency.
   - Assumption is strictly distinguished from Action.
   - Only relevant scoring fields are shown for each category.
2. Individual Review:
   - Category selector followed by actual line items.
   - Complete item details in the Suggested RAID Item Template format.
   - Original source evidence inspection.
   - Strictly single-item Approve, Reject, and Edit actions routing through RaidService.
   - Business status mapping: Draft, Open, In Progress, Closed, Rejected.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from config import RAID_TYPES, WORKSTREAMS, SAP_PHASES
from ui.service_helper import get_raid_service
from ui.components.evidence_viewer import render_evidence_box
from src.service import (
    RaidServiceError,
    RaidNotFoundError,
    InvalidRaidStatusError,
    RaidValidationError,
    DuplicateApprovalError,
    AuditPersistenceError
)

# Supported categories across all 5 RAID types
RAID_CATEGORIES = ["Risk", "Assumption", "Issue", "Decision", "Dependency"]

# Probability & Impact levels for Risk
PROBABILITY_LEVELS = ["Low", "Medium", "High"]
IMPACT_LEVELS = ["Low", "Medium", "High"]

# Severity levels for Issue
SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]


def map_business_status(raw_status: str) -> str:
    """
    Maps backend database lifecycle status to the Problem Statement PDF template business statuses:
    Draft, Open, In Progress, Closed, Rejected.
    
    Pending review items in this queue are drafts awaiting approval ('Draft').
    """
    if not raw_status:
        return "Draft"
    st_clean = str(raw_status).strip()
    if st_clean in ("Draft", "Pending_Review"):
        return "Draft"
    if st_clean == "Modified":
        return "Draft (Modified)"
    if st_clean in ("Accepted", "Approved"):
        return "Open"
    if st_clean == "Rejected":
        return "Rejected"
    if st_clean in ("In Progress", "Closed"):
        return st_clean
    return st_clean


def build_risk_table_data(drafts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Builds dataframe for R – Risk table with relevant scoring (Probability & Impact)."""
    rows = []
    for d in drafts:
        rows.append({
            "ID": d.get("id"),
            "Title": d.get("title"),
            "Workstream": d.get("workstream") or "General",
            "Phase": d.get("phase") or "Realize",
            "Probability": d.get("probability") or "Medium",
            "Impact": d.get("impact") or "Medium",
            "Owner": d.get("likely_owner") or "Unassigned",
            "Due Date": d.get("due_date") or "—",
            "Mitigation / Next Action": d.get("mitigation_action") or "—",
            "Status": map_business_status(d.get("status")),
            "Confidence": f"{float(d.get('confidence_score', 0)) * 100:.0f}%",
            "Detected On": str(d.get("detected_at", ""))[:19]
        })
    return pd.DataFrame(rows)


def build_assumption_table_data(drafts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds dataframe for A – Assumption table.
    Scoring fields (Probability, Impact, Severity) and Mitigation Action are excluded
    to prevent confusing Assumption with Action.
    """
    rows = []
    for d in drafts:
        rows.append({
            "ID": d.get("id"),
            "Title": d.get("title"),
            "Description": d.get("description"),
            "Workstream": d.get("workstream") or "General",
            "Phase": d.get("phase") or "Realize",
            "Owner": d.get("likely_owner") or "Unassigned",
            "Due Date": d.get("due_date") or "—",
            "Status": map_business_status(d.get("status")),
            "Confidence": f"{float(d.get('confidence_score', 0)) * 100:.0f}%",
            "Detected On": str(d.get("detected_at", ""))[:19]
        })
    return pd.DataFrame(rows)


def build_issue_table_data(drafts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds dataframe for I – Issue table with relevant scoring (Severity).
    Probability is excluded as issues are realized occurrences.
    """
    rows = []
    for d in drafts:
        rows.append({
            "ID": d.get("id"),
            "Title": d.get("title"),
            "Workstream": d.get("workstream") or "General",
            "Phase": d.get("phase") or "Realize",
            "Severity": d.get("severity") or d.get("impact") or "Medium",
            "Owner": d.get("likely_owner") or "Unassigned",
            "Due Date": d.get("due_date") or "—",
            "Mitigation / Next Action": d.get("mitigation_action") or "—",
            "Status": map_business_status(d.get("status")),
            "Confidence": f"{float(d.get('confidence_score', 0)) * 100:.0f}%",
            "Detected On": str(d.get("detected_at", ""))[:19]
        })
    return pd.DataFrame(rows)


def build_decision_table_data(drafts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds dataframe for D – Decision table.
    Probability, Impact, Severity are excluded as decisions record choices already agreed.
    """
    rows = []
    for d in drafts:
        rows.append({
            "ID": d.get("id"),
            "Title": d.get("title"),
            "Description": d.get("description"),
            "Workstream": d.get("workstream") or "General",
            "Phase": d.get("phase") or "Realize",
            "Decided By / Owner": d.get("likely_owner") or "Unassigned",
            "Due Date": d.get("due_date") or "—",
            "Status": map_business_status(d.get("status")),
            "Confidence": f"{float(d.get('confidence_score', 0)) * 100:.0f}%",
            "Detected On": str(d.get("detected_at", ""))[:19]
        })
    return pd.DataFrame(rows)


def build_dependency_table_data(drafts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds dataframe for DEP – Dependency table.
    Includes prerequisite ownership, target dates, and mitigation actions.
    """
    rows = []
    for d in drafts:
        rows.append({
            "ID": d.get("id"),
            "Title": d.get("title"),
            "Workstream": d.get("workstream") or "General",
            "Phase": d.get("phase") or "Realize",
            "Owner": d.get("likely_owner") or "Unassigned",
            "Due Date": d.get("due_date") or "—",
            "Mitigation / Next Action": d.get("mitigation_action") or "—",
            "Status": map_business_status(d.get("status")),
            "Confidence": f"{float(d.get('confidence_score', 0)) * 100:.0f}%",
            "Detected On": str(d.get("detected_at", ""))[:19]
        })
    return pd.DataFrame(rows)


def render_low_confidence_section(service: Any, candidates: List[Dict[str, Any]]):
    """
    Renders the Low-Confidence Candidates Review Queue (Stage 6C).
    Allows human reviewers to:
    1. Confirm as RAID (creates a Draft in raid_drafts, never approved automatically).
    2. Mark as NOT_RAID.
    3. Defer / Request secondary review.
    4. View original source content in full.
    """
    st.subheader("⚠️ Low-Confidence Candidates")
    st.caption(
        "Signals evaluated below the automated classification confidence threshold (0.75) "
        "or deferred for human review. These are NOT approved RAID items and are not in the official register."
    )

    if not candidates:
        st.info("No low-confidence candidates currently requiring human review.")
        return

    st.markdown(f"**{len(candidates)} Candidate(s) Awaiting Decision**")

    # Quick overview dataframe
    summary_rows = []
    for c in candidates:
        summary_rows.append({
            "Signal ID": c.get("signal_id"),
            "Source": c.get("source_type"),
            "Author / Sender": c.get("author") or "Unknown",
            "Subject": c.get("subject") or "—",
            "Predicted Category": c.get("predicted_category") or "Risk",
            "Model Confidence": f"{float(c.get('confidence_score', 0)) * 100:.1f}%",
            "Status": c.get("status") or "BELOW_CONFIDENCE_THRESHOLD",
            "Detected": str(c.get("timestamp", ""))[:19]
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Candidate Inspection & Disposition")

    # Selector for candidate
    cand_options = {
        f"[{c['signal_id']}] {c.get('subject') or c.get('title') or c['signal_id']} (Conf: {float(c.get('confidence_score', 0))*100:.0f}%, {c.get('predicted_category', 'Risk')})": c['signal_id']
        for c in candidates
    }
    selected_label = st.selectbox(
        "Select Candidate Signal to Inspect:",
        options=list(cand_options.keys()),
        key="low_conf_cand_selector"
    )
    sel_sig_id = cand_options[selected_label]
    cand = next((c for c in candidates if c["signal_id"] == sel_sig_id), None)
    if not cand:
        st.error("Selected candidate not found.")
        return

    # Details card
    with st.container(border=True):
        status_label = cand.get("status", "BELOW_CONFIDENCE_THRESHOLD")
        st.markdown(f"### `[{cand.get('predicted_category', 'Risk').upper()}]` {cand.get('title') or cand.get('subject')}")
        st.caption(f"Status: **{status_label}** | Signal ID: `{cand.get('signal_id')}`")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Source Type:** `{cand.get('source_type')}`")
            st.markdown(f"**Sender / Author:** `{cand.get('author') or 'Unknown'}`")
            st.markdown(f"**Subject:** {cand.get('subject') or '—'}")
        with c2:
            st.markdown(f"**Workstream:** `{cand.get('workstream') or 'General'}`")
            st.markdown(f"**SAP Phase:** `{cand.get('phase') or 'Realize'}`")
            st.markdown(f"**Timestamp:** `{cand.get('timestamp')}`")
        with c3:
            st.markdown(f"**Owner (Extracted):** `{cand.get('likely_owner') or 'Unassigned'}`")
            st.markdown(f"**Due Date (Extracted):** `{cand.get('due_date') or 'Unset'}`")
            conf_val = float(cand.get("confidence_score", 0.0))
            st.markdown(f"**Model Confidence:** `{conf_val * 100:.1f}%` ({conf_val:.3f})")
        with c4:
            st.markdown(f"**Provider:** `{cand.get('llm_provider') or 'OpenRouter'}`")
            st.markdown(f"**Model:** `{cand.get('llm_model') or 'Nemotron-3-Ultra'}`")
            st.markdown(f"**Evidence Status:** `{cand.get('evidence_status') or 'verified'}`")

        # 4. View Original Source Content
        with st.expander("📄 View Original Source Content", expanded=False):
            st.markdown(f"**Original Subject:** {cand.get('subject') or '—'}")
            st.markdown(f"**Sender / Author:** {cand.get('author') or 'Unknown'}")
            st.markdown(f"**Timestamp:** {cand.get('timestamp') or '—'}")
            st.markdown("**Raw Content:**")
            st.text_area(
                "Original Content",
                value=cand.get("content") or "",
                height=150,
                disabled=True,
                key=f"raw_content_{cand['signal_id']}"
            )

        # Supporting Evidence & Reasoning
        st.markdown("**Supporting Evidence (from LLM):**")
        st.info(cand.get("evidence_text") or "No explicit evidence captured.")

        if cand.get("reasoning_summary"):
            st.markdown("**LLM Reasoning:**")
            st.caption(cand.get("reasoning_summary"))

    # Disposition Actions
    st.markdown("##### Reviewer Actions")
    tab_confirm, tab_reject_sig, tab_defer = st.tabs([
        "✅ 1. Confirm as RAID Draft",
        "❌ 2. Mark as NOT_RAID",
        "⏸️ 3. Request Manual Review / Defer"
    ])

    with tab_confirm:
        st.caption(
            "Creates a new RAID draft with status 'Draft'. "
            "The item must be approved through the normal review queue before entering the official register."
        )
        with st.form(key=f"confirm_form_{cand['signal_id']}"):
            rev_name = st.text_input(
                "Reviewer Identity (PM Name / ID) *",
                value="",
                placeholder="e.g. John Doe (PMO Lead)",
                key=f"rev_name_conf_{cand['signal_id']}"
            ).strip()

            st.markdown("**Editable Extraction Fields**")
            cf_title = st.text_input("Title *", value=cand.get("title") or cand.get("subject") or "")
            cf_desc = st.text_area("Description *", value=cand.get("description") or cand.get("content") or "")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                cur_cat = cand.get("predicted_category") or "Risk"
                cat_idx = RAID_CATEGORIES.index(cur_cat) if cur_cat in RAID_CATEGORIES else 0
                cf_type = st.selectbox("RAID Category *", RAID_CATEGORIES, index=cat_idx, key=f"cat_sel_{cand['signal_id']}")
                cf_workstream = st.text_input("Workstream", value=cand.get("workstream") or "General", key=f"ws_{cand['signal_id']}")
            with col_b:
                cur_phase = cand.get("phase") or "Realize"
                ph_idx = SAP_PHASES.index(cur_phase) if cur_phase in SAP_PHASES else 2
                cf_phase = st.selectbox("SAP Phase", SAP_PHASES, index=ph_idx, key=f"ph_{cand['signal_id']}")
                cf_owner = st.text_input("Owner (leave empty if unassigned)", value=cand.get("likely_owner") or "", key=f"owner_{cand['signal_id']}")
            with col_c:
                cf_due = st.text_input("Due Date", value=cand.get("due_date") or "", key=f"due_{cand['signal_id']}")
                if cf_type == "Risk":
                    cur_p = cand.get("probability") or "Medium"
                    p_idx = PROBABILITY_LEVELS.index(cur_p) if cur_p in PROBABILITY_LEVELS else 1
                    cf_prob = st.selectbox("Probability", PROBABILITY_LEVELS, index=p_idx, key=f"prob_{cand['signal_id']}")
                    cur_i = cand.get("impact") or "Medium"
                    i_idx = IMPACT_LEVELS.index(cur_i) if cur_i in IMPACT_LEVELS else 1
                    cf_imp = st.selectbox("Impact", IMPACT_LEVELS, index=i_idx, key=f"imp_{cand['signal_id']}")
                    cf_sev = None
                elif cf_type == "Issue":
                    cur_s = cand.get("severity") or cand.get("impact") or "Medium"
                    s_idx = SEVERITY_LEVELS.index(cur_s) if cur_s in SEVERITY_LEVELS else 1
                    cf_sev = st.selectbox("Severity", SEVERITY_LEVELS, index=s_idx, key=f"sev_{cand['signal_id']}")
                    cf_prob = None
                    cf_imp = cf_sev
                else:
                    cf_prob = None
                    cf_imp = None
                    cf_sev = None

            cf_mitigation = st.text_area(
                "Mitigation / Next Action",
                value=cand.get("mitigation_action") or "",
                key=f"mit_{cand['signal_id']}"
            )
            cf_comments = st.text_input("Confirmation Comments / Rationale", value="", key=f"comm_{cand['signal_id']}")

            submit_conf = st.form_submit_button("✅ Confirm & Create RAID Draft")
            if submit_conf:
                if not rev_name:
                    st.error("Reviewer Identity is required.")
                elif not cf_title.strip():
                    st.error("Title cannot be empty.")
                elif not cf_desc.strip():
                    st.error("Description cannot be empty.")
                else:
                    updates = {
                        "type": cf_type,
                        "title": cf_title.strip(),
                        "description": cf_desc.strip(),
                        "workstream": cf_workstream.strip(),
                        "phase": cf_phase,
                        "likely_owner": cf_owner.strip() or None,
                        "due_date": cf_due.strip(),
                        "mitigation_action": cf_mitigation.strip(),
                    }
                    if cf_type == "Risk":
                        updates["probability"] = cf_prob
                        updates["impact"] = cf_imp
                    elif cf_type == "Issue":
                        updates["severity"] = cf_sev
                        updates["impact"] = cf_sev

                    try:
                        res = service.confirm_low_confidence_candidate(
                            signal_id=cand["signal_id"],
                            reviewer_name=rev_name,
                            updates=updates,
                            comments=cf_comments or None
                        )
                        st.success(f"Confirmed candidate '{cand['signal_id']}' as RAID draft '{res['id']}' with status 'Draft'.")
                        st.rerun()
                    except DuplicateApprovalError as exc:
                        st.error(f"Duplicate Error: {exc}")
                    except RaidValidationError as exc:
                        st.error(f"Validation Error: {exc}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    with tab_reject_sig:
        st.caption("Marks this signal as NOT_RAID. No draft is created, and the signal is excluded from future queues.")
        with st.form(key=f"reject_form_{cand['signal_id']}"):
            rej_name = st.text_input(
                "Reviewer Identity (PM Name / ID) *",
                value="",
                placeholder="e.g. John Doe (PMO Lead)",
                key=f"rej_name_{cand['signal_id']}"
            ).strip()
            rej_reason = st.text_area(
                "Reason for Marking NOT_RAID *",
                value="Not a project RAID item (operational update / routine communication).",
                key=f"rej_reason_{cand['signal_id']}"
            ).strip()
            submit_rej = st.form_submit_button("❌ Mark as NOT_RAID")
            if submit_rej:
                if not rej_name:
                    st.error("Reviewer Identity is required.")
                elif not rej_reason:
                    st.error("Rejection reason is required.")
                else:
                    try:
                        service.reject_low_confidence_candidate(
                            signal_id=cand["signal_id"],
                            reviewer_name=rej_name,
                            rejection_reason=rej_reason
                        )
                        st.success(f"Candidate '{cand['signal_id']}' marked as NOT_RAID.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    with tab_defer:
        st.caption("Defers decision on this candidate for secondary review. The candidate remains visible in the review queue.")
        with st.form(key=f"defer_form_{cand['signal_id']}"):
            def_name = st.text_input(
                "Reviewer Identity (PM Name / ID) *",
                value="",
                placeholder="e.g. John Doe (PMO Lead)",
                key=f"def_name_{cand['signal_id']}"
            ).strip()
            def_notes = st.text_area(
                "Deferral Notes / Reason *",
                value="Requires team clarification or further investigation.",
                key=f"def_notes_{cand['signal_id']}"
            ).strip()
            submit_def = st.form_submit_button("⏸️ Defer Candidate")
            if submit_def:
                if not def_name:
                    st.error("Reviewer Identity is required.")
                elif not def_notes:
                    st.error("Deferral notes are required.")
                else:
                    try:
                        service.defer_low_confidence_candidate(
                            signal_id=cand["signal_id"],
                            reviewer_name=def_name,
                            deferral_notes=def_notes
                        )
                        st.success(f"Candidate '{cand['signal_id']}' deferred for secondary review.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")


def render_page():
    st.set_page_config(
        page_title="RAID Review Queue - KTern.AI",
        page_icon="📋",
        layout="wide"
    )

    # Header with KaarTech branding
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "kaartech_logo.png"
    if logo_path.exists():
        head_col1, head_col2 = st.columns([1, 8])
        with head_col1:
            st.image(str(logo_path), width=75)
        with head_col2:
            st.title("📋 RAID Review Queue")
    else:
        st.title("📋 RAID Review Queue")

    st.caption("PMO Human-in-the-Loop Review strictly aligned with the Suggested RAID Item Template (R – Risk, A – Assumption, I – Issue, D – Decision, DEP – Dependency).")
    st.divider()

    service = get_raid_service()
    raw_drafts = service.get_pending_drafts()

    drafts = [d for d in raw_drafts if d.get("type") in RAID_CATEGORIES]
    low_conf_candidates = service.get_low_confidence_candidates()

    if not drafts and not low_conf_candidates:
        st.success("🎉 All caught up! No pending RAID drafts or low-confidence candidates in the review queue.")
        st.info("When new signals are ingested through project communication channels, new drafts will appear here.")
        return

    if not drafts:
        st.info("ℹ️ No high-confidence RAID drafts pending approval.")

def render_high_confidence_drafts(service: Any, drafts: List[Dict[str, Any]]):
    """
    Renders the high-confidence pending drafts review queue with 5 category tables,
    individual template inspection, and approve/reject/modify actions.
    """
    # --- Sidebar Filtering Controls ---
    st.sidebar.markdown("### 🔍 Filter Review Queue")
    
    available_workstreams = sorted(list(set(d.get("workstream") or "General" for d in drafts)))
    selected_workstreams = st.sidebar.multiselect("Workstream", available_workstreams, default=available_workstreams)

    available_phases = sorted(list(set(d.get("phase") or "Realize" for d in drafts)))
    selected_phases = st.sidebar.multiselect("SAP Phase", available_phases, default=available_phases)

    min_conf = st.sidebar.slider("Minimum Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    search_query = st.sidebar.text_input("Search Keyword", value="").strip().lower()

    # Apply filters
    filtered_drafts = [
        d for d in drafts
        if (d.get("workstream") or "General") in selected_workstreams
        and (d.get("phase") or "Realize") in selected_phases
        and float(d.get("confidence_score") or 0.0) >= min_conf
        and (
            not search_query
            or search_query in (d.get("title") or "").lower()
            or search_query in (d.get("description") or "").lower()
            or search_query in (d.get("id") or "").lower()
        )
    ]

    # Split filtered drafts into the 5 categories
    risk_drafts = [d for d in filtered_drafts if d.get("type") == "Risk"]
    assumption_drafts = [d for d in filtered_drafts if d.get("type") == "Assumption"]
    issue_drafts = [d for d in filtered_drafts if d.get("type") == "Issue"]
    decision_drafts = [d for d in filtered_drafts if d.get("type") == "Decision"]
    dependency_drafts = [d for d in filtered_drafts if d.get("type") == "Dependency"]

    # Top KPI Metrics for the 5 RAID categories
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="🔴 Risks (R)", value=len(risk_drafts), help="Potential future events with negative impact")
    with m2:
        st.metric(label="🟡 Assumptions (A)", value=len(assumption_drafts), help="Conditions presumed true without immediate proof")
    with m3:
        st.metric(label="🟠 Issues (I)", value=len(issue_drafts), help="Current unresolved problems requiring intervention")
    with m4:
        st.metric(label="🔵 Decisions (D)", value=len(decision_drafts), help="Official conclusions or agreed trade-offs recorded")
    with m5:
        st.metric(label="🟣 Dependencies (DEP)", value=len(dependency_drafts), help="Prerequisites and external/internal dependencies")

    st.divider()

    # =========================================================================
    # 1. RAID Overview: Five Separate Tables
    # =========================================================================
    st.subheader("📊 1. RAID Overview (5 Category Tables)")
    st.caption("Structured view of all pending drafts by RAID category. Each table presents only category-relevant scoring fields.")

    # Table 1: R – Risk
    with st.expander(f"🔴 R – Risk ({len(risk_drafts)} pending)", expanded=True):
        if risk_drafts:
            df_risk = build_risk_table_data(risk_drafts)
            st.dataframe(df_risk, use_container_width=True, hide_index=True)
        else:
            st.info("No pending Risk items matching filter criteria.")

    # Table 2: A – Assumption
    with st.expander(f"🟡 A – Assumption ({len(assumption_drafts)} pending)", expanded=True):
        if assumption_drafts:
            df_assumption = build_assumption_table_data(assumption_drafts)
            st.dataframe(df_assumption, use_container_width=True, hide_index=True)
        else:
            st.info("No pending Assumption items matching filter criteria.")

    # Table 3: I – Issue
    with st.expander(f"🟠 I – Issue ({len(issue_drafts)} pending)", expanded=True):
        if issue_drafts:
            df_issue = build_issue_table_data(issue_drafts)
            st.dataframe(df_issue, use_container_width=True, hide_index=True)
        else:
            st.info("No pending Issue items matching filter criteria.")

    # Table 4: D – Decision
    with st.expander(f"🔵 D – Decision ({len(decision_drafts)} pending)", expanded=True):
        if decision_drafts:
            df_decision = build_decision_table_data(decision_drafts)
            st.dataframe(df_decision, use_container_width=True, hide_index=True)
        else:
            st.info("No pending Decision items matching filter criteria.")

    # Table 5: DEP – Dependency
    with st.expander(f"🟣 DEP – Dependency ({len(dependency_drafts)} pending)", expanded=True):
        if dependency_drafts:
            df_dependency = build_dependency_table_data(dependency_drafts)
            st.dataframe(df_dependency, use_container_width=True, hide_index=True)
        else:
            st.info("No pending Dependency items matching filter criteria.")

    st.divider()

    # =========================================================================
    # 2. Individual Review & Template Inspection
    # =========================================================================
    st.subheader("🔎 2. Individual RAID Item Review")
    st.caption("Select a category and line item to inspect complete details in the Suggested RAID Item Template format with full evidence traceability.")

    # Category Selector
    category_counts = {
        "Risk": len(risk_drafts),
        "Assumption": len(assumption_drafts),
        "Issue": len(issue_drafts),
        "Decision": len(decision_drafts),
        "Dependency": len(dependency_drafts)
    }
    category_options = [
        f"Risk ({category_counts['Risk']})",
        f"Assumption ({category_counts['Assumption']})",
        f"Issue ({category_counts['Issue']})",
        f"Decision ({category_counts['Decision']})",
        f"Dependency ({category_counts['Dependency']})"
    ]

    sel_cat_str = st.radio(
        "Select RAID Category:",
        options=category_options,
        horizontal=True
    )
    selected_category = sel_cat_str.split(" ")[0]

    # Category drafts pool
    cat_drafts_map = {
        "Risk": risk_drafts,
        "Assumption": assumption_drafts,
        "Issue": issue_drafts,
        "Decision": decision_drafts,
        "Dependency": dependency_drafts
    }
    active_category_drafts = cat_drafts_map.get(selected_category, [])

    if not active_category_drafts:
        st.warning(f"No pending {selected_category} line items available for review.")
        return

    # Line Item Selector
    item_options = {
        f"[{d['id']}] {d['title'][:70]}": d['id']
        for d in active_category_drafts
    }
    selected_label = st.selectbox(
        f"Select {selected_category} Line Item to Review:",
        options=list(item_options.keys())
    )
    selected_id = item_options[selected_label]
    item = next((d for d in active_category_drafts if d["id"] == selected_id), None)

    if not item:
        st.error("Selected item not found.")
        return

    # -------------------------------------------------------------------------
    # Render Template Details strictly following PDF Suggested RAID Item Template
    # -------------------------------------------------------------------------
    st.markdown("#### Suggested RAID Item Template")
    
    with st.container(border=True):
        # Header banner
        st.markdown(f"### `[{item['type'].upper()}]` {item.get('title')}")
        
        # Primary Template Metadata Columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**ID:** `{item.get('id')}`")
            st.markdown(f"**Type:** `{item.get('type')}`")
            st.markdown(f"**Status:** `{map_business_status(item.get('status'))}` *(Backend: {item.get('status')})*")
        with c2:
            st.markdown(f"**Workstream:** `{item.get('workstream') or 'General'}`")
            st.markdown(f"**SAP Phase:** `{item.get('phase') or 'Realize'}`")
            st.markdown(f"**Detected On:** `{str(item.get('detected_at', ''))[:19]}`")
        with c3:
            st.markdown(f"**Owner:** `{item.get('likely_owner') or 'Unassigned'}`")
            st.markdown(f"**Due Date:** `{item.get('due_date') or 'Unset'}`")
            st.markdown(f"**Confidence:** `{float(item.get('confidence_score', 0)) * 100:.0f}%`")
        with c4:
            # Show ONLY relevant category scoring fields (do not invent)
            if selected_category == "Risk":
                st.markdown(f"**Probability:** `{item.get('probability') or 'Medium'}`")
                st.markdown(f"**Impact:** `{item.get('impact') or 'Medium'}`")
            elif selected_category == "Issue":
                st.markdown(f"**Severity:** `{item.get('severity') or item.get('impact') or 'Medium'}`")
            else:
                st.markdown("**Scoring:** `N/A for Category`")

        # Second row of template fields: Approved By and Related Items
        st.markdown("---")
        cr1, cr2 = st.columns(2)
        with cr1:
            st.markdown("**Approved By:** `Pending Approval (Draft Stage)`")
            st.caption("*(Field is formally recorded upon Project Manager approval in the official register)*")
        with cr2:
            rel_dup = item.get("duplicate_of_id")
            if rel_dup:
                st.markdown(f"**Related Items:** `{rel_dup}` *(Flagged by semantic deduplication)*")
            else:
                st.markdown("**Related Items:** `None linked`")
            st.caption("*(Deduplication and relationship cross-references)*")

        # Description
        st.markdown("---")
        st.markdown("**Description:**")
        st.write(item.get("description"))

        # Mitigation / Next Action (relevant for Risk, Issue & Dependency)
        if selected_category in ("Risk", "Issue", "Dependency"):
            st.markdown("**Mitigation / Next Action:**")
            mit_action = item.get("mitigation_action")
            if mit_action:
                st.write(mit_action)
            else:
                st.caption("*(No mitigation or next action defined yet)*")

        if item.get("scoring_reason"):
            st.markdown("**Scoring Reason / Context:**")
            st.caption(item.get("scoring_reason"))

        # Source Evidence Box
        st.markdown("---")
        render_evidence_box(item)

    st.divider()

    # =========================================================================
    # 3. Individual Item Actions: Approve, Reject, Edit
    # (Strictly single-item; category-level bulk actions are strictly prohibited)
    # =========================================================================
    st.subheader(f"⚡ Action Selected Item: `{item['id']}`")
    st.caption("Individual sign-off controls. All operations are atomic, strictly logged to audit history, and routed through RaidService.")

    tab_approve, tab_reject, tab_edit = st.tabs([
        f"✅ Approve {selected_category}",
        f"❌ Reject {selected_category}",
        f"✏️ Edit / Modify {selected_category}"
    ])

    # -------------------------------------------------------------------------
    # ACTION 1: APPROVE DRAFT
    # -------------------------------------------------------------------------
    with tab_approve:
        st.markdown(f"##### Approve `{item['id']}` into Official Project Register")
        st.caption("Atomically transitions this draft to the Official RAID Register with business status 'Open' and creates an immutable audit record.")
        
        with st.form(key=f"approve_form_{item['id']}"):
            approver_name = st.text_input(
                "Approver Identity (PM Name / User ID) *",
                value="",
                placeholder="e.g. Sarah Jenkins (FI Project Lead)"
            ).strip()
            approval_comments = st.text_area(
                "Optional Approval Comments / SteerCo Notes",
                value="",
                placeholder="Provide rationale or SteerCo context for this approval..."
            )
            submit_approve = st.form_submit_button(f"✅ Confirm Approval of {item['id']}")

            if submit_approve:
                if not approver_name:
                    st.error("Approver Identity is strictly mandatory for formal sign-off.")
                else:
                    try:
                        service.approve_raid(
                            raid_id=item["id"],
                            approved_by=approver_name,
                            comments=approval_comments or None
                        )
                        st.success(f"RAID item '{item['id']}' has been officially approved into the register!")
                        st.rerun()
                    except DuplicateApprovalError as exc:
                        st.error(f"Duplicate Approval Conflict: {exc}")
                    except InvalidRaidStatusError as exc:
                        st.error(f"Invalid Status Transition: {exc}")
                    except RaidValidationError as exc:
                        st.error(f"Validation Failure: {exc}")
                    except Exception as exc:
                        st.error(f"Database/Service Error: {exc}")

    # -------------------------------------------------------------------------
    # ACTION 2: REJECT DRAFT
    # -------------------------------------------------------------------------
    with tab_reject:
        st.markdown(f"##### Reject `{item['id']}` from Review Queue")
        st.caption("Marks the draft as Rejected and preserves full audit traceability. A valid non-empty rejection reason is required.")

        with st.form(key=f"reject_form_{item['id']}"):
            rejector_name = st.text_input(
                "Reviewer Identity (PM Name / User ID) *",
                value="",
                placeholder="e.g. David Ross (PMO Lead)"
            ).strip()
            rejection_reason = st.text_area(
                "Rejection Reason (Mandatory) *",
                value="",
                placeholder="Explain why this item is rejected (e.g. false positive, already addressed, out of project scope)..."
            ).strip()
            submit_reject = st.form_submit_button(f"❌ Confirm Rejection of {item['id']}")

            if submit_reject:
                if not rejector_name:
                    st.error("Reviewer Identity is strictly required.")
                elif not rejection_reason:
                    st.error("A non-empty rejection reason is mandatory.")
                else:
                    try:
                        service.reject_raid(
                            raid_id=item["id"],
                            rejected_by=rejector_name,
                            reason=rejection_reason
                        )
                        st.success(f"RAID item '{item['id']}' has been marked as Rejected.")
                        st.rerun()
                    except InvalidRaidStatusError as exc:
                        st.error(f"Invalid Status Transition: {exc}")
                    except RaidValidationError as exc:
                        st.error(f"Validation Failure: {exc}")
                    except Exception as exc:
                        st.error(f"Database/Service Error: {exc}")

    # -------------------------------------------------------------------------
    # ACTION 3: EDIT / MODIFY DRAFT
    # -------------------------------------------------------------------------
    with tab_edit:
        st.markdown(f"##### Edit Fields for `{item['id']}`")
        st.caption("Refines draft fields while preserving evidence lineage and keeping the item in the review queue with status 'Modified'.")

        with st.form(key=f"edit_form_{item['id']}"):
            st.markdown("**Permitted Editable Fields**")
            new_title = st.text_input("Title *", value=item.get("title") or "")
            new_desc = st.text_area("Description *", value=item.get("description") or "")
            
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                cur_cat = item.get("type", "Risk")
                cat_idx = RAID_CATEGORIES.index(cur_cat) if cur_cat in RAID_CATEGORIES else 0
                new_type = st.selectbox("Category *", RAID_CATEGORIES, index=cat_idx)
                new_workstream = st.text_input("Workstream", value=item.get("workstream") or "General")
            with ec2:
                cur_phase = item.get("phase", "Realize")
                phase_idx = SAP_PHASES.index(cur_phase) if cur_phase in SAP_PHASES else 2
                new_phase = st.selectbox("SAP Phase", SAP_PHASES, index=phase_idx)
                new_owner = st.text_input("Owner", value=item.get("likely_owner") or "")
            with ec3:
                new_due = st.text_input("Due Date", value=item.get("due_date") or "")
                
                # Category-relevant scoring inputs
                if new_type == "Risk":
                    cur_prob = item.get("probability") or "Medium"
                    prob_idx = PROBABILITY_LEVELS.index(cur_prob) if cur_prob in PROBABILITY_LEVELS else 1
                    new_prob = st.selectbox("Probability", PROBABILITY_LEVELS, index=prob_idx)

                    cur_impact = item.get("impact") or "Medium"
                    imp_idx = IMPACT_LEVELS.index(cur_impact) if cur_impact in IMPACT_LEVELS else 1
                    new_impact = st.selectbox("Impact", IMPACT_LEVELS, index=imp_idx)
                    new_sev = None
                elif new_type == "Issue":
                    cur_sev = item.get("severity") or item.get("impact") or "Medium"
                    sev_idx = SEVERITY_LEVELS.index(cur_sev) if cur_sev in SEVERITY_LEVELS else 1
                    new_sev = st.selectbox("Severity", SEVERITY_LEVELS, index=sev_idx)
                    new_prob = None
                    new_impact = new_sev
                else:
                    new_prob = None
                    new_impact = None
                    new_sev = None
                    st.caption("*(No probability/severity levels applicable for Assumption / Decision / Dependency)*")

            # Mitigation / Next Action field (relevant for Risk, Issue & Dependency)
            if new_type in ("Risk", "Issue", "Dependency"):
                new_mitigation = st.text_area("Mitigation / Next Action", value=item.get("mitigation_action") or "")
            else:
                new_mitigation = ""

            st.divider()
            st.markdown("**🔒 Read-Only Lineage Fields**")
            st.caption(f"ID: `{item.get('id')}` | Detection: `{item.get('detection_source')}` | Signal Source: `{item.get('evidence_source_id')}` | Detected: `{str(item.get('detected_at', ''))[:19]}`")

            st.divider()
            editor_name = st.text_input("Editor Identity (PM Name / User ID) *", value="").strip()
            mod_notes = st.text_input("Modification Notes / Rationale", value="PM refined draft fields per project context.")

            submit_modify = st.form_submit_button(f"💾 Save Modifications for {item['id']}")

            if submit_modify:
                if not editor_name:
                    st.error("Editor Identity is required to log modifications.")
                elif not new_title.strip():
                    st.error("Title cannot be empty.")
                elif not new_desc.strip():
                    st.error("Description cannot be empty.")
                else:
                    updates = {
                        "title": new_title.strip(),
                        "description": new_desc.strip(),
                        "type": new_type,
                        "workstream": new_workstream.strip(),
                        "phase": new_phase,
                        "likely_owner": new_owner.strip(),
                        "due_date": new_due.strip(),
                    }
                    if new_type == "Risk":
                        updates["probability"] = new_prob
                        updates["impact"] = new_impact
                        updates["mitigation_action"] = new_mitigation.strip()
                    elif new_type == "Issue":
                        updates["severity"] = new_sev
                        updates["impact"] = new_sev
                        updates["mitigation_action"] = new_mitigation.strip()
                    else:
                        # Clear or leave scoring fields clean for Assumption & Decision
                        updates["mitigation_action"] = ""

                    try:
                        service.modify_raid(
                            raid_id=item["id"],
                            modified_by=editor_name,
                            updates=updates,
                            comments=mod_notes or None
                        )
                        st.success(f"RAID item '{item['id']}' successfully modified! Status updated to 'Modified'.")
                        st.rerun()
                    except InvalidRaidStatusError as exc:
                        st.error(f"Invalid Status Transition: {exc}")
                    except RaidValidationError as exc:
                        st.error(f"Validation Failure: {exc}")
                    except Exception as exc:
                        st.error(f"Database/Service Error: {exc}")


def render_page():
    st.set_page_config(
        page_title="RAID Review Queue - KTern.AI",
        page_icon="📋",
        layout="wide"
    )

    # Header with KaarTech branding
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "kaartech_logo.png"
    if logo_path.exists():
        head_col1, head_col2 = st.columns([1, 8])
        with head_col1:
            st.image(str(logo_path), width=75)
        with head_col2:
            st.title("📋 RAID Review Queue")
    else:
        st.title("📋 RAID Review Queue")

    st.caption("PMO Human-in-the-Loop Review strictly aligned with the Suggested RAID Item Template (R – Risk, A – Assumption, I – Issue, D – Decision, DEP – Dependency).")
    st.divider()

    service = get_raid_service()
    raw_drafts = service.get_pending_drafts()

    drafts = [d for d in raw_drafts if d.get("type") in RAID_CATEGORIES]
    low_conf_candidates = service.get_low_confidence_candidates()

    if not drafts and not low_conf_candidates:
        st.success("🎉 All caught up! No pending RAID drafts or low-confidence candidates in the review queue.")
        st.info("When new signals are ingested through project communication channels, new drafts will appear here.")
        return

    # Section 1-3: High-Confidence Drafts
    if drafts:
        render_high_confidence_drafts(service, drafts)
    else:
        st.info("ℹ️ No high-confidence RAID drafts pending approval.")

    # Section 4: Low-Confidence Review Queue (Stage 6C)
    st.divider()
    render_low_confidence_section(service, low_conf_candidates)


if __name__ == "__main__":
    render_page()
