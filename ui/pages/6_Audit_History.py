"""
Page 5: Audit History
KTern.AI RAID Identification Agent POC

Dedicated compliance and audit log interface.
Displays immutable, append-only history of agent proposals, approvals,
rejections, and field modifications with human-readable before/after state snapshots.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from ui.service_helper import get_raid_service
from src.export.excel_export import export_audit_history


def _get_action_badge(action: str) -> str:
    """Returns a visual badge for the audit action verb."""
    act = (action or "").upper()
    if act == "APPROVED":
        return "✅ APPROVED"
    elif act == "REJECTED":
        return "❌ REJECTED"
    elif act == "MODIFIED":
        return "✏️ MODIFIED"
    elif act == "PROPOSED":
        return "🤖 PROPOSED"
    else:
        return f"📌 {act}"


def _render_state_card(data: Any, title: str, subtitle: str, is_final: bool = False, action: str = ""):
    """
    Renders an executive, human-readable snapshot card instead of raw developer JSON.
    """
    st.markdown(f"##### {title}")
    st.caption(subtitle)

    if not data:
        if is_final and action == "PROPOSED":
            with st.container(border=True):
                st.markdown("##### ⏳ Post-Decision State")
                st.info(
                    "**Status:** `Awaiting Human PM Review`\n\n"
                    "This audit record represents the AI agent's initial extraction. "
                    "A finalized state snapshot will be recorded once a Project Manager "
                    "approves, modifies, or rejects this item in the review queue."
                )
        else:
            with st.container(border=True):
                st.caption("*(No state snapshot recorded for this phase)*")
        return

    if not isinstance(data, dict):
        with st.container(border=True):
            st.write(data)
        return

    # Specific formatted card for rejection result
    if action == "REJECTED" and is_final and ("reason" in data or data.get("status") == "Rejected"):
        with st.container(border=True):
            st.markdown("#### ❌ Rejection Record")
            st.markdown(f"**Final Status:** `{data.get('status', 'Rejected')}`")
            st.markdown("**Documented Rejection Reason:**")
            st.error(data.get("reason") or "No detailed rationale provided.")
            st.caption("🔒 *Draft is preserved in database for historical audit compliance.*")
        return

    # Standard formatted card for drafts or approved records
    with st.container(border=True):
        item_type = data.get("type") or "RAID Item"
        item_title = data.get("title") or "Untitled Item"
        st.markdown(f"#### `[{item_type.upper()}]` {item_title}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Workstream:** `{data.get('workstream') or 'General'}`")
            st.markdown(f"**SAP Phase:** `{data.get('phase') or 'Realize'}`")
        with c2:
            owner_val = data.get("owner") or data.get("likely_owner") or "Unassigned"
            st.markdown(f"**Owner:** `{owner_val}`")
            st.markdown(f"**Due Date:** `{data.get('due_date') or 'Unset'}`")
        with c3:
            st.markdown(f"**Status:** `{data.get('status') or 'Draft'}`")
            if "confidence_score" in data:
                conf = float(data.get("confidence_score", 0))
                st.markdown(f"**Confidence:** `{conf * 100:.0f}%`")
            elif "approved_by" in data:
                st.markdown(f"**Approved By:** `{data.get('approved_by')}`")

        # Category-relevant scoring metrics
        scoring_items = []
        if data.get("probability"):
            scoring_items.append(f"**Probability:** `{data.get('probability')}`")
        if data.get("impact"):
            scoring_items.append(f"**Impact:** `{data.get('impact')}`")
        if data.get("severity"):
            scoring_items.append(f"**Severity:** `{data.get('severity')}`")
        if data.get("severity_or_impact"):
            scoring_items.append(f"**Severity/Impact:** `{data.get('severity_or_impact')}`")

        if scoring_items:
            st.markdown(" | ".join(scoring_items))

        # Description
        if data.get("description"):
            st.markdown("**Description:**")
            st.info(data.get("description"))

        # Mitigation / Next Action
        if data.get("mitigation_action"):
            st.markdown("**Mitigation / Next Action:**")
            st.write(data.get("mitigation_action"))

        # Source Evidence Citation
        ev_text = data.get("evidence_text") or data.get("evidence_links")
        if ev_text:
            st.markdown("**Source Evidence Quote:**")
            st.caption(f"❝ {ev_text} ❞")
            src_id = data.get("evidence_source_id") or data.get("draft_id")
            src_type = data.get("evidence_source_type") or "signal"
            if src_id:
                st.caption(f"Source: `{src_id}` ({src_type})")


def render_page():
    st.set_page_config(
        page_title="Audit History - KTern.AI",
        page_icon="📜",
        layout="wide"
    )

    # Header with KaarTech branding
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "kaartech_logo.png"
    if logo_path.exists():
        head_col1, head_col2 = st.columns([1, 8])
        with head_col1:
            st.image(str(logo_path), width=75)
        with head_col2:
            st.title("LOGS")
    else:
        st.title("LOGS")

    st.caption("Chronological, append-only ledger tracking every AI proposal, PM approval, rejection, and modification event.")
    st.divider()

    service = get_raid_service()
    history = service.get_audit_history()

    if not history:
        st.info("ℹ️ **Audit Trail is Empty**: No lifecycle events have been recorded yet.")
        return

    # Header & Export
    col_hdr, col_dl = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"**Total Audit Records: `{len(history)}`**")
        st.caption("🔒 *Read-Only Ledger: Records cannot be updated, pruned, or purged through the application.*")
    with col_dl:
        excel_bytes = export_audit_history(history)
        st.download_button(
            label="📥 Download Audit Trail Excel",
            data=excel_bytes,
            file_name="RAID_Audit_Trail.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # --- Filters ---
    st.sidebar.markdown("### 🔍 Filter Audit Trail")

    # Filter by Action
    actions = sorted(list(set(h.get("action", "") for h in history if h.get("action"))))
    selected_actions = st.sidebar.multiselect("Action", actions, default=actions)

    # Filter by Actor
    actors = sorted(list(set(h.get("action_by", "") for h in history if h.get("action_by"))))
    selected_actors = st.sidebar.multiselect("Actor Identity", actors, default=actors)

    # Filter by RAID Item ID
    item_search = st.sidebar.text_input("Filter by Item ID", value="").strip().lower()

    filtered_history = [
        h for h in history
        if h.get("action") in selected_actions
        and h.get("action_by") in selected_actors
        and (not item_search or item_search in (h.get("item_id") or "").lower())
    ]

    st.markdown(f"**Displaying `{len(filtered_history)}` of `{len(history)}` audit records**")

    # Table Overview
    if filtered_history:
        rows = []
        for h in filtered_history:
            rows.append({
                "Audit ID": h.get("id"),
                "Item ID": h.get("item_id"),
                "Table": h.get("item_table"),
                "Action": _get_action_badge(h.get("action")),
                "Actor": h.get("action_by"),
                "Timestamp (UTC)": (h.get("timestamp") or "")[:19],
                "Notes": h.get("action_notes") or "-"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No audit records match your current filter criteria.")
        return

    st.divider()

    # --- Beautified Before/After Inspection ---
    st.subheader("🔎 Event State Inspection")
    st.caption("Inspect the exact state transition recorded for the selected audit event.")

    event_options = {
        f"Event #{h['id']} | [{h['action']}] on {h['item_id']} by {h['action_by']} ({(h.get('timestamp') or '')[:19]})": h
        for h in filtered_history
    }
    sel_label = st.selectbox("Select Audit Event to Inspect Details", list(event_options.keys()))
    sel_event = event_options[sel_label]

    # Executive Event Summary Banner
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"**Target Item:** `{sel_event.get('item_id')}`")
            st.caption(f"Database Table: `{sel_event.get('item_table')}`")
        with m2:
            st.markdown(f"**Action Verb:** {_get_action_badge(sel_event.get('action'))}")
        with m3:
            st.markdown(f"**Actor Identity:** `{sel_event.get('action_by')}`")
        with m4:
            st.markdown(f"**Timestamp (UTC):** `{str(sel_event.get('timestamp') or '')[:19]}`")

        if sel_event.get("action_notes"):
            st.info(f"📝 **Action Notes / Rationale:** {sel_event.get('action_notes')}")

    # Parse JSON payloads safely
    parsed_prop = None
    prop_raw = sel_event.get("proposed_data")
    if prop_raw:
        try:
            parsed_prop = json.loads(prop_raw)
        except Exception:
            parsed_prop = {"raw_data": prop_raw}

    parsed_final = None
    final_raw = sel_event.get("final_data")
    if final_raw:
        try:
            parsed_final = json.loads(final_raw)
        except Exception:
            parsed_final = {"raw_data": final_raw}

    action_verb = str(sel_event.get("action", "")).upper()

    # Side-by-Side Human-Readable Cards
    c_prop, c_final = st.columns(2)

    with c_prop:
        _render_state_card(
            data=parsed_prop,
            title="📦 Prior State / Agent Proposal",
            subtitle="Snapshot of the RAID item before or at the time of this event.",
            is_final=False,
            action=action_verb
        )

    with c_final:
        _render_state_card(
            data=parsed_final,
            title="📥 Final State / Human Decision",
            subtitle="Resulting record state after human review or modification.",
            is_final=True,
            action=action_verb
        )


if __name__ == "__main__":
    render_page()
