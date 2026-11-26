"""
Page 4: Semantic Duplicate Review
KTern.AI RAID Identification Agent POC

Interactive deduplication review workflow powered by TF-IDF vectorization and Cosine Similarity:
1. Side-by-side comparison of candidate RAID item vs prior existing RAID item.
2. Direct review actions for candidate item:
   - Approve as a separate RAID (via RaidService.approve_raid)
   - Reject duplicate with mandatory reason (via RaidService.reject_raid)
   - Existing item is preserved and untouched. No physical record deletion.
3. Multi-keyword free-text search (comma-separated) across titles, descriptions, categories,
   workstreams, and source evidence.
4. Match-status filtering: All Matches, Likely Duplicate, Possible Duplicate, Related – Not Duplicate.
5. Active review queue filtering (rejected/resolved candidates removed from active queue).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from ui.service_helper import get_raid_service
from ui.components.evidence_viewer import render_evidence_box
from src.service import (
    RaidService,
    RaidServiceError,
    RaidNotFoundError,
    InvalidRaidStatusError,
    RaidValidationError,
    DuplicateApprovalError,
    AuditPersistenceError
)

# Match status display options mapped to database enum values
MATCH_STATUS_OPTIONS = [
    "All Matches",
    "Likely Duplicate",
    "Possible Duplicate",
    "Related – Not Duplicate"
]

MATCH_STATUS_TO_DB = {
    "Likely Duplicate": "LIKELY_DUPLICATE",
    "Possible Duplicate": "POSSIBLE_DUPLICATE",
    "Related – Not Duplicate": "RELATED_NOT_DUPLICATE"
}

DB_TO_DISPLAY_STATUS = {
    "LIKELY_DUPLICATE": "🔴 Likely Duplicate",
    "POSSIBLE_DUPLICATE": "🟡 Possible Duplicate",
    "RELATED_NOT_DUPLICATE": "🔵 Related – Not Duplicate",
    "NO_MATCH": "⚪ Distinct / No Match"
}


def get_status_badge(status_code: str) -> str:
    """Returns human-friendly formatted status badge."""
    return DB_TO_DISPLAY_STATUS.get(str(status_code).strip(), f"⚪ {status_code}")


def filter_matches_by_status(matches: List[Dict[str, Any]], selected_status_label: str) -> List[Dict[str, Any]]:
    """
    Filters deduplication match records based on the user-selected match-status label.
    'All Matches' retains all records.
    """
    if not selected_status_label or selected_status_label == "All Matches":
        return matches
    target_db_status = MATCH_STATUS_TO_DB.get(selected_status_label)
    if not target_db_status:
        return matches
    return [m for m in matches if str(m.get("match_status")).strip() == target_db_status]


def filter_matches_by_keywords(
    matches: List[Dict[str, Any]],
    keyword_string: str,
    service: RaidService
) -> List[Dict[str, Any]]:
    """
    Filters matches across multiple comma-separated keywords.
    Searches across:
    - Existing RAID title, description, category, workstream, evidence text
    - Candidate RAID title, description, category, workstream, evidence text
    - Match reason / explanation
    """
    if not keyword_string or not str(keyword_string).strip():
        return matches

    tokens = [t.strip().lower() for t in keyword_string.split(",") if t.strip()]
    if not tokens:
        return matches

    filtered = []
    for m in matches:
        candidate = service.get_raid_by_id(m.get("new_raid_id")) or {}
        existing = service.get_raid_by_id(m.get("matched_raid_id")) or {}

        searchable_corpus = " ".join([
            str(candidate.get("title") or ""),
            str(candidate.get("description") or ""),
            str(candidate.get("type") or ""),
            str(candidate.get("workstream") or ""),
            str(candidate.get("evidence_text") or ""),
            str(candidate.get("evidence_links") or ""),
            str(existing.get("title") or ""),
            str(existing.get("description") or ""),
            str(existing.get("type") or ""),
            str(existing.get("workstream") or ""),
            str(existing.get("evidence_text") or ""),
            str(existing.get("evidence_links") or ""),
            str(m.get("match_reason") or ""),
            str(m.get("new_raid_id") or ""),
            str(m.get("matched_raid_id") or "")
        ]).lower()

        # If any user keyword matches anywhere in the candidate or existing item
        if any(tok in searchable_corpus for tok in tokens):
            filtered.append(m)

    return filtered


def filter_active_pending_matches(
    matches: List[Dict[str, Any]],
    service: RaidService,
    include_resolved: bool = False
) -> List[Dict[str, Any]]:
    """
    Removes candidate items that have already been rejected or approved from the active review queue,
    unless include_resolved is explicitly enabled.
    """
    if include_resolved:
        return matches

    active = []
    for m in matches:
        candidate = service.get_raid_by_id(m.get("new_raid_id"))
        # If candidate item does not exist or has been rejected or accepted, it is not active pending
        if candidate:
            cand_status = str(candidate.get("status", "")).strip()
            if cand_status in ("Draft", "Pending_Review", "Modified"):
                active.append(m)
        else:
            # Candidate deleted or missing; exclude from active queue
            pass
    return active


def render_page():
    st.set_page_config(
        page_title="Semantic Duplicate Review - KTern.AI",
        page_icon="🔍",
        layout="wide"
    )

    # Header with KaarTech branding
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "kaartech_logo.png"
    if logo_path.exists():
        head_col1, head_col2 = st.columns([1, 8])
        with head_col1:
            st.image(str(logo_path), width=75)
        with head_col2:
            st.title("🔍 Semantic Duplicate Review")
    else:
        st.title("🔍 Semantic Duplicate Review")

    st.caption("Interactive duplicate review workflow powered by TF-IDF & Cosine Similarity matching with side-by-side comparison and human-in-the-loop review actions.")
    st.divider()

    service = get_raid_service()
    all_raw_matches = service.get_deduplication_matches()

    if not all_raw_matches:
        st.info("ℹ️ **No Duplicate Matches Detected**: All ingested items currently exhibit distinct content.")
        st.caption("When items with high semantic similarity are ingested, deduplication pairs will be surfaced here.")
        return

    # --- Sidebar Filtering Controls ---
    st.sidebar.markdown("### 🔍 Filter Duplicate Matches")

    # Match-Status Filter
    selected_status = st.sidebar.selectbox(
        "Match Classification",
        options=MATCH_STATUS_OPTIONS,
        index=0,
        help="Filter matches by determined similarity classification."
    )

    # Custom Free-Text Keyword Filter (supports multiple comma-separated keywords)
    keyword_input = st.sidebar.text_input(
        "Search Keywords (comma-separated)",
        value="",
        placeholder="e.g. cutover, migration, FI",
        help="Search across titles, descriptions, categories, workstreams, and source evidence of both items."
    )

    # Minimum Similarity Score Slider
    min_score = st.sidebar.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter matches by numerical cosine similarity score threshold."
    )

    # Include Resolved / Rejected Matches Toggle
    include_resolved = st.sidebar.checkbox(
        "Include Resolved / Rejected Candidates",
        value=False,
        help="Show candidates that have already been formally rejected or approved into the register."
    )

    # Reset Filters Button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    # Apply Filtering Pipeline
    # 1. Filter active vs resolved candidates
    active_matches = filter_active_pending_matches(all_raw_matches, service, include_resolved=include_resolved)

    # 2. Filter by match status
    status_filtered = filter_matches_by_status(active_matches, selected_status)

    # 3. Filter by similarity score
    score_filtered = [m for m in status_filtered if float(m.get("similarity_score") or 0.0) >= min_score]

    # 4. Filter by custom keywords
    final_matches = filter_matches_by_keywords(score_filtered, keyword_input, service)

    # Display Metrics & Summary
    st.markdown(
        f"**Displaying `{len(final_matches)}` of `{len(all_raw_matches)}` duplicate matches** "
        f"*(Active Pending: `{len(active_matches)}` | Resolved: `{len(all_raw_matches) - len(active_matches)}`)*"
    )

    if not final_matches:
        st.warning("🔍 **No duplicate matches found matching your current filter criteria.**")
        st.info("Try adjusting your keywords, selecting 'All Matches', or checking 'Include Resolved / Rejected Candidates'.")
        return

    # =========================================================================
    # Overview Table of Filtered Matches
    # =========================================================================
    rows = []
    for m in final_matches:
        score = float(m.get("similarity_score") or 0.0)
        cand = service.get_raid_by_id(m.get("new_raid_id")) or {}
        exist = service.get_raid_by_id(m.get("matched_raid_id")) or {}
        rows.append({
            "Candidate ID": m.get("new_raid_id"),
            "Candidate Title": (cand.get("title") or "—")[:45],
            "Candidate Status": cand.get("status") or "Unknown",
            "Existing ID": m.get("matched_raid_id"),
            "Existing Title": (exist.get("title") or "—")[:45],
            "Existing Status": exist.get("status") or "Unknown",
            "Similarity": f"{score * 100:.1f}%",
            "Match Status": get_status_badge(m.get("match_status")),
            "Same Category?": "Yes" if m.get("same_category") else "No",
            "Same Workstream?": "Yes" if m.get("same_workstream") else "No",
            "Match Reason": m.get("match_reason") or "—"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # =========================================================================
    # A. Side-by-Side Pair Comparison
    # =========================================================================
    st.subheader("🔎 Side-by-Side Duplicate Pair Inspection")
    st.caption("Inspect both records in parallel to analyze semantic overlap, source evidence, and alignment.")

    pair_options = {
        f"Candidate [{m['new_raid_id']}] ⟷ Existing [{m['matched_raid_id']}] ({float(m.get('similarity_score', 0))*100:.0f}%)": m
        for m in final_matches
    }
    selected_pair_label = st.selectbox("Select Match Pair to Inspect & Action:", list(pair_options.keys()))
    selected_match = pair_options[selected_pair_label]

    candidate_id = selected_match["new_raid_id"]
    existing_id = selected_match["matched_raid_id"]

    candidate_item = service.get_raid_by_id(candidate_id)
    existing_item = service.get_raid_by_id(existing_id)

    # Match Engine Diagnostic Banner
    with st.container(border=True):
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Similarity Score", f"{float(selected_match.get('similarity_score', 0)) * 100:.1f}%")
        with m_col2:
            st.markdown("**Match Status:**")
            st.markdown(get_status_badge(selected_match.get("match_status")))
        with m_col3:
            st.markdown(f"**Same Category:** `{'Yes' if selected_match.get('same_category') else 'No'}`")
            st.markdown(f"**Same Workstream:** `{'Yes' if selected_match.get('same_workstream') else 'No'}`")
        with m_col4:
            st.markdown(f"**Human Review Required:** `{'Yes' if selected_match.get('requires_human_review') else 'No'}`")
        
        st.markdown(f"**Matching Engine Assessment:** {selected_match.get('match_reason') or 'No diagnostic explanation provided.'}")

    # Side-by-side record columns
    col_candidate, col_existing = st.columns(2)

    # 1. CANDIDATE RAID ITEM (LEFT)
    with col_candidate:
        st.markdown(f"### 🆕 Candidate RAID Item (`{candidate_id}`)")
        st.caption("Newly ingested item flagged by the semantic matching engine.")
        if candidate_item:
            with st.container(border=True):
                st.markdown(f"#### `[{candidate_item.get('type')}]` {candidate_item.get('title')}")
                
                c_meta1, c_meta2 = st.columns(2)
                with c_meta1:
                    st.write(f"**Workstream:** `{candidate_item.get('workstream') or 'General'}`")
                    st.write(f"**SAP Phase:** `{candidate_item.get('phase') or 'Realize'}`")
                with c_meta2:
                    st.write(f"**Status:** `{candidate_item.get('status')}`")
                    st.write(f"**Owner:** `{candidate_item.get('likely_owner') or 'Unassigned'}`")

                st.markdown("**Description:**")
                st.info(candidate_item.get("description") or "—")

                if candidate_item.get("mitigation_action"):
                    st.markdown("**Mitigation / Next Action:**")
                    st.write(candidate_item.get("mitigation_action"))

                st.markdown("**Source Evidence:**")
                ev_quote = candidate_item.get("evidence_text") or candidate_item.get("evidence_links") or "—"
                st.caption(f"❝ {ev_quote} ❞")
                st.caption(f"Source Signal ID: `{candidate_item.get('evidence_source_id') or candidate_item.get('draft_id') or 'N/A'}`")
        else:
            st.error(f"Candidate item '{candidate_id}' not found in active database.")

    # 2. PRIOR EXISTING RAID ITEM (RIGHT)
    with col_existing:
        st.markdown(f"### 📑 Prior Existing RAID Item (`{existing_id}`)")
        st.caption("Existing database record matched for similarity. (Read-only reference)")
        if existing_item:
            with st.container(border=True):
                st.markdown(f"#### `[{existing_item.get('type')}]` {existing_item.get('title')}")
                
                e_meta1, e_meta2 = st.columns(2)
                with e_meta1:
                    st.write(f"**Workstream:** `{existing_item.get('workstream') or 'General'}`")
                    st.write(f"**SAP Phase:** `{existing_item.get('phase') or 'Realize'}`")
                with e_meta2:
                    st.write(f"**Status:** `{existing_item.get('status')}`")
                    st.write(f"**Owner:** `{existing_item.get('likely_owner') or existing_item.get('owner') or 'Unassigned'}`")

                st.markdown("**Description:**")
                st.info(existing_item.get("description") or "—")

                if existing_item.get("mitigation_action"):
                    st.markdown("**Mitigation / Next Action:**")
                    st.write(existing_item.get("mitigation_action"))

                st.markdown("**Source Evidence:**")
                e_ev_quote = existing_item.get("evidence_text") or existing_item.get("evidence_links") or "—"
                st.caption(f"❝ {e_ev_quote} ❞")
                st.caption(f"Source Signal ID: `{existing_item.get('evidence_source_id') or existing_item.get('draft_id') or 'N/A'}`")
        else:
            st.warning(f"Existing item '{existing_id}' record details not found in active database.")

    st.divider()

    # =========================================================================
    # B. Review Actions
    # =========================================================================
    st.subheader(f"⚡ Review Actions for Candidate Item: `{candidate_id}`")
    st.caption(
        "Actions apply STRICTLY to the candidate item. The prior existing RAID item is completely untouched. "
        "All transactions are logged to immutable audit history and routed through RaidService."
    )

    if not candidate_item:
        st.error("Cannot action a candidate item that does not exist in the database.")
        return

    cand_current_status = candidate_item.get("status")

    if cand_current_status in ("Accepted", "Approved"):
        st.success(f"✅ Candidate item `{candidate_id}` has already been approved into the Official RAID Register. No further review actions required.")
        return
    elif cand_current_status == "Rejected":
        st.info(f"❌ Candidate item `{candidate_id}` has already been formally rejected. No further review actions required.")
        return

    # Two clear review actions for the candidate
    tab_approve, tab_reject = st.tabs([
        f"✅ Approve Candidate as Separate RAID (`{candidate_id}`)",
        f"❌ Reject Candidate as Duplicate (`{candidate_id}`)"
    ])

    # -------------------------------------------------------------------------
    # ACTION 1: APPROVE AS SEPARATE RAID
    # -------------------------------------------------------------------------
    with tab_approve:
        st.markdown(f"##### Approve Candidate `{candidate_id}` into Official RAID Register")
        st.caption(
            "Applies when the reviewer determines that this candidate represents a distinct, genuine scope item "
            "despite high text similarity with an existing record. Moves candidate to Approved Register and preserves audit trail."
        )

        with st.form(key=f"dupe_approve_form_{candidate_id}"):
            approver_name = st.text_input(
                "Approver Identity (PM Name / User ID) *",
                value="",
                placeholder="e.g. Rachel Adams (PMO Director)"
            ).strip()
            approval_notes = st.text_area(
                "Approval Comments / SteerCo Notes",
                value=f"Approved as distinct RAID item after reviewing similarity match against existing item {existing_id}.",
                placeholder="Document why this item is distinct from the matched record..."
            )
            submit_approve = st.form_submit_button(f"✅ Confirm Separate Approval of {candidate_id}")

            if submit_approve:
                if not approver_name:
                    st.error("Approver Identity is strictly mandatory to execute approval.")
                else:
                    try:
                        service.approve_raid(
                            raid_id=candidate_id,
                            approved_by=approver_name,
                            comments=approval_notes or None
                        )
                        st.success(f"Candidate RAID item '{candidate_id}' has been approved into the Official Register as a separate item!")
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
    # ACTION 2: REJECT AS DUPLICATE
    # -------------------------------------------------------------------------
    with tab_reject:
        st.markdown(f"##### Reject Candidate `{candidate_id}` as Duplicate")
        st.caption(
            "Applies when the candidate is confirmed to be redundant. Marks the candidate as Rejected with mandatory reason "
            "and removes it from the active duplicate review queue. Record is NOT physically deleted; audit history and evidence are preserved."
        )

        with st.form(key=f"dupe_reject_form_{candidate_id}"):
            rejector_name = st.text_input(
                "Reviewer Identity (PM Name / User ID) *",
                value="",
                placeholder="e.g. Rachel Adams (PMO Director)"
            ).strip()
            default_reason = f"Duplicate of existing RAID item {existing_id}: '{existing_item.get('title') if existing_item else ''}'."
            rejection_reason = st.text_area(
                "Rejection Reason (Mandatory) *",
                value=default_reason,
                placeholder="State the justification for marking this candidate as a duplicate..."
            ).strip()
            submit_reject = st.form_submit_button(f"❌ Confirm Rejection of {candidate_id}")

            if submit_reject:
                if not rejector_name:
                    st.error("Reviewer Identity is strictly mandatory.")
                elif not rejection_reason:
                    st.error("A non-empty rejection reason is required.")
                else:
                    try:
                        service.reject_raid(
                            raid_id=candidate_id,
                            rejected_by=rejector_name,
                            reason=rejection_reason
                        )
                        st.success(f"Candidate RAID item '{candidate_id}' has been marked as Rejected and removed from active queue.")
                        st.rerun()
                    except InvalidRaidStatusError as exc:
                        st.error(f"Invalid Status Transition: {exc}")
                    except RaidValidationError as exc:
                        st.error(f"Validation Failure: {exc}")
                    except Exception as exc:
                        st.error(f"Database/Service Error: {exc}")


if __name__ == "__main__":
    render_page()
