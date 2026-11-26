"""
Page 3: Predictive Insights
KTern.AI RAID Identification Agent POC

Dedicated page for Foresight Predictive Intelligence findings.
Presents automated task slippages, defect clustering, contradicted assumptions,
and dependency blockages identified by rule-based heuristic engines.
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
from src.export.excel_export import export_predictions


def _get_severity_badge(severity: str) -> str:
    sev = (severity or "Medium").upper()
    if sev == "CRITICAL":
        return "🔴 Critical"
    elif sev == "HIGH":
        return "🟠 High"
    elif sev == "MEDIUM":
        return "🟡 Medium"
    else:
        return "🟢 Low"


def render_page():
    st.set_page_config(page_title="Predictive Insights - KTern.AI", page_icon="🔮", layout="wide")
    st.title("🔮 Foresight Predictive Insights")
    st.markdown("Heuristic foresight engine analyzing task schedule slippages, defect concentration thresholds, and contradicted assumptions.")
    st.divider()

    service = get_raid_service()
    predictions = service.get_predictions()

    if not predictions:
        st.info("ℹ️ **No Predictive Findings Stored**: The predictive engine has not generated active warnings.")
        st.markdown("Run `python3 app.py` or ingest sample project plans and defect logs to trigger predictive rules.")
        return

    # Header Controls
    col_hdr, col_dl = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"**Total Active Foresight Warnings: {len(predictions)}**")
        st.caption("⚠️ *Note: Confidence scores represent rule-based heuristic match weights, not statistically calibrated probabilities.*")
    with col_dl:
        excel_bytes = export_predictions(predictions)
        st.download_button(
            label="📥 Download Predictions Excel",
            data=excel_bytes,
            file_name="Predictive_Insights.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # Filter by prediction type
    pred_types = sorted(list(set(p.get("prediction_type", "OTHER") for p in predictions)))
    selected_types = st.multiselect("Filter by Prediction Rule Type", pred_types, default=pred_types)

    filtered_preds = [p for p in predictions if p.get("prediction_type") in selected_types]

    st.divider()

    # Categorization Groups
    groups = {
        "⏰ Overdue Tasks & Schedule Slippage": ["TASK_OVERDUE", "UPCOMING_DELAY"],
        "🐛 Defect Concentration & Clustering": ["DEFECT_CLUSTER"],
        "🔄 Contradicted Assumptions": ["ASSUMPTION_CONTRADICTED"],
        "⛓️ Prerequisite Dependency Blockages": ["DEPENDENCY_BLOCKAGE"],
    }

    # Render each group
    for group_title, type_keys in groups.items():
        group_items = [p for p in filtered_preds if p.get("prediction_type") in type_keys]
        if group_items:
            st.markdown(f"### {group_title} ({len(group_items)})")
            for item in group_items:
                sev_badge = _get_severity_badge(item.get("severity"))
                conf_val = float(item.get("confidence") or 0.0) * 100
                with st.expander(f"{sev_badge} | {item.get('title')} (Score: {conf_val:.0f}%)", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Description:** {item.get('description')}")
                        st.markdown(f"**Evidence & Indicator:**")
                        st.info(f"❝ {item.get('evidence')} ❞")
                    with c2:
                        st.write(f"**ID:** `{item.get('id')}`")
                        st.write(f"**Type:** `{item.get('prediction_type')}`")
                        st.write(f"**Suggested Category:** `{item.get('suggested_raid_category')}`")
                        st.write(f"**Severity:** `{item.get('severity') or 'Medium'}`")
                        st.write(f"**Detected:** `{(item.get('detected_at') or '')[:19]}`")
                        
                        sources = item.get("source_signal_ids")
                        if sources:
                            st.caption(f"**Signals:** `{sources}`")
                        related = item.get("related_raid_ids")
                        if related:
                            st.caption(f"**Related RAID:** `{related}`")

    # Catch-all for any other unmapped prediction types
    other_items = [
        p for p in filtered_preds
        if not any(p.get("prediction_type") in t_list for t_list in groups.values())
    ]
    if other_items:
        st.markdown(f"### 📌 Other Predictive Signals ({len(other_items)})")
        for item in other_items:
            sev_badge = _get_severity_badge(item.get("severity"))
            with st.expander(f"{sev_badge} | {item.get('title')}", expanded=False):
                st.write(item.get("description"))
                st.info(item.get("evidence"))


if __name__ == "__main__":
    render_page()
