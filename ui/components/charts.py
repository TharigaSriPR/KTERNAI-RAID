"""
Visual distribution charts component for Streamlit Dashboard.
Displays categorical summaries across Category, Status, SAP Phase, and Workstream.
"""

from typing import List, Dict, Any
import pandas as pd
import streamlit as st


def render_distribution_charts(items: List[Dict[str, Any]]):
    """Renders 4 visual breakdown charts using pandas and Streamlit charts."""
    if not items:
        st.info("ℹ️ No RAID items available to display visual distributions.")
        return

    df = pd.DataFrame(items)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📌 RAID by Category")
        if "type" in df.columns:
            cat_counts = df["type"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            st.bar_chart(cat_counts.set_index("Category"), color="#9A1C1F")
        else:
            st.caption("No category data available.")

    with col2:
        st.markdown("##### 🚦 RAID by Status")
        if "status" in df.columns:
            # Normalize Accepted to Approved in display if desired, or keep as is
            status_df = df["status"].replace({"Accepted": "Approved (Accepted)"}).value_counts().reset_index()
            status_df.columns = ["Status", "Count"]
            st.bar_chart(status_df.set_index("Status"), color="#9A1C1F")
        else:
            st.caption("No status data available.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### 🔄 RAID by SAP Project Phase")
        if "phase" in df.columns:
            phase_counts = df["phase"].fillna("Unassigned").value_counts().reset_index()
            phase_counts.columns = ["SAP Phase", "Count"]
            st.bar_chart(phase_counts.set_index("SAP Phase"), color="#9A1C1F")
        else:
            st.caption("No phase data available.")

    with col4:
        st.markdown("##### 🏢 RAID by Workstream")
        if "workstream" in df.columns:
            ws_counts = df["workstream"].fillna("General").value_counts().reset_index()
            ws_counts.columns = ["Workstream", "Count"]
            st.bar_chart(ws_counts.set_index("Workstream"), color="#9A1C1F")
        else:
            st.caption("No workstream data available.")
