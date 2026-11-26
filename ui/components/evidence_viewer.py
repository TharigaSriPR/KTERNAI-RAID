"""
Evidence viewer component for Streamlit Dashboard.
Displays source citations, exact verbatim quotes, and raw signal traceability.
Distinguishes between VERIFIED direct quotes, FALLBACK algorithmic selections, and UNAVAILABLE evidence.
Surfaces LLM provider provenance (OpenRouter, Gemini, Mock, Rule-based Foresight).
"""

from typing import Dict, Any, Optional
import streamlit as st


def render_evidence_box(item: Dict[str, Any]):
    """Renders verifiable evidence citation and traceability block with explicit status."""
    source_type = item.get("evidence_source_type") or "signal"
    source_id = item.get("evidence_source_id") or item.get("draft_id") or "N/A"
    quote = item.get("evidence_text") or item.get("evidence_links") or item.get("evidence") or ""
    detection = item.get("detection_source") or "pipeline"
    dup_id = item.get("duplicate_of_id")
    
    # Evidence Provenance Status
    raw_status = str(item.get("evidence_status") or "unknown").strip().upper()
    if raw_status in ("VERIFIED", "FALLBACK", "UNAVAILABLE"):
        status_key = raw_status
    else:
        status_key = "UNKNOWN"

    # Provider Provenance
    provider = str(item.get("llm_provider") or "unknown").strip().lower()
    model = str(item.get("llm_model") or "unknown").strip()

    st.markdown("##### 🔍 Source Evidence & Audit Trace")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.caption(f"**Source Channel:** `{source_type.upper()}`")
    with col_s2:
        st.caption(f"**Source Signal ID:** `{source_id}`")
    with col_s3:
        st.caption(f"**Origin Engine:** `{detection}`")
    with col_s4:
        if provider == "openrouter":
            st.caption(f"**Provider:** 🌐 `OpenRouter ({model})`")
        elif provider == "gemini":
            st.caption(f"**Provider:** ✨ `Gemini ({model})`")
        elif provider == "mock":
            st.caption(f"**Provider:** ⚠️ `Mock ({model})`")
        elif provider == "rule_based_foresight":
            st.caption(f"**Provider:** 📈 `Rule-Based Foresight`")
        else:
            st.caption(f"**Provider:** ❓ `Legacy / Unknown`")

    # Render Evidence Status Badge and Quote
    if status_key == "VERIFIED":
        st.success("🟢 **Evidence Status: VERIFIED** — Verbatim source quotation located and verified directly in the project signal.")
        if quote:
            st.info(f"❝ {quote} ❞")
        else:
            st.caption("*(Verbatim quote recorded as empty)*")
    elif status_key == "FALLBACK":
        st.warning("🟠 **Evidence Status: FALLBACK** — Algorithmically selected relevant sentence. **Not a direct verified quotation.**")
        if quote:
            st.markdown(f"> **Supporting Context Sentence:** *\"{quote}\"*")
        else:
            st.caption("*(Fallback sentence empty)*")
    elif status_key == "UNAVAILABLE":
        st.error("⚪ **Evidence Status: UNAVAILABLE** — No authentic supporting quote or relevant sentence was identified in the source signal.")
        st.caption("*(Evidence text unavailable)*")
    else:
        st.info("ℹ️ **Evidence Status: UNKNOWN** — Legacy record created prior to evidence provenance tracking.")
        if quote:
            st.markdown(f"> *\"{quote}\"*")

    if dup_id:
        st.warning(f"⚠️ **Duplicate Alert:** Engine flagged potential similarity with item `{dup_id}`.")
