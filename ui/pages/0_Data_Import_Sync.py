"""
Page 0: Project Data Import & Synchronisation (Stage 7C.2 - MCP Integration)
KTern.AI RAID Identification Agent POC

Provides:
- Local folder access via Model Context Protocol (MCP) STDIO Server and Client.
- Reliable macOS native folder selection with clean fallback.
- Explicit separation of button responsibilities:
  1. Configure: Validates and registers target directory in MCP Server (no scanning, no processing).
  2. Scan Folder: Discovers supported files via MCP (strictly read-only, no LLM calls).
  3. Process New Files: Reads files via MCP read_file and executes RAID pipeline.
  4. Refresh Status: Re-queries tracking status.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from config import DATA_DIR, BASE_DIR
from ui.service_helper import get_raid_service


def _get_status_badge(status: str) -> str:
    st_upper = (status or "").upper()
    if st_upper == "PROCESSED":
        return "🟢 Processed"
    elif st_upper in ("NEW", "DISCOVERED"):
        return "🟡 New"
    elif st_upper == "MODIFIED":
        return "🟠 Modified"
    elif st_upper == "FAILED":
        return "🔴 Failed"
    return f"⚪ {status}"


def _get_source_badge(source_type: str) -> str:
    s_lower = (source_type or "").lower()
    if s_lower == "email":
        return "📧 Email"
    elif s_lower == "chat":
        return "💬 Chat"
    elif s_lower in ("meeting_notes", "meeting"):
        return "📝 MOM / Notes"
    elif s_lower in ("project_plan", "plan"):
        return "📅 Project Plan"
    elif s_lower == "defect":
        return "🐛 Defect Tracker"
    return f"📄 {source_type}"


def _get_processing_result_badge(file_info: Dict[str, Any]) -> str:
    status = file_info.get("status", "DISCOVERED")
    if status == "PROCESSED":
        sig = file_info.get("signals_created", file_info.get("record_count", 0))
        drafts = file_info.get("drafts_created", 0)
        return f"✅ Ingested ({sig} signals, {drafts} drafts)"
    elif status == "FAILED":
        err = file_info.get("error_message") or "Ingestion failure"
        return f"❌ Failed: {err[:35]}..."
    elif status == "MODIFIED":
        return "⚠️ Modified on disk (re-process required)"
    return "⏳ Awaiting Ingestion"


def _browse_local_folder(default_path: Optional[str] = None) -> Optional[str]:
    """
    Attempts to open a native OS folder dialog.
    On macOS, uses osascript to open the native Finder folder picker.
    Falls back gracefully without raising uncaught exceptions or error toasts.
    """
    initial_dir = default_path or str(DATA_DIR)

    # 1. macOS native Finder picker via osascript
    if sys.platform == "darwin":
        try:
            escaped_dir = initial_dir.replace('"', '\\"')
            script = f'''
            try
                set defaultFolder to POSIX file "{escaped_dir}"
                set selectedFolder to choose folder with prompt "Select Project Data Folder:" default location defaultFolder
                return POSIX path of selectedFolder
            on error number -128
                return "USER_CANCELLED"
            end try
            '''
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30
            )
            out = proc.stdout.strip()
            if out and out != "USER_CANCELLED" and Path(out).is_dir():
                return out
            if out == "USER_CANCELLED":
                return None
        except Exception:
            pass

    # 2. Tkinter fallback if installed
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        picked = filedialog.askdirectory(initialdir=initial_dir)
        root.destroy()
        if picked and Path(picked).is_dir():
            return picked
    except Exception:
        pass

    return None


def render_page():
    st.set_page_config(
        page_title="Data Import & Synchronisation - KTern.AI",
        page_icon="📥",
        layout="wide"
    )

    # Header with KaarTech branding
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "kaartech_logo.png"
    if logo_path.exists():
        head_col1, head_col2 = st.columns([1, 8])
        with head_col1:
            st.image(str(logo_path), width=75)
        with head_col2:
            st.title("📥 Project Data Import & Synchronisation")
    else:
        st.title("📥 Project Data Import & Synchronisation")

    st.caption(
        "Local folder discovery and pipeline ingestion powered by Model Context Protocol (MCP). "
        "Strict separation of read-only scanning from LLM pipeline processing."
    )
    st.divider()

    service = get_raid_service()
    if not hasattr(service, "scan_data_folder"):
        import importlib
        import src.service
        importlib.reload(src.service)
        service = get_raid_service()

    # Session State Initialization
    if "target_folder_path" not in st.session_state:
        # Check if MCP tools has a persisted folder
        from src.mcp.tools.local_file_tools import get_current_configured_folder
        st.session_state["target_folder_path"] = str(get_current_configured_folder())

    if "mcp_config_result" not in st.session_state:
        st.session_state["mcp_config_result"] = None

    if "data_scan_results" not in st.session_state:
        st.session_state["data_scan_results"] = service.scan_data_folder(
            data_dir=st.session_state["target_folder_path"],
            use_mcp=True
        )

    if "data_process_results" not in st.session_state:
        st.session_state["data_process_results"] = None

    scan_data = st.session_state["data_scan_results"]
    proc_data = st.session_state["data_process_results"]
    all_files = scan_data.get("files", [])

    # --- Section 1: Local Folder Configuration ---
    st.subheader("📁 Local Folder Configuration")

    # Quick selection presets for convenience
    preset_folders = [
        "Select a known folder preset...",
        str(DATA_DIR),
        str(DATA_DIR / "raw"),
        str(DATA_DIR / "raw" / "emails"),
        str(DATA_DIR / "raw" / "chats"),
        str(DATA_DIR / "raw" / "meeting_notes"),
        str(DATA_DIR / "raw" / "project_plans"),
        str(DATA_DIR / "raw" / "defects"),
    ]

    col_input, col_native, col_set = st.columns([3, 1, 1])

    with col_input:
        selected_path_input = st.text_input(
            "Target Local Folder Path:",
            value=st.session_state["target_folder_path"],
            help="Enter the absolute or relative path to the project data directory."
        )

    with col_native:
        st.write("") # vertical spacing
        st.write("")
        if st.button("📂 Browse...", use_container_width=True, help="Open native macOS Finder folder dialog"):
            picked = _browse_local_folder(st.session_state["target_folder_path"])
            if picked:
                st.session_state["target_folder_path"] = picked
                st.rerun()

    with col_set:
        st.write("") # vertical spacing
        st.write("")
        # Problem 3: Configure button only validates and registers folder in MCP Server.
        # It does NOT scan or process files.
        if st.button("Configure", use_container_width=True, help="Validate and register target directory with MCP Server"):
            clean_path = selected_path_input.strip()
            if clean_path:
                with st.spinner("Registering folder with MCP Server..."):
                    cfg_res = service.configure_mcp_folder(clean_path)
                    st.session_state["mcp_config_result"] = cfg_res
                    if cfg_res.get("success"):
                        st.session_state["target_folder_path"] = cfg_res["folder_path"]
                        st.rerun()

    # Quick Preset Dropdown Helper (Fallback option without error)
    with st.expander("📌 Quick Preset Selectors", expanded=False):
        chosen_preset = st.selectbox("Choose a known folder preset:", preset_folders, index=0)
        if chosen_preset != "Select a known folder preset..." and chosen_preset != st.session_state["target_folder_path"]:
            st.session_state["target_folder_path"] = chosen_preset
            st.rerun()

    # Display configuration feedback banner if configure was clicked
    cfg_state = st.session_state.get("mcp_config_result")
    if cfg_state:
        if cfg_state.get("success"):
            st.success(f"✅ **MCP Server Configuration:** Folder registered successfully as `{cfg_state.get('folder_path')}`. Click **Scan Folder** below to discover files.")
        else:
            st.error(f"❌ **MCP Server Configuration Rejected:** {cfg_state.get('error')}")

    current_path_display = st.session_state["target_folder_path"]
    st.markdown(f"**Active Monitored Directory:** `{current_path_display}`")
    st.caption("Supports recursive scanning of emails (`.csv`, `.eml`, `.txt`), chats (`.csv`, `.json`, `.txt`), meeting notes (`.docx`, `.csv`, `.txt`), plans (`.csv`, `.xlsx`), and defects (`.csv`, `.xlsx`).")

    st.markdown("---")

    # --- Section 2: Action Controls (Scan, Process, Refresh) ---
    st.subheader("⚡ Action Controls")
    c_scan, c_proc, c_refresh = st.columns(3)

    with c_scan:
        # Problem 3: Scan Folder calls scan_folder MCP tool, discovers files, does not classify or create drafts
        if st.button("🔍 Scan Folder", use_container_width=True, help="Walk the configured local folder via MCP tool 'scan_folder'. Strictly read-only."):
            with st.spinner("Executing MCP scan_folder..."):
                scan_res = service.scan_data_folder(
                    data_dir=st.session_state["target_folder_path"],
                    use_mcp=True
                )
                st.session_state["data_scan_results"] = scan_res
                # Clear previous config result notice on scan
                st.session_state["mcp_config_result"] = None
                st.toast("Folder scan completed via MCP!", icon="🔍")
                st.rerun()

    with c_proc:
        # Problem 3: Process New Files reads files via MCP read_file and executes full RAID pipeline
        new_or_mod = scan_data.get("new_count", 0) + scan_data.get("modified_count", 0)
        btn_label = f"⚡ Process New Files ({new_or_mod})" if new_or_mod > 0 else "⚡ Process New Files"
        if st.button(btn_label, type="primary", use_container_width=True, help="Retrieve file contents via MCP read_file and execute full RAID pipeline."):
            status_box = st.empty()
            with st.spinner("Reading files via MCP and processing through RAID pipeline..."):
                def update_status(stage: str, msg: str):
                    status_box.info(f"**[{stage}]** {msg}")

                proc_results = service.process_discovered_files(
                    data_dir=st.session_state["target_folder_path"],
                    use_mcp=True,
                    stage_callback=update_status
                )
                st.session_state["data_process_results"] = proc_results
                st.session_state["data_scan_results"] = service.scan_data_folder(
                    data_dir=st.session_state["target_folder_path"],
                    use_mcp=True
                )
                status_box.empty()
                st.success(
                    f"🎉 Processed {proc_results['files_processed_successfully']} file(s). "
                    f"Ingested {proc_results['total_signals_ingested']} signals, generated {proc_results['total_drafts_created']} RAID drafts."
                )
                st.rerun()

    with c_refresh:
        # Problem 3: Refresh Status retrieves current tracking status
        if st.button("🔄 Refresh Status", use_container_width=True, help="Re-query status for all files using MCP get_file_status."):
            with st.spinner("Refreshing file status..."):
                st.session_state["data_scan_results"] = service.scan_data_folder(
                    data_dir=st.session_state["target_folder_path"],
                    use_mcp=True
                )
                st.toast("Status refreshed!", icon="🔄")
                st.rerun()

    # --- Execution Progress / Last Run Banner ---
    if proc_data:
        with st.container(border=True):
            st.markdown("#### 📊 Last Synchronisation Execution Summary")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Files Ingested", proc_data.get("files_processed_successfully", 0))
            with r2:
                st.metric("Signals Ingested", proc_data.get("total_signals_ingested", 0))
            with r3:
                st.metric("Drafts Generated", proc_data.get("total_drafts_created", 0))
            with r4:
                st.metric("Duplicates Skipped", proc_data.get("total_duplicates_skipped", 0))

            if proc_data.get("files_failed", 0) > 0:
                st.error(f"⚠️ {proc_data['files_failed']} file(s) failed processing. See file status table below for error details.")

    st.markdown("---")

    # --- Section 3: Summary Metrics ---
    st.subheader("📈 Summary Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Files Discovered", scan_data.get("total_supported", 0))
    with m2:
        st.metric("New Files", scan_data.get("new_count", 0))
    with m3:
        st.metric("Unchanged Files", scan_data.get("processed_count", 0))
    with m4:
        st.metric("Modified Files", scan_data.get("modified_count", 0))

    m5, m6, m7, m8 = st.columns(4)
    total_records = sum(int(f.get("record_count", 0)) for f in all_files)
    total_drafts = sum(int(f.get("drafts_created", 0)) for f in all_files)
    success_files = sum(1 for f in all_files if f.get("status") == "PROCESSED")

    with m5:
        st.metric("Successfully Processed", success_files)
    with m6:
        st.metric("Failed Files", scan_data.get("failed_count", 0))
    with m7:
        st.metric("Total ProjectSignals Created", total_records)
    with m8:
        st.metric("RAID Drafts Generated", total_drafts)

    st.divider()

    # --- Section 4: File-Level Status Table ---
    st.subheader("📋 Discovered Files & Processing Status")

    if not all_files:
        st.info("ℹ️ No supported project source data files discovered in the configured folder.")
        st.markdown(
            "Place project emails (`.csv`, `.eml`, `.txt`), chats (`.csv`, `.json`, `.txt`), "
            "meeting minutes (`.docx`, `.csv`, `.txt`), project plans (`.csv`, `.xlsx`), "
            "or defect registers (`.csv`, `.xlsx`) in the folder and click **Scan Folder**."
        )
        return

    # Build the required 5-column table:
    # | File Name | Type | Status | Last Modified | Processing Result |
    table_rows = []
    for f in all_files:
        table_rows.append({
            "File Name": f.get("file_name"),
            "Type": _get_source_badge(f.get("source_type")),
            "Status": _get_status_badge(f.get("status")),
            "Last Modified": str(f.get("last_modified", ""))[:19],
            "Processing Result": _get_processing_result_badge(f)
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # --- Section 5: Detailed File Provenance Expander ---
    with st.expander("🔎 Detailed File Provenance & Content Hash Inspection", expanded=False):
        file_options = {f"{f['file_name']} ({f['file_path']})": f for f in all_files}
        sel_name = st.selectbox("Select file to inspect details:", list(file_options.keys()))
        selected_file = file_options[sel_name]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**File Name:** `{selected_file.get('file_name')}`")
            st.write(f"**Source Category:** {_get_source_badge(selected_file.get('source_type'))}")
            st.write(f"**Size:** `{selected_file.get('file_size')} bytes`")
        with c2:
            st.write(f"**Status:** {_get_status_badge(selected_file.get('status'))}")
            st.write(f"**Signals Count:** `{selected_file.get('record_count')}`")
            st.write(f"**Drafts Created:** `{selected_file.get('drafts_created')}`")
        with c3:
            st.write(f"**Discovered At:** `{str(selected_file.get('discovered_at', ''))[:19]}`")
            st.write(f"**Processed At:** `{str(selected_file.get('processed_at', '—'))[:19]}`")

        st.markdown(f"**Content SHA-256 Digest:** `{selected_file.get('content_hash')}`")
        if selected_file.get("error_message"):
            st.error(f"Documented Error: {selected_file.get('error_message')}")

    # Navigation CTA
    st.divider()
    c_nav1, c_nav2 = st.columns([3, 1])
    with c_nav1:
        st.markdown("💡 **Next Step:** After synchronising data, navigate to the **RAID Review Queue** to inspect newly classified items.")
    with c_nav2:
        st.markdown("👉 [Go to RAID Review Queue](1_RAID_Review)")


if __name__ == "__main__":
    render_page()
