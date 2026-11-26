"""
Main Application Entry Point: app.py
RAID Identification Agent POC (KTern.AI / KaarTech)

Can be run:
1. Via CLI: python3 app.py
   Initializes DB, ingests sample signals, executes the pipeline, and displays summary.
2. Via Streamlit Dashboard: streamlit run app.py
   Launches the interactive Web UI for PM review queue, Approved Register, and Exports.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_PATH,
    EMAILS_DIR,
    CHATS_DIR,
    MEETING_NOTES_DIR,
    PROJECT_PLANS_DIR,
    DEFECTS_DIR
)
from src.database.db_manager import DatabaseManager
from src.pipeline import RAIDPipeline


def run_pipeline_base():
    """
    Executes the base pipeline: Ingestion -> Normalisation -> Noise Filter ->
    Detection & Predictive Engine -> Deduplication -> Evidence Linking -> SQLite DB.
    """
    print("=" * 70)
    print(" KTern.AI RAID Identification Agent")
    print("=" * 70)

    # Initialize SQLite database
    db = DatabaseManager(DB_PATH)
    print(f"✅ SQLite Database initialized at: {DB_PATH}")

    # Initialize Pipeline Orchestrator & Service Layer
    pipeline = RAIDPipeline(db)
    from src.service import RaidService
    service = RaidService(db, pipeline)
    print("✅ Pipeline components loaded:")
    print("   - Data Ingestion: Pandas / CSV / Excel adapters")
    print("   - Normalisation: Common Signal Schema")
    print("   - Noise Filtering: Rule-based heuristics")
    print("   - RAID Identification & Extraction: LLM + Pydantic schema")
    print("   - Predictive Engine: Task slippage & defect clustering rules")
    print("   - Deduplication: Similarity matching")
    print("   - Evidence Linking: Verifiable source quotation & ID trace")

    # Discover supported files dynamically
    print("\n🔍 Discovering supported project data files...")
    scan_res = service.scan_data_folder()
    print(f"   Found {scan_res['total_supported']} supported project data files ({scan_res['new_count']} new, {scan_res['processed_count']} processed):")
    for f in scan_res["files"]:
        print(f"   • {f['file_path']} ({f['source_type']}) - Status: {f['status']}")

    # Process new and modified files through the architectural pipeline
    print("\n⚙️  Executing end-to-end pipeline processing on new/modified files...")
    results = service.process_discovered_files()

    print("\n📊 Pipeline Execution Summary:")
    print(f"   • Files Targeted:           {results['files_targeted']}")
    print(f"   • Files Processed Cleanly:  {results['files_processed_successfully']}")
    print(f"   • Files Failed:             {results['files_failed']}")
    print(f"   • Total Signals Ingested:   {results['total_signals_ingested']}")
    print(f"   • Normalized Signals:       {results['total_signals_normalized']}")
    print(f"   • Discarded as Noise:       {results['total_noise_filtered']} (Zero token cost)")
    print(f"   • Total Drafts Created:     {results['total_drafts_created']}")
    print(f"   • Duplicates / Prior Skipped: {results['total_duplicates_skipped']}")

    # Display pending drafts from SQLite DB
    drafts = db.get_pending_drafts()
    print(f"\n📋 Current Review Queue in SQLite ({len(drafts)} drafts pending PM approval):")
    for idx, d in enumerate(drafts, 1):
        print(f"   {idx}. [{d['type'].upper()}] {d['title']}")
        print(f"      • Workstream: {d['workstream']} | Phase: {d['phase']} | Confidence: {d['confidence_score']}")
        print(f"      • Source: {d['detection_source']} | Evidence: \"{d['evidence_text'][:60]}...\"")

    print("\n🚀 Base is ready! You can launch the UI with: streamlit run app.py")
    return db, pipeline


# Check if executed via Streamlit or standard Python
def main():
    try:
        import streamlit as st
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            is_streamlit = get_script_run_ctx() is not None
        except Exception:
            is_streamlit = False

        # Detect if running under streamlit runtime
        if is_streamlit or (hasattr(st, "_is_running_with_streamlit") and st._is_running_with_streamlit):
            from ui.dashboard import main as render_dashboard
            render_dashboard()
            return
    except ImportError:
        pass

    # Default CLI execution
    run_pipeline_base()


if __name__ == "__main__":
    main()
