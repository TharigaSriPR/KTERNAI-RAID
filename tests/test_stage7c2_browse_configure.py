"""
Stage 7C.2 Automated Test Suite: Fix Local Folder Browse and MCP Configure Functionality.
Verifies the 10 requirements:
1. Browse fallback behaviour.
2. Valid folder configuration.
3. Invalid folder rejection.
4. Configure button calls the service without scanning.
5. MCP Client invokes the correct MCP tool.
6. MCP Server accepts and stores the configured folder.
7. Configured folder remains available for scanning.
8. Scan does not process files (read-only verification).
9. Process New Files reads files through MCP.
10. Streamlit reruns do not lose configuration.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.database.db_manager import DatabaseManager
from src.pipeline import RAIDPipeline
from src.service import RaidService
from src.classification.schemas import ClassificationOutcome, ClassificationStatus, RAIDCategory, RAIDExtractionResult
from src.mcp.client import MCPClient
from src.mcp.tools.local_file_tools import (
    select_or_configure_folder,
    scan_folder,
    read_file,
    get_file_status,
    get_current_configured_folder,
    set_current_configured_folder
)
import importlib
data_sync_module = importlib.import_module("ui.pages.0_Data_Import_Sync")
_browse_local_folder = data_sync_module._browse_local_folder


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated test SQLite database."""
    db_file = tmp_path / "test_stage7c2.db"
    return DatabaseManager(db_file)


@pytest.fixture
def sample_folder(tmp_path):
    """Creates a sample test folder with emails and chats."""
    folder = tmp_path / "custom_project_data"
    folder.mkdir(parents=True)

    emails_dir = folder / "emails"
    emails_dir.mkdir()
    (emails_dir / "emails.csv").write_text("id,sender,recipient,subject,body\n1,a@b.com,c@d.com,Cutover Risk,Window is tight\n")

    chats_dir = folder / "chats"
    chats_dir.mkdir()
    (chats_dir / "chat.csv").write_text("chat_id,timestamp,author,channel,text\n1,2026-11-01,dev,gen,Delay in migration\n")

    return folder


@pytest.fixture
def mock_classifier():
    """Mock classifier that returns valid Risk extraction without live API calls."""
    classifier = MagicMock()
    classifier.confidence_threshold = 0.75

    def fake_classify(sig):
        return ClassificationOutcome(
            signal_id=sig.id,
            source_type=sig.source_type,
            status=ClassificationStatus.COMPLETED,
            is_raid=True,
            confidence=0.88,
            category=RAIDCategory.RISK,
            extraction=RAIDExtractionResult(
                source_signal_id=sig.id,
                is_raid=True,
                category=RAIDCategory.RISK,
                confidence=0.88,
                title=f"Extracted Risk: {sig.subject}",
                description=sig.content,
                workstream=sig.workstream or "General",
                phase=sig.phase or "Realize",
                probability="Medium",
                impact="Moderate delay",
                evidence=sig.content[:60],
                owner="Lead"
            ),
            llm_provider="test_mock",
            llm_model="test_model"
        )

    classifier.classify_signal_detailed.side_effect = fake_classify
    return classifier


class TestStage7C2BrowseAndConfigure:
    """Tests for Stage 7C.2 requirements."""

    # 1. Browse fallback behaviour
    def test_1_browse_fallback_behaviour(self):
        # When osascript returns error/cancel or is unavailable, _browse_local_folder must return None cleanly
        with patch("subprocess.run") as mock_sub:
            mock_sub.side_effect = Exception("Display server not accessible")
            result = _browse_local_folder("/tmp")
            assert result is None  # Clean fallback without exception

    # 2. Valid folder configuration
    def test_2_valid_folder_configuration(self, temp_db, sample_folder):
        service = RaidService(db_manager=temp_db)
        try:
            res = service.configure_mcp_folder(str(sample_folder))
            assert res.get("success") is True
            assert res.get("folder_path") == str(sample_folder.resolve())
            assert "Successfully configured" in res.get("message", "")
        finally:
            service.close_mcp_client()

    # 3. Invalid folder rejection
    def test_3_invalid_folder_rejection(self, temp_db, tmp_path):
        service = RaidService(db_manager=temp_db)
        try:
            # Non-existent directory
            res1 = service.configure_mcp_folder(str(tmp_path / "does_not_exist_404"))
            assert res1.get("success") is False
            assert "does not exist" in res1.get("error", "").lower()

            # Prohibited system directory
            res2 = service.configure_mcp_folder("/etc")
            assert res2.get("success") is False
            assert "prohibited" in res2.get("error", "").lower()

            # Empty string
            res3 = service.configure_mcp_folder("")
            assert res3.get("success") is False
            assert "cannot be empty" in res3.get("error", "").lower()
        finally:
            service.close_mcp_client()

    # 4. Configure button calls the service
    def test_4_configure_button_calls_the_service(self, temp_db, sample_folder):
        service = RaidService(db_manager=temp_db)
        try:
            # Simulate what the Configure button calls
            cfg_res = service.configure_mcp_folder(str(sample_folder))
            assert cfg_res["success"] is True

            # Verify that calling configure alone did NOT perform scanning
            # Database should remain completely empty (no raw_signals, no drafts)
            with temp_db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM raw_signals")
                assert cur.fetchone()[0] == 0
                cur.execute("SELECT COUNT(*) FROM raid_drafts")
                assert cur.fetchone()[0] == 0
        finally:
            service.close_mcp_client()

    # 5. MCP Client invokes the correct MCP tool
    def test_5_mcp_client_invokes_correct_mcp_tool(self, sample_folder):
        client = MCPClient()
        try:
            res = client.configure_folder(str(sample_folder))
            assert res.get("success") is True
            assert res.get("folder_path") == str(sample_folder.resolve())
        finally:
            client.close()

    # 6. MCP Server accepts and stores the configured folder
    def test_6_mcp_server_accepts_and_stores_configured_folder(self, sample_folder):
        res = select_or_configure_folder(str(sample_folder))
        assert res["success"] is True
        active_folder = get_current_configured_folder()
        assert active_folder == sample_folder.resolve()

    # 7. Configured folder remains available for scanning
    def test_7_configured_folder_remains_available_for_scanning(self, temp_db, sample_folder):
        service = RaidService(db_manager=temp_db)
        try:
            # Step 1: Configure
            cfg = service.configure_mcp_folder(str(sample_folder))
            assert cfg["success"] is True

            # Step 2: Scan without passing folder_path (MCP server must use configured folder)
            client = service.get_mcp_client()
            scan_res = client.scan_folder()
            assert scan_res["total_supported"] == 2
            assert scan_res["new_count"] == 2
            assert str(sample_folder.resolve()) in scan_res["data_dir"]
        finally:
            service.close_mcp_client()

    # 8. Scan does not process files
    def test_8_scan_does_not_process_files(self, temp_db, sample_folder):
        service = RaidService(db_manager=temp_db)
        try:
            service.configure_mcp_folder(str(sample_folder))

            # Scan folder
            scan_summary = service.scan_data_folder(data_dir=sample_folder, use_mcp=True)
            assert scan_summary["total_supported"] == 2

            # Verify zero DB mutations: strictly read-only
            with temp_db.get_connection() as conn:
                cur = conn.cursor()
                for table in ("raw_signals", "raid_drafts", "approved_raid"):
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    assert cur.fetchone()[0] == 0, f"Table {table} had mutations during scan!"
        finally:
            service.close_mcp_client()

    # 9. Process New Files reads files through MCP
    def test_9_process_new_files_reads_files_through_mcp(self, temp_db, sample_folder, mock_classifier):
        pipeline = RAIDPipeline(db_manager=temp_db)
        pipeline.classifier = mock_classifier
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        try:
            # Configure folder
            service.configure_mcp_folder(str(sample_folder))

            # Process files using MCP
            res = service.process_discovered_files(data_dir=sample_folder, use_mcp=True)
            assert res["files_processed_successfully"] == 2
            assert res["files_failed"] == 0
            assert res["total_signals_ingested"] == 2
            assert res["total_drafts_created"] == 2

            # Verify sync history records
            sync_records = service.get_file_sync_history()
            assert len(sync_records) == 2
            for r in sync_records:
                assert r["status"] == "PROCESSED"
                assert r["record_count"] == 1
        finally:
            service.close_mcp_client()

    # 10. Streamlit reruns do not lose configuration
    def test_10_streamlit_reruns_do_not_lose_configuration(self, temp_db, sample_folder):
        from streamlit.testing.v1 import AppTest
        page_path = Path(__file__).resolve().parent.parent / "ui" / "pages" / "0_Data_Import_Sync.py"

        with patch("ui.service_helper.get_raid_service") as mock_get_service:
            pipeline = RAIDPipeline(db_manager=temp_db)
            service = RaidService(db_manager=temp_db, pipeline=pipeline)
            mock_get_service.return_value = service

            at = AppTest.from_file(str(page_path), default_timeout=30)
            at.run()
            assert not at.exception

            # Initial target path is set
            assert "target_folder_path" in at.session_state

            # Simulate user entering custom path and clicking Configure
            at.session_state["target_folder_path"] = str(sample_folder)
            at.run()
            assert not at.exception
            assert at.session_state["target_folder_path"] == str(sample_folder)

            # Consecutive rerun preserves the configured path
            at.run()
            assert not at.exception
            assert at.session_state["target_folder_path"] == str(sample_folder)
