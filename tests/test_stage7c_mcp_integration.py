"""
Stage 7C Automated Test Suite: Local Folder Integration Through MCP.
Tests the 14 requirements specified in Stage 7C:
1. MCP Server startup.
2. MCP Client connection.
3. Tool discovery.
4. Folder selection/path validation (security & path traversal).
5. Folder scanning (read-only, no LLM calls).
6. Reading supported file types (text & base64).
7. Unsupported and corrupted files handling.
8. New-file detection.
9. Unchanged-file skipping.
10. Modified-file detection.
11. File registry updates.
12. Existing ingestion pipeline integration with mock classifier.
13. MCP connection failure handling & graceful fallback.
14. Streamlit rerun and session-state behaviour.
"""

import os
import sys
import base64
import shutil
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.database.db_manager import DatabaseManager
from src.normalisation.schemas import SourceType, ProjectSignal
from src.pipeline import RAIDPipeline
from src.service import RaidService
from src.classification.schemas import ClassificationOutcome, ClassificationStatus, RAIDCategory, RAIDExtractionResult
from src.mcp.client import MCPClient, MCPConnectionError
from src.mcp.tools.local_file_tools import (
    select_or_configure_folder,
    scan_folder,
    read_file,
    get_file_status,
    validate_path_safety,
    MAX_FILE_SIZE_BYTES
)


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated test SQLite database."""
    db_file = tmp_path / "test_stage7c.db"
    return DatabaseManager(db_file)


@pytest.fixture
def sample_mcp_folder(tmp_path):
    """Creates a sample folder tree with supported project files and noise files."""
    data_dir = tmp_path / "project_data"
    data_dir.mkdir(parents=True)

    # 1. Emails
    emails_dir = data_dir / "emails"
    emails_dir.mkdir()
    (emails_dir / "email1.csv").write_text("id,sender,recipient,subject,body\n1,a@b.com,c@d.com,Go-Live Risk,Cutover window slips.\n")

    # 2. Chats
    chats_dir = data_dir / "chats"
    chats_dir.mkdir()
    (chats_dir / "chat1.csv").write_text("chat_id,timestamp,author,channel,text\n1,2026-11-01,dev1,general,DB migration takes longer.\n")

    # 3. Meeting Notes
    mom_dir = data_dir / "meeting_notes"
    mom_dir.mkdir()
    (mom_dir / "mom1.csv").write_text("MeetingId,Organizer,Date,Title,Notes\n1,PMO,2026-11-01,SteerCo,Decision on ERP upgrade.\n")

    # 4. Project Plans
    plans_dir = data_dir / "project_plans"
    plans_dir.mkdir()
    (plans_dir / "plan1.csv").write_text("Task ID,Task Name,Workstream,Phase,Baseline Finish,Actual Finish,Status,Owner\n1,Custom Code,ABAP,Realize,2026-11-01,,Delayed,Dev Team\n")

    # 5. Defects
    defects_dir = data_dir / "defects"
    defects_dir.mkdir()
    (defects_dir / "defects1.csv").write_text("Defect ID,Summary,Module,Severity,Status,Reporter\n1,BAPI dump,MM,High,Open,QA Lead\n")

    # Excluded noise files
    db_dir = data_dir / "db"
    db_dir.mkdir()
    (db_dir / "test.db").write_text("DB_BINARY")
    (data_dir / "summary.json").write_text('{"status": "ok"}')
    (data_dir / "code.py").write_text("print('not data')")

    return data_dir


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
                title=f"Extracted Risk from {sig.subject}",
                description=sig.content,
                workstream=sig.workstream or "General",
                phase=sig.phase or "Realize",
                probability="Medium",
                impact="Moderate project delay",
                evidence=sig.content[:60],
                owner="Integration Lead"
            ),
            llm_provider="test_mock",
            llm_model="test_model"
        )

    classifier.classify_signal_detailed.side_effect = fake_classify
    return classifier


class TestStage7CMCPIntegration:
    """Test suite for MCP Server, Client, Tools, and RAID Pipeline integration."""

    # 1. MCP Server startup
    def test_1_mcp_server_startup(self):
        from src.mcp.server import server
        assert server.name == "raid_local_file_server"

    # 2. MCP Client connection
    def test_2_mcp_client_connection(self):
        client = MCPClient()
        try:
            assert client.is_healthy()
        finally:
            client.close()

    # 3. Tool discovery
    def test_3_tool_discovery(self):
        client = MCPClient()
        try:
            tools = client.list_tools()
            tool_names = {t["name"] for t in tools}
            expected = {"select_or_configure_folder", "scan_folder", "read_file", "get_file_status"}
            assert expected.issubset(tool_names)
        finally:
            client.close()

    # 4. Folder selection/path validation
    def test_4_folder_selection_and_path_validation(self, sample_mcp_folder, tmp_path):
        client = MCPClient()
        try:
            # Valid directory
            res = client.configure_folder(str(sample_mcp_folder))
            assert res.get("success") is True
            assert res.get("folder_path") == str(sample_mcp_folder.resolve())

            # Non-existent directory
            res_bad = client.configure_folder(str(tmp_path / "non_existent_folder_123"))
            assert res_bad.get("success") is False
            assert "does not exist" in res_bad.get("error", "").lower()

            # Path traversal / forbidden roots
            res_root = client.configure_folder("/etc")
            assert res_root.get("success") is False
            assert "prohibited" in res_root.get("error", "").lower()
        finally:
            client.close()

    # 5. Folder scanning
    def test_5_folder_scanning(self, sample_mcp_folder):
        client = MCPClient()
        try:
            res = client.scan_folder(folder_path=str(sample_mcp_folder))
            assert res.get("success") is True
            assert res.get("total_supported") == 5
            assert res.get("new_count") == 5
            assert res.get("processed_count") == 0

            # Verify safe exclusion of .db and .py
            files = [f["file_name"] for f in res.get("files", [])]
            assert "test.db" not in files
            assert "code.py" not in files
        finally:
            client.close()

    # 6. Reading supported file types
    def test_6_reading_supported_file_types(self, sample_mcp_folder):
        client = MCPClient()
        try:
            client.configure_folder(str(sample_mcp_folder))

            # Test text file (CSV email)
            email_path = sample_mcp_folder / "emails" / "email1.csv"
            res_text = client.read_file(str(email_path))
            assert res_text.get("success") is True
            assert res_text.get("encoding") == "utf-8"
            assert "Go-Live Risk" in res_text.get("content", "")
            assert res_text.get("content_hash") is not None

            # Test binary file simulation (.docx)
            docx_path = sample_mcp_folder / "meeting_notes" / "dummy.docx"
            docx_path.write_bytes(b"PK\x03\x04DUMMY_DOCX_BYTES")
            res_bin = client.read_file(str(docx_path))
            assert res_bin.get("success") is True
            assert res_bin.get("encoding") == "base64"
            assert res_bin.get("extension") == ".docx"
            # Content should decode back to original bytes
            decoded = base64.b64decode(res_bin["content"])
            assert decoded == b"PK\x03\x04DUMMY_DOCX_BYTES"
        finally:
            client.close()

    # 7. Unsupported and corrupted files
    def test_7_unsupported_and_corrupted_files(self, sample_mcp_folder):
        client = MCPClient()
        try:
            client.configure_folder(str(sample_mcp_folder))

            # Unsupported file
            res_unsupported = client.read_file(str(sample_mcp_folder / "code.py"))
            assert res_unsupported.get("success") is False
            assert "unsupported" in res_unsupported.get("error", "").lower()

            # Corrupted / unreadable file
            corrupt = sample_mcp_folder / "emails" / "unreadable.csv"
            corrupt.write_text("content")
            os.chmod(corrupt, 0o000)
            try:
                res_corrupt = client.read_file(str(corrupt))
                assert res_corrupt.get("success") is False
            finally:
                os.chmod(corrupt, 0o644)
        finally:
            client.close()

    # 8. New-file detection
    def test_8_new_file_detection(self, sample_mcp_folder):
        client = MCPClient()
        try:
            scan = client.scan_folder(folder_path=str(sample_mcp_folder))
            assert scan.get("new_count") == 5
            for f in scan.get("new_files", []):
                assert f.get("status") == "DISCOVERED"
        finally:
            client.close()

    # 9. Unchanged-file skipping
    def test_9_unchanged_file_skipping(self, temp_db, sample_mcp_folder):
        # Save one file as PROCESSED in DB
        email_file = sample_mcp_folder / "emails" / "email1.csv"
        scanner_res = scan_folder(folder_path=str(sample_mcp_folder), db_manager=temp_db)
        f_info = next(f for f in scanner_res["files"] if "email1.csv" in f["file_name"])

        temp_db.save_file_sync_record(f_info)
        temp_db.update_file_sync_status(f_info["file_path"], status="PROCESSED", record_count=1)

        # Scan with temp_db
        res2 = scan_folder(folder_path=str(sample_mcp_folder), db_manager=temp_db)
        assert res2["processed_count"] == 1
        assert res2["new_count"] == 4

        proc_f = next(f for f in res2["files"] if "email1.csv" in f["file_name"])
        assert proc_f["status"] == "PROCESSED"

    # 10. Modified-file detection
    def test_10_modified_file_detection(self, temp_db, sample_mcp_folder):
        scanner_res = scan_folder(folder_path=str(sample_mcp_folder), db_manager=temp_db)
        f_info = next(f for f in scanner_res["files"] if "email1.csv" in f["file_name"])

        temp_db.save_file_sync_record(f_info)
        temp_db.update_file_sync_status(f_info["file_path"], status="PROCESSED")

        # Modify content on disk
        email_file = sample_mcp_folder / "emails" / "email1.csv"
        email_file.write_text(email_file.read_text() + "2,x@y.com,z@w.com,Additional Risk,Body\n")

        res2 = scan_folder(folder_path=str(sample_mcp_folder), db_manager=temp_db)
        assert res2["modified_count"] == 1
        mod_f = next(f for f in res2["files"] if "email1.csv" in f["file_name"])
        assert mod_f["status"] == "MODIFIED"

    # 11. File registry updates
    def test_11_file_registry_updates(self, temp_db, sample_mcp_folder):
        client = MCPClient()
        try:
            # Check unsaved file
            st1 = get_file_status("dummy_path.csv", db_manager=temp_db)
            assert st1.get("status") == "DISCOVERED"

            # Save record and check
            rec = {
                "file_path": "test/path.csv",
                "file_name": "path.csv",
                "source_type": "email",
                "file_size": 100,
                "content_hash": "abc123hash",
                "last_modified": "2026-11-01T00:00:00Z",
                "record_count": 5,
                "signals_created": 5,
                "signals_classified": 5,
                "drafts_created": 2,
                "status": "PROCESSED",
                "error_message": None,
                "user_id": "default_user"
            }
            temp_db.save_file_sync_record(rec)
            st2 = get_file_status("test/path.csv", db_manager=temp_db)
            assert st2.get("status") == "PROCESSED"
            assert st2.get("drafts_created") == 2
        finally:
            client.close()

    # 12. Existing ingestion pipeline integration
    def test_12_existing_ingestion_pipeline_integration(self, temp_db, sample_mcp_folder, mock_classifier):
        pipeline = RAIDPipeline(db_manager=temp_db)
        pipeline.classifier = mock_classifier
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        try:
            # Process files using MCP
            res = service.process_discovered_files(
                data_dir=sample_mcp_folder,
                use_mcp=True
            )
            assert res["files_processed_successfully"] == 5
            assert res["files_failed"] == 0
            assert res["total_signals_ingested"] > 0
            assert res["total_drafts_created"] > 0

            # Check DB sync records
            sync_records = service.get_file_sync_history()
            assert len(sync_records) == 5
            for r in sync_records:
                assert r["status"] == "PROCESSED"
        finally:
            service.close_mcp_client()

    # 13. MCP connection failure handling
    def test_13_mcp_connection_failure_handling(self, temp_db, sample_mcp_folder):
        service = RaidService(db_manager=temp_db)
        # Mock client to simulate connection failure
        broken_client = MagicMock()
        broken_client.scan_folder.side_effect = MCPConnectionError("Connection lost")
        service._mcp_client = broken_client

        # Should safely fall back to direct scanner without raising exception
        res = service.scan_data_folder(data_dir=sample_mcp_folder, use_mcp=True)
        assert res.get("total_supported") == 5

    # 14. Streamlit rerun and session-state behaviour
    def test_14_streamlit_rerun_and_session_state_behaviour(self, temp_db, sample_mcp_folder):
        from streamlit.testing.v1 import AppTest
        page_path = Path(__file__).resolve().parent.parent / "ui" / "pages" / "0_Data_Import_Sync.py"

        with patch("ui.service_helper.get_raid_service") as mock_get_service:
            pipeline = RAIDPipeline(db_manager=temp_db)
            service = RaidService(db_manager=temp_db, pipeline=pipeline)
            mock_get_service.return_value = service

            at = AppTest.from_file(str(page_path), default_timeout=30)
            at.run()
            assert not at.exception
            assert "Data Import & Synchronisation" in at.title[0].value
            assert len(at.subheader) >= 3

            # Simulate rerun
            at.run()
            assert not at.exception
