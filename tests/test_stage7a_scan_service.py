"""
Stage 7A Automated Tests: Service-Layer Data Folder Scanner Integration.
Verifies the 11 explicit requirements from Stage 7A Phase 4:
1. RaidService exposes scan_data_folder()
2. Valid supported files across all 5 source types are discovered
3. Unsupported files are skipped
4. Database files (.db, .bak, etc.) are excluded
5. Missing data folder is handled safely
6. Empty folder returns a valid empty result
7. Corrupted/unreadable files are reported safely without crash
8. Duplicate files are identified using content hash
9. Scan does NOT modify the database (strict read-only guarantee)
10. Scan does NOT call the LLM (zero classifier invocations)
11. Return structure contains every field consumed by the UI
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.database.db_manager import DatabaseManager
from src.pipeline import RAIDPipeline
from src.service import RaidService
from src.normalisation.schemas import SourceType


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated test SQLite database."""
    db_file = tmp_path / "test_stage7a.db"
    return DatabaseManager(db_file)


@pytest.fixture
def sample_data_tree(tmp_path):
    """Creates a sample data directory tree containing supported, unsupported, and db files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # 1. Supported: Email
    email_dir = data_dir / "raw" / "emails"
    email_dir.mkdir(parents=True)
    (email_dir / "test_emails.csv").write_text("id,sender,recipient,subject,body\n1,a@b.c,d@e.f,Test,Hello\n")

    # 2. Supported: Chat
    chat_dir = data_dir / "raw" / "chats"
    chat_dir.mkdir(parents=True)
    (chat_dir / "team_chat.csv").write_text("chat_id,timestamp,author,channel,text\n1,2026-11-01,u1,dev,Delay\n")

    # 3. Supported: Meeting Notes
    mom_dir = data_dir / "raw" / "meeting_notes"
    mom_dir.mkdir(parents=True)
    (mom_dir / "steerco_mom.csv").write_text("MeetingId,Organizer,Date,Title,Notes\n1,PMO,2026-11-01,Sync,Notes\n")

    # 4. Supported: Project Plan
    plan_dir = data_dir / "raw" / "project_plans"
    plan_dir.mkdir(parents=True)
    (plan_dir / "project_plan.csv").write_text("Task ID,Task Name,Workstream,Phase,Baseline Finish,Actual Finish,Status,Owner\n1,T1,FI,Realize,2026-11-01,,Open,Team\n")

    # 5. Supported: Defect
    defect_dir = data_dir / "raw" / "defects"
    defect_dir.mkdir(parents=True)
    (defect_dir / "bug_register.csv").write_text("Defect ID,Summary,Module,Severity,Status,Reporter\n1,Crash,MM,High,Open,Tester\n")

    # Unsupported files
    (data_dir / "script.py").write_text("print('not a data file')")
    (data_dir / "notes.unsupported_ext").write_text("random")

    # Database / Backup files
    db_dir = data_dir / "db"
    db_dir.mkdir(parents=True)
    (db_dir / "production.db").write_text("SQLITE_DUMMY")
    (db_dir / "backup.db.bak").write_text("BAK_DUMMY")
    (data_dir / "test_results.json").write_text('{"status": "ok"}')

    return data_dir


class TestStage7AScanService:
    """Test suite for RaidService.scan_data_folder integration."""

    def test_requirement_1_service_exposes_scan_data_folder(self, temp_db):
        """1. RaidService exposes scan_data_folder()."""
        service = RaidService(db_manager=temp_db)
        assert hasattr(service, "scan_data_folder")
        assert callable(service.scan_data_folder)

    def test_requirement_2_valid_supported_files_discovered(self, temp_db, sample_data_tree):
        """2. Valid supported files across all 5 source types are discovered."""
        service = RaidService(db_manager=temp_db)
        result = service.scan_data_folder(data_dir=sample_data_tree)

        assert result["total_supported"] == 5
        discovered_types = {f["source_type"] for f in result["files"]}
        assert discovered_types == {"email", "chat", "meeting_notes", "project_plan", "defect"}

    def test_requirement_3_unsupported_files_skipped(self, temp_db, sample_data_tree):
        """3. Unsupported files are skipped or listed in unsupported_files."""
        service = RaidService(db_manager=temp_db)
        result = service.scan_data_folder(data_dir=sample_data_tree)

        discovered_paths = [f["file_path"] for f in result["files"]]
        assert not any("script.py" in p for p in discovered_paths)
        assert not any("notes.unsupported_ext" in p for p in discovered_paths)

        # Check unsupported list
        unsupported_names = [f["file_name"] for f in result.get("unsupported_files", [])]
        assert "notes.unsupported_ext" in unsupported_names

    def test_requirement_4_database_files_excluded(self, temp_db, sample_data_tree):
        """4. Database files (.db, .bak, etc.) and json test summaries are excluded."""
        service = RaidService(db_manager=temp_db)
        result = service.scan_data_folder(data_dir=sample_data_tree)

        discovered_paths = [f["file_path"] for f in result["files"]]
        assert not any("production.db" in p for p in discovered_paths)
        assert not any("backup.db.bak" in p for p in discovered_paths)
        assert not any("test_results.json" in p for p in discovered_paths)

    def test_requirement_5_missing_data_folder_handled_safely(self, temp_db, tmp_path):
        """5. Missing data folder is handled safely without throwing an exception."""
        non_existent_dir = tmp_path / "does_not_exist"
        service = RaidService(db_manager=temp_db)

        # Must not raise
        result = service.scan_data_folder(data_dir=non_existent_dir)
        assert result["total_supported"] == 0
        assert result["files"] == []
        assert result["new_count"] == 0

    def test_requirement_6_empty_folder_returns_valid_structure(self, temp_db, tmp_path):
        """6. Empty folder returns a valid empty result structure."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        service = RaidService(db_manager=temp_db)

        result = service.scan_data_folder(data_dir=empty_dir)
        assert result["total_supported"] == 0
        assert result["new_count"] == 0
        assert result["processed_count"] == 0
        assert result["modified_count"] == 0
        assert result["failed_count"] == 0
        assert isinstance(result["files"], list)
        assert len(result["files"]) == 0

    def test_requirement_7_corrupted_unreadable_file_handled_safely(self, temp_db, sample_data_tree):
        """7. Corrupted / unreadable files are reported safely without crash."""
        corrupt_file = sample_data_tree / "raw" / "emails" / "corrupted_email.csv"
        corrupt_file.write_text("CORRUPTED")
        os.chmod(corrupt_file, 0o000)

        try:
            service = RaidService(db_manager=temp_db)
            result = service.scan_data_folder(data_dir=sample_data_tree)
            assert result["total_supported"] >= 5
            # Corrupted file should be captured in failed_files
            assert any("corrupted_email.csv" in f.get("file_name", "") for f in result.get("failed_files", []))
        finally:
            os.chmod(corrupt_file, 0o644)

    def test_requirement_8_duplicate_identification_by_hash(self, temp_db, sample_data_tree):
        """8. Duplicate files are identified using content hash."""
        service = RaidService(db_manager=temp_db)
        r1 = service.scan_data_folder(data_dir=sample_data_tree)
        assert r1["new_count"] == 5

        # Simulate one file processed
        f_email = next(f for f in r1["files"] if "test_emails.csv" in f["file_name"])
        temp_db.save_file_sync_record(f_email)
        temp_db.update_file_sync_status(f_email["file_path"], status="PROCESSED", record_count=1)

        # Re-scan: the processed file should be recognized by matching hash
        r2 = service.scan_data_folder(data_dir=sample_data_tree)
        assert r2["processed_count"] == 1
        assert r2["new_count"] == 4

        proc_file = next(f for f in r2["files"] if "test_emails.csv" in f["file_name"])
        assert proc_file["status"] == "PROCESSED"
        assert proc_file["content_hash"] == f_email["content_hash"]

    def test_requirement_9_scan_does_not_modify_database(self, temp_db, sample_data_tree):
        """9. Scan does NOT modify the database (strict read-only guarantee)."""
        def get_all_table_counts():
            with temp_db.get_connection() as conn:
                cursor = conn.cursor()
                counts = {}
                for tbl in ("raw_signals", "raid_drafts", "approved_raid", "audit_history", "source_file_sync"):
                    cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                    counts[tbl] = cursor.fetchone()[0]
                return counts

        before_counts = get_all_table_counts()

        service = RaidService(db_manager=temp_db)
        # Perform scan
        result = service.scan_data_folder(data_dir=sample_data_tree)
        assert result["total_supported"] == 5

        after_counts = get_all_table_counts()

        # Database must remain 100% unchanged
        assert before_counts == after_counts, f"Database was modified during scan: before={before_counts}, after={after_counts}"

    def test_requirement_10_scan_does_not_call_llm(self, temp_db, sample_data_tree):
        """10. Scan does NOT call the LLM (zero classifier invocations)."""
        mock_pipeline = MagicMock()
        mock_classifier = MagicMock()
        mock_pipeline.classifier = mock_classifier

        service = RaidService(db_manager=temp_db, pipeline=mock_pipeline)
        result = service.scan_data_folder(data_dir=sample_data_tree)

        assert result["total_supported"] == 5
        mock_classifier.classify_signal_detailed.assert_not_called()
        mock_pipeline.process_signals.assert_not_called()

    def test_requirement_11_return_structure_consumed_by_ui(self, temp_db, sample_data_tree):
        """11. Return structure contains every field consumed by the UI."""
        service = RaidService(db_manager=temp_db)
        result = service.scan_data_folder(data_dir=sample_data_tree)

        # Top-level keys required by 0_Data_Import_Sync.py
        required_top_level_keys = {
            "data_dir",
            "total_supported",
            "new_count",
            "processed_count",
            "modified_count",
            "failed_count",
            "files",
            "new_files",
            "processed_files",
            "modified_files",
            "failed_files",
            "unsupported_files"
        }
        for k in required_top_level_keys:
            assert k in result, f"Missing required top-level key: {k}"

        # Per-file keys consumed by UI tables and metrics
        required_file_keys = {
            "file_name",
            "file_path",
            "source_type",
            "file_size",
            "content_hash",
            "last_modified",
            "status",
            "record_count",
            "signals_created",
            "drafts_created",
            "error_message"
        }
        for f in result["files"]:
            for fk in required_file_keys:
                assert fk in f, f"File entry {f.get('file_name')} missing required key: {fk}"
