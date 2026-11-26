"""
Stage 7 Automated Tests: Data Folder Ingestion, Database Synchronisation & UI Integration.
Tests:
- DataFolderScanner discovery across all 5 source types
- Safe skipping of DBs, backups, validation summaries, and hidden files
- Duplicate prevention via SHA-256 hashing (skipping unchanged files)
- Modified file detection and re-scan
- Corrupt / unreadable file error handling
- End-to-end pipeline execution with mocked LLM provider
- UI page rendering with Streamlit AppTest
- Tenant isolation across sync records
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.database.db_manager import DatabaseManager
from src.normalisation.schemas import SourceType, ProjectSignal
from src.ingestion.folder_scanner import DataFolderScanner, compute_file_hash
from src.pipeline import RAIDPipeline
from src.service import RaidService
from src.classification.schemas import ClassificationOutcome, ClassificationStatus, RAIDCategory, RAIDExtractionResult


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated test SQLite database."""
    db_file = tmp_path / "test_stage7.db"
    db = DatabaseManager(db_file)
    return db


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


@pytest.fixture
def sample_data_folder(tmp_path):
    """Creates a realistic test data folder with all 5 source types and noise files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # 1. Emails
    emails_dir = data_dir / "raw" / "emails"
    emails_dir.mkdir(parents=True)
    (emails_dir / "emails.csv").write_text(
        "id,date,sender,recipient,subject,body\n"
        "MSG-101,2026-11-01,lead@test.com,dev@test.com,Cutover Risk,Cutover window may slip.\n"
        "MSG-102,2026-11-02,tester@test.com,lead@test.com,Routine Update,All tests running normally.\n"
    )

    # 2. Chats
    chats_dir = data_dir / "raw" / "chats"
    chats_dir.mkdir(parents=True)
    (chats_dir / "chats.csv").write_text(
        "chat_id,timestamp,author,channel,text\n"
        "CHAT-101,2026-11-03T10:00:00Z,dev1@test.com,general,DB migration taking longer than planned.\n"
    )

    # 3. Meeting Notes
    mom_dir = data_dir / "raw" / "meeting_notes"
    mom_dir.mkdir(parents=True)
    (mom_dir / "mom.csv").write_text(
        "MeetingId,Organizer,Date,Title,Notes,Workstream,Phase\n"
        "MOM-101,PMO Lead,2026-11-04,SteerCo,Decided on brownfield upgrade,FI,Realize\n"
    )

    # 4. Project Plans
    plans_dir = data_dir / "raw" / "project_plans"
    plans_dir.mkdir(parents=True)
    (plans_dir / "plan.csv").write_text(
        "Task ID,Task Name,Workstream,Phase,Baseline Finish,Actual Finish,Status,Owner\n"
        "TSK-201,Remediate Custom Code,ABAP,Realize,2026-11-01,,Delayed,Dev Team\n"
    )

    # 5. Defects
    defects_dir = data_dir / "raw" / "defects"
    defects_dir.mkdir(parents=True)
    (defects_dir / "defects.csv").write_text(
        "Defect ID,Summary,Module,Severity,Status,Reporter\n"
        "DEF-301,BAPI dump on batch processing,MM,High,Open,QA Lead\n"
    )

    # Excluded noise files: database, backup, summary json
    db_dir = data_dir / "db"
    db_dir.mkdir(parents=True)
    (db_dir / "dummy.db").write_text("DUMMY_SQLITE_BINARY")
    (db_dir / "dummy.db.bak").write_text("BACKUP")
    (data_dir / "validation_summary.json").write_text("{\"status\": \"passed\"}")
    (data_dir / "test_results.json").write_text("{\"passed\": 10}")

    return data_dir


class TestDataFolderScanner:
    """Tests for file discovery, extension routing, and deduplication detection."""

    def test_scanner_discovers_all_supported_sources(self, temp_db, sample_data_folder):
        scanner = DataFolderScanner(db_manager=temp_db, data_dir=sample_data_folder)
        summary = scanner.scan_data_folder(target_dir=sample_data_folder)

        assert summary["total_supported"] == 5
        assert summary["new_count"] == 5
        assert summary["processed_count"] == 0
        assert summary["modified_count"] == 0
        assert summary["failed_count"] == 0

        source_types = {f["source_type"] for f in summary["files"]}
        expected_types = {"email", "chat", "meeting_notes", "project_plan", "defect"}
        assert source_types == expected_types

    def test_scanner_safely_excludes_non_source_files(self, temp_db, sample_data_folder):
        scanner = DataFolderScanner(db_manager=temp_db, data_dir=sample_data_folder)
        summary = scanner.scan_data_folder(target_dir=sample_data_folder)

        paths = [f["file_path"] for f in summary["files"]]
        for p in paths:
            assert ".db" not in p
            assert ".bak" not in p
            assert "validation_summary" not in p
            assert "test_results" not in p

    def test_duplicate_prevention_unchanged_files(self, temp_db, sample_data_folder):
        scanner = DataFolderScanner(db_manager=temp_db, data_dir=sample_data_folder)
        s1 = scanner.scan_data_folder(target_dir=sample_data_folder)
        assert s1["new_count"] == 5

        # Mark all as PROCESSED in DB
        for f in s1["files"]:
            temp_db.save_file_sync_record(f)
            temp_db.update_file_sync_status(f["file_path"], status="PROCESSED", record_count=1)

        # Re-scan: unchanged files must be marked PROCESSED
        s2 = scanner.scan_data_folder(target_dir=sample_data_folder)
        assert s2["new_count"] == 0
        assert s2["processed_count"] == 5
        assert s2["modified_count"] == 0

    def test_modified_file_detection(self, temp_db, sample_data_folder):
        scanner = DataFolderScanner(db_manager=temp_db, data_dir=sample_data_folder)
        s1 = scanner.scan_data_folder(target_dir=sample_data_folder)

        for f in s1["files"]:
            temp_db.save_file_sync_record(f)
            temp_db.update_file_sync_status(f["file_path"], status="PROCESSED")

        # Modify email file
        email_file = sample_data_folder / "raw" / "emails" / "emails.csv"
        email_file.write_text(email_file.read_text() + "MSG-103,2026-11-03,lead@test.com,dev@test.com,New Risk,Another risk.\n")

        s2 = scanner.scan_data_folder(target_dir=sample_data_folder)
        assert s2["modified_count"] == 1
        assert s2["processed_count"] == 4

        mod_file = s2["modified_files"][0]
        assert "emails.csv" in mod_file["file_name"]
        assert mod_file["status"] == "MODIFIED"

    def test_corrupt_file_handling(self, temp_db, sample_data_folder):
        # Create unreadable file (simulate permission error)
        corrupt_file = sample_data_folder / "raw" / "emails" / "corrupted_email.csv"
        corrupt_file.write_text("INVALID_CONTENT")
        os.chmod(corrupt_file, 0o000)

        try:
            scanner = DataFolderScanner(db_manager=temp_db, data_dir=sample_data_folder)
            summary = scanner.scan_data_folder(target_dir=sample_data_folder)
            # Other files must still be discovered cleanly without crash
            assert summary["total_supported"] >= 5
        finally:
            os.chmod(corrupt_file, 0o644)


class TestServiceSynchronisation:
    """Tests for RaidService.scan_data_folder and process_discovered_files."""

    def test_service_process_discovered_files(self, temp_db, sample_data_folder, mock_classifier):
        pipeline = RAIDPipeline(db_manager=temp_db)
        pipeline.classifier = mock_classifier
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        # Execute processing
        results = service.process_discovered_files(data_dir=sample_data_folder)

        assert results["files_targeted"] == 5
        assert results["files_processed_successfully"] == 5
        assert results["files_failed"] == 0
        assert results["total_signals_ingested"] > 0
        assert results["total_drafts_created"] > 0

        # Check DB sync records
        sync_records = service.get_file_sync_history()
        assert len(sync_records) == 5
        for rec in sync_records:
            assert rec["status"] == "PROCESSED"
            assert rec["record_count"] > 0

    def test_reprocessing_skips_unchanged_files(self, temp_db, sample_data_folder, mock_classifier):
        pipeline = RAIDPipeline(db_manager=temp_db)
        pipeline.classifier = mock_classifier
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        # First run: processes 5 files
        res1 = service.process_discovered_files(data_dir=sample_data_folder)
        assert res1["files_targeted"] == 5

        # Second run: zero files targeted (all unchanged and PROCESSED)
        res2 = service.process_discovered_files(data_dir=sample_data_folder)
        assert res2["files_targeted"] == 0
        assert res2["total_signals_ingested"] == 0

    def test_tenant_isolation_on_sync_records(self, temp_db, sample_data_folder, mock_classifier):
        pipeline = RAIDPipeline(db_manager=temp_db)
        pipeline.classifier = mock_classifier
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        # Process for tenant A
        service.process_discovered_files(data_dir=sample_data_folder, user_id="tenant_a")
        # Process for tenant B
        service.process_discovered_files(data_dir=sample_data_folder, user_id="tenant_b")

        records_a = temp_db.get_all_file_sync_records(user_id="tenant_a")
        records_b = temp_db.get_all_file_sync_records(user_id="tenant_b")

        assert len(records_a) == 5
        assert len(records_b) == 5
        assert all(r["user_id"] == "tenant_a" for r in records_a)
        assert all(r["user_id"] == "tenant_b" for r in records_b)


class TestUIIntegration:
    """Streamlit AppTest integration for the Data Import & Sync page."""

    def test_render_data_import_page(self, temp_db, sample_data_folder):
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
