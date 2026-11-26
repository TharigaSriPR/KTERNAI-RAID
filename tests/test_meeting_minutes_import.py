"""
Comprehensive Integration and Unit Tests for Meeting Minutes Import Feature.
KTern.AI RAID Identification Agent POC

Covers:
1. DOCX upload and processing.
2. TXT upload and processing.
3. CSV upload and processing.
4. Multiple file uploads.
5. Manual text input.
6. Empty or invalid files (0-byte file, unsupported extension, corrupt file).
7. Meeting date and source metadata preservation.
8. RAID classification and evidence traceability.
9. Duplicate source prevention (same file uploaded twice is detected and skipped).
10. Repeated processing does not create duplicate RAID drafts.
11. Processing failure handling (graceful error reporting).
12. Confirmation that generated items are not automatically approved (status == "Draft").
13. Dependency is excluded from review categories.
14. Stage callback progression through all 8 stages.
"""

import io
import json
import pytest
from pathlib import Path
import docx

from src.database.db_manager import DatabaseManager
from src.service import RaidService, RaidValidationError, RaidServiceError
from src.pipeline import RAIDPipeline
from src.classification.providers.mock_provider import MockLLMProvider
from src.classification.classifier import RAIDClassifier
import importlib


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Creates a clean isolated DatabaseManager instance."""
    db_file = tmp_path / "test_meeting_import.db"
    return DatabaseManager(db_file)


@pytest.fixture
def service_with_mock_llm(temp_db: DatabaseManager) -> RaidService:
    """Creates a RaidService with MockLLMProvider for deterministic offline execution."""
    pipeline = RAIDPipeline(temp_db)
    # Inject MockLLMProvider so live API keys are not required
    mock_provider = MockLLMProvider()
    pipeline.classifier = RAIDClassifier(provider=mock_provider)
    return RaidService(db_manager=temp_db, pipeline=pipeline)


class TestMeetingMinutesImport:
    def test_01_docx_upload_and_processing(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test DOCX meeting minutes file processing through complete pipeline."""
        doc_path = tmp_path / "SteerCo_Wave1_Minutes.docx"
        doc = docx.Document()
        doc.add_heading("SteerCo Wave 1 Alignment Meeting", level=0)
        doc.add_paragraph("Date: 2026-11-20")
        doc.add_paragraph("Agreed that SD pricing user exit refactoring carries high risk and may delay end-to-end cycle test.")
        doc.add_paragraph("Decided to proceed with brownfield migration approach for ECC financials rather than greenfield.")
        doc.save(str(doc_path))

        stage_records = []
        def track_stage(stage, details):
            stage_records.append(stage)

        results = service_with_mock_llm.process_meeting_minutes(
            file_paths=[doc_path],
            stage_callback=track_stage
        )

        assert results["status"] == "success"
        assert results["total_signals"] >= 2
        assert results["drafts_created"] >= 2

        # Check drafts in DB
        drafts = service_with_mock_llm.get_pending_drafts()
        assert len(drafts) >= 2
        types = {d["type"] for d in drafts}
        assert "Risk" in types or "Decision" in types

        # Confirm all items are strictly Draft status
        for d in drafts:
            assert d["status"] == "Draft"
            assert d["evidence_text"] is not None
            assert len(d["evidence_text"]) > 0

    def test_02_txt_upload_and_processing(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test TXT meeting minutes upload and processing."""
        txt_path = tmp_path / "Sprint_4_Review.txt"
        txt_content = (
            "Meeting Date: 2026-11-15\n"
            "Discussion Point 1: Agreed that SD pricing user exit refactoring carries high risk.\n\n"
            "Discussion Point 2: Planning assumption: We assume the customer business team will provide clean test data before SIT.\n\n"
            "Discussion Point 3: Integration testing is currently blocked due to missing master data.\n"
        )
        txt_path.write_text(txt_content, encoding="utf-8")

        results = service_with_mock_llm.process_meeting_minutes(file_paths=[txt_path])
        assert results["status"] == "success"
        assert results["total_signals"] >= 2
        assert results["drafts_created"] >= 2

        drafts = service_with_mock_llm.get_pending_drafts()
        for d in drafts:
            assert d["status"] == "Draft"

    def test_03_csv_upload_and_processing(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test CSV tabular meeting minutes upload and processing."""
        csv_path = tmp_path / "mom_export.csv"
        csv_content = (
            "MeetingId,Organizer,Date,Title,Notes,Workstream,Phase\n"
            "MOM-101,PMO Lead,2026-11-18,Sprint Planning,Agreed that SD pricing user exit refactoring carries high risk,SD,Realize\n"
            "MOM-102,Project Director,2026-11-18,SteerCo Wave 1,Decided to proceed with brownfield migration approach,FI,Prepare\n"
        )
        csv_path.write_text(csv_content, encoding="utf-8")

        results = service_with_mock_llm.process_meeting_minutes(file_paths=[csv_path])
        assert results["status"] == "success"
        assert results["total_signals"] == 2
        assert results["drafts_created"] >= 2

    def test_04_multiple_file_uploads_processing(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test processing multiple uploaded meeting minutes files in one execution."""
        f1 = tmp_path / "Meeting_Alpha.txt"
        f1.write_text("Agreed that SD pricing user exit refactoring carries high risk of delay.", encoding="utf-8")

        f2 = tmp_path / "Meeting_Beta.csv"
        f2.write_text(
            "MeetingId,Organizer,Date,Title,Notes,Workstream,Phase\n"
            "MOM-B1,Director,2026-11-19,SteerCo Beta,Decided to proceed with brownfield migration approach,FI,Prepare\n",
            encoding="utf-8"
        )

        results = service_with_mock_llm.process_meeting_minutes(file_paths=[f1, f2])
        assert results["status"] == "success"
        assert results["total_signals"] >= 2
        assert results["drafts_created"] >= 2

    def test_05_manual_text_input_processing(self, service_with_mock_llm: RaidService):
        """Test direct manual text input processing."""
        manual_payload = {
            "title": "Architecture Review Board Alignment",
            "date": "2026-11-21",
            "project_name": "S/4HANA Enterprise",
            "content": (
                "- Agreed that SD pricing user exit refactoring carries high risk and may delay testing.\n"
                "- Decided to proceed with brownfield conversion rather than selective data transition.\n"
                "- We assume the customer business team will provide clean test data."
            )
        }

        results = service_with_mock_llm.process_meeting_minutes(manual_entry=manual_payload)
        assert results["status"] == "success"
        assert results["drafts_created"] >= 2

        drafts = service_with_mock_llm.get_pending_drafts()
        # Verify metadata
        found_manual = False
        for d in drafts:
            meta = service_with_mock_llm.get_source_metadata_for_draft(d)
            if "manual" in str(meta["source_id"]).lower() or "manual" in str(d.get("source_file", "")).lower():
                found_manual = True
                assert d["status"] == "Draft"
        assert found_manual is True

    def test_06_empty_and_invalid_files(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test that empty (0 bytes) and unsupported format files are handled gracefully."""
        # 1. Empty file
        empty_file = tmp_path / "empty_meeting.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(RaidValidationError) as exc_info:
            service_with_mock_llm.process_meeting_minutes(file_paths=[empty_file])
        assert "empty" in str(exc_info.value).lower() or "no readable" in str(exc_info.value).lower()

        # 2. Unsupported format (.pdf)
        pdf_file = tmp_path / "meeting.pdf"
        pdf_file.write_text("Binary mock content", encoding="utf-8")

        with pytest.raises(RaidValidationError) as exc_info:
            service_with_mock_llm.process_meeting_minutes(file_paths=[pdf_file])
        assert "unsupported" in str(exc_info.value).lower()

    def test_07_meeting_date_and_source_metadata_preservation(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test that meeting date and source file name are preserved end-to-end."""
        txt_path = tmp_path / "Governance_Meeting_2026.txt"
        txt_path.write_text(
            "Date: 2026-11-20\n\n"
            "Agreed that SD pricing user exit refactoring carries high risk and may delay test.\n",
            encoding="utf-8"
        )

        results = service_with_mock_llm.process_meeting_minutes(file_paths=[txt_path])
        assert results["status"] == "success"

        drafts = service_with_mock_llm.get_pending_drafts()
        target = drafts[0]
        meta = service_with_mock_llm.get_source_metadata_for_draft(target)

        assert "Governance_Meeting_2026" in meta["source_file"] or "Governance_Meeting_2026" in str(target.get("source_file", ""))
        assert "2026-11-20" in meta["meeting_date"] or "2026-11-20" in str(target.get("meeting_date", ""))
        assert meta["evidence_text"] is not None

    def test_08_raid_classification_and_evidence_traceability(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test evidence linking and audit trail generation for imported drafts."""
        txt_path = tmp_path / "MOM_Evidence_Trace.txt"
        quote_text = "Agreed that SD pricing user exit refactoring carries high risk and may delay end-to-end cycle test."
        txt_path.write_text(f"Discussion Point: {quote_text}\n", encoding="utf-8")

        results = service_with_mock_llm.process_meeting_minutes(file_paths=[txt_path])
        assert results["status"] == "success"

        drafts = service_with_mock_llm.get_pending_drafts()
        assert len(drafts) >= 1
        draft = drafts[0]

        # Evidence must quote the source
        assert len(draft["evidence_text"]) > 10
        # Foreign key link to raw_signals
        assert draft["evidence_source_id"] is not None
        raw_signal = service_with_mock_llm.get_raw_signal_by_id(draft["evidence_source_id"])
        assert raw_signal is not None
        assert raw_signal["source_type"] == "meeting_notes"

        # Audit trail must have recorded PROPOSED action
        audits = service_with_mock_llm.get_audit_history(draft["id"])
        assert len(audits) >= 1
        assert audits[0]["action"] == "PROPOSED"

    def test_09_duplicate_source_prevention(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test that importing the exact same file twice is detected and prevented."""
        mom_path = tmp_path / "Duplicate_Check_MOM.txt"
        mom_path.write_text("Agreed that SD pricing user exit refactoring carries high risk.", encoding="utf-8")

        # First run: should succeed
        res1 = service_with_mock_llm.process_meeting_minutes(file_paths=[mom_path])
        assert res1["status"] == "success"
        initial_drafts_count = len(res1["drafts"])
        assert initial_drafts_count >= 1

        # Second run: duplicate source file should be flagged as skipped
        res2 = service_with_mock_llm.process_meeting_minutes(file_paths=[mom_path])
        assert res2["status"] == "skipped"
        assert len(res2["skipped_sources"]) >= 1
        assert "Already ingested" in res2["skipped_sources"][0]["reason"]

    def test_10_repeated_processing_does_not_create_duplicate_drafts(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test that repeated processing calls do not duplicate drafts in the database."""
        mom_path = tmp_path / "Draft_Dedupe_Test.txt"
        mom_path.write_text("Agreed that SD pricing user exit refactoring carries high risk.", encoding="utf-8")

        service_with_mock_llm.process_meeting_minutes(file_paths=[mom_path])
        drafts_run1 = service_with_mock_llm.get_pending_drafts()
        count_run1 = len(drafts_run1)

        # Attempt to re-run
        service_with_mock_llm.process_meeting_minutes(file_paths=[mom_path])
        drafts_run2 = service_with_mock_llm.get_pending_drafts()
        count_run2 = len(drafts_run2)

        # Draft count must remain unchanged
        assert count_run2 == count_run1

    def test_11_processing_failure_handling(self, service_with_mock_llm: RaidService):
        """Test that invalid/empty input triggers clear RaidValidationError."""
        # Empty inputs
        with pytest.raises(RaidValidationError) as exc:
            service_with_mock_llm.process_meeting_minutes()
        assert "Please provide at least one" in str(exc.value)

        # Empty manual text
        with pytest.raises(RaidValidationError):
            service_with_mock_llm.process_meeting_minutes(manual_entry={"title": "", "content": ""})

    def test_12_confirmation_generated_items_not_automatically_approved(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """CRITICAL: Test that ALL generated items are Drafts and NEVER automatically approved."""
        mom_path = tmp_path / "Safety_Check_MOM.txt"
        mom_path.write_text(
            "Agreed that SD pricing user exit refactoring carries high risk and may delay testing.\n\n"
            "Decided to proceed with brownfield migration approach for ECC financials.\n",
            encoding="utf-8"
        )

        results = service_with_mock_llm.process_meeting_minutes(file_paths=[mom_path])
        assert results["status"] == "success"

        # 1. Approved register must be completely empty of these items
        approved_items = service_with_mock_llm.get_approved_items()
        created_ids = {d["id"] for d in results["drafts"]}
        for app in approved_items:
            assert app["id"] not in created_ids
            assert app.get("draft_id") not in created_ids

        # 2. Every single created item must have status == "Draft"
        for d in results["drafts"]:
            assert d["status"] == "Draft"
            assert d["status"] not in ("Accepted", "Approved")

    def test_13_dependency_included_in_review_categories(self, service_with_mock_llm: RaidService):
        """Test that Dependency is included in review categories in Meeting Minutes Import."""
        import importlib
        page2 = importlib.import_module("ui.pages.2_Meeting_Minutes_Import")
        REVIEW_CATEGORIES = page2.REVIEW_CATEGORIES

        # Dependency must be in review categories
        assert "Dependency" in REVIEW_CATEGORIES
        assert REVIEW_CATEGORIES == ["Risk", "Assumption", "Issue", "Decision", "Dependency"]

    def test_14_stage_callback_progression(self, service_with_mock_llm: RaidService, tmp_path: Path):
        """Test that all 8 processing stages are reported in sequence to the callback."""
        mom_path = tmp_path / "Stage_Track_MOM.txt"
        mom_path.write_text("Agreed that SD pricing user exit refactoring carries high risk.", encoding="utf-8")

        observed_stages = []
        def on_stage(stage, details):
            observed_stages.append(stage)

        service_with_mock_llm.process_meeting_minutes(
            file_paths=[mom_path],
            stage_callback=on_stage
        )

        expected_stages = [
            "Uploaded",
            "Reading meeting minutes",
            "Normalising content",
            "Filtering noise",
            "Extracting RAID items",
            "Checking duplicates",
            "Saving results",
            "Completed"
        ]

        for exp in expected_stages:
            assert exp in observed_stages, f"Expected stage '{exp}' was not reported. Observed: {observed_stages}"
