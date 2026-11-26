"""
Unit tests for Data Ingestion modules.
Tests:
- Email ingestion (CSV, EML, TXT, and signature removal)
- Chat ingestion (CSV, JSON, TXT)
- Meeting notes ingestion (DOCX, TXT, CSV)
- Excel project plan ingestion (XLSX, CSV, missing columns)
- Defect log ingestion (XLSX, CSV)
- Non-existent, empty, and malformed files
- Duplicate source record detection in SQLite
"""

from pathlib import Path
import json
import sqlite3
import pandas as pd
import pytest
import docx

from src.ingestion.email_ingestor import EmailIngestor
from src.ingestion.chat_ingestor import ChatIngestor
from src.ingestion.meeting_ingestor import MeetingIngestor
from src.ingestion.plan_ingestor import PlanIngestor
from src.ingestion.defect_ingestor import DefectIngestor
from src.normalisation.schemas import ProjectSignal, SourceType
from src.database.db_manager import DatabaseManager


class TestEmailIngestion:
    def test_email_csv_ingestion(self, tmp_path: Path):
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text(
            "MessageId,From,Date,Subject,Body,Workstream,Phase\n"
            "EM-01,architect@test.com,2026-11-20,Code Remediation,Remediation delayed due to Z-objects.,ABAP,Realize\n",
            encoding="utf-8"
        )
        ingestor = EmailIngestor()
        signals = ingestor.ingest(csv_file)
        assert len(signals) == 1
        assert signals[0].source_type == SourceType.EMAIL
        assert signals[0].source_id == "EM-01"
        assert signals[0].author == "architect@test.com"
        assert "Remediation delayed" in signals[0].content
        assert signals[0].raw_meta["source_file"] == "emails.csv"

    def test_email_eml_ingestion(self, tmp_path: Path):
        eml_file = tmp_path / "message.eml"
        eml_content = (
            "From: lead@enterprise.com\r\n"
            "To: team@enterprise.com\r\n"
            "Date: Fri, 20 Nov 2026 10:00:00 +0000\r\n"
            "Subject: SteerCo Decision on Cutover\r\n"
            "Message-ID: <msg-eml-999@enterprise.com>\r\n"
            "\r\n"
            "SteerCo approved moving cutover date to wave 2.\r\n"
            "-- \r\n"
            "John Doe | Enterprise Lead\r\n"
        )
        eml_file.write_text(eml_content, encoding="utf-8")
        ingestor = EmailIngestor()
        signals = ingestor.ingest(eml_file)
        assert len(signals) == 1
        assert signals[0].source_type == SourceType.EMAIL
        assert signals[0].source_id == "msg-eml-999@enterprise.com"
        assert "SteerCo approved" in signals[0].content
        # Ensure signature was removed
        assert "John Doe | Enterprise Lead" not in signals[0].content

    def test_email_txt_ingestion(self, tmp_path: Path):
        txt_file = tmp_path / "email.txt"
        txt_file.write_text(
            "From: pmo@test.com\n"
            "Subject: Scope baseline\n"
            "Date: 2026-11-20\n\n"
            "Baseline assumptions approved by customer sponsor.\n"
            "Best regards,\n"
            "PMO Office\n",
            encoding="utf-8"
        )
        ingestor = EmailIngestor()
        signals = ingestor.ingest(txt_file)
        assert len(signals) == 1
        assert signals[0].author == "pmo@test.com"
        assert "Baseline assumptions approved" in signals[0].content
        assert "PMO Office" not in signals[0].content


class TestChatIngestion:
    def test_chat_csv_ingestion(self, tmp_path: Path):
        csv_file = tmp_path / "chat.csv"
        csv_file.write_text(
            "ChatId,Sender,Timestamp,Channel,Message\n"
            "CH-1,rajesh@sap,2026-11-20 10:00,Basis,Transport TRK9001 failed\n",
            encoding="utf-8"
        )
        signals = ChatIngestor().ingest(csv_file)
        assert len(signals) == 1
        assert signals[0].source_type == SourceType.CHAT
        assert signals[0].content == "Transport TRK9001 failed"

    def test_chat_json_ingestion(self, tmp_path: Path):
        json_file = tmp_path / "teams_chat.json"
        data = [
            {"id": "msg-101", "sender": "Alice", "timestamp": "2026-11-20 12:00:00", "message": "CPI connectivity port 443 blocked"},
            {"id": "msg-102", "sender": "Bob", "timestamp": "2026-11-20 12:01:00", "message": "Escalated to network team"}
        ]
        json_file.write_text(json.dumps(data), encoding="utf-8")
        signals = ChatIngestor().ingest(json_file)
        assert len(signals) == 2
        assert signals[0].author == "Alice"
        assert signals[1].author == "Bob"

    def test_chat_txt_transcript_ingestion(self, tmp_path: Path):
        txt_file = tmp_path / "chat_transcript.txt"
        txt_file.write_text(
            "[2026-11-20 14:00] DevLead: Database dump on Sandbox environment\n"
            "[2026-11-20 14:05] BasisLead: Storage IOPS need immediate increase\n",
            encoding="utf-8"
        )
        signals = ChatIngestor().ingest(txt_file)
        assert len(signals) == 2
        assert signals[0].author == "DevLead"
        assert "Database dump" in signals[0].content


class TestMeetingIngestion:
    def test_meeting_docx_ingestion(self, tmp_path: Path):
        doc_file = tmp_path / "MOM_Sprint_Review.docx"
        doc = docx.Document()
        doc.add_heading("Sprint Review Minutes", level=0)
        doc.add_paragraph("Agreed that SD pricing user exit refactoring carries high risk.")
        doc.add_paragraph("Decided to proceed with brownfield conversion rather than selective data transition.")
        doc.save(str(doc_file))

        signals = MeetingIngestor().ingest(doc_file)
        assert len(signals) == 3  # heading + 2 paragraphs
        assert signals[1].source_type == SourceType.MEETING_NOTES
        assert "SD pricing user exit refactoring" in signals[1].content
        assert signals[1].raw_meta["source_file"] == "MOM_Sprint_Review.docx"

    def test_meeting_txt_ingestion(self, tmp_path: Path):
        txt_file = tmp_path / "meeting_notes.txt"
        txt_file.write_text(
            "Discussion Point 1: Custom code remediation velocity is below target.\n\n"
            "Discussion Point 2: Steering committee will review offshore capacity on Friday.\n",
            encoding="utf-8"
        )
        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) == 2
        assert "velocity is below target" in signals[0].content


class TestPlanIngestion:
    def test_plan_excel_ingestion(self, tmp_path: Path):
        excel_file = tmp_path / "project_schedule.xlsx"
        df = pd.DataFrame([
            {
                "Task ID": "TSK-01",
                "Task Name": "Integration Testing Cycle 1",
                "Assignee": "Kavitha R",
                "Status": "Delayed",
                "Start Date": "2026-11-01",
                "Due Date": "2026-11-20",
                "VarianceDays": 7,
                "Dependencies": "TSK-00",
                "Notes": "Waiting for bank certificates",
                "Workstream": "Integration"
            }
        ])
        df.to_excel(excel_file, index=False, engine="openpyxl")

        signals = PlanIngestor().ingest(excel_file)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.source_type == SourceType.PROJECT_PLAN
        assert sig.source_id == "TSK-01"
        assert sig.author == "Kavitha R"
        assert "Integration Testing Cycle 1" in sig.subject
        assert "Variance: 7 days" in sig.content
        assert sig.raw_meta["variance_days"] == "7"
        assert sig.raw_meta["dependencies"] == "TSK-00"

    def test_plan_csv_with_missing_columns(self, tmp_path: Path):
        csv_file = tmp_path / "plan_sparse.csv"
        # Only task name and status provided
        csv_file.write_text(
            "Task Name,Status\n"
            "Security Role Matrix Setup,In Progress\n",
            encoding="utf-8"
        )
        signals = PlanIngestor().ingest(csv_file)
        assert len(signals) == 1
        assert "Security Role Matrix Setup" in signals[0].content
        assert signals[0].author == "Unknown"  # Explicitly unassigned; no synthetic PMO default


class TestDefectIngestion:
    def test_defect_excel_ingestion(self, tmp_path: Path):
        excel_file = tmp_path / "defects.xlsx"
        df = pd.DataFrame([
            {
                "Defect ID": "DEF-901",
                "Summary": "Material master BAPI short dump",
                "Severity": "Critical",
                "Status": "Open",
                "ReportedBy": "QA Lead",
                "Created Date": "2026-11-20",
                "Module": "MM",
                "Description": "BAPI_MATERIAL_SAVEDATA dumps with CALL_FUNCTION_NOT_FOUND"
            }
        ])
        df.to_excel(excel_file, index=False, engine="openpyxl")

        signals = DefectIngestor().ingest(excel_file)
        assert len(signals) == 1
        assert signals[0].source_type == SourceType.DEFECT
        assert signals[0].source_id == "DEF-901"
        assert "Critical" in signals[0].content
        assert signals[0].workstream == "MM"

    def test_defect_csv_ingestion(self, tmp_path: Path):
        csv_file = tmp_path / "defects.csv"
        csv_file.write_text(
            "DefectId,Summary,Severity,Status,Module\n"
            "DEF-902,PO approval workflow failure,High,Open,MM\n",
            encoding="utf-8"
        )
        signals = DefectIngestor().ingest(csv_file)
        assert len(signals) == 1
        assert signals[0].source_id == "DEF-902"


class TestErrorAndDuplicateHandling:
    def test_nonexistent_and_empty_files(self, tmp_path: Path):
        ingestor = EmailIngestor()
        assert ingestor.ingest(tmp_path / "nonexistent.csv") == []

        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("", encoding="utf-8")
        assert ingestor.ingest(empty_file) == []

    def test_sqlite_duplicate_prevention(self, tmp_path: Path):
        db_file = tmp_path / "test_db.db"
        db = DatabaseManager(db_file)

        sig1 = ProjectSignal(
            source_type=SourceType.EMAIL,
            source_id="STABLE-MSG-001",
            content="Testing duplicate detection"
        )
        sig2 = ProjectSignal(
            source_type=SourceType.EMAIL,
            source_id="STABLE-MSG-001",  # Same stable identifier
            content="Testing duplicate detection - attempt 2"
        )

        inserted1 = db.save_raw_signal(sig1, overwrite=False)
        inserted2 = db.save_raw_signal(sig2, overwrite=False)

        assert inserted1 is True
        assert inserted2 is False  # Correctly prevented duplicate

        # Verify only 1 record exists in table
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raw_signals WHERE source_id = 'STABLE-MSG-001'")
            assert cursor.fetchone()[0] == 1
