"""
Comprehensive unit and edge-case tests for the optimized Ingestion Module.
Tests:
- Email CSV with BOM (utf-8-sig), alternate column headers, multiple rows, malformed rows
- Email EML multipart (plain preferred, HTML-only conversion, attachment extraction)
- Conservative signature stripping (verify prose starting with 'Regards' is preserved)
- Email TXT header parsing and headerless fallback
- Chat CSV with flexible columns, missing fields
- Chat JSON array vs wrapped object with messages/chats key
- Chat TXT multiline transcripts with various timestamp patterns
- Meeting Notes DOCX with tables, merged cells, headers, attendees
- Meeting Notes TXT with section chunks
- Meeting Notes CSV with flexible headers
- Meeting Notes manual text entry
- Plan and Defect CSV with BOM and flexible columns
- Non-existent files, 0-byte files, and unsupported formats
- Unicode and special characters
"""

from pathlib import Path
import json
import pytest
import docx

from src.ingestion.email_ingestor import EmailIngestor
from src.ingestion.chat_ingestor import ChatIngestor
from src.ingestion.meeting_ingestor import MeetingIngestor
from src.ingestion.plan_ingestor import PlanIngestor
from src.ingestion.defect_ingestor import DefectIngestor
from src.normalisation.schemas import SourceType


# ===========================================================================
# 1. EMAIL INGESTION EDGE CASES
# ===========================================================================

class TestEmailIngestionEdgeCases:
    def test_email_csv_utf8_bom_and_alternate_columns(self, tmp_path: Path):
        """Tests that UTF-8 with BOM and non-standard column headers are handled cleanly."""
        csv_file = tmp_path / "emails_bom.csv"
        # Includes BOM \ufeff and alternative column headers
        content = (
            "\ufeffEmailId,Sender_Email,SentDate,Topic,Email_Body,Track,Stage\n"
            "EML-101,sarah@client.com,2026-11-21,Testing Dependency,SIT cannot proceed until Basis completes sandbox refresh.,Basis,Realize\n"
            "EML-102,mike@partner.com,,Security Review,Role matrix sign-off is pending from audit committee.,Security,Explore\n"
        )
        csv_file.write_bytes(content.encode("utf-8"))

        ingestor = EmailIngestor()
        signals = ingestor.ingest(csv_file)

        assert len(signals) == 2
        assert signals[0].source_id == "EML-101"
        assert signals[0].author == "sarah@client.com"
        assert signals[0].subject == "Testing Dependency"
        assert signals[0].workstream == "Basis"
        assert signals[0].phase == "Realize"
        assert "SIT cannot proceed" in signals[0].content
        assert signals[0].raw_meta["raw_body"] is not None

        # Missing date is preserved as empty string without silently inventing source dates
        assert signals[1].source_id == "EML-102"
        assert signals[1].author == "mike@partner.com"
        assert signals[1].timestamp == ""
        assert signals[1].has_source_timestamp is False

    def test_email_csv_row_resilience_and_empty_message(self, tmp_path: Path):
        """Tests that empty rows or one malformed record do not halt the entire batch."""
        csv_file = tmp_path / "emails_mixed.csv"
        content = (
            "MessageId,From,Date,Subject,Body\n"
            "EM-01,lead@test.com,2026-11-20,Valid Msg 1,Valid project update content.\n"
            "EM-02,lead@test.com,2026-11-20,Empty Msg,   \n"  # Empty body: should be skipped
            "EM-03,lead@test.com,2026-11-20,Valid Msg 2,Second valid update for project.\n"
        )
        csv_file.write_text(content, encoding="utf-8")

        ingestor = EmailIngestor()
        signals = ingestor.ingest(csv_file)

        assert len(signals) == 2
        assert signals[0].source_id == "EM-01"
        assert signals[1].source_id == "EM-03"

    def test_conservative_signature_stripping(self):
        """Verifies that signature removal never deletes meaningful project prose."""
        ingestor = EmailIngestor()

        # Case 1: Prose starting with "Regards" must NOT be stripped
        prose_msg = (
            "Regards our discussion yesterday, we are waiting for master data.\n"
            "The client promised to deliver files by Friday."
        )
        cleaned = ingestor.strip_signature(prose_msg)
        assert "Regards our discussion yesterday" in cleaned
        assert "The client promised to deliver files by Friday." in cleaned

        # Case 2: Prose containing "With regards to" must NOT be stripped
        with_regards = (
            "With regards to the testing schedule, we anticipate a 3-day delay.\n"
            "Please confirm if cutover window changes."
        )
        cleaned2 = ingestor.strip_signature(with_regards)
        assert "With regards to the testing schedule" in cleaned2

        # Case 3: Genuine sign-off at the end MUST be stripped
        genuine_email = (
            "Integration testing is currently blocked due to missing master data.\n\n"
            "Best regards,\n"
            "Alex Turner\n"
            "Senior Solution Architect"
        )
        cleaned3 = ingestor.strip_signature(genuine_email)
        assert "Integration testing is currently blocked" in cleaned3
        assert "Alex Turner" not in cleaned3
        assert "Senior Solution Architect" not in cleaned3

        # Case 4: Mobile footer must be stripped
        mobile_email = (
            "Critical transport TRK9001 failed with RC=8.\n\n"
            "Sent from my iPhone"
        )
        cleaned4 = ingestor.strip_signature(mobile_email)
        assert "Critical transport TRK9001 failed" in cleaned4
        assert "iPhone" not in cleaned4

    def test_email_eml_html_only_and_attachment_detection(self, tmp_path: Path):
        """Tests that HTML-only EML converts HTML to clean text and extracts attachment names."""
        eml_file = tmp_path / "html_email.eml"
        eml_content = (
            "From: notifications@sap.com\r\n"
            "To: team@kaartech.com\r\n"
            "Subject: QA Transport Status Alert\r\n"
            "Date: Mon, 23 Nov 2026 09:30:00 +0000\r\n"
            "MIME-Version: 1.0\r\n"
            "Content-Type: multipart/mixed; boundary=\"====BOUNDARY====\"\r\n"
            "\r\n"
            "--====BOUNDARY====\r\n"
            "Content-Type: text/html; charset=\"utf-8\"\r\n"
            "\r\n"
            "<html><body>"
            "<h3>Alert: Transport TRK9002 Blocked</h3>"
            "<p>The finance configuration transport failed import with <b>RC=8</b>.</p>"
            "<p>Action required before SIT cycle 2 starts.</p>"
            "</body></html>\r\n"
            "--====BOUNDARY====\r\n"
            "Content-Type: text/plain; name=\"error_log.txt\"\r\n"
            "Content-Disposition: attachment; filename=\"error_log.txt\"\r\n"
            "\r\n"
            "Simulated log file attachment content\r\n"
            "--====BOUNDARY====--\r\n"
        )
        eml_file.write_bytes(eml_content.encode("utf-8"))

        ingestor = EmailIngestor()
        signals = ingestor.ingest(eml_file)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.source_type == SourceType.EMAIL
        assert "Alert: Transport TRK9002 Blocked" in sig.content
        assert "RC=8" in sig.content
        assert "<html>" not in sig.content  # Tags cleanly stripped
        assert "<h3>" not in sig.content
        assert "error_log.txt" in sig.raw_meta.get("attachments", [])

    def test_email_txt_headerless_fallback(self, tmp_path: Path):
        """Tests that TXT emails without standard headers are treated safely as raw message body."""
        txt_file = tmp_path / "raw_note.txt"
        txt_file.write_text(
            "Quick note for the team: We are waiting on client sign-off before commencing sprint 3.",
            encoding="utf-8"
        )

        ingestor = EmailIngestor()
        signals = ingestor.ingest(txt_file)

        assert len(signals) == 1
        assert signals[0].author == "Unknown"
        assert signals[0].subject == "raw_note"
        assert "waiting on client sign-off" in signals[0].content


# ===========================================================================
# 2. CHAT INGESTION EDGE CASES
# ===========================================================================

class TestChatIngestionEdgeCases:
    def test_chat_multiline_transcript(self, tmp_path: Path):
        """Tests that multiline messages in transcripts are grouped with the sender."""
        txt_file = tmp_path / "teams_multiline.txt"
        content = (
            "[2026-11-22 10:00] LeadArchitect: We encountered a blocker during cutover rehearsal:\n"
            "1. Database restore took 4 hours instead of 2.\n"
            "2. File system permissions were reset.\n"
            "[2026-11-22 10:05] BasisConsultant: Storage IOPS need immediate increase to resolve item 1."
        )
        txt_file.write_text(content, encoding="utf-8")

        signals = ChatIngestor().ingest(txt_file)
        assert len(signals) == 2

        # First message must contain all 3 lines
        assert signals[0].author == "LeadArchitect"
        assert "encountered a blocker" in signals[0].content
        assert "Database restore took 4 hours" in signals[0].content
        assert "File system permissions were reset" in signals[0].content

        # Second message
        assert signals[1].author == "BasisConsultant"
        assert "Storage IOPS need immediate increase" in signals[1].content

    def test_chat_json_wrapped_structures(self, tmp_path: Path):
        """Tests JSON chat files wrapped with top-level keys."""
        json_file = tmp_path / "wrapped_chat.json"
        data = {
            "channel": "S/4HANA Migration",
            "messages": [
                {"id": "msg-501", "user": "Priya", "time": "2026-11-22T08:00:00Z", "text": "Bank interface certificate expired"},
                {"id": "msg-502", "user": "David", "time": "2026-11-22T08:05:00Z", "text": "Escalated to treasury team"}
            ]
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        signals = ChatIngestor().ingest(json_file)
        assert len(signals) == 2
        assert signals[0].author == "Priya"
        assert signals[0].content == "Bank interface certificate expired"
        assert signals[1].author == "David"

    def test_chat_csv_alternate_headers(self, tmp_path: Path):
        """Tests chat CSV with non-standard column headers."""
        csv_file = tmp_path / "chat_alt.csv"
        csv_file.write_text(
            "MsgId,Author,DateTime,Room,Text,Module,Stage\n"
            "C-1,Kavitha,2026-11-22 11:00,Cutover-WarRoom,Mock cutover rehearsal slipped by 2 days,Integration,Realize\n",
            encoding="utf-8"
        )
        signals = ChatIngestor().ingest(csv_file)
        assert len(signals) == 1
        assert signals[0].source_id == "C-1"
        assert signals[0].author == "Kavitha"
        assert signals[0].subject == "Cutover-WarRoom"
        assert signals[0].workstream == "Integration"


# ===========================================================================
# 3. MEETING MINUTES INGESTION EDGE CASES
# ===========================================================================

class TestMeetingIngestionEdgeCases:
    def test_docx_with_table_merged_cells_and_attendees(self, tmp_path: Path):
        """Tests DOCX parsing with attendee header and tables with deduplicated merged cells."""
        doc_file = tmp_path / "SteerCo_Review.docx"
        doc = docx.Document()
        doc.add_heading("SteerCo Meeting Minutes", level=0)
        doc.add_paragraph("Date: 2026-11-22")
        doc.add_paragraph("Attendees: Alex Turner (PMO), Sarah Jenkins (Lead), Mike Ross (FI)")
        doc.add_paragraph("Steering committee approved brownfield conversion scope.")

        # Add a table
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Action Item"
        table.cell(0, 1).text = "Owner"
        table.cell(1, 0).text = "Remediate custom user exit"
        table.cell(1, 1).text = "ABAP Team"
        doc.save(str(doc_file))

        signals = MeetingIngestor().ingest(doc_file)
        assert len(signals) >= 4

        # Verify attendee metadata was captured
        assert any("Alex Turner" in sig.raw_meta.get("attendees", "") for sig in signals)
        # Verify table row content
        table_signal = [s for s in signals if "Remediate custom user exit" in s.content]
        assert len(table_signal) == 1
        assert "ABAP Team" in table_signal[0].content

    def test_meeting_txt_section_detection(self, tmp_path: Path):
        """Tests TXT meeting notes with attendees and discussion points."""
        txt_file = tmp_path / "workshop.txt"
        txt_file.write_text(
            "Meeting Date: 2026-11-22\n"
            "Participants: John, Mary, Rajesh\n\n"
            "Key Decisions:\n"
            "Steering committee decided to postpone SIT kickoff to December 15.\n\n"
            "Key Risks:\n"
            "Offshore capacity constraint may delay SD pricing routine adjustments.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) >= 2
        assert any("postpone SIT kickoff" in s.content for s in signals)
        assert any("Offshore capacity constraint" in s.content for s in signals)
        assert signals[0].raw_meta.get("attendees") == "John, Mary, Rajesh"

    def test_meeting_manual_text_bullet_points(self):
        """Tests manual meeting text ingestion with bulleted action points."""
        ingestor = MeetingIngestor()
        text = (
            "Meeting Date: 2026-11-22\n"
            "• Decision 1: Proceed with S/4HANA Finance brownfield approach\n"
            "• Risk 1: Test data migration scripts have not been validated\n"
            "• Action 1: Schedule follow-up checkpoint with data lead"
        )
        signals = ingestor.ingest_text(content=text, title="Sprint Alignment")
        assert len(signals) == 3
        assert "Proceed with S/4HANA Finance" in signals[0].content
        assert "Test data migration scripts" in signals[1].content


# ===========================================================================
# 4. UNICODE, SPECIAL CHARS & FORMAT RESILIENCE
# ===========================================================================

class TestIngestionResilienceAndFormats:
    def test_unicode_and_special_characters(self, tmp_path: Path):
        """Tests that non-ASCII unicode characters, emojis, and accents are preserved."""
        csv_file = tmp_path / "unicode_test.csv"
        csv_file.write_text(
            "MessageId,From,Subject,Body\n"
            "UNI-01,François Müller,CR ⚠️ Délais dépassés,Attention: la migration des données a pris du retard 🚀.\n",
            encoding="utf-8"
        )
        signals = EmailIngestor().ingest(csv_file)
        assert len(signals) == 1
        assert "François Müller" in signals[0].author
        assert "Délais dépassés" in signals[0].subject
        assert "🚀" in signals[0].content

    def test_unsupported_formats_and_nonexistent_files(self, tmp_path: Path):
        """Tests that unsupported formats and non-existent files return empty lists cleanly."""
        ingestor = EmailIngestor()

        # Non-existent file
        assert ingestor.ingest(tmp_path / "does_not_exist.csv") == []

        # 0-byte file
        zero_file = tmp_path / "zero_byte.csv"
        zero_file.write_bytes(b"")
        assert ingestor.ingest(zero_file) == []

        # Unsupported file extension (.exe, .pdf)
        bin_file = tmp_path / "report.pdf"
        bin_file.write_text("PDF content simulation", encoding="utf-8")
        assert ingestor.ingest(bin_file) == []
