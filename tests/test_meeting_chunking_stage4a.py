"""
Stage 4A Regression Tests: Meeting Minutes TXT Chunking.
Tests for:
1. Numbered list items (1., 2), (1)) chunking.
2. Bullet list items (-, *, •, –, +) chunking.
3. Preserving ordinary sentences across multiple lines (no sentence splitting).
4. Preserving double-newline separation.
5. Multi-line list items (continuation lines stay with parent item).
6. Preserving document metadata (meeting date, attendees) across all chunks.
"""

from pathlib import Path
import pytest
from src.ingestion.meeting_ingestor import MeetingIngestor


class TestMeetingTxtChunkingStage4A:
    """Regression test suite for improved TXT meeting notes chunking."""

    def test_numbered_list_chunking(self, tmp_path: Path):
        """Verify that numbered list items are split into discrete discussion signals."""
        txt_file = tmp_path / "numbered_meeting.txt"
        txt_file.write_text(
            "1. Outbound delivery interface is delayed because of an unresolved vendor dependency.\n"
            "2. SAP integration testing is blocked by an external API authentication failure.\n"
            "3. Steering Committee approves the S/4HANA cutover date for December 15, 2026.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) == 3
        assert "Outbound delivery interface is delayed" in signals[0].content
        assert "SAP integration testing is blocked" in signals[1].content
        assert "Steering Committee approves" in signals[2].content
        assert signals[0].source_id.endswith("chunk-1")
        assert signals[1].source_id.endswith("chunk-2")
        assert signals[2].source_id.endswith("chunk-3")

    def test_numbered_list_alternate_delimiters(self, tmp_path: Path):
        """Verify that alternate numbering formats like '1)' and '(1)' are recognized."""
        txt_file = tmp_path / "alternate_numbers.txt"
        txt_file.write_text(
            "1) Custom code remediation velocity is below the required target.\n"
            "2) Basis team completed the system kernel upgrade on sandbox.\n"
            "(3) QA sign-off pending business approval for finance workflows.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) == 3
        assert "Custom code remediation velocity" in signals[0].content
        assert "Basis team completed" in signals[1].content
        assert "QA sign-off pending" in signals[2].content

    def test_bullet_list_chunking(self, tmp_path: Path):
        """Verify that various bullet styles (-, *, •, –, +) are recognized as distinct chunks."""
        txt_file = tmp_path / "bullet_meeting.txt"
        txt_file.write_text(
            "- Outbound delivery interface delayed due to missing partner profiles.\n"
            "* Client firewall changes blocked RFC connection to SAP Gateway.\n"
            "• Cutover schedule confirmed for December 15 by PMO lead.\n"
            "+ Third-party logistics middleware upgraded to version 4.2.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) == 4
        assert "Outbound delivery interface" in signals[0].content
        assert "Client firewall changes" in signals[1].content
        assert "Cutover schedule confirmed" in signals[2].content
        assert "Third-party logistics middleware" in signals[3].content

    def test_ordinary_sentences_not_split(self, tmp_path: Path):
        """Verify that normal multiline paragraphs without list markers are NOT split."""
        txt_file = tmp_path / "normal_paragraph.txt"
        txt_file.write_text(
            "This is the first sentence of the meeting summary. "
            "The program manager presented the overall status to stakeholders.\n"
            "Workstream leads confirmed that sprint targets were met on schedule.\n"
            "No critical blockers were reported during the morning standup.\n\n"
            "This is the second distinct paragraph discussing the infrastructure refresh.\n"
            "The hardware provisioning is complete and verified by the cloud hosting team.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) == 2
        # First chunk contains all 3 lines of paragraph 1 intact
        assert "first sentence of the meeting summary" in signals[0].content
        assert "Workstream leads confirmed" in signals[0].content
        assert "No critical blockers were reported" in signals[0].content
        # Second chunk contains both lines of paragraph 2 intact
        assert "second distinct paragraph" in signals[1].content
        assert "hardware provisioning is complete" in signals[1].content

    def test_multiline_list_items_preserve_continuation(self, tmp_path: Path):
        """Verify that continuation sentences in a list item stay attached to their parent item."""
        txt_file = tmp_path / "multiline_items.txt"
        txt_file.write_text(
            "1. Outbound delivery interface is delayed because of an unresolved vendor dependency.\n"
            "   The vendor engineering team is actively investigating the API credentials.\n"
            "   Estimated turnaround is two business days according to the account manager.\n"
            "2. Steering Committee approves the S/4HANA cutover date for December 15, 2026.\n"
            "   All business stakeholders have provided signed concurrence.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        assert len(signals) == 2
        assert "Outbound delivery interface is delayed" in signals[0].content
        assert "actively investigating the API credentials" in signals[0].content
        assert "Estimated turnaround is two business days" in signals[0].content
        assert "Steering Committee approves" in signals[1].content
        assert "signed concurrence" in signals[1].content

    def test_metadata_preserved_with_chunking(self, tmp_path: Path):
        """Verify that meeting date and attendees metadata are preserved across all chunks."""
        txt_file = tmp_path / "mom_with_metadata.txt"
        txt_file.write_text(
            "Meeting Date: 2026-11-20\n"
            "Attendees: Alice Smith, Bob Jones, Charlie Brown\n\n"
            "1. Vendor API authentication failed repeatedly during integration testing.\n"
            "2. Cutover window approved for December 15 by Steering Committee.\n",
            encoding="utf-8"
        )

        signals = MeetingIngestor().ingest(txt_file)
        # 1 header chunk + 2 numbered item chunks = 3 chunks
        assert len(signals) >= 2
        item_signals = [s for s in signals if "chunk-2" in s.source_id or "chunk-3" in s.source_id]
        assert len(item_signals) == 2
        for sig in item_signals:
            assert sig.raw_meta.get("meeting_date") == "2026-11-20"
            assert "Alice Smith" in sig.raw_meta.get("attendees", "")
            assert sig.raw_meta.get("original_format") == "txt"
            assert sig.source_type == "meeting_notes"

    def test_empty_and_whitespace_txt(self, tmp_path: Path):
        """Verify that empty or whitespace-only TXT files return empty list without error."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   \n\n   \n", encoding="utf-8")
        signals = MeetingIngestor().ingest(empty_file)
        assert signals == []
