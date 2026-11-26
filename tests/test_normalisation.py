"""
Unit tests for ProjectSignal Pydantic model and SignalNormaliser.
Verifies validation rules, source type Enums, ISO 8601 timestamps,
serialization, and error handling.
"""

import json
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.normalisation.schemas import ProjectSignal, SourceType
from src.normalisation.normaliser import SignalNormaliser


class TestProjectSignalCreation:
    def test_valid_signal_with_all_fields(self):
        """Tests complete valid signal instantiation."""
        sig = ProjectSignal(
            id="sig-test-1234",
            source_type=SourceType.EMAIL,
            source_id="MSG-001",
            author="lead@example.com",
            timestamp="2026-11-20T10:00:00+00:00",
            subject="Test Subject",
            content="Detailed signal content about custom code.",
            workstream="ABAP / Custom Code",
            phase="Realize",
            raw_meta={"client_ip": "10.0.0.1"},
            is_noise=0,
            noise_reason=""
        )
        assert sig.id == "sig-test-1234"
        assert sig.source_type == "email"
        assert sig.source_id == "MSG-001"
        assert sig.author == "lead@example.com"
        assert sig.content == "Detailed signal content about custom code."
        assert sig.workstream == "ABAP / Custom Code"
        assert sig.phase == "Realize"
        assert sig.raw_meta == {"client_ip": "10.0.0.1"}
        assert sig.is_noise == 0

    def test_valid_signal_with_default_fields(self):
        """Tests that optional fields default properly and id/source_id generate automatically."""
        sig = ProjectSignal(
            source_type="chat",
            content="Database restore issue on sandbox"
        )
        assert sig.id.startswith("sig-")
        assert sig.source_id == sig.id
        assert sig.source_type == "chat"
        assert sig.author == "Unknown"
        assert sig.workstream == "General"
        assert sig.phase == "Realize"
        assert sig.raw_meta == {}
        assert sig.is_noise == 0
        assert isinstance(sig.timestamp, str)

    def test_source_type_enum_coercion(self):
        """Tests that string representations of sources (case-insensitive) are coerced correctly."""
        sources = ["email", "EMAIL", "Chat", "MEETING_NOTES", "project_plan", "Defect"]
        expected = ["email", "email", "chat", "meeting_notes", "project_plan", "defect"]
        for src_input, exp in zip(sources, expected):
            sig = ProjectSignal(source_type=src_input, content="Valid content")
            assert sig.source_type == exp

    def test_missing_required_content(self):
        """Tests that omitting required content raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectSignal(source_type="email")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content",) for e in errors)

    def test_empty_or_whitespace_content(self):
        """Tests that whitespace-only content is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectSignal(source_type="email", content="   \n\t  ")
        assert "content cannot be empty" in str(exc_info.value)

    def test_invalid_source_type(self):
        """Tests that unsupported source types are rejected with clear error message."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectSignal(source_type="whatsapp_unsupported", content="Some content")
        assert "Invalid source_type" in str(exc_info.value)

    def test_timestamp_validation_and_iso_normalization(self):
        """Tests timestamp formats are normalized to valid ISO 8601 strings."""
        # 1. Standard datetime object
        dt = datetime(2026, 11, 20, 15, 30, 0, tzinfo=timezone.utc)
        sig1 = ProjectSignal(source_type="email", content="Test", timestamp=dt)
        assert sig1.timestamp == "2026-11-20T15:30:00+00:00"

        # 2. String format YYYY-MM-DD HH:MM:SS
        sig2 = ProjectSignal(source_type="chat", content="Test", timestamp="2026-11-20 18:45:00")
        assert "2026-11-20T18:45:00" in sig2.timestamp

        # 3. String format YYYY-MM-DD
        sig3 = ProjectSignal(source_type="meeting_notes", content="Test", timestamp="2026-11-20")
        assert "2026-11-20" in sig3.timestamp

        # 4. None / Empty preserves missing timestamp without silently inventing source dates
        sig4 = ProjectSignal(source_type="defect", content="Test", timestamp=None)
        assert sig4.timestamp == ""
        assert sig4.has_source_timestamp is False

    def test_invalid_timestamp_rejected(self):
        """Tests that unparseable date formats raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectSignal(source_type="email", content="Test", timestamp="invalid-date-string-xyz")
        assert "Invalid timestamp format" in str(exc_info.value)

    def test_is_noise_validation(self):
        """Tests boolean and integer validation for noise flag."""
        sig1 = ProjectSignal(source_type="email", content="Valid", is_noise=True)
        assert sig1.is_noise == 1
        sig2 = ProjectSignal(source_type="email", content="Valid", is_noise=False)
        assert sig2.is_noise == 0

        with pytest.raises(ValidationError):
            ProjectSignal(source_type="email", content="Valid", is_noise=5)

    def test_optional_metadata_handling(self):
        """Tests that arbitrary raw metadata is preserved."""
        meta = {"message_id": "MSG-999", "tags": ["urgent", "sap_mm"], "retry_count": 2}
        sig = ProjectSignal(source_type="defect", content="BAPI dump", raw_meta=meta)
        assert sig.raw_meta["tags"] == ["urgent", "sap_mm"]
        assert sig.raw_meta["retry_count"] == 2

    def test_model_dump_and_serialization(self):
        """Tests model_dump() and model_dump_json() outputs."""
        sig = ProjectSignal(
            source_type="email",
            content="Testing model serialization",
            subject="Test Serialization"
        )
        dump_dict = sig.model_dump()
        assert isinstance(dump_dict, dict)
        assert dump_dict["source_type"] == "email"
        assert dump_dict["content"] == "Testing model serialization"

        # Backwards compatible to_dict()
        assert sig.to_dict() == dump_dict

        # JSON serialization
        json_str = sig.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == sig.id
        assert parsed["source_type"] == "email"


class TestSignalNormaliser:
    def test_normalise_single_record(self):
        """Tests normalisation from raw dictionary input."""
        raw = {
            "source_type": "email",
            "source_id": "MSG-100",
            "author": "tester@test.com",
            "timestamp": "2026-11-20 12:00:00",
            "subject": "Status Update",
            "content": "Transport TRK90012 failed in QA",
            "workstream": "FI",
            "phase": "Realize"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig is not None
        assert sig.source_id == "MSG-100"
        assert sig.source_type == "email"
        assert sig.workstream == "FI"
        assert sig.content == "Transport TRK90012 failed in QA"

    def test_normalise_batch(self):
        """Tests batch normalisation across multiple source types."""
        records = [
            {"source_type": "email", "content": "Email content 1"},
            {"source_type": "chat", "content": "Chat content 2"},
            {"source_type": "defect", "content": "Defect content 3"}
        ]
        signals = SignalNormaliser.normalise_batch(records)
        assert len(signals) == 3
        assert [s.source_type for s in signals] == ["email", "chat", "defect"]

    def test_normalise_batch_skip_on_error(self):
        """Tests batch normalisation with skip_on_error option."""
        records = [
            {"source_type": "email", "content": "Valid content 1"},
            {"source_type": "invalid_source", "content": "Should fail"},
            {"source_type": "chat", "content": "Valid content 2"}
        ]
        # skip_on_error=True drops the invalid one
        signals = SignalNormaliser.normalise_batch(records, skip_on_error=True)
        assert len(signals) == 2
        assert signals[0].source_type == "email"
        assert signals[1].source_type == "chat"
