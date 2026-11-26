"""
Comprehensive Unit and Integration Tests for the Optimised Normalisation Layer.
Covers:
- Dictionary input normalisation
- Existing ProjectSignal input standardisation & validation
- Missing and malformed timestamps
- Valid timestamps with various formats & timezones (UTC, offsets, RFC 2822, naive)
- Missing or non-dict metadata preservation
- Unknown source-specific fields preserved in raw_meta['extra_fields']
- Empty and whitespace content rejection
- Invalid source types rejection
- Batch processing with skip_on_error=True and False, error reporting, and reports
- Channel compatibility across Email, Chat, MOM, Plan, and Defect sources
- End-to-end pipeline execution with mixed inputs
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.normalisation.schemas import ProjectSignal, SourceType
from src.normalisation.normaliser import SignalNormaliser
from src.pipeline import RAIDPipeline
from src.database.db_manager import DatabaseManager


class TestNormalisationDictionaryInput:
    """Tests normalisation from raw dictionary payloads."""

    def test_complete_valid_dictionary(self):
        raw = {
            "source_type": "email",
            "source_id": "MSG-999",
            "author": "finance_lead@corp.com",
            "timestamp": "2026-11-20T10:00:00+00:00",
            "subject": "ECC Migration Risk",
            "content": "Custom z-tables in FI module will not migrate without redesign.",
            "workstream": "FI",
            "phase": "Realize",
            "raw_meta": {"priority": "High", "thread_id": "TH-123"},
            "user_id": "usr-finance"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig is not None
        assert sig.source_type == "email"
        assert sig.source_id == "MSG-999"
        assert sig.author == "finance_lead@corp.com"
        assert sig.timestamp == "2026-11-20T10:00:00+00:00"
        assert sig.workstream == "FI"
        assert sig.phase == "Realize"
        assert sig.user_id == "usr-finance"
        assert sig.raw_meta["priority"] == "High"
        assert sig.raw_meta["has_source_timestamp"] is True
        assert sig.raw_meta["_normalised"] is True

    def test_minimal_dictionary_defaults(self):
        raw = {
            "source_type": "chat",
            "content": "Database container stopped unexpectedly in test environment."
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig is not None
        assert sig.source_type == "chat"
        assert sig.id.startswith("sig-")
        assert sig.source_id == sig.id
        assert sig.author == "Unknown"
        assert sig.workstream == "General"
        assert sig.phase == "Realize"
        assert sig.timestamp == ""
        assert sig.raw_meta["has_source_timestamp"] is False
        assert sig.raw_meta["_normalised"] is True


class TestNormalisationProjectSignalInput:
    """Tests consistent standardisation when input is already a ProjectSignal."""

    def test_existing_project_signal_normalisation(self):
        original = ProjectSignal(
            source_type=SourceType.MEETING_NOTES,
            content="Team confirmed cutover window is fixed to 48 hours.",
            timestamp="2026-11-22 14:00:00",
            raw_meta={"meeting_room": "Boardroom A"}
        )
        # Not yet marked normalised
        assert "_normalised" not in original.raw_meta

        normalised = SignalNormaliser.normalise(original)
        assert normalised is original
        assert normalised.raw_meta["_normalised"] is True
        assert normalised.raw_meta["has_source_timestamp"] is True
        assert "2026-11-22T14:00:00" in normalised.timestamp

    def test_existing_project_signal_idempotency(self):
        """Calling normalise twice on the same ProjectSignal returns immediately without redundant work."""
        sig = ProjectSignal(source_type="email", content="Valid content")
        first = SignalNormaliser.normalise(sig)
        assert first.raw_meta.get("_normalised") is True

        second = SignalNormaliser.normalise(first)
        assert second is first

    def test_existing_project_signal_empty_content_rejected(self):
        sig = ProjectSignal(source_type="defect", content="Valid content initially")
        object.__setattr__(sig, "content", "   ")  # Bypass validate_assignment to test normaliser check
        sig.raw_meta.pop("_normalised", None)

        with pytest.raises(ValueError) as exc_info:
            SignalNormaliser.normalise(sig)
        assert "content cannot be empty" in str(exc_info.value)


class TestTimestampHandlingAndConsistency:
    """Tests standardisation into ISO 8601 UTC and handling of missing dates."""

    def test_missing_timestamp_preserved_as_empty_string(self):
        raw = {"source_type": "chat", "content": "Update on sandbox"}
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == ""
        assert sig.raw_meta["has_source_timestamp"] is False

    def test_explicit_none_timestamp_preserved(self):
        raw = {"source_type": "chat", "content": "Update on sandbox", "timestamp": None}
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == ""
        assert sig.raw_meta["has_source_timestamp"] is False

    def test_empty_string_timestamp_preserved(self):
        raw = {"source_type": "chat", "content": "Update on sandbox", "timestamp": "   "}
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == ""
        assert sig.raw_meta["has_source_timestamp"] is False

    def test_timezone_aware_iso_standardised_to_utc(self):
        # 15:30 in +05:30 is 10:00 UTC
        raw = {
            "source_type": "email",
            "content": "Meeting invite",
            "timestamp": "2026-11-20T15:30:00+05:30"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == "2026-11-20T10:00:00+00:00"
        assert sig.raw_meta["has_source_timestamp"] is True
        assert sig.raw_meta["original_timestamp"] == "2026-11-20T15:30:00+05:30"

    def test_negative_timezone_offset_standardised_to_utc(self):
        # 05:00 in -05:00 is 10:00 UTC
        raw = {
            "source_type": "email",
            "content": "Early update",
            "timestamp": "2026-11-20T05:00:00-05:00"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == "2026-11-20T10:00:00+00:00"

    def test_rfc2822_email_date_standardised_to_utc(self):
        raw = {
            "source_type": "email",
            "content": "Email body",
            "timestamp": "Fri, 20 Nov 2026 15:30:00 +0530"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == "2026-11-20T10:00:00+00:00"

    def test_naive_timestamp_assumed_utc_and_documented(self):
        raw = {
            "source_type": "chat",
            "content": "Chat log",
            "timestamp": "2026-11-20 18:45:00"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == "2026-11-20T18:45:00+00:00"
        assert sig.raw_meta["has_source_timestamp"] is True

    def test_date_only_format_standardised(self):
        raw = {
            "source_type": "project_plan",
            "content": "Milestone M1 due",
            "timestamp": "2026-12-01"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.timestamp == "2026-12-01T00:00:00+00:00"

    def test_malformed_timestamp_rejected(self):
        raw = {
            "source_type": "email",
            "content": "Update",
            "timestamp": "not-a-real-date-string"
        }
        with pytest.raises(ValidationError) as exc_info:
            SignalNormaliser.normalise(raw)
        assert "Invalid timestamp format" in str(exc_info.value)


class TestMetadataAndUnknownFieldsPreservation:
    """Tests preservation of unknown fields and non-dict metadata."""

    def test_unknown_fields_captured_in_extra_fields(self):
        raw = {
            "source_type": "email",
            "content": "Custom code transport failed",
            "ip_address": "192.168.1.50",
            "x_header_priority": "urgent",
            "sap_tcode": "SE38"
        }
        sig = SignalNormaliser.normalise(raw)
        assert "extra_fields" in sig.raw_meta
        assert sig.raw_meta["extra_fields"]["ip_address"] == "192.168.1.50"
        assert sig.raw_meta["extra_fields"]["x_header_priority"] == "urgent"
        assert sig.raw_meta["extra_fields"]["sap_tcode"] == "SE38"

    def test_json_string_raw_meta_parsed_safely(self):
        json_meta = json.dumps({"session_id": "sess-99", "channel": "qa-war-room"})
        raw = {
            "source_type": "chat",
            "content": "Sandbox refreshed",
            "raw_meta": json_meta
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.raw_meta["session_id"] == "sess-99"
        assert sig.raw_meta["channel"] == "qa-war-room"

    def test_non_json_string_raw_meta_preserved(self):
        raw = {
            "source_type": "chat",
            "content": "Sandbox refreshed",
            "raw_meta": "Raw non-json string data from legacy logger"
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.raw_meta.get("raw_value") == "Raw non-json string data from legacy logger"

    def test_scalar_raw_meta_preserved(self):
        raw = {
            "source_type": "defect",
            "content": "Memory dump in BAPI",
            "raw_meta": 12345
        }
        sig = SignalNormaliser.normalise(raw)
        assert sig.raw_meta.get("raw_value") == 12345


class TestValidationAndErrorHandling:
    """Tests rejection of invalid records and batch processing error reporting."""

    def test_empty_content_rejected(self):
        with pytest.raises(ValueError) as exc:
            SignalNormaliser.normalise({"source_type": "email", "content": ""})
        assert "content cannot be empty" in str(exc.value)

    def test_whitespace_content_rejected(self):
        with pytest.raises(ValueError) as exc:
            SignalNormaliser.normalise({"source_type": "email", "content": "   \n\t  "})
        assert "content cannot be empty" in str(exc.value)

    def test_invalid_source_type_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SignalNormaliser.normalise({"source_type": "unsupported_channel", "content": "Msg"})
        assert "Invalid source_type" in str(exc.value)

    def test_batch_skip_on_error_false_raises(self):
        records = [
            {"source_type": "email", "content": "Valid 1"},
            {"source_type": "invalid_source", "content": "Will fail"},
            {"source_type": "chat", "content": "Valid 2"}
        ]
        with pytest.raises(ValidationError):
            SignalNormaliser.normalise_batch(records, skip_on_error=False)

    def test_batch_skip_on_error_true_omits_and_tracks_errors(self):
        records = [
            {"id": "rec-1", "source_type": "email", "content": "Valid 1"},
            {"id": "rec-2", "source_type": "invalid_source", "content": "Fails"},
            {"id": "rec-3", "source_type": "chat", "content": "   "},  # empty content
            {"id": "rec-4", "source_type": "defect", "content": "Valid 2"}
        ]
        signals = SignalNormaliser.normalise_batch(records, skip_on_error=True)
        assert len(signals) == 2
        assert signals[0].id == "rec-1"
        assert signals[1].id == "rec-4"

        # Check structured errors
        assert len(SignalNormaliser.last_batch_errors) == 2
        err1 = SignalNormaliser.last_batch_errors[0]
        assert err1["index"] == 1
        assert err1["record_id"] == "rec-2"

        err2 = SignalNormaliser.last_batch_errors[1]
        assert err2["index"] == 2
        assert err2["record_id"] == "rec-3"

    def test_batch_return_errors_flag(self):
        records = [
            {"source_type": "email", "content": "Valid 1"},
            {"source_type": "invalid", "content": "Fails"}
        ]
        signals, errors = SignalNormaliser.normalise_batch(records, skip_on_error=True, return_errors=True)
        assert len(signals) == 1
        assert len(errors) == 1
        assert errors[0]["index"] == 1

    def test_normalise_batch_with_report(self):
        records = [
            {"source_type": "email", "content": "Valid 1"},
            {"source_type": "invalid", "content": "Fails"}
        ]
        report = SignalNormaliser.normalise_batch_with_report(records, skip_on_error=True)
        assert report["total_input"] == 2
        assert report["success_count"] == 1
        assert report["failure_count"] == 1
        assert len(report["signals"]) == 1
        assert len(report["errors"]) == 1


class TestIngestionCompatibilityAndPipelineIntegration:
    """Verifies all source types and end-to-end pipeline execution with mixed inputs."""

    def test_all_supported_source_types(self):
        sources = [
            SourceType.EMAIL,
            SourceType.CHAT,
            SourceType.MEETING_NOTES,
            SourceType.PROJECT_PLAN,
            SourceType.DEFECT
        ]
        for src in sources:
            sig = SignalNormaliser.normalise({
                "source_type": src.value,
                "content": f"Content for source {src.value}"
            })
            assert sig.source_type == src.value
            assert sig.raw_meta["_normalised"] is True

    def test_pipeline_execution_with_mixed_inputs(self, tmp_path: Path):
        """Tests that RAIDPipeline processes both dicts and ProjectSignals cleanly."""
        db_file = tmp_path / "test_pipeline.db"
        db = DatabaseManager(db_file)
        pipeline = RAIDPipeline(db)

        # 1. ProjectSignal input (as produced by Ingestors)
        sig_obj = ProjectSignal(
            source_type="email",
            source_id="EML-TEST-1",
            author="lead@test.com",
            content="Dependency on Basis team for sandbox refresh is unresolved.",
            timestamp="2026-11-20T10:00:00+00:00"
        )

        # 2. Dict input (as might come from an external API or test)
        dict_obj = {
            "source_type": "chat",
            "source_id": "CHAT-TEST-1",
            "author": "dev@test.com",
            "content": "Table BSEG transport failed during test migration run.",
            "timestamp": "2026-11-20 12:00:00",
            "extra_custom_field": "custom_val"
        }

        # 3. Malformed dict that should be gracefully skipped with structured logging
        malformed = {
            "source_type": "invalid_type",
            "content": "This will fail normalisation"
        }

        mixed_signals = [sig_obj, dict_obj, malformed]
        results = pipeline.process_signals(mixed_signals, user_id="test_user")

        assert results["total_ingested"] == 3
        # 2 successfully normalised (1 was malformed and skipped)
        assert results["normalized"] >= 1
        # Check raw signals persisted in DB
        persisted = db.get_raw_signals(user_id="test_user")
        persisted_ids = {s["source_id"] for s in persisted}
        assert "EML-TEST-1" in persisted_ids
        assert "CHAT-TEST-1" in persisted_ids
