"""
Unit and Integration Tests for Phase 1: Gemini RAID Classification Critical Integrity Fixes.

Covers:
1. Confidence Threshold Tests (Main pipeline & Meeting Minutes, thresholds 0.69, 0.70, 0.74, 0.75, 0.90, dynamic config).
2. Dependency Category Tests (Extraction from Email, Chat, Meeting Minutes, review lifecycle: approve, reject, modify, export).
3. Failed Classification Tracking Tests (Timeout, 429 rate limit, auth failure, empty response, invalid JSON, validation error, retry behavior, no duplicate drafts).
4. Evidence Tests (Exact match, casing tolerance, whitespace tolerance, hallucinated quote fallback, missing quote fallback, after greeting, before disclaimer, unavailable evidence).
"""

import json
from pathlib import Path
import pytest
from typing import Dict, Any

from config import CONFIDENCE_THRESHOLD
from src.database.db_manager import DatabaseManager
from src.normalisation.schemas import ProjectSignal, SourceType
from src.classification.schemas import (
    RAIDCategory,
    RAIDExtractionResult,
    ClassificationStatus,
    ClassificationErrorCategory,
    EvidenceStatus,
    ClassificationOutcome
)
from src.classification.classifier import RAIDClassifier, find_best_fallback_sentence, sanitize_error_message
from src.classification.providers.mock_provider import MockLLMProvider
from src.pipeline import RAIDPipeline
from src.service import RaidService


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "test_classification_phase1.db"
    return DatabaseManager(db_file)


@pytest.fixture
def test_pipeline(temp_db: DatabaseManager) -> RAIDPipeline:
    pipeline = RAIDPipeline(temp_db)
    pipeline.classifier = RAIDClassifier(provider=MockLLMProvider(), db_manager=temp_db)
    return pipeline


@pytest.fixture
def test_service(temp_db: DatabaseManager, test_pipeline: RAIDPipeline) -> RaidService:
    return RaidService(db_manager=temp_db, pipeline=test_pipeline)


# =============================================================================
# 1. Confidence Threshold Tests
# =============================================================================
class TestConfidenceThresholdIntegration:
    @pytest.mark.parametrize("confidence,expected_admitted", [
        (0.69, False),
        (0.70, False),  # 0.70 is rejected under 0.75 threshold
        (0.74, False),
        (0.75, True),   # 0.75 meets threshold
        (0.90, True)    # 0.90 meets threshold
    ])
    def test_pipeline_confidence_threshold_gate(
        self,
        temp_db: DatabaseManager,
        confidence: float,
        expected_admitted: bool
    ):
        """Main pipeline admits items strictly when confidence >= configured threshold (0.75)."""
        mock_response = json.dumps({
            "category": "Risk",
            "title": "Vendor delivery slip",
            "description": "Vendor delivery may slip past deadline.",
            "confidence": confidence,
            "evidence": "Vendor delivery may slip past deadline.",
            "source_signal_id": "sig-conf-test"
        })
        provider = MockLLMProvider(mock_response=mock_response)
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(
            provider=provider,
            confidence_threshold=0.75,
            db_manager=temp_db
        )

        signal = ProjectSignal(
            id=f"sig-conf-{int(confidence*100)}",
            source_type=SourceType.EMAIL,
            content="Vendor delivery may slip past deadline."
        )
        results = pipeline.process_signals([signal], user_id="test_user")

        if expected_admitted:
            assert results["classified_drafts"] == 1
            assert len(results["drafts"]) == 1
            assert results["drafts"][0]["confidence_score"] == confidence
            # Check DB classification status
            sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
            assert sig_rec["classification_status"] == "COMPLETED"
        else:
            assert results["classified_drafts"] == 0
            assert len(results["drafts"]) == 0
            # Check DB classification status is BELOW_CONFIDENCE_THRESHOLD
            sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
            assert sig_rec["classification_status"] == "BELOW_CONFIDENCE_THRESHOLD"

    @pytest.mark.parametrize("confidence,expected_admitted", [
        (0.69, False),
        (0.70, False),
        (0.74, False),
        (0.75, True),
        (0.90, True)
    ])
    def test_meeting_minutes_confidence_threshold_gate(
        self,
        temp_db: DatabaseManager,
        tmp_path: Path,
        confidence: float,
        expected_admitted: bool
    ):
        """Meeting Minutes workflow enforces the same authoritative threshold (0.75)."""
        mock_response = json.dumps({
            "category": "Decision",
            "title": "Agreed to extend sprint",
            "description": "SteerCo agreed to extend sprint by one week.",
            "confidence": confidence,
            "evidence": "SteerCo agreed to extend sprint by one week.",
            "source_signal_id": "sig-mom-conf"
        })
        provider = MockLLMProvider(mock_response=mock_response)
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(
            provider=provider,
            confidence_threshold=0.75,
            db_manager=temp_db
        )
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        mom_file = tmp_path / f"mom_conf_{int(confidence*100)}.txt"
        mom_file.write_text("SteerCo agreed to extend sprint by one week.", encoding="utf-8")

        results = service.process_meeting_minutes(file_paths=[mom_file], user_id="test_user")

        if expected_admitted:
            assert results["drafts_created"] == 1
        else:
            assert results["drafts_created"] == 0

    def test_dynamic_threshold_adjustment(self, temp_db: DatabaseManager):
        """Changing the threshold to 0.80 rejects 0.78, but changing to 0.65 admits 0.70."""
        mock_response = json.dumps({
            "category": "Issue",
            "title": "Database connection drop",
            "description": "DB connection dropped during load test.",
            "confidence": 0.72,
            "evidence": "DB connection dropped during load test."
        })
        provider = MockLLMProvider(mock_response=mock_response)
        pipeline = RAIDPipeline(temp_db)

        # 1. Configured at 0.80 -> 0.72 is rejected
        pipeline.classifier = RAIDClassifier(provider=provider, confidence_threshold=0.80, db_manager=temp_db)
        res1 = pipeline.process_signals([ProjectSignal(id="sig-dyn-1", source_type=SourceType.CHAT, content="DB connection dropped during load test.")])
        assert res1["classified_drafts"] == 0

        # 2. Configured at 0.65 -> 0.72 is admitted
        pipeline.classifier = RAIDClassifier(provider=provider, confidence_threshold=0.65, db_manager=temp_db)
        res2 = pipeline.process_signals([ProjectSignal(id="sig-dyn-2", source_type=SourceType.CHAT, content="DB connection dropped during load test.")])
        assert res2["classified_drafts"] == 1


# =============================================================================
# 2. Reconcile Dependency Classification Tests
# =============================================================================
class TestDependencyClassificationLifecycle:
    def test_dependency_extraction_and_persistence_from_email(self, test_pipeline: RAIDPipeline, temp_db: DatabaseManager):
        """Dependency extracted from Email is saved as a draft with type='Dependency'."""
        mock_json = json.dumps({
            "category": "Dependency",
            "title": "Basis system refresh prerequisite",
            "description": "Config testing depends on Basis completing sandbox refresh.",
            "confidence": 0.92,
            "evidence": "Configuration testing can begin only after the Basis team completes the system refresh.",
            "owner": "Basis Team",
            "due_date": "2026-12-01"
        })
        test_pipeline.classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json), db_manager=temp_db)

        signal = ProjectSignal(
            id="sig-dep-email-01",
            source_type=SourceType.EMAIL,
            author="Test Lead",
            subject="Config Test Dependency",
            content="Configuration testing can begin only after the Basis team completes the system refresh."
        )
        res = test_pipeline.process_signals([signal], user_id="test_user")

        assert res["classified_drafts"] == 1
        draft = res["drafts"][0]
        assert draft["type"] == "Dependency"
        assert draft["status"] == "Draft"
        assert draft["likely_owner"] == "Basis Team"

    def test_dependency_extraction_and_persistence_from_chat(self, test_pipeline: RAIDPipeline, temp_db: DatabaseManager):
        """Dependency extracted from Chat is saved as a draft with type='Dependency'."""
        mock_json = json.dumps({
            "category": "Dependency",
            "title": "CPI tenant provisioning dependency",
            "description": "Integration testing waiting on middleware CPI tenant.",
            "confidence": 0.88,
            "evidence": "Interface testing can start only after middleware CPI tenant is provisioned.",
            "owner": "Integration Lead"
        })
        test_pipeline.classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json), db_manager=temp_db)

        signal = ProjectSignal(
            id="sig-dep-chat-01",
            source_type=SourceType.CHAT,
            author="Dev",
            content="Interface testing can start only after middleware CPI tenant is provisioned."
        )
        res = test_pipeline.process_signals([signal], user_id="test_user")

        assert res["classified_drafts"] == 1
        assert res["drafts"][0]["type"] == "Dependency"

    def test_dependency_extraction_from_meeting_minutes(self, temp_db: DatabaseManager, tmp_path: Path):
        """Dependency extracted from Meeting Minutes is preserved and saved into drafts."""
        mock_json = json.dumps({
            "category": "Dependency",
            "title": "SecOps certificate deployment",
            "description": "Open banking integration pending TLS 1.3 certificate deployment by SecOps.",
            "confidence": 0.91,
            "evidence": "Open banking integration pending TLS 1.3 certificate deployment by SecOps."
        })
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json), db_manager=temp_db)
        service = RaidService(db_manager=temp_db, pipeline=pipeline)

        mom_file = tmp_path / "mom_dep.txt"
        mom_file.write_text("Open banking integration pending TLS 1.3 certificate deployment by SecOps.", encoding="utf-8")

        res = service.process_meeting_minutes(file_paths=[mom_file], user_id="test_user")

        assert res["status"] == "success"
        assert res["drafts_created"] == 1
        drafts = service.get_pending_drafts(user_id="test_user")
        assert len(drafts) == 1
        assert drafts[0]["type"] == "Dependency"
        assert drafts[0]["status"] == "Draft"

    def test_dependency_approval_rejection_modification_lifecycle(self, test_service: RaidService, temp_db: DatabaseManager):
        """Dependency drafts can be approved, rejected, modified, and queried seamlessly."""
        # 1. Manually insert a Dependency draft
        draft_id = "raid-dep-test-lifecycle"
        temp_db.save_raid_draft({
            "id": draft_id,
            "user_id": "test_user",
            "type": "Dependency",
            "title": "CPI tenant prerequisite",
            "description": "Waiting for tenant",
            "workstream": "Integration",
            "phase": "Realize",
            "confidence_score": 0.90,
            "evidence_text": "Waiting for tenant",
            "evidence_source_id": "sig-001",
            "evidence_source_type": "email",
            "detection_source": "llm_extraction",
            "status": "Draft",
            "evidence_status": "verified"
        }, user_id="test_user")

        # 2. Modify Dependency
        test_service.modify_raid(
            raid_id=draft_id,
            modified_by="PM Lead",
            updates={"title": "Updated CPI prerequisite"},
            comments="Clarified description",
            user_id="test_user"
        )
        updated = test_service.get_raid_by_id(draft_id, user_id="test_user")
        assert updated["title"] == "Updated CPI prerequisite"
        assert updated["type"] == "Dependency"

        # 3. Approve Dependency
        approved_id = test_service.approve_raid(
            raid_id=draft_id,
            approved_by="PM Lead",
            user_id="test_user"
        )
        assert approved_id is not None
        approved_items = test_service.get_approved_items(user_id="test_user")
        assert any(item["type"] == "Dependency" for item in approved_items)

        # 4. Reject another Dependency draft
        reject_draft_id = "raid-dep-reject"
        temp_db.save_raid_draft({
            "id": reject_draft_id,
            "user_id": "test_user",
            "type": "Dependency",
            "title": "Duplicate Dependency",
            "description": "Already tracked",
            "confidence_score": 0.85,
            "evidence_text": "Duplicate quote",
            "evidence_source_id": "sig-002",
            "evidence_source_type": "chat",
            "detection_source": "llm_extraction",
            "status": "Draft"
        }, user_id="test_user")

        test_service.reject_raid(
            raid_id=reject_draft_id,
            rejected_by="PM Lead",
            reason="Duplicate item",
            user_id="test_user"
        )
        rejected = test_service.get_rejected_drafts(user_id="test_user")
        assert any(r["id"] == reject_draft_id and r["type"] == "Dependency" for r in rejected)


# =============================================================================
# 3. Failed Classification Tracking Tests
# =============================================================================
class TestFailedClassificationTracking:
    def test_api_timeout_failure_tracking(self, temp_db: DatabaseManager):
        """Simulated API timeout records status=FAILED, error_category=API_TIMEOUT, and no draft."""
        provider = MockLLMProvider(fail_times=999, exception_to_raise=TimeoutError("Request timed out after 30s"))
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=temp_db)

        signal = ProjectSignal(id="sig-timeout-1", source_type=SourceType.EMAIL, content="Important project milestone delayed.")
        results = pipeline.process_signals([signal], user_id="test_user")

        assert results["classified_drafts"] == 0
        sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
        assert sig_rec["classification_status"] == "FAILED"
        assert sig_rec["classification_error_category"] == "API_TIMEOUT"
        assert "timed out" in sig_rec["classification_failure_reason"].lower()
        assert sig_rec["classification_attempts"] == 1

    def test_rate_limit_429_failure_tracking(self, temp_db: DatabaseManager):
        """Simulated 429 quota exhaustion records status=FAILED, error_category=RATE_LIMIT."""
        provider = MockLLMProvider(fail_times=999, exception_to_raise=RuntimeError("ResourceExhausted 429 quota exceeded"))
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=temp_db)

        signal = ProjectSignal(id="sig-429-1", source_type=SourceType.CHAT, content="Critical interface failure.")
        results = pipeline.process_signals([signal], user_id="test_user")

        assert results["classified_drafts"] == 0
        sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
        assert sig_rec["classification_status"] == "FAILED"
        assert sig_rec["classification_error_category"] == "RATE_LIMIT"

    def test_auth_failure_sanitization(self, temp_db: DatabaseManager):
        """Auth failures mask API keys and passwords from error messages."""
        secret_key = "AIzaSyD-mockSecretKey1234567890123"
        provider = MockLLMProvider(fail_times=999, exception_to_raise=ValueError(f"Invalid API key: {secret_key}"))
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=temp_db)

        signal = ProjectSignal(id="sig-auth-1", source_type=SourceType.EMAIL, content="Project update.")
        pipeline.process_signals([signal], user_id="test_user")

        sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
        assert sig_rec["classification_status"] == "FAILED"
        assert sig_rec["classification_error_category"] == "AUTHENTICATION_FAILURE"
        # Verify secret was sanitized
        assert secret_key not in sig_rec["classification_failure_reason"]
        assert "[REDACTED_API_KEY]" in sig_rec["classification_failure_reason"]

    def test_empty_llm_response_failure(self, temp_db: DatabaseManager):
        """Empty LLM response records EMPTY_RESPONSE error category."""
        provider = MockLLMProvider(mock_response="   \n  ")
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=temp_db)

        signal = ProjectSignal(id="sig-empty-1", source_type=SourceType.EMAIL, content="Content here.")
        pipeline.process_signals([signal], user_id="test_user")

        sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
        assert sig_rec["classification_status"] == "FAILED"
        assert sig_rec["classification_error_category"] == "EMPTY_RESPONSE"

    def test_invalid_json_failure(self, temp_db: DatabaseManager):
        """Malformed JSON string records INVALID_JSON error category."""
        provider = MockLLMProvider(mock_response="{invalid_json: true, unterminated")
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=temp_db)

        signal = ProjectSignal(id="sig-json-1", source_type=SourceType.EMAIL, content="Content here.")
        pipeline.process_signals([signal], user_id="test_user")

        sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
        assert sig_rec["classification_status"] == "FAILED"
        assert sig_rec["classification_error_category"] == "INVALID_JSON"

    def test_pydantic_schema_validation_failure(self, temp_db: DatabaseManager):
        """Missing mandatory category or invalid confidence bounds records SCHEMA_VALIDATION_ERROR."""
        provider = MockLLMProvider(mock_response=json.dumps({"title": "No category provided"}))
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=temp_db)

        signal = ProjectSignal(id="sig-schema-1", source_type=SourceType.EMAIL, content="Content here.")
        pipeline.process_signals([signal], user_id="test_user")

        sig_rec = temp_db.get_raw_signal_by_id(signal.id, user_id="test_user")
        assert sig_rec["classification_status"] == "FAILED"
        assert sig_rec["classification_error_category"] == "SCHEMA_VALIDATION_ERROR"

    def test_retry_failed_signals_recovery(self, temp_db: DatabaseManager):
        """Failed signals can be retried and recovered into drafts without duplicates."""
        # 1. First run with failing provider
        failing_provider = MockLLMProvider(fail_times=999, exception_to_raise=TimeoutError("Temporary timeout"))
        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=failing_provider, db_manager=temp_db)

        signal = ProjectSignal(
            id="sig-retry-recover-1",
            source_type=SourceType.EMAIL,
            content="Agreed that SD pricing user exit refactoring carries high risk."
        )
        pipeline.process_signals([signal], user_id="test_user")

        failed_signals = temp_db.get_failed_signals(user_id="test_user")
        assert len(failed_signals) == 1
        assert failed_signals[0]["id"] == "sig-retry-recover-1"

        # 2. Swap to working provider and retry
        working_provider = MockLLMProvider()
        pipeline.classifier = RAIDClassifier(provider=working_provider, db_manager=temp_db)

        retry_res = pipeline.retry_failed_signals(user_id="test_user")
        assert retry_res["retried_count"] == 1
        assert retry_res["recovered_drafts"] == 1
        assert retry_res["still_failed"] == 0

        # Check DB status updated to COMPLETED
        sig_rec = temp_db.get_raw_signal_by_id("sig-retry-recover-1", user_id="test_user")
        assert sig_rec["classification_status"] == "COMPLETED"
        assert sig_rec["classification_attempts"] == 2

        # 3. Running retry a second time does not duplicate drafts
        retry_res_2 = pipeline.retry_failed_signals(user_id="test_user")
        assert retry_res_2["retried_count"] == 0
        assert retry_res_2["recovered_drafts"] == 0

    def test_one_failure_does_not_halt_batch(self, temp_db: DatabaseManager):
        """In a multi-signal batch, a failure on signal 1 does not prevent signal 2 from classifying."""
        class AlternatingProvider(MockLLMProvider):
            def generate_text(self, prompt: str, system_instruction=None):
                if "fail_this_signal" in prompt.lower():
                    raise TimeoutError("Simulated failure for item 1")
                return json.dumps({
                    "category": "Decision",
                    "title": "Go live approved",
                    "description": "Go live approved by SteerCo",
                    "confidence": 0.95,
                    "evidence": "Go live approved by SteerCo."
                })

        pipeline = RAIDPipeline(temp_db)
        pipeline.classifier = RAIDClassifier(provider=AlternatingProvider(), db_manager=temp_db)

        sig1 = ProjectSignal(id="sig-batch-1", source_type=SourceType.EMAIL, content="Fail_this_signal please.")
        sig2 = ProjectSignal(id="sig-batch-2", source_type=SourceType.EMAIL, content="Go live approved by SteerCo.")

        res = pipeline.process_signals([sig1, sig2], user_id="test_user")

        assert res["total_ingested"] == 2
        assert res["classified_drafts"] == 1
        assert temp_db.get_raw_signal_by_id("sig-batch-1")["classification_status"] == "FAILED"
        assert temp_db.get_raw_signal_by_id("sig-batch-2")["classification_status"] == "COMPLETED"


# =============================================================================
# 4. Evidence Fallback & Verification Tests
# =============================================================================
class TestEvidenceVerificationAndFallback:
    def test_01_valid_exact_evidence(self):
        """Exact verbatim quote is marked as VERIFIED."""
        content = "The client may not provide the required master data before the integration testing phase."
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Master data delay",
            "confidence": 0.90,
            "evidence": content
        })
        classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json))
        res = classifier.classify_signal(ProjectSignal(source_type=SourceType.EMAIL, content=content))

        assert res.evidence_status == EvidenceStatus.VERIFIED
        assert res.evidence == content

    def test_02_valid_evidence_casing_tolerance(self):
        """Case differences (lowercase vs uppercase) still verify successfully."""
        content = "The Client May Not Provide Master Data Before Testing."
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Master data delay",
            "confidence": 0.90,
            "evidence": "the client may not provide master data before testing."
        })
        classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json))
        res = classifier.classify_signal(ProjectSignal(source_type=SourceType.EMAIL, content=content))

        assert res.evidence_status == EvidenceStatus.VERIFIED

    def test_03_valid_evidence_whitespace_tolerance(self):
        """Tabs, newlines, and multiple spaces still verify successfully."""
        content = "The client may not provide\n\tthe required   master data before SIT."
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Master data delay",
            "confidence": 0.90,
            "evidence": "The client may not provide the required master data before SIT."
        })
        classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json))
        res = classifier.classify_signal(ProjectSignal(source_type=SourceType.EMAIL, content=content))

        assert res.evidence_status == EvidenceStatus.VERIFIED

    def test_04_hallucinated_evidence_fallback(self):
        """Hallucinated quote is replaced by the authentic sentence with status=FALLBACK."""
        content = "Actual statement: Third-party interface testing may run late due to pending firewall open."
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Interface testing delay",
            "description": "Testing may run late",
            "confidence": 0.88,
            "evidence": "A fabricated quote that does not exist in the source content at all."
        })
        classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json))
        res = classifier.classify_signal(ProjectSignal(source_type=SourceType.EMAIL, content=content))

        assert res.evidence_status == EvidenceStatus.FALLBACK
        assert "Third-party interface testing may run late" in res.evidence
        assert "fabricated quote" not in res.evidence

    def test_05_missing_evidence_fallback(self):
        """Missing or null evidence extracts authentic sentence with status=FALLBACK."""
        content = "Integration testing is currently blocked because the client has not provided master data."
        mock_json = json.dumps({
            "category": "Issue",
            "title": "Integration testing blocked",
            "description": "Testing blocked by master data",
            "confidence": 0.92,
            "evidence": None
        })
        classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json))
        res = classifier.classify_signal(ProjectSignal(source_type=SourceType.CHAT, content=content))

        assert res.evidence_status == EvidenceStatus.FALLBACK
        assert "Integration testing is currently blocked" in res.evidence

    def test_06_relevant_statement_after_greeting(self):
        """Ignores introductory greeting and selects the substantive RAID statement."""
        content = (
            "Hi team, good morning! Hope everyone had a great weekend. "
            "Critical transport TRK9001 failed with RC=8 and blocked testing. "
            "Please check the log."
        )
        fallback = find_best_fallback_sentence(
            content=content,
            title="Critical transport TRK9001 failed",
            description="Testing blocked by transport failure",
            category="Issue"
        )
        assert fallback is not None
        assert "TRK9001 failed with RC=8" in fallback
        assert "Hi team" not in fallback

    def test_07_relevant_statement_before_disclaimer(self):
        """Ignores trailing confidentiality disclaimer and selects the substantive statement."""
        content = (
            "Decided to proceed with brownfield migration approach for ECC financials. "
            "CONFIDENTIALITY NOTICE: This transmission is intended only for the use of the addressee. "
            "If you have received this email in error, please delete it immediately."
        )
        fallback = find_best_fallback_sentence(
            content=content,
            title="Decided to proceed with brownfield migration",
            description="SteerCo decision on brownfield approach",
            category="Decision"
        )
        assert fallback is not None
        assert "brownfield migration approach" in fallback
        assert "CONFIDENTIALITY NOTICE" not in fallback

    def test_08_no_identifiable_supporting_sentence(self):
        """When source content contains only generic conversational filler, evidence is marked UNAVAILABLE."""
        content = "Ok sounds good will do."
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Master data delay",
            "description": "Potential delay",
            "confidence": 0.85,
            "evidence": "Fabricated text"
        })
        classifier = RAIDClassifier(provider=MockLLMProvider(mock_response=mock_json))
        res = classifier.classify_signal(ProjectSignal(source_type=SourceType.CHAT, content=content))

        assert res.evidence_status == EvidenceStatus.UNAVAILABLE
        assert res.evidence == ""
