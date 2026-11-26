"""
Regression tests for Stage 6C: Low-Confidence RAID Human Review Queue.

Requirements Covered:
1. Scenario 1: Below-threshold signals appear in low-confidence review section.
2. Scenario 2: Signals are not automatically inserted into approved_raid.
3. Scenario 3: Reviewer confirmation creates a Draft, not an approved record.
4. Scenario 4: NOT_RAID decisions do not create drafts in raid_drafts.
5. Scenario 5: Deferred candidates remain retrievable in subsequent calls.
6. Scenario 6: Duplicate confirmation on the same signal is prevented (DuplicateApprovalError).
7. Scenario 7: Reviewer identity and actions are audited in audit_history.
8. Scenario 8: Tenant isolation is maintained across different user_ids.
9. Scenario 9: Owner and due-date NULL values are preserved upon confirmation.
10. Scenario 10: Existing normal RAID review queue functionality remains unaffected.
"""

import json
from pathlib import Path
import pytest

from src.normalisation.schemas import ProjectSignal, SourceType
from src.classification.schemas import RAIDCategory, RAIDExtractionResult
from src.classification.classifier import RAIDClassifier
from src.classification.providers.mock_provider import MockLLMProvider
from src.pipeline import RAIDPipeline
from src.service import (
    RaidService,
    DuplicateApprovalError,
    RaidValidationError,
    RaidNotFoundError,
)
from src.database.db_manager import DatabaseManager
from src.database.migrations import run_migrations


@pytest.fixture
def clean_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "stage6c_test.db"
    db = DatabaseManager(db_file)
    run_migrations(db_file)
    return db


@pytest.fixture
def service(clean_db: DatabaseManager) -> RaidService:
    return RaidService(db_manager=clean_db)


def create_pipeline_with_mock(db: DatabaseManager, mock_payload: dict) -> RAIDPipeline:
    provider = MockLLMProvider(mock_response=json.dumps(mock_payload))
    pipeline = RAIDPipeline(db)
    pipeline.classifier = RAIDClassifier(provider=provider, db_manager=db)
    return pipeline


# ===========================================================================
# Test Cases for Stage 6C
# ===========================================================================

class TestLowConfidenceQueue:

    def test_scenario_1_below_threshold_signals_appear_in_low_confidence_queue(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 1: Signals below confidence threshold (0.75) are marked
        BELOW_CONFIDENCE_THRESHOLD, do not appear in pending drafts, but do
        appear in the low-confidence candidates list.
        """
        mock_payload = {
            "category": "Risk",
            "title": "Vendor contract negotiation delay",
            "description": "Possible SLA delay on vendor signing.",
            "confidence": 0.65,  # Below 0.75 threshold
            "owner": "Sarah Vendor Lead",
            "evidence": "Vendor negotiations are taking longer than scheduled.",
            "reasoning_summary": "Uncertain impact on baseline milestone.",
            "probability": "Medium",
            "impact": "Medium"
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-low-001",
            source_type=SourceType.EMAIL,
            author="procurement@enterprise.com",
            content="Vendor negotiations are taking longer than scheduled."
        )

        res = pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)
        assert res["classified_drafts"] == 0

        # Check raw_signal status in DB
        sig_row = clean_db.get_raw_signal_by_id("sig-low-001", user_id="tenant_1")
        assert sig_row["classification_status"] == "BELOW_CONFIDENCE_THRESHOLD"

        # Must NOT appear in standard pending drafts
        standard_drafts = service.get_pending_drafts(user_id="tenant_1")
        assert len(standard_drafts) == 0

        # MUST appear in low-confidence candidates
        low_conf = service.get_low_confidence_candidates(user_id="tenant_1")
        assert len(low_conf) == 1
        cand = low_conf[0]
        assert cand["signal_id"] == "sig-low-001"
        assert cand["predicted_category"] == "Risk"
        assert cand["confidence_score"] == pytest.approx(0.65)
        assert cand["status"] == "BELOW_CONFIDENCE_THRESHOLD"
        assert cand["likely_owner"] == "Sarah Vendor Lead"
        assert cand["reasoning_summary"] == "Uncertain impact on baseline milestone."

    def test_scenario_2_signals_not_automatically_inserted_into_approved_raid(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 2: Low-confidence signals are never automatically approved
        into approved_raid table.
        """
        mock_payload = {
            "category": "Issue",
            "title": "Test environment connectivity blip",
            "description": "Transient network timeout.",
            "confidence": 0.55,
            "owner": None,
            "evidence": "Network timeout in QA environment.",
            "reasoning_summary": "Transient event."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-low-002",
            source_type=SourceType.CHAT,
            author="qa_lead@kaartech.com",
            content="Network timeout in QA environment."
        )

        pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)

        # Neither raid_drafts nor approved_raid should have records
        assert len(clean_db.get_pending_drafts(user_id="tenant_1")) == 0
        assert len(clean_db.get_approved_items(user_id="tenant_1")) == 0

    def test_scenario_3_reviewer_confirmation_creates_draft_not_approved(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 3: Human confirmation creates a record in raid_drafts with status 'Draft'.
        It does NOT write to approved_raid.
        """
        mock_payload = {
            "category": "Dependency",
            "title": "External partner API delivery",
            "description": "Waiting for third-party API release.",
            "confidence": 0.70,
            "owner": "Partner Manager",
            "evidence": "Third-party API release scheduled for next week.",
            "reasoning_summary": "External dependency on partner timeline."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-low-003",
            source_type=SourceType.EMAIL,
            author="integrator@partner.com",
            content="Third-party API release scheduled for next week."
        )
        pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)

        # Confirm candidate
        draft = service.confirm_low_confidence_candidate(
            signal_id="sig-low-003",
            reviewer_name="Sarah PM",
            comments="Verified with partner delivery lead",
            user_id="tenant_1"
        )

        assert draft["status"] == "Draft"
        assert draft["type"] == "Dependency"
        assert draft["confidence_score"] == pytest.approx(0.70)
        assert draft["likely_owner"] == "Partner Manager"

        # Check DB: Present in pending drafts, ABSENT in approved_raid
        drafts = service.get_pending_drafts(user_id="tenant_1")
        assert len(drafts) == 1
        assert drafts[0]["id"] == draft["id"]

        approved = clean_db.get_approved_items(user_id="tenant_1")
        assert len(approved) == 0

        # Also verify signal is now excluded from low-confidence queue
        assert len(service.get_low_confidence_candidates(user_id="tenant_1")) == 0

    def test_scenario_4_not_raid_decisions_do_not_create_drafts(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 4: Marking candidate NOT_RAID updates signal status to NOT_RAID
        and does NOT create a draft in raid_drafts.
        """
        mock_payload = {
            "category": "Risk",
            "title": "Possible weather disruption to office commute",
            "description": "Rain forecast tomorrow.",
            "confidence": 0.40,
            "owner": None,
            "evidence": "Rain forecast tomorrow.",
            "reasoning_summary": "Routine chatter."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-low-004",
            source_type=SourceType.CHAT,
            author="dev1@company.com",
            content="Rain forecast tomorrow, working from home."
        )
        pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)

        # Reject candidate
        res = service.reject_low_confidence_candidate(
            signal_id="sig-low-004",
            reviewer_name="David Ross",
            rejection_reason="Chit-chat / office logistics, not project RAID.",
            user_id="tenant_1"
        )
        assert res is True

        # Signal status updated
        signal_row = clean_db.get_raw_signal_by_id("sig-low-004", user_id="tenant_1")
        assert signal_row["classification_status"] == "NOT_RAID"

        # No drafts created
        assert len(clean_db.get_pending_drafts(user_id="tenant_1")) == 0
        assert len(clean_db.get_approved_items(user_id="tenant_1")) == 0

        # No longer in low-confidence queue
        assert len(service.get_low_confidence_candidates(user_id="tenant_1")) == 0

    def test_scenario_5_deferred_candidates_remain_retrievable(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 5: Deferred candidates transition to DEFERRED status and remain
        available in subsequent get_low_confidence_candidates calls.
        """
        mock_payload = {
            "category": "Assumption",
            "title": "Assuming legacy data format is UTF-8",
            "description": "Need to confirm legacy encoding.",
            "confidence": 0.60,
            "owner": None,
            "evidence": "Assuming legacy data format is UTF-8.",
            "reasoning_summary": "Unverified assumption."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-low-005",
            source_type=SourceType.EMAIL,
            author="analyst@kaartech.com",
            content="Assuming legacy data format is UTF-8."
        )
        pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)

        # Defer candidate
        res = service.defer_low_confidence_candidate(
            signal_id="sig-low-005",
            reviewer_name="David Ross",
            deferral_notes="Waiting for data architect confirmation tomorrow.",
            user_id="tenant_1"
        )
        assert res is True

        # Signal status is DEFERRED
        signal_row = clean_db.get_raw_signal_by_id("sig-low-005", user_id="tenant_1")
        assert signal_row["classification_status"] == "DEFERRED"

        # Candidate still appears in low-confidence queue
        cands = service.get_low_confidence_candidates(user_id="tenant_1", include_deferred=True)
        assert len(cands) == 1
        assert cands[0]["signal_id"] == "sig-low-005"
        assert cands[0]["status"] == "DEFERRED"

        # If include_deferred=False, excluded
        cands_no_def = service.get_low_confidence_candidates(user_id="tenant_1", include_deferred=False)
        assert len(cands_no_def) == 0

    def test_scenario_6_duplicate_confirmation_prevented(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 6: Attempting to confirm the same candidate twice raises
        DuplicateApprovalError and does not duplicate the draft.
        """
        mock_payload = {
            "category": "Risk",
            "title": "Server provisioning delay",
            "description": "Hardware delivery delayed by 2 weeks.",
            "confidence": 0.68,
            "owner": "Infra Lead",
            "evidence": "Hardware delivery delayed by 2 weeks.",
            "reasoning_summary": "Infra delay."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-low-006",
            source_type=SourceType.EMAIL,
            author="infra@kaartech.com",
            content="Hardware delivery delayed by 2 weeks."
        )
        pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)

        # First confirmation
        service.confirm_low_confidence_candidate(
            signal_id="sig-low-006",
            reviewer_name="Sarah PM",
            user_id="tenant_1"
        )

        # Second confirmation must raise DuplicateApprovalError
        with pytest.raises(DuplicateApprovalError):
            service.confirm_low_confidence_candidate(
                signal_id="sig-low-006",
                reviewer_name="Sarah PM",
                user_id="tenant_1"
            )

        # Only 1 draft exists
        drafts = clean_db.get_pending_drafts(user_id="tenant_1")
        assert len(drafts) == 1

    def test_scenario_7_reviewer_identity_and_actions_audited(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 7: Reviewer identity and all lifecycle actions (confirm, reject, defer)
        are recorded in audit_history.
        """
        mock_payload = {
            "category": "Risk",
            "title": "Audit test signal",
            "description": "Audit trail verification.",
            "confidence": 0.50,
            "owner": None,
            "evidence": "Audit trail verification.",
            "reasoning_summary": "Audit test."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        # 1. Test confirmation audit
        sig1 = ProjectSignal(
            id="sig-audit-01",
            source_type=SourceType.EMAIL,
            author="sender@enterprise.com",
            content="Audit test signal 1."
        )
        pipeline.process_signals([sig1], user_id="tenant_1", run_predictive_engine=False)

        draft = service.confirm_low_confidence_candidate(
            signal_id="sig-audit-01",
            reviewer_name="Auditor Alice",
            comments="SteerCo review approval",
            user_id="tenant_1"
        )

        history = clean_db.get_audit_history(item_id=draft["id"], user_id="tenant_1")
        assert len(history) >= 1
        entry = history[0]
        assert entry["action"] == "CONFIRMED_LOW_CONFIDENCE"
        assert entry["action_by"] == "Auditor Alice"
        assert "SteerCo review approval" in entry["action_notes"]

        # 2. Test rejection audit
        sig2 = ProjectSignal(
            id="sig-audit-02",
            source_type=SourceType.EMAIL,
            author="sender@enterprise.com",
            content="Audit test signal 2."
        )
        pipeline.process_signals([sig2], user_id="tenant_1", run_predictive_engine=False)

        service.reject_low_confidence_candidate(
            signal_id="sig-audit-02",
            reviewer_name="Auditor Bob",
            rejection_reason="False positive rejection",
            user_id="tenant_1"
        )
        history_rej = clean_db.get_audit_history(item_id="sig-audit-02", user_id="tenant_1")
        assert len(history_rej) >= 1
        assert history_rej[0]["action"] == "REJECTED_LOW_CONFIDENCE"
        assert history_rej[0]["action_by"] == "Auditor Bob"

        # 3. Test deferral audit
        sig3 = ProjectSignal(
            id="sig-audit-03",
            source_type=SourceType.EMAIL,
            author="sender@enterprise.com",
            content="Audit test signal 3."
        )
        pipeline.process_signals([sig3], user_id="tenant_1", run_predictive_engine=False)

        service.defer_low_confidence_candidate(
            signal_id="sig-audit-03",
            reviewer_name="Auditor Charlie",
            deferral_notes="Deferred to next sprint",
            user_id="tenant_1"
        )
        history_def = clean_db.get_audit_history(item_id="sig-audit-03", user_id="tenant_1")
        assert len(history_def) >= 1
        assert history_def[0]["action"] == "DEFERRED_LOW_CONFIDENCE"
        assert history_def[0]["action_by"] == "Auditor Charlie"

    def test_scenario_8_tenant_isolation_maintained(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 8: Candidates for Tenant A are not visible to Tenant B,
        and Tenant B cannot confirm Tenant A's candidates.
        """
        mock_payload = {
            "category": "Risk",
            "title": "Tenant isolated signal",
            "description": "Tenant confidentiality.",
            "confidence": 0.60,
            "owner": None,
            "evidence": "Tenant confidentiality.",
            "reasoning_summary": "Tenant check."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-tenant-01",
            source_type=SourceType.EMAIL,
            author="tenant_a_user@client.com",
            content="Tenant confidentiality."
        )
        pipeline.process_signals([sig], user_id="tenant_A", run_predictive_engine=False)

        # Tenant A sees candidate
        assert len(service.get_low_confidence_candidates(user_id="tenant_A")) == 1

        # Tenant B sees 0 candidates
        assert len(service.get_low_confidence_candidates(user_id="tenant_B")) == 0

        # Tenant B cannot confirm Tenant A's signal
        with pytest.raises(RaidNotFoundError):
            service.confirm_low_confidence_candidate(
                signal_id="sig-tenant-01",
                reviewer_name="Malicious Tenant B",
                user_id="tenant_B"
            )

    def test_scenario_9_owner_and_due_date_null_values_preserved(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 9: When confirming a candidate with owner=None, the owner remains None.
        The sender is NOT substituted as the owner.
        """
        mock_payload = {
            "category": "Risk",
            "title": "Unassigned security scan finding",
            "description": "Open port 22 on staging server.",
            "confidence": 0.70,
            "owner": None,
            "due_date": None,
            "evidence": "Open port 22 on staging server.",
            "reasoning_summary": "Unassigned finding."
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-null-owner-01",
            source_type=SourceType.EMAIL,
            author="scanner_daemon@internal.net",
            content="Open port 22 on staging server."
        )
        pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)

        draft = service.confirm_low_confidence_candidate(
            signal_id="sig-null-owner-01",
            reviewer_name="Security PM",
            user_id="tenant_1"
        )

        # Accountable owner must be None, NOT scanner_daemon@internal.net
        assert draft["likely_owner"] is None
        assert draft["due_date"] == "" or draft["due_date"] is None

        # Check DB row
        db_draft = clean_db.get_pending_drafts(user_id="tenant_1")[0]
        assert db_draft["likely_owner"] is None

    def test_scenario_10_normal_raid_review_queue_unaffected(
        self, clean_db: DatabaseManager, service: RaidService
    ):
        """
        Scenario 10: Standard high-confidence signals (>= 0.75) continue to flow
        normally into raid_drafts, do not appear in low-confidence queue, and
        can be approved via service.approve_raid().
        """
        mock_payload = {
            "category": "Risk",
            "title": "High confidence migration risk",
            "description": "Database cutover window insufficient.",
            "confidence": 0.95,
            "owner": "DBA Lead",
            "evidence": "Database cutover window insufficient.",
            "reasoning_summary": "High risk identified.",
            "probability": "High",
            "impact": "High"
        }
        pipeline = create_pipeline_with_mock(clean_db, mock_payload)

        sig = ProjectSignal(
            id="sig-high-001",
            source_type=SourceType.EMAIL,
            author="dba@kaartech.com",
            content="Database cutover window insufficient."
        )
        res = pipeline.process_signals([sig], user_id="tenant_1", run_predictive_engine=False)
        assert res["classified_drafts"] == 1

        # Normal pending drafts has 1
        drafts = service.get_pending_drafts(user_id="tenant_1")
        assert len(drafts) == 1
        draft_id = drafts[0]["id"]

        # Low-confidence queue has 0
        assert len(service.get_low_confidence_candidates(user_id="tenant_1")) == 0

        # Standard approval flow operates unaffected
        approved = service.approve_raid(
            raid_id=draft_id,
            approved_by="Project Director",
            comments="Approved for inclusion in SteerCo register",
            user_id="tenant_1"
        )
        assert approved["id"].startswith("appr-")
        assert approved["draft_id"] == draft_id
        assert approved["status"] == "Open"
        assert approved["approved_by"] == "Project Director"
        assert len(service.get_pending_drafts(user_id="tenant_1")) == 0
        assert len(clean_db.get_approved_items(user_id="tenant_1")) == 1
