"""
Unit tests for the RaidService Layer.
Covers:
Retrieval:
1. Retrieve pending drafts.
2. Retrieve approved records.
3. Retrieve audit history.
4. Retrieve predictions and duplicate matches.
Approval:
5. Approve a valid pending draft.
6. Reject approval for a nonexistent draft.
7. Prevent approval of an already approved draft.
8. Prevent approval of a rejected draft.
9. Verify approved record and audit record are persisted.
10. Verify transaction rollback when approval persistence fails.
Rejection:
11. Reject a valid draft.
12. Reject an empty rejection reason.
13. Verify rejected records remain available.
14. Verify rejected records do not appear in approved register.
Modification:
15. Modify a single field.
16. Verify other fields remain unchanged.
17. Reject invalid field names.
18. Reject invalid field values.
19. Verify before-and-after values in audit history.
20. Prevent modification of approved records.
Audit:
21. Verify actor and timestamp are recorded.
22. Verify audit records are append-only through service operations.
23. Verify each business action creates the expected audit entry.
Data Integrity:
24. Verify repeated approval does not create duplicate approved records.
25. Verify source IDs and evidence are preserved.
26. Verify database failures do not leave partial updates.
"""

import pytest
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch

from src.database.db_manager import DatabaseManager
from src.service import (
    RaidService,
    RaidStatus,
    RaidServiceError,
    RaidNotFoundError,
    InvalidRaidStatusError,
    RaidValidationError,
    DuplicateApprovalError,
    AuditPersistenceError,
)
from src.normalisation.schemas import ProjectSignal, SourceType
from src.predictive_engine.schemas import PredictionItem, PredictionType
from src.deduplication.schemas import DeduplicationResult, MatchStatus


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Creates a clean, isolated SQLite DatabaseManager instance for testing."""
    db_file = tmp_path / "test_raid_service.db"
    return DatabaseManager(db_file)


@pytest.fixture
def service(temp_db: DatabaseManager) -> RaidService:
    """Creates a RaidService instance backed by an isolated database."""
    return RaidService(temp_db)


@pytest.fixture
def sample_draft(temp_db: DatabaseManager) -> dict:
    """Seeds a single valid raw signal and RAID draft in the database."""
    signal = ProjectSignal(
        source_type=SourceType.EMAIL,
        source_id="email-101",
        author="Integration Lead",
        subject="RFC Connection Failure",
        content="Testing blocked because RFC destination to S/4HANA target system times out.",
        workstream="Integration",
        phase="Realize"
    )
    temp_db.save_raw_signal(signal)

    draft = {
        "id": "raid-test-001",
        "type": "Issue",
        "title": "RFC connection failure to S/4HANA",
        "description": "Testing blocked because RFC destination to S/4HANA target system times out.",
        "workstream": "Integration",
        "phase": "Realize",
        "probability": None,
        "impact": "High",
        "severity": "High",
        "scoring_reason": "Automated LLM classification",
        "likely_owner": "Basis Team",
        "due_date": "2026-12-01",
        "mitigation_action": "Check gateway ports and RFC authorization user",
        "confidence_score": 0.95,
        "evidence_text": "RFC destination to S/4HANA target system times out.",
        "evidence_source_id": signal.id,
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "duplicate_of_id": None,
        "status": "Draft"
    }
    temp_db.save_raid_draft(draft)
    return draft


# ---------------------------------------------------------------------------
# Retrieval Tests
# ---------------------------------------------------------------------------

def test_1_retrieve_pending_drafts(service: RaidService, sample_draft: dict):
    """Retrieves all pending drafts awaiting PM review."""
    pending = service.get_pending_drafts()
    assert len(pending) == 1
    assert pending[0]["id"] == sample_draft["id"]
    assert pending[0]["title"] == sample_draft["title"]
    assert pending[0]["status"] == "Draft"


def test_2_retrieve_approved_records(service: RaidService, sample_draft: dict):
    """Initially empty; retrieves approved records once approved."""
    assert len(service.get_approved_items()) == 0
    service.approve_raid(sample_draft["id"], approved_by="Project Manager")
    approved = service.get_approved_items()
    assert len(approved) == 1
    assert approved[0]["draft_id"] == sample_draft["id"]
    assert approved[0]["approved_by"] == "Project Manager"


def test_3_retrieve_audit_history(service: RaidService, sample_draft: dict):
    """Retrieves full audit trail entries."""
    audit_trail = service.get_audit_history(sample_draft["id"])
    assert len(audit_trail) >= 1
    assert audit_trail[0]["item_id"] == sample_draft["id"]
    assert audit_trail[0]["action"] == "PROPOSED"


def test_4_retrieve_predictions_and_duplicate_matches(service: RaidService, temp_db: DatabaseManager):
    """Retrieves predictions and deduplication match records via service layer."""
    pred = PredictionItem(
        prediction_type=PredictionType.TASK_OVERDUE,
        title="Predictive Overdue Task",
        description="Task delay warning",
        confidence=0.88,
        evidence="Schedule evidence",
        source_signal_ids=["sig-123"]
    )
    temp_db.save_prediction(pred)

    dedupe = DeduplicationResult(
        new_raid_id="raid-new",
        matched_raid_id="raid-old",
        similarity_score=0.82,
        match_status=MatchStatus.LIKELY_DUPLICATE,
        match_reason="High text similarity",
        same_category=True,
        same_workstream=True,
        requires_human_review=True
    )
    temp_db.save_deduplication_match(dedupe)

    preds = service.get_predictions()
    assert len(preds) == 1
    assert preds[0]["title"] == "Predictive Overdue Task"

    matches = service.get_deduplication_matches("raid-new")
    assert len(matches) == 1
    assert matches[0]["matched_raid_id"] == "raid-old"


# ---------------------------------------------------------------------------
# Approval Tests
# ---------------------------------------------------------------------------

def test_5_approve_valid_pending_draft(service: RaidService, sample_draft: dict):
    """Approves a valid pending draft, creating an approved record and updating draft status."""
    approved_record = service.approve_raid(
        raid_id=sample_draft["id"],
        approved_by="Jane PM",
        comments="Approved for Sprint 5 cutover"
    )
    assert approved_record["draft_id"] == sample_draft["id"]
    assert approved_record["approved_by"] == "Jane PM"
    assert approved_record["status"] == "Open"

    # Verify draft status is now Accepted / Approved
    draft_after = service.get_raid_by_id(sample_draft["id"])
    assert draft_after["status"] in ("Accepted", "Approved")


def test_6_reject_approval_for_nonexistent_draft(service: RaidService):
    """Fails with RaidNotFoundError when attempting to approve a non-existent draft ID."""
    with pytest.raises(RaidNotFoundError) as exc_info:
        service.approve_raid("nonexistent-id-999", approved_by="Jane PM")
    assert "not found" in str(exc_info.value).lower()


def test_7_prevent_approval_of_already_approved_draft(service: RaidService, sample_draft: dict):
    """Prevents approving an already approved draft, raising DuplicateApprovalError."""
    service.approve_raid(sample_draft["id"], approved_by="Jane PM")
    with pytest.raises(DuplicateApprovalError):
        service.approve_raid(sample_draft["id"], approved_by="Jane PM")


def test_8_prevent_approval_of_rejected_draft(service: RaidService, sample_draft: dict):
    """Prevents approving a rejected draft without explicit re-submission, raising InvalidRaidStatusError."""
    service.reject_raid(sample_draft["id"], rejected_by="QA Lead", reason="Duplicate issue")
    with pytest.raises(InvalidRaidStatusError) as exc_info:
        service.approve_raid(sample_draft["id"], approved_by="Jane PM")
    assert "rejected" in str(exc_info.value).lower()


def test_9_verify_approved_record_and_audit_record_persisted(service: RaidService, sample_draft: dict):
    """Verifies that both the approved register record and the approval audit entry are saved."""
    service.approve_raid(sample_draft["id"], approved_by="Director Ops", comments="SteerCo signed off")

    approved_items = service.get_approved_items()
    assert len(approved_items) == 1
    assert approved_items[0]["title"] == sample_draft["title"]

    history = service.get_audit_history(sample_draft["id"])
    approval_entries = [h for h in history if h["action"] == "APPROVED"]
    assert len(approval_entries) == 1
    assert approval_entries[0]["action_by"] == "Director Ops"
    assert "SteerCo signed off" in approval_entries[0]["action_notes"]


def test_10_verify_transaction_rollback_when_approval_fails(service: RaidService, sample_draft: dict):
    """Simulates a database failure during approval and confirms transaction rolls back completely."""
    # Force a failure inside execute_approval_transaction by raising an error during insertion
    with patch.object(service.db, "execute_approval_transaction", side_effect=sqlite3.OperationalError("Simulated disk error")):
        with pytest.raises(AuditPersistenceError):
            service.approve_raid(sample_draft["id"], approved_by="Admin")

    # Draft status must remain 'Draft' (NOT 'Accepted')
    draft_after = service.get_raid_by_id(sample_draft["id"])
    assert draft_after["status"] == "Draft"
    # No approved items created
    assert len(service.get_approved_items()) == 0


# ---------------------------------------------------------------------------
# Rejection Tests
# ---------------------------------------------------------------------------

def test_11_reject_valid_draft(service: RaidService, sample_draft: dict):
    """Rejects a valid draft with a stated reason."""
    rejected_record = service.reject_raid(
        raid_id=sample_draft["id"],
        rejected_by="Functional Lead",
        reason="Target system configuration resolved out-of-band."
    )
    assert rejected_record["status"] == "Rejected"
    assert "Target system configuration" in rejected_record["scoring_reason"]


def test_12_reject_empty_rejection_reason(service: RaidService, sample_draft: dict):
    """Rejection requires a non-empty explanation; raises RaidValidationError if omitted."""
    with pytest.raises(RaidValidationError) as exc_info:
        service.reject_raid(sample_draft["id"], rejected_by="Lead", reason="   ")
    assert "reason is required" in str(exc_info.value).lower()


def test_13_verify_rejected_records_remain_available(service: RaidService, sample_draft: dict):
    """Rejected records remain queryable for historical review and are not deleted."""
    service.reject_raid(sample_draft["id"], rejected_by="Lead", reason="Not actionable")
    rejected_list = service.get_rejected_drafts()
    assert len(rejected_list) == 1
    assert rejected_list[0]["id"] == sample_draft["id"]


def test_14_verify_rejected_records_do_not_appear_in_approved_register(service: RaidService, sample_draft: dict):
    """Rejected items must not appear in the approved register or pending review list."""
    service.reject_raid(sample_draft["id"], rejected_by="Lead", reason="Not an issue")
    assert len(service.get_approved_items()) == 0
    assert len(service.get_pending_drafts()) == 0


# ---------------------------------------------------------------------------
# Modification Tests
# ---------------------------------------------------------------------------

def test_15_modify_single_field(service: RaidService, sample_draft: dict):
    """Updates only the requested field (e.g. likely_owner)."""
    updated = service.modify_raid(
        raid_id=sample_draft["id"],
        modified_by="PM Lead",
        updates={"likely_owner": "Security Architect"}
    )
    assert updated["likely_owner"] == "Security Architect"


def test_16_verify_other_fields_remain_unchanged(service: RaidService, sample_draft: dict):
    """Modifying one field preserves all other attributes (title, description, evidence, etc.)."""
    service.modify_raid(
        raid_id=sample_draft["id"],
        modified_by="PM Lead",
        updates={"likely_owner": "New Owner"}
    )
    updated = service.get_raid_by_id(sample_draft["id"])
    assert updated["likely_owner"] == "New Owner"
    assert updated["title"] == sample_draft["title"]
    assert updated["description"] == sample_draft["description"]
    assert updated["evidence_text"] == sample_draft["evidence_text"]
    assert updated["evidence_source_id"] == sample_draft["evidence_source_id"]


def test_17_reject_invalid_field_names(service: RaidService, sample_draft: dict):
    """Attempting to update an unknown or non-existent field raises RaidValidationError."""
    with pytest.raises(RaidValidationError) as exc_info:
        service.modify_raid(
            raid_id=sample_draft["id"],
            modified_by="PM Lead",
            updates={"non_existent_column_xyz": "value"}
        )
    assert "invalid field name" in str(exc_info.value).lower()


def test_18_reject_invalid_field_values(service: RaidService, sample_draft: dict):
    """Attempting to pass invalid field values (empty title, invalid category) raises RaidValidationError."""
    with pytest.raises(RaidValidationError):
        service.modify_raid(sample_draft["id"], modified_by="PM", updates={"title": ""})

    with pytest.raises(RaidValidationError):
        service.modify_raid(sample_draft["id"], modified_by="PM", updates={"type": "INVALID_TYPE"})

    with pytest.raises(RaidValidationError):
        service.modify_raid(sample_draft["id"], modified_by="PM", updates={"confidence_score": 1.5})


def test_19_verify_before_and_after_values_in_audit_history(service: RaidService, sample_draft: dict):
    """Audit entry preserves exact JSON diff of old and new values for modified fields."""
    service.modify_raid(
        raid_id=sample_draft["id"],
        modified_by="Editor",
        updates={"workstream": "Basis"},
        comments="Reassigned workstream to Basis"
    )
    history = service.get_audit_history(sample_draft["id"])
    mod_entries = [h for h in history if h["action"] == "MODIFIED"]
    assert len(mod_entries) == 1

    old_vals = json.loads(mod_entries[0]["proposed_data"])
    new_vals = json.loads(mod_entries[0]["final_data"])
    assert old_vals["workstream"] == "Integration"
    assert new_vals["workstream"] == "Basis"


def test_20_prevent_modification_of_approved_records(service: RaidService, sample_draft: dict):
    """Approved RAID items cannot be modified through the draft modification method."""
    service.approve_raid(sample_draft["id"], approved_by="PM")
    with pytest.raises(InvalidRaidStatusError) as exc_info:
        service.modify_raid(sample_draft["id"], modified_by="PM", updates={"workstream": "Basis"})
    assert "already approved" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Audit Tests
# ---------------------------------------------------------------------------

def test_21_verify_actor_and_timestamp_recorded(service: RaidService, sample_draft: dict):
    """Every action records the actor name and a valid UTC timestamp."""
    service.modify_raid(sample_draft["id"], modified_by="Alice Auditor", updates={"severity": "Critical"})
    history = service.get_audit_history(sample_draft["id"])
    last_action = history[0]  # Ordered DESC by timestamp
    assert last_action["action_by"] == "Alice Auditor"
    assert last_action["timestamp"] is not None


def test_22_verify_audit_records_are_append_only(service: RaidService, sample_draft: dict):
    """Audit records increase monotonically; past records are never deleted or updated."""
    initial_count = len(service.get_audit_history(sample_draft["id"]))
    service.modify_raid(sample_draft["id"], modified_by="User 1", updates={"workstream": "Basis"})
    assert len(service.get_audit_history(sample_draft["id"])) == initial_count + 1

    service.approve_raid(sample_draft["id"], approved_by="User 2")
    assert len(service.get_audit_history(sample_draft["id"])) == initial_count + 2


def test_23_verify_each_business_action_creates_audit_entry(service: RaidService, sample_draft: dict):
    """PROPOSED, MODIFIED, and APPROVED/REJECTED actions are accurately logged."""
    service.modify_raid(sample_draft["id"], modified_by="User A", updates={"impact": "Critical"})
    service.approve_raid(sample_draft["id"], approved_by="User B")

    actions = [h["action"] for h in service.get_audit_history(sample_draft["id"])]
    assert "PROPOSED" in actions
    assert "MODIFIED" in actions
    assert "APPROVED" in actions


# ---------------------------------------------------------------------------
# Data Integrity Tests
# ---------------------------------------------------------------------------

def test_24_verify_repeated_approval_does_not_create_duplicate_approved_records(service: RaidService, sample_draft: dict):
    """Attempting repeated approval fails and leaves exactly one record in approved_raid."""
    service.approve_raid(sample_draft["id"], approved_by="PM 1")
    with pytest.raises(DuplicateApprovalError):
        service.approve_raid(sample_draft["id"], approved_by="PM 2")

    approved_list = service.get_approved_items()
    assert len(approved_list) == 1


def test_25_verify_source_ids_and_evidence_preserved(service: RaidService, sample_draft: dict):
    """Approved item retains immutable traceability back to source signal and quote."""
    approved = service.approve_raid(sample_draft["id"], approved_by="PM")
    assert approved["evidence_links"] == sample_draft["evidence_text"]
    assert approved["draft_id"] == sample_draft["id"]


def test_26_verify_database_failures_do_not_leave_partial_updates(service: RaidService, sample_draft: dict):
    """If modification transaction throws an error, no partial values or orphan audit logs persist."""
    with patch.object(service.db, "execute_modification_transaction", side_effect=sqlite3.OperationalError("Disk failure")):
        with pytest.raises(AuditPersistenceError):
            service.modify_raid(sample_draft["id"], modified_by="PM", updates={"workstream": "Basis"})

    # Verify workstream was NOT modified
    current_draft = service.get_raid_by_id(sample_draft["id"])
    assert current_draft["workstream"] == "Integration"
    # Verify no MODIFIED audit log was written
    actions = [h["action"] for h in service.get_audit_history(sample_draft["id"])]
    assert "MODIFIED" not in actions


def test_27_verify_actual_sqlite_transaction_rollback_on_partial_failure(temp_db: DatabaseManager, sample_draft: dict):
    """
    Directly tests execute_approval_transaction against SQLite:
    If a database error occurs during step 2 or 3 (e.g. NOT NULL violation in approved_raid),
    SQLite rolls back step 1, leaving the draft status untouched.
    """
    invalid_approved_record = {
        "id": "appr-invalid",
        "draft_id": sample_draft["id"],
        "type": None,  # Violates NOT NULL on approved_raid.type
        "title": "Invalid Title",
        "description": "Invalid Desc",
        "workstream": "Integration",
        "phase": "Realize",
        "severity_or_impact": "High",
        "owner": "Basis",
        "due_date": "",
        "mitigation_action": "",
        "status": "Open",
        "evidence_links": "",
        "approved_by": "PM",
        "approved_at": "2026-11-21T00:00:00Z",
        "last_updated_at": "2026-11-21T00:00:00Z"
    }
    audit_record = {
        "item_id": sample_draft["id"],
        "item_table": "raid_drafts",
        "action": "APPROVED",
        "proposed_data": "{}",
        "final_data": "{}",
        "action_by": "PM",
        "action_notes": "Approval",
        "timestamp": "2026-11-21T00:00:00Z"
    }

    # Attempting transaction must fail due to NOT NULL constraint on approved_raid.type
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.execute_approval_transaction(
            draft_id=sample_draft["id"],
            approved_record=invalid_approved_record,
            audit_record=audit_record
        )

    # Confirm step 1 (status update to Accepted) was rolled back by SQLite
    draft_after = temp_db.get_raid_draft_by_id(sample_draft["id"])
    assert draft_after["status"] == "Draft", "Draft status was not rolled back after partial failure"

    # Confirm neither approved_raid nor audit_history contain partial records
    approved = temp_db.get_approved_items()
    assert len(approved) == 0, "approved_raid contains orphan record"
    audit_entries = [h for h in temp_db.get_audit_history(sample_draft["id"]) if h["action"] == "APPROVED"]
    assert len(audit_entries) == 0, "audit_history contains orphan approval entry"


def test_28_simulate_concurrent_approvals(temp_db: DatabaseManager, sample_draft: dict):
    """
    Simulates concurrent approval attempts for the same draft:
    Confirms only one succeeds, only one approved record and audit record exist,
    the other receives DuplicateApprovalError, and draft status remains consistent.
    """
    from concurrent.futures import ThreadPoolExecutor

    service1 = RaidService(temp_db)
    service2 = RaidService(temp_db)

    results = []
    errors = []

    def try_approve(svc, approver_name):
        try:
            res = svc.approve_raid(sample_draft["id"], approved_by=approver_name)
            results.append(res)
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(try_approve, service1, "Approver 1")
        f2 = executor.submit(try_approve, service2, "Approver 2")
        f1.result()
        f2.result()

    # Exactly one approval succeeded and one failed with DuplicateApprovalError
    assert len(results) == 1, f"Expected exactly 1 success, got {len(results)}"
    assert len(errors) == 1, f"Expected exactly 1 failure, got {len(errors)}"
    assert isinstance(errors[0], DuplicateApprovalError), f"Expected DuplicateApprovalError, got {type(errors[0])}: {errors[0]}"

    # Exactly one approved record in approved_raid
    approved_items = service1.get_approved_items()
    assert len(approved_items) == 1

    # Exactly one APPROVED audit entry
    history = service1.get_audit_history(sample_draft["id"])
    approval_audits = [h for h in history if h["action"] == "APPROVED"]
    assert len(approval_audits) == 1

    # Draft status is cleanly 'Accepted'
    draft = service1.get_raid_by_id(sample_draft["id"])
    assert draft["status"] in ("Accepted", "Approved")


# ===========================================================================
# Stage 6C: Low-Confidence Candidate Retrieval Regression Tests
# ===========================================================================

class TestLowConfidenceCandidatesRetrieval:
    """
    Validates the Streamlit UI -> RaidService -> DatabaseManager -> raw_signals
    retrieval path for Stage 6C.
    """

    def test_raid_service_exposes_get_low_confidence_candidates(self, temp_db):
        """1. RaidService exposes get_low_confidence_candidates as a callable method."""
        service = RaidService(temp_db)
        assert hasattr(service, "get_low_confidence_candidates")
        assert callable(service.get_low_confidence_candidates)

    def test_correct_retrieval_of_below_confidence_threshold_signals(self, temp_db):
        """2. Correct retrieval and JSON parsing of BELOW_CONFIDENCE_THRESHOLD signals."""
        service = RaidService(temp_db)
        extraction_payload = {
            "category": "Risk",
            "title": "Vendor SLA delay",
            "description": "Possible SLA contract breach.",
            "confidence": 0.65,
            "owner": "Sarah Procurement",
            "due_date": "2026-12-01",
            "evidence": "Vendor negotiations are behind schedule.",
            "reasoning_summary": "Low confidence due to missing baseline date.",
            "probability": "Medium",
            "impact": "High",
            "evidence_status": "verified"
        }
        sig = ProjectSignal(
            id="sig-below-01",
            source_type=SourceType.EMAIL,
            source_id="msg-101",
            author="vendor_mgr@enterprise.com",
            subject="Vendor Negotiation Status",
            content="Vendor negotiations are behind schedule.",
            workstream="SD",
            phase="Realize"
        )
        temp_db.save_raw_signal(sig, user_id="user_alpha")
        temp_db.update_signal_classification_status(
            signal_id="sig-below-01",
            status="BELOW_CONFIDENCE_THRESHOLD",
            failure_reason=json.dumps(extraction_payload),
            user_id="user_alpha",
            llm_provider="OpenRouter",
            llm_model="nvidia/nemotron-3-ultra-550b-a55b:free"
        )

        candidates = service.get_low_confidence_candidates(user_id="user_alpha")
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand["signal_id"] == "sig-below-01"
        assert cand["source_id"] == "msg-101"
        assert cand["source_type"] == "email"
        assert cand["author"] == "vendor_mgr@enterprise.com"
        assert cand["subject"] == "Vendor Negotiation Status"
        assert cand["predicted_category"] == "Risk"
        assert cand["confidence_score"] == pytest.approx(0.65)
        assert cand["likely_owner"] == "Sarah Procurement"
        assert cand["due_date"] == "2026-12-01"
        assert cand["workstream"] == "SD"
        assert cand["phase"] == "Realize"
        assert cand["llm_provider"] == "OpenRouter"
        assert cand["llm_model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert cand["status"] == "BELOW_CONFIDENCE_THRESHOLD"

    def test_exclusion_of_completed_and_unrelated_signals(self, temp_db):
        """3. Exclusion of completed, not_raid, failed, and already-drafted signals."""
        service = RaidService(temp_db)

        # Signal 1: COMPLETED (high confidence draft created)
        sig1 = ProjectSignal(id="sig-comp-01", source_type=SourceType.EMAIL, author="a@b.com", content="High risk 1")
        temp_db.save_raw_signal(sig1, user_id="user_filter")
        temp_db.update_signal_classification_status("sig-comp-01", status="COMPLETED", user_id="user_filter")

        # Signal 2: NOT_RAID (filtered noise)
        sig2 = ProjectSignal(id="sig-not-02", source_type=SourceType.CHAT, author="b@c.com", content="Chat noise")
        temp_db.save_raw_signal(sig2, user_id="user_filter")
        temp_db.update_signal_classification_status("sig-not-02", status="NOT_RAID", user_id="user_filter")

        # Signal 3: FAILED
        sig3 = ProjectSignal(id="sig-fail-03", source_type=SourceType.EMAIL, author="c@d.com", content="Corrupt payload")
        temp_db.save_raw_signal(sig3, user_id="user_filter")
        temp_db.update_signal_classification_status("sig-fail-03", status="FAILED", user_id="user_filter")

        # Signal 4: BELOW_CONFIDENCE_THRESHOLD but already has an existing draft in raid_drafts
        sig4 = ProjectSignal(id="sig-drafted-04", source_type=SourceType.EMAIL, author="d@e.com", content="Already confirmed")
        temp_db.save_raw_signal(sig4, user_id="user_filter")
        temp_db.update_signal_classification_status("sig-drafted-04", status="BELOW_CONFIDENCE_THRESHOLD", user_id="user_filter")
        temp_db.save_raid_draft({
            "id": "raid-existing-04",
            "user_id": "user_filter",
            "type": "Risk",
            "title": "Already drafted risk",
            "description": "Desc",
            "evidence_source_id": "sig-drafted-04",
            "status": "Draft"
        }, user_id="user_filter")

        # Signal 5: Genuine BELOW_CONFIDENCE_THRESHOLD awaiting review
        sig5 = ProjectSignal(id="sig-genuine-05", source_type=SourceType.EMAIL, author="e@f.com", content="Genuine candidate")
        temp_db.save_raw_signal(sig5, user_id="user_filter")
        temp_db.update_signal_classification_status("sig-genuine-05", status="BELOW_CONFIDENCE_THRESHOLD", user_id="user_filter")

        candidates = service.get_low_confidence_candidates(user_id="user_filter")
        assert len(candidates) == 1
        assert candidates[0]["signal_id"] == "sig-genuine-05"

    def test_empty_results(self, temp_db):
        """4. Empty results returned safely without exceptions."""
        service = RaidService(temp_db)
        candidates = service.get_low_confidence_candidates(user_id="empty_tenant")
        assert candidates == []
        assert isinstance(candidates, list)

    def test_tenant_isolation(self, temp_db):
        """5. Strict tenant isolation between users."""
        service = RaidService(temp_db)

        # Insert signal for Tenant 1
        sig1 = ProjectSignal(id="sig-t1", source_type=SourceType.EMAIL, author="t1@corp.com", content="Tenant 1 confidential")
        temp_db.save_raw_signal(sig1, user_id="tenant_1")
        temp_db.update_signal_classification_status("sig-t1", status="BELOW_CONFIDENCE_THRESHOLD", user_id="tenant_1")

        # Insert signal for Tenant 2
        sig2 = ProjectSignal(id="sig-t2", source_type=SourceType.EMAIL, author="t2@corp.com", content="Tenant 2 confidential")
        temp_db.save_raw_signal(sig2, user_id="tenant_2")
        temp_db.update_signal_classification_status("sig-t2", status="BELOW_CONFIDENCE_THRESHOLD", user_id="tenant_2")

        # Tenant 1 sees only Tenant 1
        t1_cands = service.get_low_confidence_candidates(user_id="tenant_1")
        assert len(t1_cands) == 1
        assert t1_cands[0]["signal_id"] == "sig-t1"

        # Tenant 2 sees only Tenant 2
        t2_cands = service.get_low_confidence_candidates(user_id="tenant_2")
        assert len(t2_cands) == 1
        assert t2_cands[0]["signal_id"] == "sig-t2"

        # Tenant 3 sees 0
        assert service.get_low_confidence_candidates(user_id="tenant_3") == []

    def test_required_ui_fields(self, temp_db):
        """6. All fields required by the Streamlit UI are present in each candidate."""
        service = RaidService(temp_db)
        sig = ProjectSignal(
            id="sig-ui-req-01",
            source_type=SourceType.CHAT,
            source_id="chat-888",
            author="lead@dev.com",
            subject="API timeout",
            content="API timeout observed during batch run."
        )
        temp_db.save_raw_signal(sig, user_id="user_ui")
        temp_db.update_signal_classification_status(
            "sig-ui-req-01",
            status="BELOW_CONFIDENCE_THRESHOLD",
            failure_reason="Confidence 0.58 below threshold",
            user_id="user_ui",
            llm_provider="openrouter",
            llm_model="nvidia/nemotron-3-ultra-550b-a55b:free"
        )

        candidates = service.get_low_confidence_candidates(user_id="user_ui")
        assert len(candidates) == 1
        cand = candidates[0]

        required_keys = [
            "signal_id", "source_type", "source_id", "author", "timestamp",
            "subject", "content", "workstream", "phase", "predicted_category",
            "confidence_score", "title", "description", "evidence_text",
            "evidence_status", "reasoning_summary", "likely_owner", "due_date",
            "probability", "impact", "severity", "mitigation_action",
            "llm_provider", "llm_model", "status"
        ]
        for key in required_keys:
            assert key in cand, f"Missing required UI field: {key}"

        # Verify fallback regex parsed 0.58 from 'Confidence 0.58 below threshold'
        assert cand["confidence_score"] == pytest.approx(0.58)

    def test_no_database_mutations_during_retrieval(self, temp_db):
        """7. Calling get_low_confidence_candidates does NOT mutate any database tables."""
        service = RaidService(temp_db)
        sig = ProjectSignal(id="sig-immut-01", source_type=SourceType.EMAIL, author="auditor@sec.com", content="Audit signal")
        temp_db.save_raw_signal(sig, user_id="user_audit")
        temp_db.update_signal_classification_status("sig-immut-01", status="BELOW_CONFIDENCE_THRESHOLD", user_id="user_audit")

        # Snapshot table counts before
        with temp_db.get_connection() as conn:
            c = conn.cursor()
            sig_before = c.execute("SELECT COUNT(*) FROM raw_signals").fetchone()[0]
            draft_before = c.execute("SELECT COUNT(*) FROM raid_drafts").fetchone()[0]
            appr_before = c.execute("SELECT COUNT(*) FROM approved_raid").fetchone()[0]
            audit_before = c.execute("SELECT COUNT(*) FROM audit_history").fetchone()[0]

        # Call retrieval 3 times
        for _ in range(3):
            cands = service.get_low_confidence_candidates(user_id="user_audit")
            assert len(cands) == 1

        # Snapshot table counts after
        with temp_db.get_connection() as conn:
            c = conn.cursor()
            sig_after = c.execute("SELECT COUNT(*) FROM raw_signals").fetchone()[0]
            draft_after = c.execute("SELECT COUNT(*) FROM raid_drafts").fetchone()[0]
            appr_after = c.execute("SELECT COUNT(*) FROM approved_raid").fetchone()[0]
            audit_after = c.execute("SELECT COUNT(*) FROM audit_history").fetchone()[0]

        assert sig_after == sig_before
        assert draft_after == draft_before
        assert appr_after == appr_before
        assert audit_after == audit_before

