"""
Tests for Phase 2: Database Migrations and Tenant Isolation.

Verifies:
1. User A cannot read User B's raw signals.
2. User A cannot read User B's RAID drafts.
3. User A cannot read User B's approved RAID records.
4. User A cannot read User B's predictions.
5. User A cannot access User B's audit history.
6. User A cannot update or delete User B's records.
7. User A cannot approve or reject User B's draft by guessing ID (IDOR prevention).
8. Semantic deduplication does not compare records across users.
9. Predictive engine is isolated by user.
10. Legacy data (default_user) is preserved and inaccessible to normal users.
11. Migration preserves existing data, resolves duplicates, creates backup and indexes.
12. Re-running migrations is idempotent and safe.
"""

import pytest
import sqlite3
import tempfile
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

from src.database.db_manager import DatabaseManager
from src.database.migrations import run_migrations, MIGRATION_VERSION
from src.service import RaidService, RaidNotFoundError, DuplicateApprovalError
from src.pipeline import RAIDPipeline
from src.normalisation.schemas import ProjectSignal, SourceType
from src.predictive_engine.schemas import PredictionItem, PredictionType


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated DatabaseManager backed by a temporary file."""
    db_file = tmp_path / "tenant_test.db"
    db = DatabaseManager(db_path=db_file)
    return db


@pytest.fixture
def raid_service(temp_db):
    """Provides a RaidService instance bound to the isolated temporary DB."""
    return RaidService(db_manager=temp_db)


# ---------------------------------------------------------------------------
# Scenario 1: Raw Signals Isolation
# ---------------------------------------------------------------------------

def test_user_cannot_read_other_user_raw_signals(temp_db):
    sig_a = {
        "id": "sig-a-1",
        "user_id": "user_A",
        "source_type": "email",
        "source_id": "msg-101",
        "author": "alice@corp.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": "Confidential Budget user_A",
        "content": "Alice's proprietary budget email content",
        "workstream": "Finance",
        "phase": "Realize",
        "is_noise": 0,
        "noise_reason": None
    }
    sig_b = {
        "id": "sig-b-1",
        "user_id": "user_B",
        "source_type": "email",
        "source_id": "msg-202",
        "author": "bob@corp.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": "Secret Architecture user_B",
        "content": "Bob's confidential infrastructure design",
        "workstream": "Architecture",
        "phase": "Explore",
        "is_noise": 0,
        "noise_reason": None
    }

    temp_db.save_raw_signal(sig_a, user_id="user_A")
    temp_db.save_raw_signal(sig_b, user_id="user_B")

    # User A cannot read User B's raw signal
    assert temp_db.get_raw_signal_by_id("sig-b-1", user_id="user_A") is None
    # User B can read their own
    assert temp_db.get_raw_signal_by_id("sig-b-1", user_id="user_B") is not None
    # User A cannot detect User B's source_id as existing
    assert not temp_db.signal_exists("email", "msg-202", user_id="user_A")
    # User A can ingest the SAME source_id without conflict because of composite unique key (user_id, source_type, source_id)
    sig_a_same_id = dict(sig_a)
    sig_a_same_id["id"] = "sig-a-2"
    sig_a_same_id["source_id"] = "msg-202"
    assert temp_db.save_raw_signal(sig_a_same_id, user_id="user_A") is True


# ---------------------------------------------------------------------------
# Scenario 2: RAID Drafts Isolation
# ---------------------------------------------------------------------------

def test_user_cannot_read_other_user_raid_drafts(temp_db):
    draft_a = {
        "id": "draft-a-01",
        "type": "Risk",
        "title": "Vendor SLA Risk for User A",
        "description": "Risk details for tenant A",
        "workstream": "Procurement",
        "phase": "Realize",
        "confidence_score": 0.88,
        "evidence_text": "Clause 4.1 in contract",
        "evidence_source_id": "sig-a-1",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "user_A"
    }
    draft_b = {
        "id": "draft-b-01",
        "type": "Issue",
        "title": "Database Outage Issue for User B",
        "description": "Issue details for tenant B",
        "workstream": "Technology",
        "phase": "Deploy",
        "confidence_score": 0.95,
        "evidence_text": "Log error 500",
        "evidence_source_id": "sig-b-1",
        "evidence_source_type": "defect",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "user_B"
    }

    temp_db.save_raid_draft(draft_a, user_id="user_A")
    temp_db.save_raid_draft(draft_b, user_id="user_B")

    # Pending drafts scoped by user
    drafts_for_a = temp_db.get_pending_drafts(user_id="user_A")
    assert len(drafts_for_a) == 1
    assert drafts_for_a[0]["id"] == "draft-a-01"

    drafts_for_b = temp_db.get_pending_drafts(user_id="user_B")
    assert len(drafts_for_b) == 1
    assert drafts_for_b[0]["id"] == "draft-b-01"

    # Direct ID lookup isolation
    assert temp_db.get_raid_draft_by_id("draft-b-01", user_id="user_A") is None
    assert temp_db.get_raid_draft_by_id("draft-a-01", user_id="user_B") is None

    # All drafts scoped by user
    all_a = temp_db.get_all_drafts(user_id="user_A")
    assert len(all_a) == 1
    assert all_a[0]["id"] == "draft-a-01"


# ---------------------------------------------------------------------------
# Scenario 3: Approved RAID Records Isolation
# ---------------------------------------------------------------------------

def test_user_cannot_read_other_user_approved_raid(raid_service, temp_db):
    draft_b = {
        "id": "draft-b-approve",
        "type": "Risk",
        "title": "Critical Infrastructure Risk for User B",
        "description": "Details about cloud failover",
        "confidence_score": 0.92,
        "evidence_text": "Disaster recovery audit note",
        "evidence_source_id": "sig-b-1",
        "evidence_source_type": "meeting_notes",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "user_B"
    }
    temp_db.save_raid_draft(draft_b, user_id="user_B")

    # User B approves their draft
    approved_b = raid_service.approve_raid(
        raid_id="draft-b-approve",
        approved_by="Bob Manager",
        user_id="user_B"
    )
    assert approved_b["status"] == "Open"
    assert approved_b["user_id"] == "user_B"

    # User A queries approved items
    user_a_approved = raid_service.get_approved_items(user_id="user_A")
    assert len(user_a_approved) == 0

    # User A attempts direct lookup of approved item ID
    assert temp_db.get_approved_item_by_id(approved_b["id"], user_id="user_A") is None
    assert temp_db.get_approved_item_by_draft_id("draft-b-approve", user_id="user_A") is None

    # User B sees their approved item
    user_b_approved = raid_service.get_approved_items(user_id="user_B")
    assert len(user_b_approved) == 1
    assert user_b_approved[0]["id"] == approved_b["id"]


# ---------------------------------------------------------------------------
# Scenario 4: Predictions Isolation
# ---------------------------------------------------------------------------

def test_user_cannot_read_other_user_predictions(temp_db):
    pred_a = {
        "prediction_id": "pred-a-01",
        "user_id": "user_A",
        "prediction_type": "TASK_OVERDUE",
        "title": "Task 12 Overdue for User A",
        "description": "Migration task 12 is overdue",
        "suggested_raid_category": "Issue",
        "confidence": 0.85,
        "severity": "High",
        "evidence": "Schedule baseline exceeded",
        "source_signal_ids": ["sig-a-1"],
        "related_raid_ids": [],
        "workstream": "Migration",
        "phase": "Realize",
        "status": "DETECTED"
    }
    pred_b = {
        "prediction_id": "pred-b-01",
        "user_id": "user_B",
        "prediction_type": "DEFECT_CLUSTER",
        "title": "Defect Spike on Module X for User B",
        "description": "5 defects reported in 2 days",
        "suggested_raid_category": "Risk",
        "confidence": 0.90,
        "severity": "Critical",
        "evidence": "Defect log clustering",
        "source_signal_ids": ["sig-b-1"],
        "related_raid_ids": [],
        "workstream": "QA",
        "phase": "Testing",
        "status": "DETECTED"
    }

    temp_db.save_prediction(pred_a, user_id="user_A")
    temp_db.save_prediction(pred_b, user_id="user_B")

    preds_a = temp_db.get_predictions(user_id="user_A")
    assert len(preds_a) == 1
    assert preds_a[0]["id"] == "pred-a-01"

    preds_b = temp_db.get_predictions(user_id="user_B")
    assert len(preds_b) == 1
    assert preds_b[0]["id"] == "pred-b-01"


# ---------------------------------------------------------------------------
# Scenario 5: Audit History Isolation
# ---------------------------------------------------------------------------

def test_user_cannot_access_other_user_audit_history(temp_db):
    audit_a = {
        "user_id": "user_A",
        "item_id": "item-a-1",
        "item_table": "raid_drafts",
        "action": "PROPOSED",
        "proposed_data": json.dumps({"title": "A proposal"}),
        "action_by": "Agent A",
        "action_notes": "Proposed item for user A",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    audit_b = {
        "user_id": "user_B",
        "item_id": "item-b-1",
        "item_table": "raid_drafts",
        "action": "PROPOSED",
        "proposed_data": json.dumps({"title": "B proposal"}),
        "action_by": "Agent B",
        "action_notes": "Proposed item for user B",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    temp_db.insert_audit_entry(audit_a, user_id="user_A")
    temp_db.insert_audit_entry(audit_b, user_id="user_B")

    trail_a = temp_db.get_audit_trail(user_id="user_A")
    assert len(trail_a) == 1
    assert trail_a[0]["item_id"] == "item-a-1"

    trail_b = temp_db.get_audit_trail(user_id="user_B")
    assert len(trail_b) == 1
    assert trail_b[0]["item_id"] == "item-b-1"

    # Scoped by item_id AND user_id: querying item-b-1 as user_A yields nothing
    assert len(temp_db.get_audit_trail(item_id="item-b-1", user_id="user_A")) == 0


# ---------------------------------------------------------------------------
# Scenario 6: Record Modification / Update Isolation
# ---------------------------------------------------------------------------

def test_user_cannot_update_other_user_records(raid_service, temp_db):
    draft_b = {
        "id": "draft-b-mod",
        "type": "Risk",
        "title": "Original Title by User B",
        "description": "Original description",
        "confidence_score": 0.80,
        "evidence_text": "Original evidence",
        "evidence_source_id": "sig-b-1",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "user_B"
    }
    temp_db.save_raid_draft(draft_b, user_id="user_B")

    # User A tries to modify User B's draft
    with pytest.raises(RaidNotFoundError):
        raid_service.modify_raid(
            raid_id="draft-b-mod",
            modified_by="Malicious User A",
            updates={"title": "Tampered Title by User A"},
            user_id="user_A"
        )

    # Verify User B's record remains unmodified
    b_record = temp_db.get_raid_draft_by_id("draft-b-mod", user_id="user_B")
    assert b_record["title"] == "Original Title by User B"


# ---------------------------------------------------------------------------
# Scenario 7: IDOR Protection on Approval and Rejection
# ---------------------------------------------------------------------------

def test_user_cannot_approve_or_reject_other_user_draft_idor(raid_service, temp_db):
    draft_b = {
        "id": "draft-b-idor",
        "type": "Issue",
        "title": "Target Draft for IDOR Attack",
        "description": "Sensitive internal operational issue",
        "confidence_score": 0.90,
        "evidence_text": "Sensitive quote",
        "evidence_source_id": "sig-b-1",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "user_B"
    }
    temp_db.save_raid_draft(draft_b, user_id="user_B")

    # 1. User A guesses draft_id and tries to approve
    with pytest.raises(RaidNotFoundError):
        raid_service.approve_raid(
            raid_id="draft-b-idor",
            approved_by="Attacker A",
            user_id="user_A"
        )

    # 2. User A guesses draft_id and tries to reject
    with pytest.raises(RaidNotFoundError):
        raid_service.reject_raid(
            raid_id="draft-b-idor",
            rejected_by="Attacker A",
            reason="Malicious rejection attempt",
            user_id="user_A"
        )

    # Ensure draft is untouched and still pending for User B
    b_draft = temp_db.get_raid_draft_by_id("draft-b-idor", user_id="user_B")
    assert b_draft["status"] == "Draft"
    assert b_draft["scoring_reason"] != "Rejected: Malicious rejection attempt"


# ---------------------------------------------------------------------------
# Scenario 8: Semantic Deduplication Isolated by User
# ---------------------------------------------------------------------------

def test_semantic_deduplication_isolation_across_users(temp_db):
    from src.classification.classifier import RAIDClassifier
    from src.classification.providers.mock_provider import MockLLMProvider

    # Inject mock LLM provider so classification runs deterministically offline
    pipeline = RAIDPipeline(db_manager=temp_db)
    pipeline.classifier = RAIDClassifier(provider=MockLLMProvider())

    # User B ingests an email about a potential schedule risk
    raw_signals_b = [{
        "id": "sig-b-net-1",
        "source_type": "email",
        "source_id": "email-net-001",
        "author": "bob@domain.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": "Risk: Client may not provide master data on time",
        "content": "The client may not provide the required master data before the integration testing phase. This carries risk of delay.",
        "workstream": "SD",
        "phase": "Realize"
    }]
    res_b = pipeline.process_signals(raw_signals_b, user_id="user_B")
    assert res_b["classified_drafts"] >= 1
    assert res_b["duplicates_detected"] == 0

    # User A now ingests an email with identical or highly similar phrasing
    raw_signals_a1 = [{
        "id": "sig-a-net-1",
        "source_type": "email",
        "source_id": "email-net-002",
        "author": "alice@domain.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": "Risk: Client may not provide master data on time",
        "content": "The client may not provide the required master data before the integration testing phase. This carries risk of delay.",
        "workstream": "SD",
        "phase": "Realize"
    }]
    res_a1 = pipeline.process_signals(raw_signals_a1, user_id="user_A")
    # Must NOT be marked as duplicate of User B's item!
    assert res_a1["duplicates_detected"] == 0
    draft_a = res_a1["drafts"][0]
    assert draft_a["duplicate_of_id"] is None

    # But if User A ingests another matching email, it MUST be deduplicated against User A's own draft
    raw_signals_a2 = [{
        "id": "sig-a-net-2",
        "source_type": "email",
        "source_id": "email-net-003",
        "author": "alice@domain.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": "Update: Client may not provide master data on time",
        "content": "The client may not provide the required master data before the integration testing phase. High risk of delay.",
        "workstream": "SD",
        "phase": "Realize"
    }]
    res_a2 = pipeline.process_signals(raw_signals_a2, user_id="user_A")
    assert res_a2["duplicates_detected"] == 1
    assert res_a2["drafts"][0]["duplicate_of_id"] == draft_a["id"]


# ---------------------------------------------------------------------------
# Scenario 9: Predictive Engine Isolation Across Users
# ---------------------------------------------------------------------------

def test_predictive_engine_isolation_across_users(temp_db):
    pipeline = RAIDPipeline(db_manager=temp_db)

    # User B registers an assumption
    assumption_b = {
        "id": "assump-b-1",
        "type": "Assumption",
        "title": "Assuming 10Gbps fiber link availability",
        "description": "Fiber connection will be established before UAT start date.",
        "confidence_score": 0.85,
        "evidence_text": "Assumption doc v1",
        "evidence_source_id": "sig-b-0",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "status": "Accepted",
        "user_id": "user_B"
    }
    temp_db.save_raid_draft(assumption_b, user_id="user_B")

    # User A ingests a project signal saying fiber installation has been cancelled
    contradicting_signal_a = [{
        "id": "sig-a-fiber-cancel",
        "source_type": "email",
        "source_id": "fiber-canc-01",
        "author": "carrier@telecom.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": "Fiber installation cancelled by city permit",
        "content": "City permits rejected fiber trenching. Fiber link connection cancelled.",
        "workstream": "Infrastructure",
        "phase": "Realize"
    }]

    pipeline.process_signals(contradicting_signal_a, user_id="user_A")

    # Foresight for User A must NOT trigger an ASSUMPTION_CONTRADICTED based on User B's assumption
    preds_a = temp_db.get_predictions(user_id="user_A")
    contradicted_preds = [p for p in preds_a if p["prediction_type"] == "ASSUMPTION_CONTRADICTED"]
    assert len(contradicted_preds) == 0


# ---------------------------------------------------------------------------
# Scenario 10: Legacy Data Isolation & default_user Integrity
# ---------------------------------------------------------------------------

def test_legacy_data_preserved_under_default_user_and_inaccessible(temp_db):
    # Insert legacy prototype item under default_user
    legacy_draft = {
        "id": "legacy-draft-001",
        "type": "Risk",
        "title": "Legacy Prototype Risk",
        "description": "Pre-migration prototype data",
        "confidence_score": 0.90,
        "evidence_text": "Prototype seed data",
        "evidence_source_id": "legacy-sig-01",
        "evidence_source_type": "meeting_notes",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "default_user"
    }
    temp_db.save_raid_draft(legacy_draft, user_id="default_user")

    # Newly registered or authenticated user
    new_user_id = "user_new_456"

    # New user sees zero pending drafts, zero approved items, zero KPIs
    assert len(temp_db.get_pending_drafts(user_id=new_user_id)) == 0
    assert len(temp_db.get_approved_items(user_id=new_user_id)) == 0
    assert temp_db.get_raid_draft_by_id("legacy-draft-001", user_id=new_user_id) is None

    kpis_new = temp_db.get_kpis(user_id=new_user_id)
    assert kpis_new["total_drafts"] == 0
    assert kpis_new["pending_reviews"] == 0

    # default_user data remains fully intact in the database
    kpis_default = temp_db.get_kpis(user_id="default_user")
    assert kpis_default["total_drafts"] >= 1

    # Verify default_user exists in users table and has locked credentials
    default_user_record = temp_db.get_user_by_id("default_user")
    assert default_user_record is not None
    assert default_user_record["password_hash"] == "LOCKED_SYSTEM_ACCOUNT_DO_NOT_LOGIN"


# ---------------------------------------------------------------------------
# Scenario 11: Migration Preserves Existing Data, Resolves Duplicates, Backs Up
# ---------------------------------------------------------------------------

def test_migration_preserves_data_and_resolves_duplicates(tmp_path):
    legacy_db_file = tmp_path / "legacy_v1.db"
    conn = sqlite3.connect(str(legacy_db_file))
    cursor = conn.cursor()

    # Create unmigrated tables without user_id column
    cursor.execute("""
        CREATE TABLE raw_signals (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT,
            author TEXT,
            timestamp DATETIME NOT NULL,
            subject TEXT,
            content TEXT NOT NULL,
            workstream TEXT,
            phase TEXT,
            is_noise INTEGER DEFAULT 0,
            noise_reason TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE raid_drafts (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            workstream TEXT,
            phase TEXT,
            probability TEXT,
            impact TEXT,
            severity TEXT,
            scoring_reason TEXT,
            likely_owner TEXT,
            due_date TEXT,
            mitigation_action TEXT,
            confidence_score REAL NOT NULL,
            evidence_text TEXT NOT NULL,
            evidence_source_id TEXT NOT NULL,
            evidence_source_type TEXT NOT NULL,
            detection_source TEXT NOT NULL,
            duplicate_of_id TEXT,
            status TEXT DEFAULT 'Draft'
        );
    """)
    cursor.execute("""
        CREATE TABLE approved_raid (
            id TEXT PRIMARY KEY,
            draft_id TEXT,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            workstream TEXT,
            phase TEXT,
            severity_or_impact TEXT,
            owner TEXT,
            due_date TEXT,
            mitigation_action TEXT,
            status TEXT DEFAULT 'Open',
            evidence_links TEXT,
            approved_by TEXT NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE predictions (
            id TEXT PRIMARY KEY,
            prediction_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            suggested_raid_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT,
            evidence TEXT NOT NULL,
            source_signal_ids TEXT NOT NULL,
            related_raid_ids TEXT,
            workstream TEXT,
            phase TEXT,
            status TEXT DEFAULT 'DETECTED',
            metadata TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE deduplication_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            new_raid_id TEXT,
            matched_raid_id TEXT,
            similarity_score REAL NOT NULL,
            match_status TEXT NOT NULL,
            match_reason TEXT NOT NULL,
            same_category INTEGER NOT NULL DEFAULT 0,
            same_workstream INTEGER NOT NULL DEFAULT 0,
            requires_human_review INTEGER NOT NULL DEFAULT 0
        );
    """)
    cursor.execute("""
        CREATE TABLE audit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL,
            item_table TEXT NOT NULL,
            action TEXT NOT NULL,
            proposed_data TEXT,
            final_data TEXT,
            action_by TEXT NOT NULL,
            action_notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Populate with legacy data, including duplicate source_id to test pre-check resolution
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO raw_signals VALUES ('sig-1', 'email', 'dup-msg-1', 'alice', ?, 'Subject 1', 'Content 1', 'General', 'Realize', 0, NULL)
    """, (now,))
    cursor.execute("""
        INSERT INTO raw_signals VALUES ('sig-2', 'email', 'dup-msg-1', 'alice', ?, 'Subject 2', 'Content 2', 'General', 'Realize', 0, NULL)
    """, (now,))
    cursor.execute("""
        INSERT INTO raid_drafts VALUES ('d-1', 'Risk', 'Title 1', 'Desc 1', 'General', 'Realize', 'High', 'High', NULL, 'Reason', 'PM', '2026-12-01', 'Mitigate', 0.85, 'Quote', 'sig-1', 'email', 'llm', NULL, 'Draft')
    """)
    conn.commit()
    conn.close()

    # Run migrations
    applied = run_migrations(legacy_db_file)
    assert applied is True

    # 1. Verify backup was created
    backup_path = legacy_db_file.with_suffix(".db.bak")
    assert backup_path.exists()
    assert backup_path.stat().st_size > 0

    # 2. Inspect migrated database
    conn = sqlite3.connect(str(legacy_db_file))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check schema_migrations
    cursor.execute("SELECT * FROM schema_migrations WHERE version = ?", (MIGRATION_VERSION,))
    row = cursor.fetchone()
    assert row is not None

    # Check that user_id column exists and all legacy data assigned to default_user
    cursor.execute("SELECT user_id FROM raid_drafts WHERE id = 'd-1'")
    assert cursor.fetchone()["user_id"] == "default_user"

    # Check duplicate raw signals resolution (second duplicate has -dup- appended)
    cursor.execute("SELECT id, source_id, user_id FROM raw_signals ORDER BY id ASC")
    signals = cursor.fetchall()
    assert len(signals) == 2
    assert signals[0]["source_id"] == "dup-msg-1"
    assert signals[0]["user_id"] == "default_user"
    assert "dup-msg-1-dup-" in signals[1]["source_id"]
    assert signals[1]["user_id"] == "default_user"

    # Verify unique index uq_raw_signals_user_source exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'uq_raw_signals_user_source'")
    assert cursor.fetchone() is not None

    conn.close()


# ---------------------------------------------------------------------------
# Scenario 12: Migration Idempotence
# ---------------------------------------------------------------------------

def test_migration_is_idempotent_and_safe_to_rerun(temp_db):
    # temp_db has already run migrations during DatabaseManager.__init__
    re_applied = run_migrations(temp_db.db_path)
    assert re_applied is False, "Re-running migration on an up-to-date DB should return False without error"

    # Verify database can still perform operations cleanly
    draft = {
        "id": "draft-idempotent",
        "type": "Risk",
        "title": "Idempotent Test",
        "description": "Validating DB stability after migration rerun",
        "confidence_score": 0.90,
        "evidence_text": "Evidence",
        "evidence_source_id": "sig-idem",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "status": "Draft",
        "user_id": "user_C"
    }
    temp_db.save_raid_draft(draft, user_id="user_C")
    retrieved = temp_db.get_raid_draft_by_id("draft-idempotent", user_id="user_C")
    assert retrieved is not None
    assert retrieved["title"] == "Idempotent Test"
