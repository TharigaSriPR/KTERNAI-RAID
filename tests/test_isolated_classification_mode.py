"""
Unit and integration tests for Stage 2: Isolated Classification Testing Mode.
Verifies that run_predictive_engine=False safely isolates RAID classification
from rule-based foresight predictions without altering normal classification,
evidence verification, draft creation, deduplication, or tenant isolation.
"""

import pytest
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from src.database.db_manager import DatabaseManager
from src.database.migrations import run_migrations
from src.normalisation.schemas import ProjectSignal, SourceType
from src.pipeline import RAIDPipeline
from src.classification.providers.base_provider import BaseLLMProvider
from src.classification.schemas import RAIDExtractionResult, RAIDCategory, EvidenceStatus


class StubClassificationProvider(BaseLLMProvider):
    """Deterministic mock provider for testing classification mode without live network calls."""
    def __init__(self, category: str = "Risk", confidence: float = 0.95):
        super().__init__("stub-model-v1")
        self._category = category
        self._confidence = confidence

    @property
    def provider_name(self) -> str:
        return "stub_provider"

    def is_configured(self) -> bool:
        return True

    def generate_text(self, prompt: str, system_instruction: str = None) -> str:
        return f"""{{
            "is_raid": true,
            "category": "{self._category}",
            "title": "Vendor Interface Delivery Delay",
            "description": "Third-party vendor API specifications are delayed by two weeks.",
            "workstream": "Integration",
            "phase": "Realize",
            "probability": "High",
            "impact": "High",
            "severity": "High",
            "owner": "Integration Lead",
            "due_date": "2026-12-01",
            "mitigation_action": "Escalate to vendor executive team",
            "confidence": {self._confidence},
            "evidence": "Third-party vendor API specifications are delayed by two weeks"
        }}"""


@pytest.fixture
def isolated_db(tmp_path):
    """Provides a fresh isolated SQLite database with full migrations applied."""
    db_file = tmp_path / "test_isolated.db"
    db = DatabaseManager(db_file)
    run_migrations(db_file)
    return db


@pytest.fixture
def sample_overdue_signal():
    """Signal representing an overdue task that triggers predictive schedule slippage."""
    past_date = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    return ProjectSignal(
        id="sig-task-overdue-1",
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-999",
        author="PMO",
        timestamp=datetime.now(timezone.utc).isoformat(),
        subject="Task: Bank Interface Configuration",
        content=f"Task: Bank Interface Configuration | Baseline Finish: {past_date} | Status: In Progress | Progress: 40%",
        workstream="Finance",
        phase="Realize",
        raw_meta={
            "task_id": "TASK-999",
            "task_name": "Bank Interface Configuration",
            "finish_date": past_date,
            "status": "In Progress",
            "percent_complete": 40
        }
    )


@pytest.fixture
def sample_raid_signal():
    """Signal representing a genuine project risk."""
    return ProjectSignal(
        id="sig-raid-risk-1",
        source_type=SourceType.EMAIL,
        source_id="EML-101",
        author="lead@corp.com",
        timestamp=datetime.now(timezone.utc).isoformat(),
        subject="Vendor API Delivery Delay",
        content="Third-party vendor API specifications are delayed by two weeks, risking SIT cutover.",
        workstream="Integration",
        phase="Realize"
    )


def test_default_pipeline_runs_predictive_foresight(isolated_db, sample_overdue_signal):
    """Confirms that default pipeline execution (run_predictive_engine=True) runs foresight."""
    pipeline = RAIDPipeline(db_manager=isolated_db)
    pipeline.classifier.provider = StubClassificationProvider(category="Not RAID", confidence=0.0)

    # Ingest overdue task signal with default parameter
    res = pipeline.process_signals([sample_overdue_signal], user_id="user_default")

    # Predictive drafts should be generated
    assert res["predictive_drafts"] > 0, "Default execution must run predictive foresight"

    # Verify prediction recorded in database
    predictions = isolated_db.get_predictions(user_id="user_default")
    assert len(predictions) > 0, "Predictions table must contain generated foresight predictions"

    # Verify foresight draft created in raid_drafts
    drafts = isolated_db.get_pending_drafts(user_id="user_default")
    foresight_drafts = [d for d in drafts if d.get("detection_source") == "predictive_engine"]
    assert len(foresight_drafts) > 0, "Drafts table must contain foresight draft records"
    assert foresight_drafts[0]["llm_provider"] == "rule_based_foresight"


def test_isolated_mode_skips_predictive_foresight(isolated_db, sample_overdue_signal):
    """Confirms that run_predictive_engine=False completely skips predictive foresight."""
    pipeline = RAIDPipeline(db_manager=isolated_db)
    pipeline.classifier.provider = StubClassificationProvider(category="Not RAID", confidence=0.0)

    # Ingest overdue task signal with run_predictive_engine=False
    res = pipeline.process_signals(
        [sample_overdue_signal],
        user_id="user_isolated",
        run_predictive_engine=False
    )

    # Predictive drafts must be 0
    assert res["predictive_drafts"] == 0, "Isolated mode must produce 0 predictive drafts"

    # Verify predictions table remains empty
    predictions = isolated_db.get_predictions(user_id="user_isolated")
    assert len(predictions) == 0, "Predictions table must remain empty in isolated mode"

    # Verify no foresight drafts were saved
    drafts = isolated_db.get_pending_drafts(user_id="user_isolated")
    foresight_drafts = [d for d in drafts if d.get("detection_source") == "predictive_engine"]
    assert len(foresight_drafts) == 0, "No predictive drafts may be saved in isolated mode"


def test_isolated_mode_normal_raid_classification_and_provenance(isolated_db, sample_raid_signal):
    """Verifies that normal RAID classification, evidence verification, and provenance work in isolated mode."""
    pipeline = RAIDPipeline(db_manager=isolated_db)
    stub_prov = StubClassificationProvider(category="Risk", confidence=0.95)
    pipeline.classifier.provider = stub_prov

    res = pipeline.process_signals(
        [sample_raid_signal],
        user_id="user_isolated",
        run_predictive_engine=False
    )

    # Normal classification must succeed
    assert res["classified_drafts"] == 1
    assert res["predictive_drafts"] == 0
    assert res["drafts_created"] == 1

    # Check draft in database
    drafts = isolated_db.get_pending_drafts(user_id="user_isolated")
    assert len(drafts) == 1
    d = drafts[0]

    assert d["type"] == "Risk"
    assert d["detection_source"] == "llm_extraction"
    assert d["status"] == "Draft"
    assert d["llm_provider"] == "stub_provider"
    assert d["llm_model"] == "stub-model-v1"
    assert d["confidence_score"] == 0.95
    assert d["evidence_status"] == "verified"
    assert "delayed by two weeks" in d["evidence_text"]

    # Verify raw signal record
    sig_row = isolated_db.get_raw_signal_by_id(sample_raid_signal.id, user_id="user_isolated")
    assert sig_row["classification_status"] == "COMPLETED"
    assert sig_row["llm_provider"] == "stub_provider"
    assert sig_row["llm_model"] == "stub-model-v1"


def test_isolated_mode_preserves_deduplication(isolated_db):
    """Confirms that semantic deduplication continues to protect against duplicate RAID drafts in isolated mode."""
    pipeline = RAIDPipeline(db_manager=isolated_db)
    pipeline.classifier.provider = StubClassificationProvider(category="Risk", confidence=0.92)

    sig1 = ProjectSignal(
        id="sig-dedupe-1",
        source_type=SourceType.EMAIL,
        content="Third-party vendor API specifications are delayed by two weeks.",
        subject="Vendor Delay",
        workstream="Integration"
    )
    sig2 = ProjectSignal(
        id="sig-dedupe-2",
        source_type=SourceType.EMAIL,
        content="Third-party vendor API specifications are delayed by two weeks.",
        subject="Vendor Delay",
        workstream="Integration"
    )

    # Ingest first signal
    res1 = pipeline.process_signals([sig1], user_id="user_dedupe", run_predictive_engine=False)
    assert res1["drafts_created"] == 1
    first_draft_id = res1["drafts"][0]["id"]

    # Ingest second identical signal
    res2 = pipeline.process_signals([sig2], user_id="user_dedupe", run_predictive_engine=False)
    assert res2["duplicates_detected"] >= 1, "Duplicate must be detected in isolated mode"

    # Verify deduplication record in database
    matches = isolated_db.get_deduplication_matches(user_id="user_dedupe")
    assert len(matches) >= 1
    assert matches[0]["matched_raid_id"] == first_draft_id


def test_isolated_mode_tenant_isolation(isolated_db, sample_raid_signal):
    """Verifies that isolated classification mode respects tenant boundaries."""
    pipeline = RAIDPipeline(db_manager=isolated_db)
    pipeline.classifier.provider = StubClassificationProvider(category="Risk", confidence=0.90)

    # Ingest under User A
    pipeline.process_signals([sample_raid_signal], user_id="tenant_A", run_predictive_engine=False)

    # User B should see zero drafts and zero signals
    drafts_b = isolated_db.get_pending_drafts(user_id="tenant_B")
    assert len(drafts_b) == 0

    sig_b = isolated_db.get_raw_signal_by_id(sample_raid_signal.id, user_id="tenant_B")
    assert sig_b is None


def test_service_layer_forwards_run_predictive_engine(isolated_db):
    """Verifies that RaidService.process_meeting_minutes honors run_predictive_engine=False."""
    from src.service import RaidService
    service = RaidService(db_manager=isolated_db)
    service.pipeline.classifier.provider = StubClassificationProvider(category="Risk", confidence=0.90)

    manual_input = {
        "content": "Discussion on project milestones: vendor API specs are delayed by two weeks.",
        "author": "PMO",
        "workstream": "Integration",
        "phase": "Realize"
    }

    # Run service with run_predictive_engine=False
    results = service.process_meeting_minutes(
        manual_entry=manual_input,
        user_id="service_test_user",
        run_predictive_engine=False
    )

    # Classification occurred
    assert results["status"] == "success"
    # Zero predictions in DB
    predictions = isolated_db.get_predictions(user_id="service_test_user")
    assert len(predictions) == 0, "No predictions should be saved when run_predictive_engine=False"
