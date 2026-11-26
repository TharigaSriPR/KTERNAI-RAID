"""
Regression tests for Stage 6B Data Integrity Remediation:
1. Fix 1: Correct Owner Attribution (no sender-as-owner fallback, no synthetic PMO/QA Team defaults).
2. Fix 2: Preserve LLM Reasoning (reasoning_summary survives to scoring_reason, DB, and retrieval).
3. Fix 3: Remove Few-Shot Probability Bias (prompt templates and schema support null probability/severity).
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.normalisation.schemas import ProjectSignal, SourceType
from src.classification.schemas import RAIDCategory, RAIDExtractionResult
from src.classification.classifier import RAIDClassifier
from src.classification.providers.mock_provider import MockLLMProvider
from src.classification.prompt_templates import FEW_SHOT_EXAMPLES, SYSTEM_INSTRUCTION
from src.pipeline import RAIDPipeline
from src.service import RaidService
from src.database.db_manager import DatabaseManager
from src.database.migrations import run_migrations
from src.ingestion.plan_ingestor import PlanIngestor
from src.ingestion.defect_ingestor import DefectIngestor


@pytest.fixture
def clean_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "stage6b_test.db"
    db = DatabaseManager(db_file)
    run_migrations(db_file)
    return db


# ===========================================================================
# Fix 1: Correct Owner Attribution Tests
# ===========================================================================

class TestOwnerAttributionRemediation:
    def test_owner_explicitly_stated_in_source(self, clean_db: DatabaseManager):
        """Scenario 1: Owner explicitly extracted by LLM is preserved, not replaced."""
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Data migration timeline risk",
            "description": "Risk that master data delivery will slip.",
            "confidence": 0.90,
            "owner": "Alice Smith",
            "evidence": "Alice Smith is accountable for data migration.",
            "reasoning_summary": "Explicit owner named in source."
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-owner-01",
            source_type=SourceType.EMAIL,
            author="bob.sender@enterprise.com",
            content="Alice Smith is accountable for data migration."
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        assert res["classified_drafts"] == 1
        draft = res["drafts"][0]

        # Accountable owner is Alice Smith, NOT Bob the sender
        assert draft["likely_owner"] == "Alice Smith"
        assert signal.author == "bob.sender@enterprise.com"

        # Check DB persistence
        db_drafts = clean_db.get_pending_drafts(user_id="test_user")
        assert len(db_drafts) == 1
        assert db_drafts[0]["likely_owner"] == "Alice Smith"

    def test_owner_absent_but_sender_present(self, clean_db: DatabaseManager):
        """Scenario 2: When LLM extracts owner=None, sender must NOT become the owner."""
        mock_json = json.dumps({
            "category": "Issue",
            "title": "Firewall port 443 blocked",
            "description": "Port 443 blocked between systems.",
            "confidence": 0.92,
            "owner": None,
            "evidence": "Firewall port 443 is blocked between ERP and CPI.",
            "reasoning_summary": "Active blocker reported by engineer."
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-owner-02",
            source_type=SourceType.EMAIL,
            author="reporting.engineer@kaartech.com",
            content="Firewall port 443 is blocked between ERP and CPI."
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        assert res["classified_drafts"] == 1
        draft = res["drafts"][0]

        # The reporting engineer must NOT become the owner
        assert draft["likely_owner"] is None

        # Sender is preserved in signal metadata
        assert signal.author == "reporting.engineer@kaartech.com"

        # Check SQLite persistence: stored as NULL
        db_drafts = clean_db.get_pending_drafts(user_id="test_user")
        assert len(db_drafts) == 1
        assert db_drafts[0]["likely_owner"] is None

    def test_owner_absent_and_sender_absent(self, clean_db: DatabaseManager):
        """Scenario 3: When owner is absent and sender is absent, likely_owner remains None."""
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Generic schedule risk",
            "description": "Potential delay mentioned in anonymous notes.",
            "confidence": 0.85,
            "owner": None,
            "evidence": "Schedule may slip by two weeks.",
            "reasoning_summary": "Schedule uncertainty."
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-owner-03",
            source_type=SourceType.MEETING_NOTES,
            author="",
            content="Schedule may slip by two weeks."
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        assert res["classified_drafts"] == 1
        draft = res["drafts"][0]
        assert draft["likely_owner"] is None

    def test_owner_explicitly_assigned_to_team(self, clean_db: DatabaseManager):
        """Scenario 4: Team-level accountable owner is preserved."""
        mock_json = json.dumps({
            "category": "Dependency",
            "title": "Basis system refresh prerequisite",
            "description": "Testing waiting on Basis team refresh.",
            "confidence": 0.94,
            "owner": "Basis Team",
            "evidence": "Configuration testing can begin only after Basis team completes refresh.",
            "reasoning_summary": "External team deliverable prerequisite."
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-owner-04",
            source_type=SourceType.CHAT,
            author="DevLead",
            content="Configuration testing can begin only after Basis team completes refresh."
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        draft = res["drafts"][0]
        assert draft["likely_owner"] == "Basis Team"

    def test_plan_and_defect_ingestion_no_stated_owner(self, tmp_path: Path):
        """Scenario 5: Plan and defect ingestion do not inject synthetic PMO/QA Team owners."""
        # 1. Project plan CSV with empty Assignee
        plan_csv = tmp_path / "plan_no_owner.csv"
        plan_csv.write_text(
            "Task ID,Task Name,Assignee,Status,Planned Finish\n"
            "TASK-101,Configure FI-GL,,In Progress,2026-12-01\n",
            encoding="utf-8"
        )
        plan_ingestor = PlanIngestor()
        plan_signals = plan_ingestor.ingest(plan_csv)
        assert len(plan_signals) == 1
        # Author must NOT be synthetically set to "PMO"
        assert plan_signals[0].author == "Unknown"
        assert plan_signals[0].raw_meta.get("owner") is None

        # 2. Defect CSV with empty Reporter/Assignee
        defect_csv = tmp_path / "defect_no_owner.csv"
        defect_csv.write_text(
            "Defect ID,Summary,Severity,Status,Assignee\n"
            "DEF-201,ABAP dump on billing,,Open,\n",
            encoding="utf-8"
        )
        defect_ingestor = DefectIngestor()
        defect_signals = defect_ingestor.ingest(defect_csv)
        assert len(defect_signals) == 1
        # Author must NOT be synthetically set to "QA Team"
        assert defect_signals[0].author == "Unknown"
        assert defect_signals[0].raw_meta.get("owner") is None

    def test_approval_preserves_none_owner_in_database(self, clean_db: DatabaseManager):
        """Scenario 6: Approving a draft with owner=None stores NULL in approved_raid, not 'Unassigned'."""
        service = RaidService(clean_db)
        draft_record = {
            "id": "raid-test-null-owner",
            "user_id": "test_user",
            "type": "Risk",
            "title": "Unassigned Risk Item",
            "description": "Risk item without accountable owner.",
            "workstream": "Finance",
            "phase": "Realize",
            "likely_owner": None,
            "confidence_score": 0.88,
            "evidence_text": "Evidence excerpt.",
            "evidence_status": "verified",
            "status": "Draft"
        }
        clean_db.save_raid_draft(draft_record, user_id="test_user")

        # Approve draft
        approved = service.approve_raid(
            raid_id="raid-test-null-owner",
            approved_by="Project Manager",
            user_id="test_user"
        )
        assert approved["owner"] is None

        # Query database row directly
        approved_items = clean_db.get_approved_items(user_id="test_user")
        assert len(approved_items) == 1
        assert approved_items[0]["owner"] is None


# ===========================================================================
# Fix 2: Preserve LLM Reasoning Tests
# ===========================================================================

class TestPreserveLLMReasoningRemediation:
    def test_reasoning_survives_pipeline_classification_and_db(self, clean_db: DatabaseManager):
        """LLM reasoning_summary survives into scoring_reason, DB persistence, and retrieval."""
        expected_reasoning = "Future risk: custom code remediation backlog may jeopardize end-to-end integration testing."
        mock_json = json.dumps({
            "category": "Risk",
            "title": "Custom code remediation delay",
            "description": "40% of Z-objects remain unanalysed.",
            "confidence": 0.92,
            "owner": None,
            "evidence": "40% of Z-objects are still unanalysed by the offshore ABAP team.",
            "reasoning_summary": expected_reasoning
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-reason-01",
            source_type=SourceType.EMAIL,
            content="40% of Z-objects are still unanalysed by the offshore ABAP team."
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        assert res["classified_drafts"] == 1
        draft = res["drafts"][0]

        # Meaningful reasoning is preserved in scoring_reason
        assert draft["scoring_reason"] == expected_reasoning
        # Confidence score is maintained independently
        assert draft["confidence_score"] == 0.92

        # Check DB retrieval
        db_drafts = clean_db.get_pending_drafts(user_id="test_user")
        assert len(db_drafts) == 1
        assert db_drafts[0]["scoring_reason"] == expected_reasoning
        assert db_drafts[0]["confidence_score"] == 0.92

    def test_missing_reasoning_falls_back_safely(self, clean_db: DatabaseManager):
        """When reasoning_summary is omitted by the model, fall back safely to generic confidence message."""
        mock_json = json.dumps({
            "category": "Issue",
            "title": "Transport error RC=8",
            "description": "Transport TRK90012 failed.",
            "confidence": 0.95,
            "evidence": "Transport TRK90012 to QA failed with RC=8.",
            "reasoning_summary": None
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-reason-02",
            source_type=SourceType.EMAIL,
            content="Transport TRK90012 to QA failed with RC=8."
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        draft = res["drafts"][0]
        # Falls back cleanly without breaking
        assert "Classified by LLM with confidence 0.95" in draft["scoring_reason"]

    def test_confidence_and_evidence_distinguished_from_reasoning(self, clean_db: DatabaseManager):
        """Verifies that scoring_reason, confidence_score, and evidence_text remain distinct fields."""
        reasoning = "Unverified planning assumption accepted without proof."
        evidence = "We are assuming that client will complete master data prep before SIT."
        mock_json = json.dumps({
            "category": "Assumption",
            "title": "Client master data assumption",
            "description": "Assumption regarding master data readiness.",
            "confidence": 0.87,
            "evidence": evidence,
            "reasoning_summary": reasoning
        })
        provider = MockLLMProvider(mock_response=mock_json)
        pipeline = RAIDPipeline(clean_db)
        pipeline.classifier = RAIDClassifier(provider=provider, db_manager=clean_db)

        signal = ProjectSignal(
            id="sig-reason-03",
            source_type=SourceType.MEETING_NOTES,
            content=evidence
        )

        res = pipeline.process_signals([signal], user_id="test_user", run_predictive_engine=False)
        draft = res["drafts"][0]

        assert draft["scoring_reason"] == reasoning
        assert draft["confidence_score"] == 0.87
        assert draft["evidence_text"] == evidence


# ===========================================================================
# Fix 3: Remove Few-Shot Probability Bias Tests
# ===========================================================================

class TestProbabilityAndRatingBiasRemediation:
    def test_prompt_example1_probability_is_null(self):
        """Verifies few-shot Example 1 has probability: null and strict extraction rules forbidding guessing."""
        assert '"probability": null' in FEW_SHOT_EXAMPLES
        assert "Do not force, guess, or invent risk probability" in SYSTEM_INSTRUCTION

    def test_no_probability_stated_parses_with_none(self):
        """Scenario 1: Signal with no stated probability parses with probability=None."""
        res = RAIDExtractionResult(
            category=RAIDCategory.RISK,
            title="Interface specification delay",
            description="3PL EDI specification is unresolved.",
            confidence=0.90,
            probability=None,
            source_signal_id="sig-p-01"
        )
        assert res.probability is None

    def test_explicit_high_probability_preserved(self):
        """Scenario 2: Explicitly stated High probability is preserved."""
        res = RAIDExtractionResult(
            category=RAIDCategory.RISK,
            title="High likelihood cutover delay",
            description="Risk of cutover slip.",
            confidence=0.92,
            probability="High",
            source_signal_id="sig-p-02"
        )
        assert res.probability == "High"

    def test_explicit_low_probability_preserved(self):
        """Scenario 3: Explicitly stated Low probability is preserved."""
        res = RAIDExtractionResult(
            category=RAIDCategory.RISK,
            title="Minor environmental discrepancy",
            description="Low likelihood risk.",
            confidence=0.88,
            probability="Low",
            source_signal_id="sig-p-03"
        )
        assert res.probability == "Low"

    def test_impact_described_but_probability_absent(self):
        """Scenario 4: Impact described but probability unmentioned keeps probability=None."""
        res = RAIDExtractionResult(
            category=RAIDCategory.RISK,
            title="Potential UAT delay",
            description="Defect backlog could delay UAT commencement by 1 week.",
            impact="1-week delay to UAT cycle start",
            confidence=0.91,
            probability=None,
            source_signal_id="sig-p-04"
        )
        assert res.impact == "1-week delay to UAT cycle start"
        assert res.probability is None

    def test_insufficient_information_for_severity_keeps_severity_none(self):
        """Scenario 5: Issue without stated severity keeps severity=None."""
        res = RAIDExtractionResult(
            category=RAIDCategory.ISSUE,
            title="Temporary license key expired",
            description="Sandbox license expired.",
            confidence=0.93,
            severity=None,
            source_signal_id="sig-p-05"
        )
        assert res.severity is None
