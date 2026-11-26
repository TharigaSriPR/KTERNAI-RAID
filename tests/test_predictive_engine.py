"""
Unit tests for the Rule-Based Predictive Engine (Foresight).
Covers:
1. Overdue incomplete task detection.
2. Upcoming deadline detection.
3. Completed task does not generate an overdue prediction.
4. Missing planned end date is handled safely.
5. Unknown task status is handled safely.
6. Defect cluster detection.
7. Unrelated defects are not clustered.
8. Contradicted assumption detection.
9. Assumption with no contradictory evidence is not flagged.
10. Original and contradictory evidence are preserved.
11. Duplicate predictions are prevented.
12. Configurable thresholds work.
13. Reference date can be controlled in tests.
"""

from datetime import date, timedelta
import pytest

from src.normalisation.schemas import ProjectSignal, SourceType
from src.predictive_engine.schemas import PredictionType, PredictionStatus, PredictionItem
from src.predictive_engine.rules import evaluate_task_schedule, evaluate_defect_clusters
from src.predictive_engine.assumption_monitor import AssumptionMonitor
from src.predictive_engine.predictor import PredictiveEngine


def test_1_overdue_incomplete_task_detection():
    """Confirms that an incomplete task past its planned end date triggers TASK_OVERDUE."""
    ref_date = date(2026, 11, 20)
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-101",
        author="PMO",
        subject="Task: Migrate GL Balances",
        content="Task: Migrate GL Balances | Status: In Progress | Baseline Finish: 2026-11-15 | Variance: 5 days",
        workstream="FI",
        raw_meta={
            "task_id": "TASK-101",
            "task_name": "Migrate GL Balances",
            "status": "In Progress",
            "finish_date": "2026-11-15",
            "variance_days": "5"
        }
    )
    pred = evaluate_task_schedule(sig, reference_date=ref_date)
    assert pred is not None
    assert pred.prediction_type == PredictionType.TASK_OVERDUE
    assert pred.suggested_raid_category == "Risk"
    assert "Migrate GL Balances" in pred.title
    assert "overdue" in pred.description.lower()
    assert pred.metadata["days_overdue"] == 5
    assert pred.source_signal_ids == [sig.id]


def test_2_upcoming_deadline_detection():
    """Confirms that an incomplete task within the threshold window triggers UPCOMING_DELAY."""
    ref_date = date(2026, 11, 20)
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-102",
        author="Basis Lead",
        subject="Task: System Refresh Sandbox",
        content="Task: System Refresh Sandbox | Status: Open | Baseline Finish: 2026-11-24 | Variance: 0 days",
        workstream="Basis",
        raw_meta={
            "task_id": "TASK-102",
            "task_name": "System Refresh Sandbox",
            "status": "Open",
            "finish_date": "2026-11-24",
            "variance_days": "0"
        }
    )
    pred = evaluate_task_schedule(sig, reference_date=ref_date, upcoming_threshold_days=7)
    assert pred is not None
    assert pred.prediction_type == PredictionType.UPCOMING_DELAY
    assert pred.suggested_raid_category == "Risk"
    assert pred.metadata["days_remaining"] == 4
    assert pred.confidence > 0.70


def test_3_completed_task_does_not_generate_overdue_prediction():
    """Scenario C: Completed tasks generate NO schedule risk, even if past planned finish."""
    ref_date = date(2026, 11, 20)
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-103",
        author="Security Lead",
        subject="Task: Role Build Security",
        content="Task: Role Build Security | Status: Completed | Baseline Finish: 2026-11-10",
        workstream="Security",
        raw_meta={
            "task_id": "TASK-103",
            "task_name": "Role Build Security",
            "status": "Completed",
            "finish_date": "2026-11-10"
        }
    )
    pred = evaluate_task_schedule(sig, reference_date=ref_date)
    assert pred is None


def test_4_missing_planned_end_date_handled_safely():
    """Missing or unparseable dates must not trigger exceptions or false alarms."""
    ref_date = date(2026, 11, 20)
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-104",
        author="PMO",
        subject="Task: Exploratory Meeting",
        content="Task: Exploratory Meeting | Status: In Progress",
        workstream="General",
        raw_meta={
            "task_id": "TASK-104",
            "task_name": "Exploratory Meeting",
            "status": "In Progress",
            "finish_date": None
        }
    )
    pred = evaluate_task_schedule(sig, reference_date=ref_date)
    assert pred is None


def test_5_unknown_task_status_handled_safely():
    """Requirement: Do not assume a task is incomplete when its status is unknown or missing."""
    ref_date = date(2026, 11, 20)
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-105",
        author="PMO",
        subject="Task: Unconfirmed Activity",
        content="Task: Unconfirmed Activity | Baseline Finish: 2026-11-10",
        workstream="General",
        raw_meta={
            "task_id": "TASK-105",
            "task_name": "Unconfirmed Activity",
            "status": "Unknown",
            "finish_date": "2026-11-10"
        }
    )
    pred = evaluate_task_schedule(sig, reference_date=ref_date)
    assert pred is None


def test_6_defect_cluster_detection():
    """Repeated defects in the same module sharing common themes form a DEFECT_CLUSTER prediction."""
    defects = [
        ProjectSignal(
            source_type=SourceType.DEFECT,
            source_id="DEF-001",
            author="QA Lead",
            subject="Defect: Material master validation failure",
            content="Defect DEF-001: Material master validation failure | Severity: High | Module: MM",
            workstream="MM",
            raw_meta={"defect_id": "DEF-001", "summary": "Material master validation failure", "severity": "High", "module": "MM"}
        ),
        ProjectSignal(
            source_type=SourceType.DEFECT,
            source_id="DEF-002",
            author="QA Lead",
            subject="Defect: Material master validation failure during procurement testing",
            content="Defect DEF-002: Material master validation failure during procurement testing | Severity: High | Module: MM",
            workstream="MM",
            raw_meta={"defect_id": "DEF-002", "summary": "Material master validation failure during procurement testing", "severity": "High", "module": "MM"}
        ),
        ProjectSignal(
            source_type=SourceType.DEFECT,
            source_id="DEF-003",
            author="QA Lead",
            subject="Defect: Incorrect material master data causing integration failure",
            content="Defect DEF-003: Incorrect material master data causing integration failure | Severity: High | Module: MM",
            workstream="MM",
            raw_meta={"defect_id": "DEF-003", "summary": "Incorrect material master data causing integration failure", "severity": "High", "module": "MM"}
        )
    ]
    clusters = evaluate_defect_clusters(defects, min_cluster_size=3, similarity_threshold=0.35)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.prediction_type == PredictionType.DEFECT_CLUSTER
    assert c.workstream == "MM"
    assert "DEF-001" in c.description or "DEF-001" in c.metadata["defect_ids"]
    assert len(c.source_signal_ids) == 3


def test_7_unrelated_defects_are_not_clustered():
    """Unrelated defects with different topics must not be clustered simply because they exist."""
    defects = [
        ProjectSignal(
            source_type=SourceType.DEFECT,
            source_id="DEF-101",
            author="QA",
            subject="Defect: Material master validation failure",
            content="Defect DEF-101: Material master validation failure | Severity: High | Module: MM",
            workstream="MM",
            raw_meta={"defect_id": "DEF-101", "summary": "Material master validation failure", "severity": "High", "module": "MM"}
        ),
        ProjectSignal(
            source_type=SourceType.DEFECT,
            source_id="DEF-102",
            author="QA",
            subject="Defect: User button CSS alignment on mobile tablet",
            content="Defect DEF-102: User button CSS alignment on mobile tablet | Severity: High | Module: MM",
            workstream="MM",
            raw_meta={"defect_id": "DEF-102", "summary": "User button CSS alignment on mobile tablet", "severity": "High", "module": "MM"}
        ),
        ProjectSignal(
            source_type=SourceType.DEFECT,
            source_id="DEF-103",
            author="QA",
            subject="Defect: Email SMTP notification server timeout",
            content="Defect DEF-103: Email SMTP notification server timeout | Severity: High | Module: MM",
            workstream="MM",
            raw_meta={"defect_id": "DEF-103", "summary": "Email SMTP notification server timeout", "severity": "High", "module": "MM"}
        )
    ]
    clusters = evaluate_defect_clusters(defects, min_cluster_size=3, similarity_threshold=0.45)
    assert len(clusters) == 0


def test_8_contradicted_assumption_detection():
    """Detects when subsequent communication contradicts an earlier recorded assumption."""
    monitor = AssumptionMonitor()
    assumptions = [{
        "id": "draft-asm-1",
        "title": "Client master data availability",
        "description": "Client master data will be available before SIT starts.",
        "evidence_text": "Client master data will be available before SIT starts.",
        "evidence_source_id": "sig-asm-origin",
        "workstream": "MM"
    }]

    subsequent_signals = [
        ProjectSignal(
            source_type=SourceType.CHAT,
            source_id="chat-update-99",
            author="Data Migration Lead",
            subject="Chat from Migration Channel",
            content="Client master data preparation is still incomplete, and SIT starts next week.",
            workstream="MM"
        )
    ]

    preds = monitor.check_contradictions(assumptions, subsequent_signals)
    assert len(preds) == 1
    p = preds[0]
    assert p.prediction_type == PredictionType.ASSUMPTION_CONTRADICTED
    assert "Client master data availability" in p.title
    assert "incomplete" in p.evidence.lower()
    assert "draft-asm-1" in p.related_raid_ids


def test_9_assumption_with_no_contradictory_evidence_not_flagged():
    """Unrelated negative communications must not falsely trigger an assumption contradiction."""
    monitor = AssumptionMonitor()
    assumptions = [{
        "id": "draft-asm-2",
        "title": "Client master data availability",
        "description": "Client master data will be available before SIT starts.",
        "evidence_text": "Client master data will be available before SIT starts.",
        "evidence_source_id": "sig-asm-origin-2",
        "workstream": "MM"
    }]

    # A negative signal talking about a completely different topic (laptop delivery)
    subsequent_signals = [
        ProjectSignal(
            source_type=SourceType.CHAT,
            source_id="chat-unrelated",
            author="IT Support",
            subject="Laptop shipment",
            content="Laptop delivery is delayed and incomplete for new team members.",
            workstream="General"
        )
    ]

    preds = monitor.check_contradictions(assumptions, subsequent_signals)
    assert len(preds) == 0


def test_10_original_and_contradictory_evidence_preserved():
    """Both the original assumption quote and contradictory evidence snippet must be preserved."""
    monitor = AssumptionMonitor()
    assumptions = [{
        "id": "draft-asm-3",
        "title": "Network firewall port 443 open",
        "description": "Network firewall port 443 will be opened by infrastructure team.",
        "evidence_text": "Network firewall port 443 will be opened by infrastructure team.",
        "evidence_source_id": "sig-port-asm",
        "workstream": "Basis"
    }]

    subsequent_signals = [
        ProjectSignal(
            source_type=SourceType.EMAIL,
            source_id="email-firewall-fail",
            author="Security Ops",
            subject="Firewall ticket update",
            content="Network firewall port 443 request was rejected and blocked pending audit.",
            workstream="Basis"
        )
    ]

    preds = monitor.check_contradictions(assumptions, subsequent_signals)
    assert len(preds) == 1
    p = preds[0]
    assert "Original Assumption:" in p.evidence
    assert "Contradicting Evidence:" in p.evidence
    assert "Network firewall port 443" in p.evidence
    assert "sig-port-asm" in p.source_signal_ids
    assert subsequent_signals[0].id in p.source_signal_ids


def test_11_duplicate_predictions_are_prevented():
    """PredictiveEngine must not produce duplicate predictions for the same triggering sources."""
    engine = PredictiveEngine()
    ref_date = date(2026, 11, 20)
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-200",
        author="PMO",
        subject="Task: Cutover Runbook",
        content="Task: Cutover Runbook | Status: In Progress | Baseline Finish: 2026-11-10",
        workstream="PMO",
        raw_meta={
            "task_id": "TASK-200",
            "task_name": "Cutover Runbook",
            "status": "In Progress",
            "finish_date": "2026-11-10"
        }
    )

    preds1 = engine.predict_schedule_slippage([sig], reference_date=ref_date)
    assert len(preds1) == 1

    # Second run with same signal
    preds2 = engine.predict_schedule_slippage([sig], reference_date=ref_date)
    assert len(preds2) == 0


def test_12_configurable_thresholds_work():
    """Configurable deadline threshold window adjusts upcoming prediction behavior."""
    ref_date = date(2026, 11, 20)
    # Task finish is 10 days away
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-300",
        author="PMO",
        subject="Task: UAT Test Cases",
        content="Task: UAT Test Cases | Status: Open | Baseline Finish: 2026-11-30",
        workstream="Testing",
        raw_meta={
            "task_id": "TASK-300",
            "task_name": "UAT Test Cases",
            "status": "Open",
            "finish_date": "2026-11-30"
        }
    )

    # With default 7 days threshold: outside window -> None
    pred_7 = evaluate_task_schedule(sig, reference_date=ref_date, upcoming_threshold_days=7)
    assert pred_7 is None

    # With custom 14 days threshold: within window -> UPCOMING_DELAY
    pred_14 = evaluate_task_schedule(sig, reference_date=ref_date, upcoming_threshold_days=14)
    assert pred_14 is not None
    assert pred_14.prediction_type == PredictionType.UPCOMING_DELAY


def test_13_reference_date_can_be_controlled():
    """Explicit reference date allows testing against any historical or future project date."""
    sig = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-400",
        author="PMO",
        subject="Task: Interface Testing",
        content="Task: Interface Testing | Status: In Progress | Baseline Finish: 2026-06-15",
        workstream="Integration",
        raw_meta={
            "task_id": "TASK-400",
            "task_name": "Interface Testing",
            "status": "In Progress",
            "finish_date": "2026-06-15"
        }
    )

    # When evaluated on 2026-06-01 (before finish) -> Not overdue
    pred_past = evaluate_task_schedule(sig, reference_date="2026-06-01", upcoming_threshold_days=7)
    assert pred_past is None

    # When evaluated on 2026-06-20 (after finish) -> Overdue
    pred_future = evaluate_task_schedule(sig, reference_date="2026-06-20")
    assert pred_future is not None
    assert pred_future.prediction_type == PredictionType.TASK_OVERDUE



def test_14_dependency_blockage_detected_when_prerequisite_incomplete():
    """Identifies potential blockage when Task B requires incomplete Task A."""
    sig_a = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-A",
        author="Basis Lead",
        subject="Task: Complete system refresh",
        content="Task: Complete system refresh | Status: In Progress | Baseline Finish: 2026-11-20",
        workstream="Basis",
        raw_meta={
            "task_id": "TASK-A",
            "task_name": "Complete system refresh",
            "status": "In Progress",
            "finish_date": "2026-11-20"
        }
    )
    sig_b = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-B",
        author="Functional Lead",
        subject="Task: Begin configuration testing",
        content="Task: Begin configuration testing | Status: Not Started | Dependencies: TASK-A",
        workstream="FI",
        raw_meta={
            "task_id": "TASK-B",
            "task_name": "Begin configuration testing",
            "status": "Not Started",
            "dependencies": "TASK-A",
            "finish_date": "2026-11-28"
        }
    )

    engine = PredictiveEngine()
    preds = engine.predict_dependency_blockages([sig_a, sig_b])

    assert len(preds) == 1
    p = preds[0]
    assert p.prediction_type == PredictionType.DEPENDENCY_BLOCKAGE
    assert "Begin configuration testing" in p.title
    assert "Complete system refresh" in p.title
    assert "TASK-B" in p.evidence and "TASK-A" in p.evidence
    assert p.metadata["prerequisite_status"] == "In Progress"
    assert sig_b.id in p.source_signal_ids
    assert sig_a.id in p.source_signal_ids
    assert p.rule_score == 0.90


def test_15_no_dependency_blockage_when_prerequisite_completed():
    """No dependency blockage is generated if the prerequisite task is already finished."""
    sig_a = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-A",
        author="Basis Lead",
        subject="Task: Complete system refresh",
        content="Task: Complete system refresh | Status: Completed | Baseline Finish: 2026-11-15",
        workstream="Basis",
        raw_meta={
            "task_id": "TASK-A",
            "task_name": "Complete system refresh",
            "status": "Completed",
            "finish_date": "2026-11-15"
        }
    )
    sig_b = ProjectSignal(
        source_type=SourceType.PROJECT_PLAN,
        source_id="TASK-B",
        author="Functional Lead",
        subject="Task: Begin configuration testing",
        content="Task: Begin configuration testing | Status: Not Started | Dependencies: TASK-A",
        workstream="FI",
        raw_meta={
            "task_id": "TASK-B",
            "task_name": "Begin configuration testing",
            "status": "Not Started",
            "dependencies": "TASK-A",
            "finish_date": "2026-11-28"
        }
    )

    engine = PredictiveEngine()
    preds = engine.predict_dependency_blockages([sig_a, sig_b])
    assert len(preds) == 0


def test_16_assumption_not_contradicted_by_older_historical_signal():
    """A communication that occurred BEFORE an assumption was made cannot contradict it."""
    monitor = AssumptionMonitor()
    assumptions = [{
        "id": "asm-dated-1",
        "title": "Client master data availability",
        "description": "Client master data will be available before SIT starts.",
        "evidence_text": "Client master data will be available before SIT starts.",
        "evidence_source_id": "sig-asm-dated",
        "workstream": "MM",
        "timestamp": "2026-11-15T10:00:00"
    }]

    # A historical signal that occurred before the assumption was agreed
    older_signals = [
        ProjectSignal(
            source_type=SourceType.CHAT,
            source_id="chat-old-1",
            author="Migration Lead",
            timestamp="2026-11-10T09:00:00",
            subject="Past status",
            content="Client master data preparation is still incomplete.",
            workstream="MM"
        )
    ]

    preds = monitor.check_contradictions(assumptions, older_signals)
    assert len(preds) == 0


def test_17_prediction_confidence_and_rule_score_interchangeability():
    """Confirms confidence and rule_score represent the same deterministic rule metric."""
    pred = PredictionItem(
        prediction_type=PredictionType.TASK_OVERDUE,
        title="Test Overdue Task",
        description="Overdue description",
        confidence=0.85,
        evidence="Evidence snippet",
        source_signal_ids=["sig-1"]
    )
    assert pred.confidence == 0.85
    assert pred.rule_score == 0.85
