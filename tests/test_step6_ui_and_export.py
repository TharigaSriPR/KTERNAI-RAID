"""
Unit and Integration Tests for Step 6: Streamlit Dashboard and Excel Export.
KTern.AI RAID Identification Agent POC

Verifies:
1. KPI metrics and distribution summaries against empty and populated databases.
2. Review Queue workflows (approve, reject, modify) through RaidService.
3. Approved Register isolation (excluding pending and rejected drafts).
4. Predictive Insights retrieval and empty-state safety.
5. Duplicate Review suggestions (read-only, non-destructive).
6. Audit History trail retrieval and JSON diff rendering.
7. Excel Export generation using openpyxl (headers, auto-filters, frozen panes,
   types, multi-tab combined workbook, empty dataset resilience).
"""

import io
import json
import pytest
import sqlite3
from pathlib import Path
from typing import Dict, Any
import openpyxl

from src.database.db_manager import DatabaseManager
from src.service import (
    RaidService,
    RaidStatus,
    DuplicateApprovalError,
    InvalidRaidStatusError,
    RaidValidationError
)
from src.normalisation.schemas import ProjectSignal, SourceType
from src.predictive_engine.schemas import PredictionItem, PredictionType
from src.deduplication.schemas import DeduplicationResult, MatchStatus
from src.export.excel_export import (
    export_approved_register,
    export_review_queue,
    export_predictions,
    export_audit_history,
    export_combined_report
)


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Creates a clean, isolated SQLite DatabaseManager instance for testing."""
    db_file = tmp_path / "test_step6.db"
    return DatabaseManager(db_file)


@pytest.fixture
def service(temp_db: DatabaseManager) -> RaidService:
    """Creates a RaidService instance backed by an isolated database."""
    return RaidService(temp_db)


@pytest.fixture
def populated_db(temp_db: DatabaseManager) -> DatabaseManager:
    """Populates the database with representative signals, drafts, approved items, predictions, and duplicates."""
    # 1. Raw Signal
    sig = ProjectSignal(
        source_type=SourceType.EMAIL,
        source_id="sig-email-001",
        author="Architecture Lead",
        subject="RFC Destination Timeout",
        content="Testing blocked because RFC destination to S/4HANA times out.",
        workstream="Integration",
        phase="Realize"
    )
    temp_db.save_raw_signal(sig)

    # 2. RAID Draft 1 (Pending)
    draft1 = {
        "id": "raid-001",
        "type": "Issue",
        "title": "RFC Destination Timeout to S/4HANA",
        "description": "Testing blocked because RFC destination to S/4HANA times out.",
        "workstream": "Integration",
        "phase": "Realize",
        "probability": None,
        "impact": "High",
        "severity": "High",
        "scoring_reason": "Direct communication indicates blocked testing",
        "likely_owner": "Basis Lead",
        "due_date": "2026-12-05",
        "mitigation_action": "Check gateway security ACLs",
        "confidence_score": 0.88,
        "evidence_text": "Testing blocked because RFC destination to S/4HANA times out.",
        "evidence_source_id": "sig-email-001",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "duplicate_of_id": None,
        "status": "Draft"
    }
    temp_db.save_raid_draft(draft1)

    # 3. RAID Draft 2 (Modified)
    draft2 = {
        "id": "raid-002",
        "type": "Risk",
        "title": "Custom Code Remediation Delay",
        "description": "Custom code remediation is running 3 weeks behind schedule.",
        "workstream": "ABAP",
        "phase": "Realize",
        "probability": "High",
        "impact": "High",
        "severity": "High",
        "scoring_reason": "Historical velocity indicates slippage",
        "likely_owner": "Dev Manager",
        "due_date": "2026-12-15",
        "mitigation_action": "Add offshore ABAP developers",
        "confidence_score": 0.84,
        "evidence_text": "Remediation is running 3 weeks behind schedule.",
        "evidence_source_id": "sig-email-001",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "duplicate_of_id": None,
        "status": "Modified"
    }
    temp_db.save_raid_draft(draft2)

    # 4. RAID Draft 3 (Rejected)
    draft3 = {
        "id": "raid-003",
        "type": "Assumption",
        "title": "Invalid Assumption",
        "description": "Assume vendor provides free licenses.",
        "workstream": "Finance",
        "phase": "Prepare",
        "probability": "Low",
        "impact": "Low",
        "severity": "Low",
        "scoring_reason": "Heuristic match",
        "likely_owner": "Finance Lead",
        "due_date": "2026-11-30",
        "mitigation_action": "Verify contract",
        "confidence_score": 0.50,
        "evidence_text": "Assume vendor provides free licenses.",
        "evidence_source_id": "sig-email-001",
        "evidence_source_type": "email",
        "detection_source": "llm_extraction",
        "duplicate_of_id": None,
        "status": "Rejected"
    }
    temp_db.save_raid_draft(draft3)

    # 5. Prediction
    pred = PredictionItem(
        prediction_id="pred-001",
        prediction_type=PredictionType.TASK_OVERDUE,
        title="Overdue Custom Code Task",
        description="Task 'Custom Code Remediation' is 14 days overdue.",
        suggested_raid_category="Risk",
        confidence=0.92,
        severity="High",
        evidence="Task end date was 2026-11-01, current status In Progress.",
        source_signal_ids=["sig-plan-101"]
    )
    temp_db.save_prediction(pred)

    # 6. Deduplication Match
    dedup = DeduplicationResult(
        new_raid_id="raid-002",
        matched_raid_id="raid-001",
        similarity_score=0.78,
        match_status=MatchStatus.POSSIBLE_DUPLICATE,
        match_reason="High term overlap in S/4HANA remediation context",
        same_category=False,
        same_workstream=False,
        requires_human_review=True
    )
    temp_db.save_deduplication_match(dedup)

    return temp_db


# ---------------------------------------------------------------------------
# 1. Dashboard Overview & KPI Counts
# ---------------------------------------------------------------------------

def test_1_dashboard_empty_database_kpi(temp_db: DatabaseManager):
    """Empty database reports zero across all KPI metrics without crashing."""
    service = RaidService(temp_db)
    kpis = service.get_kpis()
    assert kpis["total_drafts"] == 0
    assert kpis["pending_reviews"] == 0
    assert kpis["approved_items"] == 0
    assert kpis["rejected_drafts"] == 0
    assert kpis["active_predictions"] == 0
    assert kpis["potential_duplicates"] == 0

    all_drafts = service.get_all_drafts()
    assert len(all_drafts) == 0


def test_2_dashboard_populated_kpis(populated_db: DatabaseManager):
    """Populated database returns accurate counts matching underlying SQLite state."""
    service = RaidService(populated_db)
    kpis = service.get_kpis()
    assert kpis["total_drafts"] == 3
    assert kpis["pending_reviews"] == 2  # raid-001 (Draft) and raid-002 (Modified)
    assert kpis["approved_items"] == 0
    assert kpis["rejected_drafts"] == 1  # raid-003 (Rejected)
    assert kpis["active_predictions"] == 1
    assert kpis["potential_duplicates"] == 1


def test_3_missing_optional_fields_safety(temp_db: DatabaseManager):
    """Missing or null fields in drafts do not cause failure in retrieval or KPI calculations."""
    draft = {
        "id": "raid-sparse",
        "type": "Decision",
        "title": "Sparse Decision",
        "description": "Minimal description.",
        "workstream": None,
        "phase": None,
        "probability": None,
        "impact": None,
        "severity": None,
        "scoring_reason": None,
        "likely_owner": None,
        "due_date": None,
        "mitigation_action": None,
        "confidence_score": 0.70,
        "evidence_text": "Minimal quote",
        "evidence_source_id": "sig-000",
        "evidence_source_type": "chat",
        "detection_source": "llm_extraction",
        "duplicate_of_id": None,
        "status": "Draft"
    }
    temp_db.save_raid_draft(draft)
    service = RaidService(temp_db)

    drafts = service.get_pending_drafts()
    assert len(drafts) == 1
    assert drafts[0]["workstream"] is None
    assert drafts[0]["phase"] is None


# ---------------------------------------------------------------------------
# 2. Review Queue Workflows via RaidService
# ---------------------------------------------------------------------------

def test_4_review_queue_approve_action(populated_db: DatabaseManager):
    """Approve action moves draft from review queue to approved register."""
    service = RaidService(populated_db)
    initial_pending = len(service.get_pending_drafts())
    assert initial_pending == 2

    approved = service.approve_raid(
        raid_id="raid-001",
        approved_by="Jane PM Lead",
        comments="Approved for Sprint 5 cutover"
    )
    assert approved["draft_id"] == "raid-001"
    assert approved["approved_by"] == "Jane PM Lead"

    # Pending count decrements by 1
    assert len(service.get_pending_drafts()) == initial_pending - 1
    # Approved register increments by 1
    assert len(service.get_approved_items()) == 1


def test_5_review_queue_reject_requires_reason(populated_db: DatabaseManager):
    """Reject action fails if rejection reason is empty or whitespace."""
    service = RaidService(populated_db)
    with pytest.raises(RaidValidationError) as exc:
        service.reject_raid("raid-001", rejected_by="PM", reason="")
    assert "rejection reason is required" in str(exc.value).lower()

    with pytest.raises(RaidValidationError) as exc:
        service.reject_raid("raid-001", rejected_by="PM", reason="   ")
    assert "rejection reason is required" in str(exc.value).lower()


def test_6_review_queue_modify_action_retains_in_queue(populated_db: DatabaseManager):
    """Modify action updates fields, preserves immutable fields, and keeps draft in review queue."""
    service = RaidService(populated_db)
    service.modify_raid(
        raid_id="raid-001",
        modified_by="PM Lead",
        updates={"workstream": "Basis", "likely_owner": "Basis Architect"},
        comments="Refined ownership"
    )

    updated = service.get_raid_by_id("raid-001")
    assert updated["workstream"] == "Basis"
    assert updated["likely_owner"] == "Basis Architect"
    assert updated["status"] == "Modified"
    # Immutable fields preserved
    assert updated["evidence_source_id"] == "sig-email-001"
    assert updated["detection_source"] == "llm_extraction"

    # Remains visible in review queue
    pending_ids = [d["id"] for d in service.get_pending_drafts()]
    assert "raid-001" in pending_ids


def test_7_review_queue_blocks_invalid_transitions(populated_db: DatabaseManager):
    """Service strictly blocks invalid status transitions."""
    service = RaidService(populated_db)
    # raid-003 is already Rejected; approving it must be blocked
    with pytest.raises(InvalidRaidStatusError):
        service.approve_raid("raid-003", approved_by="PM")


# ---------------------------------------------------------------------------
# 3. Approved Register
# ---------------------------------------------------------------------------

def test_8_approved_register_exclusivity(populated_db: DatabaseManager):
    """Approved register returns strictly approved items; pending and rejected are excluded."""
    service = RaidService(populated_db)
    assert len(service.get_approved_items()) == 0

    service.approve_raid("raid-001", approved_by="PM Director")
    approved = service.get_approved_items()
    assert len(approved) == 1
    assert approved[0]["draft_id"] == "raid-001"

    # Ensure rejected draft (raid-003) and modified draft (raid-002) are not in approved register
    appr_draft_ids = [item["draft_id"] for item in approved]
    assert "raid-002" not in appr_draft_ids
    assert "raid-003" not in appr_draft_ids


# ---------------------------------------------------------------------------
# 4. Predictive Insights
# ---------------------------------------------------------------------------

def test_9_predictive_insights_read_only(populated_db: DatabaseManager):
    """Predictions are retrieved directly from SQLite without modification or invention."""
    service = RaidService(populated_db)
    preds = service.get_predictions()
    assert len(preds) == 1
    p = preds[0]
    assert p["id"] == "pred-001"
    assert p["prediction_type"] == PredictionType.TASK_OVERDUE.value
    assert p["severity"] == "High"
    assert p["confidence"] == 0.92


def test_10_predictive_insights_empty_safe(temp_db: DatabaseManager):
    """Empty predictions table returns empty list safely."""
    service = RaidService(temp_db)
    assert service.get_predictions() == []


# ---------------------------------------------------------------------------
# 5. Duplicate Review
# ---------------------------------------------------------------------------

def test_11_duplicate_review_suggestions_safe(populated_db: DatabaseManager):
    """Duplicate matches are retrieved as suggestions without altering or deleting records."""
    service = RaidService(populated_db)
    matches = service.get_deduplication_matches()
    assert len(matches) == 1
    m = matches[0]
    assert m["new_raid_id"] == "raid-002"
    assert m["matched_raid_id"] == "raid-001"
    assert m["similarity_score"] == 0.78

    # Both drafts still exist independently in drafts table
    assert service.get_raid_by_id("raid-001") is not None
    assert service.get_raid_by_id("raid-002") is not None


# ---------------------------------------------------------------------------
# 6. Audit History
# ---------------------------------------------------------------------------

def test_12_audit_history_read_only(populated_db: DatabaseManager):
    """Audit entries are retrieved with valid action verbs, actor identity, and before/after JSON."""
    service = RaidService(populated_db)
    service.modify_raid("raid-001", modified_by="Auditor PM", updates={"workstream": "Security"})
    service.approve_raid("raid-001", approved_by="Lead PM")

    history = service.get_audit_history("raid-001")
    assert len(history) >= 2

    mod_entry = next((h for h in history if h["action"] == "MODIFIED"), None)
    assert mod_entry is not None
    assert mod_entry["action_by"] == "Auditor PM"
    prop_data = json.loads(mod_entry["proposed_data"])
    final_data = json.loads(mod_entry["final_data"])
    assert prop_data["workstream"] == "Integration"
    assert final_data["workstream"] == "Security"


# ---------------------------------------------------------------------------
# 7. Excel Export Suite (openpyxl)
# ---------------------------------------------------------------------------

def test_13_export_approved_register_valid_workbook(populated_db: DatabaseManager):
    """Generates valid Excel file for approved register with frozen header and correct columns."""
    service = RaidService(populated_db)
    service.approve_raid("raid-001", approved_by="Chief Architect", comments="SteerCo approval")

    approved_items = service.get_approved_items()
    excel_bytes = export_approved_register(approved_items)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 2000

    # Re-open with openpyxl to verify spreadsheet structure
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    ws = wb.active
    assert ws.title == "Approved RAID Register"
    assert ws.freeze_panes == "A2"
    assert ws.auto_filter.ref is not None

    # Verify header values
    headers = [cell.value for cell in ws[1]]
    assert "Approved ID" in headers
    assert "Draft ID" in headers
    assert "Title" in headers
    assert "Approved By" in headers
    assert "Evidence & Source Quote" in headers

    # Verify data row
    assert ws.max_row == 2
    row2 = [cell.value for cell in ws[2]]
    assert row2[headers.index("Draft ID")] == "raid-001"
    assert row2[headers.index("Approved By")] == "Chief Architect"


def test_14_export_review_queue_valid_workbook(populated_db: DatabaseManager):
    """Generates valid Excel file for review queue with numeric confidence and evidence citations."""
    service = RaidService(populated_db)
    drafts = service.get_pending_drafts()
    excel_bytes = export_review_queue(drafts)

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    ws = wb.active
    assert ws.title == "Pending Review Queue"
    assert ws.freeze_panes == "A2"

    headers = [cell.value for cell in ws[1]]
    assert "Draft ID" in headers
    assert "Confidence Score" in headers
    assert "Evidence Quote" in headers

    # Verify rows count (2 pending drafts)
    assert ws.max_row == 3
    conf_col = headers.index("Confidence Score") + 1
    assert isinstance(ws.cell(row=2, column=conf_col).value, (float, int))


def test_15_export_predictions_valid_workbook(populated_db: DatabaseManager):
    """Generates valid Excel file for predictive foresight insights with source IDs and severity."""
    service = RaidService(populated_db)
    preds = service.get_predictions()
    excel_bytes = export_predictions(preds)

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    ws = wb.active
    assert ws.title == "Predictive Insights"
    assert ws.max_row == 2

    headers = [cell.value for cell in ws[1]]
    assert "Prediction ID" in headers
    assert "Severity" in headers
    assert "Source Signal IDs" in headers


def test_16_export_audit_history_valid_workbook(populated_db: DatabaseManager):
    """Generates valid Excel file for audit history with proposed and final states."""
    service = RaidService(populated_db)
    service.modify_raid("raid-001", modified_by="Auditor", updates={"severity": "Critical"})
    history = service.get_audit_history()

    excel_bytes = export_audit_history(history)
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    ws = wb.active
    assert ws.title == "Audit Trail"
    assert ws.max_row >= 2

    headers = [cell.value for cell in ws[1]]
    assert "Audit ID" in headers
    assert "Action" in headers
    assert "Actor Identity" in headers
    assert "Prior State / Proposed Data" in headers


def test_17_export_combined_report_multi_tab(populated_db: DatabaseManager):
    """Generates master Excel workbook containing 4 distinct worksheets."""
    service = RaidService(populated_db)
    service.approve_raid("raid-001", approved_by="Director")

    excel_bytes = export_combined_report(
        approved_items=service.get_approved_items(),
        draft_items=service.get_pending_drafts(),
        predictions=service.get_predictions(),
        audit_entries=service.get_audit_history()
    )

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Approved Register" in sheet_names
    assert "Pending Review Queue" in sheet_names
    assert "Predictive Insights" in sheet_names
    assert "Audit Trail" in sheet_names
    assert len(sheet_names) == 4


def test_18_export_empty_datasets_handled_gracefully():
    """Exporting empty datasets produces valid, openable workbooks with headers and placeholder messages."""
    empty_bytes = export_combined_report([], [], [], [])
    assert isinstance(empty_bytes, bytes)
    wb = openpyxl.load_workbook(io.BytesIO(empty_bytes))
    for name in wb.sheetnames:
        ws = wb[name]
        assert ws.max_row == 2
        # Row 2 contains clean empty message
        first_cell = ws.cell(row=2, column=1).value
        assert "No" in str(first_cell)


def test_19_excel_export_neutralizes_formula_injection():
    """
    Verifies that Excel exports neutralize formula injection (CWE-1236).
    Inputs starting with '=', '+', '-', '@' are neutralized so Excel does not treat them as formulas.
    """
    malicious_items = [
        {
            "id": "=1+1",
            "type": "Risk",
            "title": "@SUM(1,2)",
            "description": "+cmd|'/C calc'!A0",
            "workstream": "-Finance",
            "phase": "Realize",
            "severity_or_impact": "High",
            "owner": "Hacker",
            "due_date": "2026-12-01",
            "mitigation_action": "=HYPERLINK(\"http://evil.com\")",
            "status": "Open",
            "approved_by": "=cmd",
            "approved_at": "2026-11-21T00:00:00Z",
            "evidence_links": "-evid"
        }
    ]
    excel_bytes = export_approved_register(malicious_items)
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    ws = wb.active
    # In openpyxl, formula cells have data_type 'f'. All our cells must have data_type 's' (string).
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=2, column=col)
        assert cell.data_type != "f", f"Cell at column {col} was exported as a formula: {cell.value}"

    # Verify that cells starting with malicious formula triggers were safely prefixed
    id_val = str(ws.cell(row=2, column=1).value)
    title_val = str(ws.cell(row=2, column=4).value)
    desc_val = str(ws.cell(row=2, column=5).value)
    assert id_val.startswith("'="), f"Expected '= prefix, got {id_val}"
    assert title_val.startswith("'@"), f"Expected '@ prefix, got {title_val}"
    assert desc_val.startswith("'+"), f"Expected '+ prefix, got {desc_val}"


def test_20_streamlit_apptest_all_pages_render_without_exception():
    """
    Verifies using Streamlit's official AppTest framework that all 9 Streamlit pages
    (Dashboard + 8 subpages including Data Import & Sync and Meeting Minutes Import) execute and render
    their initial layouts without unhandled exceptions or crashes.
    """
    import glob
    from streamlit.testing.v1 import AppTest

    project_root = Path(__file__).resolve().parent.parent
    scripts = [project_root / "ui/dashboard.py"] + [
        p for p in sorted(project_root.glob("ui/pages/*.py"))
        if p.name != "__init__.py" and not p.name.startswith(".")
    ]
    assert len(scripts) == 9, f"Expected 9 scripts, found {len(scripts)}"

    for script_path in scripts:
        at = AppTest.from_file(str(script_path))
        at.run()
        assert not at.exception, f"Streamlit page {script_path.name} raised an uncaught exception: {at.exception}"


def test_21_controlled_database_counts_and_filters(temp_db: DatabaseManager):
    """
    Verifies data accuracy against a tightly controlled SQLite database with exact known rows:
    - 5 drafts across different statuses: 2 Draft, 1 Modified, 1 Accepted, 1 Rejected
    - 2 predictions
    - 1 duplicate match
    Validates exact KPI counts and filter segregation.
    """
    service = RaidService(temp_db)

    def _make_draft(draft_id, raid_type, title, workstream, phase, status, conf=0.85):
        return {
            "id": draft_id, "type": raid_type, "title": title, "description": f"{title} desc",
            "workstream": workstream, "phase": phase, "probability": "Medium", "impact": "Medium",
            "severity": "Medium", "scoring_reason": "Rule check", "likely_owner": "Owner",
            "due_date": "2026-12-01", "mitigation_action": "Mitigate", "confidence_score": conf,
            "evidence_text": "Quote", "evidence_source_id": "s1", "evidence_source_type": "email",
            "detection_source": "llm", "duplicate_of_id": None, "status": status
        }

    # 1. Draft 1 (Draft, Risk, FI, Explore)
    temp_db.save_raid_draft(_make_draft("c-001", "Risk", "FI Risk", "FI", "Explore", "Draft", 0.85))
    # 2. Draft 2 (Draft, Issue, SD, Realize)
    temp_db.save_raid_draft(_make_draft("c-002", "Issue", "SD Issue", "SD", "Realize", "Draft", 0.90))
    # 3. Draft 3 (Modified, Risk, FI, Realize)
    temp_db.save_raid_draft(_make_draft("c-003", "Risk", "FI Modified", "FI", "Realize", "Modified", 0.80))
    # 4. Draft 4 (Accepted, Decision, MM, Prepare)
    temp_db.save_raid_draft(_make_draft("c-004", "Decision", "MM Accepted", "MM", "Prepare", "Accepted", 0.95))
    with temp_db.get_connection() as conn:
        conn.cursor().execute("""
            INSERT INTO approved_raid (id, draft_id, type, title, description, workstream, phase, approved_by)
            VALUES ('appr-c004', 'c-004', 'Decision', 'MM Accepted', 'MM desc', 'MM', 'Prepare', 'PM Lead')
        """)
        conn.commit()

    # 5. Draft 5 (Rejected, Assumption, Basis, Realize)
    temp_db.save_raid_draft(_make_draft("c-005", "Assumption", "Basis Rejected", "Basis", "Realize", "Rejected", 0.60))

    # Predictions (2)
    temp_db.save_prediction(PredictionItem(
        prediction_id="cp-1", prediction_type=PredictionType.TASK_OVERDUE,
        title="P1", description="D1", suggested_raid_category="Risk",
        confidence=0.88, severity="High", evidence="ev1", source_signal_ids=["s1"]
    ))
    temp_db.save_prediction(PredictionItem(
        prediction_id="cp-2", prediction_type=PredictionType.DEFECT_CLUSTER,
        title="P2", description="D2", suggested_raid_category="Issue",
        confidence=0.91, severity="Critical", evidence="ev2", source_signal_ids=["s2"]
    ))

    # Duplicate match (1)
    temp_db.save_deduplication_match(DeduplicationResult(
        new_raid_id="c-001", matched_raid_id="c-003", similarity_score=0.82,
        match_status=MatchStatus.LIKELY_DUPLICATE, match_reason="Overlap",
        same_category=True, same_workstream=True, requires_human_review=True
    ))

    # 1. Check exact KPI counts
    kpis = service.get_kpis()
    assert kpis["total_drafts"] == 5
    assert kpis["pending_reviews"] == 3  # c-001, c-002, c-003 (Draft + Modified)
    assert kpis["approved_items"] == 1   # c-004
    assert kpis["rejected_drafts"] == 1  # c-005
    assert kpis["active_predictions"] == 2
    assert kpis["potential_duplicates"] == 1

    # 2. Check pending review queue filtering
    pending = service.get_pending_drafts()
    pending_ids = [d["id"] for d in pending]
    assert sorted(pending_ids) == ["c-001", "c-002", "c-003"]
    assert "c-004" not in pending_ids  # Accepted excluded
    assert "c-005" not in pending_ids  # Rejected excluded

    # 3. Check approved register exclusivity
    approved = service.get_approved_items()
    assert len(approved) == 1
    assert approved[0]["draft_id"] == "c-004"
    assert approved[0]["title"] == "MM Accepted"


def test_22_raid_review_template_table_segregation():
    """
    Verifies the 4 separate tables strictly conform to the Suggested RAID Item Template
    from the Problem Statement PDF:
    - R – Risk includes Probability and Impact, but not Severity
    - A – Assumption excludes scoring and action fields (no confusion with Action)
    - I – Issue includes Severity, but not Probability
    - D – Decision excludes Probability, Impact, Severity
    - Dependency is excluded from review categories
    - Business statuses correctly map: Draft, Open, In Progress, Closed, Rejected
    """
    import importlib
    page1 = importlib.import_module("ui.pages.1_RAID_Review")
    map_business_status = page1.map_business_status
    build_risk_table_data = page1.build_risk_table_data
    build_assumption_table_data = page1.build_assumption_table_data
    build_issue_table_data = page1.build_issue_table_data
    build_decision_table_data = page1.build_decision_table_data
    build_dependency_table_data = getattr(page1, "build_dependency_table_data", None)
    RAID_CATEGORIES = page1.RAID_CATEGORIES

    # 1. Dependency is supported across all 5 RAID categories
    assert "Dependency" in RAID_CATEGORIES
    assert RAID_CATEGORIES == ["Risk", "Assumption", "Issue", "Decision", "Dependency"]

    # 2. Status mapping
    assert map_business_status("Draft") == "Draft"
    assert map_business_status("Pending_Review") == "Draft"
    assert map_business_status("Modified") == "Draft (Modified)"
    assert map_business_status("Accepted") == "Open"
    assert map_business_status("Approved") == "Open"
    assert map_business_status("Rejected") == "Rejected"
    assert map_business_status("In Progress") == "In Progress"
    assert map_business_status("Closed") == "Closed"

    # Sample draft
    mock_draft = {
        "id": "raid-001",
        "type": "Risk",
        "title": "Data Migration Delay",
        "description": "Legacy cutover delay",
        "workstream": "FI",
        "phase": "Realize",
        "probability": "High",
        "impact": "High",
        "severity": "Critical",
        "likely_owner": "FI Lead",
        "due_date": "2026-12-15",
        "mitigation_action": "Add parallel extraction",
        "confidence_score": 0.95,
        "status": "Draft",
        "detected_at": "2026-11-20 10:00:00"
    }

    # 3. R – Risk Table
    df_risk = build_risk_table_data([mock_draft])
    assert "Probability" in df_risk.columns
    assert "Impact" in df_risk.columns
    assert "Severity" not in df_risk.columns
    assert "Mitigation / Next Action" in df_risk.columns

    # 4. A – Assumption Table (must not confuse Assumption with Action)
    df_assumption = build_assumption_table_data([mock_draft])
    assert "Probability" not in df_assumption.columns
    assert "Impact" not in df_assumption.columns
    assert "Severity" not in df_assumption.columns
    assert "Mitigation / Next Action" not in df_assumption.columns
    assert "Description" in df_assumption.columns

    # 5. I – Issue Table
    df_issue = build_issue_table_data([mock_draft])
    assert "Severity" in df_issue.columns
    assert "Probability" not in df_issue.columns
    assert "Mitigation / Next Action" in df_issue.columns

    # 6. D – Decision Table
    df_decision = build_decision_table_data([mock_draft])
    assert "Decided By / Owner" in df_decision.columns
    assert "Probability" not in df_decision.columns
    assert "Impact" not in df_decision.columns
    assert "Severity" not in df_decision.columns
    assert "Mitigation / Next Action" not in df_decision.columns


# ---------------------------------------------------------------------------
# 10. Interactive Duplicate Review Workflows
# ---------------------------------------------------------------------------

def _create_full_draft(draft_id: str, raid_type: str, title: str, description: str, workstream: str = "FI", phase: str = "Realize", status: str = "Draft", evidence_text: str = "Evidence quote", evidence_source_id: str = "sig-01", evidence_source_type: str = "email", duplicate_of_id=None):
    return {
        "id": draft_id,
        "type": raid_type,
        "title": title,
        "description": description,
        "workstream": workstream,
        "phase": phase,
        "probability": "Medium",
        "impact": "Medium",
        "severity": "Medium",
        "scoring_reason": "Matching check",
        "likely_owner": "Owner Lead",
        "due_date": "2026-12-31",
        "mitigation_action": "Mitigation steps",
        "confidence_score": 0.88,
        "evidence_text": evidence_text,
        "evidence_source_id": evidence_source_id,
        "evidence_source_type": evidence_source_type,
        "detection_source": "llm",
        "duplicate_of_id": duplicate_of_id,
        "status": status
    }


def test_23_duplicate_review_filters_match_status_and_keywords(temp_db: DatabaseManager):
    """
    Verifies match-status and comma-separated keyword filtering for the duplicate review workflow.
    """
    import importlib
    page4 = importlib.import_module("ui.pages.4_Duplicate_Review")
    filter_matches_by_status = page4.filter_matches_by_status
    filter_matches_by_keywords = page4.filter_matches_by_keywords
    MATCH_STATUS_OPTIONS = page4.MATCH_STATUS_OPTIONS

    service = RaidService(temp_db)

    # Setup drafts
    d_exist = _create_full_draft("ex-01", "Risk", "FI Cutover Timeline Risk", "Risk of month-end reconciliation delay", workstream="FI", evidence_text="Email about cutover")
    d_cand1 = _create_full_draft("cd-01", "Risk", "FI Month-End Cutover Schedule", "Reconciliation delay during wave 1 cutover", workstream="FI", evidence_text="Meeting notes cutover", duplicate_of_id="ex-01")
    d_cand2 = _create_full_draft("cd-02", "Issue", "Basis Transport Freeze Failure", "Transport imports failing in QA", workstream="Basis", evidence_text="Defect log freeze", duplicate_of_id="ex-01")
    temp_db.save_raid_draft(d_exist)
    temp_db.save_raid_draft(d_cand1)
    temp_db.save_raid_draft(d_cand2)

    matches = [
        {"new_raid_id": "cd-01", "matched_raid_id": "ex-01", "similarity_score": 0.85, "match_status": "LIKELY_DUPLICATE", "match_reason": "High text overlap"},
        {"new_raid_id": "cd-02", "matched_raid_id": "ex-01", "similarity_score": 0.55, "match_status": "POSSIBLE_DUPLICATE", "match_reason": "Moderate overlap"},
        {"new_raid_id": "cd-03", "matched_raid_id": "ex-01", "similarity_score": 0.40, "match_status": "RELATED_NOT_DUPLICATE", "match_reason": "Different category"}
    ]

    # 1. Match Status Filter
    assert len(filter_matches_by_status(matches, "All Matches")) == 3
    likely = filter_matches_by_status(matches, "Likely Duplicate")
    assert len(likely) == 1 and likely[0]["match_status"] == "LIKELY_DUPLICATE"
    possible = filter_matches_by_status(matches, "Possible Duplicate")
    assert len(possible) == 1 and possible[0]["match_status"] == "POSSIBLE_DUPLICATE"
    related = filter_matches_by_status(matches, "Related – Not Duplicate")
    assert len(related) == 1 and related[0]["match_status"] == "RELATED_NOT_DUPLICATE"

    # 2. Multi-Keyword Filter (comma-separated)
    # Search across title / description / workstream / evidence
    kw_res1 = filter_matches_by_keywords(matches, "cutover, reconciliation", service)
    assert len(kw_res1) >= 1
    assert any(m["new_raid_id"] == "cd-01" for m in kw_res1)

    kw_res2 = filter_matches_by_keywords(matches, "Basis, freeze", service)
    assert len(kw_res2) == 1
    assert kw_res2[0]["new_raid_id"] == "cd-02"

    # No match query
    kw_res3 = filter_matches_by_keywords(matches, "nonexistent_token_xyz", service)
    assert len(kw_res3) == 0

    # Empty query returns all
    assert len(filter_matches_by_keywords(matches, "", service)) == 3
    assert len(filter_matches_by_keywords(matches, "   ", service)) == 3


def test_24_duplicate_review_active_queue_filtering(temp_db: DatabaseManager):
    """
    Verifies that rejected and approved candidate items are excluded from the active duplicate review queue,
    but remain accessible when include_resolved=True.
    """
    import importlib
    page4 = importlib.import_module("ui.pages.4_Duplicate_Review")
    filter_active_pending_matches = page4.filter_active_pending_matches

    service = RaidService(temp_db)

    # 3 candidates: 1 Draft, 1 Accepted, 1 Rejected
    temp_db.save_raid_draft(_create_full_draft("cand-active", "Risk", "Active Cand", "Desc", status="Draft"))
    temp_db.save_raid_draft(_create_full_draft("cand-acc", "Risk", "Accepted Cand", "Desc", status="Accepted"))
    temp_db.save_raid_draft(_create_full_draft("cand-rej", "Risk", "Rejected Cand", "Desc", status="Rejected"))

    matches = [
        {"new_raid_id": "cand-active", "matched_raid_id": "exist-1", "similarity_score": 0.8},
        {"new_raid_id": "cand-acc", "matched_raid_id": "exist-1", "similarity_score": 0.85},
        {"new_raid_id": "cand-rej", "matched_raid_id": "exist-1", "similarity_score": 0.75}
    ]

    # Active review queue (default: include_resolved=False)
    active_only = filter_active_pending_matches(matches, service, include_resolved=False)
    assert len(active_only) == 1
    assert active_only[0]["new_raid_id"] == "cand-active"

    # When include_resolved=True, all 3 are returned
    all_with_resolved = filter_active_pending_matches(matches, service, include_resolved=True)
    assert len(all_with_resolved) == 3


def test_25_duplicate_review_action_approve_separate_raid(temp_db: DatabaseManager):
    """
    Action 1: Approving candidate as a separate RAID:
    - Atomically moves candidate to approved register
    - Leaves existing RAID item completely untouched
    - Adds audit entry for candidate approval
    - Candidate appears in approved register
    """
    service = RaidService(temp_db)

    temp_db.save_raid_draft(_create_full_draft("exist-item-1", "Risk", "Existing Scope Item", "Original", workstream="SD", phase="Explore", status="Draft"))
    temp_db.save_raid_draft(_create_full_draft("cand-item-1", "Risk", "Candidate Scope Item", "New distinct scope", workstream="SD", phase="Explore", duplicate_of_id="exist-item-1", status="Draft"))

    # Execute Approval of Candidate ONLY
    approved_cand = service.approve_raid(
        raid_id="cand-item-1",
        approved_by="Auditor PM Lead",
        comments="Approved as distinct scope item after duplicate comparison"
    )
    assert approved_cand["draft_id"] == "cand-item-1"
    assert approved_cand["approved_by"] == "Auditor PM Lead"

    # Candidate status in drafts is Accepted
    cand_draft = temp_db.get_raid_draft_by_id("cand-item-1")
    assert cand_draft["status"] == "Accepted"

    # Existing item is untouched: still Draft, NOT approved
    exist_draft = temp_db.get_raid_draft_by_id("exist-item-1")
    assert exist_draft["status"] == "Draft"
    assert temp_db.get_approved_item_by_draft_id("exist-item-1") is None

    # Approved register has candidate
    approved_items = service.get_approved_items()
    assert any(a["draft_id"] == "cand-item-1" for a in approved_items)
    assert not any(a["draft_id"] == "exist-item-1" for a in approved_items)

    # Audit history recorded for candidate
    cand_audits = service.get_audit_history("cand-item-1")
    assert any(a["action"] == "APPROVED" and a["action_by"] == "Auditor PM Lead" for a in cand_audits)


def test_26_duplicate_review_action_reject_duplicate(temp_db: DatabaseManager):
    """
    Action 2: Rejecting candidate as a duplicate:
    - Candidate marked as Rejected with mandatory reason
    - Existing item is untouched
    - Candidate is NOT physically deleted from database
    - Candidate source evidence and audit history are preserved
    - Candidate removed from active review queue
    """
    service = RaidService(temp_db)

    temp_db.save_raid_draft(_create_full_draft("exist-item-2", "Issue", "Existing Defect", "Original issue", workstream="MM", phase="Realize", status="Draft"))
    temp_db.save_raid_draft(_create_full_draft("cand-item-2", "Issue", "Duplicate Defect Notice", "Duplicate issue text", workstream="MM", phase="Realize", duplicate_of_id="exist-item-2", status="Draft"))

    # Validation: rejection requires mandatory reason
    with pytest.raises(RaidValidationError):
        service.reject_raid("cand-item-2", rejected_by="PM Lead", reason="")
    with pytest.raises(RaidValidationError):
        service.reject_raid("cand-item-2", rejected_by="PM Lead", reason="   ")

    # Execute Rejection of Candidate ONLY
    service.reject_raid(
        raid_id="cand-item-2",
        rejected_by="PM Lead",
        reason="Redundant duplicate of existing item exist-item-2"
    )

    # 1. Candidate is marked as Rejected
    cand_draft = temp_db.get_raid_draft_by_id("cand-item-2")
    assert cand_draft is not None  # NOT deleted
    assert cand_draft["status"] == "Rejected"

    # 2. Source evidence preserved
    assert cand_draft["evidence_text"] is not None
    assert cand_draft["evidence_source_id"] == "sig-01"

    # 3. Existing item completely untouched
    exist_draft = temp_db.get_raid_draft_by_id("exist-item-2")
    assert exist_draft["status"] == "Draft"

    # 4. Excluded from pending review drafts
    pending_ids = [d["id"] for d in service.get_pending_drafts()]
    assert "cand-item-2" not in pending_ids
    assert "exist-item-2" in pending_ids

    # 5. Audit trail records REJECTED with reason and actor
    audits = service.get_audit_history("cand-item-2")
    assert any(a["action"] == "REJECTED" and "Redundant duplicate of existing item exist-item-2" in a["action_notes"] for a in audits)


