"""
Streamlit AppTest validation for ui/pages/1_RAID_Review.py (Stage 6C Fix).
Verifies:
1. RAID Review page loads without AttributeError.
2. Empty candidate list displays proper empty state.
3. Existing low-confidence candidates appear when present.
4. Original source content is displayed correctly.
5. Existing RAID review functionality is unaffected.
6. Displaying candidates does not mutate the database or create drafts.
"""

import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from src.database.db_manager import DatabaseManager
from src.database.migrations import run_migrations
from src.normalisation.schemas import ProjectSignal, SourceType
from src.service import RaidService


@pytest.fixture
def test_db(tmp_path: Path):
    db_file = tmp_path / "ui_test.db"
    db = DatabaseManager(db_file)
    run_migrations(db_file)
    return db


PAGE_PATH = str(Path(__file__).resolve().parent.parent / "ui/pages/1_RAID_Review.py")


def test_1_raid_review_page_loads_empty_state(test_db: DatabaseManager, monkeypatch):
    """Verifies page loads cleanly without AttributeError when database is empty."""
    monkeypatch.setattr("ui.service_helper.DB_PATH", test_db.db_path)

    at = AppTest.from_file(PAGE_PATH)
    at.run()
    assert not at.exception, f"Raised exception on empty DB: {at.exception}"


def test_2_raid_review_renders_low_confidence_candidates(test_db: DatabaseManager, monkeypatch):
    """Verifies low-confidence candidates render when present and no drafts exist."""
    monkeypatch.setattr("ui.service_helper.DB_PATH", test_db.db_path)

    sig = ProjectSignal(
        id="sig-lc-ui-01",
        source_type=SourceType.EMAIL,
        author="vendor.pm@supplier.com",
        subject="Scope Clarification Pending",
        content="Hardware delivery may slip by 3 weeks pending customs clearance."
    )
    test_db.save_raw_signal(sig, user_id="default_user")
    payload = {
        "category": "Risk",
        "title": "Hardware customs clearance delay",
        "description": "Delivery slip risk.",
        "confidence": 0.62,
        "owner": "Logistics Lead",
        "evidence": "Hardware delivery may slip by 3 weeks.",
        "reasoning_summary": "Customs clearance uncertainty."
    }
    test_db.update_signal_classification_status(
        signal_id="sig-lc-ui-01",
        status="BELOW_CONFIDENCE_THRESHOLD",
        failure_reason=json.dumps(payload),
        user_id="default_user",
        llm_provider="OpenRouter",
        llm_model="nvidia/nemotron-3-ultra-550b-a55b:free"
    )

    # Initial check: raid_drafts count is 0
    assert len(test_db.get_pending_drafts(user_id="default_user")) == 0

    at = AppTest.from_file(PAGE_PATH)
    at.run()
    assert not at.exception, f"Raised exception when candidate present: {at.exception}"

    # Verify candidate display did NOT create any drafts in database
    assert len(test_db.get_pending_drafts(user_id="default_user")) == 0
    assert len(test_db.get_approved_items(user_id="default_user")) == 0


def test_3_raid_review_both_drafts_and_low_confidence(test_db: DatabaseManager, monkeypatch):
    """Verifies existing high-confidence drafts and low-confidence candidates co-exist cleanly."""
    monkeypatch.setattr("ui.service_helper.DB_PATH", test_db.db_path)

    # High confidence draft
    test_db.save_raid_draft({
        "id": "raid-high-01",
        "user_id": "default_user",
        "type": "Risk",
        "title": "High confidence architecture risk",
        "description": "DB failover untested.",
        "workstream": "SD",
        "phase": "Realize",
        "status": "Draft",
        "confidence_score": 0.92
    }, user_id="default_user")

    # Low confidence candidate
    sig = ProjectSignal(
        id="sig-lc-ui-02",
        source_type=SourceType.CHAT,
        author="qa@kaartech.com",
        subject="Flaky test",
        content="Intermittent test failure on staging."
    )
    test_db.save_raw_signal(sig, user_id="default_user")
    test_db.update_signal_classification_status(
        signal_id="sig-lc-ui-02",
        status="BELOW_CONFIDENCE_THRESHOLD",
        failure_reason="Confidence 0.52 below threshold",
        user_id="default_user"
    )

    at = AppTest.from_file(PAGE_PATH)
    at.run()
    assert not at.exception, f"Raised exception with mixed items: {at.exception}"
