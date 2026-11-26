"""
Unit tests for Semantic Deduplication.
Covers:
1. Exact duplicate detection.
2. Semantically similar RAID descriptions.
3. Different issues are not marked as duplicates.
4. Different RAID categories are handled appropriately (Risk vs Issue).
5. Workstream context is considered (Finance vs Procurement).
6. Low similarity returns NO_MATCH.
7. Configurable thresholds work.
8. Empty or missing text is handled safely.
9. Existing records remain unchanged.
10. Duplicate matches are not repeatedly created.
"""

import pytest
import copy

from src.deduplication.schemas import MatchStatus, DeduplicationResult
from src.deduplication.similarity import calculate_text_similarity, evaluate_similarity
from src.deduplication.deduplicator import SemanticDeduplicator


def test_1_exact_duplicate_detection():
    """Identical title and description within same category and workstream is flagged as LIKELY_DUPLICATE."""
    candidate = {
        "id": "draft-101",
        "type": "Risk",
        "title": "Data migration delay in MM module",
        "description": "Legacy material master extraction is running behind schedule.",
        "workstream": "MM"
    }
    existing = [{
        "id": "approved-001",
        "type": "Risk",
        "title": "Data migration delay in MM module",
        "description": "Legacy material master extraction is running behind schedule.",
        "workstream": "MM"
    }]

    deduper = SemanticDeduplicator()
    result = deduper.check_duplicate(candidate, existing)

    assert result.match_status == MatchStatus.LIKELY_DUPLICATE
    assert result.similarity_score >= 0.95
    assert result.matched_raid_id == "approved-001"
    assert result.same_category is True
    assert result.same_workstream is True


def test_2_semantically_similar_descriptions():
    """Different phrasing conveying the same underlying problem is detected as LIKELY_DUPLICATE."""
    existing = [{
        "id": "raid-exist-1",
        "type": "Issue",
        "title": "Integration testing is delayed because client master data is pending.",
        "description": "SIT cycle 1 cannot start due to incomplete customer master records from legacy ERP.",
        "workstream": "Testing"
    }]
    candidate = {
        "id": "raid-cand-1",
        "type": "Issue",
        "title": "SIT cannot proceed due to missing customer master records.",
        "description": "Integration testing blocked as client master data is not yet delivered.",
        "workstream": "Testing"
    }

    deduper = SemanticDeduplicator(duplicate_threshold=0.70)
    result = deduper.check_duplicate(candidate, existing)

    assert result.match_status == MatchStatus.LIKELY_DUPLICATE
    assert result.matched_raid_id == "raid-exist-1"
    assert result.similarity_score >= 0.70


def test_3_different_issues_not_marked_as_duplicates():
    """Different operational issues within the same stream return NO_MATCH."""
    existing = [{
        "id": "raid-exist-2",
        "type": "Issue",
        "title": "Purchase order approval workflow is failing.",
        "description": "Workflow engine throws an exception when approving multi-level purchase orders.",
        "workstream": "MM"
    }]
    candidate = {
        "id": "raid-cand-2",
        "type": "Issue",
        "title": "Material valuation is incorrect during goods receipt.",
        "description": "Price variance account is not derived properly during MIGO 101 posting.",
        "workstream": "MM"
    }

    deduper = SemanticDeduplicator()
    result = deduper.check_duplicate(candidate, existing)

    assert result.match_status == MatchStatus.NO_MATCH
    assert result.similarity_score < 0.50
    assert result.matched_raid_id is None


def test_4_different_raid_categories_handled_appropriately():
    """Risk vs Issue with identical topic is classified as RELATED_NOT_DUPLICATE."""
    existing = [{
        "id": "raid-risk-1",
        "type": "Risk",
        "title": "The Basis team may not complete the system refresh on time.",
        "description": "Hardware resource constraints may cause the sandbox refresh to slip.",
        "workstream": "Basis"
    }]
    candidate = {
        "id": "raid-issue-1",
        "type": "Issue",
        "title": "The system refresh has not been completed.",
        "description": "The Basis team has not completed the sandbox refresh, blocking configuration.",
        "workstream": "Basis"
    }

    deduper = SemanticDeduplicator()
    result = deduper.check_duplicate(candidate, existing)

    assert result.match_status == MatchStatus.RELATED_NOT_DUPLICATE
    assert result.same_category is False
    assert "differing categories" in result.match_reason.lower()
    assert result.requires_human_review is True


def test_5_workstream_context_is_considered():
    """Identical problem description in different workstreams is flagged as RELATED_NOT_DUPLICATE."""
    existing = [{
        "id": "raid-fi-1",
        "type": "Issue",
        "title": "Testing blocked due to missing master data",
        "description": "Testing blocked due to missing master data for chart of accounts validation.",
        "workstream": "FI"
    }]
    candidate = {
        "id": "raid-mm-1",
        "type": "Issue",
        "title": "Testing blocked due to missing master data",
        "description": "Testing blocked due to missing master data for vendor master records.",
        "workstream": "MM"
    }

    deduper = SemanticDeduplicator()
    result = deduper.check_duplicate(candidate, existing)

    assert result.match_status == MatchStatus.RELATED_NOT_DUPLICATE
    assert result.same_workstream is False
    assert "differing workstreams" in result.match_reason.lower()


def test_6_low_similarity_returns_no_match():
    """Candidate with low text similarity returns NO_MATCH and requires_human_review=False."""
    existing = [{
        "id": "raid-exist-3",
        "type": "Risk",
        "title": "User training rooms not booked",
        "description": "Training facilities have not been reserved for end user training.",
        "workstream": "Change Management"
    }]
    candidate = {
        "id": "raid-cand-3",
        "type": "Risk",
        "title": "SAP router connection timeout",
        "description": "Network latency causing OSS connection dropouts.",
        "workstream": "Basis"
    }

    deduper = SemanticDeduplicator(review_threshold=0.50)
    result = deduper.check_duplicate(candidate, existing)

    assert result.match_status == MatchStatus.NO_MATCH
    assert result.similarity_score < 0.20
    assert result.requires_human_review is False


def test_7_configurable_thresholds_work():
    """Changing similarity thresholds directly alters matching behavior."""
    cand = {"title": "Vendor master data migration delay", "description": "Delay in supplier load."}
    exist = {"title": "Customer master data migration delay", "description": "Delay in customer load."}

    # Strict threshold: duplicate_threshold = 0.95
    strict_res = evaluate_similarity(cand, exist, duplicate_threshold=0.95, review_threshold=0.50)
    assert strict_res.match_status != MatchStatus.LIKELY_DUPLICATE

    # Relaxed threshold: duplicate_threshold = 0.55
    relaxed_res = evaluate_similarity(cand, exist, duplicate_threshold=0.55, review_threshold=0.40)
    assert relaxed_res.match_status == MatchStatus.LIKELY_DUPLICATE


def test_8_empty_or_missing_text_handled_safely():
    """None or empty text strings return 0.0 similarity without throwing exceptions."""
    score_empty = calculate_text_similarity("", "")
    assert score_empty == 0.0

    score_none = calculate_text_similarity(None, "Valid text")
    assert score_none == 0.0

    res = evaluate_similarity({"title": None, "description": ""}, {"title": "", "description": ""})
    assert res.similarity_score == 0.0
    assert res.match_status == MatchStatus.NO_MATCH


def test_9_existing_records_remain_unchanged():
    """Verifies that deduplication is strictly a read-only operation and does not mutate existing records."""
    existing_original = [{
        "id": "raid-const-1",
        "type": "Risk",
        "title": "Key resource availability",
        "description": "Architect may be allocated to another conversion project.",
        "workstream": "General",
        "status": "Approved"
    }]
    existing_copy = copy.deepcopy(existing_original)

    candidate = {
        "id": "raid-cand-9",
        "type": "Risk",
        "title": "Key resource availability",
        "description": "Architect unavailable.",
        "workstream": "General"
    }

    deduper = SemanticDeduplicator()
    deduper.check_duplicate(candidate, existing_copy)

    # Existing records must be identical to original deepcopy
    assert existing_copy == existing_original


def test_10_duplicate_matches_not_repeatedly_created():
    """Repeated calls for the same pair do not generate redundant match audit entries."""
    deduper = SemanticDeduplicator()
    candidate = {
        "id": "cand-dup-1",
        "type": "Risk",
        "title": "Material master validation failure",
        "description": "Validation failure during goods movement.",
        "workstream": "MM"
    }
    existing = [{
        "id": "exist-dup-1",
        "type": "Risk",
        "title": "Material master validation failure",
        "description": "Validation failure during goods movement.",
        "workstream": "MM"
    }]

    first_matches = deduper.find_all_matches(candidate, existing)
    assert len(first_matches) == 1

    # Second call for the same pair
    second_matches = deduper.find_all_matches(candidate, existing)
    assert len(second_matches) == 0



def test_11_stable_similarity_across_repeated_evaluations():
    """Confirms that repeated vectorisation and scoring produces deterministic, stable results."""
    t1 = "Configuration testing blocked due to missing pricing procedure schema"
    t2 = "Pricing procedure schema missing in SD, preventing configuration tests"

    score1 = calculate_text_similarity(t1, t2)
    score2 = calculate_text_similarity(t1, t2)
    score3 = calculate_text_similarity(t1, t2)

    assert score1 == score2 == score3
    assert score1 > 0.45


def test_12_risk_to_issue_progression_relationship():
    """Risk-to-Issue evolution is tagged as RELATED_NOT_DUPLICATE, maintaining audit trace."""
    risk_item = {
        "id": "raid-r-50",
        "type": "Risk",
        "title": "Data load timeout on BAPI_MATERIAL_SAVEDATA",
        "description": "Mass material creation BAPI may time out during initial migration cutover.",
        "workstream": "MM"
    }
    issue_item = {
        "id": "raid-i-50",
        "type": "Issue",
        "title": "Data load timeout on BAPI_MATERIAL_SAVEDATA",
        "description": "Mass material creation BAPI is timing out during mock cutover load.",
        "workstream": "MM"
    }

    deduper = SemanticDeduplicator()
    result = deduper.check_duplicate(issue_item, [risk_item])

    assert result.match_status == MatchStatus.RELATED_NOT_DUPLICATE
    assert result.same_category is False
    assert result.requires_human_review is True
