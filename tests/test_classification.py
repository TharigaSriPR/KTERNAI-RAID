"""
Unit tests for Structured Extraction & LLM-Based RAID Classification.
Covers all 16 required test scenarios using mocked LLM responses without calling live APIs:
1. Correct Risk classification.
2. Correct Assumption classification.
3. Correct Issue classification.
4. Correct Decision classification.
5. Correct Dependency classification.
6. Correct NOT_RAID classification.
7. Pydantic validation.
8. Confidence values outside permitted range.
9. Missing owner and due date remain None.
10. Invalid JSON response handling.
11. API timeout handling.
12. Retry behavior with transient errors.
13. Missing API key handling.
14. Long input handling.
15. Evidence and source ID preservation.
16. No fabricated information in extracted fields.
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.normalisation.schemas import ProjectSignal, SourceType
from src.classification.schemas import RAIDCategory, RAIDExtractionResult
from src.classification.classifier import RAIDClassifier
from src.classification.providers.mock_provider import MockLLMProvider
from src.classification.providers.gemini_provider import GeminiProvider


class TestRAIDClassificationCategories:
    def test_correct_risk_classification(self):
        """Scenario 1: Future uncertain event with potential negative project impact."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.EMAIL,
            author="Jane",
            subject="Master data status",
            content="The client may not provide the required master data before the integration testing phase."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.category == RAIDCategory.RISK
        assert result.is_raid is True
        assert "Delay in integration testing" in result.title
        assert result.source_signal_id == signal.id
        assert result.confidence >= 0.75
        assert "may not provide the required master data" in result.evidence

    def test_correct_assumption_classification(self):
        """Scenario 2: Unverified planning assumption accepted without proof."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.MEETING_NOTES,
            author="Scribe",
            subject="SIT Kickoff Planning",
            content="We are assuming that the client will complete all master data preparation before the SIT phase."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.category == RAIDCategory.ASSUMPTION
        assert result.is_raid is True
        assert "master data" in result.title.lower()
        assert result.source_signal_id == signal.id

    def test_correct_issue_classification(self):
        """Scenario 3: An active problem or blocker that has already occurred."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.CHAT,
            author="Mike",
            subject="Integration Testing",
            content="Integration testing is currently blocked because the client has not provided master data."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.category == RAIDCategory.ISSUE
        assert result.is_raid is True
        assert "blocked" in result.title.lower()
        assert result.source_signal_id == signal.id

    def test_correct_decision_classification(self):
        """Scenario 4: An agreed choice or concluded decision made by stakeholders."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.EMAIL,
            author="SteerCo",
            subject="Steering Committee Outcomes",
            content="The project team has agreed to move the SIT start date to 15 December."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.category == RAIDCategory.DECISION
        assert result.is_raid is True
        assert "15 December" in result.title
        assert result.source_signal_id == signal.id

    def test_correct_dependency_classification(self):
        """Scenario 5: Progress depends on an external team or system deliverable."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.CHAT,
            author="Tester",
            subject="Config Test Prerequisite",
            content="Configuration testing can begin only after the Basis team completes the system refresh."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.category == RAIDCategory.DEPENDENCY
        assert result.is_raid is True
        assert "refresh" in result.title.lower()
        assert result.owner == "Basis Team"
        assert result.source_signal_id == signal.id

    def test_correct_not_raid_classification(self):
        """Scenario 6: Routine communication without RAID relevance."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.EMAIL,
            author="Alex",
            subject="MOM Share",
            content="Thanks for sharing the meeting minutes."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.category == RAIDCategory.NOT_RAID
        assert result.is_raid is False
        assert result.title is None
        assert result.description is None


class TestPydanticValidationAndConstraints:
    def test_pydantic_schema_validation(self):
        """Scenario 7: Full valid schema instantiation and serialization."""
        res = RAIDExtractionResult(
            category=RAIDCategory.RISK,
            title="Custom Code Remediation Overrun",
            description="Remediation for Z-objects running behind schedule.",
            impact="SIT timeline delay",
            owner="ABAP Lead",
            due_date="2026-11-30",
            workstream="ABAP",
            phase="Realize",
            confidence=0.88,
            evidence="Custom code remediation is 40% unanalysed",
            source_signal_id="sig-test-001",
            reasoning_summary="Identified high potential for project slip"
        )
        assert res.category == "Risk"
        assert res.confidence == 0.88
        dump = res.model_dump()
        assert dump["category"] == "Risk"
        assert dump["owner"] == "ABAP Lead"

    def test_confidence_values_outside_permitted_range(self):
        """Scenario 8: Confidence values outside 0.0 to 1.0 must be rejected."""
        with pytest.raises(ValidationError):
            RAIDExtractionResult(
                category=RAIDCategory.RISK,
                title="Test Risk",
                confidence=1.5,  # Invalid: > 1.0
                source_signal_id="sig-001"
            )

        with pytest.raises(ValidationError):
            RAIDExtractionResult(
                category=RAIDCategory.RISK,
                title="Test Risk",
                confidence=-0.1,  # Invalid: < 0.0
                source_signal_id="sig-001"
            )

    def test_missing_owner_and_due_date_remain_none(self):
        """Scenario 9: Unmentioned fields must remain None rather than being fabricated."""
        res = RAIDExtractionResult(
            category=RAIDCategory.ISSUE,
            title="Firewall port 443 blocked",
            description="Port 443 blocked between systems.",
            owner=None,
            due_date=None,
            confidence=0.92,
            source_signal_id="sig-002",
            evidence="Port 443 blocked"
        )
        assert res.owner is None
        assert res.due_date is None

        # Even if passed as "null", "none", or whitespace
        res2 = RAIDExtractionResult(
            category=RAIDCategory.ISSUE,
            title="Firewall port 443 blocked",
            description="Port 443 blocked between systems.",
            owner="null",
            due_date="   ",
            confidence=0.92,
            source_signal_id="sig-002",
            evidence="Port 443 blocked"
        )
        assert res2.owner is None
        assert res2.due_date is None

    def test_no_fabricated_information(self):
        """Scenario 16: When model output lacks factual support, unknown values remain null."""
        raw_json = json.dumps({
            "category": "Risk",
            "title": "Database restore concern",
            "description": "Storage volume might slip",
            "impact": None,
            "owner": "N/A",
            "due_date": "None",
            "confidence": 0.85,
            "evidence": "Storage volume might slip",
            "reasoning_summary": "Storage constraint"
        })
        provider = MockLLMProvider(mock_response=raw_json)
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(source_type=SourceType.CHAT, content="Storage volume might slip")
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.owner is None
        assert result.due_date is None
        assert result.impact is None


class TestErrorHandlingAndReliability:
    def test_invalid_json_response_handling(self):
        """Scenario 10: Non-JSON / malformed LLM response handled gracefully without crashing."""
        provider = MockLLMProvider(mock_response="Not valid JSON at all! {broken syntax")
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(source_type=SourceType.EMAIL, content="Some message")
        result = classifier.classify_signal(signal)

        assert result is None  # Handled safely, did not crash or fabricate

    def test_api_timeout_handling(self):
        """Scenario 11: Timeout exceptions handled gracefully without crashing."""
        provider = MockLLMProvider(fail_times=999, exception_to_raise=TimeoutError("Request timed out after 30s"))
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(source_type=SourceType.EMAIL, content="Some message")
        result = classifier.classify_signal(signal)

        assert result is None  # Handled safely

    def test_retry_behavior_with_temporary_errors(self):
        """Scenario 12: Transient errors are retried up to max_retries and succeed when resolved."""
        # Fail 1 time with transient error, succeed on 2nd attempt
        provider = MockLLMProvider(fail_times=1, exception_to_raise=ConnectionResetError("Transient network reset"))
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            source_type=SourceType.EMAIL,
            content="The client may not provide the required master data before the integration testing phase."
        )

        # In classifier, if the provider itself implements retry or raises, let's test provider retry directly:
        gemini = GeminiProvider(api_key="mock_key", max_retries=2, retry_delay=0.01)
        # Mock client to fail once then succeed
        class FakeClient:
            def __init__(self):
                self.calls = 0
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    nonlocal fake_calls
                    fake_calls += 1
                    if fake_calls == 1:
                        raise ConnectionResetError("Transient connection drop")
                    class Resp:
                        text = '{"category": "Risk", "title": "Test", "description": "Desc", "confidence": 0.9, "source_signal_id": "s1"}'
                    return Resp()

        fake_calls = 0
        gemini._client = FakeClient()
        resp_text = gemini.generate_text("test prompt")
        assert fake_calls == 2
        assert "Test" in resp_text

    def test_missing_api_key_handling(self):
        """Scenario 13: Missing API key raises clear descriptive ValueError."""
        provider = GeminiProvider(api_key="")
        with pytest.raises(ValueError) as exc_info:
            provider.generate_text("Test prompt")
        assert "Gemini API key is not configured" in str(exc_info.value)

    def test_long_input_handling(self):
        """Scenario 14: Excessively long signals are safely truncated without crashing."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider, max_input_length=500)

        very_long_content = "Critical transport failure in SAP QA environment. " * 50  # ~2500 chars
        signal = ProjectSignal(source_type=SourceType.EMAIL, content=very_long_content)

        result = classifier.classify_signal(signal)
        assert result is not None
        # Check recorded prompt was truncated
        recorded_prompt = provider.recorded_prompts[-1]
        assert "[TRUNCATED]" in recorded_prompt

    def test_evidence_and_source_id_preservation(self):
        """Scenario 15: Source ID and evidence text are strictly linked to original signal."""
        provider = MockLLMProvider()
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            id="sig-traceability-999",
            source_type=SourceType.CHAT,
            content="Integration testing is currently blocked because the client has not provided master data."
        )
        result = classifier.classify_signal(signal)

        assert result is not None
        assert result.source_signal_id == "sig-traceability-999"
        assert "Integration testing is currently blocked" in result.evidence

    def test_hallucinated_evidence_fallback(self):
        """Audit Verification: If LLM hallucinates an evidence quote not in source, fall back to authentic source content."""
        hallucinated_json = json.dumps({
            "category": "Risk",
            "title": "Vendor delivery risk",
            "description": "Vendor might be late",
            "confidence": 0.85,
            "evidence": "This is a completely fabricated sentence never present in the signal",
            "source_signal_id": "sig-001"
        })
        provider = MockLLMProvider(mock_response=hallucinated_json)
        classifier = RAIDClassifier(provider=provider)

        signal = ProjectSignal(
            id="sig-001",
            source_type=SourceType.EMAIL,
            content="Actual project content: Third-party interface testing may run late."
        )
        result = classifier.classify_signal(signal)
        assert result is not None
        # Evidence was replaced by authentic excerpt from actual content
        assert "Actual project content" in result.evidence
        assert "fabricated sentence" not in result.evidence
