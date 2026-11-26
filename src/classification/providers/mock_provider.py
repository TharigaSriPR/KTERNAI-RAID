"""
Deterministic Mock Provider for unit testing and offline execution.
Allows injecting test fixtures, testing retry behavior, and testing invalid responses.
"""

from typing import Optional, Dict, List, Any
import json
from .base_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Mock implementation of BaseLLMProvider for test suites and offline pipelines.
    """

    def __init__(
        self,
        default_category: str = "NOT_RAID",
        mock_response: Optional[str] = None,
        fail_times: int = 0,
        exception_to_raise: Optional[Exception] = None
    ):
        super().__init__("mock-llm")
        self.default_category = default_category
        self.mock_response = mock_response
        self.fail_times = fail_times
        self.exception_to_raise = exception_to_raise or TimeoutError("Simulated LLM API request timeout.")
        self.call_count = 0
        self.recorded_prompts: List[str] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        self.call_count += 1
        self.recorded_prompts.append(prompt)

        # Simulate failures for testing retry logic
        if self.fail_times > 0 and self.call_count <= self.fail_times:
            raise self.exception_to_raise

        # If an explicit raw mock response was supplied (e.g. invalid JSON, custom payload)
        if self.mock_response is not None:
            return self.mock_response

        # Inspect only the target signal portion after the few-shot examples
        if "--- NOW ANALYZE THIS PROJECT SIGNAL ---" in prompt:
            prompt_lower = prompt.split("--- NOW ANALYZE THIS PROJECT SIGNAL ---")[-1].lower()
        else:
            prompt_lower = prompt.lower()

        if "configuration testing can begin only after the basis team completes" in prompt_lower:
            return json.dumps({
                "category": "Dependency",
                "title": "System refresh required before configuration testing",
                "description": "Configuration testing depends on Basis team completing the system refresh.",
                "impact": "Config test start dependent on Basis refresh",
                "owner": "Basis Team",
                "due_date": None,
                "confidence": 0.94,
                "evidence": "Configuration testing can begin only after the Basis team completes the system refresh.",
                "reasoning_summary": "Testing cannot begin until an external team deliverable is finished."
            })

        if "client may not provide the required master data" in prompt_lower or "remediation may run late" in prompt_lower or "pricing user exit refactoring" in prompt_lower or "carries high risk" in prompt_lower or "high risk and may delay" in prompt_lower or "risk of delay" in prompt_lower:
            return json.dumps({
                "category": "Risk",
                "title": "Delay in integration testing due to pending master data",
                "description": "Potential schedule delay if master data is not delivered prior to integration test.",
                "impact": "Schedule delay in integration test",
                "owner": None,
                "due_date": None,
                "confidence": 0.92,
                "evidence": "The client may not provide the required master data before the integration testing phase." if "client may not provide" in prompt_lower else "Agreed that SD pricing user exit refactoring carries high risk.",
                "reasoning_summary": "Future uncertain condition with potential schedule impact."
            })

        if "integration testing is currently blocked" in prompt_lower or "failed with rc=8 and blocked" in prompt_lower or "transport trk9001 failed" in prompt_lower or "dump on sandbox" in prompt_lower or "critical defect" in prompt_lower or "is currently blocked" in prompt_lower:
            return json.dumps({
                "category": "Issue",
                "title": "Integration testing blocked due to missing master data",
                "description": "Integration testing is actively stalled because client master data has not arrived.",
                "impact": "Active blocker on test cycle",
                "owner": None,
                "due_date": None,
                "confidence": 0.95,
                "evidence": "Integration testing is currently blocked because the client has not provided master data." if "integration testing is currently blocked" in prompt_lower else "Critical transport TRK9001 failed and blocked testing.",
                "reasoning_summary": "Active failure directly blocking work right now."
            })

        if "assuming that the client will complete" in prompt_lower or "we assume the customer business team" in prompt_lower or "planning assumption" in prompt_lower or "assumed that" in prompt_lower or "assumption:" in prompt_lower:
            return json.dumps({
                "category": "Assumption",
                "title": "Client master data will be ready before SIT",
                "description": "Planning assumption that client will complete master data prep before SIT.",
                "impact": None,
                "owner": None,
                "due_date": None,
                "confidence": 0.89,
                "evidence": "We are assuming that the client will complete all master data preparation before the SIT phase." if "assuming that the client" in prompt_lower else "We assume the customer business team will provide clean test data.",
                "reasoning_summary": "Unverified belief treated as true for planning."
            })

        if "agreed to move the sit start date" in prompt_lower or "chose to drop legacy" in prompt_lower or "decided to proceed with brownfield" in prompt_lower or "decided to proceed" in prompt_lower or "steerco approved moving" in prompt_lower or "formal agreement" in prompt_lower:
            return json.dumps({
                "category": "Decision",
                "title": "SIT start date moved to 15 December",
                "description": "Formal agreement made by project team to postpone SIT start to 15 December.",
                "impact": "Timeline adjustment",
                "owner": "Project Team",
                "due_date": "2026-12-15",
                "confidence": 0.96,
                "evidence": "The project team has agreed to move the SIT start date to 15 December." if "agreed to move" in prompt_lower else "Decided to proceed with brownfield conversion rather than selective data transition.",
                "reasoning_summary": "Formal choice agreed by project stakeholders."
            })

        if "thanks for sharing the meeting minutes" in prompt_lower:
            return json.dumps({
                "category": "NOT_RAID",
                "title": None,
                "description": None,
                "impact": None,
                "owner": None,
                "due_date": None,
                "confidence": 0.99,
                "evidence": "Thanks for sharing the meeting minutes.",
                "reasoning_summary": "Casual communication without RAID impact."
            })

        # Generic default response
        return json.dumps({
            "category": self.default_category,
            "title": f"Identified {self.default_category}" if self.default_category != "NOT_RAID" else None,
            "description": "Generated mock item." if self.default_category != "NOT_RAID" else None,
            "impact": None,
            "owner": None,
            "due_date": None,
            "confidence": 0.85,
            "evidence": prompt[:100],
            "reasoning_summary": "Default mock response"
        })
