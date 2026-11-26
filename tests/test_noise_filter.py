"""
Unit tests for Enhanced Noise Filter.
Verifies:
- Empty and whitespace messages are filtered.
- Greetings and acknowledgements are filtered.
- Administrative noise (out of office, calendar RSVPs, food chatter) is filtered.
- Important short domain/project signals are PROTECTED and NOT filtered.
- Signal noise reasons are recorded accurately.
"""

import pytest
from src.normalisation.schemas import ProjectSignal, SourceType
from src.noise_filter.filter_rules import NoiseFilter


class TestNoiseFilter:
    def test_empty_content_filtered(self):
        class DummySignal:
            subject = ""
            content = "   "
        is_noise, reason = NoiseFilter.evaluate(DummySignal)
        assert is_noise is True
        assert "Empty" in reason

    def test_conversational_greetings_filtered(self):
        greetings = [
            "Hi",
            "Hello",
            "Hey",
            "Good morning",
            "Good afternoon",
            "Hi all",
            "Hello team",
            "Hey folks!"
        ]
        for g in greetings:
            sig = ProjectSignal(source_type=SourceType.CHAT, content=g)
            NoiseFilter.apply_filter(sig)
            assert sig.is_noise == 1, f"Expected '{g}' to be filtered as greeting"
            assert "greeting" in sig.noise_reason.lower()

    def test_acknowledgements_and_thanks_filtered(self):
        acks = [
            "Thanks",
            "Thank you so much",
            "Many thanks",
            "Thx",
            "Noted with thanks",
            "Got it",
            "Sounds good",
            "OK",
            "Okay!",
            "Great",
            "Confirmed",
            "LGTM",
            "Looks good to me"
        ]
        for ack in acks:
            sig = ProjectSignal(source_type=SourceType.CHAT, content=ack)
            NoiseFilter.apply_filter(sig)
            assert sig.is_noise == 1, f"Expected '{ack}' to be filtered as acknowledgement"
            assert "acknowledgement" in sig.noise_reason.lower()

    def test_administrative_noise_filtered(self):
        admin_messages = [
            ("Out of office: I am on annual leave until Monday", "Out of office"),
            ("Accepted: Weekly Status Call", "Calendar invitation"),
            ("Anyone up for coffee in the cafeteria?", "food"),
            ("Happy birthday to Sarah in finance!", "HR / social"),
            ("Click here to unsubscribe from newsletter digest", "marketing")
        ]
        for text, keyword in admin_messages:
            sig = ProjectSignal(source_type=SourceType.EMAIL, content=text)
            NoiseFilter.apply_filter(sig)
            assert sig.is_noise == 1, f"Expected '{text}' to be filtered"

    def test_important_short_project_messages_protected(self):
        """
        CRITICAL TEST:
        Ensures messages containing actionable domain keywords are NOT discarded merely because they are short.
        """
        protected_messages = [
            "Testing blocked due to missing data",
            "Task delayed by three days",
            "Transport TRK9001 failed in QA",
            "Database restore dump occurred",
            "Critical defect reopened",
            "Offshore capacity constraint identified",
            "SteerCo approved wave 1 scope reduction",
            "Assuming client will provide test data by Friday",
            "SD pricing user exit refactoring at risk",
            "Variance: 8 days delay on custom ABAP"
        ]
        for msg in protected_messages:
            sig = ProjectSignal(source_type=SourceType.CHAT, content=msg)
            NoiseFilter.apply_filter(sig)
            assert sig.is_noise == 0, f"Critical message was mistakenly filtered: '{msg}'"
            assert sig.noise_reason == ""

    def test_filter_batch(self):
        signals = [
            ProjectSignal(source_type=SourceType.CHAT, content="Hello team"),
            ProjectSignal(source_type=SourceType.CHAT, content="Testing blocked due to missing data"),
            ProjectSignal(source_type=SourceType.CHAT, content="Thanks")
        ]
        filtered = NoiseFilter.filter_batch(signals)
        assert len(filtered) == 3
        assert filtered[0].is_noise == 1
        assert filtered[1].is_noise == 0
        assert filtered[2].is_noise == 1
