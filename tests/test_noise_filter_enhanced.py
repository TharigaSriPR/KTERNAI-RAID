"""
Comprehensive Enhanced Unit & Regression Tests for NoiseFilter.
Validates:
1. Clear conversational noise (greetings, pure acknowledgements)
2. Formal approvals and decisions (Approved, Agreed, Consensus, Go/No-Go)
3. Short meaningful project updates (Fix applied, DB fix done, Patch ready)
4. Technical fixes and defect resolutions (hotfix, workaround, outage, downtime)
5. Budget, scope, and vendor constraints (vendor late, budget cut, scope change)
6. Ambiguous messages (Please check this, As discussed, MOM attached)
7. Positive project updates (SIT 100% pass, mock runs ahead of schedule)
8. Food-related business discussions (lunch meeting, dinner review preserved)
9. Standalone sign-offs (Cheers Sarah, Best regards John filtered)
10. Substantive emails containing signatures (preserved for classification)
11. Standalone legal disclaimers (Confidentiality notice filtered)
12. Substantive emails containing disclaimers (preserved for classification)
13. Standalone automated notifications (Do not reply, ServiceNow generic notice filtered)
14. Automated notifications containing real incidents or defects (preserved for classification)
15. Project plan and defect records with short descriptions (source-aware preservation)
16. Meeting notes with short action items (source-aware preservation)
17. Empty and whitespace-only records (filtered with clear reasons)
"""

import pytest
from src.normalisation.schemas import ProjectSignal, SourceType
from src.noise_filter.filter_rules import NoiseFilter


class TestNoiseFilterEnhanced:

    # 1. Clear conversational noise
    @pytest.mark.parametrize("msg,stype", [
        ("Hi team", SourceType.CHAT),
        ("Hello all", SourceType.CHAT),
        ("Good morning everyone!", SourceType.EMAIL),
        ("Hey folks", SourceType.CHAT),
        ("Thanks!", SourceType.CHAT),
        ("Thank you so much", SourceType.EMAIL),
        ("Noted with thanks.", SourceType.CHAT),
        ("Sounds good.", SourceType.CHAT),
        ("LGTM", SourceType.CHAT),
        ("Looks good to me", SourceType.CHAT)
    ])
    def test_clear_conversational_noise(self, msg: str, stype: SourceType):
        sig = ProjectSignal(source_type=stype, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is True
        assert any(k in reason.lower() for k in ["greeting", "acknowledgement", "thanks"])

    # 2. Formal approvals and decisions
    @pytest.mark.parametrize("msg,stype", [
        ("Approved.", SourceType.CHAT),
        ("Approved", SourceType.CHAT),
        ("Agreed: We will use standard BAPI instead of custom RFC", SourceType.MEETING_NOTES),
        ("SteerCo approved wave 1 scope reduction to meet November go-live", SourceType.MEETING_NOTES),
        ("Decision: Architecture committee approved rollback strategy", SourceType.EMAIL),
        ("Go/no-go consensus reached: Proceed with cutover window", SourceType.MEETING_NOTES),
        ("Sign-off granted for SIT cycle 2", SourceType.EMAIL)
    ])
    def test_formal_approvals_and_decisions(self, msg: str, stype: SourceType):
        sig = ProjectSignal(source_type=stype, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Formal decision was falsely filtered: '{msg}' ({reason})"

    # 3. Short meaningful project updates (<12 chars)
    @pytest.mark.parametrize("msg", [
        "Fix applied",
        "DB fix done",
        "Patch ready",
        "Budget cut",
        "Scope cut",
        "Vendor late",
        "Workaround",
        "Hotfix done",
        "UAT blocked"
    ])
    def test_short_meaningful_project_updates(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.CHAT, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Short meaningful message was falsely filtered: '{msg}' ({reason})"

    # 4. Technical fixes and defect resolutions
    @pytest.mark.parametrize("msg", [
        "The BAPI dump issue in sandbox has been resolved",
        "Database connectivity problem fixed by Basis team yesterday",
        "Hotfix applied to solve memory leak in custom RFC",
        "Temporary workaround established for tax calculation defect",
        "Unplanned server downtime occurred in QA during regression run",
        "Network bottleneck identified during peak load test"
    ])
    def test_technical_fixes_and_resolutions(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.CHAT, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Technical resolution was falsely filtered: '{msg}' ({reason})"

    # 5. Budget, scope, and vendor constraints
    @pytest.mark.parametrize("msg", [
        "Hardware server delivery postponed by vendor for two weeks",
        "Third-party payment API documentation is missing key auth specifications",
        "Client requested addition of 5 new interfaces for warehouse tracking",
        "Offshore ABAP development capacity is reduced by 3 consultants this sprint",
        "Budget overrun projected on external cloud infrastructure hosting",
        "Software license expiration will halt SIT testing next Tuesday"
    ])
    def test_budget_scope_and_vendor_constraints(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Project constraint was falsely filtered: '{msg}' ({reason})"

    # 6. Ambiguous messages (must NOT be filtered as noise)
    @pytest.mark.parametrize("msg", [
        "Please check this.",
        "As discussed, we need to close this.",
        "The same issue is still pending.",
        "We should consider an alternative.",
        "The team is waiting for confirmation.",
        "This is completed.",
        "Please find the MOM attached."
    ])
    def test_ambiguous_messages_preserved(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.CHAT, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Ambiguous message was falsely filtered: '{msg}' ({reason})"

    # 7. Positive project updates
    @pytest.mark.parametrize("msg", [
        "SIT 1 testing completed with 100% pass rate across all FI test cases",
        "All 15 data migration mock runs completed successfully ahead of schedule",
        "Security compliance audit passed with zero high-severity findings"
    ])
    def test_positive_project_updates(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Positive progress update was falsely filtered: '{msg}' ({reason})"

    # 8. Food-related business discussions (preserved) vs Casual food chatter (filtered)
    def test_food_related_business_discussions(self):
        biz_messages = [
            "Discussed the project timeline with the client over lunch and agreed to extend the testing window.",
            "Dinner will be served during cutover weekend in the war room.",
            "Vendor meeting scheduled during lunch to finalize SLA agreements.",
            "Budget review discussed during the steering committee dinner."
        ]
        for msg in biz_messages:
            sig = ProjectSignal(source_type=SourceType.EMAIL, content=msg)
            is_noise, reason = NoiseFilter.evaluate(sig)
            assert is_noise is False, f"Business message mentioning food was falsely filtered: '{msg}' ({reason})"

        chat_messages = [
            "Anyone up for pizza in the cafeteria?",
            "Anyone up for coffee in the cafeteria?",
            "Let's grab lunch",
            "Who wants coffee?",
            "Coffee break time!"
        ]
        for msg in chat_messages:
            sig = ProjectSignal(source_type=SourceType.CHAT, content=msg)
            is_noise, reason = NoiseFilter.evaluate(sig)
            assert is_noise is True, f"Casual food chatter was not filtered: '{msg}'"
            assert "food" in reason.lower()

    # 9. Standalone sign-offs (filtered)
    @pytest.mark.parametrize("msg", [
        "Cheers, Sarah",
        "Best regards,\nJohn",
        "Kind regards",
        "Sincerely, John",
        "Thanks & regards, Sarah",
        "Warm regards,\nDave Miller"
    ])
    def test_standalone_sign_offs_filtered(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is True, f"Standalone sign-off was not filtered: '{msg}'"
        assert "sign-off" in reason.lower() or "signature" in reason.lower()

    # 10. Substantive emails containing signatures (preserved)
    def test_substantive_emails_containing_signatures(self):
        email_body = (
            "The data migration mock run 3 encountered 14 data formatting errors in table KNA1.\n"
            "We need the master data lead to review the mapping sheet before tomorrow.\n\n"
            "Best regards,\n"
            "Sarah Jenkins\n"
            "Senior SAP Consultant"
        )
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=email_body)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Substantive email with signature was falsely filtered: {reason}"

    # 11. Standalone legal disclaimers (filtered)
    @pytest.mark.parametrize("msg", [
        "CONFIDENTIALITY NOTICE: This email and any attachments are confidential and intended solely for the recipient.",
        "The information contained in this message is privileged and confidential. If you are not the intended recipient, notify sender.",
        "This email was scanned by Proofpoint Enterprise Security and deemed clean."
    ])
    def test_standalone_legal_disclaimers_filtered(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is True, f"Standalone disclaimer was not filtered: '{msg}'"
        assert "disclaimer" in reason.lower()

    # 12. Substantive emails containing disclaimers (preserved)
    def test_substantive_emails_containing_disclaimers(self):
        email_body = (
            "Critical defect DEF-901 has reoccurred in QA environment during transport TRK90012.\n"
            "Testing cannot proceed until basis team applies the kernel patch.\n\n"
            "CONFIDENTIALITY NOTICE: This transmission is intended only for the use of the individual named above."
        )
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=email_body)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Substantive email with disclaimer was falsely filtered: {reason}"

    # 13. Standalone automated notifications (filtered)
    @pytest.mark.parametrize("msg", [
        "Do not reply to this email. This is an automated notification from ServiceNow.",
        "This is an automated notification confirming your password reset request. Do not reply.",
        "Generated by Jira Cloud. This message was sent automatically."
    ])
    def test_standalone_automated_notifications_filtered(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is True, f"Standalone automated notification was not filtered: '{msg}'"
        assert "automated" in reason.lower() or "notification" in reason.lower()

    # 14. Automated notifications containing real incidents or defects (preserved)
    def test_automated_notifications_with_real_incidents(self):
        notif_with_incident = (
            "Generated by ServiceNow: Critical incident INC009823 opened for Production ECC system.\n"
            "Database dump occurred during payment batch run. Do not reply to this email."
        )
        sig = ProjectSignal(source_type=SourceType.EMAIL, content=notif_with_incident)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Incident notification was falsely filtered: {reason}"

    # 15. Project plan and defect records with short descriptions (source-aware)
    @pytest.mark.parametrize("msg,stype", [
        ("Sign-off", SourceType.PROJECT_PLAN),
        ("Fix applied", SourceType.DEFECT),
        ("Patch ready", SourceType.DEFECT),
        ("UAT blocked", SourceType.PROJECT_PLAN),
        ("Vendor pending", SourceType.PROJECT_PLAN),
        ("Review", SourceType.PROJECT_PLAN)
    ])
    def test_formal_source_records_preserved(self, msg: str, stype: SourceType):
        sig = ProjectSignal(source_type=stype, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Formal {stype.value} record was falsely filtered: '{msg}' ({reason})"

    # 16. Meeting notes with short action items (source-aware)
    @pytest.mark.parametrize("msg", [
        ("Action: Sarah to test FI user exit"),
        ("Agreed: Go-live moved to Nov 15"),
        ("Approved."),
        ("Owner: Mike to verify Basis transport")
    ])
    def test_meeting_notes_action_items_preserved(self, msg: str):
        sig = ProjectSignal(source_type=SourceType.MEETING_NOTES, content=msg)
        is_noise, reason = NoiseFilter.evaluate(sig)
        assert is_noise is False, f"Meeting note action was falsely filtered: '{msg}' ({reason})"

    # 17. Empty and whitespace-only records
    @pytest.mark.parametrize("msg", ["", "   ", "\n\t  "])
    def test_empty_and_whitespace_records(self, msg: str):
        # Bare signal with bypassed validation to test noise filter robustness
        class BareSignal:
            subject = ""
            content = msg
            source_type = SourceType.CHAT
        is_noise, reason = NoiseFilter.evaluate(BareSignal)
        assert is_noise is True
        assert "empty" in reason.lower()
