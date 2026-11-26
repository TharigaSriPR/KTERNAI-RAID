"""
Enhanced Rule-Based Noise Filter.
Applies fast, zero-token Python heuristics to discard irrelevant communications
while strictly protecting critical project signals, approvals, fixes, and constraints:
- Greetings and acknowledgements in informal channels
- Standalone sign-offs and legal disclaimers
- Automated system notifications without defect/incident payloads
- Out of office auto-replies, calendar notifications, and casual food chatter
- Empty, whitespace, or meaningless filler records

Protects all signals containing actionable project/RAID keywords across:
- Decisions and approvals (approve, agreed, sign-off, consensus, go/no-go)
- Technical fixes and mitigations (fix, patch, hotfix, workaround, revert, outage)
- Constraints and governance (budget, scope, vendor, contract, deadline, milestone)
- Actions and deliverables (action item, deliverable, owner, pending)
- Core RAID terms (risk, issue, assumption, dependency, blocker, delay, failure, etc.)
"""

import re
from typing import Tuple, List, Set
from src.normalisation.schemas import ProjectSignal, SourceType
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Formal sources where conversational chat heuristics (greetings, pure acknowledgements,
# or informal conversational checks) must never be applied.
FORMAL_PROJECT_SOURCES: Set[SourceType] = {
    SourceType.PROJECT_PLAN,
    SourceType.DEFECT,
    SourceType.MEETING_NOTES
}

# Critical domain terms that must NEVER be discarded, even if the message is short
DOMAIN_PROTECTED_TERMS = [
    # Core technical and schedule risks/issues
    r"\bblock(?:ed|ing|er|s)?\b",
    r"\bdelay(?:ed|ing|s)?\b",
    r"\bslip(?:page|ped|ping|s)?\b",
    r"\brisk(?:s|y)?\b",
    r"\bissue(?:s)?\b",
    r"\bdefect(?:s)?\b",
    r"\bbug(?:s)?\b",
    r"\bfail(?:ed|ing|ure|ures|s)?\b",
    r"\bdump(?:s)?\b",
    r"\berror(?:s)?\b",
    r"\bcritical\b",
    r"\burgent\b",
    r"\bescalat(?:e|ed|ing|ion)\b",
    r"\brollback\b",
    r"\bstuck\b",
    r"\bstop(?:ped)?\b",
    r"\boverrun(?:s)?\b",
    r"\bunanalysed\b",
    r"\bdepend(?:ency|encies|ent)?\b",
    r"\bassum(?:e|ed|ing|ption|ptions)\b",
    r"\bdecid(?:e|ed|ing|ision|isions)\b",
    r"\bsteerco\b",
    r"\bvariance\b",
    r"\btransport\b",
    r"\bcutover\b",
    r"\bgo-live\b",
    r"\bhypercare\b",
    r"\bsit\b",
    r"\buat\b",
    r"\bcustom code\b",
    r"\bz-object(?:s)?\b",
    r"\bremediat(?:e|ion)\b",
    r"\bcapacity constraint\b",

    # Decisions and approvals
    r"\bapprov(?:e|ed|ing|al|als)?\b",
    r"\bagree(?:d|ment|ing)?\b",
    r"\breject(?:ed|ing|ion)?\b",
    r"\bsign-?off(?:s|ed)?\b",
    r"\bsigned\s+off\b",
    r"\bconsensus\b",
    r"\bgo\s*/\s*no-?go\b",

    # Technical fixes and mitigations
    r"\bfix(?:ed|es|ing)?\b",
    r"\bfix\s+applied\b",
    r"\bpatch(?:ed|es|ing)?\b",
    r"\bpatch\s+ready\b",
    r"\bhotfix(?:es)?\b",
    r"\bworkaround(?:s)?\b",
    r"\brevert(?:ed|ing)?\b",
    r"\boutage(?:s)?\b",
    r"\bdowntime\b",
    r"\bbottleneck(?:s)?\b",

    # Project constraints and governance
    r"\bbudget(?:s|ary)?\b",
    r"\bscope(?:s)?\b",
    r"\bvendor(?:s)?\b",
    r"\bsupplier(?:s)?\b",
    r"\bcontract(?:s)?\b",
    r"\blicen[sc]e(?:s|d|ing)?\b",
    r"\bdeadline(?:s)?\b",
    r"\bmilestone(?:s)?\b",
    r"\baudit(?:s|ed|ing)?\b",
    r"\bcomplian(?:ce|t)\b",

    # Actions and deliverables
    r"\baction(?:\s+item)?(?:s)?\b",
    r"\bdeliverable(?:s)?\b",
    r"\bowner(?:s)?\b",
    r"\bassign(?:ed|ee|ment)?\b",
    r"\bpending\b"
]

DOMAIN_PROTECTED_REGEX = re.compile("|".join(DOMAIN_PROTECTED_TERMS), re.IGNORECASE)

# Conversational Greetings (strictly anchored to whole message)
GREETINGS_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|greetings)(?:\s+(?:all|team|everyone|folks|there))?[\.!, ]*$",
    re.IGNORECASE
)

# Conversational Acknowledgements (strictly anchored to whole message)
ACKNOWLEDGEMENTS_PATTERN = re.compile(
    r"^(?:thanks|thank\s+you|thank\s+you\s+so\s+much|many\s+thanks|thx|ty|thanks\s+a\s+lot|"
    r"noted\s+with\s+thanks|noted|got\s+it|cool|ok|okay|k|great|sounds\s+good|acknowledged|"
    r"confirmed|lgtm|looks\s+good(?:\s+to\s+me)?|will\s+do|sure|perfect)[\.!, ]*$",
    re.IGNORECASE
)

# Standalone Sign-offs / Signatures (anchored, does not match substantive body + signature)
STANDALONE_SIGN_OFF_PATTERN = re.compile(
    r"^(?:(?:best|warm|kindest|kind)?\s*regards|"
    r"thanks\s*(?:&|and)\s*regards|"
    r"cheers|"
    r"sincerely|"
    r"yours\s+truly|"
    r"respectfully)"
    r"(?:,?\s*[\r\n]+.*|,?\s+[A-Za-z]+(?:\s+[A-Za-z]+)?)?[\.!, ]*$",
    re.IGNORECASE
)

# Standalone Legal Disclaimers (anchored, does not match substantive email + disclaimer footer)
STANDALONE_DISCLAIMER_PATTERN = re.compile(
    r"^(?:\s*|\W*)*(?:"
    r"confidentiality\s+notice|"
    r"this\s+email\s+(?:and\s+any\s+attachments\s+)?is\s+(?:confidential|intended)|"
    r"the\s+information\s+(?:contained\s+in\s+this|in\s+this)\s+(?:email|message)\s+is\s+(?:confidential|privileged)|"
    r"privileged\s+and\s+confidential|"
    r"intended\s+solely\s+for\s+(?:the\s+)?use|"
    r"(?:this\s+email\s+(?:was\s+)?scanned\s+by|scanned\s+by)\s+(?:proofpoint|symantec|mcafee|sophos|barracuda)"
    r")\b.*$",
    re.IGNORECASE | re.DOTALL
)

# Standalone Generic Automated Notifications (anchored, no defect/incident payload)
STANDALONE_AUTOMATED_NOTIFICATION_PATTERN = re.compile(
    r"^(?:\s*|\W*)*(?:"
    r"do\s+not\s+reply\s+to\s+this\s+email|"
    r"this\s+is\s+an\s+automated\s+notification|"
    r"this\s+is\s+a\s+system[- ]generated\s+(?:email|notification|message)|"
    r"generated\s+by\s+(?:servicenow|jira|github|confluence|jenkins|gitlab|azure\s+devops)"
    r")\b.*$",
    re.IGNORECASE | re.DOTALL
)

# Casual Social & Food Invitations (intent-based, protects business discussions mentioning lunch/dinner)
CASUAL_FOOD_INVITATION_PATTERN = re.compile(
    r"^(?:.*?\b(?:"
    r"anyone\s+(?:up\s+for|free\s+for|want\s+(?:some|to\s+grab))|"
    r"who(?:'s|\s+is)\s+(?:up\s+for|getting|ordering|bringing)|"
    r"who\s+wants?|"
    r"let's\s+(?:grab|get|order)|"
    r"grabbing\s+some|"
    r"ordering\s+some|"
    r"time\s+for|"
    r"coffee\s+break|"
    r"tea\s+break"
    r")\s+(?:coffee|lunch|cafeteria|dinner|pizza|tea|food|drinks?|snacks?)\b.*|"
    r"\b(?:coffee\s+break|tea\s+break|lunch\s+time)\b.*|"
    r"coffee\?|lunch\?|dinner\?|pizza\?)$",
    re.IGNORECASE
)

# Administrative Noise Patterns (OOO, Calendar RSVPs, HR greetings, Marketing)
ADMIN_NOISE_PATTERNS = [
    (re.compile(r"automatic\s+reply|out\s+of\s+(?:the\s+)?office|on\s+annual\s+leave|away\s+from\s+(?:the\s+)?office|vacation\s+auto-reply", re.IGNORECASE), "Out of office auto-reply"),
    (re.compile(r"^(?:accepted|declined|tentative):\s*|has\s+(?:accepted|declined)\s+this\s+invitation|updated\s+invitation", re.IGNORECASE), "Calendar invitation notification"),
    (re.compile(r"happy\s+birthday|happy\s+work\s+anniversary|congratulations|welcome\s+aboard", re.IGNORECASE), "HR / social greetings"),
    (re.compile(r"unsubscribe|marketing\s+digest|newsletter", re.IGNORECASE), "Automated marketing/bulk email")
]

# Meaningless Input Patterns (replaces arbitrary <12 character cutoff)
MEANINGLESS_INPUT_PATTERN = re.compile(
    r"^(?:"
    r"[\s\W_]+|"                         # Pure punctuation / symbols (???, ..., !!!, ---, etc.)
    r"\d+|"                              # Pure digits (123, 000, etc.)
    r"[a-zA-Z]|"                         # Single letter (k, y, x, etc.)
    r"(.)\1{2,}|"                        # Repetitive single char (aaa, zzz, etc.)
    r"(?:test|testing|asdf|qwerty|foo|bar|baz|na|n/a|none|null|nil)"  # Obvious test/filler strings
    r")[\.!, ]*$",
    re.IGNORECASE
)


class NoiseFilter:
    """
    Evaluates ProjectSignal instances and flags noise while retaining full signal traceability.
    Operates offline without token cost, using deterministic source-aware heuristics.
    """

    @staticmethod
    def is_protected_signal(text: str) -> bool:
        """
        Returns True if the text contains high-value domain keywords that must not be filtered.
        """
        return bool(DOMAIN_PROTECTED_REGEX.search(text))

    @classmethod
    def evaluate(cls, signal: ProjectSignal) -> Tuple[bool, str]:
        """
        Evaluates whether a ProjectSignal is noise.
        Returns: (is_noise: bool, reason: str)
        """
        subject_clean = (signal.subject or "").strip()
        content_clean = (signal.content or "").strip()
        full_text = f"{subject_clean} {content_clean}".strip()

        # 1. Empty or whitespace-only content
        if not content_clean:
            return True, "Empty or whitespace-only content"

        # 2. Check if content has domain-protected keywords (e.g. "Testing blocked", "Approved", "Budget cut")
        if cls.is_protected_signal(full_text):
            return False, ""

        # 3. Source-aware determination:
        # Formal sources (PROJECT_PLAN, DEFECT, MEETING_NOTES) skip casual conversational checks
        is_formal = signal.source_type in FORMAL_PROJECT_SOURCES

        # 4. Standalone legal disclaimers
        if STANDALONE_DISCLAIMER_PATTERN.match(content_clean):
            return True, "Legal / confidentiality disclaimer"

        # 5. Standalone automated system notifications
        if STANDALONE_AUTOMATED_NOTIFICATION_PATTERN.match(content_clean):
            return True, "Automated system notification"

        # 6. Standalone sign-offs / signatures (only for conversational channels like email/chat)
        if not is_formal and STANDALONE_SIGN_OFF_PATTERN.match(content_clean):
            return True, "Email sign-off or signature"

        # 7. Conversational greetings (only for email/chat)
        if not is_formal and GREETINGS_PATTERN.match(content_clean):
            return True, "Conversational greeting"

        # 8. Conversational acknowledgements / thanks (only for email/chat)
        if not is_formal and ACKNOWLEDGEMENTS_PATTERN.match(content_clean):
            return True, "Generic acknowledgement or thanks"

        # 9. Casual food / social invitations (intent-based)
        if CASUAL_FOOD_INVITATION_PATTERN.match(content_clean):
            return True, "Casual social / food conversation"

        # 10. Administrative noise patterns (OOO, Calendar RSVPs, HR greetings, Newsletters)
        for pattern, reason in ADMIN_NOISE_PATTERNS:
            if pattern.search(full_text):
                return True, reason

        # 11. Meaningless input (pure punctuation, digits, single chars, or test filler)
        if MEANINGLESS_INPUT_PATTERN.match(content_clean):
            return True, "Meaningless input or test filler"

        return False, ""

    @classmethod
    def apply_filter(cls, signal: ProjectSignal) -> ProjectSignal:
        """
        Applies filter evaluation and records is_noise flag and noise_reason directly on the signal.
        """
        is_noise, reason = cls.evaluate(signal)
        signal.is_noise = 1 if is_noise else 0
        signal.noise_reason = reason
        if is_noise:
            logger.debug("Signal %s filtered as noise: %s", signal.id, reason)
        return signal

    @classmethod
    def filter_batch(cls, signals: List[ProjectSignal]) -> List[ProjectSignal]:
        """
        Applies filter to a list of signals and returns the modified list.
        """
        for sig in signals:
            cls.apply_filter(sig)
        noise_count = sum(1 for s in signals if s.is_noise == 1)
        logger.info("Noise filtering complete: %d of %d signals marked as noise", noise_count, len(signals))
        return signals
