"""
Pydantic v2 schemas for Structured RAID Extraction & Classification.
Provides strict validation, category Enums, confidence thresholds,
and source traceability fields.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class ClassificationStatus(str, Enum):
    """
    Lifecycle processing states for ProjectSignal classification.
    """
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_RAID = "NOT_RAID"
    BELOW_CONFIDENCE_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"


class ClassificationErrorCategory(str, Enum):
    """
    Structured categorization of LLM classification failure modes.
    """
    API_TIMEOUT = "API_TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    API_ERROR = "API_ERROR"
    INVALID_JSON = "INVALID_JSON"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class EvidenceStatus(str, Enum):
    """
    Evidence verification provenance states.
    """
    VERIFIED = "verified"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


class RAIDCategory(str, Enum):
    """
    Five primary RAID categories plus non-RAID classification.
    """
    RISK = "Risk"
    ASSUMPTION = "Assumption"
    ISSUE = "Issue"
    DECISION = "Decision"
    DEPENDENCY = "Dependency"
    NOT_RAID = "NOT_RAID"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_clean = value.strip().upper()
            if val_clean in ("NOT_RAID", "NONE", "NOT RAID", "NO_RAID", "NON_RAID"):
                return cls.NOT_RAID
            for member in cls:
                if member.value.upper() == val_clean:
                    return member
        return None


class RAIDExtractionResult(BaseModel):
    """
    Validated structured extraction result produced by the LLM classifier.
    Preserves evidence and links directly to the originating ProjectSignal.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        extra="ignore"
    )

    category: RAIDCategory = Field(
        ...,
        description="Classified RAID category or NOT_RAID"
    )
    title: Optional[str] = Field(
        default=None,
        description="Short and meaningful title describing the item"
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed contextual summary of the RAID item"
    )
    impact: Optional[str] = Field(
        default=None,
        description="Project impact if explicitly stated or directly inferred from source"
    )
    owner: Optional[str] = Field(
        default=None,
        description="Responsible person or team, only when explicitly supported by the source"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Date or timeframe mentioned in the source"
    )
    workstream: Optional[str] = Field(
        default="General",
        description="Associated SAP module or track (FI, SD, MM, Basis, etc.)"
    )
    phase: Optional[str] = Field(
        default="Realize",
        description="SAP Activate phase (Prepare, Explore, Realize, Deploy, Run)"
    )
    probability: Optional[str] = Field(
        default=None,
        description="Likelihood of risk occurring: Low, Medium, High"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity of issue: Low, Medium, High, Critical"
    )
    mitigation_action: Optional[str] = Field(
        default=None,
        description="Mitigation step, next action, or resolution response"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Exact supporting verbatim text from the original signal"
    )
    source_signal_id: str = Field(
        ...,
        description="Identifier of the source ProjectSignal"
    )
    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Short explanation of why this category was selected"
    )
    evidence_status: EvidenceStatus = Field(
        default=EvidenceStatus.UNAVAILABLE,
        description="Verification state of supporting evidence: 'verified', 'fallback', or 'unavailable'"
    )
    llm_provider: Optional[str] = Field(
        default=None,
        description="LLM provider name: gemini, mock, openrouter, etc."
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model identifier used for classification"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence score {v} must be between 0.0 and 1.0")
        return round(float(v), 4)

    @field_validator("owner", "due_date", "impact", "mitigation_action", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v_str = str(v).strip()
        if not v_str or v_str.lower() in ("null", "none", "unknown", "n/a", "na", ""):
            return None
        return v_str

    @model_validator(mode="before")
    @classmethod
    def normalize_not_raid_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cat = str(data.get("category", "")).strip().upper()
            if cat in ("NOT_RAID", "NONE", "NOT RAID", "NO_RAID"):
                data["title"] = None
                data["description"] = None
                data["impact"] = None
                data["owner"] = None
                data["due_date"] = None
                data["probability"] = None
                data["severity"] = None
                data["mitigation_action"] = None
        return data

    @model_validator(mode="after")
    def validate_category_consistency(self) -> "RAIDExtractionResult":
        if self.category != RAIDCategory.NOT_RAID:
            if not self.title or not self.title.strip():
                raise ValueError(f"RAID item of category '{self.category}' must include a non-empty title.")
            if not self.description or not self.description.strip():
                object.__setattr__(self, "description", self.title)
        else:
            object.__setattr__(self, "title", None)
            object.__setattr__(self, "description", None)
            object.__setattr__(self, "impact", None)
            object.__setattr__(self, "owner", None)
            object.__setattr__(self, "due_date", None)
            object.__setattr__(self, "probability", None)
            object.__setattr__(self, "severity", None)
            object.__setattr__(self, "mitigation_action", None)
        return self

    @property
    def is_raid(self) -> bool:
        """Returns True if the extraction represents an actionable RAID item."""
        return self.category != RAIDCategory.NOT_RAID


class ClassificationOutcome(BaseModel):
    """
    Structured outcome of the classification process.
    Captures complete lifecycle state, error tracking, and extraction result.
    """
    model_config = ConfigDict(validate_assignment=True)

    status: ClassificationStatus
    extraction: Optional[RAIDExtractionResult] = None
    error_category: Optional[ClassificationErrorCategory] = None
    error_message: Optional[str] = None
    signal_id: str
    source_type: str
    attempts: int = 1
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

