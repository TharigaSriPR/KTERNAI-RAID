"""
Pydantic v2 schemas for the Rule-Based Predictive Engine (Foresight).
Provides strict validation, category Enums, confidence/rule_score levels, and traceability fields.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field, ConfigDict, model_validator


class PredictionType(str, Enum):
    """Types of foresight patterns detected by rule evaluation."""
    TASK_OVERDUE = "TASK_OVERDUE"
    UPCOMING_DELAY = "UPCOMING_DELAY"
    DEFECT_CLUSTER = "DEFECT_CLUSTER"
    ASSUMPTION_CONTRADICTED = "ASSUMPTION_CONTRADICTED"
    DEPENDENCY_BLOCKAGE = "DEPENDENCY_BLOCKAGE"


class PredictionStatus(str, Enum):
    """Lifecycle status of a generated predictive finding."""
    DETECTED = "DETECTED"
    REVIEWED = "REVIEWED"
    DISMISSED = "DISMISSED"
    CONVERTED_TO_RAID = "CONVERTED_TO_RAID"


class PredictionSeverity(str, Enum):
    """Severity classification for predictive risks and issues."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class PredictionItem(BaseModel):
    """
    Validated predictive finding produced by the rule-based foresight engine.
    Preserves exact evidence and links directly to triggering ProjectSignal IDs.
    
    NOTE ON CONFIDENCE / RULE_SCORE:
    The confidence value is a deterministic rule evaluation score (0.0 to 1.0)
    measuring condition strength and evidence completeness. It is NOT a calibrated
    statistical or event probability.
    """
    model_config = ConfigDict(
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )

    prediction_id: str = Field(
        default_factory=lambda: f"pred-{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the predictive finding"
    )
    prediction_type: PredictionType = Field(
        ...,
        description="Categorical type of predictive pattern detected"
    )
    title: str = Field(
        ...,
        description="Concise, actionable title summarizing the foresight risk or issue"
    )
    description: str = Field(
        ...,
        description="Detailed explanation of the predictive pattern and anticipated impact"
    )
    suggested_raid_category: str = Field(
        default="Risk",
        description="Suggested RAID category (Risk or Issue)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deterministic rule score reflecting condition match strength, NOT a statistical probability"
    )
    rule_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Explicit rule-score alias for confidence, clarifying it is a deterministic rule metric"
    )
    severity: Optional[str] = Field(
        default="Medium",
        description="Assessed severity level (Low, Medium, High, Critical)"
    )
    evidence: str = Field(
        ...,
        description="Original source evidence, schedule data, or contradictory quotation"
    )
    source_signal_ids: List[str] = Field(
        default_factory=list,
        description="List of ProjectSignal IDs that triggered this prediction"
    )
    related_raid_ids: List[str] = Field(
        default_factory=list,
        description="List of related existing RAID IDs (e.g., contradicted assumption)"
    )
    workstream: Optional[str] = Field(
        default="General",
        description="Associated SAP workstream or module (FI, MM, SD, Basis, etc.)"
    )
    phase: Optional[str] = Field(
        default="Realize",
        description="SAP Activate phase (Prepare, Explore, Realize, Deploy, Run)"
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the foresight pattern was detected"
    )
    status: PredictionStatus = Field(
        default=PredictionStatus.DETECTED,
        description="Lifecycle status of this prediction item"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rule evaluation metadata (e.g. days overdue, cluster size)"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_rule_score_and_confidence(cls, data: Any) -> Any:
        if isinstance(data, dict):
            conf = data.get("confidence")
            r_score = data.get("rule_score")
            if conf is not None and r_score is None:
                data["rule_score"] = conf
            elif r_score is not None and conf is None:
                data["confidence"] = r_score
        return data
