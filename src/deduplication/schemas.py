"""
Pydantic v2 schemas for Semantic Deduplication.
Defines match statuses, similarity score validation, and review recommendations.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class MatchStatus(str, Enum):
    """Classification of similarity match between a new item and an existing record."""
    LIKELY_DUPLICATE = "LIKELY_DUPLICATE"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"
    RELATED_NOT_DUPLICATE = "RELATED_NOT_DUPLICATE"
    NO_MATCH = "NO_MATCH"


class DeduplicationResult(BaseModel):
    """
    Validated outcome of comparing a candidate RAID item against an existing record.
    Explains the score and provides context on category and workstream alignment.
    """
    model_config = ConfigDict(
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )

    new_raid_id: Optional[str] = Field(
        default=None,
        description="ID of the new candidate RAID draft or signal"
    )
    new_signal_id: Optional[str] = Field(
        default=None,
        description="Originating source signal ID of the new item"
    )
    matched_raid_id: Optional[str] = Field(
        default=None,
        description="ID of the existing RAID draft or approved record that matched"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0 to 1.0) derived via TF-IDF"
    )
    match_status: MatchStatus = Field(
        ...,
        description="Determined match classification"
    )
    match_reason: str = Field(
        ...,
        description="Clear human-readable justification explaining the match status"
    )
    same_category: bool = Field(
        default=False,
        description="Whether both candidate and matched item share the same RAID category"
    )
    same_workstream: bool = Field(
        default=False,
        description="Whether both candidate and matched item share the same workstream"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Flag indicating if PM verification is recommended before merging or linking"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata (e.g. threshold settings, compared text snippets)"
    )
