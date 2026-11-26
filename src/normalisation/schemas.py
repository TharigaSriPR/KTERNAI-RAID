"""
Signal Schema for normalized project inputs using Pydantic v2.
Provides standardized validation, type enforcement, and serialization
across all communication and project system channels.
"""

from enum import Enum
from typing import Optional, Dict, Any, Union
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class SourceType(str, Enum):
    """
    Supported ingestion sources across the project lifecycle.
    """
    EMAIL = "email"
    CHAT = "chat"
    MEETING_NOTES = "meeting_notes"
    PROJECT_PLAN = "project_plan"
    DEFECT = "defect"

    @classmethod
    def _missing_(cls, value: object):
        # Allow case-insensitive lookups (e.g. "Email" -> SourceType.EMAIL)
        if isinstance(value, str):
            val_lower = value.lower().strip()
            for member in cls:
                if member.value == val_lower:
                    return member
        return None


class ProjectSignal(BaseModel):
    """
    Normalized representation of any communication or project event signal.
    Validates input constraints, source types, and ISO 8601 timestamps.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        extra="ignore"
    )

    id: str = Field(
        default_factory=lambda: f"sig-{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the normalized signal"
    )
    source_type: SourceType = Field(
        ...,
        description="Origin channel type of the signal"
    )
    source_id: str = Field(
        default="",
        description="Identifier from the origin system (MessageId, ChatId, TaskId, DefectId)"
    )
    author: str = Field(
        default="Unknown",
        description="Sender, organizer, assignee, or reporter of the signal"
    )
    timestamp: str = Field(
        default="",
        description="Signal generation or receipt timestamp in ISO 8601 UTC format, or empty string if absent"
    )
    subject: str = Field(
        default="",
        description="Subject line, channel name, meeting title, or task header"
    )
    content: str = Field(
        ...,
        description="Body, transcript excerpt, message text, or defect description"
    )
    workstream: str = Field(
        default="General",
        description="Associated SAP module or workstream (FI, SD, MM, Basis, etc.)"
    )
    phase: str = Field(
        default="Realize",
        description="SAP Activate methodology phase (Prepare, Explore, Realize, Deploy, Run)"
    )
    raw_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original unmodified metadata dictionary from source system"
    )
    is_noise: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Flag indicating if signal was classified as noise (0=relevant, 1=noise)"
    )
    noise_reason: str = Field(
        default="",
        description="Explanation when signal is marked as noise"
    )
    user_id: str = Field(
        default="default_user",
        description="Owner user identifier for tenant isolation"
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_and_generate_id(cls, value: Any) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return f"sig-{uuid.uuid4().hex[:8]}"
        return str(value).strip()

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_not_empty(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            raise ValueError("Signal content cannot be empty or purely whitespace")
        return str(value).strip()

    @field_validator("source_type", mode="before")
    @classmethod
    def validate_and_coerce_source_type(cls, value: Any) -> SourceType:
        if isinstance(value, SourceType):
            return value
        if isinstance(value, str):
            cleaned = value.strip().lower()
            try:
                return SourceType(cleaned)
            except ValueError:
                allowed = [e.value for e in SourceType]
                raise ValueError(
                    f"Invalid source_type '{value}'. Allowed source types are: {allowed}"
                )
        raise ValueError(f"Invalid type for source_type: {type(value).__name__}. Expected string or SourceType.")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_and_format_timestamp(cls, value: Any) -> str:
        """
        Validates and standardises timestamps into standard ISO 8601 UTC format.
        Preserves the distinction between an actual source timestamp and missing timestamp:
        - If value is None, empty string, or whitespace: returns empty string "" without inventing a date.
        - If timezone-aware: standardises to UTC (+00:00).
        - If timezone-naive: documents assumption and interprets as UTC (+00:00).
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return ""

        if isinstance(value, datetime):
            if value.tzinfo is None:
                # Documented assumption: naive datetime is interpreted as UTC
                dt = value.replace(tzinfo=timezone.utc)
            else:
                dt = value.astimezone(timezone.utc)
            return dt.isoformat()

        if value is not None and not isinstance(value, str):
            value = str(value)

        if isinstance(value, str):
            val_str = value.strip()
            if not val_str:
                return ""

            # Try ISO format directly
            try:
                dt = datetime.fromisoformat(val_str)
                if dt.tzinfo is None:
                    # Documented assumption: naive ISO string is interpreted as UTC
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.isoformat()
            except ValueError:
                pass

            # Try RFC 2822 / Email date format
            try:
                import email.utils
                dt = email.utils.parsedate_to_datetime(val_str)
                if dt is not None:
                    if dt.tzinfo is None:
                        # Documented assumption: naive RFC 2822 date is interpreted as UTC
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt.isoformat()
            except (TypeError, ValueError, IndexError):
                pass

            # Try common datetime formats in enterprise exports
            common_formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%Y/%m/%d %H:%M:%S"
            ]
            for fmt in common_formats:
                try:
                    # Documented assumption: naive enterprise export date is interpreted as UTC
                    dt = datetime.strptime(val_str, fmt).replace(tzinfo=timezone.utc)
                    return dt.isoformat()
                except ValueError:
                    continue

            raise ValueError(
                f"Invalid timestamp format '{val_str}'. Expected ISO 8601 or standard date format."
            )

        raise ValueError(f"Unsupported timestamp type: {type(value).__name__}")

    @field_validator("is_noise", mode="before")
    @classmethod
    def validate_is_noise_flag(cls, value: Any) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            val_int = int(value)
            if val_int in (0, 1):
                return val_int
        raise ValueError("is_noise must be 0, 1, or boolean True/False")

    @model_validator(mode="after")
    def sync_source_id_and_id(self) -> "ProjectSignal":
        # Ensure source_id defaults to self.id if empty
        if not self.source_id:
            self.source_id = self.id
        return self

    @property
    def has_source_timestamp(self) -> bool:
        """Returns True if this signal contains a valid source timestamp, False if absent."""
        return bool(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """
        Backwards-compatible dictionary conversion helper.
        Uses Pydantic's model_dump().
        """
        return self.model_dump()
