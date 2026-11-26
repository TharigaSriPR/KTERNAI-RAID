"""
Signal Normaliser module.
Converts raw dictionary payloads from various source ingestors into standardized ProjectSignal objects
with error handling, Pydantic validation, structured error reporting, and centralised logging.
Also provides consistent validation for pre-instantiated ProjectSignal instances.
"""

from typing import List, Dict, Any, Optional, Union, Sequence, Tuple
import json
from pydantic import ValidationError

from .schemas import ProjectSignal, SourceType
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Known fields on the ProjectSignal schema
KNOWN_SIGNAL_FIELDS = {
    "id",
    "user_id",
    "source_type",
    "source_id",
    "author",
    "timestamp",
    "subject",
    "content",
    "workstream",
    "phase",
    "raw_meta",
    "is_noise",
    "noise_reason"
}


class SignalNormaliser:
    """
    Transforms raw dictionaries and existing ProjectSignal instances into validated,
    standardised ProjectSignal instances with consistent field normalization,
    metadata preservation, and structured error reporting.
    """
    last_batch_errors: List[Dict[str, Any]] = []

    @classmethod
    def _sanitize_raw_meta(cls, raw_meta_val: Any) -> Dict[str, Any]:
        """
        Safely converts raw_meta into a dictionary without silently discarding information.
        - Dicts are copied.
        - JSON strings are parsed; if non-dict JSON, wrapped.
        - Other non-empty strings/values are preserved under 'raw_value'.
        - None or empty values default to empty dict {}.
        """
        if raw_meta_val is None:
            return {}
        if isinstance(raw_meta_val, dict):
            return dict(raw_meta_val)
        if isinstance(raw_meta_val, str):
            val_str = raw_meta_val.strip()
            if not val_str:
                return {}
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, dict):
                    return parsed
                return {"parsed_value": parsed, "raw": val_str}
            except Exception:
                return {"raw_value": val_str}
        return {"raw_value": raw_meta_val}

    @classmethod
    def normalise(cls, raw_record: Union[Dict[str, Any], ProjectSignal]) -> ProjectSignal:
        """
        Normalises a single record (dict or ProjectSignal) into a validated ProjectSignal.
        
        Applies consistent validation:
        1. For existing ProjectSignal:
           - Verifies content is non-empty and stripped.
           - Ensures source_id defaults to id if absent.
           - Normalises timestamp into ISO 8601 UTC string if present, or preserves empty string.
           - Sanitizes raw_meta and flags has_source_timestamp.
           - Sets enterprise defaults for author, workstream, and phase if blank.
           - Marks _normalised=True to avoid duplicate processing downstream.
        2. For dictionary inputs:
           - Extracts known signal attributes.
           - Preserves unknown source-specific fields in raw_meta['extra_fields'].
           - Sanitizes raw_meta preserving non-dict values.
           - Records original_timestamp before conversion.
           - Validates and constructs ProjectSignal via Pydantic.
           - Marks _normalised=True.
        """
        if raw_record is None:
            raise ValueError("Cannot normalise None record")

        # -------------------------------------------------------------------
        # 1. Handle pre-existing ProjectSignal instance
        # -------------------------------------------------------------------
        if isinstance(raw_record, ProjectSignal):
            # Avoid redundant processing if already standardised
            if raw_record.raw_meta.get("_normalised"):
                return raw_record

            content_str = str(raw_record.content or "").strip()
            if not content_str:
                raise ValueError("Signal content cannot be empty or purely whitespace")

            raw_meta = cls._sanitize_raw_meta(raw_record.raw_meta)

            # Standardise timestamp if present
            ts = raw_record.timestamp
            if ts and isinstance(ts, str) and ts.strip():
                standardised_ts = ProjectSignal.validate_and_format_timestamp(ts)
                raw_meta["has_source_timestamp"] = True
                if "original_timestamp" not in raw_meta:
                    raw_meta["original_timestamp"] = ts.strip()
            else:
                standardised_ts = ""
                raw_meta["has_source_timestamp"] = False

            source_id = str(raw_record.source_id or "").strip()
            if not source_id:
                source_id = raw_record.id

            raw_meta["_normalised"] = True

            raw_record.content = content_str
            raw_record.source_id = source_id
            raw_record.timestamp = standardised_ts
            raw_record.author = str(raw_record.author or "Unknown").strip() or "Unknown"
            raw_record.workstream = str(raw_record.workstream or "General").strip() or "General"
            raw_record.phase = str(raw_record.phase or "Realize").strip() or "Realize"
            raw_record.raw_meta = raw_meta

            logger.debug("Standardised existing ProjectSignal %s from source '%s'", raw_record.id, raw_record.source_type)
            return raw_record

        # -------------------------------------------------------------------
        # 2. Handle dictionary input
        # -------------------------------------------------------------------
        if not isinstance(raw_record, dict):
            raise ValueError(f"Expected dict or ProjectSignal, got {type(raw_record).__name__}")

        try:
            content_val = raw_record.get("content", "")
            if content_val is None:
                content_val = ""
            content_str = str(content_val).strip()
            if not content_str:
                raise ValueError("Signal content cannot be empty or purely whitespace")

            source_id_val = raw_record.get("source_id") or ""
            author_val = raw_record.get("author") or "Unknown"
            subject_val = raw_record.get("subject") or ""
            workstream_val = raw_record.get("workstream") or "General"
            phase_val = raw_record.get("phase") or "Realize"
            user_id_val = raw_record.get("user_id") or "default_user"

            # Safely handle raw_meta
            raw_meta = cls._sanitize_raw_meta(raw_record.get("raw_meta"))

            # Preserve unknown extra fields inside raw_meta['extra_fields']
            extra_fields = {
                k: v for k, v in raw_record.items()
                if k not in KNOWN_SIGNAL_FIELDS
            }
            if extra_fields:
                if "extra_fields" not in raw_meta:
                    raw_meta["extra_fields"] = extra_fields
                elif isinstance(raw_meta["extra_fields"], dict):
                    raw_meta["extra_fields"].update(extra_fields)
                else:
                    raw_meta["extra_fields"] = {"existing": raw_meta["extra_fields"], **extra_fields}

            # Preserve original timestamp if provided
            raw_ts = raw_record.get("timestamp")
            if raw_ts is not None and str(raw_ts).strip():
                raw_meta["original_timestamp"] = str(raw_ts)
                raw_meta["has_source_timestamp"] = True
            else:
                raw_meta["has_source_timestamp"] = False

            raw_meta["_normalised"] = True

            signal = ProjectSignal(
                id=raw_record.get("id") or None,
                user_id=str(user_id_val),
                source_type=raw_record.get("source_type"),
                source_id=str(source_id_val),
                author=str(author_val),
                timestamp=raw_ts,
                subject=str(subject_val),
                content=content_str,
                workstream=str(workstream_val),
                phase=str(phase_val),
                raw_meta=raw_meta,
                is_noise=raw_record.get("is_noise", 0),
                noise_reason=raw_record.get("noise_reason", "")
            )
            logger.debug("Successfully normalised signal %s from source '%s'", signal.id, signal.source_type)
            return signal

        except ValidationError as exc:
            logger.error("Validation failed normalising record: %s | Errors: %s", raw_record, exc.errors())
            raise
        except Exception as exc:
            logger.error("Unexpected error normalising record: %s | Error: %s", raw_record, str(exc))
            raise

    @classmethod
    def normalise_batch(
        cls,
        raw_records: Sequence[Union[Dict[str, Any], ProjectSignal]],
        skip_on_error: bool = False,
        return_errors: bool = False
    ) -> Union[List[ProjectSignal], Tuple[List[ProjectSignal], List[Dict[str, Any]]]]:
        """
        Normalises a collection of records (dicts or ProjectSignals).
        
        Args:
            raw_records: Sequence of input records.
            skip_on_error: If True, invalid records are logged as structured warnings and omitted.
                           If False, re-raises on first encountered error.
            return_errors: If True, returns a tuple (signals, errors).
                           If False, returns signals (maintains backward compatibility).
        """
        signals: List[ProjectSignal] = []
        errors: List[Dict[str, Any]] = []

        for index, record in enumerate(raw_records):
            try:
                sig = cls.normalise(record)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                rec_id = None
                src_type = None
                if isinstance(record, ProjectSignal):
                    rec_id = record.id
                    src_type = str(record.source_type)
                elif isinstance(record, dict):
                    rec_id = record.get("id") or record.get("source_id")
                    src_type = record.get("source_type")

                err_entry = {
                    "index": index,
                    "record_id": rec_id,
                    "source_type": src_type,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                errors.append(err_entry)

                if skip_on_error:
                    logger.warning(
                        "Skipping invalid record at index %d (id=%s, source=%s) due to %s: %s",
                        index, rec_id, src_type, type(exc).__name__, str(exc)
                    )
                    continue
                raise

        logger.info(
            "Batch normalisation complete: %d signals successfully created from %d inputs (%d errors)",
            len(signals), len(raw_records), len(errors)
        )
        cls.last_batch_errors = errors

        if return_errors:
            return signals, errors
        return signals

    @classmethod
    def normalise_batch_with_report(
        cls,
        raw_records: Sequence[Union[Dict[str, Any], ProjectSignal]],
        skip_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Executes batch normalisation and returns a structured summary report.
        """
        signals, errors = cls.normalise_batch(raw_records, skip_on_error=skip_on_error, return_errors=True)
        return {
            "signals": signals,
            "errors": errors,
            "total_input": len(raw_records),
            "success_count": len(signals),
            "failure_count": len(errors)
        }
