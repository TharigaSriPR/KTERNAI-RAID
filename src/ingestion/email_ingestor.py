"""
Email and Outlook Ingestion Adapter.
Supports:
- CSV exports (.csv)
- Standard MIME email files (.eml)
- Text email exports (.txt)
Extracts sender, subject, timestamp, and body, removes signatures conservatively,
converts HTML parts to clean text, and produces validated ProjectSignals.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import csv
import email
from email import policy
import re

from .base_ingestor import BaseIngestor, extract_text_from_html
from src.normalisation.schemas import ProjectSignal, SourceType

# Conservative signature detection patterns
# Strictly matches sign-offs on their own line (optionally followed by punctuation)
# Never matches running prose such as "Regards our discussion yesterday..."
SIGNATURE_PATTERNS = [
    re.compile(r"(?m)^--\s*$", re.IGNORECASE),
    re.compile(r"(?mi)^(?:best|warm|kind|warmest)?\s*regards\s*[,.-]?\s*$"),
    re.compile(r"(?mi)^(?:thanks(?:\s+and|\s*&)?\s+regards|many\s+thanks|thank\s+you)\s*[,.-]?\s*$"),
    re.compile(r"(?mi)^(?:sincerely(?:\s+yours)?|yours\s+sincerely|yours\s+truly)\s*[,.-]?\s*$"),
    re.compile(r"(?mi)^cheers\s*[,.-]?\s*$"),
    re.compile(r"(?mi)^sent\s+from\s+my\s+(?:iphone|android|ipad|galaxy|mobile|device).*$"),
    re.compile(r"(?mi)^get\s+outlook\s+for\s+(?:ios|android).*$")
]

# Supported header variations for CSV emails
CSV_ID_FIELDS = ["MessageId", "Message-ID", "Message_Id", "Id", "MsgId", "EmailId"]
CSV_FROM_FIELDS = ["From", "Sender", "Author", "From_Email", "Sender_Email"]
CSV_DATE_FIELDS = ["Date", "Timestamp", "SentDate", "Sent_Date", "DateTime", "Time"]
CSV_SUBJECT_FIELDS = ["Subject", "Title", "Topic", "Header"]
CSV_BODY_FIELDS = ["Body", "Content", "Message", "Text", "Email_Body", "Description", "Notes"]
CSV_WORKSTREAM_FIELDS = ["Workstream", "Module", "Track", "Area"]
CSV_PHASE_FIELDS = ["Phase", "Stage", "ProjectPhase"]


class EmailIngestor(BaseIngestor):
    """
    Ingestion adapter for email sources (CSV, EML, TXT).
    Extracts structured attributes, strips signatures conservatively,
    converts HTML to clean text, and creates validated ProjectSignal instances.
    """

    def __init__(self):
        super().__init__(SourceType.EMAIL)

    @staticmethod
    def strip_signature(body: str) -> str:
        """
        Removes common email signature blocks using conservative pattern matching.
        Preserves all project content prior to the signature delimiter.
        """
        if not body:
            return ""
        cleaned = body
        for pat in SIGNATURE_PATTERNS:
            match = pat.search(cleaned)
            if match:
                # Strip only from the match start to the end
                candidate = cleaned[:match.start()].strip()
                # Ensure we don't wipe out the entire body if match occurred at index 0
                if candidate:
                    cleaned = candidate
        return cleaned.strip()

    def ingest(self, file_path: Union[str, Path]) -> List[ProjectSignal]:
        """
        Ingests emails from a file path (.csv, .eml, .txt).
        Returns a list of validated ProjectSignal objects.
        """
        path = self._validate_file_path(file_path)
        if not path:
            return []

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return self._ingest_csv(path)
            elif suffix == ".eml":
                return self._ingest_eml(path)
            elif suffix == ".txt":
                return self._ingest_txt(path)
            else:
                self.logger.warning("Unsupported email file extension '%s' for file: %s", suffix, path)
                return []
        except Exception as exc:
            self.logger.error("Failed to ingest email file '%s': %s", path, exc, exc_info=True)
            return []

    def _ingest_csv(self, path: Path) -> List[ProjectSignal]:
        """
        Reads CSV email exports with support for UTF-8 with BOM, Latin-1 fallback,
        flexible header resolution, and row-level resilience.
        """
        signals: List[ProjectSignal] = []

        # Read content handling BOM and encodings gracefully
        try:
            with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()
        except Exception:
            with open(path, mode="r", encoding="latin-1", errors="replace") as f:
                content = f.read()

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            self.logger.warning("Email CSV file '%s' is empty.", path.name)
            return []

        reader = csv.DictReader(lines)
        if not reader.fieldnames:
            self.logger.warning("Email CSV file '%s' has no headers.", path.name)
            return []

        for row_idx, row in enumerate(reader, start=1):
            try:
                raw_body = self._get_field(row, CSV_BODY_FIELDS, default="")
                if not raw_body:
                    self.logger.debug("Skipping row %d in %s due to empty body content", row_idx, path.name)
                    continue

                cleaned_body = self.strip_signature(raw_body)
                if not cleaned_body:
                    self.logger.debug("Skipping row %d in %s: empty body after signature removal", row_idx, path.name)
                    continue

                source_id = str(self._get_field(row, CSV_ID_FIELDS, default=f"{path.stem}-row-{row_idx}")).strip()
                author = str(self._get_field(row, CSV_FROM_FIELDS, default="Unknown")).strip() or "Unknown"
                timestamp = self._get_field(row, CSV_DATE_FIELDS, default=None)
                subject = str(self._get_field(row, CSV_SUBJECT_FIELDS, default=f"Email-{source_id}")).strip() or path.stem
                workstream = str(self._get_field(row, CSV_WORKSTREAM_FIELDS, default="General")).strip() or "General"
                phase = str(self._get_field(row, CSV_PHASE_FIELDS, default="Explore")).strip() or "Explore"

                raw_meta = dict(row)
                raw_meta["source_file"] = path.name
                raw_meta["original_format"] = "csv"
                raw_meta["raw_body"] = raw_body

                sig = ProjectSignal(
                    source_type=SourceType.EMAIL,
                    source_id=source_id,
                    author=author,
                    timestamp=timestamp,
                    subject=subject,
                    content=cleaned_body,
                    workstream=workstream,
                    phase=phase,
                    raw_meta=raw_meta
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Failed to parse row %d in %s: %s", row_idx, path.name, exc)
                continue

        self.logger.info("Ingested %d email signals from CSV %s", len(signals), path.name)
        return signals

    def _ingest_eml(self, path: Path) -> List[ProjectSignal]:
        """
        Ingests a standard MIME email (.eml).
        Prefers plain text, converts HTML when plain text is unavailable,
        extracts attachment metadata, and preserves headers without leaking them into body.
        """
        try:
            with open(path, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
        except Exception as exc:
            self.logger.error("Failed to parse EML binary format for '%s': %s", path.name, exc)
            return []

        subject = str(msg.get("Subject", "")).strip() or path.stem
        sender = str(msg.get("From", "Unknown")).strip() or "Unknown"
        date_str = str(msg.get("Date", "")).strip() or None
        raw_msg_id = str(msg.get("Message-ID", "")).strip()
        message_id = raw_msg_id.strip("<>") if raw_msg_id else path.stem

        # Extract attachments and body content cleanly
        attachments: List[str] = []
        body_text = ""

        # Extract attachment names from message
        for part in msg.walk():
            disp = str(part.get_content_disposition() or "").lower()
            fn = part.get_filename()
            if fn or disp == "attachment":
                if fn and fn not in attachments:
                    attachments.append(fn)

        # First try email policy's get_body
        body_part = msg.get_body(preferencelist=('plain', 'html'))
        if body_part:
            try:
                ctype = body_part.get_content_type()
                raw_content = body_part.get_content()
                if ctype == "text/html":
                    body_text = extract_text_from_html(str(raw_content))
                else:
                    body_text = str(raw_content)
            except Exception as exc:
                self.logger.debug("get_body() content extraction failed for %s: %s", path.name, exc)

        # Fallback recursive multipart traversal if body_text is empty
        if not body_text.strip():
            plain_parts = []
            html_parts = []
            for part in msg.walk():
                disp = str(part.get_content_disposition() or "").lower()
                fn = part.get_filename()
                if fn or disp == "attachment":
                    continue

                ctype = part.get_content_type()
                if ctype == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        plain_parts.append(payload.decode(charset, errors="replace"))
                    except Exception:
                        pass
                elif ctype == "text/html":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        html_parts.append(payload.decode(charset, errors="replace"))
                    except Exception:
                        pass

            if plain_parts:
                body_text = "\n\n".join(plain_parts).strip()
            elif html_parts:
                body_text = extract_text_from_html("\n\n".join(html_parts))

        # Check for non-body fallback: never dump raw MIME headers into content
        if not body_text.strip():
            self.logger.warning("Email %s has no readable plain text or HTML body", path.name)
            return []

        cleaned_body = self.strip_signature(body_text)
        if not cleaned_body:
            self.logger.warning("Email %s has empty content after signature removal", path.name)
            return []

        # Header metadata capture
        header_keys = ["To", "Cc", "Date", "Subject", "From", "Message-ID"]
        headers_captured = {}
        for k in header_keys:
            v = msg.get(k)
            if v:
                headers_captured[k] = str(v)

        sig = ProjectSignal(
            source_type=SourceType.EMAIL,
            source_id=message_id,
            author=sender,
            timestamp=date_str,
            subject=subject,
            content=cleaned_body,
            workstream="General",
            phase="Explore",
            raw_meta={
                "source_file": path.name,
                "original_format": "eml",
                "headers": headers_captured,
                "attachments": attachments,
                "raw_body": body_text
            }
        )
        self.logger.info("Ingested 1 email signal from EML %s", path.name)
        return [sig]

    def _ingest_txt(self, path: Path) -> List[ProjectSignal]:
        """
        Ingests plain text email exports with header parsing.
        Prevents headers from leaking into body lines and handles headerless text safely.
        """
        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            content = path.read_text(encoding="latin-1", errors="replace")

        lines = content.splitlines()
        if not lines or not content.strip():
            self.logger.warning("Email text file %s is empty", path.name)
            return []

        sender = "Unknown"
        subject = ""
        date_str = None
        workstream = "General"
        phase = "Explore"
        body_lines = []
        in_body = False

        header_prefixes = ("from:", "subject:", "date:", "to:", "cc:", "workstream:", "phase:")

        for line in lines:
            if not in_body:
                line_lower = line.strip().lower()
                if line.strip() == "":
                    # Empty line marks end of header section
                    in_body = True
                    continue
                if line_lower.startswith("from:"):
                    sender = line.split(":", 1)[1].strip() or "Unknown"
                elif line_lower.startswith("subject:"):
                    subject = line.split(":", 1)[1].strip()
                elif line_lower.startswith("date:"):
                    date_str = line.split(":", 1)[1].strip()
                elif line_lower.startswith("workstream:"):
                    workstream = line.split(":", 1)[1].strip() or "General"
                elif line_lower.startswith("phase:"):
                    phase = line.split(":", 1)[1].strip() or "Explore"
                elif line_lower.startswith(("to:", "cc:")):
                    # Known email header; do not include in body
                    continue
                else:
                    # If line does not match known headers, header section has ended
                    in_body = True
                    body_lines.append(line)
            else:
                body_lines.append(line)

        raw_body = "\n".join(body_lines).strip()
        cleaned_body = self.strip_signature(raw_body)
        if not cleaned_body:
            self.logger.warning("Email text file %s has no body content", path.name)
            return []

        sig = ProjectSignal(
            source_type=SourceType.EMAIL,
            source_id=path.stem,
            author=sender,
            timestamp=date_str,
            subject=subject or path.stem,
            content=cleaned_body,
            workstream=workstream,
            phase=phase,
            raw_meta={
                "source_file": path.name,
                "original_format": "txt",
                "raw_body": raw_body
            }
        )
        self.logger.info("Ingested 1 email signal from TXT %s", path.name)
        return [sig]
