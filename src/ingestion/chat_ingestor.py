"""
MS Teams / Chat Ingestion Adapter.
Supports:
- CSV exports (.csv)
- JSON exports (.json)
- Plain text transcript exports (.txt) with multiline support
Extracts author, timestamp, channel, and message content, producing validated ProjectSignals.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import csv
import json
import re

from .base_ingestor import BaseIngestor
from src.normalisation.schemas import ProjectSignal, SourceType

CSV_CHAT_ID_FIELDS = ["ChatId", "Chat_Id", "MessageId", "Message_Id", "Id", "MsgId"]
CSV_SENDER_FIELDS = ["Sender", "Author", "From", "User", "Speaker", "Username"]
CSV_TIME_FIELDS = ["Timestamp", "Date", "Time", "DateTime", "Created_At"]
CSV_CHANNEL_FIELDS = ["Channel", "Thread", "Subject", "ChatName", "Room"]
CSV_MSG_FIELDS = ["Message", "Content", "Text", "Body"]
CSV_WORKSTREAM_FIELDS = ["Workstream", "Module", "Track", "Area"]
CSV_PHASE_FIELDS = ["Phase", "Stage", "ProjectPhase"]


class ChatIngestor(BaseIngestor):
    """
    Ingestion adapter for project chat channels (MS Teams, Slack, transcript exports).
    Supports CSV, JSON, and structured multiline text transcripts.
    """

    def __init__(self):
        super().__init__(SourceType.CHAT)

    def ingest(self, file_path: Union[str, Path]) -> List[ProjectSignal]:
        """
        Ingests chat records from a file path (.csv, .json, .txt).
        Returns a list of validated ProjectSignal objects in chronological/source order.
        """
        path = self._validate_file_path(file_path)
        if not path:
            return []

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return self._ingest_csv(path)
            elif suffix == ".json":
                return self._ingest_json(path)
            elif suffix == ".txt":
                return self._ingest_txt(path)
            else:
                self.logger.warning("Unsupported chat file extension '%s' for file: %s", suffix, path)
                return []
        except Exception as exc:
            self.logger.error("Failed to ingest chat file '%s': %s", path, exc, exc_info=True)
            return []

    def _ingest_csv(self, path: Path) -> List[ProjectSignal]:
        """
        Ingests CSV chat exports with flexible headers and row resilience.
        """
        signals: List[ProjectSignal] = []

        try:
            with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()
        except Exception:
            with open(path, mode="r", encoding="latin-1", errors="replace") as f:
                content = f.read()

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            self.logger.warning("Chat CSV file '%s' is empty.", path.name)
            return []

        reader = csv.DictReader(lines)
        if not reader.fieldnames:
            self.logger.warning("Chat CSV file '%s' has no headers.", path.name)
            return []

        for idx, row in enumerate(reader, start=1):
            msg = str(self._get_field(row, CSV_MSG_FIELDS, default="")).strip()
            if not msg:
                continue

            try:
                chat_id = str(self._get_field(row, CSV_CHAT_ID_FIELDS, default=f"{path.stem}-chat-{idx}")).strip()
                author = str(self._get_field(row, CSV_SENDER_FIELDS, default="Unknown")).strip() or "Unknown"
                timestamp = self._get_field(row, CSV_TIME_FIELDS, default=None)
                channel = str(self._get_field(row, CSV_CHANNEL_FIELDS, default="Teams Channel")).strip() or "Teams Channel"
                workstream = str(self._get_field(row, CSV_WORKSTREAM_FIELDS, default="General")).strip() or "General"
                phase = str(self._get_field(row, CSV_PHASE_FIELDS, default="Realize")).strip() or "Realize"

                sig = ProjectSignal(
                    source_type=SourceType.CHAT,
                    source_id=chat_id,
                    author=author,
                    timestamp=timestamp,
                    subject=channel,
                    content=msg,
                    workstream=workstream,
                    phase=phase,
                    raw_meta={"source_file": path.name, "original_format": "csv", **dict(row)}
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error parsing chat row %d in %s: %s", idx, path.name, exc)
                continue

        self.logger.info("Ingested %d chat signals from CSV %s", len(signals), path.name)
        return signals

    def _ingest_json(self, path: Path) -> List[ProjectSignal]:
        """
        Ingests JSON chat exports supporting array of messages, wrapped objects,
        or single message dictionaries.
        """
        signals: List[ProjectSignal] = []

        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            content = path.read_text(encoding="latin-1", errors="replace")

        try:
            data = json.loads(content)
        except Exception as exc:
            self.logger.error("Failed to parse JSON file '%s': %s", path.name, exc)
            return []

        # Resolve records container
        records: List[Dict[str, Any]] = []
        if isinstance(data, list):
            records = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            for key in ["messages", "chats", "data", "conversations", "items"]:
                if key in data and isinstance(data[key], list):
                    records = [item for item in data[key] if isinstance(item, dict)]
                    break
            if not records:
                # Check if the dict itself is a single message
                if any(k in data for k in ["message", "content", "text"]):
                    records = [data]

        for idx, item in enumerate(records, start=1):
            msg = str(self._get_field(item, CSV_MSG_FIELDS, default="")).strip()
            if not msg:
                continue

            try:
                msg_id = str(self._get_field(item, CSV_CHAT_ID_FIELDS, default=f"{path.stem}-msg-{idx}")).strip()
                author = str(self._get_field(item, CSV_SENDER_FIELDS, default="Unknown")).strip() or "Unknown"
                timestamp = self._get_field(item, CSV_TIME_FIELDS, default=None)
                channel = str(self._get_field(item, CSV_CHANNEL_FIELDS, default="Teams Chat")).strip() or "Teams Chat"
                workstream = str(self._get_field(item, CSV_WORKSTREAM_FIELDS, default="General")).strip() or "General"
                phase = str(self._get_field(item, CSV_PHASE_FIELDS, default="Realize")).strip() or "Realize"

                sig = ProjectSignal(
                    source_type=SourceType.CHAT,
                    source_id=msg_id,
                    author=author,
                    timestamp=timestamp,
                    subject=channel,
                    content=msg,
                    workstream=workstream,
                    phase=phase,
                    raw_meta={"source_file": path.name, "original_format": "json", **item}
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error parsing chat JSON item %d in %s: %s", idx, path.name, exc)
                continue

        self.logger.info("Ingested %d chat signals from JSON %s", len(signals), path.name)
        return signals

    def _ingest_txt(self, path: Path) -> List[ProjectSignal]:
        """
        Ingests plain text chat transcripts with multiline message preservation.
        Recognizes common transcript headers (bracket timestamps, log timestamps, author colons).
        """
        signals: List[ProjectSignal] = []

        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            content = path.read_text(encoding="latin-1", errors="replace")

        # Recognizer patterns for message headers
        # [2026-11-20 10:15] Author: Message
        pattern_bracket = re.compile(r"^\[(?P<time>[^\]]+)\]\s*(?P<author>[^:]+):\s*(?P<msg>.*)$")
        # 2026-11-20 10:15 - Author: Message
        pattern_hyphen = re.compile(r"^(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)\s*-\s*(?P<author>[^:]+):\s*(?P<msg>.*)$")
        # Author (10:15): Message
        pattern_paren = re.compile(r"^(?P<author>[A-Za-z0-9_\. ]{2,30})\s*\((?P<time>[^)]+)\):\s*(?P<msg>.*)$")
        # Author: Message (clean speaker prefix)
        pattern_simple = re.compile(r"^(?P<author>[A-Za-z0-9_\. ]{2,30}):\s*(?P<msg>.*)$")

        lines = content.splitlines()
        parsed_entries: List[Dict[str, Any]] = []
        current_entry: Optional[Dict[str, Any]] = None

        for idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str:
                # Blank line within a multiline message
                if current_entry:
                    current_entry["lines"].append("")
                continue

            # Check if this line starts a new message
            matched = False
            for pat in [pattern_bracket, pattern_hyphen, pattern_paren]:
                m = pat.match(line_str)
                if m:
                    if current_entry and current_entry["lines"]:
                        parsed_entries.append(current_entry)
                    current_entry = {
                        "line_num": idx,
                        "timestamp": m.group("time").strip() if "time" in m.groupdict() else None,
                        "author": m.group("author").strip(),
                        "lines": [m.group("msg").strip()] if m.group("msg").strip() else []
                    }
                    matched = True
                    break

            if not matched:
                m_simple = pattern_simple.match(line_str)
                # Ensure the author part doesn't look like an ordinary URL or time
                if m_simple and not line_str.lower().startswith(("http:", "https:", "note:", "ps:")):
                    if current_entry and current_entry["lines"]:
                        parsed_entries.append(current_entry)
                    current_entry = {
                        "line_num": idx,
                        "timestamp": None,
                        "author": m_simple.group("author").strip(),
                        "lines": [m_simple.group("msg").strip()] if m_simple.group("msg").strip() else []
                    }
                    matched = True

            if not matched:
                # Continuation line of active message or standalone first line
                if current_entry is not None:
                    current_entry["lines"].append(line_str)
                else:
                    current_entry = {
                        "line_num": idx,
                        "timestamp": None,
                        "author": "Unknown",
                        "lines": [line_str]
                    }

        if current_entry and current_entry["lines"]:
            parsed_entries.append(current_entry)

        # Convert entries to ProjectSignal instances
        for item_idx, entry in enumerate(parsed_entries, start=1):
            full_msg = "\n".join(entry["lines"]).strip()
            if not full_msg:
                continue

            try:
                sig = ProjectSignal(
                    source_type=SourceType.CHAT,
                    source_id=f"{path.stem}-line-{entry['line_num']}",
                    author=entry["author"] or "Unknown",
                    timestamp=entry["timestamp"],
                    subject=path.stem,
                    content=full_msg,
                    raw_meta={
                        "source_file": path.name,
                        "original_format": "txt",
                        "line_number": entry["line_num"],
                        "message_order": item_idx
                    }
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error parsing chat transcript item %d in %s: %s", item_idx, path.name, exc)
                continue

        self.logger.info("Ingested %d chat signals from TXT %s", len(signals), path.name)
        return signals
