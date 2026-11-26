"""
Meeting Notes / MOM (Minutes of Meeting) Ingestion Adapter.
Supports:
- Microsoft Word documents (.docx) via python-docx
- Plain text notes / transcripts (.txt)
- Tabular meeting exports (.csv)
- Direct manual meeting minutes text entry
Extracts discussion points, decisions, actions, attendees, and dates,
producing validated ProjectSignals without fabricating missing information.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import csv
import re

from .base_ingestor import BaseIngestor
from src.normalisation.schemas import ProjectSignal, SourceType

try:
    import docx
except ImportError:
    docx = None

DATE_PATTERN = re.compile(
    r"(?:Date|Meeting Date):\s*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4}|[A-Za-z]+ \d{1,2},? \d{4})",
    re.IGNORECASE
)

ATTENDEES_PATTERN = re.compile(
    r"(?:Attendees|Participants|Present|Members):\s*(.+)",
    re.IGNORECASE
)

CSV_NOTES_FIELDS = ["Notes", "Transcript", "Content", "Description", "Discussion", "Minutes", "Summary"]
CSV_ORGANIZER_FIELDS = ["Organizer", "Scribe", "Author", "Chair", "Presenter", "Owner", "From"]
CSV_DATE_FIELDS = ["Date", "MeetingDate", "Meeting_Date", "Timestamp", "DateTime"]
CSV_TITLE_FIELDS = ["Title", "MeetingTitle", "Meeting_Name", "Subject", "Topic", "Name"]
CSV_WORKSTREAM_FIELDS = ["Workstream", "Module", "Track", "Area"]
CSV_PHASE_FIELDS = ["Phase", "Stage", "ProjectPhase"]
CSV_ID_FIELDS = ["MeetingId", "Meeting_Id", "Id", "MOM_Id"]


class MeetingIngestor(BaseIngestor):
    """
    Ingestion adapter for Meeting Minutes and discussion notes.
    Extracts paragraphs, tables, attendees, and structured discussion points.
    """

    def __init__(self):
        super().__init__(SourceType.MEETING_NOTES)

    def ingest(
        self,
        file_path: Union[str, Path],
        original_filename: Optional[str] = None,
        meeting_date: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[ProjectSignal]:
        """
        Ingests meeting notes from a file (.docx, .txt, .csv).
        Supports optional metadata (original_filename, meeting_date, project_name).
        """
        path = self._validate_file_path(file_path)
        if not path:
            return []

        suffix = path.suffix.lower()
        effective_filename = original_filename or path.name

        try:
            if suffix == ".docx":
                return self._ingest_docx(
                    path,
                    original_filename=effective_filename,
                    meeting_date=meeting_date,
                    project_name=project_name
                )
            elif suffix == ".txt":
                return self._ingest_txt(
                    path,
                    original_filename=effective_filename,
                    meeting_date=meeting_date,
                    project_name=project_name
                )
            elif suffix == ".csv":
                return self._ingest_csv(
                    path,
                    original_filename=effective_filename,
                    meeting_date=meeting_date,
                    project_name=project_name
                )
            else:
                self.logger.warning("Unsupported meeting notes file extension '%s' for file: %s", suffix, path)
                return []
        except Exception as exc:
            self.logger.error("Failed to ingest meeting notes '%s': %s", path, exc, exc_info=True)
            return []

    def _extract_date_from_text(self, text: str) -> Optional[str]:
        """Tries to extract a meeting date string from document header text."""
        match = DATE_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_attendees_from_text(self, text: str) -> Optional[str]:
        """Tries to extract an attendee list from document header text."""
        match = ATTENDEES_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _ingest_docx(
        self,
        path: Path,
        original_filename: Optional[str] = None,
        meeting_date: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[ProjectSignal]:
        """
        Ingests Microsoft Word (.docx) meeting minutes.
        Extracts paragraphs, table rows (with merged-cell deduplication), and document metadata.
        """
        if docx is None:
            self.logger.error("python-docx is not installed. Cannot parse .docx file: %s", path)
            return []

        signals: List[ProjectSignal] = []
        try:
            doc = docx.Document(str(path))
        except Exception as exc:
            self.logger.error("Error opening or parsing DOCX document '%s': %s", path, exc)
            return []

        source_file = original_filename or path.name
        stem = Path(source_file).stem
        title = stem

        # Extract text from paragraphs
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Extract table rows with merged-cell deduplication
        for table in doc.tables:
            for row in table.rows:
                raw_cells = [c.text.strip() for c in row.cells if c.text.strip()]
                # Deduplicate consecutive identical cells caused by merged cells in Word
                deduped_cells: List[str] = []
                for cell in raw_cells:
                    if not deduped_cells or deduped_cells[-1] != cell:
                        deduped_cells.append(cell)
                if deduped_cells:
                    paragraphs.append(" | ".join(deduped_cells))

        if not paragraphs:
            self.logger.warning("DOCX meeting notes file '%s' has no readable text.", source_file)
            return []

        # Check for date and attendees in early paragraphs
        extracted_date = meeting_date
        extracted_attendees = None
        header_sample = " ".join(paragraphs[:5])
        if not extracted_date:
            extracted_date = self._extract_date_from_text(header_sample)
        extracted_attendees = self._extract_attendees_from_text(header_sample)

        for idx, text_block in enumerate(paragraphs, start=1):
            if len(text_block) < 5:
                continue

            try:
                sig_meta: Dict[str, Any] = {
                    "source_file": source_file,
                    "original_format": "docx",
                    "paragraph_index": idx
                }
                if extracted_date:
                    sig_meta["meeting_date"] = extracted_date
                if extracted_attendees:
                    sig_meta["attendees"] = extracted_attendees
                if project_name:
                    sig_meta["project_name"] = project_name

                sig = ProjectSignal(
                    source_type=SourceType.MEETING_NOTES,
                    source_id=f"{stem}-para-{idx}",
                    author="Meeting Scribe",
                    timestamp=extracted_date,
                    subject=f"{title} (Item {idx})",
                    content=text_block,
                    workstream="Cross-Functional",
                    phase="Realize",
                    raw_meta=sig_meta
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error processing docx paragraph %d in %s: %s", idx, source_file, exc)
                continue

        self.logger.info("Ingested %d meeting signals from DOCX %s", len(signals), source_file)
        return signals

    @staticmethod
    def _chunk_txt(content: str) -> List[str]:
        """
        Splits TXT meeting notes into meaningful discussion chunks.

        Chunking rules:
        1. Preserves existing double-newline separation (\n\n) between paragraphs.
        2. Within paragraphs or single blocks, additionally recognizes numbered list
           items (e.g., '1.', '2)', '(1)') and bullet list items (e.g., '-', '*', '•', '–', '—', '+').
        3. Preserves ordinary sentences spanning multiple lines without splitting them.
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        if not paragraphs:
            paragraphs = [line.strip() for line in content.splitlines() if line.strip()]

        list_marker_pattern = re.compile(
            r'^\s*(?:(?:\d+[\.\)]|\(\d+\))|[-*•–—+])\s+'
        )

        chunks: List[str] = []
        for paragraph in paragraphs:
            lines = paragraph.splitlines()
            has_list_items = any(list_marker_pattern.match(line) for line in lines)
            if not has_list_items:
                # Normal paragraph without list items; preserve intact
                chunks.append(paragraph.strip())
                continue

            current_chunk_lines: List[str] = []
            for line in lines:
                if list_marker_pattern.match(line):
                    if current_chunk_lines:
                        chunk_text = "\n".join(current_chunk_lines).strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                    current_chunk_lines = [line]
                else:
                    current_chunk_lines.append(line)

            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)

        return chunks

    def _ingest_txt(
        self,
        path: Path,
        original_filename: Optional[str] = None,
        meeting_date: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[ProjectSignal]:
        """
        Ingests plain text meeting notes with chunking and metadata detection.
        """
        signals: List[ProjectSignal] = []

        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            content = path.read_text(encoding="latin-1", errors="replace")

        if not content.strip():
            self.logger.warning("TXT meeting notes file '%s' is empty or whitespace only.", path)
            return []

        source_file = original_filename or path.name
        stem = Path(source_file).stem

        # Extract meeting date and attendees from initial text
        extracted_date = meeting_date
        header_sample = content[:600]
        if not extracted_date:
            extracted_date = self._extract_date_from_text(header_sample)
        extracted_attendees = self._extract_attendees_from_text(header_sample)

        # Split by double newline or bullet/numbered lines to get meaningful discussion chunks
        chunks = self._chunk_txt(content)

        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) < 5:
                continue

            try:
                sig_meta: Dict[str, Any] = {
                    "source_file": source_file,
                    "original_format": "txt",
                    "chunk_index": idx
                }
                if extracted_date:
                    sig_meta["meeting_date"] = extracted_date
                if extracted_attendees:
                    sig_meta["attendees"] = extracted_attendees
                if project_name:
                    sig_meta["project_name"] = project_name

                sig = ProjectSignal(
                    source_type=SourceType.MEETING_NOTES,
                    source_id=f"{stem}-chunk-{idx}",
                    author="Meeting Scribe",
                    timestamp=extracted_date,
                    subject=f"{stem} (Discussion {idx})",
                    content=chunk,
                    workstream="Cross-Functional",
                    phase="Realize",
                    raw_meta=sig_meta
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error processing text chunk %d in %s: %s", idx, source_file, exc)
                continue

        self.logger.info("Ingested %d meeting signals from TXT %s", len(signals), source_file)
        return signals

    def _ingest_csv(
        self,
        path: Path,
        original_filename: Optional[str] = None,
        meeting_date: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[ProjectSignal]:
        """
        Ingests CSV meeting notes with flexible header resolution.
        """
        signals: List[ProjectSignal] = []
        source_file = original_filename or path.name
        stem = Path(source_file).stem

        try:
            with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()
        except Exception:
            with open(path, mode="r", encoding="latin-1", errors="replace") as f:
                content = f.read()

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            self.logger.warning("Meeting CSV file '%s' is empty.", source_file)
            return []

        reader = csv.DictReader(lines)
        if not reader.fieldnames:
            self.logger.warning("CSV meeting notes file '%s' has no headers or is empty.", source_file)
            return []

        for idx, row in enumerate(reader, start=1):
            notes = str(self._get_field(row, CSV_NOTES_FIELDS, default="")).strip()
            if not notes:
                continue

            row_date = self._get_field(row, CSV_DATE_FIELDS, default=meeting_date)
            row_id = str(self._get_field(row, CSV_ID_FIELDS, default=f"{stem}-mom-{idx}")).strip()
            author = str(self._get_field(row, CSV_ORGANIZER_FIELDS, default="Meeting Scribe")).strip() or "Meeting Scribe"
            subject = str(self._get_field(row, CSV_TITLE_FIELDS, default=stem)).strip() or stem
            workstream = str(self._get_field(row, CSV_WORKSTREAM_FIELDS, default="Cross-Functional")).strip() or "Cross-Functional"
            phase = str(self._get_field(row, CSV_PHASE_FIELDS, default="Realize")).strip() or "Realize"

            sig_meta: Dict[str, Any] = {
                "source_file": source_file,
                "original_format": "csv",
                **dict(row)
            }
            if row_date:
                sig_meta["meeting_date"] = row_date
            if project_name:
                sig_meta["project_name"] = project_name

            try:
                sig = ProjectSignal(
                    source_type=SourceType.MEETING_NOTES,
                    source_id=row_id,
                    author=author,
                    timestamp=row_date,
                    subject=subject,
                    content=notes,
                    workstream=workstream,
                    phase=phase,
                    raw_meta=sig_meta
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error processing meeting CSV row %d in %s: %s", idx, source_file, exc)
                continue

        self.logger.info("Ingested %d meeting signals from CSV %s", len(signals), source_file)
        return signals

    def ingest_text(
        self,
        content: str,
        title: str = "Meeting Minutes",
        meeting_date: Optional[str] = None,
        project_name: Optional[str] = None,
        author: str = "Meeting Scribe",
        workstream: str = "Cross-Functional",
        phase: str = "Realize"
    ) -> List[ProjectSignal]:
        """
        Directly ingests pasted meeting minutes text into ProjectSignal objects.
        Preserves meeting title, meeting date, project name, author, and chunk IDs.
        """
        if not content or not content.strip():
            self.logger.warning("Manual meeting minutes text is empty.")
            return []

        cleaned_content = content.strip()
        extracted_date = meeting_date
        header_sample = cleaned_content[:600]
        if not extracted_date:
            extracted_date = self._extract_date_from_text(header_sample)
        extracted_attendees = self._extract_attendees_from_text(header_sample)

        # Clean title slug for deterministic source_id
        clean_slug = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title.strip()).strip('_')[:30] or "mom"

        # Split into discussion points / chunks (paragraphs or bullet lines)
        chunks = [c.strip() for c in cleaned_content.split("\n\n") if c.strip()]
        if len(chunks) == 1:
            # Check if text contains numbered or bulleted lines
            bullet_lines = [
                line.strip() for line in cleaned_content.splitlines()
                if line.strip() and (line.strip().startswith(("-", "*", "•")) or re.match(r"^\d+[\.\)]", line.strip()))
            ]
            if len(bullet_lines) >= 2:
                chunks = bullet_lines

        signals: List[ProjectSignal] = []
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) < 5:
                continue

            sig_meta: Dict[str, Any] = {
                "source_file": "manual_entry",
                "original_format": "text",
                "meeting_title": title.strip(),
                "chunk_index": idx
            }
            if extracted_date:
                sig_meta["meeting_date"] = extracted_date
            if extracted_attendees:
                sig_meta["attendees"] = extracted_attendees
            if project_name:
                sig_meta["project_name"] = project_name

            try:
                sig = ProjectSignal(
                    source_type=SourceType.MEETING_NOTES,
                    source_id=f"manual-{clean_slug}-chunk-{idx}",
                    author=author,
                    timestamp=extracted_date,
                    subject=f"{title.strip()} (Discussion {idx})",
                    content=chunk,
                    workstream=workstream,
                    phase=phase,
                    raw_meta=sig_meta
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error processing manual meeting chunk %d: %s", idx, exc)
                continue

        self.logger.info("Ingested %d meeting signals from manual text entry '%s'", len(signals), title)
        return signals
