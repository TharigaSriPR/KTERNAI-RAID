"""
Base class for all data ingestion adapters.
Provides standardized logging, file verification, HTML extraction, and schema helpers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import html
from html.parser import HTMLParser

from src.normalisation.schemas import ProjectSignal, SourceType
from src.utils.logger import get_logger


class SafeHTMLTextExtractor(HTMLParser):
    """
    Safely strips HTML tags and extracts visible human-readable text
    without external dependencies or security risks.
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed: List[str] = []
        self._ignore = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript", "svg"):
            self._ignore = True
        elif tag_lower in ("p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"):
            self.fed.append("\n")

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript", "svg"):
            self._ignore = False
        elif tag_lower in ("p", "div", "tr", "li", "blockquote"):
            self.fed.append("\n")

    def handle_data(self, data: str):
        if not self._ignore:
            self.fed.append(data)

    def get_text(self) -> str:
        raw_text = "".join(self.fed)
        raw_text = html.unescape(raw_text)
        lines = [line.strip() for line in raw_text.splitlines()]
        return "\n".join(chunk for chunk in lines if chunk).strip()


def extract_text_from_html(html_content: str) -> str:
    """
    Convenience function to parse an HTML string into cleaned plain text.
    Returns stripped plain text or empty string if input is empty.
    """
    if not html_content or not html_content.strip():
        return ""
    try:
        parser = SafeHTMLTextExtractor()
        parser.feed(html_content)
        parser.close()
        return parser.get_text()
    except Exception:
        # Fallback to basic regex stripping if parsing encountered malformed HTML
        import re
        stripped = re.sub(r"<[^>]+>", " ", html_content)
        return " ".join(stripped.split()).strip()


class BaseIngestor(ABC):
    """
    Abstract base class for signal ingestion adapters across all project channels.
    """

    def __init__(self, source_type: Union[SourceType, str]):
        self.source_type = SourceType(source_type) if isinstance(source_type, str) else source_type
        self.logger = get_logger(f"ingestion.{self.source_type.value}")

    @abstractmethod
    def ingest(self, file_path_or_data: Union[str, Path]) -> List[ProjectSignal]:
        """
        Ingest records from a file path or directory and return validated ProjectSignal objects.
        """
        pass

    def _validate_file_path(self, file_path: Union[str, Path]) -> Optional[Path]:
        """
        Validates that target file exists, is a regular file, and is non-empty.
        Returns Path object if valid, or None if invalid or empty.
        """
        path = Path(file_path)
        if not path.exists():
            self.logger.warning("Target file does not exist: %s", path)
            return None
        if not path.is_file():
            self.logger.warning("Target path is not a file: %s", path)
            return None
        try:
            if path.stat().st_size == 0:
                self.logger.warning("Target file is empty (0 bytes): %s", path)
                return None
        except OSError as exc:
            self.logger.warning("Cannot check file size for '%s': %s", path, exc)
            return None
        return path

    @staticmethod
    def _find_column(columns: Any, candidates: List[str]) -> Optional[str]:
        """
        Finds the matching column name from a list or collection of columns,
        matching case-insensitively and ignoring whitespace and BOM.
        """
        if columns is None or len(columns) == 0:
            return None
        cols_lower = {str(c).lstrip("\ufeff").strip().lower(): c for c in columns}
        for cand in candidates:
            cand_clean = cand.lstrip("\ufeff").strip().lower()
            if cand_clean in cols_lower:
                return cols_lower[cand_clean]
        return None

    @staticmethod
    def _get_field(mapping: Dict[str, Any], candidates: List[str], default: Any = None) -> Any:
        """
        Retrieves a value from a dictionary using candidate key names,
        checking case-insensitively and stripping whitespace/BOM.
        """
        if not mapping or not isinstance(mapping, dict):
            return default
        # Fast path exact match
        for cand in candidates:
            if cand in mapping and mapping[cand] is not None:
                val = str(mapping[cand]).strip() if isinstance(mapping[cand], str) else mapping[cand]
                if val != "":
                    return val

        # Case-insensitive / normalized lookup
        norm_keys = {str(k).lstrip("\ufeff").strip().lower(): k for k in mapping.keys()}
        for cand in candidates:
            cand_norm = cand.lstrip("\ufeff").strip().lower()
            if cand_norm in norm_keys:
                actual_k = norm_keys[cand_norm]
                val = mapping[actual_k]
                if val is not None:
                    val_clean = str(val).strip() if isinstance(val, str) else val
                    if val_clean != "":
                        return val_clean

        return default
