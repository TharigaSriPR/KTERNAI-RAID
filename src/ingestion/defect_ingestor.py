"""
Defects / Quality Tracker Ingestion Adapter.
Supports:
- Excel spreadsheets (.xlsx, .xls) via Pandas & OpenPyXL
- CSV defect exports (.csv)
Extracts defect ID, summary, severity, status, owner, module, and description into validated ProjectSignals.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import pandas as pd

from .base_ingestor import BaseIngestor
from src.normalisation.schemas import ProjectSignal, SourceType


class DefectIngestor(BaseIngestor):
    def __init__(self):
        super().__init__(SourceType.DEFECT)

    def ingest(self, file_path: Union[str, Path]) -> List[ProjectSignal]:
        path = self._validate_file_path(file_path)
        if not path:
            return []

        suffix = path.suffix.lower()
        try:
            if suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path, engine="openpyxl" if suffix == ".xlsx" else None)
            elif suffix == ".csv":
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig")
                except Exception:
                    df = pd.read_csv(path, encoding="latin-1")
            else:
                self.logger.warning("Unsupported defect file extension '%s' for file: %s", suffix, path)
                return []
            return self._process_dataframe(df, path)
        except Exception as exc:
            self.logger.error("Failed to ingest defect log '%s': %s", path, exc, exc_info=True)
            return []

    def _process_dataframe(self, df: pd.DataFrame, path: Path) -> List[ProjectSignal]:
        signals = []
        if df.empty:
            self.logger.warning("Defect log file is empty: %s", path.name)
            return []

        # Flexible column resolution using BaseIngestor helper
        id_col = self._find_column(df.columns, ["defectid", "defect id", "key", "issue id", "id", "bug id"])
        summary_col = self._find_column(df.columns, ["summary", "title", "subject", "defect name"])
        desc_col = self._find_column(df.columns, ["description", "details", "notes", "steps to reproduce"])
        sev_col = self._find_column(df.columns, ["severity", "priority", "impact"])
        status_col = self._find_column(df.columns, ["status", "state"])
        owner_col = self._find_column(df.columns, ["reportedby", "reporter", "assignee", "owner"])
        date_col = self._find_column(df.columns, ["createddate", "created date", "date", "created", "reported date"])
        module_col = self._find_column(df.columns, ["module", "workstream", "component", "area"])
        phase_col = self._find_column(df.columns, ["phase", "stage"])

        df_clean = df.fillna("")

        for idx, row in df_clean.iterrows():
            summary = str(row[summary_col]).strip() if summary_col else ""
            desc = str(row[desc_col]).strip() if desc_col else ""
            if not summary and not desc:
                continue

            defect_id = str(row[id_col]).strip() if id_col and str(row[id_col]).strip() else f"{path.stem}-def-{idx+1}"
            severity = str(row[sev_col]).strip() if sev_col else "Medium"
            status = str(row[status_col]).strip() if status_col else "Open"
            owner_raw = str(row[owner_col]).strip() if owner_col else ""
            owner = owner_raw if owner_raw and owner_raw.lower() not in ("nan", "none", "null") else None
            created_date = str(row[date_col]).strip() if date_col else ""
            module = str(row[module_col]).strip() if module_col and str(row[module_col]).strip() else "Testing"
            phase = str(row[phase_col]).strip() if phase_col and str(row[phase_col]).strip() else "Realize"

            content_parts = [
                f"Defect {defect_id}: {summary or desc}",
                f"Severity: {severity}",
                f"Status: {status}",
                f"Module: {module}"
            ]
            if desc and desc != summary:
                content_parts.append(f"Details: {desc}")
            content_str = " | ".join(content_parts)

            try:
                sig = ProjectSignal(
                    source_type=SourceType.DEFECT,
                    source_id=defect_id,
                    author=owner or "Unknown",
                    timestamp=created_date or None,
                    subject=f"Defect: {summary or defect_id}",
                    content=content_str,
                    workstream=module,
                    phase=phase,
                    raw_meta={
                        "source_file": path.name,
                        "original_format": path.suffix.lower().strip("."),
                        "defect_id": defect_id,
                        "summary": summary,
                        "description": desc,
                        "severity": severity,
                        "status": status,
                        "owner": owner,
                        "module": module
                    }
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error processing defect row %d in %s: %s", idx+1, path.name, exc)
                continue

        self.logger.info("Ingested %d defect signals from %s", len(signals), path.name)
        return signals
