"""
Project Plan & Task List Ingestion Adapter.
Supports:
- Excel spreadsheets (.xlsx, .xls) via Pandas & OpenPyXL
- Delimited task lists (.csv)
Extracts task name, planned start/end dates, actual status, owner, variance, and dependencies.
Preserves structured task metadata in raw_meta without prematurely forcing every ordinary task into a RAID item.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import pandas as pd

from .base_ingestor import BaseIngestor
from src.normalisation.schemas import ProjectSignal, SourceType


class PlanIngestor(BaseIngestor):
    def __init__(self):
        super().__init__(SourceType.PROJECT_PLAN)

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
                self.logger.warning("Unsupported plan file extension '%s' for file: %s", suffix, path)
                return []
            return self._process_dataframe(df, path)
        except Exception as exc:
            self.logger.error("Failed to ingest project plan '%s': %s", path, exc, exc_info=True)
            return []

    def _process_dataframe(self, df: pd.DataFrame, path: Path) -> List[ProjectSignal]:
        signals = []
        if df.empty:
            self.logger.warning("Project plan file is empty: %s", path.name)
            return []

        # Graceful column resolution using BaseIngestor helper
        id_col = self._find_column(df.columns, ["taskid", "task id", "id", "wbs", "code"])
        name_col = self._find_column(df.columns, ["taskname", "task name", "task", "title", "summary", "name"])
        owner_col = self._find_column(df.columns, ["assignee", "owner", "resource", "assigned to"])
        status_col = self._find_column(df.columns, ["status", "state", "progress"])
        start_col = self._find_column(df.columns, ["baselinestart", "plannedstart", "start date", "start"])
        finish_col = self._find_column(df.columns, ["baselinefinish", "plannedfinish", "currentfinish", "due date", "finish date", "finish"])
        variance_col = self._find_column(df.columns, ["variancedays", "variance", "delay", "slip days", "slip"])
        dep_col = self._find_column(df.columns, ["dependencies", "predecessors", "depends on", "dependency"])
        notes_col = self._find_column(df.columns, ["notes", "description", "comments", "remarks"])
        workstream_col = self._find_column(df.columns, ["workstream", "module", "track", "area"])
        phase_col = self._find_column(df.columns, ["phase", "stage"])

        df_clean = df.fillna("")

        for idx, row in df_clean.iterrows():
            task_name = str(row[name_col]).strip() if name_col else f"Task-{idx+1}"
            if not task_name or task_name.lower() == "nan":
                continue

            task_id = str(row[id_col]).strip() if id_col and str(row[id_col]).strip() else f"{path.stem}-task-{idx+1}"
            owner_raw = str(row[owner_col]).strip() if owner_col else ""
            owner = owner_raw if owner_raw and owner_raw.lower() not in ("nan", "none", "null") else None
            status = str(row[status_col]).strip() if status_col else "Not Started"
            start_date = str(row[start_col]).strip() if start_col else ""
            finish_date = str(row[finish_col]).strip() if finish_col else ""
            variance = str(row[variance_col]).strip() if variance_col else "0"
            dependencies = str(row[dep_col]).strip() if dep_col else ""
            notes = str(row[notes_col]).strip() if notes_col else ""
            workstream = str(row[workstream_col]).strip() if workstream_col and str(row[workstream_col]).strip() else "PMO"
            phase = str(row[phase_col]).strip() if phase_col and str(row[phase_col]).strip() else "Realize"

            # Build readable summary content for the signal
            content_parts = [
                f"Task: {task_name}",
                f"Status: {status}",
                f"Baseline Finish: {finish_date}",
                f"Variance: {variance} days"
            ]
            if dependencies:
                content_parts.append(f"Dependencies: {dependencies}")
            if notes:
                content_parts.append(f"Notes: {notes}")
            content_str = " | ".join(content_parts)

            try:
                sig = ProjectSignal(
                    source_type=SourceType.PROJECT_PLAN,
                    source_id=task_id,
                    author=owner or "Unknown",
                    timestamp=finish_date or None,
                    subject=f"Task: {task_name}",
                    content=content_str,
                    workstream=workstream,
                    phase=phase,
                    raw_meta={
                        "source_file": path.name,
                        "original_format": path.suffix.lower().strip("."),
                        "task_id": task_id,
                        "task_name": task_name,
                        "owner": owner,
                        "status": status,
                        "start_date": start_date,
                        "finish_date": finish_date,
                        "variance_days": variance,
                        "dependencies": dependencies,
                        "notes": notes
                    }
                )
                signals.append(sig)
            except Exception as exc:
                self.logger.warning("Error processing plan row %d in %s: %s", idx+1, path.name, exc)
                continue

        self.logger.info("Ingested %d task signals from project plan %s", len(signals), path.name)
        return signals
