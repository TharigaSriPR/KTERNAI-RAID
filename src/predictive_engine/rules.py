"""
Deterministic Rule-Based Foresight Evaluation.
Implements:
1. Task Schedule Slippage (Overdue & Approaching Deadlines).
2. Defect Clustering (Repeated recurring defects indicating systemic risk).
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import re
from dateutil import parser as date_parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.normalisation.schemas import ProjectSignal, SourceType
from src.predictive_engine.schemas import (
    PredictionItem,
    PredictionType,
    PredictionSeverity,
    PredictionStatus,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

COMPLETED_STATUSES = {
    "completed", "complete", "done", "closed", "finished",
    "100%", "resolved", "signed off", "signed-off"
}

INCOMPLETE_STATUSES = {
    "in progress", "in-progress", "open", "not started", "delayed",
    "blocked", "pending", "active", "on hold", "behind schedule",
    "wip", "testing", "development", "qa"
}

UNKNOWN_STATUSES = {
    "", "unknown", "n/a", "none", "null", "undefined", "?"
}


def parse_date_safe(val: Any) -> Optional[date]:
    """Safely parses a date from string, datetime, or date object."""
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        val_clean = val.strip()
        if not val_clean or val_clean.lower() in ("nan", "none", "null", "n/a"):
            return None
        try:
            dt = date_parser.parse(val_clean)
            return dt.date()
        except Exception:
            return None
    return None


def evaluate_task_schedule(
    signal: ProjectSignal,
    reference_date: Optional[Union[date, datetime, str]] = None,
    upcoming_threshold_days: int = 7
) -> Optional[PredictionItem]:
    """
    Evaluates a single project plan signal for schedule slippage.
    
    Identifies:
    - Scenario A: TASK_OVERDUE (planned end passed, task is incomplete)
    - Scenario B: UPCOMING_DELAY (planned end within threshold, task is incomplete)
    - Scenario C: Completed tasks generate NO schedule risk
    
    Adheres strictly to:
    - Never assume incomplete when status is unknown or missing.
    - Handles missing or invalid dates gracefully.
    - Supports configurable reference date for deterministic testing.
    """
    meta = signal.raw_meta or {}
    task_name = meta.get("task_name") or signal.subject.replace("Task: ", "").strip()
    status_raw = meta.get("status")
    
    # Check status
    if status_raw is None:
        # Fallback inspection of content string
        content_lower = signal.content.lower()
        if "status:" in content_lower:
            parts = signal.content.split("Status:")[1].split("|")[0].strip()
            status_raw = parts
        else:
            status_raw = ""

    status_str = str(status_raw).strip().lower()

    # Rule: If status is unknown, do NOT assume incomplete
    if not status_str or status_str in UNKNOWN_STATUSES:
        logger.debug("Task '%s' has unknown status; skipping schedule evaluation.", task_name)
        return None

    # Rule: Scenario C - Completed tasks produce no schedule risk
    if status_str in COMPLETED_STATUSES:
        logger.debug("Task '%s' is completed (%s); no schedule risk generated.", task_name, status_str)
        return None

    # Verify task is genuinely incomplete
    is_incomplete = (
        status_str in INCOMPLETE_STATUSES or
        any(inc in status_str for inc in ("progress", "open", "delay", "blocked", "pending"))
    )
    if not is_incomplete:
        # Status not explicitly known as incomplete
        logger.debug("Task '%s' status '%s' is not classified as incomplete; skipping.", task_name, status_raw)
        return None

    # Extract planned finish date
    finish_date_raw = meta.get("finish_date") or meta.get("planned_end_date")
    if not finish_date_raw and "Baseline Finish:" in signal.content:
        finish_date_raw = signal.content.split("Baseline Finish:")[1].split("|")[0].strip()

    planned_end = parse_date_safe(finish_date_raw)
    if not planned_end:
        logger.debug("Task '%s' has no valid planned end date; skipping schedule evaluation.", task_name)
        return None

    # Determine reference date
    ref_date: date
    if reference_date is None:
        ref_date = date.today()
    elif isinstance(reference_date, str):
        parsed_ref = parse_date_safe(reference_date)
        ref_date = parsed_ref if parsed_ref else date.today()
    elif isinstance(reference_date, datetime):
        ref_date = reference_date.date()
    else:
        ref_date = reference_date

    variance_val = 0
    try:
        var_raw = meta.get("variance_days", "0")
        variance_val = int(float(str(var_raw).replace("days", "").strip()))
    except Exception:
        variance_val = 0

    evidence_str = (
        f"Task: {task_name} | Planned Finish: {planned_end.isoformat()} | "
        f"Status: {status_raw} | Variance: {variance_val} days | "
        f"Source: {signal.source_id or signal.id}"
    )

    # Scenario A: Overdue task
    if planned_end < ref_date:
        days_overdue = (ref_date - planned_end).days
        severity = PredictionSeverity.HIGH.value if days_overdue >= 7 or variance_val >= 5 else PredictionSeverity.MEDIUM.value
        return PredictionItem(
            prediction_type=PredictionType.TASK_OVERDUE,
            title=f"Predictive Risk: Task '{task_name}' is overdue ({days_overdue} days)",
            description=(
                f"Task '{task_name}' is overdue: planned end date was {planned_end.isoformat()}, which has passed by {days_overdue} days. "
                f"Task status is '{status_raw}'. May affect dependent activities in {signal.workstream or 'project'}."
            ),
            suggested_raid_category="Risk",
            confidence=0.92,
            severity=severity,
            evidence=evidence_str,
            source_signal_ids=[signal.id],
            workstream=signal.workstream or "General",
            phase=signal.phase or "Realize",
            metadata={
                "task_name": task_name,
                "days_overdue": days_overdue,
                "planned_end": planned_end.isoformat(),
                "reference_date": ref_date.isoformat(),
                "variance_days": variance_val
            }
        )

    # Scenario B: Upcoming deadline
    deadline_window = ref_date + timedelta(days=upcoming_threshold_days)
    if ref_date <= planned_end <= deadline_window:
        days_remaining = (planned_end - ref_date).days
        return PredictionItem(
            prediction_type=PredictionType.UPCOMING_DELAY,
            title=f"Predictive Risk: Approaching deadline for task '{task_name}' ({days_remaining} days remaining)",
            description=(
                f"Task '{task_name}' is approaching its planned end date ({planned_end.isoformat()}) "
                f"within {days_remaining} days and is currently '{status_raw}'."
            ),
            suggested_raid_category="Risk",
            confidence=0.80,
            severity=PredictionSeverity.MEDIUM.value,
            evidence=evidence_str,
            source_signal_ids=[signal.id],
            workstream=signal.workstream or "General",
            phase=signal.phase or "Realize",
            metadata={
                "task_name": task_name,
                "days_remaining": days_remaining,
                "planned_end": planned_end.isoformat(),
                "reference_date": ref_date.isoformat(),
                "threshold_days": upcoming_threshold_days
            }
        )

    return None


def evaluate_defect_clusters(
    defect_signals: List[ProjectSignal],
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.40
) -> List[PredictionItem]:
    """
    Groups defect signals by module and text similarity to detect recurring systemic issues.
    
    Requirements:
    1. Groups defects using meaningful common attributes (module) and text similarity.
    2. Does not cluster unrelated defects simply because of matching severity.
    3. Respects configurable minimum cluster size and similarity threshold.
    4. Preserves individual defect IDs and source references in evidence.
    5. Avoids generating redundant identical cluster predictions.
    """
    if len(defect_signals) < min_cluster_size:
        return []

    # Filter to only DEFECT source types
    valid_defects = [s for s in defect_signals if s.source_type == SourceType.DEFECT]
    if len(valid_defects) < min_cluster_size:
        return []

    # Group first by module/workstream
    by_module: Dict[str, List[ProjectSignal]] = {}
    for sig in valid_defects:
        mod = (sig.workstream or sig.raw_meta.get("module") or "General").strip()
        by_module.setdefault(mod, []).append(sig)

    predictions: List[PredictionItem] = []

    for mod, signals in by_module.items():
        if len(signals) < min_cluster_size:
            continue

        # Extract text representations
        texts = []
        for s in signals:
            meta = s.raw_meta or {}
            summary = meta.get("summary") or s.subject or ""
            desc = meta.get("description") or s.content or ""
            texts.append(f"{summary} {desc}".strip())

        # Build TF-IDF similarity matrix
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception as exc:
            logger.warning("Error computing TF-IDF on defects in %s: %s", mod, exc)
            continue

        # Cluster using connected components / similarity grouping
        visited = set()
        for i in range(len(signals)):
            if i in visited:
                continue
            cluster_indices = [i]
            for j in range(i + 1, len(signals)):
                if j not in visited and sim_matrix[i, j] >= similarity_threshold:
                    cluster_indices.append(j)

            if len(cluster_indices) >= min_cluster_size:
                visited.update(cluster_indices)
                cluster_signals = [signals[idx] for idx in cluster_indices]

                # Identify key recurring terms
                defect_ids = [s.raw_meta.get("defect_id") or s.source_id or s.id for s in cluster_signals]
                summaries = [s.raw_meta.get("summary") or s.subject for s in cluster_signals]

                # Extract common theme from summaries
                theme = summaries[0].replace("Defect: ", "").strip()
                if len(theme) > 40:
                    theme = theme[:40] + "..."

                # Determine if problem is actively blocking
                has_critical = any(
                    str(s.raw_meta.get("severity", "")).lower() in ("critical", "high") or
                    "blocked" in s.content.lower()
                    for s in cluster_signals
                )
                suggested_cat = "Issue" if has_critical else "Risk"
                severity = PredictionSeverity.HIGH.value if has_critical else PredictionSeverity.MEDIUM.value

                evidence_lines = [
                    f"- Defect {s.raw_meta.get('defect_id') or s.source_id}: {s.raw_meta.get('summary') or s.subject}"
                    for s in cluster_signals
                ]
                evidence_text = f"Recurring defects in module '{mod}':\n" + "\n".join(evidence_lines)

                pred = PredictionItem(
                    prediction_type=PredictionType.DEFECT_CLUSTER,
                    title=f"Predictive Pattern: Defect cluster in {mod} module ({len(cluster_signals)} related defects)",
                    description=(
                        f"Multiple recurring defects ({', '.join(defect_ids[:4])}) identified in {mod} "
                        f"indicating a potential systemic quality instability around '{theme}'."
                    ),
                    suggested_raid_category=suggested_cat,
                    confidence=min(0.95, 0.75 + (0.05 * len(cluster_signals))),
                    severity=severity,
                    evidence=evidence_text,
                    source_signal_ids=[s.id for s in cluster_signals],
                    workstream=mod,
                    phase=cluster_signals[0].phase or "Realize",
                    metadata={
                        "module": mod,
                        "cluster_size": len(cluster_signals),
                        "defect_ids": defect_ids,
                        "theme": theme
                    }
                )
                predictions.append(pred)

    return predictions



def evaluate_task_dependencies(
    signals: List[ProjectSignal]
) -> List[PredictionItem]:
    """
    Evaluates project plan signals for dependency blockages.
    Identifies tasks that cannot progress because a prerequisite task or external dependency is incomplete.
    
    Expected behavior:
    - Identifies a potential dependency blockage when Task B requires Task A, but Task A is incomplete.
    - Preserves both task references and IDs.
    - Explains the blocking relationship clearly.
    - Does NOT generate a prediction if the dependency is already completed.
    """
    plan_signals = [s for s in signals if s.source_type == SourceType.PROJECT_PLAN]
    if not plan_signals:
        return []

    # Index tasks by ID and by name (normalized lowercase)
    tasks_by_id: Dict[str, ProjectSignal] = {}
    tasks_by_name: Dict[str, ProjectSignal] = {}

    for s in plan_signals:
        meta = s.raw_meta or {}
        tid = str(meta.get("task_id") or s.source_id or s.id).strip().lower()
        tname = str(meta.get("task_name") or s.subject.replace("Task: ", "")).strip().lower()
        if tid:
            tasks_by_id[tid] = s
        if tname:
            tasks_by_name[tname] = s

    predictions: List[PredictionItem] = []

    for sig_b in plan_signals:
        meta_b = sig_b.raw_meta or {}
        task_b_id = str(meta_b.get("task_id") or sig_b.source_id or sig_b.id).strip()
        task_b_name = str(meta_b.get("task_name") or sig_b.subject.replace("Task: ", "")).strip()
        status_b_raw = str(meta_b.get("status") or "Open").strip()
        status_b_lower = status_b_raw.lower()

        # If Task B is already completed, no blockage
        if status_b_lower in COMPLETED_STATUSES:
            continue

        deps_raw = meta_b.get("dependencies")
        if not deps_raw and "Dependencies:" in sig_b.content:
            deps_raw = sig_b.content.split("Dependencies:")[1].split("|")[0].strip()

        if not deps_raw or str(deps_raw).strip().lower() in ("nan", "none", "", "null", "-"):
            continue

        # Split dependencies if multiple
        dep_items = [d.strip() for d in re.split(r"[,;]+", str(deps_raw)) if d.strip()]

        for dep_ref in dep_items:
            dep_clean = dep_ref.lower()
            # Lookup prerequisite task A
            sig_a = tasks_by_id.get(dep_clean) or tasks_by_name.get(dep_clean)
            
            # If not matched directly, try partial name match
            if not sig_a:
                for k, v in tasks_by_name.items():
                    if dep_clean in k or k in dep_clean:
                        sig_a = v
                        break

            if not sig_a:
                continue

            meta_a = sig_a.raw_meta or {}
            task_a_id = str(meta_a.get("task_id") or sig_a.source_id or sig_a.id).strip()
            task_a_name = str(meta_a.get("task_name") or sig_a.subject.replace("Task: ", "")).strip()
            status_a_raw = str(meta_a.get("status") or "Open").strip()
            status_a_lower = status_a_raw.lower()

            # Rule: Do not generate a prediction if the dependency is already completed
            if status_a_lower in COMPLETED_STATUSES:
                logger.debug("Prerequisite task '%s' for '%s' is completed; no blockage.", task_a_name, task_b_name)
                continue

            # Prerequisite task A is incomplete (e.g. In Progress, Open, Not Started, Delayed)
            suggested_cat = "Risk"
            if "blocked" in status_b_lower or "delay" in status_a_lower:
                severity = PredictionSeverity.HIGH.value
            else:
                severity = PredictionSeverity.MEDIUM.value

            evidence_text = (
                f"Task B: '{task_b_name}' (ID: {task_b_id}, Status: {status_b_raw}) depends on "
                f"Task A: '{task_a_name}' (ID: {task_a_id}, Status: {status_a_raw}). "
                f"Prerequisite is not completed."
            )

            pred = PredictionItem(
                prediction_type=PredictionType.DEPENDENCY_BLOCKAGE,
                title=f"Predictive Risk: Task '{task_b_name}' blocked by incomplete dependency '{task_a_name}'",
                description=(
                    f"Task '{task_b_name}' ({task_b_id}, status: '{status_b_raw}') requires prerequisite task "
                    f"'{task_a_name}' ({task_a_id}) to be finished, but prerequisite status is '{status_a_raw}'. "
                    f"Execution cannot progress as scheduled until prerequisite finishes."
                ),
                suggested_raid_category=suggested_cat,
                confidence=0.90,
                rule_score=0.90,
                severity=severity,
                evidence=evidence_text,
                source_signal_ids=[sig_b.id, sig_a.id],
                workstream=sig_b.workstream or sig_a.workstream or "General",
                phase=sig_b.phase or sig_a.phase or "Realize",
                metadata={
                    "task_b_id": task_b_id,
                    "task_b_name": task_b_name,
                    "task_b_status": status_b_raw,
                    "prerequisite_task_id": task_a_id,
                    "prerequisite_task_name": task_a_name,
                    "prerequisite_status": status_a_raw,
                    "declared_dependency": dep_ref
                }
            )
            predictions.append(pred)

    return predictions
