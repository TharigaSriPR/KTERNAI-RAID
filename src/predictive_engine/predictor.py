"""
Predictive Engine (Foresight Orchestrator).
Coordinates:
1. Task schedule slippage evaluation (Overdue & Upcoming deadlines).
2. Defect clustering across modules and technical themes.
3. Contradicted assumption monitoring.
4. Dependency blockage detection across project plan tasks.
Prevents duplicate predictions for identical source sets.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import date, datetime

from src.normalisation.schemas import ProjectSignal, SourceType
from src.predictive_engine.schemas import (
    PredictionItem,
    PredictionType,
    PredictionSeverity,
    PredictionStatus,
)
from src.predictive_engine.rules import (
    evaluate_task_schedule,
    evaluate_defect_clusters,
    evaluate_task_dependencies,
)
from src.predictive_engine.assumption_monitor import AssumptionMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictiveEngine:
    """
    Foresight Orchestrator that applies transparent, rule-based logic to project signals.
    Identifies risks and systemic issues proactively without LLM hallucinations.
    """

    def __init__(self):
        self.assumption_monitor = AssumptionMonitor()
        self._seen_predictions: Set[Tuple[str, Tuple[str, ...]]] = set()

    def _is_duplicate_prediction(self, item: PredictionItem) -> bool:
        """Determines if a prediction for the same type and source signals was already produced."""
        sources = tuple(sorted(item.source_signal_ids))
        sig = (item.prediction_type, sources)
        if sig in self._seen_predictions:
            return True
        self._seen_predictions.add(sig)
        return False

    def predict_schedule_slippage(
        self,
        signals: List[ProjectSignal],
        reference_date: Optional[Union[date, datetime, str]] = None,
        upcoming_threshold_days: int = 7
    ) -> List[PredictionItem]:
        """Evaluates all project plan signals for overdue tasks or upcoming deadlines."""
        predictions: List[PredictionItem] = []
        for sig in signals:
            if sig.source_type != SourceType.PROJECT_PLAN:
                continue
            pred = evaluate_task_schedule(
                sig,
                reference_date=reference_date,
                upcoming_threshold_days=upcoming_threshold_days
            )
            if pred and not self._is_duplicate_prediction(pred):
                predictions.append(pred)

        return predictions

    def predict_defect_clusters(
        self,
        signals: List[ProjectSignal],
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.40
    ) -> List[PredictionItem]:
        """Evaluates defect signals to identify recurring quality clusters."""
        defect_signals = [s for s in signals if s.source_type == SourceType.DEFECT]
        clusters = evaluate_defect_clusters(
            defect_signals,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold
        )
        unique_clusters: List[PredictionItem] = []
        for c in clusters:
            if not self._is_duplicate_prediction(c):
                unique_clusters.append(c)

        return unique_clusters

    def predict_dependency_blockages(
        self,
        signals: List[ProjectSignal]
    ) -> List[PredictionItem]:
        """Evaluates tasks with prerequisites to identify blocked dependencies."""
        blockages = evaluate_task_dependencies(signals)
        unique_blockages: List[PredictionItem] = []
        for b in blockages:
            if not self._is_duplicate_prediction(b):
                unique_blockages.append(b)

        return unique_blockages

    def predict_contradicted_assumptions(
        self,
        assumptions: List[Dict[str, Any]],
        signals: List[ProjectSignal]
    ) -> List[PredictionItem]:
        """Monitors recorded assumptions against new signals for contradictory developments."""
        contradictions = self.assumption_monitor.check_contradictions(assumptions, signals)
        unique_preds: List[PredictionItem] = []
        for p in contradictions:
            if not self._is_duplicate_prediction(p):
                unique_preds.append(p)

        return unique_preds

    def run_foresight(
        self,
        signals: List[ProjectSignal],
        existing_assumptions: Optional[List[Dict[str, Any]]] = None,
        reference_date: Optional[Union[date, datetime, str]] = None,
        upcoming_threshold_days: int = 7,
        min_cluster_size: int = 3,
        defect_similarity_threshold: float = 0.40
    ) -> List[PredictionItem]:
        """
        Executes complete foresight evaluation across all rule patterns:
        1. Schedule slippage
        2. Defect clustering
        3. Dependency blockages
        4. Contradicted assumptions
        """
        results: List[PredictionItem] = []

        # 1. Schedule slippage
        schedule_preds = self.predict_schedule_slippage(
            signals,
            reference_date=reference_date,
            upcoming_threshold_days=upcoming_threshold_days
        )
        results.extend(schedule_preds)

        # 2. Defect clustering
        defect_preds = self.predict_defect_clusters(
            signals,
            min_cluster_size=min_cluster_size,
            similarity_threshold=defect_similarity_threshold
        )
        results.extend(defect_preds)

        # 3. Dependency blockages
        dependency_preds = self.predict_dependency_blockages(signals)
        results.extend(dependency_preds)

        # 4. Contradicted assumptions
        if existing_assumptions:
            assumption_preds = self.predict_contradicted_assumptions(
                existing_assumptions,
                signals
            )
            results.extend(assumption_preds)

        logger.info(
            "Foresight evaluation complete: %d total predictions generated (%d schedule, %d defect clusters, %d dependencies, %d assumptions)",
            len(results), len(schedule_preds), len(defect_preds), len(dependency_preds),
            len(results) - len(schedule_preds) - len(defect_preds) - len(dependency_preds)
        )
        return results

    # Backward compatibility helpers
    def check_task_slippage(self, plan_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []

    def check_defect_clustering(self, defect_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []
