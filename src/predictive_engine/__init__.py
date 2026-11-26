"""
Predictive Engine package.
Provides rule-based foresight detection for schedule slippage, defect clustering,
dependency blockages, and contradicted assumptions.
"""

from .schemas import (
    PredictionType,
    PredictionStatus,
    PredictionSeverity,
    PredictionItem,
)
from .rules import (
    evaluate_task_schedule,
    evaluate_defect_clusters,
    evaluate_task_dependencies,
)
from .assumption_monitor import AssumptionMonitor
from .predictor import PredictiveEngine

__all__ = [
    "PredictionType",
    "PredictionStatus",
    "PredictionSeverity",
    "PredictionItem",
    "evaluate_task_schedule",
    "evaluate_defect_clusters",
    "evaluate_task_dependencies",
    "AssumptionMonitor",
    "PredictiveEngine",
]
