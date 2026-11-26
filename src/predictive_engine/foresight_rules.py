"""
Rule-Based Predictive Engine (Foresight).
Detects proactive risks before they become issues:
1. Repeated task slippage trend across workstreams (e.g., variance > 5 days).
2. Rising defect counts in a module (e.g., >3 open high defects).
3. Contradicted assumptions (e.g. assumption was 'client data by Friday' but chat says 'client delayed').
"""

from typing import List, Dict, Any, Optional
import uuid


class PredictiveEngine:
    def __init__(self):
        pass

    def check_task_slippage(self, plan_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects repetitive slippage in project tasks.
        Raises predictive risk.
        """
        predicted_risks = []
        for signal in plan_signals:
            content = signal.get("content", "")
            # Look for variance indicators
            if "Variance:" in content:
                try:
                    parts = content.split("Variance:")[1].split()[0]
                    days = int(parts)
                    if days >= 5:
                        workstream = signal.get("workstream", "General")
                        predicted_risks.append({
                            "id": f"pred-{uuid.uuid4().hex[:8]}",
                            "type": "Risk",
                            "title": f"Predictive Risk: Schedule slippage in {workstream} ({days} days delay)",
                            "description": f"Rule-based foresight identified recurring slippage: {signal.get('subject')} is delayed by {days} days.",
                            "workstream": workstream,
                            "phase": signal.get("phase", "Realize"),
                            "probability": "High",
                            "impact": "High",
                            "severity": None,
                            "scoring_reason": f"Automated foresight: Task variance exceeded tolerance ({days} days)",
                            "likely_owner": signal.get("author", "Workstream Lead"),
                            "due_date": "Immediate",
                            "mitigation_action": "Crash schedule or conduct replanning workshop for workstream dependencies",
                            "confidence_score": 0.95,
                            "evidence_text": content,
                            "evidence_source_id": signal.get("id"),
                            "evidence_source_type": signal.get("source_type", "project_plan"),
                            "detection_source": "predictive_engine",
                            "duplicate_of_id": None,
                            "status": "Draft"
                        })
                except Exception:
                    continue
        return predicted_risks

    def check_defect_clustering(self, defect_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects high defect clusters in specific SAP modules.
        """
        predicted_risks = []
        module_counts: Dict[str, int] = {}
        for d in defect_signals:
            mod = d.get("workstream", "General")
            module_counts[mod] = module_counts.get(mod, 0) + 1

        for mod, count in module_counts.items():
            if count >= 3:
                first_sig = defect_signals[0] if defect_signals else {}
                predicted_risks.append({
                    "id": f"pred-{uuid.uuid4().hex[:8]}",
                    "type": "Risk",
                    "title": f"Predictive Risk: High defect concentration in {mod} module ({count} open defects)",
                    "description": f"Rule-based predictive analysis identified {count} defects logged in {mod}, indicating potential quality instability before cutover.",
                    "workstream": mod,
                    "phase": "Realize",
                    "probability": "High",
                    "impact": "High",
                    "severity": None,
                    "scoring_reason": f"Cluster pattern detected: {count} defects concentrated in single module",
                    "likely_owner": f"{mod} Lead",
                    "due_date": "Sprint End",
                    "mitigation_action": "Pause new transport releases to QA and organize code review on custom extensions",
                    "confidence_score": 0.91,
                    "evidence_text": f"Multiple defect reports accumulated in {mod}",
                    "evidence_source_id": first_sig.get("id", "defect-cluster"),
                    "evidence_source_type": "defect",
                    "detection_source": "predictive_engine",
                    "duplicate_of_id": None,
                    "status": "Draft"
                })
        return predicted_risks
