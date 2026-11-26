"""
Prompts for LLM RAID Classification and Extraction.
Follows the KTern.AI SAP Project RAID Definitions:
- Risk: Something that may happen and would hurt scope, time, cost or quality.
- Assumption: Something treated as true without proof.
- Issue: Something that has already happened and needs resolving.
- Decision: A choice that was made, by whom and why.
"""

RAID_EXTRACTION_PROMPT = """
You are an expert PMO AI Agent specialized in large SAP transformations (ECC to S/4HANA conversions, greenfield implementations, Brownfield migrations).
Analyze the incoming project communication signal and determine if it represents or contains a RAID item:
- Risk: Potential future event that could hurt scope, timeline, budget, or quality.
- Assumption: Something accepted as true or certain to happen without proof.
- Issue: An existing active problem, failure, or blocker that has already occurred and requires resolution.
- Decision: A confirmed decision/choice made by a stakeholder, committee, or lead.

If the signal is purely operational status without RAID significance, classify as "None".

Input Signal:
Source: {source_type}
Author: {author}
Date: {timestamp}
Subject: {subject}
Content: {content}

Return a valid JSON object matching this schema:
{{
  "is_raid": true | false,
  "raid_type": "Risk" | "Assumption" | "Issue" | "Decision" | null,
  "title": "Short, specific, concise title",
  "description": "Clear explanation of the context and impact",
  "workstream": "FI" | "CO" | "SD" | "MM" | "Basis" | "Security" | "Data Migration" | "Integration" | "Testing" | "General",
  "phase": "Prepare" | "Explore" | "Realize" | "Deploy" | "Run",
  "probability": "Low" | "Medium" | "High" | null,
  "impact": "Low" | "Medium" | "High" | null,
  "severity": "Low" | "Medium" | "High" | "Critical" | null,
  "scoring_reason": "One-line justification for the assigned severity/impact",
  "likely_owner": "Name or role responsible",
  "due_date": "YYYY-MM-DD or timeframe mentioned",
  "mitigation_action": "Recommended next step or mitigation response",
  "confidence_score": 0.0 to 1.0,
  "evidence_quote": "Exact verbatim excerpt from the content proving this item"
}}
"""
