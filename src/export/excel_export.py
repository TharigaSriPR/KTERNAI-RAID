"""
Excel Export Module for RAID Identification Agent POC.
Uses openpyxl to generate professional, multi-tab and single-tab Excel workbooks
with frozen header rows, auto-filters, custom widths, and proper cell types.
"""

import io
import json
from typing import List, Dict, Any, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Formatting Constants
HEADER_BG = "1F4E78"        # Dark Navy Blue
HEADER_FONT_COLOR = "FFFFFF" # White
BORDER_COLOR = "D9D9D9"      # Light Gray
EMPTY_ROW_COLOR = "F2F2F2"   # Very Light Gray


def _get_header_font() -> Font:
    return Font(name="Calibri", size=11, bold=True, color=HEADER_FONT_COLOR)


def _get_data_font(italic: bool = False) -> Font:
    return Font(name="Calibri", size=11, bold=False, italic=italic)


def _get_header_fill() -> PatternFill:
    return PatternFill(start_color=HEADER_BG, end_color=HEADER_BG, fill_type="solid")


def _get_thin_border() -> Border:
    thin = Side(border_style="thin", color=BORDER_COLOR)
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _apply_sheet_styling(ws: openpyxl.worksheet.worksheet.Worksheet, is_empty: bool = False):
    """
    Applies professional styling:
    - Bold, filled header row.
    - Thin borders on all cells.
    - Freezes top row.
    - Enables auto-filter.
    - Dynamically computes column widths.
    """
    header_font = _get_header_font()
    header_fill = _get_header_fill()
    border = _get_thin_border()

    # Style header row
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    # Set row height
    ws.row_dimensions[1].height = 28

    # Style data rows
    data_font = _get_data_font()
    empty_font = _get_data_font(italic=True)

    for row_idx in range(2, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 20
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = border
            if is_empty:
                cell.font = empty_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.font = data_font
                # Align numbers and dates nicely
                if isinstance(cell.value, (int, float)):
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

    # Freeze header row
    ws.freeze_panes = "A2"

    # Auto-filter
    if ws.max_column > 0 and ws.max_row > 0:
        ws.auto_filter.ref = ws.dimensions

    # Auto-fit column widths
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for row in range(1, ws.max_row + 1):
            val = ws.cell(row=row, column=col_idx).value
            if val is not None:
                # Approximate length, truncate multiline checks
                first_line = str(val).split("\n")[0]
                max_len = max(max_len, len(first_line))
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 14), 60)


def _clean_str(val: Any) -> str:
    """
    Safely converts value to string, turning None into empty string,
    and neutralizing Excel formula injection risks (CWE-1236).
    """
    if val is None:
        return ""
    s = str(val)
    if s.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + s
    return s


def _safe_json_format(val: Any) -> str:
    """Formats JSON array or dict cleanly into readable string if applicable, neutralized against formula injection."""
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        res = json.dumps(val, indent=2)
    else:
        val_str = str(val).strip()
        if (val_str.startswith("[") and val_str.endswith("]")) or (val_str.startswith("{") and val_str.endswith("}")):
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, list):
                    res = ", ".join(str(x) for x in parsed)
                else:
                    res = json.dumps(parsed, indent=2)
            except Exception:
                res = val_str
        else:
            res = val_str

    if res.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + res
    return res


# ---------------------------------------------------------------------------
# Individual Worksheet Writers
# ---------------------------------------------------------------------------

def write_approved_register_sheet(wb: openpyxl.Workbook, items: List[Dict[str, Any]], title: str = "Approved Register"):
    """Populates a worksheet with approved RAID register records."""
    ws = wb.create_sheet(title=title)
    headers = [
        "Approved ID", "Draft ID", "RAID Type", "Title", "Description",
        "Workstream", "SAP Phase", "Severity / Impact", "Owner", "Due Date",
        "Mitigation Action", "Register Status", "Approved By", "Approved At",
        "Evidence Status", "Evidence & Source Quote"
    ]
    ws.append(headers)

    if not items:
        ws.append(["(No approved records found)"] + [""] * (len(headers) - 1))
        _apply_sheet_styling(ws, is_empty=True)
        return ws

    for item in items:
        raw_ev_status = item.get("evidence_status")
        ev_status_str = str(raw_ev_status).upper() if raw_ev_status else "UNKNOWN"
        row = [
            _clean_str(item.get("id")),
            _clean_str(item.get("draft_id")),
            _clean_str(item.get("type")),
            _clean_str(item.get("title")),
            _clean_str(item.get("description")),
            _clean_str(item.get("workstream")),
            _clean_str(item.get("phase")),
            _clean_str(item.get("severity_or_impact")),
            _clean_str(item.get("owner")),
            _clean_str(item.get("due_date")),
            _clean_str(item.get("mitigation_action")),
            _clean_str(item.get("status")),
            _clean_str(item.get("approved_by")),
            _clean_str(item.get("approved_at")),
            _clean_str(ev_status_str),
            _clean_str(item.get("evidence_links")),
        ]
        ws.append(row)

    _apply_sheet_styling(ws, is_empty=False)
    return ws


def write_review_queue_sheet(wb: openpyxl.Workbook, items: List[Dict[str, Any]], title: str = "Review Queue"):
    """Populates a worksheet with pending/reviewable RAID drafts."""
    ws = wb.create_sheet(title=title)
    headers = [
        "Draft ID", "RAID Type", "Title", "Description", "Workstream",
        "SAP Phase", "Probability", "Impact", "Severity", "Scoring Reason",
        "Likely Owner", "Target Due Date", "Mitigation Action", "Confidence Score",
        "Detection Source", "LLM Provider", "LLM Model", "Evidence Source Type", "Evidence Source ID",
        "Evidence Status", "Evidence Quote", "Duplicate Of ID", "Status", "Detected At"
    ]
    ws.append(headers)

    if not items:
        ws.append(["(No pending drafts in review queue)"] + [""] * (len(headers) - 1))
        _apply_sheet_styling(ws, is_empty=True)
        return ws

    for item in items:
        conf = item.get("confidence_score")
        conf_val = round(float(conf), 2) if conf is not None else ""
        raw_ev_status = item.get("evidence_status")
        ev_status_str = str(raw_ev_status).upper() if raw_ev_status else "UNKNOWN"
        row = [
            _clean_str(item.get("id")),
            _clean_str(item.get("type")),
            _clean_str(item.get("title")),
            _clean_str(item.get("description")),
            _clean_str(item.get("workstream")),
            _clean_str(item.get("phase")),
            _clean_str(item.get("probability")),
            _clean_str(item.get("impact")),
            _clean_str(item.get("severity")),
            _clean_str(item.get("scoring_reason")),
            _clean_str(item.get("likely_owner")),
            _clean_str(item.get("due_date")),
            _clean_str(item.get("mitigation_action")),
            conf_val,
            _clean_str(item.get("detection_source")),
            _clean_str(item.get("llm_provider") or "unknown"),
            _clean_str(item.get("llm_model") or "unknown"),
            _clean_str(item.get("evidence_source_type")),
            _clean_str(item.get("evidence_source_id")),
            _clean_str(ev_status_str),
            _clean_str(item.get("evidence_text")),
            _clean_str(item.get("duplicate_of_id")),
            _clean_str(item.get("status")),
            _clean_str(item.get("detected_at")),
        ]
        ws.append(row)

    _apply_sheet_styling(ws, is_empty=False)
    return ws


def write_predictions_sheet(wb: openpyxl.Workbook, items: List[Dict[str, Any]], title: str = "Predictive Insights"):
    """Populates a worksheet with predictive engine findings."""
    ws = wb.create_sheet(title=title)
    headers = [
        "Prediction ID", "Type", "Title", "Description",
        "Suggested Category", "Severity", "Confidence Score",
        "Source Signal IDs", "Related RAID IDs", "Evidence & Rationale", "Detected At"
    ]
    ws.append(headers)

    if not items:
        ws.append(["(No predictions recorded)"] + [""] * (len(headers) - 1))
        _apply_sheet_styling(ws, is_empty=True)
        return ws

    for item in items:
        conf = item.get("confidence")
        conf_val = round(float(conf), 2) if conf is not None else ""
        row = [
            _clean_str(item.get("id")),
            _clean_str(item.get("prediction_type")),
            _clean_str(item.get("title")),
            _clean_str(item.get("description")),
            _clean_str(item.get("suggested_raid_category")),
            _clean_str(item.get("severity")),
            conf_val,
            _safe_json_format(item.get("source_signal_ids")),
            _safe_json_format(item.get("related_raid_ids")),
            _clean_str(item.get("evidence")),
            _clean_str(item.get("detected_at")),
        ]
        ws.append(row)

    _apply_sheet_styling(ws, is_empty=False)
    return ws


def write_audit_history_sheet(wb: openpyxl.Workbook, items: List[Dict[str, Any]], title: str = "Audit History"):
    """Populates a worksheet with immutable audit trail records."""
    ws = wb.create_sheet(title=title)
    headers = [
        "Audit ID", "Target Item ID", "Table", "Action", "Actor Identity",
        "Timestamp (UTC)", "Action Notes", "Prior State / Proposed Data", "Final State / Updated Data"
    ]
    ws.append(headers)

    if not items:
        ws.append(["(No audit history records)"] + [""] * (len(headers) - 1))
        _apply_sheet_styling(ws, is_empty=True)
        return ws

    for item in items:
        row = [
            _clean_str(item.get("id")),
            _clean_str(item.get("item_id")),
            _clean_str(item.get("item_table")),
            _clean_str(item.get("action")),
            _clean_str(item.get("action_by")),
            _clean_str(item.get("timestamp")),
            _clean_str(item.get("action_notes")),
            _safe_json_format(item.get("proposed_data")),
            _safe_json_format(item.get("final_data")),
        ]
        ws.append(row)

    _apply_sheet_styling(ws, is_empty=False)
    return ws


# ---------------------------------------------------------------------------
# Public Export Functions
# ---------------------------------------------------------------------------

def export_approved_register(items: List[Dict[str, Any]]) -> bytes:
    """Generates a standalone Excel workbook containing approved RAID items."""
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)
    write_approved_register_sheet(wb, items, title="Approved RAID Register")
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def export_review_queue(items: List[Dict[str, Any]]) -> bytes:
    """Generates a standalone Excel workbook containing pending review drafts."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_review_queue_sheet(wb, items, title="Pending Review Queue")
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def export_predictions(items: List[Dict[str, Any]]) -> bytes:
    """Generates a standalone Excel workbook containing predictive engine items."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_predictions_sheet(wb, items, title="Predictive Insights")
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def export_audit_history(items: List[Dict[str, Any]]) -> bytes:
    """Generates a standalone Excel workbook containing audit history records."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_audit_history_sheet(wb, items, title="Audit Trail")
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def export_combined_report(
    approved_items: List[Dict[str, Any]],
    draft_items: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    audit_entries: List[Dict[str, Any]]
) -> bytes:
    """
    Generates a master comprehensive multi-tab Excel workbook
    containing all 4 primary datasets in dedicated worksheets.
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    write_approved_register_sheet(wb, approved_items, title="Approved Register")
    write_review_queue_sheet(wb, draft_items, title="Pending Review Queue")
    write_predictions_sheet(wb, predictions, title="Predictive Insights")
    write_audit_history_sheet(wb, audit_entries, title="Audit Trail")

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
