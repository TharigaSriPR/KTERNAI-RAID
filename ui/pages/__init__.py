"""
UI Pages Package.
Streamlit multipage discovery package with backward-compatible aliases for tests and dynamic loaders.
"""

import sys
import importlib

# Backward-compatibility alias mappings for relocated page files
_ALIAS_MAP = {
    "ui.pages.4_Duplicate_Review": "ui.pages.5_Duplicate_Review",
    "ui.pages.7_Meeting_Minutes_Import": "ui.pages.2_Meeting_Minutes_Import",
    "ui.pages.2_Approved_Register": "ui.pages.3_Approved_Register",
    "ui.pages.3_Predictive_Insights": "ui.pages.4_Predictive_Insights",
    "ui.pages.5_Audit_History": "ui.pages.6_Audit_History",
    "ui.pages.6_Excel_Export": "ui.pages.7_Excel_Export",
}

for legacy_name, target_name in _ALIAS_MAP.items():
    if legacy_name not in sys.modules:
        try:
            sys.modules[legacy_name] = importlib.import_module(target_name)
        except Exception:
            pass
