"""Phase 5 dashboard package.

`controller.py` holds all non-UI logic (importable and testable without
Streamlit installed running anything). `app.py` is the Streamlit UI that
calls it — see ARCHITECTURE.md's Phase 5 section.
"""

from smart_scan_ew.dashboard.controller import (
    SCENARIO_PRESETS,
    SCHEDULER_DISPLAY_NAMES,
    SCHEDULER_NAMES,
    DashboardController,
    DecisionReason,
    StepResult,
    run_four_way_comparison,
)

__all__ = [
    "DashboardController",
    "DecisionReason",
    "SCENARIO_PRESETS",
    "SCHEDULER_DISPLAY_NAMES",
    "SCHEDULER_NAMES",
    "StepResult",
    "run_four_way_comparison",
]
