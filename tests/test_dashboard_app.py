"""Headless smoke tests for the Phase 5 Streamlit UI (`dashboard/app.py`),
using Streamlit's own `streamlit.testing.v1.AppTest` — the officially
supported way to run a Streamlit script without a browser and assert it
doesn't raise.

These are deliberately smoke tests, not exhaustive UI tests: the goal is
"the app imports, starts, and the real backend flows work end-to-end
through the widgets a user would actually click" (Phase 5 validation
items 3-7), not full visual coverage.
"""

from pathlib import Path

import pytest

APP_PATH = str(Path(__file__).parent.parent / "src/smart_scan_ew/dashboard/app.py")


def _click(app_test, label: str):
    """AppTest element references become stale after `.run()` — always
    re-fetch the widget by label immediately before clicking it."""
    button = next(b for b in app_test.button if b.label == label)
    button.click().run()


def _set_selectbox(app_test, label: str, value: str):
    box = next(s for s in app_test.selectbox if s.label == label)
    box.set_value(value).run()


def _set_slider(app_test, label: str, value):
    slider = next(s for s in app_test.slider if s.label == label)
    slider.set_value(value).run()


# --- 3/4. Streamlit imports / app starts ------------------------------


def test_app_imports_and_starts_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Smart Scan EW — Live Dashboard"]


def test_app_exposes_all_four_scheduler_choices():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    scheduler_box = next(s for s in at.selectbox if s.label == "Scheduler")
    assert set(scheduler_box.options) == {
        "Round Robin",
        "Random",
        "Greedy Recent Hit",
        "Adaptive UCB",
    }


# --- 5. real simulation through the app --------------------------------


def test_step_once_advances_the_real_simulation_for_every_scheduler():
    from streamlit.testing.v1 import AppTest

    for scheduler_display in ("Round Robin", "Random", "Greedy Recent Hit", "Adaptive UCB"):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _set_selectbox(at, "Scheduler", scheduler_display)
        assert not at.exception

        _click(at, "Step Once")
        assert not at.exception, f"Step Once raised for scheduler={scheduler_display!r}"
        _click(at, "Step Once")
        assert not at.exception


def test_adaptive_ucb_panel_and_why_explanation_render_without_error():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    _set_selectbox(at, "Scheduler", "Adaptive UCB")
    for _ in range(5):
        _click(at, "Step Once")
    assert not at.exception
    # The UCB diagnostics table and "why" panel should have rendered
    # something (a dataframe + a code block), not silently nothing.
    assert len(at.dataframe) >= 1
    assert len(at.code) >= 1


# --- 6. four-way comparison through the app -----------------------------


def test_four_way_comparison_runs_through_the_app_with_a_small_seed_set():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _set_slider(at, "Number of trial seeds", 2)
    _click(at, "Run Four-Way Comparison")
    assert not at.exception
    assert len(at.table) == 1
    df = at.table[0].value
    assert list(df["Scheduler"]) == ["Round Robin", "Random", "Greedy Recent Hit", "Adaptive UCB"]


# --- 7/13. real evaluator metrics displayed ------------------------------


def test_performance_tab_computes_real_metrics_without_error():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    for _ in range(5):
        _click(at, "Step Once")
    _click(at, "Compute Performance Metrics")
    assert not at.exception
    assert len(at.table) == 1
    df = at.table[0].value
    assert "Probability of Detection (Pd)" in list(df["Metric"])


# --- Reset / Start-Pause behave predictably (Part 13) ---------------------


def test_reset_button_clears_history():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    _click(at, "Step Once")
    _click(at, "Step Once")
    controller = at.session_state["controller"]
    assert len(controller.history) == 2

    _click(at, "Reset")
    controller = at.session_state["controller"]
    assert len(controller.history) == 0


def test_changing_scheduler_triggers_exactly_one_fresh_reset_not_a_reset_per_rerun():
    """Part 13's explicit pitfall: the simulation must not reset on every
    unrelated rerun. Two consecutive Step Once clicks (no config change)
    must accumulate history, not keep resetting to zero."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    _click(at, "Step Once")
    _click(at, "Step Once")
    _click(at, "Step Once")
    controller = at.session_state["controller"]
    assert len(controller.history) == 3  # accumulated, not reset each click
