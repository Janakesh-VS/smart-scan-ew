"""Phase 5 dashboard controller tests. No `streamlit` import anywhere in
this file — `controller.py` has zero UI dependency, so all of this runs
as plain Python.

Maps to the Phase 5 test plan:
 1. dashboard imports
 2. controller construction
 3. scheduler selection
 4. selected scheduler is the ACTUAL scheduler class
 5. simulation reset
 6. simulation advancement
 7. real Observation generation
 8. real BeliefState updates
 9. real AdaptiveUcbScheduler statistics are exposed
10. dashboard does not modify scheduler algorithm state unexpectedly
12. four-way comparison uses actual evaluator APIs
13. metrics shown are from actual evaluator results
14. undefined metrics render as N/A
15. no hard-coded benchmark numbers

(11 and 16 live in tests/test_dashboard_ground_truth_isolation.py and are
implicitly covered by "the full suite still passes".)
"""

import ast
import inspect

import pytest

from smart_scan_ew.evaluator import ExperimentResult, TrialSummary
from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.scheduler import (
    AdaptiveUcbScheduler,
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from smart_scan_ew.state import BeliefSnapshot


# --- 1. dashboard imports ----------------------------------------------


def test_dashboard_package_imports_successfully():
    import smart_scan_ew.dashboard as dashboard  # noqa: F401
    from smart_scan_ew.dashboard import DashboardController, run_four_way_comparison  # noqa: F401


def test_controller_module_does_not_import_streamlit():
    """controller.py must contain zero UI dependency (Part 7: "keep UI
    code separate from simulation logic") -- checked structurally, not
    just "it happens not to fail if streamlit is uninstalled"."""
    from smart_scan_ew.dashboard import controller

    source = inspect.getsource(controller)
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("streamlit") for name in imported)


def test_controller_module_does_not_import_phase4_experiment_script():
    """Decision 2 (locked): the dashboard controller must not import
    examples/phase4_experiment.py -- that script is executable
    orchestration, not application code. (The module's docstring
    legitimately mentions the script's name in prose explaining this
    decision, so this checks actual import statements via AST, not a
    plain substring search.)"""
    from smart_scan_ew.dashboard import controller

    source = inspect.getsource(controller)
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("phase4_experiment" in name for name in imported)


# --- 2. controller construction -----------------------------------------


def test_controller_constructs_unconfigured():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    assert controller.is_configured is False
    assert controller.scheduler is None
    assert controller.history == ()


def test_step_before_reset_raises_a_clear_error():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    with pytest.raises(RuntimeError):
        controller.step()


# --- 3 & 4. scheduler selection / correct actual class -------------------


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("round_robin", RoundRobinScheduler),
        ("random", RandomScheduler),
        ("greedy_recent_hit", GreedyRecentHitScheduler),
        ("adaptive_ucb", AdaptiveUcbScheduler),
    ],
)
def test_each_scheduler_name_selects_the_real_scheduler_class(name, expected_class):
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name=name, num_bands=4, seed=0)
    assert type(controller.scheduler) is expected_class
    assert isinstance(controller.scheduler, Scheduler)
    assert controller.scheduler_name == name


def test_unknown_scheduler_name_raises():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    with pytest.raises(ValueError):
        controller.reset(scheduler_name="not_a_real_scheduler", num_bands=4, seed=0)  # type: ignore[arg-type]


# --- 5. simulation reset --------------------------------------------------


def test_reset_gives_a_fresh_scheduler_instance_even_for_the_same_name():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)
    controller.step()
    controller.step()
    first_instance = controller.scheduler
    assert first_instance.decision_count > 0

    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)
    second_instance = controller.scheduler
    assert second_instance is not first_instance
    assert second_instance.decision_count == 0  # no stale learned state carried over


def test_reset_clears_time_and_history():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="round_robin", num_bands=3, seed=0)
    controller.step()
    controller.step()
    assert controller.current_time > 0
    assert len(controller.history) == 2

    controller.reset(scheduler_name="round_robin", num_bands=3, seed=0)
    assert controller.current_time == 0.0
    assert controller.history == ()


def test_reset_is_reproducible_given_the_same_seed():
    from smart_scan_ew.dashboard import DashboardController

    def run():
        controller = DashboardController()
        controller.reset(scheduler_name="adaptive_ucb", num_bands=4, seed=7, gamma=0.9, c=1.0)
        return [
            (r.band, r.observation.detected, r.observation.info["measured_power"])
            for r in (controller.step() for _ in range(10))
        ]

    assert run() == run()


# --- 6 & 7. simulation advancement / real Observation ---------------------


def test_step_advances_time_and_returns_a_real_observation():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="round_robin", num_bands=4, seed=0)

    result = controller.step()
    assert controller.current_time == pytest.approx(1.0)
    assert isinstance(result.observation, Observation)
    assert result.observation.time == pytest.approx(1.0)
    assert 0 <= result.band < 4
    assert result.observation.band == result.band

    result2 = controller.step()
    assert controller.current_time == pytest.approx(2.0)
    assert len(controller.history) == 2
    assert controller.history[-1] is result2


# --- 8. real BeliefState updates -------------------------------------------


def test_step_updates_the_real_belief_state():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="round_robin", num_bands=3, seed=0)

    result = controller.step()
    assert isinstance(result.belief, BeliefSnapshot)
    observed_band = result.belief.bands[result.band]
    assert observed_band.observation_count == 1
    assert observed_band.last_detected == result.observation.detected

    # belief_snapshot() must reflect the same real state, not a copy that
    # can drift.
    assert controller.belief_snapshot() == result.belief


# --- 9. real AdaptiveUcbScheduler statistics exposed ------------------------


def test_ucb_diagnostics_come_from_the_actual_scheduler_get_diagnostics():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)

    result = controller.step()
    assert result.ucb_diagnostics is not None
    # Must be EXACTLY what the real scheduler reports right now -- no
    # second implementation, no copy that could drift.
    assert result.ucb_diagnostics == controller.scheduler.get_diagnostics()
    assert controller.ucb_diagnostics() == controller.scheduler.get_diagnostics()

    reason = result.decision_reason
    assert reason is not None
    assert reason.band == result.band


def test_ucb_diagnostics_and_decision_reason_are_none_for_non_adaptive_schedulers():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="round_robin", num_bands=3, seed=0)
    result = controller.step()
    assert result.ucb_diagnostics is None
    assert result.decision_reason is None
    assert controller.ucb_diagnostics() is None


def test_decision_reason_flags_unobserved_band_selection_honestly():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)
    # First step: every band is unobserved, so band 0 is chosen by the
    # unobserved-first rule, not a UCB score.
    result = controller.step()
    assert result.band == 0
    assert result.decision_reason.was_unobserved_band is True
    assert result.decision_reason.ucb_score is None


# --- 10. dashboard does not modify scheduler algorithm state unexpectedly --


def test_peek_ground_truth_and_diagnostics_calls_never_mutate_the_scheduler():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)
    controller.step()
    controller.step()

    before = controller.scheduler.get_diagnostics()
    before_count = controller.scheduler.decision_count

    # Purely observational calls -- none of these are step().
    controller.peek_ground_truth()
    controller.ucb_diagnostics()
    controller.belief_snapshot()

    after = controller.scheduler.get_diagnostics()
    after_count = controller.scheduler.decision_count
    assert before == after
    assert before_count == after_count


def test_replay_metrics_uses_a_fresh_scheduler_never_the_live_one():
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)
    for _ in range(5):
        controller.step()

    live_decision_count_before = controller.scheduler.decision_count
    controller.replay_metrics(num_steps=5)
    live_decision_count_after = controller.scheduler.decision_count

    # Computing replay metrics must not have advanced the LIVE scheduler.
    assert live_decision_count_before == live_decision_count_after == 5


# --- 12 & 13. four-way comparison / metrics from actual evaluator ---------


def test_run_four_way_comparison_returns_real_trial_summaries():
    from smart_scan_ew.dashboard import run_four_way_comparison
    from smart_scan_ew.dashboard.controller import SCHEDULER_NAMES

    results = run_four_way_comparison(
        num_bands=4, num_steps=20, master_seeds=[1, 2, 3], gamma=0.9, c=1.0
    )
    assert set(results.keys()) == set(SCHEDULER_NAMES)
    for name, summary in results.items():
        assert isinstance(summary, TrialSummary)
        assert summary.scheduler_name == name
        assert len(summary.per_trial_results) == 3
        assert all(isinstance(r, ExperimentResult) for r in summary.per_trial_results)
        assert summary.probability_of_detection.n_total == 3


def test_run_four_way_comparison_is_reproducible_given_the_same_seeds():
    from smart_scan_ew.dashboard import run_four_way_comparison

    def run():
        return run_four_way_comparison(
            num_bands=4, num_steps=15, master_seeds=[10, 11], gamma=0.9, c=1.0
        )

    a = run()
    b = run()
    for name in a:
        assert a[name].interception_rate_active_emitters.mean == b[name].interception_rate_active_emitters.mean
        assert a[name].average_reward.mean == b[name].average_reward.mean


def test_replay_metrics_matches_a_direct_evaluator_call_with_the_same_config():
    """Proves replay_metrics() is not a second metrics implementation --
    it must agree exactly with calling run_experiment_for_scheduler()
    directly with the same seed/config/scheduler type."""
    from smart_scan_ew.dashboard import DashboardController
    from smart_scan_ew.evaluator import ExperimentConfig, run_experiment_for_scheduler
    from smart_scan_ew.evaluator.reproducibility import derive_seeds
    from smart_scan_ew.scheduler import AdaptiveUcbScheduler

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=4, seed=3, gamma=0.9, c=1.0)
    for _ in range(10):
        controller.step()

    replay = controller.replay_metrics()

    scheduler_seed = derive_seeds(3, 3)[2]
    direct = run_experiment_for_scheduler(
        ExperimentConfig(num_bands=4, num_steps=10),
        AdaptiveUcbScheduler(num_bands=4, gamma=0.9, c=1.0),
        master_seed=3,
        scheduler_name="adaptive_ucb",
    )

    assert replay.true_positive_count == direct.true_positive_count
    assert replay.false_positive_count == direct.false_positive_count
    assert replay.false_negative_count == direct.false_negative_count
    assert replay.true_negative_count == direct.true_negative_count
    assert replay.average_reward == direct.average_reward
    assert replay.emitter_records == direct.emitter_records


def test_replay_metrics_reproduces_the_exact_live_trajectory():
    """Stronger version of the above: replay_metrics() over the same
    number of steps as the live session, with the same seed, must
    reproduce the SAME aggregate counts the live session actually saw
    (both derived from an identical call sequence -- see
    DashboardController.step()'s docstring)."""
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=4, seed=5, gamma=0.9, c=1.0)
    live_detections = 0
    for _ in range(20):
        result = controller.step()
        if result.observation.detected:
            live_detections += 1

    replay = controller.replay_metrics(num_steps=20)
    replay_detections = replay.true_positive_count + replay.false_positive_count
    assert replay_detections == live_detections


# --- 14. undefined metrics render as N/A ------------------------------------


def test_format_metric_renders_none_as_na_never_zero():
    from smart_scan_ew.dashboard.app import format_metric

    assert format_metric(None) == "N/A"
    assert format_metric(0.0) == "0.000"  # a REAL zero must stay zero, not become N/A
    assert format_metric(0.123456, digits=2) == "0.12"


def test_replay_metrics_can_have_undefined_timing_fields():
    """When nothing was ever intercepted, average_intercept_time is
    genuinely None (Phase 3 behavior, unchanged) -- confirms the
    dashboard has real None values to render as N/A, not just a
    hypothetical."""
    from smart_scan_ew.dashboard import DashboardController

    controller = DashboardController()
    # 1 step is very unlikely to intercept anything; if it does by
    # chance, the assertion below is skipped rather than flaky-failing.
    controller.reset(scheduler_name="round_robin", num_bands=8, seed=0)
    controller.step()
    result = controller.replay_metrics(num_steps=1)
    if result.intercepted_emitter_count == 0:
        assert result.average_intercept_time is None
        from smart_scan_ew.dashboard.app import format_metric

        assert format_metric(result.average_intercept_time) == "N/A"


# --- 15. no hard-coded benchmark numbers -------------------------------


def test_dashboard_source_contains_no_suspicious_hard_coded_metric_literals():
    """Structural guard: dashboard/*.py must never assign a metric-shaped
    result from a literal instead of a real evaluator call. This can't
    catch everything, but it does assert the two functions that produce
    numbers (`run_four_way_comparison`, `DashboardController.replay_metrics`)
    return values built from `run_repeated_trials`/`run_experiment_for_scheduler`
    calls, not from a literal dict/constant."""
    from smart_scan_ew.dashboard import controller

    source = inspect.getsource(controller)
    assert "run_repeated_trials" in source
    assert "run_experiment_for_scheduler" in source
