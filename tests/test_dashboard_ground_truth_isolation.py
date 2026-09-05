"""Ground-truth isolation tests for the Phase 5 dashboard controller.

Mirrors the Phase 1 (tests/test_ground_truth_isolation.py), Phase 2
(tests/test_phase2_ground_truth_isolation.py), and Phase 4
(tests/test_phase4_ground_truth_isolation.py) convention: a structural
check on the source, plus a runtime spy proving the decision path
(`DashboardController.step()`) never touches ground truth, even though
the dashboard is explicitly allowed to DISPLAY it via
`peek_ground_truth()`.
"""

import inspect

from smart_scan_ew.dashboard.controller import DashboardController
from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.scheduler import AdaptiveUcbScheduler


def test_step_method_source_never_calls_get_ground_truth():
    """Structural check: the `step()` method's own source text must not
    contain a `get_ground_truth` call. (`peek_ground_truth()` is a
    separate method and is allowed to call it -- that's the whole point
    of having it split out.)"""
    source = inspect.getsource(DashboardController.step)
    assert "get_ground_truth" not in source
    assert "peek_ground_truth" not in source


def test_replay_metrics_does_not_leak_ground_truth_into_the_live_session():
    """`replay_metrics()` calls the real evaluator (which DOES use ground
    truth internally, for scoring only, exactly like Phase 3) but must
    never feed anything back into the live `self._scheduler`/`self._state`
    -- it builds a completely separate, fresh scheduler and a separate
    environment/receiver/state inside `run_experiment_for_scheduler()`.
    """
    source = inspect.getsource(DashboardController.replay_metrics)
    # The live scheduler/state attributes must never be passed as
    # arguments to run_experiment_for_scheduler in this method.
    assert "self._scheduler," not in source.replace(" ", "")
    assert "self._state," not in source.replace(" ", "")
    assert "run_experiment_for_scheduler" in source
    assert "fresh_scheduler" in source  # uses a fresh instance, not the live one


class _GroundTruthSpy(RFEnvironment):
    """Wraps a real RFEnvironment and records calls to get_ground_truth()
    and sense(), without changing behavior. (Duplicated from the Phase
    1/2/4 test modules rather than imported, so each isolation test file
    stays self-contained -- the established convention in this repo.)
    """

    def __init__(self, wrapped: RFEnvironment):
        self._wrapped = wrapped
        self.sense_call_count = 0
        self.ground_truth_call_count = 0

    def reset(self, seed=None) -> None:
        self._wrapped.reset(seed=seed)

    def step(self, dt: float = 1.0) -> None:
        self._wrapped.step(dt=dt)

    def sense(self, band, t):
        self.sense_call_count += 1
        return self._wrapped.sense(band, t)

    def get_ground_truth(self):
        self.ground_truth_call_count += 1
        return self._wrapped.get_ground_truth()


def _install_spy(controller: DashboardController) -> _GroundTruthSpy:
    """Swap the controller's real environment for a spy-wrapped one,
    after reset() has already built everything else. This only touches
    the private `_environment` attribute from a test, which is
    acceptable here specifically because the point of this test IS to
    inspect environment access from the outside — production code
    (`app.py`) never does this.
    """
    spy = _GroundTruthSpy(controller._environment)
    controller._environment = spy
    return spy


NUM_BANDS = 5
NUM_STEPS = 25


def test_dashboard_step_loop_never_calls_get_ground_truth_for_any_scheduler():
    for scheduler_name in ("round_robin", "random", "greedy_recent_hit", "adaptive_ucb"):
        controller = DashboardController()
        controller.reset(
            scheduler_name=scheduler_name,
            num_bands=NUM_BANDS,
            seed=0,
            emitter_specs=[
                EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=5.0),
                EmitterSpec(
                    emitter_id="hopper-1", kind="hopping", band_id=1, power=6.0,
                    hop_interval=3.0, hop_bands=tuple(range(NUM_BANDS)),
                ),
            ],
            gamma=0.9,
            c=1.0,
        )
        spy = _install_spy(controller)

        for _ in range(NUM_STEPS):
            band = controller.step().band
            assert 0 <= band < NUM_BANDS

        assert spy.sense_call_count == NUM_STEPS
        assert spy.ground_truth_call_count == 0, (
            f"step() called get_ground_truth() for scheduler={scheduler_name!r} -- "
            "this violates ground-truth isolation."
        )


def test_peek_ground_truth_is_the_only_call_site_and_does_not_affect_step():
    """`peek_ground_truth()` MAY call get_ground_truth() (that's its
    entire purpose), but calling it interleaved with step() must not
    change step()'s own zero-ground-truth-calls behavior, and its return
    value is never threaded back into the controller."""
    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=NUM_BANDS, seed=0, gamma=0.9, c=1.0)
    spy = _install_spy(controller)

    for _ in range(10):
        controller.step()
        gt = controller.peek_ground_truth()  # display-only; must not feed back in
        assert gt.time == controller.current_time

    # 10 calls to step() -> 0 ground-truth calls from step() itself, plus
    # exactly 10 from the 10 explicit peek_ground_truth() calls above.
    assert spy.ground_truth_call_count == 10
    assert spy.sense_call_count == 10


def test_adaptive_ucb_scheduler_in_the_dashboard_still_has_no_environment_reference():
    """Belt-and-braces: even inside the dashboard's own wiring, the
    AdaptiveUcbScheduler instance in use is never handed anything
    environment-shaped -- it only ever receives `state` and
    `Observation`, exactly as in Phase 4."""
    controller = DashboardController()
    controller.reset(scheduler_name="adaptive_ucb", num_bands=3, seed=0, gamma=0.9, c=1.0)
    assert isinstance(controller.scheduler, AdaptiveUcbScheduler)
    for attr_name in vars(controller.scheduler):
        value = getattr(controller.scheduler, attr_name)
        assert not isinstance(value, SimpleRFEnvironment)
        assert not isinstance(value, RFEnvironment)
