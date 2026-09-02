"""Ground-truth isolation at the evaluator boundary (Phase 3's specific
new risk).

Phase 1/2's isolation tests proved Receiver/State/Scheduler never call
get_ground_truth() themselves. Phase 3 introduces the first LEGITIMATE
caller of get_ground_truth() (SimpleEvaluator, for scoring) — so the risk
shifts: does the evaluator's own orchestration loop leak that ground
truth back into the scheduler/state it drives? These tests prove it does
not, using a scheduler-argument spy (not just an environment call-count
spy) plus a direct content check on the reward value.
"""

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.evaluator import SimpleEvaluator
from smart_scan_ew.interfaces import RFEnvironment
from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import (
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from smart_scan_ew.state import SimpleBeliefState

import pytest


class _GroundTruthSpy(RFEnvironment):
    """Wraps a real RFEnvironment and counts calls to get_ground_truth()
    and sense(). Same pattern as Phase 1/2's isolation tests."""

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


class _ArgSpyScheduler(Scheduler):
    """Wraps a real Scheduler and records/validates every argument it
    receives, so a Phase 3 bug that threads ground truth (or anything
    derived from it) into the scheduler would be caught here — not just
    "get_ground_truth was called N times" (that's expected now, from the
    evaluator itself)."""

    def __init__(self, wrapped: Scheduler):
        self._wrapped = wrapped
        self.select_band_state_types: list[type] = []
        self.update_calls: list[tuple[Observation, float]] = []

    def reset(self) -> None:
        self._wrapped.reset()

    def select_band(self, state: State) -> Band:
        assert isinstance(state, State), f"expected a State, got {type(state)}"
        assert not hasattr(state, "get_ground_truth"), (
            "the object passed to select_band() exposes get_ground_truth "
            "-- this is not a plain belief State"
        )
        self.select_band_state_types.append(type(state))
        return self._wrapped.select_band(state)

    def update(self, observation: Observation, reward: float) -> None:
        assert isinstance(observation, Observation)
        assert isinstance(reward, float)
        # The one deliberately-open field (`info`) is the only place
        # ground truth could sneak through without changing Observation's
        # shape -- guard it explicitly.
        assert "ground_truth" not in observation.info
        assert "emitters" not in observation.info
        assert "emitter_id" not in observation.info
        self.update_calls.append((observation, reward))
        self._wrapped.update(observation, reward)


class _ArgSpyState(State):
    """Wraps a real State and validates every Observation it receives."""

    def __init__(self, wrapped: State):
        self._wrapped = wrapped
        self.update_calls: list[Observation] = []

    def reset(self) -> None:
        self._wrapped.reset()

    def update(self, observation: Observation) -> None:
        assert isinstance(observation, Observation)
        assert "ground_truth" not in observation.info
        assert "emitters" not in observation.info
        self.update_calls.append(observation)
        self._wrapped.update(observation)

    def get_features(self):
        return self._wrapped.get_features()


NUM_BANDS = 5
NUM_STEPS = 15


def _make_spy_environment() -> _GroundTruthSpy:
    real_env = SimpleRFEnvironment(
        emitter_specs=[
            EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=5.0),
            EmitterSpec(
                emitter_id="hopper-1", kind="hopping", band_id=1, power=6.0,
                hop_interval=3.0, hop_bands=(0, 1, 2, 3, 4),
            ),
        ],
        num_bands=NUM_BANDS,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    return _GroundTruthSpy(real_env)


@pytest.mark.parametrize(
    "make_scheduler",
    [
        lambda: RoundRobinScheduler(num_bands=NUM_BANDS),
        lambda: RandomScheduler(num_bands=NUM_BANDS, seed=0),
        lambda: GreedyRecentHitScheduler(num_bands=NUM_BANDS),
    ],
    ids=["round_robin", "random", "greedy_recent_hit"],
)
def test_evaluator_never_leaks_ground_truth_to_scheduler_or_state(make_scheduler):
    spy_env = _make_spy_environment()
    spy_env.reset(seed=0)

    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=1.0, seed=0)
    receiver.reset()

    real_state = SimpleBeliefState(num_bands=NUM_BANDS)
    real_state.reset()
    spy_state = _ArgSpyState(real_state)

    real_scheduler = make_scheduler()
    real_scheduler.reset()
    spy_scheduler = _ArgSpyScheduler(real_scheduler)

    evaluator = SimpleEvaluator(dt=1.0)
    run_record = evaluator.run_experiment(
        spy_env, receiver, spy_scheduler, spy_state, num_steps=NUM_STEPS
    )

    # The evaluator IS expected to call get_ground_truth() now -- once per
    # step, for its own scoring. That is the whole point of Phase 3.
    assert spy_env.ground_truth_call_count == NUM_STEPS
    assert spy_env.sense_call_count == NUM_STEPS

    # But none of that ever reached the scheduler or state:
    assert len(spy_scheduler.update_calls) == NUM_STEPS
    assert len(spy_state.update_calls) == NUM_STEPS
    assert all(t is _ArgSpyState for t in spy_scheduler.select_band_state_types)

    # And the recorded run itself really does carry ground truth (proving
    # the isolation is a real design property, not just an artifact of a
    # trivial/empty run):
    assert any(
        any(e.active for e in step.ground_truth.emitters) for step in run_record.steps
    )


def test_reward_passed_to_scheduler_is_not_ground_truth_derived():
    # The Phase 3 placeholder reward (1.0 if detected else 0.0) must be
    # computable from the Observation ALONE. Verify this directly: two
    # otherwise-identical observations that differ only in `detected`
    # produce different rewards, and the reward function's signature/usage
    # never receives a GroundTruthSnapshot.
    from smart_scan_ew.evaluator.simple_evaluator import default_reward

    detected_obs = Observation(time=1.0, band=0, detected=True)
    missed_obs = Observation(time=1.0, band=0, detected=False)

    assert default_reward(detected_obs) == 1.0
    assert default_reward(missed_obs) == 0.0

    import inspect

    params = inspect.signature(default_reward).parameters
    assert list(params.keys()) == ["observation"]
    assert "ground_truth" not in params
    assert "environment" not in params
