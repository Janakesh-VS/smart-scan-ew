"""Ground-truth isolation across the complete Phase 2 loop:
RFEnvironment -> Receiver -> State -> Scheduler.

Extends the Phase 1 spy pattern (tests/test_ground_truth_isolation.py) to
cover all three Phase 2 baseline schedulers wired together with the real
SimpleRFEnvironment/SimpleReceiver/SimpleBeliefState, over a short manual
loop (no Evaluator — that's Phase 3). Asserts get_ground_truth() is never
called anywhere in the loop.
"""

import pytest

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.interfaces import RFEnvironment
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import (
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from smart_scan_ew.state import SimpleBeliefState


class _GroundTruthSpy(RFEnvironment):
    """Wraps a real RFEnvironment and records calls to get_ground_truth()
    and sense(), without changing behavior. (Duplicated from Phase 1's
    test module rather than imported, so each test file stays
    self-contained and independently readable.)"""

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
    "scheduler_factory",
    [
        lambda: RoundRobinScheduler(num_bands=NUM_BANDS),
        lambda: RandomScheduler(num_bands=NUM_BANDS, seed=0),
        lambda: GreedyRecentHitScheduler(num_bands=NUM_BANDS),
    ],
    ids=["round_robin", "random", "greedy_recent_hit"],
)
def test_full_phase2_loop_never_touches_ground_truth(scheduler_factory):
    spy_env = _make_spy_environment()
    spy_env.reset(seed=0)

    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=1.0, seed=0)
    receiver.reset()

    state = SimpleBeliefState(num_bands=NUM_BANDS)
    state.reset()

    scheduler = scheduler_factory()
    scheduler.reset()

    for step in range(NUM_STEPS):
        spy_env.step(dt=1.0)
        t = float(step + 1)

        band = scheduler.select_band(state)
        assert 0 <= band < NUM_BANDS

        receiver.tune(band)
        observation = receiver.observe(spy_env, t=t)

        state.update(observation)
        scheduler.update(observation, reward=0.0)

    assert spy_env.sense_call_count == NUM_STEPS
    assert spy_env.ground_truth_call_count == 0, (
        "Ground truth was accessed somewhere in the Phase 2 loop — this "
        "violates the ground-truth isolation rule (CLAUDE.md rule 4/5)."
    )
