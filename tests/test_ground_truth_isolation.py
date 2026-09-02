"""Dedicated test for CLAUDE.md rule 4/5 and ARCHITECTURE.md's
"Ground-truth isolation" section: SimpleReceiver must call
`sense()` and must never call `get_ground_truth()`.

This wraps a real SimpleRFEnvironment in a spy that records which methods
are called, so the assertion is about actual runtime behavior, not just
code review.
"""

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.interfaces import RFEnvironment
from smart_scan_ew.receiver import SimpleReceiver


class _GroundTruthSpy(RFEnvironment):
    """Wraps a real RFEnvironment and records calls to get_ground_truth()
    and sense(), without changing behavior."""

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


def test_receiver_never_calls_get_ground_truth_across_a_multi_step_run():
    real_env = SimpleRFEnvironment(
        emitter_specs=[
            EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=5.0),
        ],
        num_bands=5,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    spy_env = _GroundTruthSpy(real_env)
    spy_env.reset(seed=0)

    receiver = SimpleReceiver(detection_threshold=1.0, noise_std=0.5, seed=0)

    for step in range(10):
        spy_env.step(dt=1.0)
        receiver.tune(band=step % 5)
        receiver.observe(spy_env, t=float(step))

    assert spy_env.sense_call_count == 10
    assert spy_env.ground_truth_call_count == 0, (
        "SimpleReceiver.observe() called get_ground_truth() — this "
        "violates the ground-truth isolation rule (CLAUDE.md rule 4/5)."
    )
