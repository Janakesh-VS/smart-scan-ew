"""Phase 1 tests for SimpleReceiver.

Uses a tiny stub environment (not SimpleRFEnvironment) so the receiver's
noise/threshold logic is tested in isolation from emitter/band behavior.
"""

import pytest

from smart_scan_ew.interfaces import Observation, RFEnvironment
from smart_scan_ew.receiver import SimpleReceiver


class _FixedPowerEnvironment(RFEnvironment):
    """Stub environment: sense() always returns a fixed value, regardless
    of band/time. Used to isolate receiver noise/threshold behavior."""

    def __init__(self, fixed_power: float):
        self.fixed_power = fixed_power
        self.sense_calls = 0
        self.ground_truth_calls = 0

    def reset(self, seed=None) -> None:
        pass

    def step(self, dt: float = 1.0) -> None:
        pass

    def sense(self, band, t):
        self.sense_calls += 1
        return self.fixed_power

    def get_ground_truth(self):
        self.ground_truth_calls += 1
        return {}


def test_observe_requires_tune_first():
    receiver = SimpleReceiver(detection_threshold=1.0, noise_std=0.0)
    env = _FixedPowerEnvironment(fixed_power=5.0)
    with pytest.raises(RuntimeError):
        receiver.observe(env, t=0.0)


def test_observe_returns_observation_with_expected_band_and_time():
    receiver = SimpleReceiver(detection_threshold=1.0, noise_std=0.0)
    env = _FixedPowerEnvironment(fixed_power=5.0)
    receiver.tune(band=3)
    obs = receiver.observe(env, t=2.5)
    assert isinstance(obs, Observation)
    assert obs.band == 3
    assert obs.time == 2.5


def test_zero_noise_is_deterministic_threshold_comparison():
    receiver = SimpleReceiver(detection_threshold=4.0, noise_std=0.0)
    receiver.tune(band=0)

    above = _FixedPowerEnvironment(fixed_power=10.0)
    obs_above = receiver.observe(above, t=0.0)
    assert obs_above.detected is True

    below = _FixedPowerEnvironment(fixed_power=1.0)
    obs_below = receiver.observe(below, t=0.0)
    assert obs_below.detected is False


def test_noise_is_reproducible_given_same_seed():
    def run(seed):
        receiver = SimpleReceiver(detection_threshold=1.0, noise_std=2.0, seed=seed)
        receiver.tune(band=0)
        env = _FixedPowerEnvironment(fixed_power=0.0)
        return [receiver.observe(env, t=float(i)).info["measured_power"] for i in range(5)]

    assert run(seed=123) == run(seed=123)


def test_reset_reproduces_the_same_noise_sequence():
    receiver = SimpleReceiver(detection_threshold=1.0, noise_std=2.0, seed=99)
    env = _FixedPowerEnvironment(fixed_power=0.0)

    receiver.tune(band=0)
    first_run = [receiver.observe(env, t=float(i)).info["measured_power"] for i in range(5)]

    receiver.reset()
    receiver.tune(band=0)
    second_run = [receiver.observe(env, t=float(i)).info["measured_power"] for i in range(5)]

    assert first_run == second_run


def test_reset_clears_tuned_band():
    receiver = SimpleReceiver(detection_threshold=1.0, noise_std=0.0)
    receiver.tune(band=2)
    receiver.reset()
    env = _FixedPowerEnvironment(fixed_power=5.0)
    with pytest.raises(RuntimeError):
        receiver.observe(env, t=0.0)


def test_noise_can_cause_a_miss_when_signal_present():
    # A real transmission (power=5.0) with high threshold + noise that
    # always pulls it below threshold should be reported as not detected.
    receiver = SimpleReceiver(detection_threshold=100.0, noise_std=0.1, seed=1)
    receiver.tune(band=0)
    env = _FixedPowerEnvironment(fixed_power=5.0)
    obs = receiver.observe(env, t=0.0)
    assert obs.detected is False  # miss: signal present but far below threshold


def test_noise_can_cause_a_false_alarm_when_nothing_present():
    # Nothing transmitting (power=0.0), but a low threshold combined with
    # positive noise should sometimes report detected=True.
    env = _FixedPowerEnvironment(fixed_power=0.0)
    detected_any = False
    for seed in range(50):
        receiver = SimpleReceiver(detection_threshold=0.5, noise_std=2.0, seed=seed)
        receiver.tune(band=0)
        obs = receiver.observe(env, t=0.0)
        if obs.detected:
            detected_any = True
            break
    assert detected_any, "expected at least one false alarm across 50 seeds"


def test_observe_only_calls_sense_never_ground_truth():
    receiver = SimpleReceiver(detection_threshold=1.0, noise_std=0.0)
    receiver.tune(band=0)
    env = _FixedPowerEnvironment(fixed_power=5.0)
    receiver.observe(env, t=0.0)
    assert env.sense_calls == 1
    assert env.ground_truth_calls == 0
