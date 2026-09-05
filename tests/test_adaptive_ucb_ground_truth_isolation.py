"""Ground-truth isolation tests for AdaptiveUcbScheduler (Phase 4).

Mirrors the established Phase 2/3 spy pattern (see
tests/test_phase2_ground_truth_isolation.py and
tests/test_evaluator_ground_truth_isolation.py), applied to the new
scheduler specifically. Also proves the `reward` argument passed by the
evaluator has zero effect on this scheduler's behavior -- an additional,
Phase-4-specific isolation property beyond "never touches ground truth
directly": since AdaptiveUcbScheduler ignores `reward` and reads
`observation.detected` instead, it is also immune to whatever reward_fn
the evaluator happens to be configured with.
"""

import pytest

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.evaluator import SimpleEvaluator
from smart_scan_ew.interfaces import RFEnvironment
from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import AdaptiveUcbScheduler
from smart_scan_ew.state import SimpleBeliefState


class _GroundTruthSpy(RFEnvironment):
    """Wraps a real RFEnvironment and counts calls to get_ground_truth()
    and sense(). Same pattern as Phase 1/2/3's isolation tests."""

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
    """Wraps a real Scheduler and validates every argument it receives."""

    def __init__(self, wrapped: Scheduler):
        self._wrapped = wrapped
        self.update_calls: list[tuple[Observation, float]] = []

    def reset(self) -> None:
        self._wrapped.reset()

    def select_band(self, state: State) -> Band:
        assert isinstance(state, State)
        assert not hasattr(state, "get_ground_truth")
        return self._wrapped.select_band(state)

    def update(self, observation: Observation, reward: float) -> None:
        assert isinstance(observation, Observation)
        assert isinstance(reward, float)
        assert "ground_truth" not in observation.info
        assert "emitters" not in observation.info
        assert "emitter_id" not in observation.info
        self.update_calls.append((observation, reward))
        self._wrapped.update(observation, reward)


NUM_BANDS = 5
NUM_STEPS = 20


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


def test_adaptive_ucb_never_leaks_ground_truth():
    spy_env = _make_spy_environment()
    spy_env.reset(seed=0)

    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=1.0, seed=0)
    receiver.reset()

    state = SimpleBeliefState(num_bands=NUM_BANDS)
    state.reset()

    real_scheduler = AdaptiveUcbScheduler(num_bands=NUM_BANDS, gamma=0.9, exploration_constant=1.0)
    real_scheduler.reset()
    spy_scheduler = _ArgSpyScheduler(real_scheduler)

    evaluator = SimpleEvaluator(dt=1.0)
    run_record = evaluator.run_experiment(
        spy_env, receiver, spy_scheduler, state, num_steps=NUM_STEPS
    )

    # Evaluator IS expected to call get_ground_truth() -- once per step,
    # for its own scoring. That call never reaches the scheduler.
    assert spy_env.ground_truth_call_count == NUM_STEPS
    assert spy_env.sense_call_count == NUM_STEPS
    assert len(spy_scheduler.update_calls) == NUM_STEPS
    assert any(
        any(e.active for e in step.ground_truth.emitters) for step in run_record.steps
    )


def test_adaptive_ucb_source_imports_no_ground_truth_types():
    import smart_scan_ew.scheduler.adaptive_ucb_scheduler as module
    import inspect

    source = inspect.getsource(module)
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_block = "\n".join(import_lines)
    for forbidden in ("RFEnvironment", "GroundTruthSnapshot", "EmitterSpec"):
        assert forbidden not in import_block, (
            f"{forbidden} must not be imported by adaptive_ucb_scheduler.py"
        )
    # get_ground_truth is a method name, not an importable symbol -- check
    # it's never called/referenced anywhere in the module at all (this
    # check DOES scan the whole source, since a call site can't hide in a
    # docstring the way a type name explaining "we don't import X" can).
    assert "get_ground_truth(" not in source


def test_reward_argument_value_has_no_effect_end_to_end():
    # Run the SAME observation sequence through the evaluator twice, once
    # with SimpleEvaluator's default reward_fn and once with a reward_fn
    # that returns a completely different (even nonsensical) value --
    # AdaptiveUcbScheduler's resulting internal statistics and band
    # choices must be identical either way, proving end-to-end that the
    # reward argument truly has zero effect on this scheduler.
    def run(reward_fn):
        environment = SimpleRFEnvironment(
            emitter_specs=[EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=10.0)],
            num_bands=3,
            band_start_frequency_hz=2.4e9,
            band_width_hz=20e6,
        )
        environment.reset(seed=0)
        receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
        receiver.reset()
        state = SimpleBeliefState(num_bands=3)
        state.reset()
        scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, exploration_constant=1.0)
        scheduler.reset()
        evaluator = SimpleEvaluator(dt=1.0, reward_fn=reward_fn)
        run_record = evaluator.run_experiment(environment, receiver, scheduler, state, num_steps=10)
        return [s.band for s in run_record.steps], scheduler._counts, scheduler._successes

    bands_a, counts_a, successes_a = run(lambda obs: 1.0 if obs.detected else 0.0)
    bands_b, counts_b, successes_b = run(lambda obs: -7.0 if obs.detected else 123.0)

    assert bands_a == bands_b
    assert counts_a == pytest.approx(counts_b)
    assert successes_a == pytest.approx(successes_b)
