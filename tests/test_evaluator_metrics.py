"""Phase 3 metric-correctness tests.

The first test (`test_hand_computed_deterministic_scenario`) is the most
important one in this file: a fully deterministic scenario (noise_std=0)
with every TP/FP/FN/TN, Pd, Pfa, interception, and timing value computed
by hand in the test's comments, asserted exactly. Everything else is
edge cases the design review called out explicitly.
"""

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.evaluator import SimpleEvaluator
from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import RoundRobinScheduler
from smart_scan_ew.state import SimpleBeliefState


class _FixedBandScheduler(Scheduler):
    """Test-only fixture: always selects the same band, regardless of
    state. Used to force specific edge cases deterministically. Not one of
    the three approved baselines — never used outside this test file."""

    def __init__(self, band: Band):
        self._band = band

    def reset(self) -> None:
        pass

    def select_band(self, state: State) -> Band:
        return self._band

    def update(self, observation: Observation, reward: float) -> None:
        pass


def _run(environment, receiver, scheduler, state, num_steps, dt=1.0):
    evaluator = SimpleEvaluator(dt=dt)
    run_record = evaluator.run_experiment(environment, receiver, scheduler, state, num_steps)
    return evaluator.compute_metrics(run_record)


def test_hand_computed_deterministic_scenario():
    # One CW emitter, band 0, power 10.0. noise_std=0.0, threshold=5.0 ->
    # detection is a pure, noiseless threshold comparison: band 0 always
    # reads 10.0 > 5.0 (detected), every other band always reads 0.0 (not
    # detected). RoundRobinScheduler over 3 bands visits 0,1,2,0,1,2 for
    # 6 steps at t=1..6.
    #
    # Hand-computed per-step outcomes (band 0 = signal present):
    #   t=1 band=0 -> TP    t=2 band=1 -> TN    t=3 band=2 -> TN
    #   t=4 band=0 -> TP    t=5 band=1 -> TN    t=6 band=2 -> TN
    # TP=2, FP=0, FN=0, TN=4
    # Pd  = TP/(TP+FN) = 2/2 = 1.0
    # Pfa = FP/(FP+TN) = 0/4 = 0.0
    # Interception: cw-1 first active at t=1.0 (active from the first
    # step); first true positive at t=1.0 -> intercept_time=1.0,
    # intercept_time_error=0.0. Only 1 emitter, intercepted -> both
    # interception rates = 1/1 = 1.0.
    # average_intercept_time = 1.0, average_intercept_time_error = 0.0
    # average_reward = (2 rewarded steps of 6) = 2/6 = 1/3
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
    scheduler = RoundRobinScheduler(num_bands=3)
    scheduler.reset()

    metrics = _run(environment, receiver, scheduler, state, num_steps=6)

    assert metrics["true_positive_count"] == 2
    assert metrics["false_positive_count"] == 0
    assert metrics["false_negative_count"] == 0
    assert metrics["true_negative_count"] == 4
    assert metrics["total_observations"] == 6

    assert metrics["probability_of_detection"] == 1.0
    assert metrics["probability_of_false_alarm"] == 0.0

    assert metrics["total_emitters"] == 1
    assert metrics["active_emitters"] == 1
    assert metrics["intercepted_emitter_count"] == 1
    assert metrics["interception_rate_all_emitters"] == 1.0
    assert metrics["interception_rate_active_emitters"] == 1.0

    (record,) = metrics["emitter_records"]
    assert record.emitter_id == "cw-1"
    assert record.first_active_time == 1.0
    assert record.intercepted is True
    assert record.intercept_time == 1.0
    assert record.intercept_time_error == 0.0

    assert metrics["average_intercept_time"] == 1.0
    assert metrics["average_intercept_time_error"] == 0.0
    assert abs(metrics["average_reward"] - (2 / 6)) < 1e-12
    assert abs(metrics["average_cost"] - (1 - 2 / 6)) < 1e-12


def test_repeated_detection_of_already_intercepted_emitter_does_not_double_count():
    # From the hand-computed scenario above: cw-1 is detected TWICE
    # (t=1 and t=4) but only counted as ONE interception.
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
    scheduler = RoundRobinScheduler(num_bands=3)
    scheduler.reset()

    metrics = _run(environment, receiver, scheduler, state, num_steps=6)

    assert metrics["true_positive_count"] == 2  # two real detections...
    assert metrics["intercepted_emitter_count"] == 1  # ...one interception
    (record,) = metrics["emitter_records"]
    assert record.intercept_time == 1.0  # fixed at the FIRST true positive


def test_pd_is_none_when_receiver_never_lands_on_an_occupied_band():
    # Emitter lives on band 0; the (test-only) scheduler always tunes to
    # band 1 -> zero opportunities where signal was present in the tuned
    # band -> Pd's denominator (TP+FN) is 0.
    environment = SimpleRFEnvironment(
        emitter_specs=[EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=10.0)],
        num_bands=2,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=2)
    state.reset()
    scheduler = _FixedBandScheduler(band=1)

    metrics = _run(environment, receiver, scheduler, state, num_steps=5)

    assert metrics["false_negative_count"] == 0
    assert metrics["true_positive_count"] == 0
    assert metrics["probability_of_detection"] is None
    # Pfa IS defined here: band 1 is always empty, always correctly TN.
    assert metrics["probability_of_false_alarm"] == 0.0


def test_pfa_is_none_when_every_observation_lands_on_an_occupied_band():
    # Scheduler always tunes to the one band with an always-on emitter ->
    # every step is TP -> zero opportunities where nothing was present ->
    # Pfa's denominator (FP+TN) is 0.
    environment = SimpleRFEnvironment(
        emitter_specs=[EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=10.0)],
        num_bands=2,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=2)
    state.reset()
    scheduler = _FixedBandScheduler(band=0)

    metrics = _run(environment, receiver, scheduler, state, num_steps=5)

    assert metrics["true_positive_count"] == 5
    assert metrics["false_positive_count"] == 0
    assert metrics["true_negative_count"] == 0
    assert metrics["probability_of_false_alarm"] is None
    assert metrics["probability_of_detection"] == 1.0


def test_interception_rates_are_none_with_zero_emitters_in_scenario():
    environment = SimpleRFEnvironment(
        emitter_specs=[],
        num_bands=3,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=3)
    state.reset()
    scheduler = RoundRobinScheduler(num_bands=3)
    scheduler.reset()

    metrics = _run(environment, receiver, scheduler, state, num_steps=5)

    assert metrics["total_emitters"] == 0
    assert metrics["interception_rate_all_emitters"] is None
    assert metrics["interception_rate_active_emitters"] is None
    assert metrics["average_intercept_time"] is None
    assert metrics["average_intercept_time_error"] is None
    # Pfa IS defined (every step is TN, nothing ever transmits):
    assert metrics["probability_of_false_alarm"] == 0.0


def test_average_intercept_time_is_none_when_nothing_is_ever_intercepted():
    # Emitter present and active, but detection_threshold is unreachable
    # (noise_std=0.0, so detection is impossible) -> never intercepted.
    environment = SimpleRFEnvironment(
        emitter_specs=[EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=1.0)],
        num_bands=2,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=1000.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=2)
    state.reset()
    scheduler = _FixedBandScheduler(band=0)

    metrics = _run(environment, receiver, scheduler, state, num_steps=5)

    assert metrics["intercepted_emitter_count"] == 0
    assert metrics["average_intercept_time"] is None
    assert metrics["average_intercept_time_error"] is None
    (record,) = metrics["emitter_records"]
    assert record.intercepted is False
    assert record.intercept_time is None
    assert record.intercept_time_error is None
    # Still present in the audit records, not dropped:
    assert record.emitter_id == "cw-1"
    assert record.first_active_time == 1.0


def test_emitter_never_active_during_run_is_excluded_from_active_denominator():
    # period=1000 means this pulsed emitter never turns on within 5 steps.
    environment = SimpleRFEnvironment(
        emitter_specs=[
            EmitterSpec(
                emitter_id="dormant", kind="pulsed", band_id=0, power=10.0,
                period=1000.0, pulse_width=1.0,
            ),
        ],
        num_bands=2,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=2)
    state.reset()
    scheduler = RoundRobinScheduler(num_bands=2)
    scheduler.reset()

    metrics = _run(environment, receiver, scheduler, state, num_steps=5)

    assert metrics["total_emitters"] == 1
    assert metrics["active_emitters"] == 0
    assert metrics["interception_rate_all_emitters"] == 0.0
    assert metrics["interception_rate_active_emitters"] is None  # 0/0
    (record,) = metrics["emitter_records"]
    assert record.first_active_time is None
    assert record.intercepted is False


def test_co_band_emitters_are_both_credited_by_one_true_positive():
    # Two CW emitters sharing band 0 -> a single true-positive observation
    # on band 0 credits BOTH (documented Phase 1 sensing-model limitation:
    # sense() sums power, can't distinguish co-band emitters).
    environment = SimpleRFEnvironment(
        emitter_specs=[
            EmitterSpec(emitter_id="cw-a", kind="cw", band_id=0, power=5.0),
            EmitterSpec(emitter_id="cw-b", kind="cw", band_id=0, power=5.0),
        ],
        num_bands=2,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=2)
    state.reset()
    scheduler = _FixedBandScheduler(band=0)

    metrics = _run(environment, receiver, scheduler, state, num_steps=3)

    assert metrics["intercepted_emitter_count"] == 2
    by_id = {r.emitter_id: r for r in metrics["emitter_records"]}
    assert by_id["cw-a"].intercepted is True
    assert by_id["cw-b"].intercepted is True
    assert by_id["cw-a"].intercept_time == by_id["cw-b"].intercept_time == 1.0


def test_zero_step_run_yields_no_undefined_crashes():
    environment = SimpleRFEnvironment(
        emitter_specs=[EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=10.0)],
        num_bands=2,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    environment.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=5.0, noise_std=0.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=2)
    state.reset()
    scheduler = RoundRobinScheduler(num_bands=2)
    scheduler.reset()

    metrics = _run(environment, receiver, scheduler, state, num_steps=0)

    assert metrics["total_observations"] == 0
    assert metrics["probability_of_detection"] is None
    assert metrics["probability_of_false_alarm"] is None
    assert metrics["average_reward"] is None
    assert metrics["average_cost"] is None
    assert metrics["total_emitters"] == 0  # no steps -> no ground truth ever recorded
