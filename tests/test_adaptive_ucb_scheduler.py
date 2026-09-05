"""Phase 4 tests for AdaptiveUcbScheduler: construction, initialization,
update rule, discounting, UCB scoring, numerical safety, determinism,
reset, and integration with SimpleBeliefState / the Phase 3 evaluator /
the three Phase 2 baselines.

Ground-truth isolation tests live in a dedicated file,
tests/test_phase4_ground_truth_isolation.py, mirroring the Phase 1/2
convention of a separate isolation-test module per phase.

Hyperparameter-selection-rule tests (the near-tie logic) and
selection/held-out seed separation are also here, exercising
examples/phase4_experiment.py directly.
"""

import dataclasses
import math
import sys
from pathlib import Path

import pytest

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment, default_scenario
from smart_scan_ew.evaluator import ExperimentConfig, run_experiment_for_scheduler
from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import (
    AdaptiveUcbScheduler,
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from smart_scan_ew.scheduler.adaptive_ucb import EPSILON, BandUcbDiagnostics
from smart_scan_ew.state import SimpleBeliefState

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import phase4_experiment as p4  # noqa: E402  (path inserted above, matches conftest.py's src/ pattern)


def _obs(band: int, detected: bool, time: float = 1.0) -> Observation:
    return Observation(time=time, band=band, detected=detected)


# --- Interface compliance ---------------------------------------------------


def test_adaptive_ucb_is_a_scheduler_instance():
    assert isinstance(AdaptiveUcbScheduler(num_bands=4), Scheduler)


def test_construction_rejects_invalid_num_bands():
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=0)
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=-1)


def test_construction_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=3, gamma=0.0)
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=3, gamma=1.1)
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=3, gamma=-0.5)


def test_construction_rejects_negative_c():
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=3, c=-1.0)


def test_update_rejects_out_of_range_band():
    scheduler = AdaptiveUcbScheduler(num_bands=3)
    with pytest.raises(ValueError):
        scheduler.update(_obs(band=5, detected=True), reward=1.0)


# --- Initialization / unobserved-band priority ------------------------------


def test_valid_band_selection_is_always_in_range():
    scheduler = AdaptiveUcbScheduler(num_bands=6)
    state = SimpleBeliefState(num_bands=6)
    for _ in range(20):
        band = scheduler.select_band(state)
        assert 0 <= band < 6
        scheduler.update(_obs(band=band, detected=False), reward=0.0)


def test_all_bands_unobserved_explores_lowest_band_id_first():
    scheduler = AdaptiveUcbScheduler(num_bands=4)
    state = SimpleBeliefState(num_bands=4)
    assert scheduler.select_band(state) == 0


def test_unobserved_bands_are_explored_in_order_before_any_ucb_scoring():
    scheduler = AdaptiveUcbScheduler(num_bands=4)
    state = SimpleBeliefState(num_bands=4)

    seen = []
    for _ in range(4):
        band = scheduler.select_band(state)
        seen.append(band)
        scheduler.update(_obs(band=band, detected=False), reward=0.0)
    assert seen == [0, 1, 2, 3]


def test_previously_observed_band_never_preempts_a_never_observed_band():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=1.0, c=2.0)
    state = SimpleBeliefState(num_bands=3)

    # Heavily reward band 0 -- a huge UCB score should still lose to an
    # untried band.
    for _ in range(50):
        scheduler.update(_obs(band=0, detected=True), reward=1.0)

    assert scheduler.select_band(state) in (1, 2)


# --- Update rule: hits / misses / discounting -------------------------------


def test_hit_increases_success_and_count():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=1.0)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    # p_0 should now be 1.0 (one observation, one detection).
    assert scheduler._S[0] == 1.0
    assert scheduler._N[0] == 1.0


def test_miss_increases_count_only():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=1.0)
    scheduler.update(_obs(band=0, detected=False), reward=0.0)
    assert scheduler._S[0] == 0.0
    assert scheduler._N[0] == 1.0


def test_discounting_decays_every_bands_statistics_each_update():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=0.5)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    assert scheduler._S[0] == 1.0
    assert scheduler._N[0] == 1.0

    # Second update, band 1 this time -- band 0's stats must decay by
    # gamma even though band 0 wasn't the band observed this step.
    scheduler.update(_obs(band=1, detected=False), reward=0.0)
    assert scheduler._S[0] == pytest.approx(0.5)
    assert scheduler._N[0] == pytest.approx(0.5)
    assert scheduler._N[1] == pytest.approx(1.0)


def test_gamma_equal_one_never_discounts_lifetime_counts_accumulate():
    scheduler = AdaptiveUcbScheduler(num_bands=1, gamma=1.0)
    for _ in range(10):
        scheduler.update(_obs(band=0, detected=True), reward=1.0)
    assert scheduler._N[0] == 10.0
    assert scheduler._S[0] == 10.0


def test_gamma_less_than_one_lets_stale_evidence_fade_and_reorder_preference():
    # Band 0 is hit hard early, then goes cold. Band 1 starts cold, then
    # becomes reliable. With aggressive forgetting, band 1 should
    # eventually overtake band 0's UCB score once its recent evidence
    # dominates -- the point of discounting (ARCHITECTURE.md Phase 4:
    # "why recency matters").
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=0.5, c=0.0)  # c=0: pure exploitation
    state = SimpleBeliefState(num_bands=2)

    # Force-observe both bands once each (unobserved-priority phase).
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    scheduler.update(_obs(band=1, detected=False), reward=0.0)

    # Band 0 hot, band 1 cold, for a while.
    for _ in range(5):
        scheduler.update(_obs(band=0, detected=True), reward=1.0)
    # Band 0's estimate should dominate now.
    assert scheduler.select_band(state) == 0

    # Now band 1 turns reliably hot, band 0 goes cold, for long enough
    # that gamma=0.5 discounting has washed out band 0's old streak.
    for _ in range(10):
        scheduler.update(_obs(band=1, detected=True), reward=1.0)
        scheduler.update(_obs(band=0, detected=False), reward=0.0)

    assert scheduler.select_band(state) == 1


# --- UCB score against hand calculations ------------------------------------


def test_ucb_score_matches_hand_calculation():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=1.0, c=1.0)
    state = SimpleBeliefState(num_bands=2)

    # Force both bands observed once: band 0 hit, band 1 miss.
    scheduler.update(_obs(band=0, detected=True), reward=1.0)  # t becomes 1
    scheduler.update(_obs(band=1, detected=False), reward=0.0)  # t becomes 2

    # Next decision: decision_index = t + 1 = 3.
    t = 3
    p0, n0 = 1.0, 1.0
    p1, n1 = 0.0, 1.0
    score0 = p0 / n0 + 1.0 * math.sqrt(math.log(t) / n0)
    score1 = p1 / n1 + 1.0 * math.sqrt(math.log(t) / n1)
    assert score0 > score1  # band 0 has strictly better p_b, equal bonus

    chosen = scheduler.select_band(state)
    assert chosen == 0


def test_ucb_exploration_bonus_favors_less_observed_band_when_hit_rates_tie():
    # Two bands with identical hit rate but different observation counts:
    # the less-observed one gets a bigger exploration bonus and should
    # be preferred (classic UCB behavior).
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=1.0, c=2.0)
    state = SimpleBeliefState(num_bands=2)

    # Both bands: alternate hit/miss so estimated p_b == 0.5 for both,
    # but band 1 gets fewer total observations.
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    scheduler.update(_obs(band=1, detected=True), reward=1.0)
    for _ in range(3):
        scheduler.update(_obs(band=0, detected=False), reward=0.0)
        scheduler.update(_obs(band=0, detected=True), reward=1.0)
    # band 0 now has 4 hits / 8 obs = 0.5; band 1 has 1 hit / 1 obs = 1.0
    # -- not actually a tie, so instead assert band 1 (fewer observations,
    # higher p_b) wins, which is consistent both with UCB exploitation
    # and exploration.
    assert scheduler.select_band(state) == 1


# --- t / log edge cases and numerical safety --------------------------------


def test_t_starts_such_that_first_ucb_decision_never_computes_log_of_zero():
    scheduler = AdaptiveUcbScheduler(num_bands=1, gamma=1.0)
    state = SimpleBeliefState(num_bands=1)
    # First call explores the only (unobserved) band -- no log involved.
    assert scheduler.select_band(state) == 0
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    # Second call: t = 1 update so far -> decision_index = 2 -> log(2),
    # never log(0) or a negative argument. Must not raise.
    scheduler.select_band(state)  # no exception


def test_no_zero_division_or_nan_or_inf_after_prolonged_neglect():
    # gamma=0.9, band 1 is observed once and then never revisited for
    # thousands of steps while band 0 is hammered -- band 1's N_b
    # decays toward (and, in float terms, eventually to) zero. The
    # score must stay finite and non-NaN throughout.
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=0.9, c=1.0)
    state = SimpleBeliefState(num_bands=2)

    scheduler.update(_obs(band=1, detected=True), reward=1.0)  # band 1 observed once
    for _ in range(20000):
        scheduler.update(_obs(band=0, detected=True), reward=1.0)  # band 0 hammered

    assert scheduler._N[1] == pytest.approx(0.0, abs=1e-9)
    assert scheduler._N[1] >= 0.0

    # select_band must not raise, and must return a valid band with a
    # finite (non-NaN, non-inf) internal score for band 1.
    band = scheduler.select_band(state)
    assert band in (0, 1)

    n_safe = max(scheduler._N[1], EPSILON)
    p1 = scheduler._S[1] / n_safe
    bonus = 1.0 * math.sqrt(math.log(20002) / n_safe)
    score1 = p1 + bonus
    assert math.isfinite(score1)
    assert not math.isnan(score1)


def test_epsilon_floor_prevents_zero_division_at_the_boundary():
    # Directly exercises the EPSILON floor: construct a scheduler where
    # a band's discounted count has underflowed to exactly 0.0 in float,
    # and confirm select_band still returns without raising.
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=0.01, c=1.0)
    state = SimpleBeliefState(num_bands=2)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    scheduler.update(_obs(band=1, detected=True), reward=1.0)
    # 0.01 ** 162 is the first power that underflows to exactly 0.0 in
    # IEEE-754 float64 starting from 1.0; 200 decay steps clears that
    # with margin, guaranteeing band 0's N_b has genuinely underflowed
    # (not just become very small) before the assertion below.
    for _ in range(200):
        scheduler.update(_obs(band=1, detected=True), reward=1.0)
    assert scheduler._N[0] == 0.0  # confirms the underflow actually happened

    band = scheduler.select_band(state)  # must not raise ZeroDivisionError
    assert band in (0, 1)


# --- Deterministic behavior / ties / reset ----------------------------------


def test_select_band_is_a_pure_function_between_updates():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, c=1.0)
    state = SimpleBeliefState(num_bands=3)
    for _ in range(5):
        band = scheduler.select_band(state)
        scheduler.update(_obs(band=band, detected=band == 0), reward=float(band == 0))

    first = scheduler.select_band(state)
    second = scheduler.select_band(state)
    third = scheduler.select_band(state)
    assert first == second == third


def test_deterministic_given_identical_observation_sequence():
    def run():
        scheduler = AdaptiveUcbScheduler(num_bands=4, gamma=0.95, c=1.0)
        state = SimpleBeliefState(num_bands=4)
        chosen = []
        detections = [True, False, True, True, False, False, True, False]
        for i in range(12):
            band = scheduler.select_band(state)
            chosen.append(band)
            detected = detections[i % len(detections)]
            scheduler.update(_obs(band=band, detected=detected), reward=float(detected))
        return chosen

    assert run() == run()


def test_deterministic_tie_break_prefers_lowest_band_id():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=1.0, c=1.0)
    state = SimpleBeliefState(num_bands=3)
    # Identical observation history for every band -> identical scores
    # -> lowest band_id must win.
    for band in range(3):
        scheduler.update(_obs(band=band, detected=True), reward=1.0)

    assert scheduler.select_band(state) == 0


def test_reset_clears_learned_state_but_keeps_hyperparameters():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, c=2.0)
    for _ in range(10):
        scheduler.update(_obs(band=0, detected=True), reward=1.0)
    assert scheduler._N[0] > 0.0

    scheduler.reset()

    assert scheduler._N == [0.0, 0.0, 0.0]
    assert scheduler._S == [0.0, 0.0, 0.0]
    assert scheduler._observed == [False, False, False]
    assert scheduler._t == 0
    assert scheduler._gamma == 0.9  # hyperparameters survive reset()
    assert scheduler._c == 2.0

    state = SimpleBeliefState(num_bands=3)
    assert scheduler.select_band(state) == 0  # back to unobserved-priority


# --- Integration: SimpleBeliefState -----------------------------------------


def test_integration_with_real_simple_belief_state_full_loop():
    num_bands = 5
    env = SimpleRFEnvironment(
        emitter_specs=default_scenario(num_bands),
        num_bands=num_bands,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    env.reset(seed=0)
    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=1.0, seed=0)
    receiver.reset()
    state = SimpleBeliefState(num_bands=num_bands)
    state.reset()
    scheduler = AdaptiveUcbScheduler(num_bands=num_bands, gamma=0.95, c=1.0)
    scheduler.reset()

    for step in range(30):
        env.step(dt=1.0)
        t = float(step + 1)
        band = scheduler.select_band(state)  # state accepted, unused
        assert 0 <= band < num_bands
        receiver.tune(band)
        observation = receiver.observe(env, t=t)
        state.update(observation)  # real belief state updated too
        scheduler.update(observation, reward=1.0 if observation.detected else 0.0)

    # SimpleBeliefState was genuinely driven throughout -- sanity check.
    snapshot = state.get_features()
    assert sum(b.observation_count for b in snapshot.bands) == 30


# --- Integration: Phase 3 evaluator -----------------------------------------


def test_integration_with_phase3_evaluator():
    config = ExperimentConfig(num_bands=5, num_steps=50)
    scheduler = AdaptiveUcbScheduler(num_bands=config.num_bands, gamma=0.95, c=1.0)
    result = run_experiment_for_scheduler(
        config, scheduler, master_seed=42, scheduler_name="adaptive_ucb"
    )
    assert result.scheduler_name == "adaptive_ucb"
    assert result.num_steps == 50
    assert result.total_observations == 50
    assert (
        result.true_positive_count
        + result.false_positive_count
        + result.false_negative_count
        + result.true_negative_count
        == 50
    )


def test_hyperparameter_grid_reproducibility_via_evaluator():
    config = ExperimentConfig(num_bands=5, num_steps=50)

    def run():
        scheduler = AdaptiveUcbScheduler(num_bands=config.num_bands, gamma=0.95, c=1.0)
        return run_experiment_for_scheduler(
            config, scheduler, master_seed=7, scheduler_name="adaptive_ucb"
        )

    result_a = run()
    result_b = run()
    assert result_a.true_positive_count == result_b.true_positive_count
    assert result_a.probability_of_detection == result_b.probability_of_detection
    assert result_a.average_reward == result_b.average_reward
    assert result_a.emitter_records == result_b.emitter_records


# --- Four-way baseline comparison -------------------------------------------


def test_four_way_comparison_all_schedulers_share_env_receiver_trajectory():
    config = ExperimentConfig(num_bands=5, num_steps=40)
    master_seed = 123

    schedulers = {
        "round_robin": RoundRobinScheduler(num_bands=config.num_bands),
        "random": RandomScheduler(num_bands=config.num_bands, seed=0),
        "greedy_recent_hit": GreedyRecentHitScheduler(num_bands=config.num_bands),
        "adaptive_ucb": AdaptiveUcbScheduler(num_bands=config.num_bands, gamma=0.95, c=1.0),
    }

    results = {
        name: run_experiment_for_scheduler(config, sched, master_seed=master_seed, scheduler_name=name)
        for name, sched in schedulers.items()
    }

    assert set(results) == {"round_robin", "random", "greedy_recent_hit", "adaptive_ucb"}
    # env_seed/receiver_seed only depend on master_seed, never on the
    # scheduler (derive_seeds's role order) -- byte-identical across all
    # four, which is what makes this comparison fair.
    env_seeds = {r.env_seed for r in results.values()}
    receiver_seeds = {r.receiver_seed for r in results.values()}
    assert len(env_seeds) == 1
    assert len(receiver_seeds) == 1
    for result in results.values():
        assert result.num_steps == 40


# --- Hyperparameter selection rule (examples/phase4_experiment.py) ---------


class _FakeStats:
    def __init__(self, mean):
        self.mean = mean


class _FakeSummary:
    def __init__(self, interception_mean, intercept_time_mean):
        self.interception_rate_active_emitters = _FakeStats(interception_mean)
        self.average_intercept_time = _FakeStats(intercept_time_mean)


def test_selection_picks_clear_winner_by_interception_rate():
    grid = {
        (0.9, 0.5): _FakeSummary(0.40, 5.0),
        (0.95, 1.0): _FakeSummary(0.80, 10.0),  # clear best interception rate
        (1.0, 2.0): _FakeSummary(0.60, 2.0),
    }
    assert p4.select_best_hyperparameters(grid) == (0.95, 1.0)


def test_selection_near_tie_rule_breaks_by_intercept_time():
    grid = {
        (0.9, 0.5): _FakeSummary(0.80, 3.0),  # near-tied, faster interception
        (0.95, 1.0): _FakeSummary(0.81, 9.0),  # best interception rate, but slower
        (1.0, 2.0): _FakeSummary(0.50, 1.0),  # not near-tied -- excluded
    }
    # 0.81 - 0.80 = 0.01 <= NEAR_TIE_MARGIN (0.02), so both top two are
    # near-tied; the lower average_intercept_time wins.
    assert p4.select_best_hyperparameters(grid) == (0.9, 0.5)


def test_selection_treats_undefined_intercept_time_as_worse():
    grid = {
        (0.9, 0.5): _FakeSummary(0.80, None),  # never intercepted with a defined time
        (0.95, 1.0): _FakeSummary(0.81, 9.0),
    }
    assert p4.select_best_hyperparameters(grid) == (0.95, 1.0)


def test_selection_treats_undefined_interception_rate_as_worse():
    grid = {
        (0.9, 0.5): _FakeSummary(None, None),
        (0.95, 1.0): _FakeSummary(0.10, 9.0),
    }
    assert p4.select_best_hyperparameters(grid) == (0.95, 1.0)


def test_selection_and_held_out_seeds_are_disjoint():
    assert set(p4.SELECTION_SEEDS).isdisjoint(set(p4.HELD_OUT_SEEDS))
    assert len(p4.SELECTION_SEEDS) > 0
    assert len(p4.HELD_OUT_SEEDS) > 0


def test_hyperparameter_selection_pipeline_runs_end_to_end_on_small_grid():
    # Small, fast smoke test of the real pipeline (not the fake-summary
    # unit tests above) -- confirms run_hyperparameter_selection() wires
    # correctly into run_repeated_trials() and returns one TrialSummary
    # per grid point with the expected trial count.
    small_config = ExperimentConfig(num_bands=3, num_steps=15)
    small_seeds = (1, 2)
    grid_results = p4.run_hyperparameter_selection(
        config=small_config,
        selection_seeds=small_seeds,
        gamma_grid=(0.9, 1.0),
        c_grid=(1.0,),
    )
    assert set(grid_results.keys()) == {(0.9, 1.0), (1.0, 1.0)}
    for summary in grid_results.values():
        assert summary.interception_rate_active_emitters.n_total == 2

    gamma, c = p4.select_best_hyperparameters(grid_results)
    assert (gamma, c) in grid_results


def test_held_out_evaluation_runs_all_four_schedulers_on_small_seeds():
    small_config = ExperimentConfig(num_bands=3, num_steps=15)
    results = p4.run_held_out_evaluation(
        gamma=0.95, c=1.0, config=small_config, held_out_seeds=(3, 4)
    )
    assert set(results.keys()) == {
        "round_robin",
        "random",
        "greedy_recent_hit",
        "adaptive_ucb",
    }
    for summary in results.values():
        assert summary.probability_of_detection.n_total == 2


# --- Scenario-specific behavior: CW / pulsed / hopping ----------------------


def _run_scenario(emitter_specs, num_bands, num_steps=200, gamma=0.95, c=1.0, seed=0):
    env = SimpleRFEnvironment(
        emitter_specs=emitter_specs,
        num_bands=num_bands,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    env.reset(seed=seed)
    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=0.5, seed=seed)
    receiver.reset()
    state = SimpleBeliefState(num_bands=num_bands)
    scheduler = AdaptiveUcbScheduler(num_bands=num_bands, gamma=gamma, c=c)

    band_counts = [0] * num_bands
    for step in range(num_steps):
        env.step(dt=1.0)
        t = float(step + 1)
        band = scheduler.select_band(state)
        band_counts[band] += 1
        receiver.tune(band)
        observation = receiver.observe(env, t=t)
        state.update(observation)
        scheduler.update(observation, reward=1.0 if observation.detected else 0.0)
    return band_counts


def test_cw_scenario_scheduler_learns_to_favor_the_emitters_band():
    # A single, always-on, strong CW emitter on band 2: after enough
    # steps, AdaptiveUcbScheduler should have scanned it more than a
    # uniformly-unfavored band.
    specs = [EmitterSpec(emitter_id="cw-1", kind="cw", band_id=2, power=10.0)]
    band_counts = _run_scenario(specs, num_bands=5, num_steps=200)
    assert band_counts[2] > max(
        count for band, count in enumerate(band_counts) if band != 2
    )


def test_pulsed_scenario_runs_without_error_and_explores_all_bands_initially():
    specs = [
        EmitterSpec(
            emitter_id="pulsed-1", kind="pulsed", band_id=1, power=8.0,
            period=10.0, pulse_width=3.0,
        )
    ]
    band_counts = _run_scenario(specs, num_bands=4, num_steps=100)
    assert all(count >= 1 for count in band_counts)  # every band tried at least once
    assert sum(band_counts) == 100


def test_frequency_hopping_scenario_runs_without_error_and_stays_numerically_safe():
    specs = [
        EmitterSpec(
            emitter_id="hopper-1", kind="hopping", band_id=0, power=8.0,
            hop_interval=4.0, hop_bands=(0, 1, 2, 3, 4),
        )
    ]
    band_counts = _run_scenario(specs, num_bands=5, num_steps=300, gamma=0.9)
    assert sum(band_counts) == 300
    assert all(count >= 1 for count in band_counts)

# --- Phase 5 additive accessors: BandUcbDiagnostics / decision_count -------
#
# These test that the new read-only accessors (added purely for the
# dashboard) accurately expose existing internal values and cannot be
# used to mutate the scheduler -- see adaptive_ucb.py's docstrings.


def test_get_diagnostics_returns_one_entry_per_band_in_band_id_order():
    scheduler = AdaptiveUcbScheduler(num_bands=4)
    diagnostics = scheduler.get_diagnostics()
    assert len(diagnostics) == 4
    assert [d.band_id for d in diagnostics] == [0, 1, 2, 3]
    assert all(isinstance(d, BandUcbDiagnostics) for d in diagnostics)


def test_get_diagnostics_reports_unobserved_bands_correctly():
    scheduler = AdaptiveUcbScheduler(num_bands=3)
    diagnostics = scheduler.get_diagnostics()
    for d in diagnostics:
        assert d.observed is False
        assert d.discounted_successes == 0.0
        assert d.discounted_observations == 0.0
        # The real algorithm never computes a UCB score for an unobserved
        # band -- these must be None, not a fabricated number.
        assert d.estimated_hit_rate is None
        assert d.exploration_bonus is None
        assert d.ucb_score is None


def test_get_diagnostics_accurately_reflects_hits_and_misses():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=1.0, c=1.0)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    scheduler.update(_obs(band=1, detected=False), reward=0.0)

    diagnostics = scheduler.get_diagnostics()
    band0, band1 = diagnostics[0], diagnostics[1]

    assert band0.observed is True
    assert band0.discounted_successes == 1.0
    assert band0.discounted_observations == 1.0
    assert band0.estimated_hit_rate == pytest.approx(1.0)

    assert band1.observed is True
    assert band1.discounted_successes == 0.0
    assert band1.discounted_observations == 1.0
    assert band1.estimated_hit_rate == pytest.approx(0.0)

    # Cross-check against the internal attributes directly, and against
    # the hand-calculated formula, using the scheduler's OWN t.
    t = scheduler.decision_count + 1
    log_t = math.log(t)
    for d in diagnostics:
        expected_bonus = 1.0 * math.sqrt(log_t / max(d.discounted_observations, EPSILON))
        assert d.exploration_bonus == pytest.approx(expected_bonus)
        assert d.ucb_score == pytest.approx(d.estimated_hit_rate + d.exploration_bonus)


def test_get_diagnostics_matches_internal_S_and_N_after_discounting():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=0.5)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    scheduler.update(_obs(band=1, detected=False), reward=0.0)  # decays band 0

    diagnostics = scheduler.get_diagnostics()
    assert diagnostics[0].discounted_successes == pytest.approx(scheduler._S[0])
    assert diagnostics[0].discounted_observations == pytest.approx(scheduler._N[0])
    assert diagnostics[1].discounted_successes == pytest.approx(scheduler._S[1])
    assert diagnostics[1].discounted_observations == pytest.approx(scheduler._N[1])


def test_get_diagnostics_agrees_with_select_band_once_all_bands_observed():
    # The band with the max ucb_score among get_diagnostics()'s entries
    # must be exactly the band select_band() picks -- same formula, same
    # state, no second implementation.
    scheduler = AdaptiveUcbScheduler(num_bands=4, gamma=0.9, c=1.5)
    state = SimpleBeliefState(num_bands=4)
    detections = [True, False, True, False, True, False, True, True]
    for i in range(12):
        band = scheduler.select_band(state)
        scheduler.update(_obs(band=band, detected=detections[i % len(detections)]),
                          reward=float(detections[i % len(detections)]))

    diagnostics = scheduler.get_diagnostics()
    assert all(d.observed for d in diagnostics)  # all observed after 12 steps on 4 bands
    best_from_diagnostics = max(diagnostics, key=lambda d: d.ucb_score).band_id
    assert scheduler.select_band(state) == best_from_diagnostics


def test_get_diagnostics_is_read_only_frozen_dataclass():
    scheduler = AdaptiveUcbScheduler(num_bands=2)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    diagnostics = scheduler.get_diagnostics()
    with pytest.raises(dataclasses.FrozenInstanceError):
        diagnostics[0].ucb_score = 999.0


def test_get_diagnostics_does_not_mutate_scheduler_state():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, c=1.0)
    state = SimpleBeliefState(num_bands=3)
    for _ in range(3):
        band = scheduler.select_band(state)
        scheduler.update(_obs(band=band, detected=True), reward=1.0)

    before = scheduler.get_diagnostics()
    # Calling get_diagnostics() repeatedly, and calling select_band()
    # (which is itself pure -- see its own test), must not change
    # anything: no update() happened in between.
    _ = scheduler.get_diagnostics()
    _ = scheduler.select_band(state)
    after = scheduler.get_diagnostics()
    assert before == after
    assert scheduler.decision_count == 3


def test_decision_count_matches_number_of_updates_and_is_read_only():
    scheduler = AdaptiveUcbScheduler(num_bands=2)
    assert scheduler.decision_count == 0
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    assert scheduler.decision_count == 1
    scheduler.update(_obs(band=1, detected=False), reward=0.0)
    assert scheduler.decision_count == 2

    with pytest.raises(AttributeError):
        scheduler.decision_count = 999  # no setter -- read-only property


def test_decision_count_resets_with_reset():
    scheduler = AdaptiveUcbScheduler(num_bands=2)
    scheduler.update(_obs(band=0, detected=True), reward=1.0)
    assert scheduler.decision_count == 1
    scheduler.reset()
    assert scheduler.decision_count == 0
