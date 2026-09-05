"""Phase 4 tests for AdaptiveUcbScheduler.

The hand-computed tests here reproduce, exactly, the arithmetic worked
through in ARCHITECTURE.md's Phase 4 "one CW emitter" example (gamma=0.9,
c=1.0, 5 bands, CW on band 2) -- these are the golden values that prove
the implementation matches the approved design, not just "a plausible
number came out."
"""

import math

import pytest

from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.scheduler.adaptive_ucb_scheduler import AdaptiveUcbScheduler
from smart_scan_ew.state import SimpleBeliefState


def _cw_observation(t: float, band: int, cw_band: int = 2) -> Observation:
    return Observation(time=t, band=band, detected=(band == cw_band))


def test_is_a_scheduler():
    assert isinstance(AdaptiveUcbScheduler(num_bands=5), Scheduler)


def test_constructor_validates_parameters():
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=0)
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=5, gamma=0.0)
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=5, gamma=1.5)
    with pytest.raises(ValueError):
        AdaptiveUcbScheduler(num_bands=5, exploration_constant=-1.0)
    # gamma == 1.0 is explicitly valid (no discounting).
    AdaptiveUcbScheduler(num_bands=5, gamma=1.0)


def test_initialization_visits_every_band_lowest_id_first():
    scheduler = AdaptiveUcbScheduler(num_bands=5, gamma=0.9, exploration_constant=1.0)
    state = SimpleBeliefState(num_bands=5)

    selected = []
    for t in range(1, 6):
        band = scheduler.select_band(state)
        selected.append(band)
        obs = _cw_observation(float(t), band)
        state.update(obs)
        scheduler.update(obs, reward=0.0)

    assert selected == [0, 1, 2, 3, 4]


def test_hand_computed_discounted_statistics_and_ucb_scores():
    # Reproduces ARCHITECTURE.md's Phase 4 worked example exactly:
    # 5 bands, gamma=0.9, c=1.0, CW emitter on band 2 (always detected).
    scheduler = AdaptiveUcbScheduler(num_bands=5, gamma=0.9, exploration_constant=1.0)
    state = SimpleBeliefState(num_bands=5)

    for t in range(1, 6):  # t=1..5: initialization phase
        band = scheduler.select_band(state)
        obs = _cw_observation(float(t), band)
        state.update(obs)
        scheduler.update(obs, reward=0.0)

    # After t=5 (hand-computed in the design doc):
    # N = [0.6561*... let's check against the actual decay chain]
    # N_0: observed at t=1 (miss), decayed 4 times (t=2,3,4,5) -> 0.9^4
    # N_1: observed at t=2 (miss), decayed 3 times -> 0.9^3
    # N_2: observed at t=3 (HIT), decayed 2 times -> 0.9^2, S_2 = 0.9^2
    # N_3: observed at t=4 (miss), decayed 1 time -> 0.9^1
    # N_4: observed at t=5 (miss), decayed 0 times -> 1.0
    expected_n = [0.9**4, 0.9**3, 0.9**2, 0.9**1, 1.0]
    expected_s = [0.0, 0.0, 0.9**2, 0.0, 0.0]
    for i in range(5):
        assert scheduler._counts[i] == pytest.approx(expected_n[i])
        assert scheduler._successes[i] == pytest.approx(expected_s[i])

    # t=6: every band has N_b >= epsilon now, so the UCB formula is used.
    band = scheduler.select_band(state)
    log_t = math.log(6)
    scores = [
        (scheduler._successes[i] / scheduler._counts[i])
        + 1.0 * math.sqrt(log_t / scheduler._counts[i])
        for i in range(5)
    ]
    assert band == max(range(5), key=lambda i: scores[i])
    assert band == 2  # band 2's p_b=1.0 dominates -- matches the design doc


def test_gamma_one_reproduces_plain_undiscounted_ucb1():
    # With gamma=1.0, decaying every band by 1.0 changes nothing, so S_b
    # and N_b become ordinary (undiscounted) cumulative counts -- this
    # test proves that numerically, not just by reading the formula.
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=1.0, exploration_constant=1.0)
    state = SimpleBeliefState(num_bands=3)

    observations = [
        (1, 0, False), (2, 1, False), (3, 2, True),
        (4, 2, True), (5, 0, False), (6, 2, False),
    ]
    for t, band, detected in observations:
        obs = Observation(time=float(t), band=band, detected=detected)
        state.update(obs)
        scheduler.update(obs, reward=0.0)

    # Plain cumulative counts: band0 observed twice (0 hits), band1 once
    # (0 hits), band2 three times (2 hits).
    assert scheduler._counts == pytest.approx([2.0, 1.0, 3.0])
    assert scheduler._successes == pytest.approx([0.0, 0.0, 2.0])


def test_exploration_prefers_under_sampled_band_when_hit_rates_are_close():
    # Two bands with identical hit rates but very different sample
    # counts: the confidence bound should favor the less-sampled one.
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=1.0, exploration_constant=2.0)
    # Manually set internal state to simulate "band 0 heavily sampled at
    # 50% hit rate, band 1 lightly sampled at 50% hit rate".
    scheduler._counts = [20.0, 2.0]
    scheduler._successes = [10.0, 1.0]
    scheduler._t = 20

    state = SimpleBeliefState(num_bands=2)
    band = scheduler.select_band(state)
    assert band == 1  # same p_b=0.5, but band 1's exploration bonus is larger


def test_previously_observed_band_with_decayed_n_gets_revisited_eventually():
    # A band observed once early, then never again for a long stretch,
    # should eventually be re-selected once its exploration bonus grows
    # large enough -- demonstrating the recency/neglect-prevention
    # mechanism (not a formal guarantee, an observed structural property).
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.8, exploration_constant=1.0)
    state = SimpleBeliefState(num_bands=3)

    # Initialization: bands 0,1,2 each observed once (all misses).
    for t in range(1, 4):
        obs = Observation(time=float(t), band=t - 1, detected=False)
        state.update(obs)
        scheduler.update(obs, reward=0.0)

    # Now repeatedly "camp" on band 1 (always a miss there too), letting
    # bands 0 and 2's N decay toward zero.
    revisited_others = False
    for t in range(4, 60):
        band = scheduler.select_band(state)
        if band != 1:
            revisited_others = True
            break
        obs = Observation(time=float(t), band=1, detected=False)
        state.update(obs)
        scheduler.update(obs, reward=0.0)

    assert revisited_others, "expected the scheduler to eventually recheck a neglected band"


def test_t_progression_and_reset():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, exploration_constant=1.0)
    state = SimpleBeliefState(num_bands=3)
    assert scheduler._t == 0

    scheduler.select_band(state)
    assert scheduler._t == 1
    scheduler.select_band(state)
    assert scheduler._t == 2

    scheduler.reset()
    assert scheduler._t == 0
    assert scheduler._counts == [0.0, 0.0, 0.0]
    assert scheduler._successes == [0.0, 0.0, 0.0]


def test_reset_fully_clears_learned_statistics():
    scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, exploration_constant=1.0)
    state = SimpleBeliefState(num_bands=3)
    for t in range(1, 10):
        band = scheduler.select_band(state)
        obs = Observation(time=float(t), band=band, detected=(band == 1))
        state.update(obs)
        scheduler.update(obs, reward=0.0)

    assert any(n > 0 for n in scheduler._counts)  # something was learned

    scheduler.reset()
    assert scheduler._counts == [0.0, 0.0, 0.0]
    assert scheduler._successes == [0.0, 0.0, 0.0]
    assert scheduler._t == 0
    # After reset, behaves like a fresh instance: re-initializes from band 0.
    assert scheduler.select_band(state) == 0


def test_deterministic_tie_breaking_lowest_band_id():
    # Identical N and S for two bands -> identical scores -> lowest
    # band_id wins, deterministically, every time.
    scheduler = AdaptiveUcbScheduler(num_bands=4, gamma=1.0, exploration_constant=1.0)
    scheduler._counts = [5.0, 5.0, 5.0, 5.0]
    scheduler._successes = [2.0, 2.0, 2.0, 2.0]
    scheduler._t = 10

    state = SimpleBeliefState(num_bands=4)
    for _ in range(3):
        assert scheduler.select_band(state) == 0


def test_numerical_edge_case_n_near_epsilon_is_treated_as_never_observed():
    scheduler = AdaptiveUcbScheduler(num_bands=2, gamma=0.9, exploration_constant=1.0)
    # Simulate a band whose discounted N has decayed to just below the
    # epsilon floor -- must be treated as "never observed", not fed into
    # the UCB formula (which would otherwise divide by a near-zero N_b).
    scheduler._counts = [1e-10, 3.0]
    scheduler._successes = [0.0, 1.5]
    scheduler._t = 5

    state = SimpleBeliefState(num_bands=2)
    # Must not raise (no division producing inf/nan), and must select the
    # near-zero band via the initialization branch, not the formula.
    band = scheduler.select_band(state)
    assert band == 0


def test_reward_argument_is_ignored():
    # Two runs with identical detection outcomes but wildly different
    # (even nonsensical) reward values must produce identical learned
    # statistics -- proving `reward` has zero effect on this scheduler.
    def run(rewards):
        scheduler = AdaptiveUcbScheduler(num_bands=3, gamma=0.9, exploration_constant=1.0)
        state = SimpleBeliefState(num_bands=3)
        for t, (band, detected, reward) in enumerate(rewards, start=1):
            obs = Observation(time=float(t), band=band, detected=detected)
            state.update(obs)
            scheduler.update(obs, reward=reward)
        return scheduler._counts, scheduler._successes

    sequence_a = [(0, False, 0.0), (1, True, 0.0), (2, False, 0.0), (1, True, 0.0)]
    sequence_b = [(0, False, -999.0), (1, True, 42.0), (2, False, 1e6), (1, True, -0.001)]

    counts_a, successes_a = run(sequence_a)
    counts_b, successes_b = run(sequence_b)
    assert counts_a == pytest.approx(counts_b)
    assert successes_a == pytest.approx(successes_b)
