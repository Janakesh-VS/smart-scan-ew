"""Phase 2 tests for the baseline schedulers: RoundRobinScheduler,
RandomScheduler, GreedyRecentHitScheduler.
"""

import inspect

import pytest

from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.scheduler import (
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from smart_scan_ew.state import SimpleBeliefState


@pytest.mark.parametrize(
    "scheduler",
    [
        RoundRobinScheduler(num_bands=4),
        RandomScheduler(num_bands=4, seed=0),
        GreedyRecentHitScheduler(num_bands=4),
    ],
)
def test_all_baselines_are_scheduler_instances(scheduler):
    assert isinstance(scheduler, Scheduler)


def test_scheduler_interface_never_accepts_an_environment():
    # Structural pin (same style as the Phase 0 test): no Scheduler method
    # accepts anything environment-shaped.
    select_band_params = inspect.signature(Scheduler.select_band).parameters
    update_params = inspect.signature(Scheduler.update).parameters
    assert "environment" not in select_band_params
    assert "environment" not in update_params


# --- RoundRobinScheduler ----------------------------------------------------

def test_round_robin_cycles_in_order_and_wraps():
    scheduler = RoundRobinScheduler(num_bands=3)
    state = SimpleBeliefState(num_bands=3)  # present but must be ignored
    sequence = [scheduler.select_band(state) for _ in range(7)]
    assert sequence == [0, 1, 2, 0, 1, 2, 0]


def test_round_robin_ignores_belief_contents():
    scheduler = RoundRobinScheduler(num_bands=3)
    state = SimpleBeliefState(num_bands=3)
    # Load band 2 up with hits — should have zero effect on round robin.
    for _ in range(5):
        state.update(Observation(time=1.0, band=2, detected=True))
    sequence = [scheduler.select_band(state) for _ in range(3)]
    assert sequence == [0, 1, 2]


def test_round_robin_reset_returns_to_band_zero():
    scheduler = RoundRobinScheduler(num_bands=3)
    state = SimpleBeliefState(num_bands=3)
    scheduler.select_band(state)
    scheduler.select_band(state)
    scheduler.reset()
    assert scheduler.select_band(state) == 0


# --- RandomScheduler ---------------------------------------------------------

def test_random_scheduler_values_are_in_range():
    scheduler = RandomScheduler(num_bands=5, seed=0)
    state = SimpleBeliefState(num_bands=5)
    for _ in range(50):
        band = scheduler.select_band(state)
        assert 0 <= band < 5


def test_random_scheduler_is_reproducible_given_same_seed():
    state = SimpleBeliefState(num_bands=5)

    scheduler_a = RandomScheduler(num_bands=5, seed=42)
    sequence_a = [scheduler_a.select_band(state) for _ in range(20)]

    scheduler_b = RandomScheduler(num_bands=5, seed=42)
    sequence_b = [scheduler_b.select_band(state) for _ in range(20)]

    assert sequence_a == sequence_b


def test_random_scheduler_reset_reproduces_the_same_sequence():
    state = SimpleBeliefState(num_bands=5)
    scheduler = RandomScheduler(num_bands=5, seed=7)

    first_run = [scheduler.select_band(state) for _ in range(10)]
    scheduler.reset()
    second_run = [scheduler.select_band(state) for _ in range(10)]

    assert first_run == second_run


def test_random_scheduler_ignores_belief_contents():
    state = SimpleBeliefState(num_bands=5)
    for _ in range(5):
        state.update(Observation(time=1.0, band=3, detected=True))

    scheduler_a = RandomScheduler(num_bands=5, seed=1)
    sequence_a = [scheduler_a.select_band(state) for _ in range(10)]

    empty_state = SimpleBeliefState(num_bands=5)
    scheduler_b = RandomScheduler(num_bands=5, seed=1)
    sequence_b = [scheduler_b.select_band(empty_state) for _ in range(10)]

    assert sequence_a == sequence_b


# --- GreedyRecentHitScheduler -------------------------------------------------

def test_greedy_explores_least_observed_band_when_no_hits_yet():
    state = SimpleBeliefState(num_bands=4)
    scheduler = GreedyRecentHitScheduler(num_bands=4)

    # No observations at all: all bands tied at observation_count=0 ->
    # deterministic tie-break picks lowest band_id.
    assert scheduler.select_band(state) == 0

    # Observe band 0 (a miss) so it's no longer least-observed.
    state.update(Observation(time=1.0, band=0, detected=False))
    assert scheduler.select_band(state) == 1

    state.update(Observation(time=2.0, band=1, detected=False))
    assert scheduler.select_band(state) == 2


def test_greedy_prefers_band_with_most_recent_hit():
    state = SimpleBeliefState(num_bands=3)
    scheduler = GreedyRecentHitScheduler(num_bands=3)

    state.update(Observation(time=1.0, band=0, detected=True))
    state.update(Observation(time=2.0, band=2, detected=True))
    # band 2's hit is more recent than band 0's hit.
    assert scheduler.select_band(state) == 2


def test_greedy_tie_break_uses_lowest_band_id_among_equally_recent_hits():
    # Two bands can't literally share a last_observed_time from the same
    # State (each update() call advances current_time), so this test
    # constructs the tie via two updates carrying the SAME explicit time.
    state = SimpleBeliefState(num_bands=3)
    scheduler = GreedyRecentHitScheduler(num_bands=3)

    state.update(Observation(time=5.0, band=2, detected=True))
    state.update(Observation(time=5.0, band=1, detected=True))
    # Both last_observed_time == 5.0 -> tie-break picks lowest band_id (1).
    assert scheduler.select_band(state) == 1


def test_greedy_falls_back_after_previously_hit_band_goes_stale():
    state = SimpleBeliefState(num_bands=3)
    scheduler = GreedyRecentHitScheduler(num_bands=3)

    state.update(Observation(time=1.0, band=1, detected=True))
    assert scheduler.select_band(state) == 1  # sticks to the hit

    # band 1 is re-observed and now reports a miss -> no bands have
    # last_detected True anymore -> falls back to least-observed.
    state.update(Observation(time=2.0, band=1, detected=False))
    band_ids_by_observation_count = {
        b.band_id: b.observation_count for b in state.get_features().bands
    }
    assert band_ids_by_observation_count == {0: 0, 1: 2, 2: 0}
    # bands 0 and 2 are tied least-observed at 0 -> lowest band_id (0).
    assert scheduler.select_band(state) == 0


def test_greedy_reset_is_a_no_op_since_it_reads_state_fresh():
    state = SimpleBeliefState(num_bands=2)
    scheduler = GreedyRecentHitScheduler(num_bands=2)
    state.update(Observation(time=1.0, band=1, detected=True))
    before = scheduler.select_band(state)
    scheduler.reset()
    after = scheduler.select_band(state)
    assert before == after == 1
