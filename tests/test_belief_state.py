"""Phase 2 tests for SimpleBeliefState.

SimpleBeliefState must only ever be driven by Observation objects — these
tests never import or construct anything from the environment package.
"""

import pytest

from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.state import State
from smart_scan_ew.state import BandBeliefView, BeliefSnapshot, SimpleBeliefState


def test_is_a_state():
    assert isinstance(SimpleBeliefState(num_bands=3), State)


def test_fresh_state_reports_all_bands_as_never_observed():
    state = SimpleBeliefState(num_bands=4)
    snapshot = state.get_features()

    assert isinstance(snapshot, BeliefSnapshot)
    assert snapshot.current_time == 0.0
    assert len(snapshot.bands) == 4
    assert [b.band_id for b in snapshot.bands] == [0, 1, 2, 3]

    for band in snapshot.bands:
        assert isinstance(band, BandBeliefView)
        assert band.observation_count == 0
        assert band.hit_count == 0
        assert band.last_detected is None
        assert band.last_observed_time is None
        assert band.time_since_last_observed is None
        assert band.estimated_probability is None


def test_update_increments_counts_for_the_observed_band_only():
    state = SimpleBeliefState(num_bands=3)
    state.update(Observation(time=1.0, band=1, detected=True))

    snapshot = state.get_features()
    by_id = {b.band_id: b for b in snapshot.bands}

    assert by_id[1].observation_count == 1
    assert by_id[1].hit_count == 1
    assert by_id[1].last_detected is True
    assert by_id[1].last_observed_time == 1.0

    # Other bands untouched.
    assert by_id[0].observation_count == 0
    assert by_id[2].observation_count == 0


def test_update_with_miss_records_last_detected_false():
    state = SimpleBeliefState(num_bands=2)
    state.update(Observation(time=1.0, band=0, detected=False))

    band = state.get_features().bands[0]
    assert band.observation_count == 1
    assert band.hit_count == 0
    assert band.last_detected is False


def test_time_since_last_observed_tracks_current_time():
    state = SimpleBeliefState(num_bands=2)
    state.update(Observation(time=5.0, band=0, detected=True))
    state.update(Observation(time=8.0, band=1, detected=False))

    snapshot = state.get_features()
    by_id = {b.band_id: b for b in snapshot.bands}

    assert snapshot.current_time == 8.0
    # band 0 was last observed at t=5, current_time is now 8.0
    assert by_id[0].time_since_last_observed == 3.0
    # band 1 was just observed at the current time
    assert by_id[1].time_since_last_observed == 0.0


def test_estimated_probability_is_plain_hit_ratio():
    state = SimpleBeliefState(num_bands=1)
    state.update(Observation(time=1.0, band=0, detected=True))
    state.update(Observation(time=2.0, band=0, detected=True))
    state.update(Observation(time=3.0, band=0, detected=False))
    state.update(Observation(time=4.0, band=0, detected=False))

    band = state.get_features().bands[0]
    assert band.observation_count == 4
    assert band.hit_count == 2
    assert band.estimated_probability == 0.5


def test_update_rejects_band_outside_configured_range():
    state = SimpleBeliefState(num_bands=2)
    with pytest.raises(ValueError):
        state.update(Observation(time=1.0, band=5, detected=True))


def test_reset_returns_to_fresh_state():
    state = SimpleBeliefState(num_bands=2)
    state.update(Observation(time=1.0, band=0, detected=True))
    state.update(Observation(time=2.0, band=1, detected=False))

    state.reset()
    snapshot = state.get_features()

    assert snapshot.current_time == 0.0
    for band in snapshot.bands:
        assert band.observation_count == 0
        assert band.last_detected is None
        assert band.last_observed_time is None


def test_snapshot_is_frozen():
    state = SimpleBeliefState(num_bands=1)
    snapshot = state.get_features()
    with pytest.raises(Exception):
        snapshot.current_time = 99.0
    with pytest.raises(Exception):
        snapshot.bands[0].observation_count = 99
