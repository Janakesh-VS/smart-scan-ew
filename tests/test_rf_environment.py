"""Phase 1 tests for SimpleRFEnvironment."""

import pytest

from smart_scan_ew.environment import (
    EmitterSpec,
    GroundTruthSnapshot,
    SimpleRFEnvironment,
    default_scenario,
)
from smart_scan_ew.interfaces import RFEnvironment


def make_env(num_bands=5, emitters=None, seed=0):
    env = SimpleRFEnvironment(
        emitter_specs=emitters if emitters is not None else default_scenario(num_bands),
        num_bands=num_bands,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    env.reset(seed=seed)
    return env


def test_simple_rf_environment_is_an_rf_environment():
    env = make_env()
    assert isinstance(env, RFEnvironment)


def test_reset_is_reproducible_for_hopping_bands():
    def run(seed):
        env = make_env(seed=seed)
        bands = []
        for _ in range(10):
            env.step(dt=1.0)
            bands.append(env.get_ground_truth())
        return bands

    run_a = run(seed=7)
    run_b = run(seed=7)
    assert [gt.time for gt in run_a] == [gt.time for gt in run_b]
    for snap_a, snap_b in zip(run_a, run_b):
        bands_a = {e.emitter_id: e.band for e in snap_a.emitters}
        bands_b = {e.emitter_id: e.band for e in snap_b.emitters}
        assert bands_a == bands_b


def test_step_advances_time():
    env = make_env()
    assert env.get_ground_truth().time == 0.0
    env.step(dt=1.0)
    assert env.get_ground_truth().time == 1.0
    env.step(dt=2.5)
    assert env.get_ground_truth().time == 3.5


def test_step_rejects_non_positive_dt():
    env = make_env()
    with pytest.raises(ValueError):
        env.step(dt=0.0)
    with pytest.raises(ValueError):
        env.step(dt=-1.0)


def test_sense_on_band_with_active_emitter_exceeds_empty_band():
    emitters = [
        EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=10.0),
    ]
    env = make_env(num_bands=5, emitters=emitters)
    env.step(dt=1.0)
    t = env.get_ground_truth().time

    occupied = env.sense(band=0, t=t)
    empty = env.sense(band=4, t=t)
    assert occupied == 10.0
    assert empty == 0.0


def test_sense_sums_multiple_emitters_on_same_band():
    emitters = [
        EmitterSpec(emitter_id="a", kind="cw", band_id=1, power=3.0),
        EmitterSpec(emitter_id="b", kind="cw", band_id=1, power=4.0),
    ]
    env = make_env(num_bands=3, emitters=emitters)
    env.step(dt=1.0)
    t = env.get_ground_truth().time
    assert env.sense(band=1, t=t) == 7.0


def test_get_ground_truth_reports_full_per_emitter_snapshot():
    emitters = [
        EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=5.0),
        EmitterSpec(
            emitter_id="pulsed-1", kind="pulsed", band_id=1, power=2.0,
            period=10.0, pulse_width=1.0,
        ),
    ]
    env = make_env(num_bands=3, emitters=emitters)
    env.step(dt=1.0)  # t=1.0: pulsed-1 still within pulse_width=1.0? 1.0 % 10 = 1.0, not < 1.0 -> inactive

    snapshot = env.get_ground_truth()
    assert isinstance(snapshot, GroundTruthSnapshot)
    by_id = {e.emitter_id: e for e in snapshot.emitters}

    assert by_id["cw-1"].active is True
    assert by_id["cw-1"].band == 0
    assert by_id["cw-1"].power == 5.0

    assert by_id["pulsed-1"].active is False
    assert by_id["pulsed-1"].power == 0.0


def test_rejects_emitter_spec_with_out_of_range_band():
    bad_spec = EmitterSpec(emitter_id="oops", kind="cw", band_id=99, power=1.0)
    with pytest.raises(ValueError):
        SimpleRFEnvironment(
            emitter_specs=[bad_spec],
            num_bands=5,
            band_start_frequency_hz=2.4e9,
            band_width_hz=20e6,
        )



def test_rejects_hopping_emitter_with_invalid_hop_band():
    bad_spec = EmitterSpec(
        emitter_id="hopper",
        kind="hopping",
        band_id=0,
        power=1.0,
        hop_interval=1.0,
        hop_bands=(0, 1, 999),
    )
    with pytest.raises(ValueError, match="invalid band IDs"):
        SimpleRFEnvironment(
            emitter_specs=[bad_spec],
            num_bands=5,
            band_start_frequency_hz=2.4e9,
            band_width_hz=20e6,
        )

def test_band_table_has_expected_shape():
    env = make_env(num_bands=5)
    table = env.band_table
    assert len(table) == 5
    assert [b.band_id for b in table] == [0, 1, 2, 3, 4]
    assert table[1].center_frequency == 2.4e9 + 20e6
    assert all(b.bandwidth == 20e6 for b in table)
