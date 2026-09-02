"""Phase 1 tests for emitter behavior classes.

These test the internal Emitter classes directly (not through
RFEnvironment) to isolate behavior: timing/duty-cycle correctness for
ContinuousWaveEmitter/PulsedEmitter, and hop determinism (given a seeded
RNG) for FrequencyHoppingEmitter.
"""

import random

import pytest

from smart_scan_ew.environment.emitters import (
    ContinuousWaveEmitter,
    EmitterSpec,
    FrequencyHoppingEmitter,
    PulsedEmitter,
    build_emitter,
)


def test_continuous_wave_always_active_on_fixed_band():
    spec = EmitterSpec(emitter_id="cw", kind="cw", band_id=2, power=5.0)
    emitter = ContinuousWaveEmitter(spec)
    rng = random.Random(0)
    emitter.reset(rng)
    for t in [0.0, 1.0, 100.0, 999.5]:
        emitter.advance(t, rng)
        assert emitter.is_active(t) is True
        assert emitter.current_band(t) == 2
        assert emitter.current_power(t) == 5.0


def test_pulsed_duty_cycle_timing():
    # period=10, pulse_width=3 -> active for t in [0,3), [10,13), [20,23), ...
    spec = EmitterSpec(
        emitter_id="pulsed",
        kind="pulsed",
        band_id=1,
        power=4.0,
        period=10.0,
        pulse_width=3.0,
    )
    emitter = PulsedEmitter(spec)
    rng = random.Random(0)
    emitter.reset(rng)

    active_times = [0.0, 1.0, 2.9, 10.0, 12.9, 20.5]
    inactive_times = [3.0, 5.0, 9.9, 13.0, 19.9]

    for t in active_times:
        emitter.advance(t, rng)
        assert emitter.is_active(t) is True, f"expected active at t={t}"
        assert emitter.current_power(t) == 4.0

    for t in inactive_times:
        emitter.advance(t, rng)
        assert emitter.is_active(t) is False, f"expected inactive at t={t}"
        assert emitter.current_power(t) == 0.0


def test_pulsed_emitter_rejects_invalid_duty_cycle():
    bad_specs = [
        EmitterSpec(
            emitter_id="e", kind="pulsed", band_id=0, power=1.0,
            period=0.0, pulse_width=1.0,
        ),
        EmitterSpec(
            emitter_id="e", kind="pulsed", band_id=0, power=1.0,
            period=5.0, pulse_width=6.0,  # pulse_width > period
        ),
        EmitterSpec(
            emitter_id="e", kind="pulsed", band_id=0, power=1.0,
            period=5.0, pulse_width=None,
        ),
    ]
    for spec in bad_specs:
        with pytest.raises(ValueError):
            PulsedEmitter(spec)


def test_frequency_hopping_changes_band_at_hop_boundaries():
    spec = EmitterSpec(
        emitter_id="hopper",
        kind="hopping",
        band_id=0,
        power=6.0,
        hop_interval=5.0,
        hop_bands=(0, 1, 2, 3, 4),
    )
    emitter = FrequencyHoppingEmitter(spec, available_bands=(0, 1, 2, 3, 4))
    rng = random.Random(0)
    emitter.reset(rng)

    band_at_0 = emitter.current_band(0.0)
    assert band_at_0 == 0  # spec.band_id before any hop

    emitter.advance(4.9, rng)
    assert emitter.current_band(4.9) == 0  # no hop yet

    emitter.advance(5.0, rng)
    band_after_first_hop = emitter.current_band(5.0)
    assert band_after_first_hop in (0, 1, 2, 3, 4)

    # Always active regardless of band.
    assert emitter.is_active(5.0) is True
    assert emitter.current_power(5.0) == 6.0


def test_frequency_hopping_is_reproducible_given_same_seed():
    def run(seed: int) -> list[int]:
        spec = EmitterSpec(
            emitter_id="hopper",
            kind="hopping",
            band_id=0,
            power=6.0,
            hop_interval=2.0,
            hop_bands=(0, 1, 2, 3, 4),
        )
        emitter = FrequencyHoppingEmitter(spec, available_bands=(0, 1, 2, 3, 4))
        rng = random.Random(seed)
        emitter.reset(rng)
        bands = []
        for t in [2.0, 4.0, 6.0, 8.0, 10.0]:
            emitter.advance(t, rng)
            bands.append(emitter.current_band(t))
        return bands

    assert run(seed=42) == run(seed=42)


def test_frequency_hopping_large_dt_still_advances_correctly():
    # Jumping straight to t=17 (with hop_interval=5) should behave the same
    # as advancing step by step, since advance() loops over every crossed
    # hop boundary and consumes one rng draw each time.
    def run_stepwise(seed: int) -> int:
        spec = EmitterSpec(
            emitter_id="hopper", kind="hopping", band_id=0, power=1.0,
            hop_interval=5.0, hop_bands=(0, 1, 2),
        )
        emitter = FrequencyHoppingEmitter(spec, available_bands=(0, 1, 2))
        rng = random.Random(seed)
        emitter.reset(rng)
        for t in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 15.0, 17.0]:
            emitter.advance(t, rng)
        return emitter.current_band(17.0)

    def run_one_jump(seed: int) -> int:
        spec = EmitterSpec(
            emitter_id="hopper", kind="hopping", band_id=0, power=1.0,
            hop_interval=5.0, hop_bands=(0, 1, 2),
        )
        emitter = FrequencyHoppingEmitter(spec, available_bands=(0, 1, 2))
        rng = random.Random(seed)
        emitter.reset(rng)
        emitter.advance(17.0, rng)
        return emitter.current_band(17.0)

    assert run_stepwise(seed=1) == run_one_jump(seed=1)


def test_build_emitter_factory_dispatches_on_kind():
    cw_spec = EmitterSpec(emitter_id="a", kind="cw", band_id=0, power=1.0)
    assert isinstance(build_emitter(cw_spec, (0, 1)), ContinuousWaveEmitter)

    pulsed_spec = EmitterSpec(
        emitter_id="b", kind="pulsed", band_id=0, power=1.0,
        period=1.0, pulse_width=0.5,
    )
    assert isinstance(build_emitter(pulsed_spec, (0, 1)), PulsedEmitter)

    hop_spec = EmitterSpec(
        emitter_id="c", kind="hopping", band_id=0, power=1.0, hop_interval=1.0,
    )
    assert isinstance(build_emitter(hop_spec, (0, 1)), FrequencyHoppingEmitter)


def test_build_emitter_rejects_unknown_kind():
    bad_spec = EmitterSpec(emitter_id="x", kind="unknown", band_id=0, power=1.0)
    with pytest.raises(ValueError):
        build_emitter(bad_spec, (0, 1))
