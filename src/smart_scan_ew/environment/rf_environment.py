"""SimpleRFEnvironment: a real, but intentionally simple, RFEnvironment.

See ARCHITECTURE.md's "Phase 1" section for the decisions behind this
implementation (noise/detection model lives in the receiver, not here;
this module only owns ground truth and the physically-shallow `sense()`
aggregation).
"""

import random
from dataclasses import dataclass

from smart_scan_ew.environment.bands import BandSpec, build_band_table
from smart_scan_ew.environment.emitters import Emitter, EmitterSpec, build_emitter
from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.interfaces.observation import Band


@dataclass(frozen=True)
class EmitterGroundTruth:
    """Ground-truth state of a single emitter at one instant."""

    emitter_id: str
    active: bool
    band: int
    power: float


@dataclass(frozen=True)
class GroundTruthSnapshot:
    """Full ground-truth snapshot returned by `get_ground_truth()`.

    Evaluator-only information — see CLAUDE.md rule 4/5. Never pass this
    (or any piece of it) to a Receiver, State, or Scheduler.
    """

    time: float
    emitters: tuple[EmitterGroundTruth, ...]


class SimpleRFEnvironment(RFEnvironment):
    """A simple simulated RF world: a fixed band table plus a list of
    emitters (each an instance of ContinuousWaveEmitter, PulsedEmitter, or
    FrequencyHoppingEmitter).

    Explicitly out of scope (see ARCHITECTURE.md / PROJECT_CONTRACT.md):
    real waveform simulation, propagation, antenna, or multipath modeling.
    `sense()` is just a sum of transmit powers of whatever is active on
    the requested band.
    """

    def __init__(
        self,
        emitter_specs: list[EmitterSpec],
        num_bands: int,
        band_start_frequency_hz: float,
        band_width_hz: float,
    ):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")

        self._band_table: tuple[BandSpec, ...] = build_band_table(
            num_bands, band_start_frequency_hz, band_width_hz
        )
        valid_band_ids = tuple(b.band_id for b in self._band_table)

        for spec in emitter_specs:
            if spec.band_id not in valid_band_ids:
                raise ValueError(
                    f"Emitter {spec.emitter_id!r} references band_id "
                    f"{spec.band_id}, but the band table only has "
                    f"{num_bands} bands (0..{num_bands - 1})"
                )

        self._emitters: tuple[Emitter, ...] = tuple(
            build_emitter(spec, valid_band_ids) for spec in emitter_specs
        )

        self._t = 0.0
        self._rng = random.Random()  # replaced with a seeded one on reset()

    @property
    def band_table(self) -> tuple[BandSpec, ...]:
        """Read-only access to the band frequency table (metadata only,
        not ground truth about emitters)."""
        return self._band_table

    def reset(self, seed: int | None = None) -> None:
        self._t = 0.0
        self._rng = random.Random(seed)
        for emitter in self._emitters:
            emitter.reset(self._rng)

    def step(self, dt: float = 1.0) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._t += dt
        for emitter in self._emitters:
            emitter.advance(self._t, self._rng)

    def sense(self, band: Band, t: float) -> float:
        """Return the summed transmit power of active emitters on `band`.

        NOTE (documented limitation): `t` is accepted for interface
        compatibility but the environment is authoritative for simulation
        time — internal emitter state reflects the time of the most recent
        `step()` call, not the `t` argument. Callers are expected to pass
        the same time value the environment itself is at (as would
        naturally happen when an Evaluator harness calls `step()` then
        immediately has the Receiver observe). This is not validated here;
        see ARCHITECTURE.md Limitations.
        """
        return sum(
            emitter.current_power(self._t)
            for emitter in self._emitters
            if emitter.current_band(self._t) == band
        )

    def get_ground_truth(self) -> GroundTruthSnapshot:
        return GroundTruthSnapshot(
            time=self._t,
            emitters=tuple(
                EmitterGroundTruth(
                    emitter_id=emitter.spec.emitter_id,
                    active=emitter.is_active(self._t),
                    band=emitter.current_band(self._t),
                    power=emitter.current_power(self._t),
                )
                for emitter in self._emitters
            ),
        )
