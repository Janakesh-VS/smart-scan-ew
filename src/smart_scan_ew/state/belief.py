"""SimpleBeliefState: a real, but intentionally simple, State/belief.

Per the project owner's Phase 2 approval, this implements exactly:
observation_count, hit_count, last_detected, last_observed_time,
time_since_last_observed, estimated_probability — per band. Deliberately
NOT included: Bayesian priors, smoothing, exponential decay, confidence
scores, hidden-state inference, or any ML.

`estimated_probability` is calculated and exposed for future
learning/analysis modules, but no Phase 2 baseline scheduler uses it as a
decision criterion.

This module imports only from `interfaces/` (specifically
`interfaces.observation.Observation` and `interfaces.state.State`). It
must never import `RFEnvironment`, `EmitterSpec`, `GroundTruthSnapshot`,
or anything from the `environment` package — the only information that
ever enters this class is the `Observation` objects passed to `update()`.
"""

from dataclasses import dataclass

from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.state import State


@dataclass(frozen=True)
class BandBeliefView:
    """Read-only snapshot of what is believed about one band, derived
    strictly from past Observations of that band."""

    band_id: int
    observation_count: int
    hit_count: int
    last_detected: bool | None
    """Result of the most recent observation of this band, or None if this
    band has never been observed."""
    last_observed_time: float | None
    """Simulation time of the most recent observation of this band, or
    None if this band has never been observed."""
    time_since_last_observed: float | None
    """current_time - last_observed_time, or None if this band has never
    been observed."""
    estimated_probability: float | None
    """hit_count / observation_count, or None if observation_count == 0.
    No smoothing or prior — a plain frequency ratio."""


@dataclass(frozen=True)
class BeliefSnapshot:
    """The full belief state at one instant, as returned by
    SimpleBeliefState.get_features()."""

    current_time: float
    bands: tuple[BandBeliefView, ...]
    """One entry per band, ordered by band_id 0..num_bands-1. Every band
    is always present, even if never observed."""


@dataclass
class _MutableBandBelief:
    """Internal, mutable per-band record. Not exposed outside this
    module — get_features() converts these into frozen BandBeliefViews."""

    band_id: int
    observation_count: int = 0
    hit_count: int = 0
    last_detected: bool | None = None
    last_observed_time: float | None = None


class SimpleBeliefState(State):
    """Tracks, per band, only what past Observations reveal.

    Constructed with an explicit `num_bands` (the receiver's own scan
    range — a known, non-secret configuration value, not ground truth
    about emitters). Every band 0..num_bands-1 has a belief record from
    construction, so `get_features()` always reports on the full band
    range, including bands never yet observed.
    """

    def __init__(self, num_bands: int):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        self._num_bands = num_bands
        self._current_time = 0.0
        self._bands: dict[int, _MutableBandBelief] = {}
        self._init_bands()

    def _init_bands(self) -> None:
        self._bands = {
            band_id: _MutableBandBelief(band_id=band_id)
            for band_id in range(self._num_bands)
        }

    def reset(self) -> None:
        self._current_time = 0.0
        self._init_bands()

    def update(self, observation: Observation) -> None:
        band_id = observation.band
        if band_id not in self._bands:
            raise ValueError(
                f"Observation references band {band_id!r}, which is "
                f"outside this belief state's range of 0..{self._num_bands - 1}"
            )
        record = self._bands[band_id]
        record.observation_count += 1
        if observation.detected:
            record.hit_count += 1
        record.last_detected = observation.detected
        record.last_observed_time = observation.time
        self._current_time = observation.time

    def get_features(self) -> BeliefSnapshot:
        views = tuple(
            self._to_view(self._bands[band_id])
            for band_id in range(self._num_bands)
        )
        return BeliefSnapshot(current_time=self._current_time, bands=views)

    def _to_view(self, record: _MutableBandBelief) -> BandBeliefView:
        if record.last_observed_time is None:
            time_since_last_observed = None
        else:
            time_since_last_observed = self._current_time - record.last_observed_time

        if record.observation_count == 0:
            estimated_probability = None
        else:
            estimated_probability = record.hit_count / record.observation_count

        return BandBeliefView(
            band_id=record.band_id,
            observation_count=record.observation_count,
            hit_count=record.hit_count,
            last_detected=record.last_detected,
            last_observed_time=record.last_observed_time,
            time_since_last_observed=time_since_last_observed,
            estimated_probability=estimated_probability,
        )
