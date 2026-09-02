"""GreedyRecentHitScheduler: revisit wherever activity was most recently
confirmed; otherwise explore whatever has been looked at least.

See ARCHITECTURE.md's Phase 2 section for the full rationale, including
the emergent self-correcting behavior when a previously-hit band later
reports a miss.
"""

from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State


class GreedyRecentHitScheduler(Scheduler):
    """Two-tier, fully deterministic decision rule, read fresh from
    `state.get_features()` on every call (no internal state of its own):

    1. Among bands with `last_detected is True`, pick the one with the
       largest `last_observed_time`. Ties broken by lowest `band_id`.
    2. If no band has ever recorded a hit, pick the band with the fewest
       `observation_count`. Ties broken by lowest `band_id`.

    Deliberately does NOT use `estimated_probability`, exploration
    probabilities, or decay — see the project owner's Phase 2 decisions.

    NOTE on coupling: `State.get_features()` is typed `-> Any` in
    `interfaces/state.py` — the abstract interface deliberately does not
    fix a snapshot shape. This scheduler therefore relies, structurally
    (duck-typed, not via import), on the snapshot exposing an iterable
    `.bands` of objects with `.last_detected`, `.last_observed_time`,
    `.band_id`, and `.observation_count`. It does not import
    `smart_scan_ew.state` — doing so would make `scheduler/` depend on
    `state/`'s concrete internals rather than only on `interfaces/`,
    which rule 1/6 (CLAUDE.md) reserves for cross-module contracts. In
    practice, `SimpleBeliefState`'s `BeliefSnapshot`/`BandBeliefView`
    (Phase 2) satisfy this shape.
    """

    def __init__(self, num_bands: int):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        self._num_bands = num_bands

    def reset(self) -> None:
        pass  # no internal state to clear — decisions are read fresh from `state` each call

    def select_band(self, state: State) -> Band:
        snapshot = state.get_features()
        bands = snapshot.bands

        best_hit = None
        for band in bands:
            if band.last_detected is True:
                if best_hit is None or band.last_observed_time > best_hit.last_observed_time:
                    best_hit = band
        if best_hit is not None:
            return best_hit.band_id

        # No band has ever recorded a hit: explore the least-observed one.
        # `bands` is ordered by band_id ascending, and we only replace on
        # a strictly smaller count, so the first (lowest band_id) among
        # ties is kept automatically.
        least_observed = bands[0]
        for band in bands[1:]:
            if band.observation_count < least_observed.observation_count:
                least_observed = band
        return least_observed.band_id

    def update(self, observation: Observation, reward: float) -> None:
        pass  # non-learning baseline: decisions come entirely from `state`
