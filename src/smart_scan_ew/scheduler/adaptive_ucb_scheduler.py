"""AdaptiveUcbScheduler: Adaptive Discounted-UCB Multi-Armed Bandit
Scheduler (Phase 4).

Approved specification (see ARCHITECTURE.md's "Phase 4" section for the
full design rationale). Per-band discounted statistics:

    On every update() call, for EVERY band b:
        S_b <- gamma * S_b
        N_b <- gamma * N_b

    For the band b* actually observed this step:
        N_b* <- N_b* + 1
        if observation.detected:
            S_b* <- S_b* + 1

    Estimated hit rate:  p_b = S_b / N_b   (only defined once N_b >= epsilon)

    UCB score:  score(b) = p_b + c * sqrt(ln(t) / N_b)

Where `gamma` in (0, 1] is the single recency/memory parameter (no
separate alpha), `c >= 0` is the exploration constant, and `t` is the
scheduler's own ordinary (undiscounted) decision counter — NOT simulation
time, NOT a discounted quantity — starting at 0 after reset() and
becoming 1 for the first decision. Using an ordinary integer for `t`
(rather than a discounted total) is what keeps `ln(t)` always
well-defined (t >= 1 whenever the formula is evaluated at all — see
`select_band`'s cold-start handling), even though the per-band `N_b`
values are discounted and can become arbitrarily small.

NOT a contextual bandit: there is no shared feature model across bands,
just N independent per-band statistics.

GROUND-TRUTH ISOLATION: this class imports only from `interfaces/`. It
never imports RFEnvironment, GroundTruthSnapshot, or EmitterSpec, and
never receives them — `select_band(state)` and `update(observation,
reward)` are exactly the existing Scheduler interface, unchanged. `state`
is used only to enumerate bands and check whether a band has ever been
observed (via BeliefSnapshot.bands); this scheduler's actual decision
statistics (S_b, N_b) are private, scheduler-owned, and updated only from
`observation.detected` — never from `reward`, which is accepted (the
interface requires it) but deliberately ignored, and never from anything
ground-truth-derived.
"""

import math

from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State

_EPSILON = 1e-9
"""Numerical-safety floor, not a tunable hyperparameter. A band's
discounted N_b is treated as 'never observed' whenever it is below this
value — true for a genuinely never-observed band (N_b stays exactly 0.0
forever under repeated multiplication by gamma) and, in principle, for a
band decayed so far below 1.0 that using it in the UCB formula would risk
numerical instability rather than a meaningful score."""


class AdaptiveUcbScheduler(Scheduler):
    """Adaptive Discounted-UCB Multi-Armed Bandit Scheduler.

    Constructor parameters:
        num_bands: size of the scan range (explicit, matching the Phase 2
            convention — no shared config object).
        gamma: recency/memory factor in (0, 1]. gamma=1.0 means no
            discounting at all (decaying by exactly 1.0 changes nothing),
            so the formula reduces exactly to ordinary lifetime-average
            UCB1. Smaller gamma means faster forgetting and more weight
            on recent evidence — this is what lets the scheduler respond
            to frequency-hopping emitters better than a plain lifetime
            average (see the class-level docstring and ARCHITECTURE.md).
        exploration_constant: `c` in the score formula. Larger values
            favor exploration (uncertain bands); smaller values favor
            exploitation (the current best estimate).
    """

    def __init__(
        self,
        num_bands: int,
        gamma: float = 1.0,
        exploration_constant: float = 1.0,
    ):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if exploration_constant < 0.0:
            raise ValueError(
                f"exploration_constant must be non-negative, got {exploration_constant}"
            )

        self._num_bands = num_bands
        self._gamma = gamma
        self._c = exploration_constant

        self._successes: list[float] = [0.0] * num_bands
        """S_b per band — discounted count of detections."""
        self._counts: list[float] = [0.0] * num_bands
        """N_b per band — discounted count of observations."""
        self._t = 0
        """Ordinary (undiscounted) decision counter. 0 after reset(); the
        first select_band() call increments it to 1 before deciding."""

    def reset(self) -> None:
        self._successes = [0.0] * self._num_bands
        self._counts = [0.0] * self._num_bands
        self._t = 0

    def select_band(self, state: State) -> Band:
        self._t += 1

        # Initialization: visit every never-observed band first, lowest
        # band_id first. A band is "never observed" here iff N_b < epsilon
        # -- true for a genuinely unvisited band (N_b stays exactly 0.0
        # forever, since 0.0 * gamma == 0.0), and used as the same
        # deliberate floor for numerical safety generally (see module
        # docstring). We check this scheduler's OWN N_b array directly
        # (not `state`) since N_b is this scheduler's private statistic,
        # not something Phase 2's belief tracks. `state` itself is not
        # otherwise needed for this decision, but is still accepted
        # (interface requirement) and would be consulted for band
        # enumeration if this scheduler needed a different band count
        # than its own configured `num_bands` -- it does not.
        for band_id in range(self._num_bands):
            if self._counts[band_id] < _EPSILON:
                return band_id

        # Every band has N_b >= epsilon: evaluate the UCB score for all
        # of them. `self._t >= 1` is guaranteed here (the cold-start case
        # above always fires at t=1, since every N_b starts at 0.0), so
        # ln(self._t) is always well-defined and non-negative.
        log_t = math.log(self._t)

        best_band = 0
        best_score = float("-inf")
        for band_id in range(self._num_bands):
            n_b = self._counts[band_id]
            p_b = self._successes[band_id] / n_b
            score = p_b + self._c * math.sqrt(log_t / n_b)
            # Strict '>' means ties keep the lowest band_id, since we
            # scan in ascending order and only replace on a strictly
            # better score.
            if score > best_score:
                best_score = score
                best_band = band_id

        return best_band

    def update(self, observation: Observation, reward: float) -> None:
        # `reward` is accepted because the Scheduler interface requires
        # it, and deliberately ignored: this scheduler reads
        # observation.detected directly, so its behavior is completely
        # independent of whatever reward_fn a given evaluator is
        # configured with. See ARCHITECTURE.md's Phase 4 section.
        for band_id in range(self._num_bands):
            self._successes[band_id] *= self._gamma
            self._counts[band_id] *= self._gamma

        band = observation.band
        self._counts[band] += 1.0
        if observation.detected:
            self._successes[band] += 1.0
