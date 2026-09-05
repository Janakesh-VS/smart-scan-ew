"""AdaptiveUcbScheduler: Adaptive Discounted-UCB multi-armed bandit scheduler.

Phase 4's first learning-based Scheduler. See ARCHITECTURE.md's "Phase 4"
section for the full rationale and the project owner's locked design
decisions (this docstring summarizes the decisions of record):

- Maintains its OWN per-band discounted statistics (S_b, N_b, an explicit
  observed_b flag) entirely internally. `SimpleBeliefState`/`BeliefSnapshot`
  are NOT modified and are NOT used as this scheduler's data source.
  `select_band(state)` accepts `state` only because the `Scheduler`
  interface requires it; its contents are ignored — the same pattern
  `RoundRobinScheduler` and `RandomScheduler` already use.
- Uses ONE recency parameter, `gamma`, applied to both S_b and N_b on
  every `update()` call, for every band (not just the one observed). No
  separate EWMA alpha exists.
- `update()` reads `observation.detected` directly. The `reward`
  argument exists only because the `Scheduler` interface requires it and
  is intentionally unused here — a real receiver only ever reveals
  hit/miss, never a ground-truth-derived reward (see
  `evaluator/simple_evaluator.py`'s `default_reward` docstring, which
  already documents that its reward is a Phase 3 placeholder, not a
  tuned Phase 4 objective).
- Distinguishes "never observed" from "heavily discounted, N_b near
  zero" via an explicit `observed_b` flag, never by comparing N_b to
  zero — floating-point decay can (and, over a long enough run at
  gamma < 1, will) drive an observed-but-abandoned band's N_b to exactly
  0.0 in IEEE-754 arithmetic, which must not be confused with "never
  observed".
- This module imports ONLY from `interfaces/` and the standard library —
  never `environment/`, `evaluator/`, or any ground-truth-shaped type.
  See `tests/test_phase4_ground_truth_isolation.py`.

Bandit objective vs. evaluator objective (documented limitation, not a
bug): this scheduler maximizes discounted OBSERVED detections
(`observation.detected`), which is not the same objective as the
evaluator's ground-truth interception rate. The receiver never reveals
emitter identity, so the scheduler cannot target individual emitters; a
scenario with one strong emitter and one weak emitter may see it favor
the strong one, and that is expected, not a defect (see
ARCHITECTURE.md's Phase 4 section for the full discussion, echoing the
project owner's original framing).

What discounting does and does not do (be precise about this — no
unsupported claims): decaying S_b and N_b by the same factor `gamma`
every step leaves their ratio `p_b = S_b / N_b` UNCHANGED for a band
that is not itself being updated — discounting alone never moves a
band's point estimate. What it DOES do is (a) make a revisited band's
estimate a genuine recency-weighted average, so new evidence outweighs
old evidence once the band is revisited, and (b) let an unvisited band's
`N_b` shrink over time even without new data about it, which grows that
band's UCB exploration bonus and makes it progressively more attractive
to revisit. It does NOT predict where a `FrequencyHoppingEmitter` will
move to next, and it makes no formal regret-bound or "never permanently
abandoned" guarantee — this is a standard, numerically-guarded
heuristic, not a proof of optimality.
"""

import math
from dataclasses import dataclass

from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State

EPSILON = 1e-12
"""Floor applied to an observed band's discounted count (N_b) before it
is used as a divisor. Prevents ZeroDivisionError / inf / NaN from
floating-point underflow after many decay steps without a revisit (see
`tests/test_adaptive_ucb_scheduler.py`'s numerical-safety tests, which
run this scheduler for tens of thousands of steps specifically to
exercise this path). Never used to distinguish "never observed" from
"observed" — see `observed_b` in the class docstring; that distinction
is made solely via the explicit flag, never via a float comparison to
zero.
"""


@dataclass(frozen=True)
class BandUcbDiagnostics:
    """Read-only, per-band diagnostic snapshot of AdaptiveUcbScheduler's
    internal state — a Phase 5 (dashboard) addition, purely for
    observability. See `AdaptiveUcbScheduler.get_diagnostics()`.

    This is an ADDITIVE API-only change: nothing about `select_band()`,
    `update()`, `reset()`, or the equations they implement was touched to
    add this (see ARCHITECTURE.md's Phase 5 section for the project
    owner's locked decision). Nothing in this class is ever written back
    into the scheduler — nothing reads `BandUcbDiagnostics` except a
    caller displaying it.
    """

    band_id: int
    discounted_successes: float
    """S_b at the moment this snapshot was taken."""
    discounted_observations: float
    """N_b at the moment this snapshot was taken (before any EPSILON
    flooring — the raw, possibly-near-zero discounted count)."""
    observed: bool
    """Whether this band has ever been observed (the scheduler's
    `observed_b` flag) — the same flag `select_band()` itself checks
    before ever computing a UCB score for this band."""
    estimated_hit_rate: float | None
    """S_b / max(N_b, EPSILON), or None if `observed` is False. The real
    algorithm never computes this for an unobserved band (see
    `select_band()`'s unobserved-first branch) — reporting a number here
    for an unobserved band would misrepresent the actual decision rule,
    so it is None instead."""
    exploration_bonus: float | None
    """c * sqrt(ln(t) / max(N_b, EPSILON)), or None if `observed` is
    False, for the same reason as `estimated_hit_rate`."""
    ucb_score: float | None
    """estimated_hit_rate + exploration_bonus, or None if `observed` is
    False. This is exactly the score `select_band()` compares across
    bands once every band has been observed — computed by the same
    `_score_components()` helper, not a second implementation."""


class AdaptiveUcbScheduler(Scheduler):
    """Adaptive Discounted-UCB multi-armed bandit.

    Per band `b`, maintains discounted successes `S_b` and discounted
    observation count `N_b`, decayed by `gamma` on every `update()` call
    (for every band, not just the one observed this step), then
    incremented for whichever band was actually observed. Selection
    rule, applied in `select_band()`:

    1. If any band has never been observed (`observed_b is False`),
       return the lowest-numbered such band, deterministically. This
       guarantees every band gets at least one data point before UCB
       scoring is used at all, and is checked BEFORE any division, log,
       or sqrt is computed — so those never see an `N_b == 0` band.
    2. Otherwise, for every band: `p_b = S_b / max(N_b, EPSILON)`,
       `score(b) = p_b + c * sqrt(ln(t) / max(N_b, EPSILON))`, where `t`
       is the 1-indexed episode-local decision count for the decision
       about to be made (`1 + number of update() calls so far this
       episode`). Return `argmax(score)`, ties broken by lowest
       `band_id` (strict `>` comparison during the scan, so the first
       — lowest-id — band among any tie is kept).

    `t >= 1` always by construction (`self._t` starts at 0 and only
    ever increases), so `ln(t) >= 0` and the sqrt argument is always
    non-negative — there is no invalid-log or invalid-sqrt case,
    independent of the `EPSILON` floor, which exists purely to guard the
    division once `N_b` has decayed toward (or, in raw float terms,
    exactly to) zero for a band that has not been revisited in a long
    time.

    `select_band()` never mutates any internal state on its own — it is
    a pure function of the scheduler's current `S`/`N`/`observed`/`t`.
    Only `update()` advances anything. This means calling `select_band()`
    more than once in a row without an intervening `update()` always
    returns the same band (verified in
    `tests/test_adaptive_ucb_scheduler.py`).

    Phase 5 addition (observability only, no algorithmic change):
    `get_diagnostics()` and the `decision_count` property expose this
    internal state read-only, for the dashboard. See their docstrings.
    """

    def __init__(self, num_bands: int, gamma: float = 1.0, c: float = 1.0):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if c < 0.0:
            raise ValueError(f"c must be non-negative, got {c}")

        self._num_bands = num_bands
        self._gamma = gamma
        self._c = c

        # Populated by _init_stats(); declared here only so attributes
        # exist before the first call for readability.
        self._S: list[float] = []
        self._N: list[float] = []
        self._observed: list[bool] = []
        self._t = 0
        self._init_stats()

    def _init_stats(self) -> None:
        self._S = [0.0] * self._num_bands
        self._N = [0.0] * self._num_bands
        self._observed = [False] * self._num_bands
        self._t = 0  # number of update() calls so far this episode

    def reset(self) -> None:
        # Episode-local statistics reset; constructor hyperparameters
        # (num_bands, gamma, c) persist across reset() — the same
        # pattern RandomScheduler already uses for its own seed (reset()
        # re-derives from a stored value, it does not forget it).
        self._init_stats()

    def select_band(self, state: State) -> Band:
        # `state` is accepted only because the Scheduler interface
        # requires it — this scheduler's learning statistics are
        # maintained entirely internally (see class docstring and
        # ARCHITECTURE.md's Phase 4 section). Intentionally ignored,
        # exactly like RoundRobinScheduler / RandomScheduler.
        for band_id in range(self._num_bands):
            if not self._observed[band_id]:
                return band_id

        decision_index = self._t + 1  # 1-indexed; always >= 1, see docstring
        log_t = math.log(decision_index)  # decision_index >= 1 => log_t >= 0

        best_band = 0
        best_score = float("-inf")
        for band_id in range(self._num_bands):
            _, _, score = self._score_components(band_id, log_t)
            if score > best_score:
                best_score = score
                best_band = band_id
        return best_band

    def _score_components(self, band_id: int, log_t: float) -> tuple[float, float, float]:
        """(p_b, exploration_bonus, score) for one band, using the exact
        formula and EPSILON floor documented in the class docstring.

        SHARED by `select_band()` and `get_diagnostics()` (Phase 5) so
        there is exactly ONE implementation of the UCB formula in this
        class — a dashboard or any other consumer must never compute a
        second one for display purposes; this is why `get_diagnostics()`
        exists at all.
        """
        n_safe = max(self._N[band_id], EPSILON)
        p_b = self._S[band_id] / n_safe
        bonus = self._c * math.sqrt(log_t / n_safe)
        return p_b, bonus, p_b + bonus

    def get_diagnostics(self) -> tuple[BandUcbDiagnostics, ...]:
        """Read-only, per-band snapshot of this scheduler's CURRENT
        internal state, in band_id order — a Phase 5 (dashboard)
        addition, purely observational (see class/module docstrings and
        `BandUcbDiagnostics`).

        Calling this method never mutates anything (same guarantee as
        `select_band()` — see its docstring), and it uses the exact same
        `decision_index`/`log_t`/`_score_components()` `select_band()`
        itself uses, so a call to `get_diagnostics()` made immediately
        before or after a `select_band()` call, with no intervening
        `update()` in between, reports precisely the values that
        decision was (or would be) based on.
        """
        decision_index = self._t + 1
        log_t = math.log(decision_index)

        diagnostics = []
        for band_id in range(self._num_bands):
            observed = self._observed[band_id]
            if observed:
                p_b, bonus, score = self._score_components(band_id, log_t)
            else:
                p_b, bonus, score = None, None, None
            diagnostics.append(
                BandUcbDiagnostics(
                    band_id=band_id,
                    discounted_successes=self._S[band_id],
                    discounted_observations=self._N[band_id],
                    observed=observed,
                    estimated_hit_rate=p_b,
                    exploration_bonus=bonus,
                    ucb_score=score,
                )
            )
        return tuple(diagnostics)

    @property
    def decision_count(self) -> int:
        """Number of `update()` calls so far this episode — the same `t`
        used internally by `select_band()`/`get_diagnostics()` (as
        `decision_index = decision_count + 1`). Read-only: this is a
        property with no setter, added in Phase 5 purely for
        observability."""
        return self._t

    def update(self, observation: Observation, reward: float) -> None:
        # `reward` is accepted because the Scheduler interface requires
        # it, but is intentionally unused — this scheduler learns
        # directly from `observation.detected`, the only receiver-
        # visible signal a real deployment would ever have (see module
        # docstring and PROJECT_CONTRACT.md rule 4).
        band = observation.band
        if not (isinstance(band, int) and 0 <= band < self._num_bands):
            raise ValueError(
                f"Observation references band {band!r}, which is outside "
                f"this scheduler's range of 0..{self._num_bands - 1}"
            )

        for band_id in range(self._num_bands):
            self._S[band_id] *= self._gamma
            self._N[band_id] *= self._gamma

        self._N[band] += 1.0
        self._observed[band] = True
        if observation.detected:
            self._S[band] += 1.0

        self._t += 1
