"""Phase 5 dashboard controller: the ONLY non-UI code the dashboard uses.

Wraps the real, UNMODIFIED-in-behavior Phase 1-4 components
(`SimpleRFEnvironment`, `SimpleReceiver`, `SimpleBeliefState`, and the
four `Scheduler` implementations) and the real Phase 3 evaluator APIs
(`ExperimentConfig`, `run_experiment_for_scheduler`, `run_repeated_trials`,
`derive_seeds`) behind a small, UI-framework-agnostic API that
`dashboard/app.py` calls. Contains zero `streamlit` imports and no
simulation logic of its own — see ARCHITECTURE.md's Phase 5 section for
the project owner's locked decisions this module implements:

- Decision 1: `AdaptiveUcbScheduler`'s new, additive, read-only
  `get_diagnostics()`/`decision_count` accessors are the ONLY source of
  Adaptive UCB display data. This module never reaches into `_`-prefixed
  attributes and never recomputes the UCB formula.
- Decision 2: this module does NOT import `examples/phase4_experiment.py`
  (an executable script, not application code). Four-way comparison is
  built directly on `run_repeated_trials`, `ExperimentConfig`, and
  `derive_seeds` — the same reusable Phase 3 evaluator APIs that script
  also uses, with no evaluator code duplicated or modified.

GROUND-TRUTH ISOLATION (unchanged from Phase 1-4, re-verified for this
module's own code path — see
`tests/test_dashboard_ground_truth_isolation.py`): `DashboardController.step()`
never calls `get_ground_truth()` and never passes ground truth to
`state.update()`, `scheduler.select_band()`, or `scheduler.update()`.
`peek_ground_truth()` exists ONLY for the dashboard's clearly-labelled
debug/ground-truth visualization, is never called from `step()`, and its
return value is never fed back into `step()`, `state`, or `scheduler`.
"""

from dataclasses import dataclass
from typing import Callable, Literal

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment, default_scenario
from smart_scan_ew.environment.rf_environment import GroundTruthSnapshot
from smart_scan_ew.evaluator import (
    ExperimentConfig,
    ExperimentResult,
    TrialSummary,
    run_experiment_for_scheduler,
    run_repeated_trials,
)
from smart_scan_ew.evaluator.reproducibility import derive_seeds
from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import (
    AdaptiveUcbScheduler,
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from smart_scan_ew.scheduler.adaptive_ucb import BandUcbDiagnostics
from smart_scan_ew.state import BeliefSnapshot, SimpleBeliefState

SchedulerName = Literal["round_robin", "random", "greedy_recent_hit", "adaptive_ucb"]

SCHEDULER_NAMES: tuple[SchedulerName, ...] = (
    "round_robin",
    "random",
    "greedy_recent_hit",
    "adaptive_ucb",
)

SCHEDULER_DISPLAY_NAMES: dict[SchedulerName, str] = {
    "round_robin": "Round Robin",
    "random": "Random",
    "greedy_recent_hit": "Greedy Recent Hit",
    "adaptive_ucb": "Adaptive UCB",
}


def _build_scheduler(
    name: SchedulerName,
    num_bands: int,
    seed: int | None,
    gamma: float,
    c: float,
) -> Scheduler:
    """Construct one of the four REAL scheduler classes. No behavior of
    any scheduler is reimplemented here — this is a plain constructor
    dispatch."""
    if name == "round_robin":
        return RoundRobinScheduler(num_bands=num_bands)
    if name == "random":
        return RandomScheduler(num_bands=num_bands, seed=seed)
    if name == "greedy_recent_hit":
        return GreedyRecentHitScheduler(num_bands=num_bands)
    if name == "adaptive_ucb":
        return AdaptiveUcbScheduler(num_bands=num_bands, gamma=gamma, c=c)
    raise ValueError(f"Unknown scheduler name: {name!r}")


# --- Scenario presets --------------------------------------------------
#
# These are compositions of the EXISTING EmitterSpec/kind values ("cw",
# "pulsed", "hopping") already implemented in environment/emitters.py —
# no new emitter model is introduced (Part 14). "Default" reuses the
# existing default_scenario() unchanged.


def _cw_only_scenario(num_bands: int) -> list[EmitterSpec]:
    return [EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=5.0)]


def _pulsed_only_scenario(num_bands: int) -> list[EmitterSpec]:
    band_id = min(1, num_bands - 1)
    return [
        EmitterSpec(
            emitter_id="pulsed-1", kind="pulsed", band_id=band_id, power=5.0,
            period=10.0, pulse_width=3.0,
        )
    ]


def _hopping_only_scenario(num_bands: int) -> list[EmitterSpec]:
    return [
        EmitterSpec(
            emitter_id="hopper-1", kind="hopping", band_id=0, power=5.0,
            hop_interval=5.0, hop_bands=tuple(range(num_bands)),
        )
    ]


SCENARIO_PRESETS: dict[str, Callable[[int], list[EmitterSpec]]] = {
    "Default (CW + hopping + pulsed)": default_scenario,
    "CW only": _cw_only_scenario,
    "Pulsed only": _pulsed_only_scenario,
    "Frequency hopping only": _hopping_only_scenario,
}
"""Display name -> (num_bands) -> list[EmitterSpec]. "Default" requires
num_bands >= 3 (see default_scenario()'s own validation); the other
presets work for num_bands >= 1."""


@dataclass(frozen=True)
class DecisionReason:
    """Presentation-safe explanation of why AdaptiveUcbScheduler picked
    `band` for one step. Every value is read directly from
    `AdaptiveUcbScheduler.get_diagnostics()`, called immediately BEFORE
    `update()` was applied for this step — i.e. the exact internal state
    `select_band()` itself used to make the decision. Nothing here is
    recomputed independently of the scheduler."""

    band: int
    was_unobserved_band: bool
    """True if the unobserved-band-first rule is what selected `band`
    (see AdaptiveUcbScheduler.select_band()'s first branch) — in that
    case the score fields below are None, exactly mirroring the real
    algorithm never computing a score for an unobserved band."""
    estimated_hit_rate: float | None
    exploration_bonus: float | None
    ucb_score: float | None


@dataclass(frozen=True)
class StepResult:
    """Everything the UI needs to render one simulation step, and
    nothing more. Ground truth is deliberately NOT a field here — see
    `DashboardController.peek_ground_truth()` for the separate,
    explicitly-labelled debug path."""

    time: float
    band: int
    observation: Observation
    belief: BeliefSnapshot
    ucb_diagnostics: tuple[BandUcbDiagnostics, ...] | None
    """None unless the active scheduler is AdaptiveUcbScheduler."""
    decision_reason: DecisionReason | None
    """None unless the active scheduler is AdaptiveUcbScheduler."""


def _explain_decision(
    band: int, diagnostics_before: tuple[BandUcbDiagnostics, ...]
) -> DecisionReason:
    band_diag = diagnostics_before[band]
    return DecisionReason(
        band=band,
        was_unobserved_band=not band_diag.observed,
        estimated_hit_rate=band_diag.estimated_hit_rate,
        exploration_bonus=band_diag.exploration_bonus,
        ucb_score=band_diag.ucb_score,
    )


class DashboardController:
    """Owns one live simulation session: one environment, one receiver,
    one belief state, and one scheduler — all real Phase 1-4 classes.
    This class adds no simulation logic of its own, only sequencing and
    read-only bookkeeping for the UI (history, current time).
    """

    def __init__(self) -> None:
        self._scheduler_name: SchedulerName | None = None
        self._environment: SimpleRFEnvironment | None = None
        self._receiver: SimpleReceiver | None = None
        self._state: SimpleBeliefState | None = None
        self._scheduler: Scheduler | None = None
        self._num_bands = 0
        self._dt = 1.0
        self._time = 0.0
        self._history: list[StepResult] = []

    @property
    def is_configured(self) -> bool:
        return self._scheduler is not None

    @property
    def scheduler_name(self) -> SchedulerName | None:
        return self._scheduler_name

    @property
    def scheduler(self) -> Scheduler | None:
        """The actual scheduler instance in use — read-only access for
        the UI/tests (e.g. to assert
        `isinstance(controller.scheduler, AdaptiveUcbScheduler)`)."""
        return self._scheduler

    @property
    def num_bands(self) -> int:
        return self._num_bands

    @property
    def current_time(self) -> float:
        return self._time

    @property
    def history(self) -> tuple[StepResult, ...]:
        return tuple(self._history)

    def reset(
        self,
        scheduler_name: SchedulerName,
        num_bands: int = 5,
        seed: int | None = 0,
        dt: float = 1.0,
        noise_std: float = 1.0,
        detection_threshold: float = 3.0,
        band_start_frequency_hz: float = 2.4e9,
        band_width_hz: float = 20e6,
        emitter_specs: list[EmitterSpec] | None = None,
        gamma: float = 0.95,
        c: float = 1.0,
    ) -> None:
        """(Re)build the whole session from scratch: a fresh environment,
        receiver, belief state, and a FRESH scheduler instance. Reset
        always means "start over" — even re-selecting the same scheduler
        type gets a brand-new instance, so stale learned state can never
        leak across a Reset.

        `env_seed`/`receiver_seed`/`scheduler_seed` are derived from one
        `seed` via the same `derive_seeds()` role ordering (0=env,
        1=receiver, 2=scheduler) the Phase 3 evaluator uses, so a live
        session built with a given `seed` uses exactly the same
        environment/receiver trajectory a later
        `run_experiment_for_scheduler(..., master_seed=seed, ...)` replay
        would (see `replay_metrics()` below).
        """
        env_seed, receiver_seed, scheduler_seed = derive_seeds(seed, 3)

        specs = emitter_specs if emitter_specs is not None else default_scenario(num_bands)

        self._scheduler_name = scheduler_name
        self._num_bands = num_bands
        self._dt = dt
        self._time = 0.0
        self._history = []
        self._last_config = dict(
            scheduler_name=scheduler_name,
            num_bands=num_bands,
            seed=seed,
            dt=dt,
            noise_std=noise_std,
            detection_threshold=detection_threshold,
            band_start_frequency_hz=band_start_frequency_hz,
            band_width_hz=band_width_hz,
            emitter_specs=specs,
            gamma=gamma,
            c=c,
        )

        self._environment = SimpleRFEnvironment(
            emitter_specs=specs,
            num_bands=num_bands,
            band_start_frequency_hz=band_start_frequency_hz,
            band_width_hz=band_width_hz,
        )
        self._environment.reset(seed=env_seed)

        self._receiver = SimpleReceiver(
            detection_threshold=detection_threshold,
            noise_std=noise_std,
            seed=receiver_seed,
        )
        self._receiver.reset()

        self._state = SimpleBeliefState(num_bands=num_bands)
        self._state.reset()

        self._scheduler = _build_scheduler(
            scheduler_name, num_bands, seed=scheduler_seed, gamma=gamma, c=c
        )
        self._scheduler.reset()

    def step(self) -> StepResult:
        """Advance the simulation by exactly one decision:

            environment.step() -> scheduler.select_band(state) ->
            receiver.tune()/observe() -> state.update() -> scheduler.update()

        — the identical call sequence and identical `t` computation
        `SimpleEvaluator.run_experiment()` uses (see
        `evaluator/simple_evaluator.py`), so a live session and a
        `run_experiment_for_scheduler()` replay with the same seed and
        the same number of steps produce the same trajectory.

        Ground truth is never touched anywhere in this method.
        """
        if not self.is_configured:
            raise RuntimeError("DashboardController.reset() must be called before step()")

        self._environment.step(dt=self._dt)
        self._time += self._dt

        ucb_diagnostics_before = None
        if isinstance(self._scheduler, AdaptiveUcbScheduler):
            ucb_diagnostics_before = self._scheduler.get_diagnostics()

        band = self._scheduler.select_band(self._state)

        self._receiver.tune(band)
        observation = self._receiver.observe(self._environment, t=self._time)

        self._state.update(observation)
        reward = 1.0 if observation.detected else 0.0
        self._scheduler.update(observation, reward)

        ucb_diagnostics_after = None
        decision_reason = None
        if isinstance(self._scheduler, AdaptiveUcbScheduler):
            ucb_diagnostics_after = self._scheduler.get_diagnostics()
            decision_reason = _explain_decision(band, ucb_diagnostics_before)

        result = StepResult(
            time=self._time,
            band=band,
            observation=observation,
            belief=self._state.get_features(),
            ucb_diagnostics=ucb_diagnostics_after,
            decision_reason=decision_reason,
        )
        self._history.append(result)
        return result

    def peek_ground_truth(self) -> GroundTruthSnapshot:
        """DEBUG/EVALUATOR-ONLY view — calls
        `RFEnvironment.get_ground_truth()` for DISPLAY purposes ONLY.

        The caller (app.py) must never pass this return value into
        `step()`, `self._state`, or `self._scheduler` — this method does
        not do so itself, and is never called from within `step()` (see
        `tests/test_dashboard_ground_truth_isolation.py`).
        """
        if not self.is_configured:
            raise RuntimeError(
                "DashboardController.reset() must be called before peek_ground_truth()"
            )
        return self._environment.get_ground_truth()

    def belief_snapshot(self) -> BeliefSnapshot:
        if not self.is_configured:
            raise RuntimeError("DashboardController.reset() must be called first")
        return self._state.get_features()

    def ucb_diagnostics(self) -> tuple[BandUcbDiagnostics, ...] | None:
        """Current Adaptive UCB diagnostics, or None if the active
        scheduler isn't AdaptiveUcbScheduler. Delegates entirely to
        `AdaptiveUcbScheduler.get_diagnostics()` — no recomputation."""
        if isinstance(self._scheduler, AdaptiveUcbScheduler):
            return self._scheduler.get_diagnostics()
        return None

    def replay_metrics(self, num_steps: int | None = None) -> ExperimentResult:
        """Compute REAL Phase 3 evaluator metrics for a fresh experiment
        using the CURRENT session's exact configuration (same seed,
        scenario, scheduler type, and hyperparameters), via
        `run_experiment_for_scheduler()` — the actual evaluator, not a
        second metrics implementation.

        `num_steps` defaults to the number of steps taken so far in this
        live session; because environment/receiver/scheduler
        construction and the per-step call sequence are identical to
        `step()` (see its docstring), this reproduces the exact live
        trajectory when `num_steps == len(self.history)`, using a freshly
        constructed scheduler instance (never the live, already-advanced
        one — `run_experiment_for_scheduler` never reuses a scheduler
        across runs).
        """
        if not self.is_configured:
            raise RuntimeError("DashboardController.reset() must be called first")
        if num_steps is None:
            num_steps = len(self._history)

        cfg = self._last_config
        config = ExperimentConfig(
            num_bands=cfg["num_bands"],
            num_steps=num_steps,
            dt=cfg["dt"],
            emitter_specs=cfg["emitter_specs"],
            band_start_frequency_hz=cfg["band_start_frequency_hz"],
            band_width_hz=cfg["band_width_hz"],
            noise_std=cfg["noise_std"],
            detection_threshold=cfg["detection_threshold"],
        )
        fresh_scheduler = _build_scheduler(
            cfg["scheduler_name"],
            cfg["num_bands"],
            seed=derive_seeds(cfg["seed"], 3)[2],
            gamma=cfg["gamma"],
            c=cfg["c"],
        )
        return run_experiment_for_scheduler(
            config,
            fresh_scheduler,
            master_seed=cfg["seed"],
            scheduler_name=cfg["scheduler_name"],
        )


def run_four_way_comparison(
    num_bands: int,
    num_steps: int,
    master_seeds: list[int],
    gamma: float,
    c: float,
    emitter_specs: list[EmitterSpec] | None = None,
    dt: float = 1.0,
    noise_std: float = 1.0,
    detection_threshold: float = 3.0,
) -> dict[SchedulerName, TrialSummary]:
    """Run all four schedulers via the REAL `run_repeated_trials()` across
    the SAME list of master seeds and the SAME `ExperimentConfig`, so
    seed-for-seed environment/receiver trajectories match across all four
    (`derive_seeds()`'s env/receiver sub-seeds never depend on which
    scheduler is under test — see `evaluator/reproducibility.py`).

    Pure orchestration: no evaluator code is duplicated or modified, and
    this function does not import `examples/phase4_experiment.py`
    (Decision 2) even though it follows the same reusable-API pattern
    that script also uses.
    """
    config = ExperimentConfig(
        num_bands=num_bands,
        num_steps=num_steps,
        dt=dt,
        emitter_specs=emitter_specs,
        noise_std=noise_std,
        detection_threshold=detection_threshold,
    )

    factories: dict[SchedulerName, Callable[[int | None], Scheduler]] = {
        "round_robin": lambda seed: RoundRobinScheduler(num_bands=num_bands),
        "random": lambda seed: RandomScheduler(num_bands=num_bands, seed=seed),
        "greedy_recent_hit": lambda seed: GreedyRecentHitScheduler(num_bands=num_bands),
        "adaptive_ucb": lambda seed: AdaptiveUcbScheduler(num_bands=num_bands, gamma=gamma, c=c),
    }

    return {
        name: run_repeated_trials(
            config=config,
            scheduler_factory=factory,
            master_seeds=list(master_seeds),
            scheduler_name=name,
        )
        for name, factory in factories.items()
    }
