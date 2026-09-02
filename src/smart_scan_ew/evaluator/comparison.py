"""Baseline comparison and repeated-trials aggregation.

`compare_baselines` runs the three Phase 2 baseline schedulers against
the *same* environment/receiver seeds (derived once from one master
seed), so any difference in their results comes only from the scheduling
policy — not from incidental randomness differences. No ML scheduler is
included; the result is keyed by name so adding one later is additive.

`run_repeated_trials` is deliberately simple and composable: it runs ONE
scheduler across multiple seeds and reports mean/stdev/min/max per
metric. Comparing multiple schedulers across multiple seeds is done by
calling this once per scheduler and collecting the TrialSummary objects
yourself — no combined N-scheduler x M-seed structure, per the project
owner's explicit Phase 3 scope decision.
"""

import statistics
from dataclasses import dataclass
from typing import Callable

from smart_scan_ew.evaluator.experiment import ExperimentConfig, run_experiment_for_scheduler
from smart_scan_ew.evaluator.records import ExperimentResult
from smart_scan_ew.evaluator.reproducibility import derive_seeds
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.scheduler import (
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)

# Fixed, documented metric names aggregated by run_repeated_trials — kept
# to the six Phase 3 metrics the project owner asked to be able to compare
# across seeds. Extend this tuple, not ad hoc, if more are needed later.
_TRIAL_METRIC_NAMES = (
    "probability_of_detection",
    "probability_of_false_alarm",
    "interception_rate_all_emitters",
    "interception_rate_active_emitters",
    "average_intercept_time",
    "average_intercept_time_error",
    "average_reward",
)


@dataclass(frozen=True)
class ComparisonResult:
    """Result of running all three baseline schedulers against the same
    scenario and the same master seed (hence the same environment/receiver
    trajectories — see reproducibility.py)."""

    master_seed: int | None
    config: ExperimentConfig
    results: dict[str, ExperimentResult]
    """Keyed by scheduler label: 'round_robin', 'random', 'greedy_recent_hit'."""


def compare_baselines(config: ExperimentConfig, master_seed: int | None) -> ComparisonResult:
    """Run RoundRobinScheduler, RandomScheduler, and GreedyRecentHitScheduler
    against the same scenario, with the same environment/receiver seeds.

    `RandomScheduler`'s own seed is derived from the same `master_seed`
    (role index 2, per reproducibility.py's documented ordering) so a full
    comparison run is entirely reproducible from `master_seed` alone.
    """
    _, _, scheduler_seed = derive_seeds(master_seed, 3)

    schedulers: dict[str, Scheduler] = {
        "round_robin": RoundRobinScheduler(num_bands=config.num_bands),
        "random": RandomScheduler(num_bands=config.num_bands, seed=scheduler_seed),
        "greedy_recent_hit": GreedyRecentHitScheduler(num_bands=config.num_bands),
    }

    results = {
        name: run_experiment_for_scheduler(
            config, scheduler, master_seed=master_seed, scheduler_name=name
        )
        for name, scheduler in schedulers.items()
    }

    return ComparisonResult(master_seed=master_seed, config=config, results=results)


@dataclass(frozen=True)
class MetricStats:
    """Simple summary statistics for one metric across repeated trials.
    Deliberately stdlib-only (mean/stdev/min/max) — no confidence
    intervals or bootstrapping in Phase 3."""

    mean: float | None
    stdev: float | None
    """None if fewer than 2 trials had a defined (non-None) value."""
    minimum: float | None
    maximum: float | None
    n_total: int
    """Total number of trials attempted."""
    n_defined: int
    """How many of those trials had a non-None value for this metric —
    e.g. a scenario where some seeds produced zero interceptions will have
    n_defined < n_total for average_intercept_time."""

    @classmethod
    def from_values(cls, values: list[float], n_total: int) -> "MetricStats":
        n_defined = len(values)
        if n_defined == 0:
            return cls(None, None, None, None, n_total, 0)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if n_defined >= 2 else None
        return cls(mean, stdev, min(values), max(values), n_total, n_defined)


@dataclass(frozen=True)
class TrialSummary:
    """Mean/stdev/min/max, per metric, across repeated trials of ONE
    scheduler under otherwise-identical experiment configuration."""

    scheduler_name: str
    per_trial_results: tuple[ExperimentResult, ...]
    probability_of_detection: MetricStats
    probability_of_false_alarm: MetricStats
    interception_rate_all_emitters: MetricStats
    interception_rate_active_emitters: MetricStats
    average_intercept_time: MetricStats
    average_intercept_time_error: MetricStats
    average_reward: MetricStats

    @classmethod
    def from_results(
        cls, scheduler_name: str, results: tuple[ExperimentResult, ...]
    ) -> "TrialSummary":
        n_total = len(results)
        stats = {
            metric_name: MetricStats.from_values(
                [
                    value
                    for r in results
                    if (value := getattr(r, metric_name)) is not None
                ],
                n_total,
            )
            for metric_name in _TRIAL_METRIC_NAMES
        }
        return cls(
            scheduler_name=scheduler_name,
            per_trial_results=results,
            **stats,
        )


def run_repeated_trials(
    config: ExperimentConfig,
    scheduler_factory: Callable[[int | None], Scheduler],
    master_seeds: list[int],
    scheduler_name: str,
) -> TrialSummary:
    """Run one scheduler across multiple master seeds and summarize.

    `scheduler_factory(scheduler_seed)` must construct a fresh Scheduler
    instance for each trial — a fresh instance per seed, unlike
    `run_experiment_for_scheduler`'s single-run reset-not-reconstruct rule
    (each trial here is a genuinely independent experiment, not successive
    episodes of one persistent scheduler). `scheduler_seed` is derived
    from the trial's own master seed (role index 2) so a stochastic
    scheduler's randomness is also reproducible per trial.
    """
    results = []
    for seed in master_seeds:
        _, _, scheduler_seed = derive_seeds(seed, 3)
        scheduler = scheduler_factory(scheduler_seed)
        result = run_experiment_for_scheduler(
            config, scheduler, master_seed=seed, scheduler_name=scheduler_name
        )
        results.append(result)

    return TrialSummary.from_results(scheduler_name, tuple(results))
