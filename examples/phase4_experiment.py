"""Phase 4 experiment: hyperparameter selection, held-out evaluation, and
a four-way scheduler comparison for AdaptiveUcbScheduler.

This script performs no new evaluation logic of its own — it is pure
orchestration on top of the existing Phase 3 evaluator package
(`ExperimentConfig`, `run_repeated_trials`, `run_experiment_for_scheduler`).
No `evaluator/` file is imported for anything it doesn't already export.

Three distinct concepts, kept explicitly separate (do not conflate them):

A. ONLINE LEARNING — happens automatically, inside a single experiment
   run, every time `AdaptiveUcbScheduler.update()` is called by the
   Phase 3 evaluator loop. Not something this script drives directly.

B. HYPERPARAMETER SELECTION (`run_hyperparameter_selection` +
   `select_best_hyperparameters`) — a grid search over
   `GAMMA_GRID x C_GRID`, using ONLY `SELECTION_SEEDS`. This is
   conventional grid search / model selection, not "training" in the
   neural-network sense — nothing here does gradient descent or fits
   parameters to data; it evaluates a small number of fixed
   configurations and picks one by a documented rule.

C. FINAL HELD-OUT EVALUATION (`run_held_out_evaluation`) — runs the
   frozen (gamma, c), plus the three Phase 2 baselines, using ONLY
   `HELD_OUT_SEEDS`, which is disjoint from `SELECTION_SEEDS` by
   construction (see tests/test_adaptive_ucb_scheduler.py's
   `test_selection_and_held_out_seeds_are_disjoint`). No hyperparameter
   is ever selected using data it is later scored on.

Run with: python examples/phase4_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smart_scan_ew.evaluator import (
    ExperimentConfig,
    MetricStats,
    TrialSummary,
    run_repeated_trials,
)
from smart_scan_ew.scheduler import (
    AdaptiveUcbScheduler,
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)

# --- Selection vs. held-out seed separation --------------------------------
#
# SELECTION_SEEDS are used ONLY by run_hyperparameter_selection().
# HELD_OUT_SEEDS are used ONLY by run_held_out_evaluation(), after gamma/c
# are frozen. Disjoint by construction; checked by a dedicated test.
SELECTION_SEEDS: tuple[int, ...] = tuple(range(1000, 1010))  # 10 seeds
HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(2000, 2010))  # 10 seeds

GAMMA_GRID: tuple[float, ...] = (0.90, 0.95, 0.99, 1.00)
C_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)

NEAR_TIE_MARGIN = 0.02
"""Absolute interception-rate margin defining a "near tie" (locked
decision, PROJECT_CONTRACT.md Phase 4 section): any configuration within
this margin of the best mean `interception_rate_active_emitters` is
treated as tied on the primary criterion; among those, the configuration
with the lowest mean `average_intercept_time` is preferred. No weighted
composite score is used."""

DEFAULT_CONFIG = ExperimentConfig(num_bands=5, num_steps=200)

_TRIAL_METRIC_NAMES = (
    "probability_of_detection",
    "probability_of_false_alarm",
    "interception_rate_all_emitters",
    "interception_rate_active_emitters",
    "average_intercept_time",
    "average_intercept_time_error",
    "average_reward",
)
"""The metrics this script reports for the final held-out comparison —
matches the seven metrics run_repeated_trials()/TrialSummary already
compute (Pd, Pfa, both interception-rate variants, timing, reward), each
with mean/stdev/min/max/n_defined."""


def _adaptive_ucb_factory(gamma: float, c: float, num_bands: int):
    """Build a scheduler_factory(seed) for run_repeated_trials(), fixing
    gamma/c. AdaptiveUcbScheduler owns no RNG (fully deterministic given
    its observation sequence — see its module docstring), so `seed` is
    accepted only for signature compatibility with
    run_repeated_trials's Callable[[int | None], Scheduler] contract and
    is otherwise unused, exactly as GreedyRecentHitScheduler's `seed`-
    ignoring factories already do elsewhere in this script.
    """

    def factory(seed: int | None) -> AdaptiveUcbScheduler:
        del seed
        return AdaptiveUcbScheduler(num_bands=num_bands, gamma=gamma, c=c)

    return factory


def run_hyperparameter_selection(
    config: ExperimentConfig = DEFAULT_CONFIG,
    selection_seeds: tuple[int, ...] = SELECTION_SEEDS,
    gamma_grid: tuple[float, ...] = GAMMA_GRID,
    c_grid: tuple[float, ...] = C_GRID,
) -> dict[tuple[float, float], TrialSummary]:
    """Stage B: run every (gamma, c) grid point across `selection_seeds`
    ONLY, via the existing `run_repeated_trials()` — no new evaluator
    logic. Returns one TrialSummary per (gamma, c)."""
    results: dict[tuple[float, float], TrialSummary] = {}
    for gamma in gamma_grid:
        for c in c_grid:
            summary = run_repeated_trials(
                config=config,
                scheduler_factory=_adaptive_ucb_factory(gamma, c, config.num_bands),
                master_seeds=list(selection_seeds),
                scheduler_name=f"adaptive_ucb(gamma={gamma}, c={c})",
            )
            results[(gamma, c)] = summary
    return results


def select_best_hyperparameters(
    grid_results: dict[tuple[float, float], TrialSummary],
    near_tie_margin: float = NEAR_TIE_MARGIN,
) -> tuple[float, float]:
    """Two-stage hyperparameter selection rule (locked decision):

    1. Primary criterion: maximize mean `interception_rate_active_emitters`.
    2. Near-tie rule: among configurations within `near_tie_margin`
       (absolute) of the best mean interception rate, choose the one
       with the lowest mean `average_intercept_time`.

    A configuration with an undefined (`None`) mean
    `interception_rate_active_emitters` or `average_intercept_time` (no
    trial ever intercepted anything / no trial ever intercepted anything
    with a computable time) is treated as strictly worse on that
    criterion than any configuration with a defined value — "never
    intercepted anything" cannot win a timing comparison.

    Deliberately NOT a weighted composite score, per the locked decision.
    """

    def interception_mean(summary: TrialSummary) -> float:
        mean = summary.interception_rate_active_emitters.mean
        return mean if mean is not None else float("-inf")

    best_interception = max(interception_mean(s) for s in grid_results.values())

    near_tied = {
        config: summary
        for config, summary in grid_results.items()
        if best_interception - interception_mean(summary) <= near_tie_margin
    }

    def intercept_time_mean(summary: TrialSummary) -> float:
        mean = summary.average_intercept_time.mean
        return mean if mean is not None else float("inf")

    best_config = min(near_tied, key=lambda cfg: intercept_time_mean(near_tied[cfg]))
    return best_config


def run_held_out_evaluation(
    gamma: float,
    c: float,
    config: ExperimentConfig = DEFAULT_CONFIG,
    held_out_seeds: tuple[int, ...] = HELD_OUT_SEEDS,
) -> dict[str, TrialSummary]:
    """Stage C/D: four-way comparison — Round Robin, Random, Greedy Recent
    Hit, and the frozen Adaptive UCB — using ONLY `held_out_seeds`.

    Each scheduler is run via `run_repeated_trials()` across the SAME
    list of master seeds, so for a given seed the environment/receiver
    trajectories are byte-identical across all four schedulers (seeds 0
    and 1 of `derive_seeds()` depend only on the master seed, never on
    which scheduler is under test — see `evaluator/reproducibility.py`).
    This is the same fairness property `compare_baselines()` provides
    for a single run, extended here across repeated trials and a fourth
    scheduler without modifying `evaluator/comparison.py` itself.
    """
    scheduler_factories = {
        "round_robin": lambda seed: RoundRobinScheduler(num_bands=config.num_bands),
        "random": lambda seed: RandomScheduler(num_bands=config.num_bands, seed=seed),
        "greedy_recent_hit": lambda seed: GreedyRecentHitScheduler(
            num_bands=config.num_bands
        ),
        "adaptive_ucb": _adaptive_ucb_factory(gamma, c, config.num_bands),
    }

    return {
        name: run_repeated_trials(
            config=config,
            scheduler_factory=factory,
            master_seeds=list(held_out_seeds),
            scheduler_name=name,
        )
        for name, factory in scheduler_factories.items()
    }


def _print_trial_summary(name: str, summary: TrialSummary) -> None:
    print(f"\n--- {name} ---")
    for metric_name in _TRIAL_METRIC_NAMES:
        stats: MetricStats = getattr(summary, metric_name)
        print(
            f"  {metric_name:38s} mean={stats.mean} stdev={stats.stdev} "
            f"min={stats.minimum} max={stats.maximum} "
            f"n_defined={stats.n_defined}/{stats.n_total}"
        )


def main() -> None:
    print(
        f"[Stage B] Hyperparameter selection: {len(SELECTION_SEEDS)} "
        f"selection seeds x {len(GAMMA_GRID) * len(C_GRID)} configs "
        f"(gamma in {GAMMA_GRID}, c in {C_GRID})"
    )
    grid_results = run_hyperparameter_selection()
    for (gamma, c), summary in grid_results.items():
        mean_ir = summary.interception_rate_active_emitters.mean
        mean_t = summary.average_intercept_time.mean
        print(
            f"  gamma={gamma:<5} c={c:<4} "
            f"mean interception_rate_active_emitters={mean_ir} "
            f"mean average_intercept_time={mean_t}"
        )

    gamma, c = select_best_hyperparameters(grid_results)
    print(f"\n[Frozen] gamma={gamma}, c={c}")

    print(
        f"\n[Stage C/D] Held-out evaluation: {len(HELD_OUT_SEEDS)} seeds "
        f"never used during selection, four-way comparison"
    )
    held_out_results = run_held_out_evaluation(gamma, c)
    for name, summary in held_out_results.items():
        _print_trial_summary(name, summary)


if __name__ == "__main__":
    main()
