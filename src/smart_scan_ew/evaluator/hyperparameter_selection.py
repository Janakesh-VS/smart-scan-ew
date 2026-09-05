"""Phase 4 hyperparameter selection for AdaptiveUcbScheduler.

This is NOT a training loop in any neural-network sense — it is a small,
fixed grid search over two scalars (`gamma`, `exploration_constant`),
each candidate evaluated via Phase 3's existing, unmodified
`run_repeated_trials()`. See ARCHITECTURE.md's Phase 4 section for the
online-learning / hyperparameter-selection / held-out-evaluation
distinction this module implements the middle step of.

Selection seeds and final held-out evaluation seeds must be disjoint —
this module only ever consumes the seeds it's given (the caller is
responsible for keeping the two sets separate; see
tests/test_hyperparameter_selection.py and the Phase 4 held-out
comparison script for how that separation is maintained in practice).
"""

from dataclasses import dataclass

from smart_scan_ew.evaluator.comparison import TrialSummary, run_repeated_trials
from smart_scan_ew.evaluator.experiment import ExperimentConfig
from smart_scan_ew.scheduler.adaptive_ucb_scheduler import AdaptiveUcbScheduler

DEFAULT_GAMMA_GRID: tuple[float, ...] = (0.90, 0.95, 0.99, 1.00)
"""gamma=1.00 is included deliberately: it corresponds to no discounting
at all (ordinary lifetime-average UCB1), so whether discounting helps on
a given scenario is itself something the grid search discovers, not
assumed."""

DEFAULT_EXPLORATION_CONSTANT_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class HyperparameterCandidate:
    gamma: float
    exploration_constant: float


@dataclass(frozen=True)
class HyperparameterSelectionResult:
    """Result of the grid search: every candidate's TrialSummary (for
    full auditability — nothing hidden), plus which one was selected and
    why (the metric it was ranked on)."""

    candidates: tuple[tuple[HyperparameterCandidate, TrialSummary], ...]
    ranking_metric: str
    best_candidate: HyperparameterCandidate
    best_trial_summary: TrialSummary


def select_ucb_hyperparameters(
    config: ExperimentConfig,
    selection_seeds: list[int],
    gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID,
    exploration_constant_grid: tuple[float, ...] = DEFAULT_EXPLORATION_CONSTANT_GRID,
    ranking_metric: str = "interception_rate_active_emitters",
) -> HyperparameterSelectionResult:
    """Evaluate every (gamma, exploration_constant) combination on
    `selection_seeds` and return the one with the best mean
    `ranking_metric` (default: interception_rate_active_emitters, the
    Phase 3 "headline" metric).

    A candidate whose mean is `None` for the ranking metric (e.g. zero
    interceptions across every selection seed) is never chosen over one
    with a defined mean, but is still included in the returned
    `candidates` tuple for auditability.
    """
    results: list[tuple[HyperparameterCandidate, TrialSummary]] = []

    for gamma in gamma_grid:
        for c in exploration_constant_grid:
            candidate = HyperparameterCandidate(gamma=gamma, exploration_constant=c)
            summary = run_repeated_trials(
                config,
                scheduler_factory=lambda _seed, g=gamma, cc=c: AdaptiveUcbScheduler(
                    num_bands=config.num_bands, gamma=g, exploration_constant=cc
                ),
                master_seeds=selection_seeds,
                scheduler_name=f"adaptive_ucb(gamma={gamma},c={c})",
            )
            results.append((candidate, summary))

    def sort_key(item: tuple[HyperparameterCandidate, TrialSummary]) -> float:
        _, summary = item
        mean = getattr(summary, ranking_metric).mean
        # None (undefined metric) ranks last, never selected over a
        # candidate with a real value.
        return mean if mean is not None else float("-inf")

    best_candidate, best_summary = max(results, key=sort_key)

    return HyperparameterSelectionResult(
        candidates=tuple(results),
        ranking_metric=ranking_metric,
        best_candidate=best_candidate,
        best_trial_summary=best_summary,
    )
