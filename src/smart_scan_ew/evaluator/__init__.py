"""Phase 3 evaluation framework.

The only package allowed to see both ground truth and a Scheduler/State
in the same place (CLAUDE.md rule 6). `SimpleEvaluator` implements the
abstract `Evaluator` interface exactly as declared in Phase 0; everything
else here (`ExperimentConfig`, `run_experiment_for_scheduler`,
`compare_baselines`, `run_repeated_trials`) is additive orchestration, not
part of any interface.
"""

from smart_scan_ew.evaluator.comparison import (
    ComparisonResult,
    MetricStats,
    TrialSummary,
    compare_baselines,
    run_repeated_trials,
)
from smart_scan_ew.evaluator.experiment import ExperimentConfig, run_experiment_for_scheduler
from smart_scan_ew.evaluator.hyperparameter_selection import (
    DEFAULT_EXPLORATION_CONSTANT_GRID,
    DEFAULT_GAMMA_GRID,
    HyperparameterCandidate,
    HyperparameterSelectionResult,
    select_ucb_hyperparameters,
)
from smart_scan_ew.evaluator.records import (
    EmitterInterceptionRecord,
    ExperimentResult,
    RunRecord,
    StepRecord,
)
from smart_scan_ew.evaluator.simple_evaluator import SimpleEvaluator, default_reward

__all__ = [
    "SimpleEvaluator",
    "default_reward",
    "StepRecord",
    "RunRecord",
    "EmitterInterceptionRecord",
    "ExperimentResult",
    "ExperimentConfig",
    "run_experiment_for_scheduler",
    "ComparisonResult",
    "compare_baselines",
    "MetricStats",
    "TrialSummary",
    "run_repeated_trials",
    "HyperparameterCandidate",
    "HyperparameterSelectionResult",
    "select_ucb_hyperparameters",
    "DEFAULT_GAMMA_GRID",
    "DEFAULT_EXPLORATION_CONSTANT_GRID",
]
