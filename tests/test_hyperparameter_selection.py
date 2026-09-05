"""Phase 4 tests for select_ucb_hyperparameters()."""

from smart_scan_ew.evaluator import ExperimentConfig
from smart_scan_ew.evaluator.hyperparameter_selection import (
    DEFAULT_EXPLORATION_CONSTANT_GRID,
    DEFAULT_GAMMA_GRID,
    HyperparameterCandidate,
    HyperparameterSelectionResult,
    select_ucb_hyperparameters,
)


def test_default_grids_match_the_approved_specification():
    assert DEFAULT_GAMMA_GRID == (0.90, 0.95, 0.99, 1.00)
    assert DEFAULT_EXPLORATION_CONSTANT_GRID == (0.5, 1.0, 2.0)


def test_select_ucb_hyperparameters_evaluates_every_combination():
    config = ExperimentConfig(num_bands=4, num_steps=15)
    gamma_grid = (0.9, 1.0)
    c_grid = (0.5, 1.0)

    result = select_ucb_hyperparameters(
        config,
        selection_seeds=[1, 2, 3],
        gamma_grid=gamma_grid,
        exploration_constant_grid=c_grid,
    )

    assert isinstance(result, HyperparameterSelectionResult)
    assert len(result.candidates) == len(gamma_grid) * len(c_grid)

    expected_pairs = {(g, c) for g in gamma_grid for c in c_grid}
    actual_pairs = {(cand.gamma, cand.exploration_constant) for cand, _ in result.candidates}
    assert actual_pairs == expected_pairs


def test_select_ucb_hyperparameters_picks_the_best_by_ranking_metric():
    config = ExperimentConfig(num_bands=4, num_steps=15)
    result = select_ucb_hyperparameters(
        config,
        selection_seeds=[1, 2, 3],
        gamma_grid=(0.9, 1.0),
        exploration_constant_grid=(0.5, 1.0),
        ranking_metric="interception_rate_active_emitters",
    )

    assert isinstance(result.best_candidate, HyperparameterCandidate)
    best_mean = getattr(result.best_trial_summary, result.ranking_metric).mean

    for candidate, summary in result.candidates:
        mean = getattr(summary, result.ranking_metric).mean
        if mean is not None and best_mean is not None:
            assert mean <= best_mean + 1e-12  # best is truly the max


def test_select_ucb_hyperparameters_is_deterministic_given_same_seeds():
    config = ExperimentConfig(num_bands=4, num_steps=15)
    result_a = select_ucb_hyperparameters(
        config, selection_seeds=[10, 20], gamma_grid=(0.9, 1.0), exploration_constant_grid=(1.0,)
    )
    result_b = select_ucb_hyperparameters(
        config, selection_seeds=[10, 20], gamma_grid=(0.9, 1.0), exploration_constant_grid=(1.0,)
    )
    assert result_a.best_candidate == result_b.best_candidate


def test_select_ucb_hyperparameters_all_candidates_have_full_trial_summaries():
    config = ExperimentConfig(num_bands=4, num_steps=15)
    result = select_ucb_hyperparameters(
        config,
        selection_seeds=[1, 2],
        gamma_grid=(1.0,),
        exploration_constant_grid=(0.5, 1.0, 2.0),
    )
    for candidate, summary in result.candidates:
        assert len(summary.per_trial_results) == 2
        assert summary.scheduler_name == f"adaptive_ucb(gamma={candidate.gamma},c={candidate.exploration_constant})"
