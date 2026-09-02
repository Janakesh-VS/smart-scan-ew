"""Phase 3 tests for compare_baselines() and run_repeated_trials()."""

import pytest

from smart_scan_ew.evaluator import (
    ExperimentConfig,
    MetricStats,
    TrialSummary,
    compare_baselines,
    run_repeated_trials,
)
from smart_scan_ew.scheduler import RandomScheduler


def test_compare_baselines_runs_exactly_the_three_approved_schedulers():
    config = ExperimentConfig(num_bands=5, num_steps=20)
    comparison = compare_baselines(config, master_seed=1)

    assert set(comparison.results.keys()) == {"round_robin", "random", "greedy_recent_hit"}


def test_compare_baselines_results_are_internally_consistent():
    config = ExperimentConfig(num_bands=5, num_steps=20)
    comparison = compare_baselines(config, master_seed=1)

    for name, result in comparison.results.items():
        assert result.scheduler_name == name
        assert result.num_steps == 20
        assert result.total_observations == 20
        assert (
            result.true_positive_count
            + result.false_positive_count
            + result.false_negative_count
            + result.true_negative_count
            == result.total_observations
        )
        # Manually recompute Pd/Pfa from the raw counts -- this is the
        # auditability property: nothing here should be opaque.
        tp, fn = result.true_positive_count, result.false_negative_count
        fp, tn = result.false_positive_count, result.true_negative_count
        expected_pd = tp / (tp + fn) if (tp + fn) > 0 else None
        expected_pfa = fp / (fp + tn) if (fp + tn) > 0 else None
        assert result.probability_of_detection == expected_pd
        assert result.probability_of_false_alarm == expected_pfa


def test_compare_baselines_produces_different_scheduler_names_per_result():
    config = ExperimentConfig(num_bands=5, num_steps=20)
    comparison = compare_baselines(config, master_seed=1)
    for name, result in comparison.results.items():
        assert result.scheduler_name == name


def test_compare_baselines_is_reproducible_given_same_master_seed():
    config = ExperimentConfig(num_bands=5, num_steps=20)
    comparison_a = compare_baselines(config, master_seed=5)
    comparison_b = compare_baselines(config, master_seed=5)

    for name in comparison_a.results:
        a = comparison_a.results[name]
        b = comparison_b.results[name]
        assert a.true_positive_count == b.true_positive_count
        assert a.probability_of_detection == b.probability_of_detection
        assert a.interception_rate_active_emitters == b.interception_rate_active_emitters


def test_run_repeated_trials_reports_correct_mean_and_stdev():
    config = ExperimentConfig(num_bands=5, num_steps=15)
    summary = run_repeated_trials(
        config,
        scheduler_factory=lambda seed: RandomScheduler(num_bands=5, seed=seed),
        master_seeds=[1, 2, 3, 4, 5],
        scheduler_name="random",
    )

    assert isinstance(summary, TrialSummary)
    assert summary.scheduler_name == "random"
    assert len(summary.per_trial_results) == 5

    # Hand-check against Python's own statistics module for one metric.
    import statistics as stats_module

    pd_values = [
        r.probability_of_detection
        for r in summary.per_trial_results
        if r.probability_of_detection is not None
    ]
    stat = summary.probability_of_detection
    assert stat.n_total == 5
    assert stat.n_defined == len(pd_values)
    if pd_values:
        assert stat.mean == pytest.approx(stats_module.mean(pd_values))
        assert stat.minimum == min(pd_values)
        assert stat.maximum == max(pd_values)
        if len(pd_values) >= 2:
            assert stat.stdev == pytest.approx(stats_module.stdev(pd_values))


def test_metric_stats_stdev_is_none_for_a_single_trial():
    config = ExperimentConfig(num_bands=5, num_steps=15)
    summary = run_repeated_trials(
        config,
        scheduler_factory=lambda seed: RandomScheduler(num_bands=5, seed=seed),
        master_seeds=[1],
        scheduler_name="random",
    )
    assert summary.average_reward.n_total == 1
    assert summary.average_reward.stdev is None


def test_metric_stats_handles_all_none_values():
    stats = MetricStats.from_values([], n_total=3)
    assert stats.mean is None
    assert stats.stdev is None
    assert stats.minimum is None
    assert stats.maximum is None
    assert stats.n_total == 3
    assert stats.n_defined == 0
