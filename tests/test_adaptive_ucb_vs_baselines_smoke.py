"""Sanity/regression smoke test for AdaptiveUcbScheduler compared against
the three Phase 2 baselines. Explicitly NOT a superiority claim -- just
confirms all four schedulers produce internally-consistent ExperimentResult
fields under the same conditions, using Phase 3's existing, unmodified
run_experiment_for_scheduler().
"""

from smart_scan_ew.evaluator import ExperimentConfig, run_experiment_for_scheduler
from smart_scan_ew.scheduler import (
    AdaptiveUcbScheduler,
    GreedyRecentHitScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)


def _make_schedulers(num_bands: int):
    return {
        "round_robin": RoundRobinScheduler(num_bands=num_bands),
        "random": RandomScheduler(num_bands=num_bands, seed=0),
        "greedy_recent_hit": GreedyRecentHitScheduler(num_bands=num_bands),
        "adaptive_ucb": AdaptiveUcbScheduler(num_bands=num_bands, gamma=0.95, exploration_constant=1.0),
    }


def test_all_four_schedulers_run_under_identical_conditions():
    config = ExperimentConfig(num_bands=5, num_steps=100)
    master_seed = 123

    results = {
        name: run_experiment_for_scheduler(config, scheduler, master_seed=master_seed, scheduler_name=name)
        for name, scheduler in _make_schedulers(5).items()
    }

    env_seeds = {r.env_seed for r in results.values()}
    receiver_seeds = {r.receiver_seed for r in results.values()}
    assert len(env_seeds) == 1, "all four schedulers must share the same env_seed"
    assert len(receiver_seeds) == 1, "all four schedulers must share the same receiver_seed"

    for name, result in results.items():
        assert result.scheduler_name == name
        assert result.total_observations == 100
        assert (
            result.true_positive_count
            + result.false_positive_count
            + result.false_negative_count
            + result.true_negative_count
            == result.total_observations
        )
        # Auditability: Pd/Pfa recomputable by hand from raw counts.
        tp, fn = result.true_positive_count, result.false_negative_count
        fp, tn = result.false_positive_count, result.true_negative_count
        expected_pd = tp / (tp + fn) if (tp + fn) > 0 else None
        expected_pfa = fp / (fp + tn) if (fp + tn) > 0 else None
        assert result.probability_of_detection == expected_pd
        assert result.probability_of_false_alarm == expected_pfa


def test_adaptive_ucb_result_is_reproducible():
    config = ExperimentConfig(num_bands=5, num_steps=50)
    r1 = run_experiment_for_scheduler(
        config,
        AdaptiveUcbScheduler(num_bands=5, gamma=0.95, exploration_constant=1.0),
        master_seed=7,
        scheduler_name="adaptive_ucb",
    )
    r2 = run_experiment_for_scheduler(
        config,
        AdaptiveUcbScheduler(num_bands=5, gamma=0.95, exploration_constant=1.0),
        master_seed=7,
        scheduler_name="adaptive_ucb",
    )
    assert r1.true_positive_count == r2.true_positive_count
    assert r1.false_positive_count == r2.false_positive_count
    assert r1.probability_of_detection == r2.probability_of_detection
    assert r1.interception_rate_active_emitters == r2.interception_rate_active_emitters
