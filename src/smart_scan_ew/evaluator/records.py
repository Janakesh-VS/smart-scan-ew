"""Data shapes for Phase 3's evaluation framework.

`StepRecord` and `RunRecord` are Evaluator-internal — they hold the full
`GroundTruthSnapshot` for every step, which is exactly the information
that must never reach a `Receiver`, `State`, or `Scheduler` (CLAUDE.md
rule 4/5). Only `SimpleEvaluator` constructs these; nothing outside the
`evaluator/` package should need to.

`EmitterInterceptionRecord` and `ExperimentResult` are the audit-friendly,
public output: raw counts alongside every derived metric, so a reviewer
can recompute Pd/Pfa by hand from the stored counts (CLAUDE.md rule 10 —
no opaque, unauditable numbers).
"""

from dataclasses import dataclass

from smart_scan_ew.environment.rf_environment import GroundTruthSnapshot


@dataclass(frozen=True)
class StepRecord:
    """One simulation step, as seen by the Evaluator: the receiver-visible
    outcome (band/detected/info/reward) paired with the full ground-truth
    snapshot at that instant, captured for scoring only.
    """

    step_index: int
    time: float
    band: int
    detected: bool
    info: dict
    reward: float
    ground_truth: GroundTruthSnapshot


@dataclass(frozen=True)
class RunRecord:
    """The raw record of one full experiment run — what `run_experiment`
    returns. `SimpleEvaluator.compute_metrics()` derives everything else
    from this; nothing here is discarded before metrics are computed, so
    every derived number can be traced back to the steps that produced it.
    """

    steps: tuple[StepRecord, ...]

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class EmitterInterceptionRecord:
    """Per-emitter audit record. Always present for every emitter that
    appeared anywhere in the run's ground truth, whether or not it was
    ever intercepted — never-intercepted emitters are NOT dropped."""

    emitter_id: str
    first_active_time: float | None
    """Earliest time (from ground truth, independent of receiver tuning)
    this emitter was active at all. None if it was never active during
    the run's duration."""
    intercepted: bool
    intercept_time: float | None
    """Time of the first true-positive observation that credited this
    emitter. None if never intercepted."""
    intercept_time_error: float | None
    """intercept_time - first_active_time (detection delay from the
    earliest theoretical opportunity). None if never intercepted."""


@dataclass(frozen=True)
class ExperimentResult:
    """Structured, auditable result of one experiment run for one
    scheduler. Raw counts are always included alongside derived ratios."""

    # --- provenance / audit ---
    scheduler_name: str
    num_steps: int
    dt: float
    num_bands: int
    env_seed: int | None
    receiver_seed: int | None

    # --- raw counts ---
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    total_observations: int
    """Sanity-check field: should always equal num_steps."""

    # --- Pd / Pfa ---
    probability_of_detection: float | None
    """TP / (TP + FN). None if TP + FN == 0 (the receiver never once
    landed on an occupied band this run)."""
    probability_of_false_alarm: float | None
    """FP / (FP + TN). None if FP + TN == 0."""

    # --- interception ---
    total_emitters: int
    active_emitters: int
    """Number of emitters that were active at least once during the run."""
    intercepted_emitter_count: int
    interception_rate_all_emitters: float | None
    """intercepted / total_emitters. Auditability variant — denominator
    includes emitters that could never have been found (e.g. a duty
    cycle longer than the run)."""
    interception_rate_active_emitters: float | None
    """intercepted / active_emitters. Primary/headline metric for
    presentation and cross-scheduler comparison."""
    emitter_records: tuple[EmitterInterceptionRecord, ...]

    # --- timing ---
    average_intercept_time: float | None
    """Mean over INTERCEPTED emitters only. None if none were intercepted."""
    average_intercept_time_error: float | None
    """Mean over INTERCEPTED emitters only. None if none were intercepted."""

    # --- reward / cost ---
    average_reward: float | None
    """Mean of the Phase 3 placeholder reward across all steps. NOT a
    substitute for Pd/Pfa — it mixes true and false detections together.
    None only if num_steps == 0 (nothing to average)."""
    average_cost: float | None
    """1 - average_reward. None under the same condition as average_reward."""
