"""SimpleEvaluator: the one place ground truth and Scheduler/State/Receiver
meet, for scoring only.

Implements the existing Evaluator interface exactly as declared —
`run_experiment(environment, receiver, scheduler, state, num_steps) -> Any`
and `compute_metrics(run_result) -> dict` — with no signature changes.
`dt` and the reward function are concrete-class constructor parameters,
not interface additions (the abstract Evaluator does not constrain
`__init__`).

GROUND-TRUTH ISOLATION (CLAUDE.md rule 4/5, non-negotiable): this class
calls `environment.get_ground_truth()` once per step, for its own
internal `StepRecord`, and for nothing else. That snapshot — and the
`reward` value derived from it never being ground-truth-derived — never
reaches `scheduler.select_band()`, `scheduler.update()`, or
`state.update()`. Those three calls only ever receive the plain `state`
object and a plain `Observation` (with `info={"measured_power": ...}`,
exactly as produced by `SimpleReceiver` in Phase 1). See
`tests/test_evaluator_ground_truth_isolation.py`.
"""

from typing import Callable

from smart_scan_ew.evaluator.records import (
    EmitterInterceptionRecord,
    RunRecord,
    StepRecord,
)
from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.interfaces.evaluator import Evaluator
from smart_scan_ew.interfaces.observation import Observation
from smart_scan_ew.interfaces.receiver import Receiver
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State


def default_reward(observation: Observation) -> float:
    """Phase 3 placeholder reward: 1.0 if detected, else 0.0.

    Deliberately derived ONLY from the receiver-visible Observation, never
    from ground truth. This is explicitly a placeholder for driving
    Scheduler.update() during Phase 3 — it is NOT a tuned optimization
    objective and must not be presented as one (real reward-shaping is a
    Phase 4 concern once an RL formulation is chosen).
    """
    return 1.0 if observation.detected else 0.0


class SimpleEvaluator(Evaluator):
    """Runs a fixed number of steps of the environment/receiver/state/
    scheduler loop, recording enough raw data (Observation + full ground
    truth per step) that every metric in `compute_metrics` can be audited
    back to the steps that produced it.

    `run_experiment` does NOT reset anything — matching the existing
    Phase 0 fixture convention, resetting is the caller's responsibility
    (see `experiment.run_experiment_for_scheduler` for the orchestration
    layer that does this).
    """

    def __init__(
        self,
        dt: float = 1.0,
        reward_fn: Callable[[Observation], float] = default_reward,
    ):
        self._dt = dt
        self._reward_fn = reward_fn

    def run_experiment(
        self,
        environment: RFEnvironment,
        receiver: Receiver,
        scheduler: Scheduler,
        state: State,
        num_steps: int,
    ) -> RunRecord:
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}")

        steps: list[StepRecord] = []

        for step_index in range(num_steps):
            environment.step(dt=self._dt)

            band = scheduler.select_band(state)
            receiver.tune(band)
            t = (step_index + 1) * self._dt
            observation = receiver.observe(environment, t=t)

            # The ONLY ground-truth access in the whole loop — for this
            # step's audit record only. Never passed to state/scheduler.
            ground_truth = environment.get_ground_truth()

            reward = self._reward_fn(observation)

            # These two calls are the entire information surface exposed
            # to the belief/decision side. Neither `ground_truth` nor
            # anything derived from it appears in either call.
            state.update(observation)
            scheduler.update(observation, reward)

            steps.append(
                StepRecord(
                    step_index=step_index,
                    time=observation.time,
                    band=observation.band,
                    detected=observation.detected,
                    info=observation.info,
                    reward=reward,
                    ground_truth=ground_truth,
                )
            )

        return RunRecord(steps=tuple(steps))

    def compute_metrics(self, run_result: RunRecord) -> dict:
        steps = run_result.steps

        tp = fp = fn = tn = 0
        for step in steps:
            signal_present = any(
                e.active and e.band == step.band for e in step.ground_truth.emitters
            )
            if signal_present and step.detected:
                tp += 1
            elif signal_present and not step.detected:
                fn += 1
            elif not signal_present and step.detected:
                fp += 1
            else:
                tn += 1

        pd = tp / (tp + fn) if (tp + fn) > 0 else None
        pfa = fp / (fp + tn) if (fp + tn) > 0 else None

        emitter_records, intercepted_count, active_count, total_count = (
            self._compute_emitter_records(steps)
        )

        intercepted_times = [
            r.intercept_time for r in emitter_records if r.intercepted
        ]
        intercepted_errors = [
            r.intercept_time_error for r in emitter_records if r.intercepted
        ]
        average_intercept_time = (
            sum(intercepted_times) / len(intercepted_times)
            if intercepted_times
            else None
        )
        average_intercept_time_error = (
            sum(intercepted_errors) / len(intercepted_errors)
            if intercepted_errors
            else None
        )

        interception_rate_all = (
            intercepted_count / total_count if total_count > 0 else None
        )
        interception_rate_active = (
            intercepted_count / active_count if active_count > 0 else None
        )

        if steps:
            average_reward = sum(s.reward for s in steps) / len(steps)
            average_cost = 1.0 - average_reward
        else:
            average_reward = None
            average_cost = None

        return {
            "true_positive_count": tp,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "true_negative_count": tn,
            "total_observations": len(steps),
            "probability_of_detection": pd,
            "probability_of_false_alarm": pfa,
            "total_emitters": total_count,
            "active_emitters": active_count,
            "intercepted_emitter_count": intercepted_count,
            "interception_rate_all_emitters": interception_rate_all,
            "interception_rate_active_emitters": interception_rate_active,
            "emitter_records": emitter_records,
            "average_intercept_time": average_intercept_time,
            "average_intercept_time_error": average_intercept_time_error,
            "average_reward": average_reward,
            "average_cost": average_cost,
        }

    @staticmethod
    def _compute_emitter_records(
        steps: tuple[StepRecord, ...],
    ) -> tuple[tuple[EmitterInterceptionRecord, ...], int, int, int]:
        """Build one EmitterInterceptionRecord per emitter_id seen anywhere
        in this run's ground truth, in first-seen order."""
        emitter_ids: list[str] = []
        seen = set()
        for step in steps:
            for e in step.ground_truth.emitters:
                if e.emitter_id not in seen:
                    seen.add(e.emitter_id)
                    emitter_ids.append(e.emitter_id)

        records = []
        intercepted_count = 0
        active_count = 0

        for emitter_id in emitter_ids:
            first_active_time = None
            intercept_time = None

            for step in steps:
                emitter_state = next(
                    (e for e in step.ground_truth.emitters if e.emitter_id == emitter_id),
                    None,
                )
                if emitter_state is not None and emitter_state.active:
                    if first_active_time is None:
                        first_active_time = step.time

                if (
                    intercept_time is None
                    and step.detected
                    and emitter_state is not None
                    and emitter_state.active
                    and emitter_state.band == step.band
                ):
                    intercept_time = step.time

            intercepted = intercept_time is not None
            if intercepted:
                intercepted_count += 1
            if first_active_time is not None:
                active_count += 1

            intercept_time_error = (
                intercept_time - first_active_time
                if intercepted and first_active_time is not None
                else None
            )

            records.append(
                EmitterInterceptionRecord(
                    emitter_id=emitter_id,
                    first_active_time=first_active_time,
                    intercepted=intercepted,
                    intercept_time=intercept_time,
                    intercept_time_error=intercept_time_error,
                )
            )

        return tuple(records), intercepted_count, active_count, len(emitter_ids)
