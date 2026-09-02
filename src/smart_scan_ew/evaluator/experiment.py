"""Experiment configuration and single-scheduler orchestration.

This is where the experimental protocol lives: scenario, seed, band
count, simulation duration, and the reset behavior for each component.
None of this is part of the Evaluator interface — `run_experiment` (the
interface method) deliberately does no resetting of its own (matching the
Phase 0 fixture convention), so this module is the caller that does it,
using the reproducibility strategy in `reproducibility.py`.
"""

from dataclasses import dataclass

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment, default_scenario
from smart_scan_ew.evaluator.records import ExperimentResult
from smart_scan_ew.evaluator.reproducibility import derive_seeds
from smart_scan_ew.evaluator.simple_evaluator import SimpleEvaluator
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.state import SimpleBeliefState


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to run one experiment, except the scheduler under
    test and the master seed (both supplied separately — see
    `run_experiment_for_scheduler` and `compare_baselines`).

    "Number of emitters" is deliberately not a separate field — it is
    `len(emitter_specs)` (or `len(default_scenario(num_bands))` when
    `emitter_specs` is None), so it can never disagree with the actual
    scenario. "Simulation duration" is `num_steps * dt`, computed, not
    stored.
    """

    num_bands: int
    num_steps: int
    dt: float = 1.0
    emitter_specs: list[EmitterSpec] | None = None
    band_start_frequency_hz: float = 2.4e9
    band_width_hz: float = 20e6
    noise_std: float = 1.0
    detection_threshold: float = 3.0

    def resolved_emitter_specs(self) -> list[EmitterSpec]:
        if self.emitter_specs is not None:
            return self.emitter_specs
        return default_scenario(self.num_bands)


def run_experiment_for_scheduler(
    config: ExperimentConfig,
    scheduler: Scheduler,
    master_seed: int | None,
    scheduler_name: str,
) -> ExperimentResult:
    """Run one full experiment for one (already-constructed) scheduler.

    Reset behavior (the approved Phase 3 protocol):
    - `environment` is freshly constructed from `config` and
      `reset(seed=env_seed)`.
    - `receiver` is freshly constructed from `config` with `receiver_seed`
      and then `reset()` (re-applies that same seed — see Phase 1 notes).
    - `state` is freshly constructed with `config.num_bands` and `reset()`.
    - `scheduler` is NEVER reconstructed here — only `reset()` is called
      on the instance the caller passed in, so a future learned
      scheduler's persistent parameters survive across episodes while its
      episode-local counters are cleared.

    `env_seed`/`receiver_seed` are derived once from `master_seed` via
    `derive_seeds` — see that module for why this makes comparisons
    across different schedulers fair.
    """
    env_seed, receiver_seed = derive_seeds(master_seed, 2)

    environment = SimpleRFEnvironment(
        emitter_specs=config.resolved_emitter_specs(),
        num_bands=config.num_bands,
        band_start_frequency_hz=config.band_start_frequency_hz,
        band_width_hz=config.band_width_hz,
    )
    environment.reset(seed=env_seed)

    receiver = SimpleReceiver(
        detection_threshold=config.detection_threshold,
        noise_std=config.noise_std,
        seed=receiver_seed,
    )
    receiver.reset()

    state = SimpleBeliefState(num_bands=config.num_bands)
    state.reset()

    scheduler.reset()

    evaluator = SimpleEvaluator(dt=config.dt)
    run_record = evaluator.run_experiment(
        environment=environment,
        receiver=receiver,
        scheduler=scheduler,
        state=state,
        num_steps=config.num_steps,
    )
    metrics = evaluator.compute_metrics(run_record)

    return ExperimentResult(
        scheduler_name=scheduler_name,
        num_steps=config.num_steps,
        dt=config.dt,
        num_bands=config.num_bands,
        env_seed=env_seed,
        receiver_seed=receiver_seed,
        **metrics,
    )
