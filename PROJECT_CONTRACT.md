# PROJECT_CONTRACT.md

## Problem Statement

- **ID:** 26055
- **Title:** Smart Scan Strategy for Electronic Warfare
- **Repository:** `Janakesh-VS/smart-scan-ew`

## Goal

Build a modular Python software **simulation** in which a receiver must
decide, time-step by time-step, which RF band to scan next in an
environment containing simulated emitters, under the constraint that the
scanning strategy (the "scheduler") never has access to the true state of
the environment — only to what its own observations reveal over time.
Classical scheduling strategies and, later, a learning-based scheduler will
be compared against each other using a common evaluation harness.

## In-Scope Modules (eventual)

| Module | Responsibility |
|---|---|
| Simulated RF environment | Owns ground-truth world state (emitters, bands, timing). Advances over time. |
| Emitter models | Defines how individual emitters behave (on/off, frequency, hopping, power) — a detail *inside* the environment. |
| Receiver model | Tunes to a band, observes it, and produces a receiver-visible `Observation`. |
| Observation / hit-miss system | The data contract between receiver and everything downstream. |
| State / belief representation | Derives features/state for the scheduler from observation history only. |
| Classical scanning schedulers | Deterministic/heuristic strategies (e.g. round-robin, priority-based) implementing the `Scheduler` interface. |
| Learning-based scheduler | A later `Scheduler` implementation that learns from observations/rewards. |
| Evaluation and metrics | Runs experiments, is the only component allowed to read ground truth, and scores schedulers. |
| Dashboard | Visualizes runs/metrics. Consumes evaluator output only. |

## Non-Negotiable Architecture Rules

These are the rules the project owner specified. They apply to every phase:

1. Keep each module independent.
2. Modules communicate only through clearly defined interfaces.
3. No hidden global state.
4. The scheduler must never receive complete RF ground truth.
5. The evaluator may access ground truth; the scheduler may not.
6. Keep simulation, scheduling, evaluation, and UI separate.
7. Experimental parameters are configurable, not hard-coded.
8. Every module gets tests.
9. No complex ML until explicitly approved.
10. No fabricated performance results, ever.
11. Don't modify `README.md`/`.gitignore` unnecessarily.
12. Never delete existing files.
13. No unnecessary dependencies.

## Phase Plan

- **Phase 0 — Foundation (this change).** Docs, folder structure,
  interfaces/contracts, placeholder no-op implementations, import/contract
  tests. No simulation logic, no detection logic, no ML/RL, no dashboard.
- **Phase 1 — RF environment & receiver (complete).** Real (but simple)
  emitter models (`ContinuousWaveEmitter`, `PulsedEmitter`,
  `FrequencyHoppingEmitter`), a real `RFEnvironment`, and a real
  `SimpleReceiver` that produces genuine `Observation`s via a seeded
  Gaussian-noise detection model. Ground-truth isolation verified by
  tests. See `ARCHITECTURE.md`'s Phase 1 section for details.
- **Phase 2 — Classical schedulers & belief/state (complete).**
  `SimpleBeliefState` (per-band belief derived strictly from
  `Observation`s) and three baseline schedulers — `RoundRobinScheduler`,
  `RandomScheduler`, `GreedyRecentHitScheduler`. Ground-truth isolation
  verified across the complete environment→receiver→state→scheduler loop.
  See `ARCHITECTURE.md`'s Phase 2 section for details.
- **Phase 3 — Evaluation harness (complete).** `SimpleEvaluator`
  implements the existing `Evaluator` interface unchanged. Additive
  orchestration (`ExperimentConfig`, `run_experiment_for_scheduler`,
  `compare_baselines`, `run_repeated_trials`) runs the three baseline
  schedulers under identical environment/receiver conditions and computes
  Pd, Pfa, interception rate, average intercept time, intercept time
  error, and average reward/cost from real recorded runs — raw TP/FP/FN/TN
  counts are kept alongside every derived ratio. See `ARCHITECTURE.md`'s
  Phase 3 section for details.
- **Phase 4 — Learning-based scheduler (complete).**
  `AdaptiveUcbScheduler`: an Adaptive Discounted-UCB multi-armed bandit
  (not a contextual bandit — no shared feature model across bands),
  maintaining private per-band discounted statistics updated only from
  `Observation.detected`; the `reward` argument is accepted but ignored.
  A small, fixed hyperparameter grid search (`gamma` × `exploration_
  constant`, 12 combinations) selects hyperparameters on disjoint
  selection seeds before held-out evaluation. No neural networks, RL,
  DQN, policy gradients, or reward shaping were introduced. See
  `ARCHITECTURE.md`'s Phase 4 section for the full specification and real
  held-out comparison results against the three Phase 2 baselines.
- **Phase 5 — Dashboard.** Visualizes evaluator output; no direct access to
  environment or scheduler internals.

Each phase should end with something runnable and tested before moving to
the next.

## Explicitly Out of Scope for This Change

- Emitter behavior implementation
- RF simulation logic
- Detection/hit-miss algorithms
- Any ML or RL
- Dashboard implementation
- Any performance experiments or numbers

## Decisions Log

- **Phase 0** provisional decisions (band representation, config format,
  package naming) were reviewed by the owner before Phase 1 began.
- **Phase 1** decisions (emitter types, noise/detection model, `BandSpec`
  fields, `default_scenario()`, ground-truth snapshot fields, RNG
  ownership) were explicitly approved by the owner and are recorded in
  `ARCHITECTURE.md`'s "Phase 1" section.
- **Phase 2** decisions (belief schema, `BeliefSnapshot`/`BandBeliefView`
  as `State`'s output type, the three baseline schedulers' exact decision
  rules and tie-breaks, `num_bands` as an explicit constructor argument
  rather than a shared config object) were explicitly approved by the
  owner and are recorded in `ARCHITECTURE.md`'s "Phase 2" section.
- **Phase 3** decisions (exact Pd/Pfa/interception/timing/reward
  definitions, the seed-derivation reproducibility strategy, co-band
  emitter credit-sharing, never-intercepted-emitter handling, the
  five-file `evaluator/` layout) were explicitly approved by the owner
  and are recorded in `ARCHITECTURE.md`'s "Phase 3" section. No Phase
  0/1/2 interface was modified — confirmed by an empty `git diff` against
  every existing interface/implementation file.
- **Phase 4** decisions (Adaptive Discounted-UCB algorithm, exact
  discounted-statistic/UCB-score equations, ignoring the evaluator's
  `reward` argument in favor of reading `observation.detected` directly,
  the hyperparameter grid, and the honest coverage/reward-maximization
  limitation) were explicitly approved by the owner across three design
  review rounds and are recorded in `ARCHITECTURE.md`'s "Phase 4"
  section. No Phase 0-3 interface, and no other existing implementation
  file, was modified — confirmed by an empty `git diff` against every
  file under `interfaces/`, `environment/`, `receiver/`, `state/`, the
  Phase 2 scheduler files, `evaluator/experiment.py`, and
  `evaluator/simple_evaluator.py`.
- Remaining open items (config file format) are listed in
  `ARCHITECTURE.md` under "What is deliberately still NOT decided" and
  will be raised again when the relevant phase begins.
