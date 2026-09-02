# CLAUDE.md — Working Rules for AI-Assisted Development on this Repo

This file is for any AI assistant (or human) picking up work on `smart-scan-ew`.
Read this before writing or changing code. It encodes decisions already made
so they are not silently re-litigated in a later session.

## Project

Smart Scan Strategy for Electronic Warfare (Problem Statement 26055).
A modular Python **simulation** of a receiver that must scan a set of RF
bands under time constraints, and a scheduler (classical, later
learning-based) that decides which band to look at next based only on what
it has observed so far — never on the true RF environment state.

## Current Phase

**Phase 3 — Evaluation framework (complete).** A real `SimpleEvaluator`
(implements the Phase 0 `Evaluator` interface exactly, no changes), plus
additive orchestration (`ExperimentConfig`, `run_experiment_for_scheduler`,
`compare_baselines`, `run_repeated_trials`) now exist under
`src/smart_scan_ew/evaluator/`. Six metrics (Pd, Pfa, interception rate,
average intercept time, intercept time error, average reward/cost) are
computed from real recorded runs, with raw TP/FP/FN/TN counts kept
alongside every derived ratio for auditability. See `ARCHITECTURE.md`'s
"Phase 3" section for the concrete decisions of record. No ML/RL or
dashboard exists yet — see `PROJECT_CONTRACT.md` for the phase plan.

Do not jump ahead of the current phase without the project owner's sign-off.

## Hard Rules (do not violate these while extending the project)

1. **Module independence.** `environment`, `receiver`, `scheduler`, `state`,
   `evaluator`, and (later) `dashboard` are separate packages/modules. A
   module may depend on interfaces, not on another module's internals.
2. **Communicate only through the defined interfaces**
   (see `ARCHITECTURE.md`). If a new interaction is needed, extend an
   interface deliberately and document it — don't reach around it.
3. **No hidden global state.** No module-level mutable singletons, no
   implicit shared state, no monkeypatching. All state is passed explicitly
   (constructor injection or method arguments).
4. **The scheduler must NEVER receive full ground truth.** It only ever sees
   `Observation` objects and whatever `State`/belief is derived from them.
   Do not pass the `RFEnvironment` (or anything with ground-truth access)
   into a `Scheduler` implementation.
5. **Only the `Evaluator` may access ground truth**, via a distinct accessor
   (`RFEnvironment.get_ground_truth()`), and only for scoring — never to
   influence the scheduler or receiver during a run.
6. **Keep simulation, scheduling, evaluation, and UI separate.** No
   cross-imports between `interfaces/environment.py`-implementations and
   `interfaces/scheduler.py`-implementations, etc. The `Evaluator` is the
   only place allowed to wire all of them together.
7. **Experimental parameters are configurable, not hard-coded.** New
   constants (band counts, dwell times, hop rates, thresholds, reward
   shaping, etc.) belong in a config object/dataclass, not inline literals.
8. **Every module gets tests.** A module without at least an
   import/contract-level test is not considered done.
9. **No complex ML yet.** Don't introduce learning-based components,
   RL frameworks, or heavy dependencies until the project owner explicitly
   moves the project into that phase.
10. **Never fabricate performance results.** No invented metrics, plots, or
    numbers. If something hasn't been run, say so; don't simulate an
    outcome to make a placeholder look complete.
11. **Don't touch `README.md` or `.gitignore` unless a change is actually
    needed.** Both currently work fine; leave them alone by default.
12. **Never delete existing files.**
13. **Don't add dependencies you don't immediately need.** Prefer the
    standard library. Add a library only when a concrete module requires it,
    and note why in `requirements.txt`.

## Layout

```
src/smart_scan_ew/
    interfaces/        # abstract contracts — the only things other modules import
    config.py          # configurable experiment parameters
    environment/        # Phase 1: SimpleRFEnvironment, emitter models, band table
    receiver/            # Phase 1: SimpleReceiver
    state/                # Phase 2: SimpleBeliefState, BeliefSnapshot, BandBeliefView
    scheduler/            # Phase 2: RoundRobinScheduler, RandomScheduler, GreedyRecentHitScheduler
    evaluator/            # Phase 3: SimpleEvaluator, ExperimentConfig, compare_baselines, run_repeated_trials
tests/                 # one test module per interface/contract for now
```

Future concrete modules (learning-based scheduler, dashboard) will each
get their own subpackage under `src/smart_scan_ew/`, implementing the
interfaces defined in `src/smart_scan_ew/interfaces/`.

## Before adding a new module

- Check `ARCHITECTURE.md` — does an interface already cover this?
- Check `PROJECT_CONTRACT.md` — is this in scope for the current phase?
- Add/extend tests alongside the code, not after.
- If you must change an interface, update `ARCHITECTURE.md` in the same
  change and explain why in the commit message.
