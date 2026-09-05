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

**Phase 5 — Dashboard (complete).** `src/smart_scan_ew/dashboard/`
(`controller.py` + `app.py`, Streamlit) is a pure consumer of the
existing Phase 1-4 components and Phase 3 evaluator APIs — no
simulation, scheduler, or evaluator logic was reimplemented inside it.
`controller.py` has zero `streamlit` import (all non-UI logic, fully
unit-testable); `app.py` is the UI only. The one change to Phase 4 code
is additive and non-algorithmic: `AdaptiveUcbScheduler` gained
`get_diagnostics()` (returns `BandUcbDiagnostics`, one per band) and a
read-only `decision_count` property, both computed via a shared
`_score_components()` helper `select_band()` itself now also uses — the
UCB equations, update order, initialization, exploration, tie-breaking,
and reset semantics are byte-for-byte unchanged (verified: all 138
pre-existing tests still pass after the refactor, plus new tests
checking the accessors are read-only and match internal state exactly).
The dashboard does not import `examples/phase4_experiment.py`; its
four-way comparison is built directly on `run_repeated_trials`,
`ExperimentConfig`, and `derive_seeds`. Ground-truth isolation is
re-verified for the dashboard's own step loop (structural + runtime
spy). See `ARCHITECTURE.md`'s "Phase 4" and "Phase 5" sections for full
details, including Phase 5's documented Streamlit session-state and
auto-run limitations.

There is no Phase 6 yet.

Do not jump ahead of the current phase without the project owner's sign-off.

## Hard Rules (do not violate these while extending the project)

1. **Module independence.** `environment`, `receiver`, `scheduler`, `state`,
   `evaluator`, and `dashboard` are separate packages/modules. A
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
                          # Phase 4: AdaptiveUcbScheduler (adaptive_ucb.py)
    evaluator/            # Phase 3: SimpleEvaluator, ExperimentConfig, compare_baselines, run_repeated_trials
    dashboard/            # Phase 5: controller.py (no UI import) + app.py (Streamlit)
examples/
    phase4_experiment.py  # Phase 4: hyperparameter selection + held-out evaluation + 4-way comparison
tests/                 # one test module per interface/contract for now
```

Streamlit constraints (Phase 5, see `dashboard/app.py`'s module
docstring for the full explanation): the `DashboardController` instance
lives in `st.session_state`, never a module-level global; the simulation
is reset only when the resolved configuration actually changes or
"Reset" is pressed, never on an unrelated rerun; "Start"/"Pause" drive a
bounded auto-advance-then-`st.rerun()` loop, which is best-effort, not a
guaranteed-smooth animation — "Step Once" is the reliable alternative
for a live demo.

No further concrete modules are planned. Any new one still gets its own
subpackage under `src/smart_scan_ew/`, implementing the interfaces
defined in `src/smart_scan_ew/interfaces/`.

## Before adding a new module

- Check `ARCHITECTURE.md` — does an interface already cover this?
- Check `PROJECT_CONTRACT.md` — is this in scope for the current phase?
- Add/extend tests alongside the code, not after.
- If you must change an interface, update `ARCHITECTURE.md` in the same
  change and explain why in the commit message.
