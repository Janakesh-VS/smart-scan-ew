# ARCHITECTURE.md

This document defines the conceptual interfaces for the project and how
data flows between them. It is intentionally implementation-light: methods
are specified by name, purpose, and rough signature so that later phases can
fill in real logic without changing the contracts other modules rely on.

All interfaces live in `src/smart_scan_ew/interfaces/` as abstract base
classes. Nothing outside `interfaces/` should be imported across module
boundaries — a `Scheduler` implementation imports `interfaces.scheduler`
and `interfaces.observation`/`interfaces.state`, never a concrete
`RFEnvironment` class.

## Data flow (per time step, conceptually)

```
                ┌───────────────────────────────────────────┐
                │              RFEnvironment                 │
                │  (ground-truth world: emitters, bands,     │
                │   time). Advanced by step().                │
                └───────────────┬────────────────┬───────────┘
                                 │                │
                     sense(band)│                │get_ground_truth()
                                 │                │ (Evaluator ONLY)
                                 ▼                ▼
                          ┌────────────┐   ┌─────────────┐
                          │  Receiver   │   │  Evaluator  │
                          │ tune/observe│   │  (scoring)  │
                          └─────┬───────┘   └─────▲───────┘
                                │ Observation      │ metrics
                                ▼                  │
                          ┌────────────┐           │
                          │   State /   │           │
                          │   Belief    │           │
                          └─────┬───────┘           │
                                │ features/state     │
                                ▼                    │
                          ┌────────────┐             │
                          │  Scheduler  │─────────────┘
                          │select_band, │   (reward, actions logged
                          │  update     │    for evaluation)
                          └────────────┘
```

Key point: the arrow from `RFEnvironment` to `Scheduler` **does not exist**.
The scheduler only ever receives a `State` (built purely from
`Observation`s) and, via `update()`, an `Observation` and a scalar reward.
Only the `Evaluator` is allowed to call `get_ground_truth()`.

## Interfaces

### `Band`

A placeholder type alias (currently `int`) identifying which band is
selected. Kept abstract on purpose — Phase 1 can turn this into a richer
type (e.g. a frequency range) without changing any other interface, since
everything else treats it as an opaque, comparable/hashable value.

### `Observation` (`interfaces/observation.py`)

Immutable data object — the only thing that crosses from the sensing side
(`Receiver`) to the belief/decision side (`State`, `Scheduler`).

Fields:
- `time: float` — simulation time (or step index) of the observation.
- `band: Band` — which band was observed.
- `detected: bool` — hit/miss result.
- `info: dict` — optional additional receiver-visible information (e.g. a
  signal strength estimate), deliberately open-ended and empty by default
  so Phase 1 can add fields without breaking the contract. Must never
  contain ground-truth-only information (e.g. true emitter identity).

### `RFEnvironment` (`interfaces/environment.py`)

Owns the ground-truth simulation state. Abstract methods:
- `reset(seed: int | None = None) -> None` — (re)initialize the world.
- `step(dt: float = 1.0) -> None` — advance the world state by one tick.
- `sense(band: Band, t: float) -> Any` — produce whatever raw, physically
  observable signal exists at `band`/`t`. This is consumed by `Receiver`,
  not read directly by `Scheduler` or `State`.
- `get_ground_truth() -> Any` — returns the full true world state. **By
  contract, only `Evaluator` implementations may call this.** Nothing
  prevents it in Python at the language level; this is an architectural
  rule enforced by code review and tests, not the type system.

### `Receiver` (`interfaces/receiver.py`)

- `reset() -> None`
- `tune(band: Band) -> None` — select which band the receiver is currently
  looking at.
- `observe(environment: RFEnvironment, t: float) -> Observation` — using
  the currently tuned band, call `environment.sense(...)` and turn the
  result into an `Observation`. This is the *only* place `RFEnvironment`
  is touched outside of the `Evaluator`'s setup/teardown and ground-truth
  scoring calls.

### `State` / belief (`interfaces/state.py`)

- `reset() -> None`
- `update(observation: Observation) -> None` — incorporate a new
  observation into the belief.
- `get_features() -> Any` — return whatever representation (dict, vector,
  etc.) the scheduler needs to make its next decision. Must be derivable
  strictly from the history of `Observation`s passed to `update()`.

### `Scheduler` (`interfaces/scheduler.py`)

- `reset() -> None`
- `select_band(state: State) -> Band` — decide which band to observe next,
  using only the belief/state, never the environment.
- `update(observation: Observation, reward: float) -> None` — let the
  scheduler learn from/react to the outcome of its last choice.

### `Evaluator` (`interfaces/evaluator.py`)

- `run_experiment(environment, receiver, scheduler, state, num_steps) -> Any`
  — wires the above together for `num_steps`, calling `environment.step()`,
  `receiver.observe()`, `state.update()`, `scheduler.select_band()`/
  `update()` in the right order, and recording whatever data metrics need.
  This is the only place all five interfaces are imported together.
- `compute_metrics(run_result: Any) -> dict` — turn a recorded run into
  real, computed metrics. Must never return placeholder/fabricated numbers;
  in this phase it is not implemented at all (see below).

## Config (`config.py`)

A single place for experiment parameters that would otherwise be hard-coded
(number of bands, dwell time, episode length, RNG seed, etc.). Phase 0
defines the dataclass shape only, with clearly named placeholder defaults;
real values arrive with Phase 1+.

## Phase 1 — RF Environment & Receiver (decisions of record)

These decisions were approved by the project owner before implementation
and apply to everything under `src/smart_scan_ew/environment/` and
`src/smart_scan_ew/receiver/`.

### Band representation

The public `Band` type stays exactly `int`, unchanged from Phase 0. A
`BandSpec` dataclass (`environment/bands.py`) is used internally by
`RFEnvironment` to attach frequency metadata to each band index:

- `band_id: int`
- `center_frequency: float` (Hz)
- `bandwidth: float` (Hz)

`Receiver`, `State`, and `Scheduler` never see `BandSpec` — only the plain
`int` `Band` values. Bands are contiguous and non-overlapping by
construction (a simplification; see Limitations).

### Emitter models

Exactly three emitter types, each a small internal behavior class in
`environment/emitters.py` (not part of the cross-module `interfaces/`
contracts — these are implementation details of the environment module,
same as the rule that emitter behavior lives "inside" `RFEnvironment"
per `PROJECT_CONTRACT.md`):

- **`ContinuousWaveEmitter`** — always active, fixed band, fixed power.
- **`PulsedEmitter`** — fixed band, fixed power, but active only during a
  duty cycle (`pulse_width` on, `period` total).
- **`FrequencyHoppingEmitter`** — always active, fixed power, but its
  current band changes every `hop_interval`, chosen from an owned,
  explicitly seeded RNG (never Python's global `random` module).

All three are built from a single `EmitterSpec` config dataclass (a `kind`
field selects which behavior class to instantiate) so scenarios are data,
not hardcoded logic (rule 7). A `default_scenario()` factory
(`environment/scenarios.py`) provides one instance of each type for
convenience/demo/testing; callers may instead supply their own explicit
list of `EmitterSpec`s.

No chirp/swept emitter type is included in Phase 1.

### Noise / detection model

Explicitly a **simplified simulation model, not a physically complete RF
model**:

- `RFEnvironment.sense(band, t)` returns the sum of the transmit power of
  whatever emitters are active on `band` at the environment's current
  time (no propagation, path-loss, antenna, or multipath modeling).
- `SimpleReceiver` adds independent Gaussian noise (mean 0, configurable
  `noise_std`) drawn from its own seeded RNG, then thresholds against a
  configurable `detection_threshold` to set `Observation.detected`.
- Because noise is additive and can push the measured value either up or
  down, both **misses** (real transmission present, noise pulls the
  reading below threshold) and **false alarms** (nothing present, noise
  pushes the reading above threshold) are possible outcomes — this is
  intentional and is what makes "smart scanning" a non-trivial problem
  later.

### Ground truth snapshot

`RFEnvironment.get_ground_truth()` returns a full per-emitter snapshot:
simulation time, and for each emitter — `emitter_id`, `active` (bool),
`band` (int), and `power` (float, 0 if inactive). This is Evaluator-only
information; see "Ground-truth isolation" below.

### RNG ownership

No global random state anywhere. Two independent, explicitly owned and
seeded RNG instances exist:

- `RFEnvironment` owns one `random.Random` instance (seeded via
  `reset(seed=...)`, matching the existing interface signature), used for
  all frequency-hopping band draws. Emitters never create their own RNGs;
  the environment passes its RNG into each emitter's `advance()` call, and
  draw order follows a fixed emitter list order — so a given seed
  reproduces the same run.
- `SimpleReceiver` owns a separate `random.Random` instance, seeded at
  construction time (`seed=...` constructor argument) and re-seeded
  identically on `reset()` (the `Receiver.reset()` interface takes no
  arguments, so the seed is supplied once, at construction, not per-reset
  call), used for detection noise.

### Ground-truth isolation (unchanged, now implemented)

`SimpleReceiver.observe()` calls `environment.sense(...)` and nothing
else. It has no code path that calls `get_ground_truth()`, and a test
(`tests/test_ground_truth_isolation.py`) uses a spy environment to assert
this at runtime, not just by code review.

## Phase 2 — Belief/State & Baseline Schedulers (decisions of record)

Approved by the project owner before implementation. Applies to
`src/smart_scan_ew/state/` and `src/smart_scan_ew/scheduler/`.

### State output shape (resolves the Phase 0/1 open item)

`State.get_features()` for `SimpleBeliefState` returns a `BeliefSnapshot`
(frozen dataclass): `current_time: float` and `bands: tuple[BandBeliefView, ...]`
— one entry per band, `0..num_bands-1`, always present even if a band has
never been observed. Each `BandBeliefView` (also frozen) carries:
`band_id`, `observation_count`, `hit_count`, `last_detected` (`bool | None`),
`last_observed_time` (`float | None`), `time_since_last_observed`
(`float | None`), `estimated_probability` (`float | None`, a plain
`hit_count / observation_count` ratio — no smoothing, prior, or decay).
This is a typed, explicit choice over a generic `dict`, made specifically
so Phase 2 schedulers (and later the evaluator/ML scheduler) have a
stable, documented shape to code against.

`estimated_probability` is calculated and exposed for future
learning/analysis use, but no Phase 2 baseline scheduler uses it as a
decision criterion.

### `SimpleBeliefState`

Constructed with an explicit `num_bands` (the receiver's own scan range —
known configuration, not ground truth). Only ever driven by `Observation`
objects passed to `update()`; never imports anything from `environment/`.
`reset()` returns every band to "never observed."

### Baseline schedulers

Three `Scheduler` implementations, one per file under
`scheduler/`, each also taking an explicit `num_bands`:

- **`RoundRobinScheduler`** — cycles `0, 1, ..., num_bands-1, 0, ...`.
  Ignores `state` entirely. No randomness.
- **`RandomScheduler`** — uniform random band from an owned
  `random.Random`, seeded at construction (the `Scheduler.reset()`
  interface takes no arguments, so — same pattern as `SimpleReceiver` in
  Phase 1 — the seed lives at construction and `reset()` re-seeds from
  it). Ignores `state` entirely.
- **`GreedyRecentHitScheduler`** — stateless between calls; reads
  `state.get_features()` fresh each time. Decision rule: (1) among bands
  with `last_detected is True`, pick the largest `last_observed_time`,
  ties broken by lowest `band_id`; (2) if no band has ever hit, pick the
  fewest `observation_count`, ties broken by lowest `band_id`. No
  exploration probability, no decay, no use of `estimated_probability`.
  Does not import `state/`'s concrete types (see the module's docstring
  for the duck-typing rationale) — it depends only on `interfaces/`.

All three baseline schedulers ignore the `reward` argument to `update()`
entirely — reward-driven behavior is reserved for the Phase 4 learning
scheduler; these exist specifically to be fair, non-learning comparison
points (see "Comparing baselines with an ML scheduler" below).

### `num_bands` consistency (explicit tradeoff, not automated)

`SimpleRFEnvironment`, `SimpleBeliefState`, and each scheduler all take
`num_bands` as an independent constructor argument — no shared config
object ties them together yet (owner's explicit decision). Whoever wires
a run together (currently: test code; later: the Phase 3 `Evaluator`) is
responsible for passing the same value everywhere. Nothing enforces this
automatically today.

### Ground-truth isolation, extended

`tests/test_phase2_ground_truth_isolation.py` extends the Phase 1 spy
pattern across the full `environment → receiver → state → scheduler`
loop for all three baseline schedulers, asserting `get_ground_truth()` is
never called anywhere in the loop.

### Comparing baselines with an ML scheduler later

The plan (to be implemented in Phase 3/4, not now): the Phase 3
`Evaluator` will run the *same* `RFEnvironment` scenario and seed, and
the *same* `Receiver` seed, against each candidate `Scheduler` in turn —
the three baselines here, and later the learning-based scheduler. Holding
the emitter/noise randomness identical across runs isolates the outcome
difference to the scheduling policy itself, which is what makes the
baselines a *fair* comparison point rather than an arbitrary one.

## Phase 3 — Evaluation Framework (decisions of record)

Approved by the project owner before implementation. Applies to
`src/smart_scan_ew/evaluator/`. `SimpleEvaluator` implements the existing
`Evaluator` interface exactly as declared in Phase 0 — no interface
change was made or needed (see "Interface stability" below).

### Terminology

- **Detection**: `Observation.detected == True` — receiver-level, no
  regard to correctness (same meaning as Phase 2's `hit_count`).
- **True positive**: a detection where ground truth confirms a real
  emitter was active on the observed band at that instant.
- **False alarm**: a detection where ground truth shows nothing was
  active on that band.
- **Miss**: `detected == False` while ground truth shows something was
  active on that band.
- **Interception** (per-emitter, latched, one-time): the first true
  positive that credits a given emitter. Repeated true positives on an
  already-intercepted emitter add to raw TP counts but never create a
  second interception or move `intercept_time`.

### Metric definitions

Given per-step classification (`signal_present` = any ground-truth-active
emitter on the observed band at that instant):

- **Pd** = `TP / (TP + FN)`, `None` if `TP+FN == 0`. Explicitly a
  receiver/detector-quality metric, conditional on the receiver already
  observing an occupied band — **not** a search/interception metric.
- **Pfa** = `FP / (FP + TN)`, `None` if `FP+TN == 0`.
- **Interception Rate**: reported as **two** numbers —
  `interception_rate_all_emitters` (denominator = every emitter in the
  scenario) and `interception_rate_active_emitters` (denominator = only
  emitters active at least once during the run). **The active-emitters
  variant is the primary/headline metric** for presentation and
  cross-scheduler comparison; the all-emitters variant remains available
  for auditability (an emitter with a duty cycle longer than the run
  duration can never be intercepted regardless of scheduler quality, and
  folding that into one ratio would misattribute a scenario-design
  limitation to scheduler performance).
- **Average Intercept Time**: mean of `intercept_time` over intercepted
  emitters only. `None` if none intercepted. Never-intercepted emitters
  are excluded, not assigned a censored value — no Kaplan–Meier-style
  survival analysis in Phase 3 (documented simplification).
- **Intercept Time Error**: `intercept_time - first_active_time`
  (`first_active_time` = earliest time, from ground truth across the
  whole run regardless of receiver tuning, that the emitter was active at
  all). Interpreted as detection/interception delay from the earliest
  theoretical opportunity — always `>= 0` by construction. Averaged over
  intercepted emitters only, same `None` rule as above.
- **Co-band emitters**: a single true-positive observation credits
  *every* ground-truth-active emitter on that band simultaneously —
  documented limitation of the Phase 1 sensing model (`sense()` sums
  power and cannot distinguish co-band emitters), not a Phase 3 bug.
- **Average Reward / Cost**: `reward = 1.0 if detected else 0.0`,
  applied per step (a Phase 3 **placeholder**, not a tuned optimization
  objective — Phase 4 owns real reward-shaping). `average_reward` = mean
  over all steps; `average_cost = 1 - average_reward`. Mixes true and
  false detections together — **not** a substitute for Pd/Pfa.
- Every metric's raw counts (`true_positive_count`, etc.) are stored
  alongside the derived ratio specifically so Pd/Pfa can be recomputed by
  hand from the stored counts — no opaque numbers.

### Reproducibility strategy

`SimpleRFEnvironment`, `SimpleReceiver`, and `RandomScheduler` already
each own an independent `random.Random` (Phase 1/2 decisions) — no
component's draws affect any other's. `evaluator/reproducibility.py`'s
`derive_seeds(master_seed, count)` builds a fixed, documented mapping
from one master seed to `count` independent sub-seeds (role order:
0=environment, 1=receiver, 2=scheduler-if-stochastic), using a
throwaway `random.Random(master_seed)` consumed only for this derivation.
`compare_baselines()` derives `env_seed`/`receiver_seed` **once** and
reuses them for all three schedulers, so ground truth and receiver noise
trajectories are byte-identical across the comparison — the only
difference between runs is the scheduling policy. Verified directly (not
just asserted) in `tests/test_reproducibility.py` by comparing raw
`RunRecord` ground-truth and `measured_power` sequences.

### Experiment lifecycle / reset behavior

`run_experiment_for_scheduler()` (the orchestration layer, not the
`Evaluator` interface): constructs a fresh `SimpleRFEnvironment` and
`reset(seed=env_seed)`s it; constructs a fresh `SimpleReceiver(seed=
receiver_seed)` and `reset()`s it; constructs a fresh `SimpleBeliefState`
and `reset()`s it; **never reconstructs the scheduler** — only calls
`scheduler.reset()` on the instance the caller passed in, so a future
learned scheduler's persistent parameters survive across episodes while
episode-local counters clear. `SimpleEvaluator.run_experiment()` itself
resets nothing (matches the Phase 0 `NullEvaluator` fixture convention —
resetting is the caller's job, before `run_experiment` is invoked).

### Interface stability

No Phase 0/1/2 interface was changed (confirmed by an empty `git diff`
against every file under `interfaces/`, `environment/`, `receiver/`,
`state/`, `scheduler/`, and `config.py`). One near-miss considered and
resolved without a change: `Evaluator.run_experiment()`'s signature has
no `dt` parameter, so `SimpleEvaluator` takes `dt` (and `reward_fn`) as
**constructor** parameters instead — the abstract `Evaluator` doesn't
constrain `__init__`, so this isn't an interface change.

### Ground-truth isolation, extended again

Phase 3 introduces the first *legitimate* caller of `get_ground_truth()`
(`SimpleEvaluator`, once per step, for its own `StepRecord` only).
`tests/test_evaluator_ground_truth_isolation.py` proves the boundary
holds: an environment spy confirms `get_ground_truth()` is called exactly
once per step (expected now, unlike Phase 1/2's "must be zero"), while a
*scheduler-argument* spy and *state-argument* spy independently confirm
neither ever receives anything beyond a plain `State`/`Observation`, and
that `Observation.info` never carries a `ground_truth`/`emitters`/
`emitter_id` key (guarding the one deliberately-open field that could
leak truth in a careless future change). A separate test confirms the
placeholder reward function's only parameter is the `Observation` itself.

### File layout

```
src/smart_scan_ew/evaluator/
    __init__.py
    records.py        # StepRecord, RunRecord, EmitterInterceptionRecord, ExperimentResult
    simple_evaluator.py  # SimpleEvaluator(Evaluator)
    reproducibility.py    # derive_seeds()
    experiment.py           # ExperimentConfig, run_experiment_for_scheduler()
    comparison.py            # compare_baselines(), ComparisonResult, run_repeated_trials(), TrialSummary, MetricStats
```

### Repeated trials

`run_repeated_trials()` is deliberately composable, not combined: it runs
**one** scheduler across multiple master seeds and reports
mean/stdev/min/max (stdlib `statistics` only — no confidence intervals or
bootstrapping) per metric, plus `n_defined` (how many trials had a
non-`None` value for that metric, since e.g. a scenario where some seeds
produce zero interceptions leaves `average_intercept_time` undefined for
those trials). Comparing multiple schedulers across multiple seeds means
calling this once per scheduler and collecting the `TrialSummary`s
yourself — no N-scheduler × M-seed combined structure was built.

## What is deliberately still NOT decided

- The learning algorithm for the eventual learning-based `Scheduler` —
  intentionally unspecified.
- Config file format (dataclass-only for now; whether it's loaded from
  YAML/JSON later is an open decision).
- Real propagation/antenna/multipath modeling — explicitly out of scope,
  not just deferred; see Phase 1 Limitations in the implementation summary.
- A shared config object tying `num_bands` together across environment,
  state, and scheduler — currently independent constructor arguments by
  explicit Phase 2 decision; may be revisited when the Evaluator (Phase 3)
  needs to wire all of them together itself.
