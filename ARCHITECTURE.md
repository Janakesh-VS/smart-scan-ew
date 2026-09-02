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
