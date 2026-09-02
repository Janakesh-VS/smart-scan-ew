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

## What is deliberately NOT decided yet

- The concrete `Band` representation beyond "opaque, hashable value."
- The internal structure of `info` on `Observation`.
- Whether `State.get_features()` returns a dict, a fixed-size vector, or a
  small custom type — deferred until a scheduler actually needs one.
- The learning algorithm for the eventual learning-based `Scheduler` —
  intentionally unspecified per the project owner's instruction not to
  assume the final ML approach.
- Config file format (dataclass-only for now; whether it's loaded from
  YAML/JSON later is an open decision).
