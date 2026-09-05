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

## Phase 4 — Adaptive Discounted-UCB Scheduler (decisions of record)

Approved by the project owner before implementation. Applies to
`src/smart_scan_ew/scheduler/adaptive_ucb.py` and
`examples/phase4_experiment.py`. No `interfaces/`, `environment/`,
`receiver/`, `state/`, `evaluator/`, or existing `scheduler/` baseline
file was modified — only `scheduler/__init__.py`'s export list changed
(confirmed by `git diff`).

### Terminology

Explicitly **not** a contextual bandit: `AdaptiveUcbScheduler` maintains
completely independent statistics per band, with no shared model and no
feature vector. Correctly described as a non-stationary/recency-aware
multi-armed bandit — specifically a discounted-UCB variant.

### Where the learning state lives

`SimpleBeliefState`/`BeliefSnapshot` are unchanged and are not this
scheduler's data source. `AdaptiveUcbScheduler` maintains its own private
per-band statistics, exactly like `RandomScheduler` already owns its own
RNG state. `select_band(state)` accepts `state` only because the
`Scheduler` interface requires it; its contents are never read — the
same choice `RoundRobinScheduler`/`RandomScheduler` already make.

### State and update rule

Per band `b`: `S_b` (discounted successes), `N_b` (discounted
observation count), `observed_b` (bool, whether `b` has EVER been
observed — never derived from `N_b`, see "Numerical safety" below). Plus
one scheduler-wide `t`: the 1-indexed, episode-local decision count.

On every `update(observation, reward)` call (`reward` is accepted per
the interface but unused — see "Reward" below):

```
for every band:      S_b <- gamma * S_b ;  N_b <- gamma * N_b
for the observed band b:
    N_b <- N_b + 1
    observed_b[b] <- True
    if observation.detected:  S_b <- S_b + 1
```

There is exactly one recency parameter, `gamma`; no separate EWMA
smoothing constant exists. `gamma == 1.0` means no discounting —
`S_b`/`N_b` behave as ordinary lifetime UCB1 running counts. `gamma < 1`
means older observations lose influence geometrically; smaller `gamma`
forgets faster.

### Initialization / exploration priority

Before any UCB scoring: if any band has `observed_b[b] is False`, return
the lowest-numbered such band, deterministically. This guarantees every
band gets at least one observation before UCB scoring is used at all,
and is checked before any division, log, or sqrt is computed.

### UCB score and `t`

Once every band has been observed at least once:

```
p_b     = S_b / max(N_b, EPSILON)
score_b = p_b + c * sqrt(ln(t) / max(N_b, EPSILON))
```

`t` is the 1-indexed episode-local decision count for the decision about
to be made (`1 + number of update() calls so far this episode`). `t >= 1`
always by construction, so `ln(t) >= 0` and the `sqrt` argument is never
negative — independent of the `EPSILON` floor. `argmax(score)` is
returned; ties are broken deterministically by lowest `band_id`.
`select_band()` never mutates internal state — it is a pure function of
the scheduler's current `S`/`N`/`observed`/`t`, so calling it repeatedly
without an intervening `update()` always returns the same band.

### Numerical safety

`EPSILON = 1e-12`. Geometric decay of `N_b` by `gamma < 1` on every step
(for every band, not just the one observed) means an unrevisited band's
`N_b` shrinks toward, and — in IEEE-754 float64, after enough steps —
eventually reaches exactly `0.0`. `max(N_b, EPSILON)` prevents this from
ever causing a `ZeroDivisionError`, an invalid `sqrt`, or an invalid
`log`. This floor is applied only when computing a score for an already-
observed band; it is never used to decide whether a band has been
observed at all — that distinction is made solely by the explicit
`observed_b` flag, never by comparing `N_b` to zero (a "never observed"
band and a "heavily discounted, `N_b` near/at zero" band are different
things and must not be confused). Verified by
`tests/test_adaptive_ucb_scheduler.py`'s numerical-safety tests, which
run the scheduler for tens of thousands of steps and for enough
`gamma < 1` decay steps to force an actual `N_b` underflow to `0.0`, and
assert the resulting score stays finite and non-NaN.

Discounting `S_b` and `N_b` by the same factor leaves their ratio `p_b`
unchanged for a band not itself being updated — discounting alone never
moves a band's point estimate. What it does do is (a) make a revisited
band's estimate a genuine recency-weighted average, and (b) shrink an
unvisited band's `N_b`, which grows its exploration bonus over time. No
formal regret bound or "never permanently abandoned" guarantee is
claimed — this is a standard, numerically-guarded heuristic.

### Reward

`update()`'s `reward` argument exists only because the `Scheduler`
interface requires it; this scheduler reads `observation.detected`
directly instead. A real receiver cannot distinguish a true positive
from a false alarm without ground truth, which this scheduler must never
see — so every detected observation is simply an observable detection,
with no true/false qualification to learn from. No reward shaping
(first-hit bonus, miss penalty, or anything ground-truth-derived) is
used.

### Objective mismatch (documented limitation, not a bug)

This scheduler maximizes cumulative *observed* detections. The project's
actual goal is to intercept as many distinct emitters as possible — but
the scheduler cannot know emitter identity, only
`Observation.detected`. A scenario with one strong emitter and one weak
emitter may see the scheduler favor the strong one; this is an inherent
property of a detection-reward bandit, not a defect. Discounting does
**not** predict where a `FrequencyHoppingEmitter` will move next — it
only makes recent evidence outweigh old evidence once a band is
revisited.

### Hyperparameters

Grid: `gamma ∈ {0.90, 0.95, 0.99, 1.00}`, `c ∈ {0.5, 1.0, 2.0}` (12
configurations). No additional hyperparameters.

### Three distinct concepts (do not conflate)

- **Online learning**: automatic, inside a single experiment run, driven
  entirely by `update()` calls from the Phase 3 evaluator loop.
- **Hyperparameter selection**: a grid search over the 12 configurations
  above, using `run_repeated_trials()` (unchanged) on dedicated
  selection-only master seeds — conventional grid search, not
  neural-network-style training; nothing here performs gradient descent.
- **Final held-out evaluation**: the frozen `(gamma, c)` re-evaluated on
  a disjoint set of master seeds never used during selection.

### Selection procedure (`examples/phase4_experiment.py`)

Two-stage rule, deliberately not a weighted composite score:

1. **Primary**: maximize mean `interception_rate_active_emitters` across
   the selection seeds.
2. **Near-tie rule**: among configurations within `0.02` absolute of the
   best mean interception rate, choose the one with the lowest mean
   `average_intercept_time`. A configuration with an undefined
   (`None`) mean on either metric (no trial ever intercepted anything)
   is treated as strictly worse than any configuration with a defined
   value.

### Held-out evaluation / four-way comparison

`run_held_out_evaluation()` runs Round Robin, Random, Greedy Recent Hit,
and the frozen Adaptive UCB via `run_repeated_trials()`, each across the
identical list of held-out master seeds. Because `derive_seeds()`'s
environment/receiver sub-seeds depend only on the master seed, never on
which scheduler is under test, all four schedulers see byte-identical
environment/receiver trajectories for a given seed — the same fairness
property `compare_baselines()` provides for a single run, extended here
across repeated trials and a fourth scheduler without modifying
`evaluator/comparison.py`. Reports Pd, Pfa, both interception-rate
variants, average intercept time, intercept time error, and average
reward/cost, each with mean/stdev/min/max/`n_defined`
(`SELECTION_SEEDS`/`HELD_OUT_SEEDS` are disjoint tuples, checked by a
dedicated test).

### Ground-truth isolation, extended again

`adaptive_ucb.py` imports only `interfaces/` and the standard library —
never `environment/` or `evaluator/`. Verified by
`tests/test_phase4_ground_truth_isolation.py`: a structural test walks
the module's AST for forbidden imports/identifiers, and a runtime spy
test runs the complete environment → receiver → state → scheduler loop
and asserts `get_ground_truth()` is called zero times.

### File layout

```
src/smart_scan_ew/scheduler/adaptive_ucb.py   # AdaptiveUcbScheduler, EPSILON
examples/phase4_experiment.py                  # selection, held-out eval, 4-way comparison
tests/test_adaptive_ucb_scheduler.py           # unit + integration tests
tests/test_phase4_ground_truth_isolation.py    # structural + runtime isolation tests
```

## Phase 5 — Dashboard (decisions of record)

Approved by the project owner before implementation. Applies to
`src/smart_scan_ew/dashboard/` (`controller.py`, `app.py`,
`__init__.py`) and one additive change to
`src/smart_scan_ew/scheduler/adaptive_ucb.py`. No `interfaces/`,
`environment/`, `receiver/`, `state/`, `evaluator/`, or existing
baseline `scheduler/` file was modified — confirmed by `git diff`.

### Decision 1 — Adaptive UCB state access (additive, non-algorithmic)

`AdaptiveUcbScheduler` gained two new, read-only members, purely for
observability:

- `get_diagnostics() -> tuple[BandUcbDiagnostics, ...]` — one frozen
  `BandUcbDiagnostics` per band (`band_id`, `discounted_successes`,
  `discounted_observations`, `observed`, `estimated_hit_rate`,
  `exploration_bonus`, `ucb_score`). For an unobserved band, the last
  three fields are `None` — the real algorithm never computes a UCB
  score for such a band (see the unobserved-first branch in
  `select_band()`), so reporting a number there would misrepresent the
  actual decision rule.
- `decision_count` — a read-only property returning `self._t` (the same
  `t` `select_band()` uses internally, as `decision_index = decision_count + 1`).

Implementation approach: `select_band()`'s per-band score computation
(`n_safe = max(N_b, EPSILON); p_b = S_b/n_safe; bonus = c*sqrt(log_t/n_safe)`)
was extracted, verbatim, into a private `_score_components(band_id, log_t)`
helper. `select_band()` now calls this helper instead of inlining the
same three lines; `get_diagnostics()` calls the identical helper for
every observed band. This means there is exactly ONE implementation of
the UCB formula in the class — a dashboard (or any future consumer)
displaying a score can never drift from what the algorithm actually
used to decide.

**No algorithmic behavior changed.** The UCB equations, update order,
initialization/exploration-priority rule, `gamma`/`c` semantics, tie-
breaking, and `reset()` semantics are byte-for-byte identical before and
after this refactor. Verified directly: the full 138-test pre-Phase-5
suite was re-run immediately after the refactor, before any dashboard
code existed, and still passed 138/138 — including the exact
hand-calculation test (`test_ucb_score_matches_hand_calculation`) that
would catch any numeric drift in the formula. New tests
(`tests/test_adaptive_ucb_scheduler.py`'s "Phase 5 additive accessors"
section) additionally verify: `get_diagnostics()` reports `None` for
unobserved bands; its numbers exactly match the scheduler's internal
`_S`/`_N` and the same hand-calculated formula; the band with the
highest `ucb_score` among `get_diagnostics()`'s entries is always the
band `select_band()` itself picks (once all bands are observed); the
returned `BandUcbDiagnostics` tuple is immutable
(`dataclasses.FrozenInstanceError` on assignment); calling
`get_diagnostics()`/`decision_count` any number of times never mutates
scheduler state; and `decision_count` has no setter.

### Decision 2 — Four-way comparison does not import the Phase 4 script

`dashboard/controller.py` does **not** import
`examples/phase4_experiment.py` — that script is executable
orchestration (a `main()` entry point meant to be run directly), not
application code a UI should depend on. Instead,
`controller.run_four_way_comparison()` is built directly on the same
reusable Phase 3 evaluator APIs that script also uses —
`ExperimentConfig`, `run_repeated_trials`, and (via
`DashboardController.reset()`/`replay_metrics()`) `derive_seeds` — with
no evaluator code duplicated or modified. A structural test
(`tests/test_dashboard_controller.py::test_controller_module_does_not_import_phase4_experiment_script`)
walks `controller.py`'s AST for import statements referencing
`phase4_experiment` and asserts there are none.

### Architecture: `controller.py` (no UI) / `app.py` (Streamlit)

`controller.py` imports zero `streamlit` (checked structurally by a
dedicated test) and contains all non-UI logic: it owns one
`DashboardController` per live session, wrapping one real
`SimpleRFEnvironment`, `SimpleReceiver`, `SimpleBeliefState`, and one of
the four real `Scheduler` implementations (constructed via
`_build_scheduler()` — a plain dispatch, no scheduler behavior
reimplemented). `app.py` contains only Streamlit calls and formatting;
it never touches simulation, scheduler, or evaluator logic directly —
everything routes through the controller.

`DashboardController.step()` performs exactly the same call sequence
`SimpleEvaluator.run_experiment()` uses — `environment.step()` →
`scheduler.select_band(state)` → `receiver.tune()`/`observe()` →
`state.update()` → `scheduler.update()` — and computes `t` the same way
(`self._time += dt`, matching `(step_index + 1) * dt`). This is why
`DashboardController.replay_metrics()` (a fresh
`run_experiment_for_scheduler()` call using the live session's exact
seed/scenario/scheduler-type/hyperparameters) reproduces the live
session's trajectory when given the same `num_steps` — verified
directly (`test_replay_metrics_reproduces_the_exact_live_trajectory`,
`test_replay_metrics_matches_a_direct_evaluator_call_with_the_same_config`),
not just asserted in a comment. `replay_metrics()` always constructs a
**fresh** scheduler instance for the replay — it never reuses or
advances the live, already-stepped one.

### "Why this band was selected" explanation

`StepResult.decision_reason` (a `DecisionReason`) is built by
`_explain_decision()`, which takes the band chosen this step and the
`get_diagnostics()` snapshot taken **immediately before** `update()` was
applied for that step — i.e. the exact internal state `select_band()`
itself used. If that band's `observed` flag was `False` at that moment,
`was_unobserved_band=True` and the score fields are `None` (mirroring
the real algorithm never computing a score for it); otherwise the three
score components are copied straight from that pre-update diagnostics
entry. No second implementation of the algorithm exists anywhere in
this path.

### Live simulation view

Built entirely from real objects: the time-frequency map's ground-truth
layer comes from `DashboardController.peek_ground_truth()` (called by
`app.py` after each `step()`, purely for display, and appended to a
`st.session_state` list `app.py` owns — never read back into
`controller.step()`); the receiver-visible layer (selected band,
HIT/MISS) comes from `StepResult`/`Observation`. The belief-state panel
comes from `SimpleBeliefState.get_features()` unchanged. The Adaptive UCB
panel and "why" explanation come entirely from `get_diagnostics()`/
`DecisionReason` (Decision 1). Undefined metrics (a `None` from
`MetricStats`/`ExperimentResult`) render as the literal string `"N/A"`
via one shared `format_metric()` helper in `app.py` — never silently
converted to `0`; a real `0.0` still renders as `"0.000"`, not `"N/A"`
(both directions tested).

### Scenario presets

`SCENARIO_PRESETS` in `controller.py` maps a display name to a
`(num_bands) -> list[EmitterSpec]` builder. Every preset is a
composition of the three EXISTING emitter kinds
(`"cw"`/`"pulsed"`/`"hopping"`) already implemented in
`environment/emitters.py` — "Default" reuses `default_scenario()`
unchanged; "CW only" / "Pulsed only" / "Frequency hopping only" are new
`EmitterSpec` lists using those same three kinds. No new emitter model
was introduced (Part 14 of the approved brief).

### Streamlit session-state handling

The `DashboardController` instance lives in `st.session_state`, never a
module-level global — the standard Streamlit multi-session-safety
requirement. A fresh `controller.reset()` is triggered only when the
resolved configuration tuple `(scheduler_name, scenario, num_bands,
seed, gamma, c)` differs from `st.session_state.applied_config`, or when
"Reset" is pressed — clicking "Step Once" or any button that doesn't
touch those values causes zero resets (tested:
`test_changing_scheduler_triggers_exactly_one_fresh_reset_not_a_reset_per_rerun`).
"Start"/"Pause" implement a bounded auto-advance loop (`speed` steps,
then `time.sleep(0.3)`, then `st.rerun()`) — this is the standard,
if imperfect, way to get a "live" feel under Streamlit's
rerun-per-interaction execution model, and is explicitly **not**
claimed to be a smooth, frame-accurate animation. "Step Once" always
works regardless of `running` state and is the reliable, precise control
for a live demo walkthrough.

### Ground-truth isolation, re-verified for the dashboard

`DashboardController.step()`'s own source contains no
`get_ground_truth`/`peek_ground_truth` call (structural test). A runtime
spy (`_GroundTruthSpy`, the same pattern used in every previous phase's
isolation tests) wraps the controller's environment and asserts zero
`get_ground_truth()` calls across a full step loop, for all four
scheduler choices. `peek_ground_truth()` — the one intentional call
site — is verified to be additive only: interleaving it with `step()`
calls doesn't change `step()`'s own zero-ground-truth-calls behavior,
and its return value is never threaded back into `state`/`scheduler`.

### Testing approach

Three new test files, mirroring the existing one-file-per-concern
convention:
- `tests/test_dashboard_controller.py` — pure-Python logic tests (no
  `streamlit` import), covering construction, scheduler selection,
  reset, stepping, real-belief/real-diagnostics exposure, four-way
  comparison, replay-metrics correctness, and `N/A` formatting.
- `tests/test_dashboard_ground_truth_isolation.py` — structural +
  runtime-spy isolation tests, matching the Phase 1/2/4 convention.
- `tests/test_dashboard_app.py` — headless smoke tests using Streamlit's
  own `streamlit.testing.v1.AppTest` (the officially supported way to
  run a Streamlit script without a browser): confirms the app imports
  and starts, all four scheduler choices step without error, the
  Adaptive UCB panel and "why" explanation render, the four-way
  comparison and performance buttons complete without exception using
  real backend calls, and Reset/rerun behavior is predictable. This
  caught one real bug during development — mixing `bool` and the
  string `"N/A"` in one pandas column broke `st.dataframe`'s Arrow
  serialization — fixed by rendering `last_detected` as a single
  string type (`"HIT"`/`"MISS"`/`"N/A"`) in every row.

### File layout

```
src/smart_scan_ew/scheduler/adaptive_ucb.py   # + BandUcbDiagnostics, get_diagnostics(), decision_count (additive)
src/smart_scan_ew/dashboard/__init__.py        # re-exports the controller API
src/smart_scan_ew/dashboard/controller.py      # all non-UI logic; zero streamlit import
src/smart_scan_ew/dashboard/app.py             # Streamlit UI only
tests/test_dashboard_controller.py             # logic tests
tests/test_dashboard_ground_truth_isolation.py # structural + runtime isolation tests
tests/test_dashboard_app.py                    # headless Streamlit AppTest smoke tests
```

### Known limitations

- "Start"/"Pause" auto-run is a best-effort Streamlit rerun loop
  (fixed steps + sleep + `st.rerun()`), not a guaranteed-smooth,
  frame-accurate animation — a Streamlit architectural constraint, not
  something this dashboard can fix without a different UI framework
  (explicitly out of scope per Part 6).
- The Performance tab's metrics are computed by a fresh
  `run_experiment_for_scheduler()` replay of the current configuration,
  not a running tally of the exact steps displayed in the Live
  Simulation tab as they happen — they agree exactly when
  `num_steps == len(history)` (verified), but the two panels are
  computed independently, not from one shared incremental accumulator.
- Scenario presets are a small, fixed set of compositions of the three
  existing emitter kinds — not a general scenario editor.

## What is deliberately still NOT decided

- Config file format (dataclass-only for now; whether it's loaded from
  YAML/JSON later is an open decision).
- Real propagation/antenna/multipath modeling — explicitly out of scope,
  not just deferred; see Phase 1 Limitations in the implementation summary.
- A shared config object tying `num_bands` together across environment,
  state, and scheduler — currently independent constructor arguments by
  explicit Phase 2 decision; may be revisited when the Evaluator (Phase 3)
  needs to wire all of them together itself.
