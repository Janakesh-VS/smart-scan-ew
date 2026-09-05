# smart-scan-ew
ML-based Smart Scan Strategy for Electronic Warfare PS 26055 – SAH/SIH 2026

## Project status

The project has completed Phase 5.

### Phase 0 — Foundation
Completed:
- Project architecture
- Module contracts
- Interfaces
- Initial test structure

### Phase 1 — RF Environment & Receiver
Completed:
- Continuous-wave emitter
- Pulsed emitter
- Frequency-hopping emitter
- Simulated RF environment
- Receiver with noise and detection threshold
- Ground-truth isolation
- Phase 1 tests

### Phase 2 — State/Belief & Baseline Schedulers
Completed:
- Band belief representation
- Observation history
- Estimated transmission probability
- Round-robin scheduler
- Random scheduler
- Greedy recent-hit scheduler
- Ground-truth isolation tests
- Phase 2 tests

### Phase 3 — Evaluation Framework
Completed:
- SimpleEvaluator (implements the existing Evaluator interface unchanged)
- Pd, Pfa, interception rate (all-emitter and active-emitter variants),
  average intercept time, intercept time error, average reward/cost
- Raw TP/FP/FN/TN counts stored alongside every derived metric
- Reproducibility strategy (`derive_seeds`) — same environment/receiver
  trajectory across schedulers for a fixed master seed
- `compare_baselines()` and `run_repeated_trials()`
- Ground-truth isolation tests at the evaluator boundary
- Phase 3 tests

### Phase 4 — Learning-based scheduler
Completed:
- `AdaptiveUcbScheduler` — a non-stationary/recency-aware, discounted-UCB
  multi-armed bandit (explicitly NOT a contextual bandit: independent
  per-band statistics, no shared model). Learns online from
  `observation.detected` only; never sees ground truth or `state`'s
  contents.
- Deterministic unobserved-band-first exploration, then
  `score(b) = p_b + c*sqrt(ln(t)/N_b)` with an `EPSILON`-floored `N_b`
  and an explicit `observed_b` flag, so long, unrevisited bands can
  never trigger a division-by-zero or NaN/inf score.
- Hyperparameter grid (`gamma in {0.90, 0.95, 0.99, 1.00}`,
  `c in {0.5, 1.0, 2.0}`) selected on dedicated selection-only seeds
  by maximizing mean active-emitter interception rate, then frozen
  and evaluated on disjoint held-out seeds.
  (`examples/phase4_experiment.py`).
- Four-way comparison against the three Phase 2 baselines using the
  existing Phase 3 `run_repeated_trials()` — no evaluator code changed.
- Ground-truth isolation tests (structural + runtime spy), matching the
  Phase 1/2 convention.
- Phase 4 tests.

### Phase 5 — Dashboard
Completed:
- A Streamlit dashboard (`src/smart_scan_ew/dashboard/`) built entirely
  on top of the existing Phase 1-4 components and Phase 3 evaluator
  APIs — no simulation, scheduler, or evaluator logic was reimplemented.
  `controller.py` holds all non-UI logic (zero `streamlit` import);
  `app.py` is the UI.
- Live, step-by-step simulation view: time-frequency map (with a
  clearly labelled "GROUND TRUTH — EVALUATOR / DEBUG VIEW" panel, never
  fed to the scheduler), latest observation, Phase 2 belief state, and —
  for Adaptive UCB — a live per-band `S_b`/`N_b`/hit-rate/UCB-score
  table and a "why this band was selected" explanation, both sourced
  entirely from `AdaptiveUcbScheduler`'s new read-only
  `get_diagnostics()`/`decision_count` accessors (an additive,
  observability-only change — see `ARCHITECTURE.md`'s Phase 5 section;
  no equation, update rule, or algorithmic behavior changed).
- Real evaluator-backed Performance tab (`run_experiment_for_scheduler`)
  and a Four-Way Comparison tab (`run_repeated_trials`, reusing the same
  Phase 3 APIs `examples/phase4_experiment.py` uses — the dashboard does
  not import that script). Undefined metrics render as `N/A`, never `0`.
- Ground-truth isolation re-verified for the dashboard's own code path
  (structural + runtime spy, matching the Phase 1/2/4 convention).
- Phase 5 tests, including headless Streamlit `AppTest` smoke tests.

### Current phase

Phase 5 complete. See `ARCHITECTURE.md`'s "Phase 4" and "Phase 5"
sections for the full algorithm/dashboard design, and this repo's known
limitations (documented, not hidden): the UCB scheduler optimizes
observable detections, not true interception count; the dashboard's
"Start/Pause" auto-run is a best-effort Streamlit rerun loop, not a
guaranteed-smooth animation (see `CLAUDE.md`).

### Running tests

```bash
pip install -r requirements.txt
pytest
```

### Running the dashboard

```bash
pip install -r requirements.txt
streamlit run src/smart_scan_ew/dashboard/app.py
```
