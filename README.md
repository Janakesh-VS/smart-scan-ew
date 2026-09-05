# smart-scan-ew
ML-based Smart Scan Strategy for Electronic Warfare PS 26055 – SAH/SIH 2026

## Project status

The project has completed Phase 4.

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

### Phase 4 — Learning-Based Scheduler
Completed:
- AdaptiveUcbScheduler (Adaptive Discounted-UCB multi-armed bandit,
  not a contextual bandit — no shared feature model across bands)
- Discounted per-band statistics (S_b, N_b), UCB score with a scheduler-
  owned ordinary decision counter t
- reward argument accepted (interface requirement) but ignored;
  observation.detected used directly
- Hyperparameter grid search (gamma x exploration_constant), reusing
  run_repeated_trials unmodified — not a neural-network training loop
- Real held-out comparison against Round Robin, Random, and Greedy
  Recent Hit, with honest, non-cherry-picked interpretation
- No changes to any Phase 0-3 interface or to the Phase 3 evaluator/
  reward framework
- Full Phase 4 test suite, including ground-truth isolation

### Current phase

Phase 4 complete. See ARCHITECTURE.md's "Phase 4" section for the full
specification, known limitations, and real held-out experiment results.

### Running tests

```bash
pip install -r requirements.txt
pytest
```
