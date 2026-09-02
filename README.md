# smart-scan-ew
ML-based Smart Scan Strategy for Electronic Warfare PS 26055 – SAH/SIH 2026

## Project status

The project has completed Phase 3.

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

### Current phase

Phase 3 complete. Phase 4 will introduce a learning-based scheduler,
evaluated against the Phase 2 baselines using the Phase 3 framework.

### Running tests

```bash
pip install -r requirements.txt
pytest
```
