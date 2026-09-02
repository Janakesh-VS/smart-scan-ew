# smart-scan-ew
ML-based Smart Scan Strategy for Electronic Warfare PS 26055 – SAH/SIH 2026

## Project status

The project has completed Phase 2.

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

### Current phase

Phase 2 complete. Phase 3 will implement the evaluation framework
and performance metrics for comparing scanning strategies.
### Running tests

```bash
pip install -r requirements.txt
pytest
```
