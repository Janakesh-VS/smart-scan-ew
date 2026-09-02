# smart-scan-ew
ML-based Smart Scan Strategy for Electronic Warfare PS 26055 – SAH/SIH 2026

## Project status

The project has completed Phase 1.

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

### Current phase

Phase 1 complete. Phase 2 will implement the state/belief representation
and baseline scanning schedulers.
### Running tests

```bash
pip install -r requirements.txt
pytest
```
