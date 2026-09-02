"""RandomScheduler: uniform random band selection from an owned, seeded
RNG. See ARCHITECTURE.md's Phase 2 section.
"""

import random

from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State


class RandomScheduler(Scheduler):
    """Selects a uniformly random band each call.

    Ignores `state` entirely. Owns a single `random.Random` instance,
    never Python's global `random` module. The `Scheduler.reset()`
    interface takes no arguments, so the seed is fixed at construction and
    `reset()` re-seeds from it — the same pattern used by SimpleReceiver
    in Phase 1, for the same reason (reproducibility without a per-reset
    seed parameter).
    """

    def __init__(self, num_bands: int, seed: int | None = None):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        self._num_bands = num_bands
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_band(self, state: State) -> Band:
        return self._rng.randrange(self._num_bands)

    def update(self, observation: Observation, reward: float) -> None:
        pass  # non-learning baseline: nothing to react to
