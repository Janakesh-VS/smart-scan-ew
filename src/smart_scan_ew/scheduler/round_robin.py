"""RoundRobinScheduler: cycles through every band in fixed order.

Belief-blind by design — it is the simplest possible fair baseline
(guaranteed even coverage, zero adaptivity). See ARCHITECTURE.md's
Phase 2 section.
"""

from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State


class RoundRobinScheduler(Scheduler):
    """Selects band 0, 1, 2, ..., num_bands - 1, 0, 1, ... forever.

    Ignores `state` entirely — `select_band` takes it only because the
    Scheduler interface requires it.
    """

    def __init__(self, num_bands: int):
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        self._num_bands = num_bands
        self._next_band = 0

    def reset(self) -> None:
        self._next_band = 0

    def select_band(self, state: State) -> Band:
        band = self._next_band
        self._next_band = (self._next_band + 1) % self._num_bands
        return band

    def update(self, observation: Observation, reward: float) -> None:
        pass  # non-learning baseline: nothing to react to
