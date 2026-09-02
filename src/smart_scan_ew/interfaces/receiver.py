"""Receiver contract.

Tunes to a band and turns whatever RFEnvironment.sense() returns into a
receiver-visible Observation. Phase 0 defines the shape only — no real
signal processing or detection logic here yet.
"""

from abc import ABC, abstractmethod

from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.interfaces.observation import Band, Observation


class Receiver(ABC):
    """Abstract base class for the receiver model."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal receiver state (e.g. currently tuned band)."""
        raise NotImplementedError

    @abstractmethod
    def tune(self, band: Band) -> None:
        """Select which band the receiver is currently looking at."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, environment: RFEnvironment, t: float) -> Observation:
        """Observe the currently tuned band and return an Observation.

        Implementations should call `environment.sense(...)` — and only
        that method on `environment` — to obtain the raw signal, then turn
        it into a receiver-visible Observation. Must never call
        `environment.get_ground_truth()`.
        """
        raise NotImplementedError
