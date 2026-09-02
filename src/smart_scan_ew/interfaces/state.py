"""State / belief contract.

Maintains whatever representation the Scheduler needs, derived strictly
from the history of Observations it has been given. Must never be
constructed from, or updated with, ground-truth information.
"""

from abc import ABC, abstractmethod
from typing import Any

from smart_scan_ew.interfaces.observation import Observation


class State(ABC):
    """Abstract base class for the observation-derived belief/state."""

    @abstractmethod
    def reset(self) -> None:
        """Clear any accumulated belief."""
        raise NotImplementedError

    @abstractmethod
    def update(self, observation: Observation) -> None:
        """Incorporate a new observation into the belief."""
        raise NotImplementedError

    @abstractmethod
    def get_features(self) -> Any:
        """Return the representation the Scheduler should decide from.

        The shape (dict, vector, custom object, ...) is deliberately
        unspecified in Phase 0 — see ARCHITECTURE.md.
        """
        raise NotImplementedError
