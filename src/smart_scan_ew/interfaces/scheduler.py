"""Scheduler contract.

Decides which band to observe next. This is the interface with the
strictest rule attached to it (CLAUDE.md rule 4 / PROJECT_CONTRACT.md
rule 4): a Scheduler implementation must NEVER be given, store, or use a
reference to an RFEnvironment or its ground truth. It may only see a
State and, via `update`, an Observation and a scalar reward.
"""

from abc import ABC, abstractmethod

from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.state import State


class Scheduler(ABC):
    """Abstract base class for scanning strategies (classical or learned).

    Note: no method on this interface accepts an RFEnvironment. That is
    intentional and must not change without revisiting
    PROJECT_CONTRACT.md rule 4.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal scheduler state (e.g. learned parameters'
        episode-local counters — not the parameters themselves)."""
        raise NotImplementedError

    @abstractmethod
    def select_band(self, state: State) -> Band:
        """Choose the next band to observe, using only `state`."""
        raise NotImplementedError

    @abstractmethod
    def update(self, observation: Observation, reward: float) -> None:
        """React to / learn from the outcome of the last selection."""
        raise NotImplementedError
