"""RFEnvironment contract.

Owns ground-truth world state (emitters, bands, timing). Nothing in this
file simulates anything yet — Phase 0 defines the shape only.

IMPORTANT (see CLAUDE.md rules 4-5 and ARCHITECTURE.md): `get_ground_truth`
exists so an Evaluator can score a run. A Scheduler implementation must
never be given a reference to an RFEnvironment, and a Receiver
implementation must only ever call `sense`, never `get_ground_truth`.
"""

from abc import ABC, abstractmethod
from typing import Any

from smart_scan_ew.interfaces.observation import Band


class RFEnvironment(ABC):
    """Abstract base class for the simulated RF world."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """(Re)initialize the environment's internal ground-truth state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float = 1.0) -> None:
        """Advance the internal world state by `dt`."""
        raise NotImplementedError

    @abstractmethod
    def sense(self, band: Band, t: float) -> Any:
        """Return the raw, physically observable signal at `band`/`t`.

        This is the only environment method a Receiver may call. It must
        not leak ground-truth information beyond what a real receiver
        tuned to `band` could plausibly measure.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self) -> Any:
        """Return the full true world state.

        By architectural contract, only Evaluator implementations may call
        this method, and only for scoring after (or independently of) a
        scheduler's decisions — never to influence a Scheduler or Receiver
        during a run. This is not enforced by the type system; it is
        enforced by code review and tests.
        """
        raise NotImplementedError
