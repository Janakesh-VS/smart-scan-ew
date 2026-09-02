"""Evaluator contract.

The only component allowed to see both an RFEnvironment's ground truth and
a Scheduler's decisions in the same place, for the sole purpose of scoring.
It must not feed ground truth to the scheduler/receiver during a run.

Phase 0 defines the shape only. No real experiment-running or metrics
logic exists yet — see PROJECT_CONTRACT.md rule 10: no fabricated results.
"""

from abc import ABC, abstractmethod
from typing import Any

from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.interfaces.receiver import Receiver
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.state import State


class Evaluator(ABC):
    """Abstract base class for running experiments and scoring them."""

    @abstractmethod
    def run_experiment(
        self,
        environment: RFEnvironment,
        receiver: Receiver,
        scheduler: Scheduler,
        state: State,
        num_steps: int,
    ) -> Any:
        """Run `num_steps` of environment/receiver/state/scheduler
        interaction and return a record of what happened, suitable for
        `compute_metrics`. This is the only place all four interfaces are
        wired together.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, run_result: Any) -> dict:
        """Turn a recorded run into real, computed metrics.

        Must never return placeholder or fabricated numbers. In Phase 0
        this method has no real implementation anywhere in the codebase.
        """
        raise NotImplementedError
