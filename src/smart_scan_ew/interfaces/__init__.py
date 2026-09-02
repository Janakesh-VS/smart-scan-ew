"""Abstract interfaces / contracts for smart_scan_ew.

Every concrete module (real RF environment, real receiver, real scheduler,
real state/belief, real evaluator) implements one of these. Concrete
implementations should import from here, not from each other's modules —
see ARCHITECTURE.md for the allowed data-flow and CLAUDE.md rule 2/6.
"""

from smart_scan_ew.interfaces.observation import Observation, Band
from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.interfaces.receiver import Receiver
from smart_scan_ew.interfaces.state import State
from smart_scan_ew.interfaces.scheduler import Scheduler
from smart_scan_ew.interfaces.evaluator import Evaluator

__all__ = [
    "Band",
    "Observation",
    "RFEnvironment",
    "Receiver",
    "State",
    "Scheduler",
    "Evaluator",
]
