"""The Observation contract — the only thing that crosses from the sensing
side (Receiver) to the belief/decision side (State, Scheduler).

Kept deliberately minimal in Phase 0. See ARCHITECTURE.md for the full
rationale, especially why `info` exists and what must never go in it.
"""

from dataclasses import dataclass, field
from typing import Any, Hashable

# Opaque placeholder type. Anything hashable/comparable can identify a band
# for now (e.g. a plain int index). Phase 1 may replace this with a richer
# type without breaking callers, as long as it stays hashable.
Band = Hashable


@dataclass(frozen=True)
class Observation:
    """A single, receiver-visible observation.

    This must contain only information a real receiver could plausibly
    know. It must never carry ground-truth-only information (true emitter
    identity, exact emitter position, etc.) — that stays inside
    RFEnvironment and is only readable by an Evaluator via
    `RFEnvironment.get_ground_truth()`.
    """

    time: float
    band: Band
    detected: bool
    info: dict[str, Any] = field(default_factory=dict)
