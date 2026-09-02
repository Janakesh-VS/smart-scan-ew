"""Experiment configuration.

Central place for parameters that must stay configurable rather than
hard-coded (CLAUDE.md rule 7 / PROJECT_CONTRACT.md rule 7). This is a
Phase 0 placeholder: field names and defaults are provisional and expected
to grow as real modules (environment, receiver, scheduler) are implemented.

Nothing in this file performs simulation, scheduling, or evaluation — it
only describes parameters that those future modules will consume.
"""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Provisional experiment parameters.

    All values are placeholders for Phase 0. None of them are used by any
    simulation logic yet, since no simulation logic exists yet.
    """

    num_bands: int = 8
    """Number of RF bands the receiver can choose between."""

    time_step: float = 1.0
    """Simulation time advanced per RFEnvironment.step() call."""

    episode_length: int = 100
    """Number of steps in one evaluation episode."""

    random_seed: int | None = None
    """Seed passed to RFEnvironment.reset() for reproducibility."""

    extra: dict = field(default_factory=dict)
    """Escape hatch for module-specific parameters not yet promoted to a
    named field. Prefer adding a named field once a parameter is actually
    used by real code."""
