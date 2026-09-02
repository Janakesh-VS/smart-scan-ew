"""Experiment configuration.

Central place for parameters that must stay configurable rather than
hard-coded (CLAUDE.md rule 7 / PROJECT_CONTRACT.md rule 7).

Phase 0 defined the shape with placeholder fields only. Phase 1 adds the
band-table and noise/detection parameters that `SimpleRFEnvironment` and
`SimpleReceiver` actually consume — see ARCHITECTURE.md's "Phase 1"
section for the decisions behind these defaults.

Nothing in this file performs simulation, scheduling, or evaluation — it
only describes parameters that those modules consume.
"""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Experiment parameters shared across modules.

    Phase 0 fields (`num_bands`, `time_step`, `episode_length`,
    `random_seed`, `extra`) are unchanged. Phase 1 fields below configure
    the concrete `SimpleRFEnvironment` / `SimpleReceiver` implementations;
    none of them existed before Phase 1 since no simulation logic existed
    to consume them.
    """

    num_bands: int = 8
    """Number of RF bands the receiver can choose between."""

    time_step: float = 1.0
    """Simulation time advanced per RFEnvironment.step() call."""

    episode_length: int = 100
    """Number of steps in one evaluation episode."""

    random_seed: int | None = None
    """Seed passed to RFEnvironment.reset() for reproducibility. Governs
    all frequency-hopping band draws (Phase 1)."""

    extra: dict = field(default_factory=dict)
    """Escape hatch for module-specific parameters not yet promoted to a
    named field. Prefer adding a named field once a parameter is actually
    used by real code."""

    # --- Phase 1: band table -------------------------------------------------
    band_start_frequency_hz: float = 2.4e9
    """Center frequency of band 0. Placeholder value (2.4 GHz, ISM band) —
    not tied to any specific real-world scenario."""

    band_width_hz: float = 20e6
    """Width of each band, and the spacing between consecutive band center
    frequencies (bands are contiguous and non-overlapping)."""

    # --- Phase 1: noise / detection model -------------------------------------
    noise_std: float = 1.0
    """Standard deviation of the receiver's additive Gaussian noise. See
    ARCHITECTURE.md — this is an explicitly simplified model, not a
    physical noise-figure calculation."""

    detection_threshold: float = 3.0
    """Measured-power threshold above which SimpleReceiver reports
    detected=True."""

    receiver_seed: int | None = None
    """Seed for SimpleReceiver's own noise RNG. Independent of
    `random_seed` (which governs the environment/emitters) by design —
    see ARCHITECTURE.md's RNG ownership section."""

    # --- Phase 1: emitters -----------------------------------------------------
    emitters: list | None = None
    """Explicit list of EmitterSpec objects for the scenario. If None,
    callers building a SimpleRFEnvironment fall back to
    `environment.scenarios.default_scenario()`. Kept as None here (rather
    than importing EmitterSpec into this module) to avoid a dependency
    from config.py onto the environment package."""
