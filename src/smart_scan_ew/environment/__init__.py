"""Concrete RF environment implementation (Phase 1).

Exposes `SimpleRFEnvironment` (implements `interfaces.RFEnvironment`),
`EmitterSpec` (config for one emitter), and `default_scenario()` (a
convenience emitter list). `BandSpec` and the emitter behavior classes
(`ContinuousWaveEmitter`, `PulsedEmitter`, `FrequencyHoppingEmitter`) are
internal to this package — other modules should not import them directly;
they interact with bands only via the plain `int` `Band` type and with
emitters only indirectly, through `RFEnvironment.sense()` /
`get_ground_truth()`.
"""

from smart_scan_ew.environment.bands import BandSpec, build_band_table
from smart_scan_ew.environment.emitters import (
    ContinuousWaveEmitter,
    EmitterSpec,
    FrequencyHoppingEmitter,
    PulsedEmitter,
)
from smart_scan_ew.environment.rf_environment import (
    GroundTruthSnapshot,
    SimpleRFEnvironment,
)
from smart_scan_ew.environment.scenarios import default_scenario

__all__ = [
    "BandSpec",
    "build_band_table",
    "EmitterSpec",
    "ContinuousWaveEmitter",
    "PulsedEmitter",
    "FrequencyHoppingEmitter",
    "GroundTruthSnapshot",
    "SimpleRFEnvironment",
    "default_scenario",
]
