"""Internal band metadata.

`Band` (the public, cross-module type from `interfaces.observation`) stays
a plain `int` index everywhere outside this module. `BandSpec` attaches
frequency metadata to each index and is used only inside
`SimpleRFEnvironment` — Receiver/State/Scheduler never see it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BandSpec:
    """Frequency metadata for one band index.

    A simplification, not a real band plan: bands are assumed contiguous
    and non-overlapping, all of equal width.
    """

    band_id: int
    center_frequency: float
    """Hz."""
    bandwidth: float
    """Hz."""


def build_band_table(
    num_bands: int, start_frequency_hz: float, band_width_hz: float
) -> tuple[BandSpec, ...]:
    """Build `num_bands` contiguous, equal-width BandSpecs.

    Band `i` is centered at `start_frequency_hz + i * band_width_hz`.
    """
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}")
    if band_width_hz <= 0:
        raise ValueError(f"band_width_hz must be positive, got {band_width_hz}")

    return tuple(
        BandSpec(
            band_id=i,
            center_frequency=start_frequency_hz + i * band_width_hz,
            bandwidth=band_width_hz,
        )
        for i in range(num_bands)
    )
