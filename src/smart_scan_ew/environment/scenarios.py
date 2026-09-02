"""Convenience scenario builders.

`default_scenario()` exists for demos/tests/quick starts. Real experiments
should generally construct their own explicit `list[EmitterSpec]` instead —
`SimpleRFEnvironment` accepts either.
"""

from smart_scan_ew.environment.emitters import EmitterSpec


def default_scenario(num_bands: int = 5) -> list[EmitterSpec]:
    """One emitter of each of the three Phase 1 types.

    Requires `num_bands >= 3` (the emitters below reference band ids 0, 1,
    and 2). Values are illustrative, not derived from any real scenario.
    """
    if num_bands < 3:
        raise ValueError(
            f"default_scenario() needs num_bands >= 3, got {num_bands}"
        )

    return [
        EmitterSpec(
            emitter_id="cw-1",
            kind="cw",
            band_id=0,
            power=5.0,
        ),
        EmitterSpec(
            emitter_id="hopper-1",
            kind="hopping",
            band_id=1,
            power=6.0,
            hop_interval=5.0,
            hop_bands=tuple(range(num_bands)),
        ),
        EmitterSpec(
            emitter_id="pulsed-1",
            kind="pulsed",
            band_id=2,
            power=4.0,
            period=10.0,
            pulse_width=3.0,
        ),
    ]
