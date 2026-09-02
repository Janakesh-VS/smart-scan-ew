"""Emitter models: ground-truth behavior inside RFEnvironment.

Exactly three emitter types, per the project owner's Phase 1 approval:
`ContinuousWaveEmitter`, `PulsedEmitter`, `FrequencyHoppingEmitter`. No
chirp/swept emitter is implemented in Phase 1.

These classes are internal to the environment module — not part of the
cross-module `interfaces/` contracts. `RFEnvironment` is the only thing
that talks to them directly. Nothing here is imported by `Receiver`,
`State`, or `Scheduler` implementations.

All time-dependent behavior is advanced explicitly via `advance(t, rng)`,
called by `SimpleRFEnvironment.step()` — there is no background timer and
no emitter-owned RNG; the only randomness (frequency-hop band selection)
draws from the RNG instance the environment passes in, in a fixed emitter
order, so a given environment seed reproduces the whole run.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmitterSpec:
    """Configuration for one emitter. Data only — no behavior.

    Which fields matter depends on `kind`:
    - "cw": `band_id`, `power`.
    - "pulsed": `band_id`, `power`, `period`, `pulse_width`.
    - "hopping": `band_id` (initial band before the first hop), `power`,
      `hop_interval`, `hop_bands` (candidate bands to hop between; if
      None, defaults to every band in the environment's band table).
    """

    emitter_id: str
    kind: Literal["cw", "pulsed", "hopping"]
    band_id: int
    power: float
    period: float | None = None
    pulse_width: float | None = None
    hop_interval: float | None = None
    hop_bands: tuple[int, ...] | None = None


class Emitter(ABC):
    """Internal runtime behavior for one emitter. Not a cross-module
    interface — see module docstring."""

    def __init__(self, spec: EmitterSpec):
        self.spec = spec

    @abstractmethod
    def reset(self, rng: random.Random) -> None:
        """Reset any internal time-dependent state."""
        raise NotImplementedError

    @abstractmethod
    def advance(self, t: float, rng: random.Random) -> None:
        """Advance internal state to simulation time `t`."""
        raise NotImplementedError

    @abstractmethod
    def is_active(self, t: float) -> bool:
        """Whether the emitter is transmitting at time `t` (state must
        already have been advanced to `t` via `advance`)."""
        raise NotImplementedError

    @abstractmethod
    def current_band(self, t: float) -> int:
        """Which band the emitter is transmitting on at time `t` (state
        must already have been advanced to `t` via `advance`)."""
        raise NotImplementedError

    def current_power(self, t: float) -> float:
        """Transmit power at time `t`, or 0.0 if inactive."""
        return self.spec.power if self.is_active(t) else 0.0


class ContinuousWaveEmitter(Emitter):
    """Always on, fixed band, fixed power. The baseline case."""

    def reset(self, rng: random.Random) -> None:
        pass  # no internal state

    def advance(self, t: float, rng: random.Random) -> None:
        pass  # nothing changes over time

    def is_active(self, t: float) -> bool:
        return True

    def current_band(self, t: float) -> int:
        return self.spec.band_id


class PulsedEmitter(Emitter):
    """Fixed band, fixed power, but on/off on a duty cycle.

    Active whenever `t modulo period < pulse_width`. Fully deterministic
    given `t` — no internal state needed.
    """

    def __init__(self, spec: EmitterSpec):
        super().__init__(spec)
        if spec.period is None or spec.pulse_width is None:
            raise ValueError(
                "PulsedEmitter requires spec.period and spec.pulse_width"
            )
        if spec.period <= 0:
            raise ValueError(f"period must be positive, got {spec.period}")
        if not (0 < spec.pulse_width <= spec.period):
            raise ValueError(
                "pulse_width must be in (0, period], got "
                f"pulse_width={spec.pulse_width}, period={spec.period}"
            )

    def reset(self, rng: random.Random) -> None:
        pass  # no internal state

    def advance(self, t: float, rng: random.Random) -> None:
        pass  # deterministic function of t, nothing to advance

    def is_active(self, t: float) -> bool:
        return (t % self.spec.period) < self.spec.pulse_width

    def current_band(self, t: float) -> int:
        return self.spec.band_id


class FrequencyHoppingEmitter(Emitter):
    """Always on, fixed power, but its band changes every `hop_interval`.

    Band selection draws from the RNG passed into `advance()` — never an
    emitter-owned RNG — so the sequence is reproducible given the
    environment's seed and emitter ordering (see module docstring).
    """

    def __init__(self, spec: EmitterSpec, available_bands: tuple[int, ...]):
        super().__init__(spec)
        if spec.hop_interval is None:
            raise ValueError("FrequencyHoppingEmitter requires spec.hop_interval")
        if spec.hop_interval <= 0:
            raise ValueError(
                f"hop_interval must be positive, got {spec.hop_interval}"
            )
        self._candidate_bands = spec.hop_bands or available_bands
        if not self._candidate_bands:
            raise ValueError("FrequencyHoppingEmitter has no candidate bands")

        valid_bands = set(available_bands)
        invalid_bands = tuple(
            band for band in self._candidate_bands if band not in valid_bands
        )
        if invalid_bands:
            raise ValueError(
                "FrequencyHoppingEmitter hop_bands contains invalid band IDs: "
                f"{invalid_bands}; available bands are {available_bands}"
            )

        self._current_band = spec.band_id
        self._next_hop_time = spec.hop_interval

    def reset(self, rng: random.Random) -> None:
        self._current_band = self.spec.band_id
        self._next_hop_time = self.spec.hop_interval

    def advance(self, t: float, rng: random.Random) -> None:
        # A while-loop (not `if`) so a large `dt` that skips over several
        # hop boundaries in one step() call still lands on the correct
        # band, consuming one rng draw per boundary crossed.
        while t >= self._next_hop_time:
            self._current_band = rng.choice(self._candidate_bands)
            self._next_hop_time += self.spec.hop_interval

    def is_active(self, t: float) -> bool:
        return True

    def current_band(self, t: float) -> int:
        return self._current_band


def build_emitter(spec: EmitterSpec, available_bands: tuple[int, ...]) -> Emitter:
    """Factory: construct the right Emitter subclass for `spec.kind`."""
    if spec.kind == "cw":
        return ContinuousWaveEmitter(spec)
    if spec.kind == "pulsed":
        return PulsedEmitter(spec)
    if spec.kind == "hopping":
        return FrequencyHoppingEmitter(spec, available_bands)
    raise ValueError(f"Unknown emitter kind: {spec.kind!r}")
