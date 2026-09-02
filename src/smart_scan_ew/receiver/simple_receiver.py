"""SimpleReceiver: a real, but intentionally simple, Receiver.

Detection model (see ARCHITECTURE.md "Phase 1 — Noise / detection model"
for the full rationale): the environment's `sense()` value is treated as
the true received power; independent Gaussian noise (owned, seeded RNG,
never global `random` state) is added, and the result is thresholded.
This is an explicitly simplified simulation model, not a physically
complete RF/noise-figure model, and can produce both:

- misses (a real transmission is present, but noise pulls the reading
  below `detection_threshold`), and
- false alarms (nothing is present, but noise pushes the reading above
  `detection_threshold`).
"""

import random

from smart_scan_ew.interfaces.environment import RFEnvironment
from smart_scan_ew.interfaces.observation import Band, Observation
from smart_scan_ew.interfaces.receiver import Receiver


class SimpleReceiver(Receiver):
    """Tunes to one band at a time; observing it adds seeded Gaussian
    noise to the environment's `sense()` value and thresholds the result.
    """

    def __init__(
        self,
        detection_threshold: float,
        noise_std: float,
        seed: int | None = None,
    ):
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        self._threshold = detection_threshold
        self._noise_std = noise_std
        self._seed = seed
        self._rng = random.Random(seed)
        self._tuned_band: Band | None = None

    def reset(self) -> None:
        # Receiver.reset() takes no arguments by interface contract, so the
        # noise seed is fixed at construction time and re-applied here —
        # this makes repeated reset()+run sequences reproducible without
        # needing a seed parameter on reset() itself.
        self._tuned_band = None
        self._rng = random.Random(self._seed)

    def tune(self, band: Band) -> None:
        self._tuned_band = band

    def observe(self, environment: RFEnvironment, t: float) -> Observation:
        if self._tuned_band is None:
            raise RuntimeError("SimpleReceiver.observe() called before tune()")

        # The only environment call this receiver ever makes — see
        # ARCHITECTURE.md "Ground-truth isolation" and
        # tests/test_ground_truth_isolation.py.
        true_power = environment.sense(self._tuned_band, t)

        noise = self._rng.gauss(0.0, self._noise_std)
        measured_power = true_power + noise
        detected = measured_power > self._threshold

        return Observation(
            time=t,
            band=self._tuned_band,
            detected=detected,
            info={"measured_power": measured_power},
        )
