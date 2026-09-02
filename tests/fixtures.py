"""Minimal no-op implementations of each interface, used ONLY by tests to
verify the contracts are importable, instantiable, and wireable together.

These are not real modules and must never be imported by application code —
they don't simulate RF, don't detect anything, and don't learn anything.
They exist purely to exercise the interface shapes defined in
src/smart_scan_ew/interfaces/.
"""

from typing import Any

from smart_scan_ew.interfaces import (
    Band,
    Evaluator,
    Observation,
    Receiver,
    RFEnvironment,
    Scheduler,
    State,
)


class NullEnvironment(RFEnvironment):
    """No-op environment: fixed, trivial ground truth, no real physics."""

    def __init__(self) -> None:
        self._t = 0.0
        self._reset_count = 0

    def reset(self, seed: int | None = None) -> None:
        self._t = 0.0
        self._reset_count += 1

    def step(self, dt: float = 1.0) -> None:
        self._t += dt

    def sense(self, band: Band, t: float) -> Any:
        # Always "nothing detected" — this is a contract fixture, not a
        # simulator.
        return None

    def get_ground_truth(self) -> Any:
        return {"t": self._t, "reset_count": self._reset_count}


class NullReceiver(Receiver):
    """No-op receiver: always reports "not detected"."""

    def __init__(self) -> None:
        self._band: Band | None = None

    def reset(self) -> None:
        self._band = None

    def tune(self, band: Band) -> None:
        self._band = band

    def observe(self, environment: RFEnvironment, t: float) -> Observation:
        environment.sense(self._band, t)  # only allowed environment call
        return Observation(time=t, band=self._band, detected=False)


class NullState(State):
    """No-op belief: just counts how many observations it has seen."""

    def __init__(self) -> None:
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def update(self, observation: Observation) -> None:
        self._count += 1

    def get_features(self) -> Any:
        return {"observation_count": self._count}


class NullScheduler(Scheduler):
    """No-op scheduler: always selects band 0. Never touches an
    RFEnvironment — only ever sees State/Observation/reward."""

    def __init__(self) -> None:
        self._updates = 0

    def reset(self) -> None:
        self._updates = 0

    def select_band(self, state: State) -> Band:
        state.get_features()  # exercise the contract; ignore the result
        return 0

    def update(self, observation: Observation, reward: float) -> None:
        self._updates += 1


class NullEvaluator(Evaluator):
    """No-op evaluator: wires the other fixtures together for a few steps
    without computing any real metrics."""

    def run_experiment(
        self,
        environment: RFEnvironment,
        receiver: Receiver,
        scheduler: Scheduler,
        state: State,
        num_steps: int,
    ) -> Any:
        history: list[Observation] = []
        for step in range(num_steps):
            environment.step()
            band = scheduler.select_band(state)
            receiver.tune(band)
            observation = receiver.observe(environment, t=float(step))
            state.update(observation)
            scheduler.update(observation, reward=0.0)
            history.append(observation)
        return history

    def compute_metrics(self, run_result: Any) -> dict:
        # Intentionally minimal and honest: only reports what was actually
        # recorded, no fabricated performance numbers.
        return {"num_observations": len(run_result)}
