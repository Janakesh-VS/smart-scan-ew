"""Phase 1 demo: run a small 5-band scenario and print what actually
happened, step by step.

This is NOT an evaluator and computes NO metrics or scores — that's out
of scope until Phase 3 (see PROJECT_CONTRACT.md). It exists only to show
the environment/receiver wired together and producing real, non-fabricated
observations. The scheduling here (round-robin band selection) is a
throwaway loop for demo purposes, not a Scheduler implementation — that's
Phase 2.

Run with: python examples/phase1_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smart_scan_ew.environment import SimpleRFEnvironment, default_scenario
from smart_scan_ew.receiver import SimpleReceiver

NUM_BANDS = 5
NUM_STEPS = 20


def main() -> None:
    env = SimpleRFEnvironment(
        emitter_specs=default_scenario(NUM_BANDS),
        num_bands=NUM_BANDS,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    env.reset(seed=0)

    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=1.0, seed=0)
    receiver.reset()

    print(f"Scenario: {NUM_BANDS} bands, "
          f"emitters={[s.emitter_id for s in default_scenario(NUM_BANDS)]}")
    print(f"{'t':>4}  {'band':>4}  {'measured_power':>15}  {'detected':>8}  "
          f"{'ground_truth (evaluator-only, shown here for demo)'}")

    for step in range(NUM_STEPS):
        env.step(dt=1.0)
        t = step + 1.0

        # Simple round-robin band selection for demo purposes ONLY — this
        # is not a Scheduler implementation (Phase 2).
        band = step % NUM_BANDS
        receiver.tune(band)
        obs = receiver.observe(env, t=t)

        # Ground truth is fetched here only to print it alongside the
        # observation for demo/illustration. In a real evaluation harness
        # this call belongs to the Evaluator, never to code that also
        # drives the receiver/scheduler loop.
        truth = env.get_ground_truth()
        active_emitters = [
            f"{e.emitter_id}@band{e.band}" for e in truth.emitters if e.active
        ]

        print(
            f"{t:4.1f}  {band:4d}  {obs.info['measured_power']:15.3f}  "
            f"{str(obs.detected):>8}  {active_emitters}"
        )


if __name__ == "__main__":
    main()
