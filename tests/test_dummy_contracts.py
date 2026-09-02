"""Phase 0 sanity tests using the no-op fixtures in tests/fixtures.py.

These verify the interfaces are actually usable together end-to-end (can be
instantiated, wired, and run for a few steps) — without any real RF
simulation, detection, or learning logic. They also act as a smoke test for
the ground-truth isolation rule: the scheduler fixture is never given the
environment.
"""

from tests.fixtures import (
    NullEnvironment,
    NullEvaluator,
    NullReceiver,
    NullScheduler,
    NullState,
)


def test_fixtures_satisfy_their_interfaces():
    # Instantiating at all proves every abstract method was implemented.
    NullEnvironment()
    NullReceiver()
    NullState()
    NullScheduler()
    NullEvaluator()


def test_environment_reset_and_step():
    env = NullEnvironment()
    env.reset(seed=42)
    env.step(dt=1.0)
    truth = env.get_ground_truth()
    assert truth["t"] == 1.0
    assert truth["reset_count"] == 1


def test_receiver_tune_and_observe_only_uses_sense():
    env = NullEnvironment()
    env.reset()
    receiver = NullReceiver()
    receiver.tune(band=3)
    obs = receiver.observe(env, t=0.0)
    assert obs.band == 3
    assert obs.detected is False


def test_state_accumulates_observation_count():
    state = NullState()
    receiver = NullReceiver()
    env = NullEnvironment()
    env.reset()
    receiver.tune(0)
    for i in range(3):
        obs = receiver.observe(env, t=float(i))
        state.update(obs)
    assert state.get_features() == {"observation_count": 3}


def test_scheduler_never_receives_the_environment():
    # The Scheduler interface's methods don't accept an RFEnvironment at
    # all — this test documents/pins that by construction: select_band and
    # update only take (state) and (observation, reward) respectively.
    import inspect

    from smart_scan_ew.interfaces import Scheduler

    select_band_params = inspect.signature(Scheduler.select_band).parameters
    update_params = inspect.signature(Scheduler.update).parameters
    assert "environment" not in select_band_params
    assert "environment" not in update_params


def test_end_to_end_run_with_null_fixtures():
    env = NullEnvironment()
    env.reset(seed=0)
    receiver = NullReceiver()
    state = NullState()
    scheduler = NullScheduler()
    evaluator = NullEvaluator()

    history = evaluator.run_experiment(
        environment=env,
        receiver=receiver,
        scheduler=scheduler,
        state=state,
        num_steps=5,
    )
    assert len(history) == 5

    metrics = evaluator.compute_metrics(history)
    # Honest, non-fabricated metric: just what was actually recorded.
    assert metrics == {"num_observations": 5}
