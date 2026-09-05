"""Ground-truth isolation tests for Phase 4's AdaptiveUcbScheduler.

Two layers, mirroring the Phase 1 (tests/test_ground_truth_isolation.py)
and Phase 2 (tests/test_phase2_ground_truth_isolation.py) convention:

1. A structural test on the scheduler module's source: it must not import
   `smart_scan_ew.environment` or anything ground-truth-shaped.
2. A runtime spy test: the complete environment -> receiver -> state ->
   scheduler loop, with AdaptiveUcbScheduler as the scheduler, must never
   call `get_ground_truth()`.
"""

import ast
import inspect

from smart_scan_ew.environment import EmitterSpec, SimpleRFEnvironment
from smart_scan_ew.interfaces import RFEnvironment
from smart_scan_ew.receiver import SimpleReceiver
from smart_scan_ew.scheduler import adaptive_ucb
from smart_scan_ew.scheduler.adaptive_ucb import AdaptiveUcbScheduler
from smart_scan_ew.state import SimpleBeliefState

FORBIDDEN_MODULE_PREFIXES = (
    "smart_scan_ew.environment",
    "smart_scan_ew.evaluator",
)
FORBIDDEN_NAMES = {
    "RFEnvironment",
    "EmitterSpec",
    "GroundTruthSnapshot",
    "get_ground_truth",
}


def test_adaptive_ucb_module_does_not_import_environment_or_evaluator():
    """Structural check on the actual source text: no import statement in
    adaptive_ucb.py references environment/ or evaluator/ (the only two
    packages that can see ground truth)."""
    source = inspect.getsource(adaptive_ucb)
    tree = ast.parse(source)

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    for module_name in imported_modules:
        assert not module_name.startswith(FORBIDDEN_MODULE_PREFIXES), (
            f"adaptive_ucb.py imports {module_name!r} -- a Scheduler must "
            "never import environment/ or evaluator/ (CLAUDE.md rule 4/5)."
        )


def test_adaptive_ucb_source_never_names_ground_truth_symbols():
    """Belt-and-braces textual check: none of the ground-truth-only
    identifiers appear anywhere in the module source at all (not just in
    import statements) -- guards against e.g. a local duck-typed
    reimplementation that reaches for the same concepts under an alias."""
    source = inspect.getsource(adaptive_ucb)
    for name in FORBIDDEN_NAMES:
        assert name not in source, (
            f"adaptive_ucb.py's source contains {name!r}, which is "
            "ground-truth-shaped and must never appear in a Scheduler."
        )


class _GroundTruthSpy(RFEnvironment):
    """Wraps a real RFEnvironment and records calls to get_ground_truth()
    and sense(), without changing behavior. (Duplicated from the Phase
    1/2 test modules rather than imported, so each isolation test file
    stays self-contained -- same convention those files already use.)
    """

    def __init__(self, wrapped: RFEnvironment):
        self._wrapped = wrapped
        self.sense_call_count = 0
        self.ground_truth_call_count = 0

    def reset(self, seed=None) -> None:
        self._wrapped.reset(seed=seed)

    def step(self, dt: float = 1.0) -> None:
        self._wrapped.step(dt=dt)

    def sense(self, band, t):
        self.sense_call_count += 1
        return self._wrapped.sense(band, t)

    def get_ground_truth(self):
        self.ground_truth_call_count += 1
        return self._wrapped.get_ground_truth()


NUM_BANDS = 5
NUM_STEPS = 25


def test_full_phase4_loop_never_touches_ground_truth():
    real_env = SimpleRFEnvironment(
        emitter_specs=[
            EmitterSpec(emitter_id="cw-1", kind="cw", band_id=0, power=5.0),
            EmitterSpec(
                emitter_id="hopper-1", kind="hopping", band_id=1, power=6.0,
                hop_interval=3.0, hop_bands=(0, 1, 2, 3, 4),
            ),
        ],
        num_bands=NUM_BANDS,
        band_start_frequency_hz=2.4e9,
        band_width_hz=20e6,
    )
    spy_env = _GroundTruthSpy(real_env)
    spy_env.reset(seed=0)

    receiver = SimpleReceiver(detection_threshold=3.0, noise_std=1.0, seed=0)
    receiver.reset()

    state = SimpleBeliefState(num_bands=NUM_BANDS)
    state.reset()

    scheduler = AdaptiveUcbScheduler(num_bands=NUM_BANDS, gamma=0.95, c=1.0)
    scheduler.reset()

    for step in range(NUM_STEPS):
        spy_env.step(dt=1.0)
        t = float(step + 1)

        band = scheduler.select_band(state)
        assert 0 <= band < NUM_BANDS

        receiver.tune(band)
        observation = receiver.observe(spy_env, t=t)

        state.update(observation)
        scheduler.update(observation, reward=1.0 if observation.detected else 0.0)

    assert spy_env.sense_call_count == NUM_STEPS
    assert spy_env.ground_truth_call_count == 0, (
        "Ground truth was accessed somewhere in the Phase 4 loop -- this "
        "violates the ground-truth isolation rule (CLAUDE.md rule 4/5)."
    )
