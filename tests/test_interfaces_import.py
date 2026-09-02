"""Phase 0 sanity tests: the interfaces exist, import cleanly, and are
enforced as abstract (i.e. you can't accidentally instantiate a stub that
does nothing)."""

import inspect

import pytest

from smart_scan_ew.interfaces import (
    Band,
    Evaluator,
    Observation,
    Receiver,
    RFEnvironment,
    Scheduler,
    State,
)


@pytest.mark.parametrize(
    "cls", [RFEnvironment, Receiver, State, Scheduler, Evaluator]
)
def test_interface_is_abstract(cls):
    assert inspect.isabstract(cls)
    with pytest.raises(TypeError):
        cls()  # cannot instantiate an ABC with unimplemented methods


def test_observation_is_a_plain_data_object():
    obs = Observation(time=0.0, band=1, detected=True)
    assert obs.time == 0.0
    assert obs.band == 1
    assert obs.detected is True
    assert obs.info == {}


def test_observation_is_frozen():
    obs = Observation(time=0.0, band=1, detected=False)
    with pytest.raises(Exception):
        obs.detected = True  # frozen dataclass must reject mutation


def test_band_is_just_a_type_alias():
    # Band is intentionally an opaque, hashable-value alias in Phase 0.
    assert Band is not None
