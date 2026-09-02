"""Reproducibility strategy: deterministic, independent sub-seeds derived
from one master seed.

See ARCHITECTURE.md's "Phase 3" section for the full rationale. The short
version: SimpleRFEnvironment, SimpleReceiver, and RandomScheduler already
each own an independent random.Random instance (Phase 1/2 decisions) — no
component's random draws affect any other's. `derive_seeds` builds on
that by giving a fixed, documented way to turn one master seed into
several independent, reproducible sub-seeds, so that swapping which
scheduler is under test never changes the environment's or receiver's own
random sequence.
"""

import random


def derive_seeds(master_seed: int | None, count: int) -> tuple[int | None, ...]:
    """Derive `count` independent, reproducible sub-seeds from one master
    seed.

    Fixed role order used throughout Phase 3: index 0 = environment,
    index 1 = receiver, index 2 = scheduler (if stochastic). Calling this
    twice with the same `master_seed` and `count` always returns the same
    tuple.

    If `master_seed is None`, returns `(None,) * count` — no derivation is
    meaningful for a non-reproducible run; every component falls back to
    seeding itself from OS entropy independently.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if master_seed is None:
        return (None,) * count

    # A single random.Random used ONLY for this derivation step, then
    # discarded — never reused for anything else, so it can't become an
    # accidental shared-RNG dependency between components.
    rng = random.Random(master_seed)
    return tuple(rng.getrandbits(32) for _ in range(count))
