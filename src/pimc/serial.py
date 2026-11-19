"""Serial point-counting functions and the serial estimators.

The counting logic lives in two top-level functions, ``count_pure`` and
``count_numpy``. They are deliberately *not* methods:

1. ``multiprocessing.Pool`` can only send top-level functions to worker
   processes (methods would drag ``self`` -- and its non-picklable pool -- along).
2. The serial and parallel estimators then run the *identical* counting code,
   so any timing difference comes from parallelisation alone.
"""

import random
import numpy as np
from pimc.base import PiEstimator


# Default number of points NumPy generates per block. It bounds memory use
# (two float64 arrays of this length) and, as the experiments show, strongly
# affects cache behaviour and therefore parallel scalability.
CHUNK = 1_000_000

def count_pure(n: int, seed_seq: np.random.SeedSequence) -> int:
    """Count quarter-circle hits with a plain Python loop (slow baseline).

    Args:
        n: Number of random points to sample.
        seed_seq: ``SeedSequence`` the generator is seeded from. For the
            parallel estimator each worker receives its own spawned child
            sequence, which keeps the random streams independent.

    Returns:
        Number of points with ``x**2 + y**2 <= 1``.

    Note:
        ``random.Random`` (Mersenne Twister) is seeded with a single 32-bit
        integer derived from ``seed_seq``.
    """
    rng = random.Random(int(seed_seq.generate_state(1)[0]))
    inside = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def count_numpy(n: int, seed_seq: np.random.SeedSequence, chunk: int = CHUNK) -> int:
    """Count quarter-circle hits with vectorised NumPy operations.

    Points are generated in blocks of at most ``chunk`` points so that memory
    stays bounded even for very large ``n``.

    Args:
        n: Number of random points to sample.
        seed_seq: ``SeedSequence`` used to create a ``default_rng`` (PCG64)
            generator.
        chunk: Maximum number of points generated per block. Smaller blocks
            keep the working set in the CPU caches; too small a block makes
            Python loop overhead dominate.

    Returns:
        Number of points with ``x**2 + y**2 <= 1``.

    Note:
        Changing ``chunk`` changes how the random stream is split into calls,
        so the resulting estimate may differ slightly. Both values are valid
        Monte Carlo estimates.
    """
    rng = np.random.default_rng(seed_seq)
    inside = 0
    remaining = n
    while remaining > 0:
        m = min(chunk, remaining)  # last block may be smaller than `chunk`
        x = rng.random(m)
        y = rng.random(m)
        inside += int(np.count_nonzero(x * x + y * y <= 1.0))
        remaining -= m
    return inside


class PureSerialEstimator(PiEstimator):
    """Serial estimator using a pure-Python loop (the reference baseline)."""

    name = "pure_serial"

    def count_inside(self, n: int, seed: int) -> int:
        """Count hits with :func:`count_pure`, seeded from ``seed``."""
        return count_pure(n, np.random.SeedSequence(seed))


class NumpySerialEstimator(PiEstimator):
    """Serial estimator using vectorised NumPy (the fast serial baseline).

    Args:
        chunk: Block size passed to :func:`count_numpy`. The serial baseline
            should use the same chunk as the parallel estimator it is compared
            against, otherwise the speedup would mix in cache effects.
    """

    name = "numpy_serial"

    def __init__(self, chunk: int = CHUNK):
        self.chunk = chunk

    def count_inside(self, n: int, seed: int) -> int:
        """Count hits with :func:`count_numpy`, seeded from ``seed``."""
        return count_numpy(n, np.random.SeedSequence(seed), self.chunk)