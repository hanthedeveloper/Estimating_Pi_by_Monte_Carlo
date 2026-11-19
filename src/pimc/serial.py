import random
import numpy as np
from pimc.base import PiEstimator


CHUNK = 1_000_000

def count_pure(n: int, seed_seq: np.random.SeedSequence) -> int:
    rng = random.Random(int(seed_seq.generate_state(1)[0]))
    inside = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def count_numpy(n: int, seed_seq: np.random.SeedSequence, chunk: int = CHUNK) -> int:
    rng = np.random.default_rng(seed_seq)
    inside = 0
    remaining = n
    while remaining > 0:
        m = min(chunk, remaining)
        x = rng.random(m)
        y = rng.random(m)
        inside += int(np.count_nonzero(x * x + y * y <= 1.0))
        remaining -= m
    return inside


class PureSerialEstimator(PiEstimator):
    name = "pure_serial"
    def count_inside(self, n: int, seed: int) -> int:
        return count_pure(n, np.random.SeedSequence(seed))


class NumpySerialEstimator(PiEstimator):
    name = "numpy_serial"

    def __init__(self, chunk: int = CHUNK):
        self.chunk = chunk

    def count_inside(self, n: int, seed: int) -> int:
        return count_numpy(n, np.random.SeedSequence(seed), self.chunk)