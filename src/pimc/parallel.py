from multiprocessing import Pool

import numpy as np

from pimc.base import PiEstimator
from pimc.serial import count_numpy, count_pure
from pimc.serial import count_numpy, count_pure, CHUNK


_METHODS = {"pure": count_pure, "numpy": count_numpy}

def _worker(args):
    method, n, seed_seq, chunk = args
    if method == "numpy":
        return count_numpy(n, seed_seq, chunk)
    return count_pure(n, seed_seq)


class ParallelEstimator(PiEstimator):
    def __init__(self, workers: int, method: str = "numpy", chunk: int = CHUNK):
        if method not in _METHODS:
            raise ValueError(f"method 'pure' ya da 'numpy' olmalı, gelen: {method}")
        self.workers = workers
        self.method = method
        self.chunk = chunk
        self.name = f"parallel_{method}"
        self._pool = None

    def _worker_count(self) -> int:
        return self.workers

    def _prepare(self) -> None:
        self._pool = Pool(self.workers)

    def _cleanup(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def count_inside(self, n: int, seed: int) -> int:
        seeds = np.random.SeedSequence(seed).spawn(self.workers)
        base, extra = divmod(n, self.workers)
        chunks = [base + (1 if i < extra else 0) for i in range(self.workers)]
        tasks = [(self.method, c, s, self.chunk) for c, s in zip(chunks, seeds)]
        return sum(self._pool.map(_worker, tasks))