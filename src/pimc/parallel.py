"""Process-based parallel estimator built on ``multiprocessing.Pool``.

Processes are used instead of threads because the Global Interpreter Lock
(GIL) prevents threads from running CPU-bound Python code in parallel.

Algorithm: the total ``n`` is split into ``p`` nearly equal parts, each worker
counts hits in its own part using an independent random stream, and the partial
counts are summed. The samples are independent, so the problem is
"embarrassingly parallel" -- the only communication is sending the task in and
the integer count back.
"""

from multiprocessing import Pool

import numpy as np

from pimc.base import PiEstimator
from pimc.serial import count_numpy, count_pure, CHUNK


# Valid values for the ``method`` argument of ParallelEstimator.
_METHODS = {"pure": count_pure, "numpy": count_numpy}

def _worker(args):
    """Entry point executed inside each worker process.

    It is a top-level function so that it can be pickled and sent to the
    workers. Only plain values (a string, ints and a ``SeedSequence``) are
    passed in -- never ``self`` -- to keep the pickling overhead small and to
    avoid sending the (non-picklable) Pool object.

    Args:
        args: Tuple ``(method, n, seed_seq, chunk)`` where ``method`` is
            ``"pure"`` or ``"numpy"``, ``n`` is this worker's share of points,
            ``seed_seq`` is its independent seed sequence and ``chunk`` is the
            NumPy block size (ignored by the pure-Python method).

    Returns:
        The number of hits counted by this worker.
    """
    method, n, seed_seq, chunk = args
    if method == "numpy":
        return count_numpy(n, seed_seq, chunk)
    return count_pure(n, seed_seq)


class ParallelEstimator(PiEstimator):
    """Parallel pi estimator that distributes the sampling over ``p`` processes.

    Args:
        workers: Number of worker processes ``p``.
        method: ``"numpy"`` (vectorised, default) or ``"pure"`` (Python loop).
            Using the same counting code as the serial estimators keeps the
            comparison fair.
        chunk: NumPy block size, forwarded to :func:`count_numpy`.

    Raises:
        ValueError: If ``method`` is not ``"pure"`` or ``"numpy"``.

    Note:
        The pool is created in :meth:`_prepare` and closed in
        :meth:`_cleanup`, both outside the timed section. Pool start-up time is
        therefore **not** included in the reported ``elapsed``.
    """

    def __init__(self, workers: int, method: str = "numpy", chunk: int = CHUNK):
        if method not in _METHODS:
            raise ValueError(f"method must be 'pure' or 'numpy', got: {method}")
        self.workers = workers
        self.method = method
        self.chunk = chunk
        self.name = f"parallel_{method}"
        self._pool = None

    def _worker_count(self) -> int:
        """Hook override: report the configured number of workers."""
        return self.workers

    def _prepare(self) -> None:
        """Hook override: create the process pool before timing starts."""
        self._pool = Pool(self.workers)

    def _cleanup(self) -> None:
        """Hook override: close the pool and wait for the workers to exit."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def count_inside(self, n: int, seed: int) -> int:
        """Split ``n`` across the workers and sum their hit counts.

        Steps:
            1. ``SeedSequence(seed).spawn(p)`` creates ``p`` statistically
               independent child seeds. Giving every worker the same seed would
               make them generate identical points and ruin the estimate.
            2. ``divmod`` divides ``n`` into ``p`` parts; the remainder is
               spread one point each over the first workers, so part sizes
               differ by at most 1.
            3. ``Pool.map`` runs one task per worker and the partial counts
               are summed.

        Args:
            n: Total number of points over all workers.
            seed: Seed from which the per-worker seeds are derived.

        Returns:
            The total number of hits across all workers.
        """
        seeds = np.random.SeedSequence(seed).spawn(self.workers)
        base, extra = divmod(n, self.workers)
        chunks = [base + (1 if i < extra else 0) for i in range(self.workers)]
        tasks = [(self.method, c, s, self.chunk) for c, s in zip(chunks, seeds)]
        return sum(self._pool.map(_worker, tasks))