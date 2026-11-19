"""Generic benchmark driver shared by all experiment scripts."""

from pimc.base import PiEstimator
from pimc.io_utils import append_row


def run_benchmark(estimators: list[PiEstimator], ns: list[int],
                  repeats: int, csv_path: str, base_seed: int = 1000) -> None:
    """Run every estimator for every ``n`` several times and log to CSV.

    The function only calls ``estimate()``, so it works for any
    :class:`PiEstimator` regardless of whether it is serial or parallel
    (polymorphism).

    Args:
        estimators: Estimators to measure, run one after another.
        ns: Sample sizes to test.
        repeats: Repetitions per (estimator, n) pair.
        csv_path: Output CSV; each measurement is appended immediately.
        base_seed: Repetition ``r`` uses seed ``base_seed + r``. Repetitions
            are therefore independent samples, yet the whole experiment is
            reproducible (timings vary, pi estimates do not).

    Note:
        A failing measurement is reported and skipped so that one error does
        not discard a long-running experiment.
    """
    for est in estimators:
        for n in ns:
            for rep in range(repeats):
                seed = base_seed + rep
                try:
                    result = est.estimate(n, seed=seed)
                except Exception as e:
                    print(f"ERROR: {est.name} n={n} rep={rep}: {e}")
                    continue
                append_row(csv_path, result, repeat=rep, seed=seed)
                print(f"{result.estimator:<20} workers={result.workers:<3} "
                      f"n={n:<11} rep={rep} elapsed={result.elapsed:.4f}")