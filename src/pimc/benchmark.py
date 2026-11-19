from pimc.base import PiEstimator
from pimc.io_utils import append_row


def run_benchmark(estimators: list[PiEstimator], ns: list[int],
                  repeats: int, csv_path: str, base_seed: int = 1000) -> None:
    for est in estimators:
        for n in ns:
            for rep in range(repeats):
                seed = base_seed + rep
                try:
                    result = est.estimate(n, seed=seed)
                except Exception as e:
                    print(f"HATA: {est.name} n={n} rep={rep}: {e}")
                    continue
                append_row(csv_path, result, repeat=rep, seed=seed)
                print(f"{result.estimator:<20} workers={result.workers:<3} "
                      f"n={n:<11} rep={rep} elapsed={result.elapsed:.4f}")