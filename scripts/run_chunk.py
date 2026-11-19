from pimc.benchmark import run_benchmark
from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator

P_LIST = [1, 2, 4, 6, 12]


def main():
    for chunk in [1_000_000, 100_000, 10_000]:
        est = [NumpySerialEstimator(chunk=chunk)]
        est += [ParallelEstimator(workers=p, method="numpy", chunk=chunk)
                for p in P_LIST]
        run_benchmark(est, ns=[10**8], repeats=5,
                      csv_path=f"results/chunk_{chunk}.csv")


if __name__ == "__main__":
    main()