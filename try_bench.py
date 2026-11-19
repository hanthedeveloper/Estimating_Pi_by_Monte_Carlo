from pimc.benchmark import run_benchmark
from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator, PureSerialEstimator


def main():
    estimators = [
        PureSerialEstimator(),
        NumpySerialEstimator(),
        ParallelEstimator(workers=2, method="pure"),
        ParallelEstimator(workers=2, method="numpy"),
    ]
    run_benchmark(estimators, ns=[10_000, 100_000], repeats=3,
                  csv_path="results/try.csv")


if __name__ == "__main__":
    main()