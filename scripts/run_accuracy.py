from pimc.benchmark import run_benchmark
from pimc.serial import NumpySerialEstimator
from pimc.parallel import ParallelEstimator


def main():
    estimators = [
        NumpySerialEstimator(),
        ParallelEstimator(workers=4, method="numpy"),
    ]
    ns = [10**k for k in range(3, 9)]   # 10^3 ... 10^8
    run_benchmark(estimators, ns=ns, repeats=20,
                  csv_path="results/accuracy.csv")


if __name__ == "__main__":
    main()