from pimc.benchmark import run_benchmark
from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator, PureSerialEstimator

P_LIST = [1, 2, 4, 6, 8, 12]


def main():
    # Deney A: NumPy, n = 10^8, 10 tekrar
    est_numpy = [NumpySerialEstimator()]
    est_numpy += [ParallelEstimator(workers=p, method="numpy") for p in P_LIST]
    run_benchmark(est_numpy, ns=[10**8], repeats=10,
                  csv_path="results/speedup_numpy.csv")

    # Deney B: saf Python, n = 10^7, 5 tekrar
    est_pure = [PureSerialEstimator()]
    est_pure += [ParallelEstimator(workers=p, method="pure") for p in P_LIST]
    run_benchmark(est_pure, ns=[10**7], repeats=5,
                  csv_path="results/speedup_pure.csv")


if __name__ == "__main__":
    main()