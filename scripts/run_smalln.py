from pimc.benchmark import run_benchmark
from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator

CHUNK = 10_000
P_LIST = [2, 4, 6, 12]
NS = [10**k for k in range(3, 9)]   # 10^3 ... 10^8


def main():
    est = [NumpySerialEstimator(chunk=CHUNK)]
    est += [ParallelEstimator(workers=p, method="numpy", chunk=CHUNK)
            for p in P_LIST]
    run_benchmark(est, ns=NS, repeats=10,
                  csv_path="results/smalln.csv")


if __name__ == "__main__":
    main()