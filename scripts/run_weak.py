from pimc.benchmark import run_benchmark
from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator

CHUNK = 10_000
PER_WORKER = 10_000_000
P_LIST = [1, 2, 4, 6, 12]


def main():
    for p in P_LIST:
        n = p * PER_WORKER
        est = [ParallelEstimator(workers=p, method="numpy", chunk=CHUNK)]
        if p == 1:
            est.insert(0, NumpySerialEstimator(chunk=CHUNK))
        run_benchmark(est, ns=[n], repeats=10,
                      csv_path="results/weak.csv")


if __name__ == "__main__":
    main()