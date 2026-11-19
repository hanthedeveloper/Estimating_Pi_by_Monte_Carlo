from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator


def main():
    n = 10_000_000
    seri = NumpySerialEstimator().estimate(n, seed=1)
    print("seri   ", seri.workers, seri.pi_estimate, seri.elapsed)

    for p in [1, 2, 4, 6, 8, 12]:
        r = ParallelEstimator(workers=p).estimate(n, seed=1)
        print("paralel", r.workers, r.pi_estimate, r.elapsed)


if __name__ == "__main__":
    main()