from pimc.parallel import ParallelEstimator
from pimc.serial import PureSerialEstimator


def main():
    n = 10_000_000
    seri = PureSerialEstimator().estimate(n, seed=1)
    print("seri", seri.workers, seri.pi_estimate, seri.elapsed)

    for p in [1, 2, 4, 6, 12]:
        r = ParallelEstimator(workers=p, method="pure").estimate(n, seed=1)
        print("paralel pure", p, r.pi_estimate, r.elapsed)


if __name__ == "__main__":
    main()