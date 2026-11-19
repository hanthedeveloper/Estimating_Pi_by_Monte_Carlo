from pimc.parallel import ParallelEstimator


def main():
    n = 100_000_000
    for c in [1_000_000, 100_000]:
        r = ParallelEstimator(workers=4, chunk=c).estimate(n, seed=1)
        print(c, r.pi_estimate, round(r.elapsed, 4))


if __name__ == "__main__":
    main()