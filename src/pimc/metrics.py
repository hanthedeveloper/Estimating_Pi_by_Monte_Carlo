"""Performance and accuracy metrics used in the analysis."""

import math


def speedup(t_serial: float, t_parallel: float) -> float:
    """Speedup ``S = T_serial / T_parallel``.

    Values above 1 mean the parallel version is faster; the ideal speedup with
    ``p`` workers is ``p``.
    """
    return t_serial / t_parallel


def efficiency(t_serial: float, t_parallel: float, workers: int) -> float:
    """Parallel efficiency ``E = S / p``.

    ``1.0`` means perfect use of all workers; lower values indicate overhead,
    shared-resource contention or load imbalance.
    """
    return speedup(t_serial, t_parallel) / workers


def abs_error(pi_estimate: float) -> float:
    """Absolute error ``|pi_estimate - pi|`` of a pi estimate."""
    return abs(pi_estimate - math.pi)