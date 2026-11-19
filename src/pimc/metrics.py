import math


def speedup(t_serial: float, t_parallel: float) -> float:
    return t_serial / t_parallel


def efficiency(t_serial: float, t_parallel: float, workers: int) -> float:
    return speedup(t_serial, t_parallel) / workers


def abs_error(pi_estimate: float) -> float:
    return abs(pi_estimate - math.pi)