import math

import numpy as np
import pytest

from pimc.base import RunConfig
from pimc.parallel import ParallelEstimator
from pimc.serial import NumpySerialEstimator, PureSerialEstimator, count_numpy


def test_numpy_serial_close_to_pi():
    r = NumpySerialEstimator().estimate(1_000_000, seed=1)
    assert abs(r.pi_estimate - math.pi) < 0.01


def test_pure_serial_close_to_pi():
    r = PureSerialEstimator().estimate(200_000, seed=1)
    assert abs(r.pi_estimate - math.pi) < 0.03


def test_parallel_close_to_pi_and_reports_workers():
    r = ParallelEstimator(workers=2).estimate(1_000_000, seed=1)
    assert r.workers == 2
    assert r.n == 1_000_000
    assert abs(r.pi_estimate - math.pi) < 0.01


def test_same_seed_is_reproducible():
    a = NumpySerialEstimator().estimate(100_000, seed=5)
    b = NumpySerialEstimator().estimate(100_000, seed=5)
    assert a.inside == b.inside


def test_non_positive_n_raises():
    with pytest.raises(ValueError):
        NumpySerialEstimator().estimate(0)


def test_n_smaller_than_workers_raises():
    with pytest.raises(ValueError):
        ParallelEstimator(workers=4).estimate(3)


def test_invalid_workers_and_chunk_raise():
    with pytest.raises(ValueError):
        ParallelEstimator(workers=0)
    with pytest.raises(ValueError):
        ParallelEstimator(workers=2, chunk=0)
    with pytest.raises(ValueError):
        count_numpy(10, np.random.SeedSequence(1), chunk=0)


def test_run_config_rejects_bad_values():
    with pytest.raises(ValueError):      # pydantic ValidationError, ValueError'ın alt sınıfı
        RunConfig(n=0)
    with pytest.raises(ValueError):
        RunConfig(n=10, workers=0)

def test_benchmark_writes_csv(tmp_path):
    from pimc.benchmark import run_benchmark

    csv_path = tmp_path / "out.csv"
    run_benchmark([NumpySerialEstimator()], ns=[1000], repeats=2,
                  csv_path=str(csv_path))
    assert len(csv_path.read_text().splitlines()) == 3   # başlık + 2 satır