import math
import time
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    n: int = Field(gt=0, description="Toplam örnek sayısı")
    workers: int = Field(default=1, ge=1)
    repeats: int = Field(default=5, ge=1)
    seed: int = Field(default=12345, ge=0)


class EstimateResult(BaseModel):
    estimator: str
    n: int = Field(gt=0)
    workers: int = Field(default=1, ge=1)
    inside: int = Field(ge=0)
    elapsed: float = Field(ge=0)
    pi_estimate: float

    @property
    def error(self) -> float:
        return abs(self.pi_estimate - math.pi)


class PiEstimator(ABC):
    name: str = "base"

    @abstractmethod
    def count_inside(self, n: int, seed: int) -> int:
        pass

    @staticmethod
    def _compute_pi(inside: int, n: int) -> float:
        return 4 * inside / n

    def _validate(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n should be positive.")

    def _worker_count(self) -> int:
        return 1

    def _prepare(self) -> None:
        pass

    def _cleanup(self) -> None:
        pass

    def _timed_count(self, n: int, seed: int) -> tuple[int, float]:
        start = time.perf_counter()
        inside = self.count_inside(n, seed)
        elapsed = time.perf_counter() - start
        return inside, elapsed

    def _build_result(self, n: int, inside: int, elapsed: float) -> EstimateResult:
        return EstimateResult(
            estimator = self.name,
            n = n,
            workers = self._worker_count(),
            inside = inside,
            elapsed = elapsed,
            pi_estimate = self._compute_pi(inside, n),
        )

    def estimate(self, n: int, seed: int = 0) -> EstimateResult:
        self._validate(n)
        self._prepare()
        try:
            inside, elapsed = self._timed_count(n, seed)
        finally:
            self._cleanup()
        return self._build_result(n, inside, elapsed)