"""Core abstractions for the Monte Carlo pi estimation package.

This module defines:

* ``RunConfig``       -- validated experiment settings (Pydantic model).
* ``EstimateResult``  -- the outcome of a single estimation run (Pydantic model).
* ``PiEstimator``     -- abstract base class implementing the *Template Method*
  pattern. The base class owns the parts that must be identical for every
  estimator (input validation, timing, the ``4 * inside / n`` formula, result
  packaging), while subclasses only supply the point-counting step.

Keeping timing and the pi formula in one place guarantees that serial and
parallel estimators are measured in exactly the same way, which makes the
performance comparison fair.
"""

import math
import time
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Validated settings for one experiment configuration.

    Attributes:
        n: Total number of random points to sample (must be > 0).
        workers: Number of worker processes (must be >= 1).
        repeats: Number of independent repetitions (must be >= 1).
        seed: Base seed for reproducibility (must be >= 0).

    Note:
        The benchmark scripts currently pass these values directly to
        ``run_benchmark``; this model is available for validating them up front.
    """

    n: int = Field(gt=0, description="Total number of samples")
    workers: int = Field(default=1, ge=1)
    repeats: int = Field(default=5, ge=1)
    seed: int = Field(default=12345, ge=0)


class EstimateResult(BaseModel):
    """Result of a single call to :meth:`PiEstimator.estimate`.

    Attributes:
        estimator: Name of the estimator that produced the result
            (e.g. ``"numpy_serial"``, ``"parallel_pure"``).
        n: Total number of sampled points.
        workers: Number of worker processes used (1 for serial estimators).
        inside: Number of points that fell inside the quarter circle.
        elapsed: Wall-clock time of the counting step, in seconds.
        pi_estimate: The estimate ``4 * inside / n``.

    Pydantic validates the fields on construction, so an invalid result
    (e.g. a negative ``n``) raises ``ValidationError`` immediately.
    """

    estimator: str
    n: int = Field(gt=0)
    workers: int = Field(default=1, ge=1)
    inside: int = Field(ge=0)
    elapsed: float = Field(ge=0)
    pi_estimate: float

    @property
    def error(self) -> float:
        """Absolute error ``|pi_estimate - math.pi|``.

        Implemented as a property because it is derived from ``pi_estimate``
        and should never be stored separately (it could become inconsistent).
        Properties are not included in ``model_dump()``, so code that writes
        results to CSV must add this value explicitly.
        """
        return abs(self.pi_estimate - math.pi)


class PiEstimator(ABC):
    """Abstract base class for all pi estimators (Template Method pattern).

    Method kinds:

    * **Abstract**: :meth:`count_inside` -- every subclass must implement it.
    * **Hooks** (``_worker_count``, ``_prepare``, ``_cleanup``) -- have a
      default implementation that subclasses may override.
    * **Concrete**: :meth:`estimate` and its helpers -- shared by all
      subclasses and normally not overridden.

    Subclasses should set the class attribute ``name`` so that results can be
    labelled in the output CSV files.
    """

    name: str = "base"

    @abstractmethod
    def count_inside(self, n: int, seed: int) -> int:
        """Count how many of ``n`` random points fall inside the quarter circle.

        Args:
            n: Number of random points to sample in the unit square.
            seed: Seed that makes the run reproducible.

        Returns:
            The number of points with ``x**2 + y**2 <= 1``.
        """
        pass

    @staticmethod
    def _compute_pi(inside: int, n: int) -> float:
        """Convert a hit count into a pi estimate.

        The quarter circle has area pi/4 and the unit square has area 1, so
        ``inside / n`` approximates pi/4 and therefore ``pi ~= 4 * inside / n``.
        """
        return 4 * inside / n

    def _validate(self, n: int) -> None:
        """Raise ``ValueError`` if ``n`` is not a positive integer.

        This is an instance method (not static) so that subclasses can override
        it with checks that need ``self`` (e.g. ``n >= self.workers``).
        """
        if n <= 0:
            raise ValueError("n should be positive.")

    def _worker_count(self) -> int:
        """Hook: number of workers reported in the result. Serial default: 1."""
        return 1

    def _prepare(self) -> None:
        """Hook: runs *before* timing starts (e.g. creating a process pool).

        Because it runs outside the timed section, setup cost is not included
        in ``EstimateResult.elapsed``. The default does nothing.
        """
        pass

    def _cleanup(self) -> None:
        """Hook: runs *after* timing ends (e.g. closing a process pool).

        Called from a ``finally`` block, so it runs even if counting fails.
        The default does nothing.
        """
        pass

    def _timed_count(self, n: int, seed: int) -> tuple[int, float]:
        """Run :meth:`count_inside` and measure its wall-clock time.

        ``time.perf_counter()`` is a monotonic, high-resolution clock suited
        for benchmarking. All timing in the project goes through this method.

        Returns:
            A tuple ``(inside, elapsed_seconds)``.
        """
        start = time.perf_counter()
        inside = self.count_inside(n, seed)
        elapsed = time.perf_counter() - start
        return inside, elapsed

    def _build_result(self, n: int, inside: int, elapsed: float) -> EstimateResult:
        """Package raw numbers into a validated :class:`EstimateResult`."""
        return EstimateResult(
            estimator = self.name,
            n = n,
            workers = self._worker_count(),
            inside = inside,
            elapsed = elapsed,
            pi_estimate = self._compute_pi(inside, n),
        )

    def estimate(self, n: int, seed: int = 0) -> EstimateResult:
        """Estimate pi using ``n`` random points (the template method).

        Execution order: validate -> prepare -> timed counting -> cleanup ->
        build result.

        Args:
            n: Total number of random points (must be positive).
            seed: Seed for reproducibility.

        Returns:
            An :class:`EstimateResult` with the estimate, hit count and time.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        self._validate(n)
        self._prepare()
        try:
            inside, elapsed = self._timed_count(n, seed)
        finally:
            # Always release resources (e.g. the process pool), even on error.
            self._cleanup()
        return self._build_result(n, inside, elapsed)