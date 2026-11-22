# Estimating Pi by Monte Carlo (`pimc`)

Estimating π with the Monte Carlo method: **serial** and **parallel** (`multiprocessing.Pool`) versions, the experiment scripts that compare them, and the analysis scripts that plot the results.

Method: `n` random points are drawn in the unit square, the points with `x² + y² ≤ 1` (those that fall inside the quarter circle) are counted, and `π ≈ 4 · inside / n` is computed.

Questions the project asks:

- How does the error decrease as `n` grows? (expected: `O(1/√n)`)
- How much faster do we get as the number of workers `p` increases? (speedup, efficiency)
- Why do NumPy and pure Python scale differently? (chunk size, cache, memory bandwidth)
- From which point on does parallel execution pay off for small `n`?
- If the work grows with the number of workers, does the run time stay constant? (weak scaling)

---

## Table of contents

1. [Project structure](#1-project-structure)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Reading the result files](#4-reading-the-result-files)
5. [Notes and limitations](#5-notes-and-limitations)

---

## 1. Project structure

```
Estimating_Pi_by_Monte_Carlo/
├── pyproject.toml          # Poetry settings, dependencies
├── poetry.lock
├── README.md
│
├── src/pimc/               # library (import pimc...)
│   ├── base.py             # RunConfig, EstimateResult, PiEstimator (abstract class)
│   ├── serial.py           # count_pure, count_numpy, PureSerialEstimator, NumpySerialEstimator
│   ├── parallel.py         # ParallelEstimator (Pool based)
│   ├── benchmark.py        # run_benchmark: runs a list of estimators repeatedly and writes to CSV
│   ├── io_utils.py         # appending a row to a CSV (append_row)
│   └── metrics.py          # speedup, efficiency, abs_error
│
├── scripts/                # EXPERIMENTS: take measurements, produce results/*.csv
│   ├── run_accuracy.py
│   ├── run_speedup.py
│   ├── run_chunk.py
│   ├── run_smalln.py
│   └── run_weak.py
│
├── analyze_accuracy.py     # ANALYSIS: reads results/*.csv, prints a table, produces results/*.png
├── analyze_speedup.py
├── analyze_chunk.py
├── analyze_smalln.py
├── analyze_weak.py
└── results/                # experiment outputs (CSV) and plots (PNG)
```

The workflow has two steps:

```
scripts/run_*.py  ──►  results/*.csv  ──►  analyze_*.py  ──►  results/*.png + table in the terminal
  (measurement)          (raw data)          (analysis)
```

---

## 2. Installation

Requirements: Python ≥ 3.11 (the measurements were taken with Python 3.13.1) and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

This command installs the `pimc` package, its dependencies (`numpy`, `pandas`, `matplotlib`, `pydantic`) and `pytest` from the development group.
Do not edit `poetry.lock` by hand; to add a dependency use `poetry add ...` and commit the updated `pyproject.toml` and `poetry.lock` together.

> **Important:** Run all commands from the **project root folder**. The scripts use relative `results/...` paths; if you run them from another folder, files will be written to the wrong place or will not be found.

---

## 3. Usage

### 3.1 As a library

```python
from pimc.serial import NumpySerialEstimator, PureSerialEstimator
from pimc.parallel import ParallelEstimator

# Serial (NumPy)
r = NumpySerialEstimator().estimate(n=10_000_000, seed=1)
print(r.pi_estimate, r.error, r.elapsed)

# Parallel (4 workers, NumPy)
r = ParallelEstimator(workers=4, method="numpy").estimate(n=100_000_000, seed=1)
print(r.workers, r.pi_estimate, r.elapsed)
```

`estimate()` returns an `EstimateResult`: `estimator`, `n`, `workers`, `inside`, `elapsed` (seconds), `pi_estimate` and `error` (= `|pi_estimate − π|`).

| Class | Description |
|---|---|
| `PureSerialEstimator` | Pure Python loop, serial. Reference (slowest) baseline. |
| `NumpySerialEstimator(chunk=...)` | NumPy vectorised, serial. Fast baseline. |
| `ParallelEstimator(workers=p, method="numpy" \| "pure", chunk=...)` | Splits `n` points across `p` worker processes and sums the partial counts. |

`chunk`: the number of points NumPy generates at a time (default `1_000_000`). It determines memory usage and cache behaviour and has a significant effect on the results (see section 4).

Invalid inputs raise `ValueError`: `n ≤ 0`, `workers < 1`, `chunk < 1` and `n < workers` (each worker must get at least one point).

### 3.2 Running the experiments

```bash
poetry run python scripts/run_accuracy.py
poetry run python scripts/run_speedup.py
poetry run python scripts/run_chunk.py
poetry run python scripts/run_smalln.py
poetry run python scripts/run_weak.py
```

| Script | What it measures | Output |
|---|---|---|
| `run_accuracy.py` | `n = 10³ … 10⁸`, NumPy serial and 4-worker parallel, 20 repetitions | `results/accuracy.csv` |
| `run_speedup.py` | A: NumPy, `n = 10⁸`, p ∈ {1,2,4,6,8,12}, 10 repetitions. B: pure Python, `n = 10⁷`, same p values, 5 repetitions | `results/speedup_numpy.csv`, `results/speedup_pure.csv` |
| `run_chunk.py` | NumPy, `n = 10⁸`, chunk ∈ {10⁶, 10⁵, 10⁴}, p ∈ {1,2,4,6,12}, 5 repetitions | `results/chunk_1000000.csv`, `chunk_100000.csv`, `chunk_10000.csv` |
| `run_smalln.py` | chunk = 10⁴, `n = 10³ … 10⁸`, p ∈ {2,4,6,12}, 10 repetitions | `results/smalln.csv` |
| `run_weak.py` | chunk = 10⁴, 10⁷ points per worker (`n = p · 10⁷`), p ∈ {1,2,4,6,12}, 10 repetitions | `results/weak.csv` |

Warnings:

- **CSV files are appended to, not overwritten.** If you run an experiment again, the rows are doubled and the median/mean are distorted. Delete the relevant CSV before re-running.
- Measurements can take a long time (especially with `n = 10⁸` and pure Python). Since every measurement is written to the file immediately, the rows collected so far are kept even if an experiment is interrupted.
- Before starting, `run_benchmark` validates the settings (`n`, `repeats`, `base_seed`) with `RunConfig`. If a value is invalid, it raises an error before the long experiment begins.
- If an error occurs during a measurement, the message is printed, that measurement is skipped and the experiment continues.
- Do not run other heavy processes on the computer during measurements; they distort the timings.

### 3.3 Analysis and plots

Once the CSVs are ready:

```bash
poetry run python analyze_accuracy.py   # -> results/accuracy.png
poetry run python analyze_speedup.py    # -> results/speedup.png
poetry run python analyze_chunk.py      # -> results/speedup_chunk.png
poetry run python analyze_smalln.py     # -> results/smalln.png
poetry run python analyze_weak.py       # -> results/weak.png
```

Each script prints a summary table to the terminal and saves the plot. The mapping:

| Analysis script | CSV it reads | Plot it produces |
|---|---|---|
| `analyze_accuracy.py` | `accuracy.csv` | `accuracy.png` |
| `analyze_speedup.py` | `speedup_numpy.csv`, `speedup_pure.csv` | `speedup.png` |
| `analyze_chunk.py` | `chunk_*.csv`, `speedup_pure.csv` | `speedup_chunk.png` |
| `analyze_smalln.py` | `smalln.csv` | `smalln.png` |
| `analyze_weak.py` | `weak.csv` | `weak.png` |

`analyze_chunk.py` uses the `make_table` function in `analyze_speedup.py`; therefore the two files must stay in the same folder.

### 3.4 Tests

```bash
poetry run pytest
```

`tests/test_estimators.py` checks the following: that the serial and parallel estimates are close to π, that the same seed gives the same result, that invalid `n`, `workers` and `chunk` values raise `ValueError`, and that `RunConfig` rejects invalid values.

---

## 4. Reading the result files

### 4.1 CSV format

All CSV files have the same columns. **Each row is a single measurement.**

| Column | Meaning |
|---|---|
| `estimator` | `pure_serial`, `numpy_serial`, `parallel_pure`, `parallel_numpy` |
| `n` | Total number of points (sum over all workers) |
| `workers` | Number of worker processes `p` (1 for the serial estimators) |
| `repeat` | Repetition number (starts at 0) |
| `seed` | Seed used in this measurement: `1000 + repeat` |
| `elapsed` | Wall-clock time of the counting step, in seconds |
| `pi_estimate` | `4 · inside / n` |
| `error` | `abs(pi_estimate − π)` (absolute error) |

Points to note:

- **Time measurement:** `elapsed` includes only the point-counting step. The time to create and close the `Pool` is **not included**.
- **Reproducibility:** For the same `(estimator, n, workers, chunk, seed)`, `pi_estimate` and `error` are identical in every run. `elapsed`, however, varies with the machine and the current load.
- **Difference between serial and `p=1` parallel:** `numpy_serial` runs in the main process, while `parallel_numpy` with `workers=1` runs in a newly started worker process. Even though both run the same code, their times may not be exactly the same.
- **The chunk value is not written in the CSV.** Which chunk was used can be found from the file name and the corresponding `run_*.py` script (see the table below).

### 4.2 What does each CSV file tell?

| File | Rows | Estimators | n | p | chunk | Question it answers |
|---|---|---|---|---|---|---|
| `accuracy.csv` | 240 | `numpy_serial`, `parallel_numpy` | 10³ … 10⁸ | 1, 4 | 10⁶ | How does the error decrease with `n`? Are serial and parallel equally accurate? |
| `speedup_numpy.csv` | 70 | `numpy_serial`, `parallel_numpy` | 10⁸ | 1, 2, 4, 6, 8, 12 | 10⁶ | Speedup and efficiency with NumPy |
| `speedup_pure.csv` | 35 | `pure_serial`, `parallel_pure` | 10⁷ | 1, 2, 4, 6, 8, 12 | none | Speedup and efficiency with pure Python |
| `chunk_1000000.csv` | 30 | `numpy_serial`, `parallel_numpy` | 10⁸ | 1, 2, 4, 6, 12 | 10⁶ | Effect of chunk on scaling |
| `chunk_100000.csv` | 30 | same | 10⁸ | 1, 2, 4, 6, 12 | 10⁵ | same |
| `chunk_10000.csv` | 30 | same | 10⁸ | 1, 2, 4, 6, 12 | 10⁴ | same |
| `smalln.csv` | 300 | `numpy_serial`, `parallel_numpy` | 10³ … 10⁸ | 2, 4, 6, 12 | 10⁴ | From which `n` is parallel faster than serial? |
| `weak.csv` | 60 | `numpy_serial` (only p=1), `parallel_numpy` | `p · 10⁷` | 1, 2, 4, 6, 12 | 10⁴ | When the work grows with the workers, does the time stay constant? |

### 4.3 Reading the CSV yourself

```python
import pandas as pd

df = pd.read_csv("results/speedup_numpy.csv")

# For time, use the MEDIAN rather than the mean (it is not affected by occasional slow measurements)
med = df.groupby(["estimator", "workers"])["elapsed"].median()

t_serial = med[("numpy_serial", 1)]
t_p4     = med[("parallel_numpy", 4)]
print("speedup(p=4) =", t_serial / t_p4)

# For error, use the mean and standard deviation over the repetitions
acc = pd.read_csv("results/accuracy.csv")
print(acc.groupby(["estimator", "n"])["error"].agg(["mean", "std"]))
```

### 4.4 Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Speedup | `S = T_serial / T_parallel` | If greater than 1, parallel is faster. The ideal value is `p`. |
| Efficiency | `E = S / p` | 1 = perfect. A low value indicates overhead, contention for shared resources (memory/cache) or load imbalance. |
| Weak-scaling efficiency | `T(1) / T(p)` | 1 if the time stays constant while the work per worker is fixed. |
| Scaled speedup | `p · T(1) / T(p)` | The ideal value in weak scaling is `p`. |

In the calculations, the serial baseline is **the median of the serial measurement in the same CSV** (same `n`, same chunk). Do not mix serial times from different files.

`efficiency > 1` can occur. This is not a physical advantage; it is mostly measurement noise or the serial baseline having been measured slightly slower in the main process. When the number of repetitions is small (5), the median is more exposed to noise.

### 4.5 Reading the plots

**`accuracy.png`** (accuracy and convergence)
- Axes: x = `n`, y = mean absolute error. Both are logarithmic.
- Lines: `numpy_serial`, `parallel_numpy` (each point is the mean of 20 repetitions) and the black dashed theoretical curve `0.8·σ`, where `σ = 4·√(p̂(1−p̂)/n)`.
- Reading: if the error shows a linear decrease and the slope is about **−0.5**, the error decreases as `O(1/√n)`. If the two estimator lines are close to each other and to the theory line, the serial and parallel versions are equally accurate. For small `n`, the spread between repetitions is large, so the line may fluctuate in this region.

**`speedup.png`** (speedup, NumPy and pure Python)
- Axes: x = number of workers `p`, y = speedup.
- Lines: NumPy (`n = 10⁸`), pure Python (`n = 10⁷`) and the black dashed ideal line `S = p`. The vertical dotted line marks the 6 physical cores.
- Reading: the closer to the ideal line, the more efficient. The value of `p` where the curve departs from the ideal line is where scaling stalls. For `p` values beyond the number of physical cores (the vertical line), the increase shrinks or may even drop: the extra workers share the same core through hyper-threading.

**`speedup_chunk.png`** (effect of chunk)
- Axes: x = `p`, y = speedup (each line relative to the serial measurement with its own chunk).
- Lines: three chunk values for NumPy (10⁶, 10⁵, 10⁴), pure Python as a grey dashed line, and the ideal line.
- Reading: same code, same machine, only the chunk changes. If the curve approaches the ideal as the chunk gets smaller, the cause of the weak scaling is not the algorithm but whether the working set fits in the cache.

**`smalln.png`** (small `n` and start-up cost)
- Axes: x = `n` (log), y = speedup (log).
- Lines: one line per `p` and the black dashed `speedup = 1` (break-even line).
- Reading: if a line is **below the break-even line, parallel is slower than serial**. The `n` at which the line crosses 1 is the threshold where parallel starts to pay off. For small `n`, the cost of spawning processes and sending data is larger than the work itself.

**`weak.png`** (weak scaling, two panels)
- Left panel: x = `p`, y = time. The dot is the median and the error bar is the min–max range. The black dashed line shows the ideal case (constant time).
- Right panel: x = `p`, y = `T(1)/T(p)`. The black dashed line is 1 (ideal).
- Reading: if the time stays constant and the efficiency is close to 1, scaling is good. Wide error bars mean the measurements at that point are scattered. This may be caused by background load or memory pressure.

### 4.6 Tables printed in the terminal

`analyze_speedup.py` and `analyze_chunk.py` print a table with the columns `workers`, `median_s` (median time), `speedup` and `efficiency`. The top line shows the serial median time of the relevant file.
`analyze_smalln.py` prints a speedup matrix with `n` as rows and `p` as columns.
`analyze_weak.py` prints `n`, `median_s`, `efficiency` and `scaled_speedup` for each `p`.
`analyze_accuracy.py` prints the mean and standard deviation of the error for each `(estimator, n)`.

---

## 5. Notes and limitations

- **Results are machine-specific.** The current measurements were taken on an Intel Core i5-10500 (6 physical cores, 12 logical cores). On another processor, the speedup curves and the chunk effect may turn out differently. To compare on your own machine, re-run the experiments.
- **Pool set-up time is not included in the measurement.** The worker processes are started before the timer starts. The difference is negligible for large `n`, but for small `n` (the `smalln` experiment) the advantage of parallel execution in real use is smaller than what is shown here; the `n` threshold at which parallel overtakes serial is actually larger.
- **External influences:** Other processes were running and there may have been memory pressure during the measurements. This may increase the spread between repetitions, especially at `p = 6` and `p = 12`.
- **The cache explanation is a hypothesis.** The interpretation of chunk and memory bandwidth is *consistent* with the measurements, but the cache hit rate was not measured directly (`perf` or similar was not used).
- **Multiprocessing start method:** On Python 3.13 the Linux default is `fork`. If the default changes with Python 3.14, the timings may be affected.
- **Random number streams are independent:** Each worker gets its own seed through `SeedSequence(seed).spawn(p)`. Changing the chunk changes how the random stream is split, so the `pi_estimate` values may differ slightly. Both are valid Monte Carlo estimates.
