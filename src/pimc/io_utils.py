"""CSV persistence for benchmark results."""

import csv
from pathlib import Path

from pimc.base import EstimateResult

# Column order of every results CSV. ``inside`` is intentionally not stored.
FIELDS = ["estimator", "n", "workers", "repeat", "seed",
          "elapsed", "pi_estimate", "error"]


def append_row(path: str, result: EstimateResult, repeat: int, seed: int) -> None:
    """Append one measurement to a CSV file, creating it if necessary.

    The file is opened in append mode and written after every single
    measurement, so an interrupted experiment keeps all rows written so far.

    Args:
        path: CSV file path. Missing parent directories are created.
        result: The measurement to store.
        repeat: Index of the repetition (0-based).
        seed: Seed used for this run.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not file.exists() or file.stat().st_size == 0  # header for new/empty file
    row = result.model_dump()
    row["repeat"] = repeat
    row["seed"] = seed
    # `error` is a property, so model_dump() does not include it; add it here.
    row["error"] = result.error

    with open(file, "a", newline="") as f:
        # extrasaction="ignore" silently drops keys not listed in FIELDS
        # (e.g. `inside`).
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)