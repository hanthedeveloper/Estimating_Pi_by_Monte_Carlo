import csv
from pathlib import Path

from pimc.base import EstimateResult

FIELDS = ["estimator", "n", "workers", "repeat", "seed",
          "elapsed", "pi_estimate", "error"]


def append_row(path: str, result: EstimateResult, repeat: int, seed: int) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not file.exists()

    row = result.model_dump()
    row["repeat"] = repeat
    row["seed"] = seed
    row["error"] = result.error

    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)