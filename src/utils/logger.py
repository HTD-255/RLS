"""
logger.py — Experiment data logger for AutoCar III experiments.

Writes per-step telemetry to CSV files and experiment-level summaries to
JSON, following the directory layout defined in the research plan (Section 7).
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ExperimentLogger:
    """CSV + JSON logger for a single experiment run.

    Usage
    -----
    >>> logger = ExperimentLogger("data/exp1", "RLS", run_id=3)
    >>> logger.log_step(record_dict)          # called every dt
    >>> logger.save_summary(metrics_dict)      # called once at end
    >>> logger.close()

    Files created::

        data/exp1/RLS_run03.csv        ← per-step telemetry
        data/exp1/RLS_run03_meta.json  ← summary + config

    Parameters
    ----------
    base_dir : str | Path
        Experiment directory (e.g., ``data/exp1``).
    controller_name : str
        Used as filename prefix.
    run_id : int
        Run number (zero-padded to 2 digits in filename).
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        controller_name: str,
        run_id: int = 0,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        tag = controller_name.replace(" ", "_").replace("(", "").replace(")", "")
        self.csv_path = self.base_dir / f"{tag}_run{run_id:02d}.csv"
        self.meta_path = self.base_dir / f"{tag}_run{run_id:02d}_meta.json"

        self._csv_file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None
        self._step_count: int = 0
        self._start_time: float = time.time()

        # Metadata collected during run
        self.meta: Dict[str, Any] = {
            "controller": controller_name,
            "run_id": run_id,
            "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # --------------------------------------------------------------------- #
    #  Per-step logging
    # --------------------------------------------------------------------- #
    def log_step(self, record: Dict[str, Any]) -> None:
        """Append one telemetry record to the CSV file.

        On the first call the CSV header is inferred from the record keys.
        """
        # Flatten any list/array values for CSV compatibility
        flat = self._flatten(record)

        if self._writer is None:
            self._fieldnames = list(flat.keys())
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()

        self._writer.writerow(flat)
        self._step_count += 1

    # --------------------------------------------------------------------- #
    #  Summary / metadata
    # --------------------------------------------------------------------- #
    def save_summary(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write experiment metadata + metrics to a JSON sidecar file."""
        self.meta["end_time_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.meta["duration_s"] = round(time.time() - self._start_time, 2)
        self.meta["n_steps"] = self._step_count
        self.meta["csv_file"] = str(self.csv_path.name)

        if metrics:
            self.meta["metrics"] = self._make_serialisable(metrics)
        if config:
            self.meta["config"] = self._make_serialisable(config)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)
        return self.meta_path

    # --------------------------------------------------------------------- #
    #  Teardown
    # --------------------------------------------------------------------- #
    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _flatten(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list/array values to semicolon-separated strings."""
        flat: Dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, (list, tuple)):
                flat[k] = ";".join(str(x) for x in v)
            elif hasattr(v, "tolist"):  # numpy array
                serialised = v.tolist()
                if isinstance(serialised, list):
                    flat[k] = ";".join(str(x) for x in serialised)
                else:
                    flat[k] = serialised
            else:
                flat[k] = v
        return flat

    @staticmethod
    def _make_serialisable(obj: Any) -> Any:
        """Recursively convert numpy types for JSON serialisation."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: ExperimentLogger._make_serialisable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ExperimentLogger._make_serialisable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
