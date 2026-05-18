"""Artifact serialization: JSON summaries and CSV sweeps."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class _JsonEncoder(json.JSONEncoder):
    """Encoder that converts NumPy/Torch scalars and arrays to plain Python types."""

    def default(self, obj: Any) -> Any:
        # NumPy types — checked before torch to avoid unnecessary import
        try:
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        # Torch types
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
        except ImportError:
            pass

        # pathlib.Path — always use forward slashes so artifacts are portable
        if isinstance(obj, Path):
            return obj.as_posix()

        return super().default(obj)


def save_json(data: Dict, path) -> None:
    """Write dict to JSON, creating parent directories as needed.

    NumPy scalars/arrays and Torch scalars/tensors are converted to plain
    Python types before serialization.  Path values are converted to strings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_JsonEncoder)


def load_json(path) -> Dict:
    """Read JSON file."""
    with open(path) as f:
        return json.load(f)


def save_csv(rows: List[Dict], path) -> None:
    """Write list of dicts to CSV, creating parent directories as needed."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
