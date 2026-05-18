"""Unit tests for core/io.py — JSON and CSV artifact serialization."""

import json
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from submissions.solver.core.io import save_json, load_json, save_csv


# ── JSON ──────────────────────────────────────────────────────────────────────


def test_save_json_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        save_json({"key": "value", "n": 42}, path)
        assert os.path.exists(path)


def test_save_json_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        data = {"name": "ibm01", "score": 1.234, "valid": True}
        save_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data


def test_save_json_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "dir", "out.json")
        save_json({"x": 1}, path)
        assert os.path.exists(path)


def test_load_json_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rt.json")
        data = {"benchmark": "ibm01", "proxy_cost": 1.5, "overlaps": 0}
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data


def test_save_json_stable():
    # Same data written twice should produce identical files
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "a.json")
        p2 = os.path.join(tmpdir, "b.json")
        data = {"z": 1, "a": 2, "m": 3}
        save_json(data, p1)
        save_json(data, p2)
        with open(p1) as f1, open(p2) as f2:
            assert f1.read() == f2.read()


# ── CSV ───────────────────────────────────────────────────────────────────────


def test_save_csv_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.csv")
        rows = [{"name": "ibm01", "score": 1.2}, {"name": "ibm02", "score": 1.8}]
        save_csv(rows, path)
        assert os.path.exists(path)


def test_save_csv_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.csv")
        rows = [
            {"benchmark": "ibm01", "proxy_cost": 1.5, "overlaps": 0},
            {"benchmark": "ibm02", "proxy_cost": 1.9, "overlaps": 0},
        ]
        save_csv(rows, path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            loaded = list(reader)
        assert len(loaded) == 2
        assert loaded[0]["benchmark"] == "ibm01"
        assert loaded[1]["benchmark"] == "ibm02"


def test_save_csv_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "deep", "path", "out.csv")
        save_csv([{"x": 1}], path)
        assert os.path.exists(path)


def test_save_csv_empty_noop():
    # Empty rows should not raise and should not create a file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.csv")
        save_csv([], path)
        assert not os.path.exists(path)


# ── NumPy / Torch / Path serialization ───────────────────────────────────────


def test_save_json_numpy_scalar_float():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"v": np.float32(1.5)}, p)
        loaded = json.load(open(p))
        assert loaded["v"] == pytest.approx(1.5)
        assert isinstance(loaded["v"], float)


def test_save_json_numpy_scalar_int():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"v": np.int64(42)}, p)
        loaded = json.load(open(p))
        assert loaded["v"] == 42
        assert isinstance(loaded["v"], int)


def test_save_json_numpy_scalar_bool():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"v": np.bool_(True)}, p)
        loaded = json.load(open(p))
        assert loaded["v"] is True


def test_save_json_numpy_array():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"arr": np.array([1.0, 2.0, 3.0])}, p)
        loaded = json.load(open(p))
        assert loaded["arr"] == pytest.approx([1.0, 2.0, 3.0])
        assert isinstance(loaded["arr"], list)


def test_save_json_torch_scalar_tensor():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"v": torch.tensor(2.5)}, p)
        loaded = json.load(open(p))
        assert loaded["v"] == pytest.approx(2.5)


def test_save_json_torch_tensor():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"t": torch.tensor([1.0, 2.0])}, p)
        loaded = json.load(open(p))
        assert loaded["t"] == pytest.approx([1.0, 2.0])
        assert isinstance(loaded["t"], list)


def test_save_json_path_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        save_json({"p": Path("/some/artifact/path")}, p)
        loaded = json.load(open(p))
        assert loaded["p"] == "/some/artifact/path"
        assert isinstance(loaded["p"], str)


def test_save_json_nested_mixed_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "out.json")
        data = {
            "name": "ibm01",
            "cost": np.float32(2.2109),
            "counts": {"hard": np.int64(246), "soft": np.int64(894)},
            "coords": torch.tensor([1.0, 2.0]),
            "path": Path("/artifacts/run1"),
            "flags": [np.bool_(True), np.bool_(False)],
        }
        save_json(data, p)
        loaded = json.load(open(p))
        assert loaded["name"] == "ibm01"
        assert loaded["cost"] == pytest.approx(2.2109, abs=1e-4)
        assert loaded["counts"]["hard"] == 246
        assert loaded["coords"] == pytest.approx([1.0, 2.0])
        assert loaded["path"] == "/artifacts/run1"
        assert loaded["flags"] == [True, False]
