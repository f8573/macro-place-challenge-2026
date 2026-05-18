"""
Baseline regression guard.

Loads the stored golden baseline artifact and verifies the recorded average
proxy cost is within the expected range for the M1 shelf-pack placer.

This test does NOT run the evaluator.  It only checks that the artifact has
not drifted outside the acceptable window, providing a cheap signal if
someone accidentally commits an updated artifact with a bad value.
"""

from pathlib import Path

import pytest

from submissions.solver.core.io import load_json

_GOLDEN = (
    Path(__file__).resolve().parent.parent / "artifacts" / "golden_baseline.json"
)

_EXPECTED_LOW = 2.20
_EXPECTED_HIGH = 2.22


def test_golden_baseline_artifact_exists():
    assert _GOLDEN.exists(), f"Golden baseline artifact missing: {_GOLDEN}"


def test_baseline_avg_proxy_in_range():
    artifact = load_json(_GOLDEN)
    avg = artifact["avg_proxy_cost"]
    assert isinstance(avg, (int, float)), "avg_proxy_cost must be numeric"
    assert _EXPECTED_LOW <= avg <= _EXPECTED_HIGH, (
        f"Baseline avg_proxy_cost {avg:.4f} is outside expected range "
        f"[{_EXPECTED_LOW}, {_EXPECTED_HIGH}].  "
        f"If the placer changed intentionally, update artifacts/golden_baseline.json."
    )


def test_golden_baseline_has_required_keys():
    artifact = load_json(_GOLDEN)
    for key in ("description", "placer_version", "avg_proxy_cost"):
        assert key in artifact, f"Golden baseline missing key: {key}"
