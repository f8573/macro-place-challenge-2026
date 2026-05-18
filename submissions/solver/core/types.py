"""Minimal stable types for Milestone 1."""

from typing import Dict, List, NamedTuple, Optional


class SmokeResult(NamedTuple):
    """Result of a single smoke benchmark run."""

    benchmark_name: str
    is_valid: bool
    violations: List[str]
    costs: Optional[Dict]  # None when plc unavailable
    runtime_s: float
