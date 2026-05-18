"""Solver path constants."""

from pathlib import Path

_SOLVER_DIR = Path(__file__).parent
_REPO_ROOT = _SOLVER_DIR.parent.parent

ARTIFACTS_DIR = _SOLVER_DIR / "artifacts"
BENCHMARKS_PT_DIR = _REPO_ROOT / "benchmarks" / "processed" / "public"
IBM_TESTCASES_DIR = _REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
