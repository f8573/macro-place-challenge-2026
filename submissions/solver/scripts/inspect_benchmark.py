"""
Benchmark inspection script -- Milestone 1.

Prints statistics for a benchmark without running placement.

Usage (from repo root):
    uv run python -m submissions.solver.scripts.inspect_benchmark
    uv run python -m submissions.solver.scripts.inspect_benchmark -b ibm01
    uv run python -m submissions.solver.scripts.inspect_benchmark --pt benchmarks/processed/public/ibm01.pt
"""

import argparse
import sys

from submissions.solver._env import SubmoduleMissingError, require_submodule

try:
    require_submodule()
except SubmoduleMissingError as exc:
    print(exc)
    sys.exit(1)

from macro_place.benchmark import Benchmark  # noqa: E402
from submissions.solver.config import BENCHMARKS_PT_DIR, IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.benchmark_adapter import inspect as inspect_bm  # noqa: E402


def _load(name: str) -> Benchmark:
    ibm_dir = IBM_TESTCASES_DIR / name
    if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
        from macro_place.loader import load_benchmark_from_dir

        # plc_client_os.py splits on '/' so always pass a POSIX path
        benchmark, _ = load_benchmark_from_dir(ibm_dir.as_posix())
        return benchmark

    pt_path = BENCHMARKS_PT_DIR / f"{name}.pt"
    if pt_path.exists():
        return Benchmark.load(str(pt_path))

    raise FileNotFoundError(f"Benchmark '{name}' not found at {ibm_dir} or {pt_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a benchmark.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--benchmark", default="ibm01")
    group.add_argument("--pt", metavar="PATH")
    args = parser.parse_args()

    if args.pt:
        benchmark = Benchmark.load(args.pt)
    else:
        benchmark = _load(args.benchmark)

    stats = inspect_bm(benchmark)
    print(f"\nBenchmark: {stats['name']}")
    print(f"  Canvas         : {stats['canvas_width']:.3f} x {stats['canvas_height']:.3f} um")
    print(f"  Canvas area    : {stats['canvas_area_um2']:.2f} um2")
    print(f"  Hard macros    : {stats['num_hard_macros']}")
    print(f"  Soft macros    : {stats['num_soft_macros']}")
    print(f"  Fixed          : {stats['num_fixed']}")
    print(f"  Movable hard   : {stats['num_movable_hard']}")
    print(f"  Nets           : {stats['num_nets']}")
    print(f"  Hard area      : {stats['hard_macro_area_um2']:.2f} um2")
    print(f"  Utilization    : {stats['utilization']:.1%}")
    print(f"  Grid           : {stats['grid_rows']} x {stats['grid_cols']}")
    print()


if __name__ == "__main__":
    main()
