"""
Smoke benchmark script -- Milestone 1.

Loads a benchmark, runs the baseline placer, validates the result,
optionally scores (when a live plc is available), and saves a JSON artifact.

Usage (from repo root):
    uv run python -m submissions.solver.scripts.smoke_benchmark
    uv run python -m submissions.solver.scripts.smoke_benchmark -b ibm01
    uv run python -m submissions.solver.scripts.smoke_benchmark -b ibm01 --vis
"""

import argparse
import sys
import time
from pathlib import Path

from submissions.solver._env import SubmoduleMissingError, require_submodule

try:
    require_submodule()
except SubmoduleMissingError as exc:
    print(exc)
    sys.exit(1)

from macro_place.benchmark import Benchmark  # noqa: E402
from submissions.solver.config import ARTIFACTS_DIR, BENCHMARKS_PT_DIR, IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.benchmark_adapter import inspect as inspect_bm  # noqa: E402
from submissions.solver.core.io import save_json  # noqa: E402
from submissions.solver.core.scoring import score  # noqa: E402
from submissions.solver.core.types import SmokeResult  # noqa: E402
from submissions.solver.core.validation import validate  # noqa: E402


def _load(name: str):
    """Try IBM testcase dir first, fall back to preprocessed .pt file.

    Returns (benchmark, plc|None).  plc is None when loaded from .pt.
    """
    ibm_dir = IBM_TESTCASES_DIR / name
    if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
        from macro_place.loader import load_benchmark_from_dir

        print(f"  Loading from {ibm_dir}")
        # plc_client_os.py splits on '/' so always pass a POSIX path
        return load_benchmark_from_dir(ibm_dir.as_posix())

    pt_path = BENCHMARKS_PT_DIR / f"{name}.pt"
    if pt_path.exists():
        print(f"  IBM testcases not found - falling back to {pt_path}")
        print("  (scoring will be skipped; initialize submodule for full evaluation)")
        return Benchmark.load(str(pt_path)), None

    raise FileNotFoundError(
        f"Benchmark '{name}' not found.\n"
        f"  Tried: {ibm_dir}\n"
        f"  Tried: {pt_path}\n"
        "Run: git submodule update --init external/MacroPlacement"
    )


def _load_pt(pt_path: str):
    """Load from an explicit .pt file.  No plc available."""
    path = Path(pt_path)
    if not path.exists():
        raise FileNotFoundError(f".pt file not found: {path}")
    print(f"  Loading from {path}")
    print("  (scoring will be skipped; initialize submodule for full evaluation)")
    return Benchmark.load(str(path)), None


def run(benchmark_name: str, plc, benchmark: Benchmark, vis: bool, save: bool) -> SmokeResult:
    from submissions.solver.placer import SolverPlacer

    placer = SolverPlacer()
    t0 = time.perf_counter()
    placement = placer.place(benchmark)
    runtime = time.perf_counter() - t0

    is_valid, violations = validate(placement, benchmark)
    costs = score(placement, benchmark, plc)

    result = SmokeResult(
        benchmark_name=benchmark_name,
        is_valid=is_valid,
        violations=violations,
        costs=costs,
        runtime_s=runtime,
    )

    status = "VALID" if is_valid else f"INVALID ({len(violations)} violations)"
    print(f"  status  : {status}")
    print(f"  runtime : {runtime:.3f}s")
    for v in violations[:5]:
        print(f"    ! {v}")

    if costs:
        print(
            f"  proxy   : {costs['proxy_cost']:.4f}  "
            f"(wl={costs['wirelength_cost']:.3f}  "
            f"den={costs['density_cost']:.3f}  "
            f"cong={costs['congestion_cost']:.3f})"
        )
        print(f"  overlaps: {costs['overlap_count']}")
    else:
        print("  scoring : skipped (no live plc)")

    if save:
        artifact = {
            "benchmark": benchmark_name,
            "is_valid": is_valid,
            "violations": violations,
            "runtime_s": runtime,
            "costs": costs,
            "stats": inspect_bm(benchmark),
        }
        out = ARTIFACTS_DIR / f"smoke_{benchmark_name}.json"
        save_json(artifact, out)
        print(f"  artifact: {out}")

    if vis:
        from submissions.solver.viz.draw import draw

        vis_dir = ARTIFACTS_DIR / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_png = str(vis_dir / f"{benchmark_name}.png")
        draw(placement, benchmark, save_path=out_png, plc=plc)

    return result


def main():
    parser = argparse.ArgumentParser(
        prog="smoke_benchmark",
        description="Milestone 1 smoke test: load / place / validate / score.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--benchmark", default="ibm01", help="Benchmark name (default: ibm01)")
    group.add_argument("--pt", metavar="PATH", help="Load from explicit .pt file")
    parser.add_argument("--vis", action="store_true", help="Save visualization PNG")
    parser.add_argument("--no-save", action="store_true", help="Skip JSON artifact")
    args = parser.parse_args()

    print("=" * 60)
    if args.pt:
        name = Path(args.pt).stem
        benchmark, plc = _load_pt(args.pt)
    else:
        name = args.benchmark
        benchmark, plc = _load(name)

    print(f"smoke_benchmark -- {name}")
    print("=" * 60)

    stats = inspect_bm(benchmark)
    print(
        f"  macros  : {stats['num_hard_macros']} hard, {stats['num_soft_macros']} soft"
        f"  (fixed={stats['num_fixed']})"
    )
    print(
        f"  canvas  : {stats['canvas_width']:.1f} x {stats['canvas_height']:.1f} um"
        f"  util={stats['utilization']:.1%}"
    )
    print(f"  nets    : {stats['num_nets']}")

    run(name, plc, benchmark, vis=args.vis, save=not args.no_save)
    print("=" * 60)


if __name__ == "__main__":
    main()
