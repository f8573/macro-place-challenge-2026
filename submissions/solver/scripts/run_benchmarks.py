"""
Benchmark runner with profiles for M2A/M2B.

Profiles:
  smoke      — 1-3 small benchmarks, console only
  standard   — all public benchmarks, console + CSV
  heavy      — all benchmarks, repeat x3, console + CSV + JSON
  m2b-prep   — all benchmarks with full candidate budget, ranked output
  stress     — all benchmarks, 3x repeat, JSON only

Usage (from repo root):
    python -m submissions.solver.scripts.run_benchmarks --profile smoke
    python -m submissions.solver.scripts.run_benchmarks --profile standard --out results/
    python -m submissions.solver.scripts.run_benchmarks --profile m2b-prep -b ibm01
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure solver dir is on path
_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.config import BENCHMARKS_PT_DIR, IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.io import save_csv, save_json  # noqa: E402

# Profile definitions -------------------------------------------------------

_PROFILES: Dict[str, Dict] = {
    "smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console"],
        "description": "Quick sanity check — 3 small benchmarks",
    },
    "standard": {
        "benchmarks": None,  # All available
        "repeat": 1,
        "output": ["console", "csv"],
        "description": "All public benchmarks, one run",
    },
    "heavy": {
        "benchmarks": None,
        "repeat": 3,
        "output": ["console", "csv", "json"],
        "description": "All benchmarks x3 repeats",
    },
    "m2b-prep": {
        "benchmarks": None,
        "repeat": 1,
        "output": ["console", "csv", "json"],
        "description": "Full M2B candidate pipeline, ranked output",
        "show_candidates": True,
    },
    "stress": {
        "benchmarks": None,
        "repeat": 5,
        "output": ["json"],
        "description": "Stress test — all benchmarks x5",
    },
}

# ---------------------------------------------------------------------------


def _discover_benchmarks(names: Optional[List[str]]) -> List[Path]:
    """Return list of .pt file paths to run."""
    if names is not None:
        paths = []
        for name in names:
            p = BENCHMARKS_PT_DIR / f"{name}.pt"
            if p.exists():
                paths.append(p)
            else:
                print(f"  Warning: benchmark '{name}' not found at {p}")
        return paths

    if not BENCHMARKS_PT_DIR.exists():
        print(f"  Warning: benchmark dir not found: {BENCHMARKS_PT_DIR}")
        return []

    return sorted(BENCHMARKS_PT_DIR.glob("*.pt"))


def _load_benchmark(pt_path: Path):
    """Load benchmark from .pt file. Returns (benchmark, plc=None)."""
    try:
        from macro_place.benchmark import Benchmark

        bm = Benchmark.load(str(pt_path))
        return bm, None
    except Exception as exc:
        print(f"  Error loading {pt_path.name}: {exc}")
        return None, None


def _run_one(benchmark, plc, show_candidates: bool = False) -> Dict:
    """Run M2B pipeline on one benchmark. Return result dict."""
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select

    t0 = time.perf_counter()
    candidates = generate_candidates(benchmark)
    best, ranked = score_and_select(candidates, benchmark, plc=plc)
    runtime_ms = (time.perf_counter() - t0) * 1000

    orig = next((s for s in ranked if s.name == "original"), None)
    orig_cost = orig.proxy_cost if orig is not None else None

    row = {
        "benchmark": benchmark.name,
        "valid": best.valid if best else False,
        "best_candidate": best.name if best else "none",
        "proxy_cost": best.proxy_cost if best else None,
        "original_cost": orig_cost,
        "delta": best.delta_vs_original if best else None,
        "num_candidates": len(ranked),
        "num_valid": sum(1 for s in ranked if s.valid),
        "runtime_ms": round(runtime_ms, 1),
        "overlaps": best.num_overlaps if best else -1,
        "oob": best.num_out_of_bounds if best else -1,
    }

    if show_candidates:
        row["candidates"] = [
            {
                "rank": k + 1,
                "name": s.name,
                "family": s.family,
                "valid": s.valid,
                "proxy_cost": round(s.proxy_cost, 6) if s.proxy_cost is not None else None,
                "delta": round(s.delta_vs_original, 6) if s.delta_vs_original is not None else None,
                "runtime_ms": round(s.total_ms, 1),
            }
            for k, s in enumerate(ranked)
        ]

    return row


def _print_table(rows: List[Dict]) -> None:
    """Print ASCII results table to console."""
    if not rows:
        print("  (no results)")
        return
    header = f"{'Benchmark':<25} {'Valid':>5} {'Best':>35} {'Cost':>8} {'Orig':>8} {'Delta':>8} {'#Cands':>6} {'ms':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cost = f"{r['proxy_cost']:.4f}" if r["proxy_cost"] is not None else "N/A"
        orig = f"{r['original_cost']:.4f}" if r["original_cost"] is not None else "N/A"
        delta = f"{r['delta']:+.4f}" if r["delta"] is not None else "N/A"
        print(
            f"{r['benchmark']:<25} {str(r['valid']):>5} {r['best_candidate']:>35} "
            f"{cost:>8} {orig:>8} {delta:>8} {r['num_candidates']:>6} {r['runtime_ms']:>7.1f}"
        )


def run_profile(
    profile_name: str,
    benchmark_names: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> List[Dict]:
    """Run a named profile. Returns list of result dicts."""
    if profile_name not in _PROFILES:
        print(f"Unknown profile '{profile_name}'. Available: {list(_PROFILES.keys())}")
        return []

    cfg = _PROFILES[profile_name]
    names = benchmark_names or cfg["benchmarks"]
    show_candidates = cfg.get("show_candidates", False)
    pt_paths = _discover_benchmarks(names)

    print(f"\n{'='*65}")
    print(f"Profile: {profile_name} — {cfg['description']}")
    print(f"Benchmarks: {len(pt_paths)}  Repeats: {cfg['repeat']}")
    print(f"{'='*65}")

    if not pt_paths:
        print("  No benchmarks found. Check BENCHMARKS_PT_DIR in config.py.")
        return []

    all_rows: List[Dict] = []
    for pt_path in pt_paths:
        benchmark, plc = _load_benchmark(pt_path)
        if benchmark is None:
            continue

        for rep in range(cfg["repeat"]):
            try:
                row = _run_one(benchmark, plc, show_candidates=show_candidates)
                if cfg["repeat"] > 1:
                    row["repeat"] = rep
                all_rows.append(row)
                if "console" in cfg["output"]:
                    cost_str = f"{row['proxy_cost']:.4f}" if row["proxy_cost"] else "N/A"
                    print(
                        f"  [{row['benchmark']:20s}] valid={row['valid']}  "
                        f"best={row['best_candidate']:30s}  cost={cost_str:>8}  "
                        f"{row['runtime_ms']:.0f}ms"
                    )
            except Exception as exc:
                print(f"  ERROR {pt_path.name} rep{rep}: {exc}")

    if "console" in cfg["output"]:
        print()
        _print_table(all_rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if "csv" in cfg["output"]:
            csv_path = out_dir / f"run_{profile_name}.csv"
            # Flatten candidates list before CSV
            csv_rows = [{k: v for k, v in r.items() if k != "candidates"} for r in all_rows]
            save_csv(csv_rows, csv_path)
            print(f"\nCSV: {csv_path}")
        if "json" in cfg["output"]:
            json_path = out_dir / f"run_{profile_name}.json"
            save_json({"profile": profile_name, "results": all_rows}, json_path)
            print(f"JSON: {json_path}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        prog="run_benchmarks",
        description="M2A/M2B benchmark runner.",
    )
    parser.add_argument(
        "--profile",
        default="smoke",
        choices=list(_PROFILES.keys()),
        help="Run profile (default: smoke)",
    )
    parser.add_argument(
        "-b", "--benchmark",
        action="append",
        dest="benchmarks",
        metavar="NAME",
        help="Override benchmark names (can be repeated)",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Output directory for CSV/JSON (default: artifacts/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "artifacts"
    )
    run_profile(args.profile, benchmark_names=args.benchmarks, out_dir=out_dir)


if __name__ == "__main__":
    main()
