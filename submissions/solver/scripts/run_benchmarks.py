"""
Benchmark runner with profiles for M2A/M2B.

Profiles:
  smoke          — 1-3 small benchmarks, console only
  standard       — all public benchmarks, console + CSV
  heavy          — all benchmarks, repeat x3, console + CSV + JSON
  m2b-prep       — all benchmarks with full candidate budget, ranked output
  audit          — ibm01/02/03 with candidate diversity + connectivity diagnostics
  official-smoke — run on competition server with plc_client_os available
  stress         — all benchmarks, 3x repeat, JSON only

Usage (from repo root):
    python -m submissions.solver.scripts.run_benchmarks --profile smoke
    python -m submissions.solver.scripts.run_benchmarks --profile audit
    python -m submissions.solver.scripts.run_benchmarks --profile standard --out results/
    python -m submissions.solver.scripts.run_benchmarks --profile m2b-prep -b ibm01
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
        "benchmarks": None,
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
    "audit": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console"],
        "description": "Candidate diversity + connectivity diagnostics for ibm01/02/03",
        "show_candidates": True,
        "show_audit": True,
    },
    "official-smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "Official scoring smoke — requires IBM testcases + plc_client_os",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
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


def _get_benchmark_class():
    """Import Benchmark, bypassing __init__.py if plc_client_os is missing."""
    try:
        from macro_place.benchmark import Benchmark
        return Benchmark
    except ImportError:
        pass
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "macro_place.benchmark",
        Path(__file__).resolve().parent.parent.parent.parent / "macro_place" / "benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    if "macro_place" not in sys.modules:
        sys.modules["macro_place"] = types.ModuleType("macro_place")
    sys.modules["macro_place.benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod.Benchmark


def _load_benchmark(pt_path: Path, require_official: bool = False):
    """Load benchmark from .pt file. Returns (benchmark, plc=None).

    When require_official=True, also tries IBM testcases for official plc.
    """
    name = pt_path.stem

    if require_official:
        ibm_dir = IBM_TESTCASES_DIR / name
        if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
            try:
                from macro_place.loader import load_benchmark_from_dir
                print(f"  Loading from {ibm_dir} (official plc)")
                return load_benchmark_from_dir(ibm_dir.as_posix())
            except Exception as exc:
                print(f"  Failed to load official plc for {name}: {exc}")
        else:
            print(f"  IBM testcases not found for '{name}' — cannot use official scoring")

    try:
        Benchmark = _get_benchmark_class()
        bm = Benchmark.load(str(pt_path))
        return bm, None
    except Exception as exc:
        print(f"  Error loading {pt_path.name}: {exc}")
        return None, None


def _candidate_diversity(ranked, benchmark) -> Dict:
    """Compute candidate diversity metrics from legalized ranked candidates."""
    import torch
    from submissions.solver.core.candidate_scoring import placement_hash

    ref_pos = benchmark.macro_positions.float()
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()

    all_hashes = []
    families = set()
    displacements = []

    for sc in ranked:
        h = placement_hash(sc.positions)
        all_hashes.append((sc.name, h))
        families.add(sc.family)
        if movable_mask.any():
            disp = torch.norm(sc.positions - ref_pos, dim=1)[movable_mask]
            displacements.append((sc.name, float(disp.mean()), float(disp.max())))

    unique_hashes = {h for _, h in all_hashes}
    num_distinct = len(unique_hashes)
    all_collapse = num_distinct == 1

    avg_disp_mean = sum(d[1] for d in displacements) / max(len(displacements), 1)
    avg_disp_max = max((d[2] for d in displacements), default=0.0)

    return {
        "num_candidates": len(ranked),
        "num_valid": sum(1 for s in ranked if s.valid),
        "num_invalid": sum(1 for s in ranked if not s.valid),
        "families": sorted(families),
        "num_distinct_placements": num_distinct,
        "all_candidates_collapse_to_same": all_collapse,
        "avg_displacement_mean_um": round(avg_disp_mean, 2),
        "max_displacement_max_um": round(avg_disp_max, 2),
        "candidate_hashes": {name: h for name, h in all_hashes},
    }


def _run_one(
    benchmark,
    plc,
    show_candidates: bool = False,
    show_audit: bool = False,
) -> Dict:
    """Run M2B pipeline on one benchmark. Return result dict."""
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select, connectivity_audit

    t0 = time.perf_counter()
    candidates = generate_candidates(benchmark)
    best, ranked, diag = score_and_select(candidates, benchmark, plc=plc)
    runtime_ms = (time.perf_counter() - t0) * 1000

    row = {
        "benchmark": benchmark.name,
        "valid": best.valid if best else False,
        "best_candidate": best.name if best else "none",
        "proxy_cost": best.proxy_cost if best else None,
        "raw_original_cost": diag.raw_original_proxy_cost,
        "raw_original_valid": diag.raw_original_valid,
        "delta_vs_raw_original": diag.delta_vs_raw_original,
        "num_candidates": len(ranked),
        "num_valid": sum(1 for s in ranked if s.valid),
        "runtime_ms": round(runtime_ms, 1),
        "overlaps": best.num_overlaps if best else -1,
        "oob": best.num_out_of_bounds if best else -1,
        # Scoring diagnostics
        "scoring_available": diag.scoring_available,
        "scoring_mode": diag.scoring_mode,
        "score_is_degenerate": diag.score_is_degenerate,
        "num_unique_scores": diag.num_unique_scores,
        "selected_due_to": diag.selected_due_to,
    }

    if show_candidates:
        row["candidates"] = [
            {
                "rank": k + 1,
                "name": s.name,
                "family": s.family,
                "valid": s.valid,
                "no_op": s.no_op,
                "proxy_cost": round(s.proxy_cost, 6) if s.proxy_cost is not None else None,
                "delta_vs_raw_original": round(s.delta_vs_original, 6) if s.delta_vs_original is not None else None,
                "num_moved": s.num_moved,
                "runtime_ms": round(s.total_ms, 1),
            }
            for k, s in enumerate(ranked)
        ]

    if show_audit:
        row["connectivity"] = connectivity_audit(benchmark)
        row["diversity"] = _candidate_diversity(ranked, benchmark)

    return row


def _print_table(rows: List[Dict]) -> None:
    if not rows:
        print("  (no results)")
        return
    header = (
        f"{'Benchmark':<25} {'Valid':>5} {'Best':>35} {'Cost':>8} "
        f"{'Orig':>8} {'Delta':>8} {'Mode':>12} {'Sel':>16} {'ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        cost = f"{r['proxy_cost']:.4f}" if r["proxy_cost"] is not None else "N/A"
        orig = f"{r['raw_original_cost']:.4f}" if r.get("raw_original_cost") is not None else "N/A"
        delta = f"{r['delta_vs_raw_original']:+.4f}" if r.get("delta_vs_raw_original") is not None else "N/A"
        mode = r.get("scoring_mode", "?")
        sel = r.get("selected_due_to", "?")
        print(
            f"{r['benchmark']:<25} {str(r['valid']):>5} {r['best_candidate']:>35} "
            f"{cost:>8} {orig:>8} {delta:>8} {mode:>12} {sel:>16} {r['runtime_ms']:>7.1f}"
        )


def _print_audit(row: Dict) -> None:
    """Print candidate diversity and connectivity diagnostics for one benchmark."""
    bm = row["benchmark"]
    print(f"\n  --- Audit: {bm} ---")

    conn = row.get("connectivity")
    if conn:
        print(f"  Connectivity:")
        print(f"    num_macros={conn['num_macros']}  num_nets={conn['num_nets']}")
        print(f"    num_net_edges={conn['num_net_edges']}  "
              f"macros_with_degree>0={conn['num_macros_with_degree_gt_0']}")
        print(f"    num_fixed_endpoints={conn['num_fixed_endpoints']}")
        print(f"    spectral_available={conn['spectral_available']}  "
              f"terminal_anchor_available={conn['terminal_anchor_available']}")

    div = row.get("diversity")
    if div:
        print(f"  Candidate Diversity:")
        print(f"    total={div['num_candidates']}  "
              f"valid={div['num_valid']}  invalid={div['num_invalid']}")
        print(f"    families={div['families']}")
        print(f"    distinct_placements={div['num_distinct_placements']}  "
              f"all_collapse={div['all_candidates_collapse_to_same']}")
        print(f"    avg_displacement={div['avg_displacement_mean_um']} µm  "
              f"max_displacement={div['max_displacement_max_um']} µm")
        print(f"  Hashes:")
        for name, h in div["candidate_hashes"].items():
            print(f"    {name:<45} {h}")

    print(f"  Scoring: mode={row['scoring_mode']}  "
          f"available={row['scoring_available']}  "
          f"degenerate={row['score_is_degenerate']}  "
          f"unique_scores={row['num_unique_scores']}  "
          f"selected_due_to={row['selected_due_to']}")


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
    show_audit = cfg.get("show_audit", False)
    require_official = cfg.get("require_official", False)
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
        benchmark, plc = _load_benchmark(pt_path, require_official=require_official)
        if benchmark is None:
            continue

        if require_official and plc is None:
            print(
                f"  SKIP {pt_path.stem}: official-smoke requires plc_client_os. "
                "Run 'git submodule update --init external/MacroPlacement' then retry."
            )
            continue

        for rep in range(cfg["repeat"]):
            try:
                row = _run_one(
                    benchmark, plc,
                    show_candidates=show_candidates,
                    show_audit=show_audit,
                )
                if cfg["repeat"] > 1:
                    row["repeat"] = rep
                all_rows.append(row)
                if "console" in cfg["output"]:
                    cost_str = f"{row['proxy_cost']:.4f}" if row["proxy_cost"] is not None else "N/A"
                    print(
                        f"  [{row['benchmark']:20s}] valid={row['valid']}  "
                        f"best={row['best_candidate']:35s}  cost={cost_str:>8}  "
                        f"mode={row['scoring_mode']:12s}  {row['runtime_ms']:.0f}ms"
                    )
                    if show_audit:
                        _print_audit(row)
            except Exception as exc:
                print(f"  ERROR {pt_path.name} rep{rep}: {exc}")

    if "console" in cfg["output"]:
        print()
        _print_table(all_rows)

        # Scoring honesty summary
        unavailable = [r for r in all_rows if r.get("scoring_mode") == "unavailable"]
        degenerate = [r for r in all_rows if r.get("score_is_degenerate")]
        if unavailable:
            print(
                f"\n  NOTE: {len(unavailable)}/{len(all_rows)} benchmarks have no net connectivity "
                "(net_nodes empty). Candidate ranking is validity-only; M2B cannot claim "
                "score improvement locally. Run with --profile official-smoke on the "
                "competition server to evaluate real proxy cost."
            )
        elif degenerate:
            print(
                f"\n  NOTE: {len(degenerate)}/{len(all_rows)} benchmarks have degenerate scores "
                "(all candidates tie). 'original' selected as deterministic fallback."
            )
        else:
            print(
                f"\n  NOTE: Scoring mode is '{all_rows[0]['scoring_mode'] if all_rows else '?'}'. "
                "Score-based optimization is active."
            )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if "csv" in cfg["output"]:
            csv_path = out_dir / f"run_{profile_name}.csv"
            csv_rows = [{k: v for k, v in r.items() if k not in ("candidates", "connectivity", "diversity")} for r in all_rows]
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
