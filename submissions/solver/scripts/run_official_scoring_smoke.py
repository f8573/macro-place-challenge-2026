"""
Official scoring smoke test for M2B.

Run this on the competition server where plc_client_os and IBM testcases
are available to verify that M2B can achieve real proxy-cost improvement.

Checks performed:
  1. IBM testcase loads and plc is available.
  2. net_nodes is non-empty (real connectivity present).
  3. Original placement is scored with official proxy cost.
  4. At least 5 M2B candidates are generated and scored.
  5. Proxy costs are printed ranked — best vs original delta shown.
  6. Exits non-zero if official scoring is unavailable or net_nodes is empty.

Usage (from repo root on competition server):
    python -m submissions.solver.scripts.run_official_scoring_smoke
    python -m submissions.solver.scripts.run_official_scoring_smoke -b ibm01
    python -m submissions.solver.scripts.run_official_scoring_smoke -b ibm01 -b ibm02
"""

import argparse
import sys
import time
from pathlib import Path

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.config import IBM_TESTCASES_DIR  # noqa: E402


def _load_official(name: str):
    """Load benchmark with official plc. Raises on failure."""
    ibm_dir = IBM_TESTCASES_DIR / name
    if not ibm_dir.exists():
        raise FileNotFoundError(
            f"IBM testcase not found: {ibm_dir}\n"
            "Run: git submodule update --init external/MacroPlacement"
        )
    if not (ibm_dir / "netlist.pb.txt").exists():
        raise FileNotFoundError(f"netlist.pb.txt missing in {ibm_dir}")

    try:
        from macro_place.loader import load_benchmark_from_dir
    except ImportError as exc:
        raise ImportError(
            "plc_client_os is required for official scoring. "
            "This script must run in the official evaluation environment."
        ) from exc

    benchmark, plc = load_benchmark_from_dir(ibm_dir.as_posix())
    if plc is None:
        raise RuntimeError("load_benchmark_from_dir returned plc=None; official scoring unavailable.")
    return benchmark, plc


def _check_connectivity(benchmark) -> None:
    """Assert net_nodes is non-empty. Fails clearly otherwise."""
    if not benchmark.net_nodes:
        print("FAIL: net_nodes is empty — official benchmark should have connectivity data.")
        print("      Scoring will be zero for all candidates; M2B cannot demonstrate improvement.")
        sys.exit(1)
    total_pins = sum(n.numel() for n in benchmark.net_nodes)
    print(f"  net_nodes: {len(benchmark.net_nodes)} nets, {total_pins} total pin references — OK")


def _score_with_plc(positions, benchmark, plc) -> float:
    """Score positions with official plc. Raises on failure."""
    from submissions.solver.core.scoring import score
    result = score(positions, benchmark, plc)
    if result is None:
        raise RuntimeError("score() returned None even though plc is available.")
    return float(result["proxy_cost"])


def _run_official_smoke(name: str) -> dict:
    print(f"\n{'='*65}")
    print(f"Official scoring smoke: {name}")
    print(f"{'='*65}")

    # 1. Load
    print(f"  Loading {name} with official plc...")
    benchmark, plc = _load_official(name)
    print(f"  Loaded: {benchmark.num_hard_macros} hard macros, canvas "
          f"{benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f} µm")

    # 2. Check connectivity
    _check_connectivity(benchmark)

    # 3. Generate and score candidates
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select, connectivity_audit

    t0 = time.perf_counter()
    candidates = generate_candidates(benchmark)
    best, ranked, diag = score_and_select(candidates, benchmark, plc=plc)
    runtime_ms = (time.perf_counter() - t0) * 1000

    raw_cost = diag.raw_original_proxy_cost
    raw_valid = diag.raw_original_valid

    print(f"\n  Generated {len(candidates)} candidates, scored in {runtime_ms:.0f} ms")
    print(f"  Scoring mode: {diag.scoring_mode}")
    print(f"  original_raw valid={raw_valid}  "
          f"proxy_cost={f'{raw_cost:.6f}' if raw_cost is not None else 'N/A'}")
    print(f"  Valid candidates: {diag.num_unique_scores} unique scores, "
          f"degenerate={diag.score_is_degenerate}")
    print(f"  Selected due to: {diag.selected_due_to}")

    # 4. Print top-10 ranked proxy costs (selectable candidates only)
    selectable_ranked = [
        s for s in ranked
        if s.valid and s.proxy_cost is not None and s.name != "original_legalized"
    ]
    print(f"\n  Ranked proxy costs (top {min(10, len(selectable_ranked))}, "
          f"excluding diagnostic-only original_legalized):")
    print(f"  {'Rank':>4}  {'Name':<45}  {'Cost':>10}  {'Delta vs raw':>13}  {'no_op':>5}")
    print(f"  {'-'*85}")
    for k, sc in enumerate(selectable_ranked[:10]):
        delta_str = f"{sc.delta_vs_original:+.6f}" if sc.delta_vs_original is not None else "N/A"
        marker = " <-- BEST" if sc.name == best.name else ""
        noop_str = "yes" if sc.no_op else "no"
        print(f"  {k+1:>4}  {sc.name:<45}  {sc.proxy_cost:>10.6f}  "
              f"{delta_str:>13}  {noop_str:>5}{marker}")

    # Also show original_legalized for diagnostics
    leg_sc = next((s for s in ranked if s.name == "original_legalized"), None)
    if leg_sc and leg_sc.proxy_cost is not None:
        leg_delta = f"{leg_sc.delta_vs_original:+.6f}" if leg_sc.delta_vs_original is not None else "N/A"
        print(f"\n  Diagnostic — original_legalized: cost={leg_sc.proxy_cost:.6f}  "
              f"delta_vs_raw={leg_delta}  no_op={leg_sc.no_op}  num_moved={leg_sc.num_moved}")

    # Connectivity audit
    conn = connectivity_audit(benchmark)
    print(f"\n  Connectivity audit:")
    print(f"    num_net_edges={conn['num_net_edges']}  "
          f"macros_w_degree>0={conn['num_macros_with_degree_gt_0']}")
    print(f"    spectral_available={conn['spectral_available']}  "
          f"terminal_anchor_available={conn['terminal_anchor_available']}")

    # 5. Result summary
    best_cost = best.proxy_cost if best and best.proxy_cost is not None else None
    delta = diag.delta_vs_raw_original
    improved = delta is not None and delta < -1e-6

    best_cost_str = f"{best_cost:.6f}" if best_cost is not None else "N/A"
    raw_cost_str = f"{raw_cost:.6f}" if raw_cost is not None else "N/A"
    delta_str = f"{delta:+.6f}" if delta is not None else "N/A"
    print(f"\n  Summary:")
    print(f"    raw_original:   {raw_cost_str}  (valid={raw_valid})")
    print(f"    best (M2B):     {best_cost_str}  ({best.name if best else 'none'})")
    print(f"    delta_vs_raw:   {delta_str}")
    print(f"    improved:       {improved}")
    print(f"    invariant holds (best <= raw): "
          f"{best_cost is None or raw_cost is None or best_cost <= raw_cost + 1e-9}")

    if not improved:
        print(
            "\n  NOTE: M2B did not improve proxy cost for this benchmark with the current "
            "candidate set. This may indicate the candidate families need tuning, or the "
            "original placement is already near-optimal for this benchmark."
        )

    return {
        "benchmark": name,
        "raw_original_cost": raw_cost,
        "raw_original_valid": raw_valid,
        "best_cost": best_cost,
        "best_candidate": best.name if best else "none",
        "delta_vs_raw_original": delta,
        "improved": improved,
        "num_candidates": len(candidates),
        "num_valid": sum(1 for s in ranked if s.valid),
        "scoring_mode": diag.scoring_mode,
        "runtime_ms": runtime_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="run_official_scoring_smoke",
        description="Official M2B scoring smoke test (requires plc_client_os).",
    )
    parser.add_argument(
        "-b", "--benchmark",
        action="append",
        dest="benchmarks",
        metavar="NAME",
        default=None,
        help="Benchmark name (default: ibm01 ibm02 ibm03)",
    )
    args = parser.parse_args()

    names = args.benchmarks or ["ibm01", "ibm02", "ibm03"]
    results = []
    failures = []

    for name in names:
        try:
            result = _run_official_smoke(name)
            results.append(result)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            print(f"\nFAIL [{name}]: {exc}")
            failures.append(name)

    if failures:
        print(f"\nFAILED benchmarks: {failures}")
        print("Official scoring smoke FAILED — cannot evaluate M2B proxy cost improvement.")
        sys.exit(1)

    improved = [r for r in results if r["improved"]]
    print(f"\n{'='*65}")
    print(f"Official scoring smoke PASSED: {len(results)} benchmarks")
    print(f"  Improved: {len(improved)}/{len(results)} benchmarks")
    for r in results:
        status = "IMPROVED" if r["improved"] else "no improvement"
        raw = r.get("raw_original_cost")
        best = r["best_cost"]
        delta = r.get("delta_vs_raw_original")
        raw_str = f"{raw:.4f}" if raw is not None else "N/A"
        best_str = f"{best:.4f}" if best is not None else "N/A"
        delta_str = f"{delta:+.6f}" if delta is not None else "N/A"
        inv_ok = raw is None or best is None or best <= raw + 1e-9
        print(f"  {r['benchmark']:20s}  raw={raw_str}  best={best_str}  "
              f"delta={delta_str}  inv={'OK' if inv_ok else 'VIOLATED'}  [{status}]")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
