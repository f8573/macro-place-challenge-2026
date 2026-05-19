"""
Spectral graph inspection script -- Milestone 2A.

Builds the macro-net clique adjacency, computes the graph Laplacian, diagnoses
connected components, and prints the k smallest eigenvalues.

Usage (from repo root):
    uv run python -m submissions.solver.scripts.inspect_spectral_graph -b ibm01
    uv run python -m submissions.solver.scripts.inspect_spectral_graph --pt benchmarks/processed/public/ibm01.pt
    uv run python -m submissions.solver.scripts.inspect_spectral_graph -b ibm01 -k 10

Loading priority:
    1. IBM testcase directory (requires TILOS submodule)
    2. Pre-processed .pt file under benchmarks/processed/public/
    -b flag falls back to .pt automatically when the submodule is absent.
"""

import argparse
import sys
from pathlib import Path

from submissions.solver.config import BENCHMARKS_PT_DIR, IBM_TESTCASES_DIR


def _load_benchmark_class():
    from macro_place.benchmark import Benchmark

    return Benchmark


def _load_spectral_helpers():
    from submissions.solver.core.hypergraph import clique_adjacency, macro_net_members
    from submissions.solver.core.laplacian import graph_laplacian, normalized_laplacian
    from submissions.solver.core.spectral import (
        connected_components,
        spectral_eigenvectors,
    )

    return (
        connected_components,
        spectral_eigenvectors,
        clique_adjacency,
        macro_net_members,
        graph_laplacian,
        normalized_laplacian,
    )


def _load(name: str):
    """Load by name: IBM testcase dir first, then pre-processed .pt fallback."""
    ibm_dir = IBM_TESTCASES_DIR / name
    if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
        from submissions.solver._env import require_submodule

        require_submodule()
        from macro_place.loader import load_benchmark_from_dir

        benchmark, _ = load_benchmark_from_dir(ibm_dir.as_posix())
        return benchmark

    pt_path = BENCHMARKS_PT_DIR / f"{name}.pt"
    if pt_path.exists():
        print(f"  (submodule absent - loading from {pt_path})")
        Benchmark = _load_benchmark_class()
        return Benchmark.load(str(pt_path))

    raise FileNotFoundError(
        f"Benchmark '{name}' not found.\n"
        f"  Tried: {ibm_dir}\n"
        f"  Tried: {pt_path}\n"
        "Run: git submodule update --init external/MacroPlacement"
    )


def _print_separator(width: int = 60) -> None:
    print("=" * width)


def run(benchmark, k: int) -> None:
    (
        connected_components,
        spectral_eigenvectors,
        clique_adjacency,
        macro_net_members,
        graph_laplacian,
        normalized_laplacian,
    ) = _load_spectral_helpers()

    name = benchmark.name
    n_hard = benchmark.num_hard_macros

    _print_separator()
    print(f"inspect_spectral_graph -- {name}")
    _print_separator()
    print(f"  Hard macros  : {n_hard}")
    print(f"  Total nets   : {benchmark.num_nets}")

    # -- Net membership stats --------------------------------------------------
    nets = macro_net_members(benchmark)
    n_clique_nets = len(nets)
    total_pins = sum(t.numel() for t, _ in nets)
    avg_pins = total_pins / n_clique_nets if n_clique_nets else 0.0
    print(f"  Clique nets  : {n_clique_nets}  (nets with >= 2 hard pins)")
    print(f"  Avg pins/net : {avg_pins:.2f}")

    # -- Adjacency -------------------------------------------------------------
    print()
    print("Building clique adjacency...")
    adj = clique_adjacency(benchmark)
    nnz = adj.nnz
    density = nnz / (n_hard * n_hard) if n_hard > 0 else 0.0
    print(f"  Adjacency    : {n_hard} x {n_hard}  nnz={nnz}  density={density:.4%}")

    # -- Laplacian -------------------------------------------------------------
    L = graph_laplacian(adj)
    L_norm = normalized_laplacian(adj)
    print(f"  Laplacian    : {L.shape}  nnz={L.nnz}")
    print(f"  L_norm       : {L_norm.shape}  nnz={L_norm.nnz}")

    # -- Connected components --------------------------------------------------
    print()
    print("Connected components...")
    import numpy as np

    n_comp, labels = connected_components(adj)
    comp_sizes = np.bincount(labels)
    largest = int(comp_sizes.max())
    isolated = int((comp_sizes == 1).sum())
    print(f"  Components   : {n_comp}")
    print(f"  Largest CC   : {largest} nodes")
    print(f"  Isolated     : {isolated} nodes (degree-0 in clique graph)")

    # -- Spectral eigenvalues --------------------------------------------------
    k_eff = min(k, n_hard - 1) if n_hard > 1 else 0
    if k_eff <= 0:
        print()
        print("  (too few nodes for eigensolve)")
        _print_separator()
        return

    print()
    print(f"Smallest {k_eff} eigenvalues of L_norm...")
    vals, _vecs = spectral_eigenvectors(L_norm, k=k_eff)

    for i, v in enumerate(vals):
        marker = "  <-- Fiedler" if i == 1 else ""
        print(f"  lambda[{i}] = {v:.6f}{marker}")

    algebraic_connectivity = float(vals[1]) if len(vals) > 1 else 0.0
    print()
    print(f"  Algebraic connectivity (lambda[1]) : {algebraic_connectivity:.6f}")
    if algebraic_connectivity < 1e-8:
        print("  WARNING: graph is disconnected or near-disconnected")

    _print_separator()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inspect_spectral_graph",
        description="M2A: inspect macro-net spectral graph structure.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--benchmark", default="ibm01", help="Benchmark name (default: ibm01)")
    group.add_argument("--pt", metavar="PATH", help="Load from explicit .pt file")
    parser.add_argument(
        "-k",
        type=int,
        default=6,
        help="Number of smallest eigenvalues to compute (default: 6)",
    )
    args = parser.parse_args()

    try:
        import scipy  # noqa: F401
    except ImportError:
        print(
            "Error: scipy is required for spectral analysis.\n"
            "Install with: pip install 'macro-place[baselines]'"
        )
        sys.exit(1)

    Benchmark = _load_benchmark_class()

    if args.pt:
        pt_path = Path(args.pt)
        if not pt_path.exists():
            print(f"Error: .pt file not found: {pt_path}")
            sys.exit(1)
        benchmark = Benchmark.load(str(pt_path))
    else:
        try:
            benchmark = _load(args.benchmark)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    run(benchmark, k=args.k)


if __name__ == "__main__":
    main()
