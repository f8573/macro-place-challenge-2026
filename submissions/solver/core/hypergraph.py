"""
Macro-net extraction and clique hypergraph construction.

For each net with k hard-macro pins and weight w, clique expansion adds
w / (k - 1) to every undirected pair (i, j). Dividing by (k - 1) keeps the
total contribution of one net proportional to w regardless of pin count,
preventing large high-degree nets from dominating the Laplacian spectrum.

Requires scipy (listed under [baselines] optional dependencies).
"""

from typing import List, Tuple

import numpy as np
import torch
from macro_place.benchmark import Benchmark


def macro_net_members(benchmark: Benchmark) -> List[Tuple[torch.Tensor, float]]:
    """Return (hard_pin_indices, net_weight) for each net with >= 2 hard-macro pins.

    Nets with fewer than 2 hard-macro pins contribute no clique edges and are
    omitted.  Indices reference rows of benchmark.macro_positions (0-based).
    """
    n_hard = benchmark.num_hard_macros
    result: List[Tuple[torch.Tensor, float]] = []
    for ni, nodes in enumerate(benchmark.net_nodes):
        hard_pins = torch.unique(nodes[nodes < n_hard], sorted=True)
        if hard_pins.numel() >= 2:
            w = float(benchmark.net_weights[ni].item())
            result.append((hard_pins, w))
    return result


def extract_macro_nets(
    benchmark: Benchmark, *, ignore_singletons: bool = True
) -> List[tuple[int, ...]]:
    """Compatibility wrapper returning only hard-macro membership per net."""
    min_pins = 2 if ignore_singletons else 1
    n_hard = benchmark.num_hard_macros
    macro_nets: List[tuple[int, ...]] = []
    for nodes in benchmark.net_nodes:
        hard_pins = torch.unique(nodes[nodes < n_hard], sorted=True)
        if hard_pins.numel() >= min_pins:
            macro_nets.append(tuple(int(pin) for pin in hard_pins.tolist()))
    return macro_nets


def clique_adjacency(benchmark: Benchmark):
    """Build normalized clique sparse adjacency for hard macros.

    For each net with k hard-macro pins and weight w, accumulates w / (k - 1)
    on every symmetric pair (i, j).  Duplicate pairs from multiple nets are
    summed.

    Args:
        benchmark: official Benchmark object.

    Returns:
        scipy.sparse.csr_matrix of shape (n_hard, n_hard), dtype float64.
        The matrix is symmetric with non-negative entries.

    Raises:
        ImportError: if scipy is not installed.
    """
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spectral helpers. "
            "Install with: pip install 'macro-place[baselines]'"
        ) from exc

    n = benchmark.num_hard_macros
    nets = macro_net_members(benchmark)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for hard_pins, w in nets:
        k = hard_pins.numel()
        norm = w / (k - 1)
        pins = hard_pins.tolist()
        for a in range(k):
            for b in range(a + 1, k):
                i, j = pins[a], pins[b]
                rows.append(i)
                cols.append(j)
                vals.append(norm)
                rows.append(j)
                cols.append(i)
                vals.append(norm)

    if not rows:
        from scipy.sparse import csr_matrix

        return csr_matrix((n, n), dtype=np.float64)

    adj = coo_matrix(
        (
            np.array(vals, dtype=np.float64),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(n, n),
    ).tocsr()
    return adj
