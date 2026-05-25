"""
M3A pair enumeration: build top-K net-coupled macro pairs from the M2B winner.

A pair (a, b) satisfies:
  - Both a and b are movable hard macros (not fixed-hard).
  - a < b  (stable macro-id ordering — pair orientation is always (smaller, larger)).
  - Their shared-net count > 0.

Ranking: descending shared-net count, tie-broken by (a, b) ascending for determinism.
"""

from typing import List, Tuple

import torch

from macro_place.benchmark import Benchmark

# (a, b, shared_net_count)
MacroPair = Tuple[int, int, int]


def enumerate_net_coupled_pairs(
    benchmark: Benchmark,
    top_k: int,
) -> List[MacroPair]:
    """Return up to top_k macro pairs ranked by shared-net count.

    Only movable hard macros participate.  Fixed-hard macros are obstacles
    and are never included in a pair regardless of net connectivity.

    Determinism guarantee: given the same benchmark and top_k, the returned
    list is identical across calls (sort is stable; tie-break by (a, b)).
    """
    if top_k <= 0:
        return []

    n_hard = benchmark.num_hard_macros
    if n_hard < 2:
        return []

    fixed_arr = benchmark.macro_fixed[:n_hard]
    movable_set: set = {int(i) for i in range(n_hard) if not bool(fixed_arr[i].item())}

    if len(movable_set) < 2:
        return []

    pair_count: dict = {}

    for net_nodes in benchmark.net_nodes:
        # Collect unique movable hard macro ids in this net.
        seen_in_net: List[int] = []
        seen_ids: set = set()
        for n in net_nodes.tolist():
            ni = int(n)
            if ni < n_hard and ni in movable_set and ni not in seen_ids:
                seen_in_net.append(ni)
                seen_ids.add(ni)

        # Count each distinct pair in this net.
        for i in range(len(seen_in_net)):
            for j in range(i + 1, len(seen_in_net)):
                a = min(seen_in_net[i], seen_in_net[j])
                b = max(seen_in_net[i], seen_in_net[j])
                pair_count[(a, b)] = pair_count.get((a, b), 0) + 1

    if not pair_count:
        return []

    # Sort: (-count, a, b) — deterministic under any top_k.
    sorted_pairs = sorted(
        pair_count.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )

    return [(a, b, cnt) for (a, b), cnt in sorted_pairs[:top_k]]
