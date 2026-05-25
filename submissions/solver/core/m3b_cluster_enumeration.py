"""
M3B cluster enumeration: build top-K net-coupled 3-macro clusters from the M2B/M3A winner.

A cluster (a, b, c) satisfies:
  - All of a, b, c are movable hard macros (not fixed-hard).
  - a < b < c  (stable macro-id ordering — always smallest-first canonical form).
  - Each of the three pairs (a,b), (a,c), (b,c) has shared-net count > 0.

Ranking: descending aggregate pair coupling [shared(a,b) + shared(a,c) + shared(b,c)],
tie-broken by (a, b, c) ascending for determinism.
"""

from collections import defaultdict
from typing import List, Tuple

from macro_place.benchmark import Benchmark

# (a, b, c, aggregate_shared_net_count)
MacroTriple = Tuple[int, int, int, int]


def enumerate_net_coupled_triples(
    benchmark: Benchmark,
    top_k: int,
) -> List[MacroTriple]:
    """Return up to top_k 3-macro clusters ranked by aggregate shared-net coupling.

    Only movable hard macros participate.  Fixed-hard macros are excluded.

    Determinism guarantee: given the same benchmark and top_k, the returned
    list is identical across calls (sort is stable; tie-break by (a, b, c)).
    """
    if top_k <= 0:
        return []

    n_hard = benchmark.num_hard_macros
    if n_hard < 3:
        return []

    fixed_arr = benchmark.macro_fixed[:n_hard]
    movable_set: set = {int(i) for i in range(n_hard) if not bool(fixed_arr[i].item())}

    if len(movable_set) < 3:
        return []

    # Build pair shared-net counts — same approach as M3A pair enumeration.
    pair_count: dict = {}

    for net_nodes in benchmark.net_nodes:
        seen_in_net: List[int] = []
        seen_ids: set = set()
        for n in net_nodes.tolist():
            ni = int(n)
            if ni < n_hard and ni in movable_set and ni not in seen_ids:
                seen_in_net.append(ni)
                seen_ids.add(ni)

        for i in range(len(seen_in_net)):
            for j in range(i + 1, len(seen_in_net)):
                a = min(seen_in_net[i], seen_in_net[j])
                b = max(seen_in_net[i], seen_in_net[j])
                pair_count[(a, b)] = pair_count.get((a, b), 0) + 1

    if not pair_count:
        return []

    # Build undirected adjacency from pairs with count > 0.
    adjacency: dict = defaultdict(set)
    for a, b in pair_count:
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Enumerate all canonical triples (a < b < c) where every pair is connected.
    # Iterating in sorted order guarantees a < b < c without extra checks.
    triples: dict = {}
    for a in sorted(movable_set):
        neighbors_a = adjacency.get(a, set())
        for b in sorted(neighbors_a):
            if b <= a:
                continue
            neighbors_b = adjacency.get(b, set())
            common_bc = neighbors_a & neighbors_b
            for c in sorted(common_bc):
                if c <= b:
                    continue
                # All three pair keys are in canonical form (smaller, larger) because a < b < c.
                score = (
                    pair_count.get((a, b), 0)
                    + pair_count.get((a, c), 0)
                    + pair_count.get((b, c), 0)
                )
                triples[(a, b, c)] = score

    if not triples:
        return []

    # Sort: (-score, a, b, c) — fully deterministic under any top_k.
    sorted_triples = sorted(
        triples.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    )

    return [(a, b, c, score) for (a, b, c), score in sorted_triples[:top_k]]
