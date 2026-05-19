"""
Focused deterministic tests for core/hypergraph.py.

All benchmarks are synthetic - no filesystem access required.
scipy is required; tests are skipped if it is absent.
"""

import pytest
import torch

pytest.importorskip("scipy")

from macro_place.benchmark import Benchmark
from submissions.solver.core.hypergraph import clique_adjacency, macro_net_members


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_benchmark(
    n_hard: int,
    net_nodes,  # list of lists or list of tensors
    net_weights=None,
    canvas: float = 100.0,
) -> Benchmark:
    net_nodes_t = [
        (t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long))
        for t in net_nodes
    ]
    n_nets = len(net_nodes_t)
    if net_weights is None:
        net_weights = torch.ones(n_nets)
    else:
        net_weights = torch.tensor(net_weights, dtype=torch.float32)

    positions = torch.zeros(n_hard, 2)
    sizes = torch.ones(n_hard, 2)
    fixed = torch.zeros(n_hard, dtype=torch.bool)
    return Benchmark(
        name="test",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_hard,
        num_hard_macros=n_hard,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_hard)],
        num_nets=n_nets,
        net_nodes=net_nodes_t,
        net_weights=net_weights,
        grid_rows=8,
        grid_cols=8,
    )


# ---------------------------------------------------------------------------
# macro_net_members
# ---------------------------------------------------------------------------


def test_members_basic_two_pin():
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1], [2, 3]])
    members = macro_net_members(bm)
    assert len(members) == 2
    pins_0, w_0 = members[0]
    assert set(pins_0.tolist()) == {0, 1}
    assert w_0 == pytest.approx(1.0)


def test_members_omits_single_pin_nets():
    # net with only one hard pin - no clique edge possible
    bm = _make_benchmark(n_hard=3, net_nodes=[[0], [1, 2]])
    members = macro_net_members(bm)
    assert len(members) == 1
    pins, _ = members[0]
    assert set(pins.tolist()) == {1, 2}


def test_members_omits_empty_nets():
    bm = _make_benchmark(n_hard=2, net_nodes=[[], [0, 1]])
    members = macro_net_members(bm)
    assert len(members) == 1


def test_members_three_pin_net():
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1, 2]])
    members = macro_net_members(bm)
    assert len(members) == 1
    pins, _ = members[0]
    assert pins.numel() == 3


def test_members_net_weight_preserved():
    bm = _make_benchmark(n_hard=3, net_nodes=[[0, 1], [1, 2]], net_weights=[2.5, 0.5])
    members = macro_net_members(bm)
    assert len(members) == 2
    _, w0 = members[0]
    _, w1 = members[1]
    assert w0 == pytest.approx(2.5)
    assert w1 == pytest.approx(0.5)


def test_members_no_nets():
    bm = _make_benchmark(n_hard=4, net_nodes=[])
    assert macro_net_members(bm) == []


def test_members_all_soft_macro_nets():
    # n_hard=2, net has only index 2 (soft macro) - should be omitted
    bm = _make_benchmark(n_hard=2, net_nodes=[[2, 3]])
    assert macro_net_members(bm) == []


# ---------------------------------------------------------------------------
# clique_adjacency - structure
# ---------------------------------------------------------------------------


def test_clique_adj_shape():
    bm = _make_benchmark(n_hard=5, net_nodes=[[0, 1], [2, 3]])
    adj = clique_adjacency(bm)
    assert adj.shape == (5, 5)


def test_clique_adj_symmetric():
    import numpy as np

    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1, 2], [1, 3]])
    adj = clique_adjacency(bm)
    diff = (adj - adj.T).data
    assert np.allclose(diff, 0.0)


def test_clique_adj_diagonal_zero():
    import numpy as np

    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1], [1, 2], [2, 3]])
    adj = clique_adjacency(bm)
    assert np.allclose(adj.diagonal(), 0.0)


def test_clique_adj_no_nets():
    bm = _make_benchmark(n_hard=3, net_nodes=[])
    adj = clique_adjacency(bm)
    assert adj.shape == (3, 3)
    assert adj.nnz == 0


# ---------------------------------------------------------------------------
# clique_adjacency - weight normalization
# ---------------------------------------------------------------------------


def test_clique_adj_two_pin_weight():
    # 2-pin net with weight 1 -> edge weight = 1/(2-1) = 1.0
    bm = _make_benchmark(n_hard=2, net_nodes=[[0, 1]], net_weights=[1.0])
    adj = clique_adjacency(bm)
    assert adj[0, 1] == pytest.approx(1.0)
    assert adj[1, 0] == pytest.approx(1.0)


def test_clique_adj_three_pin_normalization():
    # 3-pin net with weight 1 -> each of the 3 clique edges gets 1/(3-1) = 0.5
    bm = _make_benchmark(n_hard=3, net_nodes=[[0, 1, 2]], net_weights=[1.0])
    adj = clique_adjacency(bm)
    assert adj[0, 1] == pytest.approx(0.5)
    assert adj[0, 2] == pytest.approx(0.5)
    assert adj[1, 2] == pytest.approx(0.5)


def test_clique_adj_accumulates_multiple_nets():
    # Two nets sharing edge (0,1) -> weights add
    bm = _make_benchmark(n_hard=3, net_nodes=[[0, 1], [0, 1]], net_weights=[1.0, 1.0])
    adj = clique_adjacency(bm)
    assert adj[0, 1] == pytest.approx(2.0)


def test_clique_adj_nonuniform_weights():
    bm = _make_benchmark(n_hard=3, net_nodes=[[0, 1], [1, 2]], net_weights=[3.0, 0.5])
    adj = clique_adjacency(bm)
    assert adj[0, 1] == pytest.approx(3.0)
    assert adj[1, 2] == pytest.approx(0.5)
    assert adj[0, 2] == pytest.approx(0.0)


def test_clique_adj_non_negative():
    import numpy as np

    bm = _make_benchmark(
        n_hard=4,
        net_nodes=[[0, 1, 2], [1, 2, 3], [0, 3]],
        net_weights=[1.0, 2.0, 0.5],
    )
    adj = clique_adjacency(bm)
    assert np.all(adj.data >= 0.0)
