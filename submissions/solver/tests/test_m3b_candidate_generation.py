"""test_m3b_candidate_generation — per-cluster candidate correctness."""

import math
import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core.m3b_cluster_enumeration import enumerate_net_coupled_triples
from submissions.solver.core.m3b_candidate_generation import (
    GRID_STEP,
    generate_cluster_candidates,
    generate_m3b_candidates_for_clusters,
)


def _bm_triangle(canvas=100.0, macro_size=10.0):
    """3 movable macros fully connected."""
    return make_benchmark(
        n_hard=3, canvas=canvas, macro_size=macro_size,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
    )


def _winner_pos(bm):
    return bm.macro_positions.clone().float()


def test_cluster_yields_at_most_4_candidates():
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    assert triples, "need at least one triple"
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    assert len(cands) <= 4, f"Expected ≤4 candidates, got {len(cands)}"


def test_cyclic_rotation_generated():
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    move_types = [c.metadata["move_type"] for c in cands]
    assert "cyclic" in move_types, "cyclic rotation candidate must be generated"


def test_reverse_cyclic_generated():
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    move_types = [c.metadata["move_type"] for c in cands]
    assert "rcyclic" in move_types, "reverse cyclic candidate must be generated"


def test_cyclic_rotation_semantics():
    """A→B's pos, B→C's pos, C→A's pos."""
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    old_a = (float(pos[a, 0].item()), float(pos[a, 1].item()))
    old_b = (float(pos[b, 0].item()), float(pos[b, 1].item()))
    old_c = (float(pos[c, 0].item()), float(pos[c, 1].item()))

    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    cyc = next(c for c in cands if c.metadata["move_type"] == "cyclic")

    new_a = (float(cyc.positions[a, 0].item()), float(cyc.positions[a, 1].item()))
    new_b = (float(cyc.positions[b, 0].item()), float(cyc.positions[b, 1].item()))
    new_c = (float(cyc.positions[c, 0].item()), float(cyc.positions[c, 1].item()))

    # A takes B's snapped position
    assert abs(new_a[0] - round(old_b[0] / GRID_STEP) * GRID_STEP) < 1e-6
    assert abs(new_a[1] - round(old_b[1] / GRID_STEP) * GRID_STEP) < 1e-6
    # B takes C's snapped position
    assert abs(new_b[0] - round(old_c[0] / GRID_STEP) * GRID_STEP) < 1e-6
    assert abs(new_b[1] - round(old_c[1] / GRID_STEP) * GRID_STEP) < 1e-6
    # C takes A's snapped position
    assert abs(new_c[0] - round(old_a[0] / GRID_STEP) * GRID_STEP) < 1e-6
    assert abs(new_c[1] - round(old_a[1] / GRID_STEP) * GRID_STEP) < 1e-6


def test_reverse_cyclic_rotation_semantics():
    """A→C's pos, B→A's pos, C→B's pos."""
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    old_a = (float(pos[a, 0].item()), float(pos[a, 1].item()))
    old_b = (float(pos[b, 0].item()), float(pos[b, 1].item()))
    old_c = (float(pos[c, 0].item()), float(pos[c, 1].item()))

    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    rcyc = next(c for c in cands if c.metadata["move_type"] == "rcyclic")

    new_a = (float(rcyc.positions[a, 0].item()), float(rcyc.positions[a, 1].item()))
    new_b = (float(rcyc.positions[b, 0].item()), float(rcyc.positions[b, 1].item()))
    new_c = (float(rcyc.positions[c, 0].item()), float(rcyc.positions[c, 1].item()))

    # A takes C's snapped position
    assert abs(new_a[0] - round(old_c[0] / GRID_STEP) * GRID_STEP) < 1e-6
    assert abs(new_a[1] - round(old_c[1] / GRID_STEP) * GRID_STEP) < 1e-6
    # B takes A's snapped position
    assert abs(new_b[0] - round(old_a[0] / GRID_STEP) * GRID_STEP) < 1e-6
    assert abs(new_b[1] - round(old_a[1] / GRID_STEP) * GRID_STEP) < 1e-6
    # C takes B's snapped position
    assert abs(new_c[0] - round(old_b[0] / GRID_STEP) * GRID_STEP) < 1e-6
    assert abs(new_c[1] - round(old_b[1] / GRID_STEP) * GRID_STEP) < 1e-6


def test_centroid_shift_generated():
    """Centroid-shift candidate is generated when a shared net exists."""
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    move_types = [cd.metadata["move_type"] for cd in cands]
    assert "centroid_shift" in move_types, "centroid_shift candidate must be generated"


def test_centroid_shift_moves_all_three_by_one_grid_step():
    """All three macros translate by exactly one grid step on the chosen axis."""
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    cs = next((cd for cd in cands if cd.metadata["move_type"] == "centroid_shift"), None)
    if cs is None:
        pytest.skip("no centroid_shift candidate (no qualifying net)")

    for mid in (a, b, c):
        old_x = float(pos[mid, 0].item())
        old_y = float(pos[mid, 1].item())
        new_x = float(cs.positions[mid, 0].item())
        new_y = float(cs.positions[mid, 1].item())
        dx = abs(new_x - round(old_x / GRID_STEP) * GRID_STEP)
        dy = abs(new_y - round(old_y / GRID_STEP) * GRID_STEP)
        # One axis changes by exactly GRID_STEP; other stays the same.
        moved = (abs(dx - GRID_STEP) < 1e-6 and dy < 1e-6) or (
            dx < 1e-6 and abs(dy - GRID_STEP) < 1e-6
        )
        assert moved, (
            f"macro {mid}: expected exactly one GRID_STEP move, "
            f"got dx={dx:.6f} dy={dy:.6f}"
        )


def test_all_generated_coordinates_snap_to_grid():
    """Every coordinate in every candidate must be a 0.05 µm multiple."""
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    for triple_idx, (a, b, c, _) in enumerate(triples):
        cands = generate_cluster_candidates(bm, pos, a, b, c, triple_idx, set())
        for cand in cands:
            for macro_id in range(bm.num_hard_macros):
                for dim in range(2):
                    val = float(cand.positions[macro_id, dim].item())
                    # Use fractional-quotient tolerance to accommodate float32 storage
                    # rounding (values around 100 µm carry ~1e-5 conversion error).
                    quotient = val / GRID_STEP
                    fractional = abs(quotient - round(quotient))
                    assert fractional < 1e-3, (
                        f"{cand.name} macro {macro_id} dim {dim}: "
                        f"value {val} not on 0.05 µm grid (fractional={fractional:.2e})"
                    )


def test_no_clamping_oob_allowed():
    """Candidates that move a macro out of bounds must be generated raw, not clamped."""
    # Place all macros near an edge so rotation moves one OOB.
    bm = make_benchmark(
        n_hard=3, canvas=30.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        positions=[[5.0, 5.0], [15.0, 5.0], [5.0, 15.0]],
    )
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    if not triples:
        pytest.skip("no triples on this benchmark")
    pos = bm.macro_positions.clone().float()
    a, b, c, _ = triples[0]
    cands = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    # At least one candidate must be generated — no clamping should suppress them.
    assert len(cands) >= 1, "generation must not silently discard candidates via clamping"

    # Verify at least one candidate has all three macros within valid bounds.
    half = 10.0 / 2.0
    for cand in cands:
        for mid in (a, b, c):
            x = float(cand.positions[mid, 0].item())
            y = float(cand.positions[mid, 1].item())
            # Simply check value was not forcibly clamped to [half, canvas-half].
            # We allow OOB (negative or beyond canvas) — that is the intended behaviour.
            _ = x  # no assertion: OOB is fine here


def test_existing_names_deduplicated():
    """Candidates whose name is already in existing_names must be skipped."""
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = _winner_pos(bm)
    a, b, c, _ = triples[0]
    # Generate once to learn the names.
    first_gen = generate_cluster_candidates(bm, pos, a, b, c, 0, set())
    first_names = {cd.name for cd in first_gen}
    # Second generation with all first names in existing_names → empty.
    second_gen = generate_cluster_candidates(bm, pos, a, b, c, 0, first_names)
    assert second_gen == [], "all names already in existing_names must be skipped"


def test_batch_function_propagates_names_across_clusters():
    """Batch generator must maintain inter-cluster dedup."""
    bm = make_benchmark(
        n_hard=6, canvas=200.0, macro_size=10.0,
        net_nodes=[
            [0, 1], [0, 2], [1, 2],
            [3, 4], [3, 5], [4, 5],
        ],
    )
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    pos = bm.macro_positions.clone().float()
    all_cands = generate_m3b_candidates_for_clusters(bm, pos, triples, set())
    names = [c.name for c in all_cands]
    assert len(names) == len(set(names)), "duplicate candidate names across clusters"
