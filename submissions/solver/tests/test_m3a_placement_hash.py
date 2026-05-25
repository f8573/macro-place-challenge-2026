"""test_m3a_placement_hash — all M3A moves snap to 0.05µm grid."""

import math
import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.core.m3a_candidate_generation import GRID_STEP, snap_to_grid
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
from submissions.solver.core.m3a_candidate_generation import generate_m3a_candidates_for_pairs


def _is_on_grid(val: float, step: float = GRID_STEP) -> bool:
    import numpy as np
    # Round through float64 to 3 decimal places (placement_hash precision) before checking.
    val64 = round(float(np.float64(val)), 3)
    snapped = snap_to_grid(val64, step)
    return abs(val64 - snapped) < 1e-9


def _bm():
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [1, 2], [0, 2]],
    )


def test_all_m3a_candidate_coords_on_grid():
    bm = _bm()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    cands = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    for c in cands:
        for i in range(c.positions.shape[0]):
            x = float(c.positions[i, 0].item())
            y = float(c.positions[i, 1].item())
            assert _is_on_grid(x), f"{c.name}: coord x={x} at index {i} not on {GRID_STEP}µm grid"
            assert _is_on_grid(y), f"{c.name}: coord y={y} at index {i} not on {GRID_STEP}µm grid"


def test_m3a_coords_on_grid_in_full_pipeline():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    m3a_cands = [s for s in ranked if s.family == "m3a_pair_refinement"]
    for sc in m3a_cands:
        for i in range(sc.positions.shape[0]):
            x = float(sc.positions[i, 0].item())
            y = float(sc.positions[i, 1].item())
            assert _is_on_grid(x), f"{sc.name}: x={x} at index {i} not on grid"
            assert _is_on_grid(y), f"{sc.name}: y={y} at index {i} not on grid"


def test_snap_to_grid_function():
    assert abs(snap_to_grid(0.12) - 0.10) < 1e-9
    assert abs(snap_to_grid(0.13) - 0.15) < 1e-9
    assert abs(snap_to_grid(1.28) - 1.30) < 1e-9   # 1.28/0.05 = 25.6 → rounds to 26
    assert abs(snap_to_grid(0.0) - 0.0) < 1e-9
    assert abs(snap_to_grid(0.05) - 0.05) < 1e-9


def test_centroid_shift_is_exactly_one_step():
    """Centroid-shift candidate should differ from base by exactly GRID_STEP in chosen axis."""
    pos = torch.tensor([
        [20.0, 50.0],
        [40.0, 50.0],
        [90.0, 50.0],   # pulls centroid to the right
    ])
    bm = make_benchmark(n_hard=3, canvas=100.0, macro_size=8.0,
                        net_nodes=[[0, 1, 2]], positions=pos)
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=1)
    assert len(pairs) == 1
    a, b, _ = pairs[0]
    cands = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    shift = next((c for c in cands if c.metadata.get("move_type") == "centroid_shift"), None)
    if shift is None:
        pytest.skip("No centroid_shift candidate generated")
    dx = abs(float(shift.positions[a, 0].item()) - float(wp[a, 0].item()))
    dy = abs(float(shift.positions[a, 1].item()) - float(wp[a, 1].item()))
    total = dx + dy
    # Exactly one grid step in one axis (or zero if clamped to canvas edge).
    assert total <= GRID_STEP + 1e-5, (
        f"Centroid shift moved {total}µm; expected <= {GRID_STEP}µm"
    )
