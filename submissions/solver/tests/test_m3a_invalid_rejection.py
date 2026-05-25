"""test_m3a_invalid_rejection — overlapping/OOB candidates rejected before scoring."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidateScoringConfig,
)
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
from submissions.solver.core.m3a_candidate_generation import generate_pair_candidates
from submissions.solver.core.candidate_scoring import _prepare_candidate
import torch


def _bm_tight():
    """Macros packed closely so edge-align moves will often produce overlaps."""
    pos = torch.tensor([
        [10.0, 10.0],
        [22.0, 10.0],   # only 2 µm gap between macro 0 and 1
        [50.0, 50.0],
        [80.0, 80.0],
    ])
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1]],
        positions=pos,
    )


def test_invalid_candidates_not_scored():
    """Valid M3A candidates reach scoring; invalid ones do not."""
    bm = _bm_tight()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
        m3a_score_budget=100,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    candidates = generate_candidates(bm, config=gen_cfg)
    best, ranked, diag = score_and_select(candidates, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    # All M3A candidates that were scored must be valid.
    m3a_scored = [s for s in ranked if s.family == "m3a_pair_refinement" and s.was_scored]
    for sc in m3a_scored:
        assert sc.valid, f"Scored but invalid M3A candidate: {sc.name}"


def test_invalid_candidates_have_skip_reason_invalid():
    """Invalid candidates get skip_reason='invalid', not 'scored'."""
    bm = _bm_tight()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    candidates = generate_candidates(bm, config=gen_cfg)
    _best, ranked, _diag = score_and_select(candidates, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    m3a_invalid = [s for s in ranked if s.family == "m3a_pair_refinement" and not s.valid]
    for sc in m3a_invalid:
        reason = sc.metadata.get("skip_reason", "")
        assert reason == "invalid", (
            f"Invalid M3A candidate {sc.name} has skip_reason={reason!r}, expected 'invalid'"
        )


def test_oob_candidates_are_invalid():
    """A candidate that would place a macro outside the canvas is rejected."""
    # Place two macros near the edge; left-align will push one off-canvas.
    pos = torch.tensor([
        [5.0, 50.0],   # macro 0 (a) — near left wall
        [20.0, 50.0],  # macro 1 (b)
        [50.0, 50.0],
        [80.0, 80.0],
    ])
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1]],
        positions=pos,
    )
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=1)
    assert len(pairs) >= 1
    a, b, _ = pairs[0]
    cands = generate_pair_candidates(bm, wp, a, b, 0, set())
    # Validate each candidate; any invalid one should be OOB or overlap.
    movable_mask = bm.get_movable_mask() & bm.get_hard_macro_mask()
    obstacle_mask = bm.macro_fixed & bm.get_hard_macro_mask()
    for c in cands:
        sc = _prepare_candidate(c, bm, movable_mask, obstacle_mask, legalizer_max_rings=25)
        if not sc.valid:
            # Correct — invalid candidates are rejected.
            assert sc.num_out_of_bounds > 0 or sc.num_overlaps > 0, (
                f"Invalid candidate {sc.name} has no OOB or overlap"
            )


def test_oob_candidate_produced_and_rejected_not_clamped():
    """Regression: OOB M3A candidates must be generated with raw OOB coords and rejected by
    validation — not silently clamped into bounds.

    Setup: macro_b (index 1) is at cx=2, near the left wall.  Placing macro_a
    immediately LEFT of b requires cx_a = 2 - 5 - 5 = -8, which is off-canvas.
    Without clamping the position tensor holds -8 and check_placement reports
    num_out_of_bounds > 0.  With the old snap-then-clamp, the position was
    silently repaired to cx_a = 5 and validated as in-bounds.
    """
    pos = torch.tensor([
        [50.0, 50.0],   # macro 0 (a)
        [2.0,  50.0],   # macro 1 (b) — near left wall
        [50.0, 80.0],
        [80.0, 80.0],
    ])
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1]],
        positions=pos,
    )
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=1)
    assert len(pairs) >= 1, "Expected at least one pair"
    a, b, _ = pairs[0]

    cands = generate_pair_candidates(bm, wp, a, b, 0, set())
    movable_mask = bm.get_movable_mask() & bm.get_hard_macro_mask()
    obstacle_mask = bm.macro_fixed & bm.get_hard_macro_mask()

    left_cands = [c for c in cands if c.metadata.get("move_type") == "left"]
    w_a = float(bm.macro_sizes[a, 0].item())

    found_oob_in_generated = False
    for c in left_cands:
        raw_x = float(c.positions[a, 0].item())
        if raw_x < w_a / 2.0:  # center is so close to left that macro extends off-canvas
            found_oob_in_generated = True
            sc = _prepare_candidate(c, bm, movable_mask, obstacle_mask, legalizer_max_rings=25)
            assert not sc.valid, (
                f"OOB candidate {c.name} (cx={raw_x}) should be invalid after validation "
                f"but got valid=True — was it clamped back into bounds?"
            )
            assert sc.num_out_of_bounds > 0, (
                f"OOB candidate {c.name} (cx={raw_x}) expected num_out_of_bounds > 0, "
                f"got {sc.num_out_of_bounds}"
            )

    assert found_oob_in_generated, (
        "Expected a 'left' candidate with cx_a < w_a/2 (OOB) to be generated. "
        "If clamping is still active it silently repairs the coordinate and this "
        "assertion catches the regression."
    )


def test_generated_m3a_coordinates_on_grid():
    """Regression: all generated M3A coordinates must lie on the 0.05 µm grid.

    Snap-then-clamp could produce off-grid values when the clamp boundary is not
    a grid multiple.  Snap-only always keeps coordinates on-grid.

    Values are read back via float64 and rounded to 3 dp (placement-hash precision)
    to tolerate float32 storage rounding (e.g. 10.05 stored as 10.050000190...).
    """
    import numpy as np
    from submissions.solver.core.m3a_candidate_generation import snap_to_grid

    bm = _bm_tight()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=4)
    for pair_idx, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pair_idx, set())
        for c in cands:
            for i in range(bm.num_hard_macros):
                for axis in range(2):
                    val = float(c.positions[i, axis].item())
                    # Round via float64 to 3 dp to absorb float32 representation error.
                    val64 = round(float(np.float64(val)), 3)
                    snapped = snap_to_grid(val64)
                    assert abs(snapped - val64) < 1e-9, (
                        f"Candidate {c.name}: macro {i} axis {axis} "
                        f"coord {val} (rounded {val64}) is not on 0.05 µm grid "
                        f"(off by {abs(snapped - val64)})"
                    )


def test_generated_m3a_coordinates_on_grid_non_aligned_boundary():
    """Regression: OOB coords at non-grid-aligned canvas boundaries must remain on-grid.

    With macro_size=10.03, w/2=5.015 and max_valid_cx = 100-5.015 = 94.985, which is
    NOT a 0.05 µm grid multiple (94.985/0.05 = 1899.7).  Under old snap-then-clamp:
      snap(99.03) = 99.05, clamp(99.05, max=94.985) = 94.985  ← off grid
    Under new snap-only:
      snap(99.03) = 99.05  ← on grid, OOB, rejected by validation

    The test verifies:
    - All generated coordinates (valid and OOB) are on the 0.05 µm grid.
    - A candidate with an OOB coordinate is generated (right-of-b move places
      macro_a past the canvas boundary).
    - That OOB candidate is rejected by validation (num_out_of_bounds > 0).
    - Valid candidates are also on-grid (sanity check).
    """
    import numpy as np
    from submissions.solver.core.m3a_candidate_generation import snap_to_grid

    # macro_size=10.03 → w/2=5.015 → canvas-w/2=94.985 (NOT a grid multiple).
    # macro 1 at x=89 is near the right edge: a "right" move places macro 0 at
    # x ≈ 89+5.015+5.015=99.03 → snapped to 99.05 > 94.985 (OOB).
    bm = make_benchmark(
        n_hard=4,
        canvas=100.0,
        macro_size=10.03,
        net_nodes=[[0, 1], [1, 2], [2, 3]],
        positions=torch.tensor([
            [50.0, 50.0],
            [89.0, 50.0],   # near right edge — right-of-b move puts macro 0 OOB
            [50.0, 20.0],
            [20.0, 50.0],
        ]),
    )

    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=4)
    movable_mask = bm.get_movable_mask() & bm.get_hard_macro_mask()
    obstacle_mask = bm.macro_fixed & bm.get_hard_macro_mask()

    w = float(bm.macro_sizes[0, 0].item())          # 10.03
    max_valid_cx = bm.canvas_width - w / 2           # 94.985

    # Confirm fixture property: clamp boundary must NOT be grid-aligned.
    assert abs(snap_to_grid(max_valid_cx) - max_valid_cx) > 1e-4, (
        f"Fixture invalid: max_valid_cx={max_valid_cx} is a grid multiple. "
        "Adjust macro_size so the boundary tests the off-grid clamp scenario."
    )

    found_oob_on_grid = False
    found_valid_on_grid = False

    for pair_idx, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pair_idx, set())
        for c in cands:
            sc = _prepare_candidate(c, bm, movable_mask, obstacle_mask, legalizer_max_rings=25)

            # ALL generated coordinates must be on the 0.05 µm grid regardless of validity.
            for i in range(bm.num_hard_macros):
                for axis in range(2):
                    val = float(c.positions[i, axis].item())
                    val64 = round(float(np.float64(val)), 3)
                    snapped = snap_to_grid(val64)
                    assert abs(snapped - val64) < 1e-9, (
                        f"Candidate {c.name} macro {i} axis {axis}: "
                        f"coord {val64} is off-grid by {abs(snapped - val64)}. "
                        f"Snap-then-clamp regression: clamping to {max_valid_cx} "
                        f"(not grid-aligned) would produce this off-grid value."
                    )

            if sc.valid:
                found_valid_on_grid = True
            elif sc.num_out_of_bounds > 0:
                # Find the specific OOB coordinate and confirm it is on-grid.
                for i in range(bm.num_hard_macros):
                    for axis in range(2):
                        coord = float(c.positions[i, axis].item())
                        half = float(bm.macro_sizes[i, axis].item()) / 2.0
                        canvas_dim = bm.canvas_width if axis == 0 else bm.canvas_height
                        if coord < half or coord > canvas_dim - half:
                            coord64 = round(float(np.float64(coord)), 3)
                            snapped = snap_to_grid(coord64)
                            # Key assertion: OOB coord must still be on the grid.
                            # Old snap-then-clamp would have produced coord=94.985 here
                            # (off-grid), new snap-only produces 99.05 (on-grid, OOB).
                            assert abs(snapped - coord64) < 1e-9, (
                                f"OOB coord in {c.name} macro {i} axis {axis}: "
                                f"{coord64} is off-grid by {abs(snapped - coord64)}. "
                                f"Expected snap-only behaviour, not snap-then-clamp."
                            )
                            found_oob_on_grid = True

    assert found_oob_on_grid, (
        "No OOB candidate generated with non-grid-aligned boundary. "
        "Adjust macro_size or positions so a right/left/above/below move goes OOB."
    )
    assert found_valid_on_grid, (
        "No valid M3A candidate generated — check fixture positions."
    )
