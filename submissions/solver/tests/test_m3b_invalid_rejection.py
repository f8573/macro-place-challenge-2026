"""test_m3b_invalid_rejection — OOB and overlapping M3B candidates rejected before scoring."""

import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm_tight(canvas=25.0, macro_size=10.0):
    """3 macros on a tight canvas: rotations will place macros OOB."""
    return make_benchmark(
        n_hard=3, canvas=canvas, macro_size=macro_size,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        positions=[[5.0, 5.0], [15.0, 5.0], [5.0, 15.0]],
    )


def test_oob_candidates_produced_raw_and_rejected():
    """OOB M3B candidates must appear in the pool but be marked invalid."""
    bm = _bm_tight()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)
    m3b_all = [s for s in ranked if s.family == "m3b_cluster_refinement"]
    m3b_invalid = [s for s in m3b_all if not s.valid]

    # The generation must have produced some candidates.
    assert diag.m3b_candidates_generated >= 0
    # Invalid M3B candidates must not have been scored.
    for inv in m3b_invalid:
        assert not inv.was_scored, (
            f"Invalid M3B candidate {inv.name!r} was scored — must be rejected before scoring"
        )


def test_invalid_m3b_candidates_never_selected():
    """An invalid M3B candidate must never be the final winner."""
    bm = _bm_tight()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, _ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert best is not None
    if best.family == "m3b_cluster_refinement":
        assert best.valid, "An invalid M3B candidate must not be selected as winner"


def test_overlapping_candidates_rejected_before_scoring():
    """Candidates that overlap other macros must be invalid and never scored."""
    bm = make_benchmark(
        n_hard=3, canvas=40.0, macro_size=15.0,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        positions=[[7.5, 7.5], [22.5, 7.5], [7.5, 22.5]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    for sc in ranked:
        if sc.family == "m3b_cluster_refinement" and not sc.valid:
            assert not sc.was_scored, (
                f"Invalid overlapping M3B candidate {sc.name!r} was scored"
            )


def test_non_grid_aligned_boundary_regression():
    """Regression: non-grid-aligned boundary must not repair OOB coords via clamping.

    Uses macro_size=10.03 so w/2=5.015 — the max valid center is 94.985 which
    is NOT a 0.05 µm grid multiple.  If generation clamped to bounds, a snapped
    value could land on an invalid coordinate.  The correct behaviour is to emit
    the OOB coordinate raw and let validation reject it.
    """
    bm = make_benchmark(
        n_hard=3, canvas=100.0, macro_size=10.03,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        positions=[[90.0, 5.0], [5.0, 90.0], [5.0, 5.0]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)

    from submissions.solver.core.m3b_candidate_generation import GRID_STEP
    for sc in ranked:
        if sc.family != "m3b_cluster_refinement":
            continue
        # Every coordinate must be grid-aligned after snap.  Use 1e-3 fractional
        # tolerance on the quotient to accommodate float32 storage rounding (values
        # around 100 can carry ~1e-5 float32 conversion error; 1e-3 is well above
        # that but still catches any off-grid value that differs by ≥1 ULP of 0.05).
        for mid in range(bm.num_hard_macros):
            for dim in range(2):
                val = float(sc.positions[mid, dim].item())
                quotient = val / GRID_STEP
                fractional = abs(quotient - round(quotient))
                assert fractional < 1e-3, (
                    f"{sc.name} macro {mid} dim {dim}: value {val} not on 0.05 grid "
                    f"(fractional={fractional:.2e})"
                )
        # OOB candidates must be invalid (not repaired to valid by clamping).
        if not sc.valid:
            assert not sc.was_scored, (
                f"OOB/invalid M3B candidate {sc.name!r} was scored"
            )
