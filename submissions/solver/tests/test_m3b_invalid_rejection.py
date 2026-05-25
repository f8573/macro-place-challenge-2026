"""test_m3b_invalid_rejection — OOB and overlapping M3B candidates rejected before scoring."""

import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm_tight(canvas=25.0, macro_size=10.0):
    """3 macros on a tight canvas."""
    return make_benchmark(
        n_hard=3, canvas=canvas, macro_size=macro_size,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        positions=[[5.0, 5.0], [15.0, 5.0], [5.0, 15.0]],
    )


def _bm_oob_guaranteed(canvas=100.0, macro_size=10.03):
    """3 macros where cyclic rotation is guaranteed to produce OOB candidates.

    macro_size=10.03 → half=5.015 → max_valid_cx=94.985 (not a 0.05 µm multiple).
    B.x=94.98 is a valid initial center (94.98 < 94.985) but snap(94.98)=95.0 (OOB).
    Cyclic rotation puts macro A at snap(B.pos)=(95.0, 5.05): guaranteed OOB.
    """
    return make_benchmark(
        n_hard=3, canvas=canvas, macro_size=macro_size,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        positions=[[5.05, 5.05], [94.98, 5.05], [5.05, 94.98]],
    )


def test_oob_candidates_produced_raw_and_rejected():
    """OOB M3B candidates must appear in the pool, be marked invalid, and not be scored.

    Uses a fixture where snap(B.x)=95.0 > max_valid=94.985 so cyclic rotation
    is guaranteed to emit at least one OOB candidate.  The test asserts:
    - At least one m3b candidate is invalid (OOB was actually emitted, not suppressed).
    - No invalid candidate was scored.
    """
    bm = _bm_oob_guaranteed()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    # max_official_scores must be non-zero so pass 1-4 candidates get proxy-scored,
    # which enables M3B (it gates on _pre_m3b_valid_scored being non-empty).
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)
    m3b_all = [s for s in ranked if s.family == "m3b_cluster_refinement"]
    m3b_invalid = [s for s in m3b_all if not s.valid]

    # Generation must have produced at least one OOB candidate (fixture guarantees this).
    # diag.m3b_invalid > 0 proves the invalid path was actually exercised, not just skipped.
    assert diag.m3b_invalid > 0, (
        f"Expected at least one OOB/invalid M3B candidate to be generated "
        f"(diag.m3b_invalid={diag.m3b_invalid}).  Fixture uses snap(B.x)=95.0 > "
        f"max_valid=94.985, so cyclic rotation must emit an OOB coordinate.  "
        f"A clamping implementation would suppress the OOB candidate and fail here."
    )
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

    # Fixture positions contain y=5.0 < half=5.015, so snap(5.0)=5.0 is OOB.
    # At least one m3b candidate must be invalid — asserts OOB was emitted, not clamped.
    m3b_invalid_found = any(
        sc.family == "m3b_cluster_refinement" and not sc.valid
        for sc in ranked
    )
    assert m3b_invalid_found, (
        "Expected at least one invalid M3B candidate (positions include y=5.0 < half=5.015 "
        "which snaps to 5.0 — OOB).  A clamping implementation would repair to half=5.015 "
        "(off-grid) or 5.05 (wrong grid step), hiding the OOB and passing validation."
    )

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
