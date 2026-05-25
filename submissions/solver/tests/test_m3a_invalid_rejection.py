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
