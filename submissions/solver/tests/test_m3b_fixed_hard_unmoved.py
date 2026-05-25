"""test_m3b_fixed_hard_unmoved — fixed-hard macros never enter clusters and positions unchanged."""

import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core.m3b_cluster_enumeration import enumerate_net_coupled_triples
from submissions.solver.core.m3b_candidate_generation import (
    generate_cluster_candidates,
    generate_m3b_candidates_for_clusters,
)
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm_with_fixed(fixed_id=0):
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]],
        fixed_mask=[i == fixed_id for i in range(4)],
    )


def test_fixed_macro_not_in_any_cluster():
    bm = _bm_with_fixed(0)
    triples = enumerate_net_coupled_triples(bm, top_k=20)
    for a, b, c, _ in triples:
        assert a != 0 and b != 0 and c != 0, "fixed macro 0 must not appear in clusters"


def test_generate_raises_on_fixed_hard_macro():
    """generate_cluster_candidates must raise ValueError if a fixed-hard macro is passed."""
    bm = _bm_with_fixed(0)
    pos = bm.macro_positions.clone().float()
    with pytest.raises(ValueError, match="fixed-hard"):
        generate_cluster_candidates(bm, pos, 0, 1, 2, 0, set())


def test_fixed_positions_unchanged_in_all_candidates():
    """For every generated M3B candidate, fixed-hard macro positions must be unchanged."""
    bm = _bm_with_fixed(0)
    triples = enumerate_net_coupled_triples(bm, top_k=20)
    pos = bm.macro_positions.clone().float()
    fixed_x = float(pos[0, 0].item())
    fixed_y = float(pos[0, 1].item())

    all_cands = generate_m3b_candidates_for_clusters(bm, pos, triples, set())
    for cand in all_cands:
        cx = float(cand.positions[0, 0].item())
        cy = float(cand.positions[0, 1].item())
        assert cx == fixed_x and cy == fixed_y, (
            f"Fixed macro 0 was moved in {cand.name}: ({cx},{cy}) vs original ({fixed_x},{fixed_y})"
        )


def test_pipeline_fixed_positions_unchanged():
    """End-to-end: fixed-hard macro positions unchanged after M3B pipeline."""
    bm = _bm_with_fixed(0)
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=20)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    fixed_x = float(bm.macro_positions[0, 0].item())
    fixed_y = float(bm.macro_positions[0, 1].item())
    for sc in ranked:
        if sc.family == "m3b_cluster_refinement":
            cx = float(sc.positions[0, 0].item())
            cy = float(sc.positions[0, 1].item())
            assert cx == fixed_x and cy == fixed_y, (
                f"Fixed macro 0 was moved in {sc.name}"
            )
