"""test_m3a_original_raw_invariant — original_raw stays in candidate pool unconditionally."""

import torch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _run_m3a(bm, top_k=4):
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=top_k,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands = generate_candidates(bm, config=gen_cfg)
    return score_and_select(cands, bm, plc=None,
                            scoring_config=score_cfg,
                            generation_config=gen_cfg)


def test_original_raw_in_ranked_pool():
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                        net_nodes=[[0, 1], [1, 2]])
    _best, ranked, _diag = _run_m3a(bm)
    names = {s.name for s in ranked}
    assert "original_raw" in names, "original_raw must always be in the ranked pool"


def test_original_raw_positions_unchanged():
    """original_raw's positions must match the benchmark's initial positions."""
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                        net_nodes=[[0, 1], [1, 2]])
    _best, ranked, _diag = _run_m3a(bm)
    raw = next(s for s in ranked if s.name == "original_raw")
    assert torch.allclose(raw.positions.float(), bm.macro_positions.float(), atol=1e-6)


def test_original_raw_in_pool_when_m3a_wins():
    """Even when an M3A candidate wins, original_raw must remain selectable."""
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                        net_nodes=[[0, 1], [1, 2], [0, 2]])
    _best, ranked, _diag = _run_m3a(bm, top_k=10)
    names = {s.name for s in ranked}
    assert "original_raw" in names
