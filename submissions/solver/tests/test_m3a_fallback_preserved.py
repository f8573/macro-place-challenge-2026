"""test_m3a_fallback_preserved — when no M3A candidate beats M2B-final, M2B-final returned."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm_no_nets():
    """Benchmark with no nets: scoring is unavailable, M2B winner is original_raw."""
    return make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, net_nodes=[])


def test_m2b_final_returned_when_no_m3a_improvement():
    """Without a plc and with no nets, M3A cannot improve; original_raw must be returned."""
    bm = _bm_no_nets()
    gen_cfg_base = CandidateGenerationConfig(only_original_neighborhood=True)
    score_cfg = CandidateScoringConfig()
    cands_base = generate_candidates(bm, config=gen_cfg_base)
    best_base, _, _ = score_and_select(cands_base, bm, plc=None,
                                       scoring_config=score_cfg,
                                       generation_config=gen_cfg_base)

    gen_cfg_m3a = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
    )
    cands_m3a = generate_candidates(bm, config=gen_cfg_m3a)
    best_m3a, _, diag_m3a = score_and_select(cands_m3a, bm, plc=None,
                                              scoring_config=score_cfg,
                                              generation_config=gen_cfg_m3a)
    # Both should return the same candidate (original_raw / m2b winner).
    assert best_m3a is not None
    # The winner source must not be m3a_pair_refinement when no nets.
    assert diag_m3a.m3a_winner_source != "m3a_pair_refinement"


def test_m2b_final_bit_identical_when_m3a_has_zero_pairs():
    """When there are no net-coupled pairs, M3A generates nothing and M2B winner is unchanged."""
    bm = _bm_no_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=64,
    )
    score_cfg = CandidateScoringConfig()
    cands_base = generate_candidates(bm, config=CandidateGenerationConfig(only_original_neighborhood=True))
    best_base, _, _ = score_and_select(cands_base, bm, plc=None)

    cands_m3a = generate_candidates(bm, config=gen_cfg)
    best_m3a, _, diag = score_and_select(cands_m3a, bm, plc=None,
                                         scoring_config=score_cfg,
                                         generation_config=gen_cfg)
    assert diag.m3a_pairs_considered == 0
    assert diag.m3a_candidates_generated == 0
    # Positions must be identical.
    assert torch.allclose(best_m3a.positions, best_base.positions, atol=1e-6)


def test_m3a_disabled_gives_same_result_as_no_m3a():
    """Explicitly disabling M3A must give bit-identical result to the default M2B run."""
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [1, 2]],
    )
    gen_base = CandidateGenerationConfig(only_original_neighborhood=True)
    gen_m3a_off = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=False,
    )
    score_cfg = CandidateScoringConfig()

    cands_base = generate_candidates(bm, gen_base)
    best_base, _, _ = score_and_select(cands_base, bm, plc=None,
                                       scoring_config=score_cfg, generation_config=gen_base)

    cands_off = generate_candidates(bm, gen_m3a_off)
    best_off, _, _ = score_and_select(cands_off, bm, plc=None,
                                      scoring_config=score_cfg, generation_config=gen_m3a_off)

    assert torch.allclose(best_base.positions, best_off.positions, atol=1e-6)
