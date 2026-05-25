"""test_m3a_m2b_final_invariant — M2B-final winner stays in candidate pool unconditionally."""

import torch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                          net_nodes=[[0, 1], [1, 2], [2, 3]])


def test_m2b_winner_still_in_ranked_pool_with_m3a():
    """After M3A, the M2B winner (from passes 1–3) must still be in ranked."""
    bm = _bm()

    # Identify M2B winner without M3A.
    gen_base = CandidateGenerationConfig(only_original_neighborhood=True)
    cands_base = generate_candidates(bm, gen_base)
    best_m2b, _, _ = score_and_select(cands_base, bm, plc=None, generation_config=gen_base)

    # Run M3A.
    gen_m3a = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands_m3a = generate_candidates(bm, gen_m3a)
    _best, ranked, _diag = score_and_select(cands_m3a, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_m3a)

    # The M2B winner should appear in the ranked pool.
    names = {s.name for s in ranked}
    assert best_m2b.name in names, (
        f"M2B winner '{best_m2b.name}' not found in M3A ranked pool"
    )


def test_m3a_never_removes_m2b_winner_positions():
    """M3A must not mutate any existing candidate's positions."""
    bm = _bm()
    gen_m3a = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands = generate_candidates(bm, gen_m3a)
    # Snapshot positions of non-M3A candidates before scoring.
    pre = {c.name: c.positions.clone() for c in cands}

    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_m3a)

    for sc in ranked:
        if sc.family == "m3a_pair_refinement":
            continue
        if sc.name in pre:
            assert torch.allclose(sc.positions.float(), pre[sc.name].float(), atol=1e-6), (
                f"Existing candidate {sc.name} positions changed during M3A"
            )
