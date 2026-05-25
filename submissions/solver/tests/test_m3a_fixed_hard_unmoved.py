"""test_m3a_fixed_hard_unmoved — fixed-hard macro coordinates unchanged in all M3A candidates."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
from submissions.solver.core.m3a_candidate_generation import generate_pair_candidates


def _bm_with_fixed():
    pos = torch.tensor([
        [50.0, 50.0],  # macro 0 — FIXED
        [20.0, 20.0],  # macro 1 — movable
        [80.0, 20.0],  # macro 2 — movable
        [20.0, 80.0],  # macro 3 — movable
    ])
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=8.0,
        net_nodes=[[1, 2], [2, 3], [1, 3]],
        fixed_mask=[True, False, False, False],
        positions=pos,
    )


def test_fixed_macro_position_unchanged_in_each_m3a_candidate():
    bm = _bm_with_fixed()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    assert len(pairs) >= 1, "expected net-coupled pairs"
    existing: set = set()
    for pi, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pi, existing)
        for c in cands:
            existing.add(c.name)
            for fi in range(bm.num_hard_macros):
                if bool(bm.macro_fixed[fi].item()):
                    orig_x = float(wp[fi, 0].item())
                    orig_y = float(wp[fi, 1].item())
                    new_x = float(c.positions[fi, 0].item())
                    new_y = float(c.positions[fi, 1].item())
                    assert new_x == orig_x, (
                        f"Fixed macro {fi} x changed in {c.name}: {orig_x} -> {new_x}"
                    )
                    assert new_y == orig_y, (
                        f"Fixed macro {fi} y changed in {c.name}: {orig_y} -> {new_y}"
                    )


def test_fixed_macro_not_in_any_pair():
    bm = _bm_with_fixed()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    for a, b, _ in pairs:
        assert not bool(bm.macro_fixed[a].item()), f"Fixed macro {a} in pair"
        assert not bool(bm.macro_fixed[b].item()), f"Fixed macro {b} in pair"


def test_fixed_macro_unchanged_after_full_m3a_pipeline():
    bm = _bm_with_fixed()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)

    orig_fixed_x = float(bm.macro_positions[0, 0].item())
    orig_fixed_y = float(bm.macro_positions[0, 1].item())

    for sc in ranked:
        if sc.family != "m3a_pair_refinement":
            continue
        assert float(sc.positions[0, 0].item()) == orig_fixed_x, (
            f"Fixed macro 0 x changed in {sc.name}"
        )
        assert float(sc.positions[0, 1].item()) == orig_fixed_y, (
            f"Fixed macro 0 y changed in {sc.name}"
        )
