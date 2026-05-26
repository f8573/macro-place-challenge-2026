import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core import m4b_region_repair as m4b
from submissions.solver.core.candidate_scoring import placement_hash, score_and_select
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidateScoringConfig,
)
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.diagnostics import PlacementDiagnostics
from submissions.solver.core.m3d_candidate_export import export_candidate_rows
from submissions.solver.core.m3d_family_summary import summarize_candidate_families
from submissions.solver.legalization.greedy_legalizer import LegalizationResult


def _bm():
    return make_benchmark(
        n_hard=6,
        canvas=90.0,
        macro_size=5.0,
        positions=[
            [10.0, 10.0],
            [20.0, 10.0],
            [40.0, 40.0],
            [50.0, 40.0],
            [70.0, 70.0],
            [80.0, 70.0],
        ],
        net_nodes=[[0, 1, 2], [3, 4, 5]],
        name="m4b_test",
    )


def _m4b_gen_cfg(**overrides):
    cfg = dict(
        only_original_neighborhood=True,
        m4b_region_repair=True,
        m4b_reserved_scores=2,
        m4b_grid_dims=(3, 3),
        m4b_min_macros_per_region=2,
        m4b_max_combos_per_region=16,
        m4b_legalization_max_displacement_um=200.0,
        m4b_perturbation_fraction=0.5,
    )
    cfg.update(overrides)
    return CandidateGenerationConfig(**cfg)


def _score_cfg(max_scores=8):
    return CandidateScoringConfig(max_official_scores=max_scores, prefilter_mode="off")


def _run_with_m4b(**gen_overrides):
    bm = _bm()
    gen_cfg = _m4b_gen_cfg(**gen_overrides)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, gen_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(),
        generation_config=gen_cfg,
    )
    return bm, best, ranked, diag


def test_m4b_region_repair_generates_candidate_metadata():
    bm = _bm()
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    assert rows
    first = rows[0]
    assert first.family == "m4b_region_repair"
    assert first.metadata["source_stage"] == 6
    assert first.metadata["region_id"] is not None
    assert first.metadata["moved_macro_ids"]
    assert first.metadata["move_type"] in {"centroid_shift", "spread"}


def test_legal_candidates_enter_score_pool():
    _bm_obj, _best, ranked, diag = _run_with_m4b(m4b_reserved_scores=3)
    m4b_rows = [s for s in ranked if s.family == "m4b_region_repair"]
    assert diag.m4b_legalized_count > 0
    assert any(s.valid and s.was_scored for s in m4b_rows)
    assert all(
        s.metadata["legalization_status"] == "legalized"
        for s in m4b_rows
        if s.valid
    )


def test_failed_legalization_rows_are_persisted_but_not_scored(monkeypatch):
    bm = _bm()

    def fake_legalize(**kwargs):
        return LegalizationResult(
            positions=kwargs["positions"].clone(),
            valid=False,
            num_moved=0,
            max_move=0.0,
            total_move=0.0,
            runtime_ms=0.0,
            messages=["forced failure"],
        )

    monkeypatch.setattr(m4b, "legalize", fake_legalize)
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    assert rows
    assert all(not row.valid for row in rows)
    assert all(not row.was_scored for row in rows)
    assert all(row.metadata["legalization_status"] == "failed" for row in rows)
    assert all(row.metadata["legalization_failure_reason"] == "overlap_unresolved" for row in rows)
    assert all(row.metadata["placement_hash"] is None for row in rows)


def test_duplicate_after_legalization_counted_separately_and_marked_duplicate():
    bm = _bm()
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    first_hash = next(row.metadata["placement_hash"] for row in rows if row.valid)
    rows2, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
        existing_hashes={first_hash: "existing"},
    )
    dup = next(row for row in rows2 if row.metadata["legalization_failure_reason"] == "duplicate_after_legalization")
    assert dup.duplicate_of == "existing"
    assert dup.metadata["placement_hash"] == first_hash
    assert not dup.was_scored
    summary = m4b.summarize_m4b_audit_rows(rows2)
    assert summary["duplicate_after_legalization_count"] == 1


def test_adjusted_vs_raw_legalized_rate_distinguishes_duplicates():
    bm = _bm()
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    first_hash = next(row.metadata["placement_hash"] for row in rows if row.valid)
    rows2, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
        existing_hashes={first_hash: "existing"},
    )
    summary = m4b.summarize_m4b_audit_rows(rows2)
    assert summary["adjusted_legalized_rate"] > summary["raw_legalized_rate"]


def test_failure_reason_priority_is_deterministic():
    diag = PlacementDiagnostics(
        valid=False,
        num_macros=1,
        num_out_of_bounds=1,
        num_overlaps=1,
        num_nonfinite=0,
    )
    leg = LegalizationResult(
        positions=torch.zeros(1, 2),
        valid=False,
        num_moved=1,
        max_move=999.0,
        total_move=999.0,
        runtime_ms=0.0,
    )
    assert (
        m4b.classify_legalization_failure(
            diagnostics=diag,
            legalization_result=leg,
            max_displacement_um=1.0,
            duplicate=True,
        )
        == "out_of_bounds"
    )


def test_displacement_threshold_enforced(monkeypatch):
    bm = _bm()

    def fake_legalize(**kwargs):
        return LegalizationResult(
            positions=kwargs["positions"].clone(),
            valid=True,
            num_moved=1,
            max_move=999.0,
            total_move=999.0,
            runtime_ms=0.0,
        )

    monkeypatch.setattr(m4b, "legalize", fake_legalize)
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
        legalization_max_displacement_um=1.0,
    )
    assert rows
    assert all(row.metadata["legalization_failure_reason"] == "displacement_too_large" for row in rows)
    assert all(not row.valid for row in rows)


def test_approx_delta_populated_for_admitted_m4b_candidates():
    _bm_obj, _best, ranked, _diag = _run_with_m4b(m4b_reserved_scores=4)
    admitted = [
        s for s in ranked
        if s.family == "m4b_region_repair" and s.valid and s.metadata.get("skip_reason") == "scored"
    ]
    assert admitted
    assert all(s.metadata.get("approx_hpwl_delta") is not None for s in admitted)
    assert all(
        s.metadata["approx_hpwl_delta"] == s.metadata["post_legalization_approx_delta"]
        for s in admitted
    )


def test_approx_delta_does_not_affect_admission_ranking_or_scoring(monkeypatch):
    bm = _bm()
    gen_cfg = _m4b_gen_cfg(m4b_reserved_scores=3)

    def run_with_delta(value):
        monkeypatch.setattr(m4b, "compute_approx_delta", lambda *_args, **_kwargs: value)
        _best, ranked, _diag = score_and_select(
            generate_candidates(bm, gen_cfg),
            bm,
            plc=None,
            scoring_config=_score_cfg(),
            generation_config=gen_cfg,
        )
        return [
            s.name for s in ranked
            if s.family == "m4b_region_repair" and s.was_scored
        ]

    assert run_with_delta(1000.0) == run_with_delta(-1000.0)


def test_placement_hash_stable_for_legalized_candidates():
    bm = _bm()
    rows1, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    rows2, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    hashes1 = [row.metadata["placement_hash"] for row in rows1 if row.valid]
    hashes2 = [row.metadata["placement_hash"] for row in rows2 if row.valid]
    assert hashes1 == hashes2
    assert all(h == placement_hash(row.positions) for h, row in zip(hashes1, [r for r in rows1 if r.valid]))


def test_failed_non_legalized_candidates_have_null_hash(monkeypatch):
    bm = _bm()

    def fake_legalize(**kwargs):
        return LegalizationResult(
            positions=kwargs["positions"].clone(),
            valid=False,
            num_moved=0,
            max_move=0.0,
            total_move=0.0,
            runtime_ms=0.0,
        )

    monkeypatch.setattr(m4b, "legalize", fake_legalize)
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    assert all(row.metadata["placement_hash"] is None for row in rows)


def test_reserved_m4b_bucket_does_not_starve_existing_families():
    bm = _bm()
    base_cfg = CandidateGenerationConfig(only_original_neighborhood=True)
    m4b_cfg = _m4b_gen_cfg(m4b_reserved_scores=1)
    _base_best, base_ranked, _base_diag = score_and_select(
        generate_candidates(bm, base_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(max_scores=5),
        generation_config=base_cfg,
    )
    m4b_best, m4b_ranked, _m4b_diag = score_and_select(
        generate_candidates(bm, m4b_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(max_scores=5),
        generation_config=m4b_cfg,
    )
    base_scored = sorted(s.name for s in base_ranked if s.was_scored)
    m4b_existing_scored = sorted(
        s.name for s in m4b_ranked if s.was_scored and s.family != "m4b_region_repair"
    )
    assert m4b_existing_scored == base_scored
    assert m4b_best.proxy_cost <= min(s.proxy_cost for s in base_ranked if s.was_scored)


def test_existing_families_unchanged_when_m4b_disabled():
    bm = _bm()
    cfg_a = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=4,
        m3c_budget_allocation=True,
        m3c_pre_m3_budget=4,
        m3c_m3a_reserved_budget=2,
        m3c_m3b_reserved_budget=2,
    )
    cfg_b = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=4,
        m3c_budget_allocation=True,
        m3c_pre_m3_budget=4,
        m3c_m3a_reserved_budget=2,
        m3c_m3b_reserved_budget=2,
        m4b_region_repair=False,
        m4b_reserved_scores=20,
    )
    best_a, ranked_a, diag_a = score_and_select(
        generate_candidates(bm, cfg_a), bm, plc=None,
        scoring_config=_score_cfg(max_scores=8), generation_config=cfg_a,
    )
    best_b, ranked_b, diag_b = score_and_select(
        generate_candidates(bm, cfg_b), bm, plc=None,
        scoring_config=_score_cfg(max_scores=8), generation_config=cfg_b,
    )
    assert best_a.name == best_b.name
    assert [s.name for s in ranked_a] == [s.name for s in ranked_b]
    assert diag_a.m3a_candidates_scored == diag_b.m3a_candidates_scored
    assert diag_a.m3b_scored == diag_b.m3b_scored


def test_no_scorer_invocation_in_unit_tests():
    sys.modules.pop("submissions.solver.core.scoring", None)
    _run_with_m4b(m4b_reserved_scores=1)
    assert "submissions.solver.core.scoring" not in sys.modules


def test_m4a_compatible_artifact_fields_emitted():
    bm, best, ranked, diag = _run_with_m4b(m4b_reserved_scores=2)
    rows = export_candidate_rows(ranked, best, diag, benchmark=bm.name, profile="m4b-default")
    required = {
        "benchmark",
        "profile",
        "candidate_name",
        "family",
        "valid",
        "duplicate",
        "admitted",
        "not_admitted",
        "scored",
        "skip_reason",
        "proxy_cost",
        "approx_delta",
        "placement_hash",
        "source_stage",
        "legalization_status",
        "legalization_failure_reason",
    }
    assert required.issubset(rows[0])
    assert any(row["family"] == "m4b_region_repair" for row in rows)


def test_region_partition_covers_canvas():
    regions = m4b.partition_regions(90.0, 90.0, (3, 3))
    assert len(regions) == 9
    assert regions[0].x0 == 0.0 and regions[0].y0 == 0.0
    assert regions[-1].x1 == 90.0 and regions[-1].y1 == 90.0
    bm = _bm()
    assigned = m4b.assign_macros_to_regions(
        bm.macro_positions,
        list(range(bm.num_hard_macros)),
        regions,
        (3, 3),
        90.0,
        90.0,
    )
    assert sorted(mid for mids in assigned.values() for mid in mids) == list(range(6))


def test_candidate_name_pattern_and_no_swap():
    bm = _bm()
    rows, _ = m4b.generate_m4b_region_repair_candidates(
        benchmark=bm,
        base_positions=bm.macro_positions,
    )
    pattern = re.compile(r"^m4b_r\d+(_m\d+)+_(centroid_shift|spread)$")
    assert rows
    assert all(pattern.match(row.name) for row in rows)
    assert all("swap" not in row.name for row in rows)


def test_family_summary_reports_raw_and_adjusted_rates():
    bm, best, ranked, diag = _run_with_m4b(m4b_reserved_scores=2)
    rows = export_candidate_rows(ranked, best, diag, benchmark=bm.name, profile="m4b-default")
    summaries = summarize_candidate_families(rows)
    m4b_summary = next(row for row in summaries if row["family"] == "m4b_region_repair")
    assert "raw_legalized_rate" in m4b_summary
    assert "adjusted_legalized_rate" in m4b_summary
    assert m4b_summary["legalized_count"] >= m4b_summary["scored_count"]


def test_smoke_m4b_default_profile_registered():
    from submissions.solver.scripts.run_benchmarks import _PROFILES

    profile = _PROFILES["m4b-default"]
    assert profile["max_official_scores"] == 80
    assert profile["m4b_reserved_scores"] == 20
    assert profile["m4b_region_repair"] is True
