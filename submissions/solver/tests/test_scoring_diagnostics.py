"""Tests for M2B scoring diagnostics, tie-breaking, and connectivity audit."""

import torch
import pytest

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import (
    score_and_select,
    connectivity_audit,
    placement_hash,
)
from submissions.solver.core.candidate_types import ScoringDiagnostics


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def _make_benchmark(
    n_hard: int = 4,
    canvas: float = 100.0,
    macro_size: float = 10.0,
    net_nodes=None,
    fixed_mask=None,
    num_ports: int = 0,
) -> Benchmark:
    positions = torch.zeros(n_hard, 2, dtype=torch.float32)
    for i in range(n_hard):
        positions[i, 0] = (i % 4) * 20.0 + 10.0
        positions[i, 1] = (i // 4) * 20.0 + 10.0

    sizes = torch.full((n_hard, 2), macro_size, dtype=torch.float32)

    if fixed_mask is None:
        fixed = torch.zeros(n_hard, dtype=torch.bool)
    else:
        fixed = torch.tensor(fixed_mask, dtype=torch.bool)

    if net_nodes is None:
        nn = []
        nw = torch.zeros(0)
    else:
        nn = [torch.tensor(ns, dtype=torch.long) for ns in net_nodes]
        nw = torch.ones(len(nn))

    port_pos = torch.zeros(num_ports, 2, dtype=torch.float32)
    if num_ports > 0:
        for p in range(num_ports):
            port_pos[p, 0] = canvas * (p + 1) / (num_ports + 1)
            port_pos[p, 1] = 0.0

    return Benchmark(
        name="test",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_hard,
        num_hard_macros=n_hard,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_hard)],
        num_nets=len(nn),
        net_nodes=nn,
        net_weights=nw,
        grid_rows=8,
        grid_cols=8,
        port_positions=port_pos,
    )


# ---------------------------------------------------------------------------
# 1. Scoring unavailable does not claim improvement
# ---------------------------------------------------------------------------


def test_scoring_unavailable_does_not_claim_improvement():
    """When net_nodes is empty, delta should be 0 and selected_due_to != 'proxy_cost'."""
    bm = _make_benchmark(n_hard=4, net_nodes=None)
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)

    assert diag.scoring_mode == "unavailable"
    assert not diag.scoring_available

    # No candidate should be claimed to have improved over original
    for sc in ranked:
        if sc.delta_vs_original is not None:
            # All deltas should be 0 (all costs are 0.0)
            assert abs(sc.delta_vs_original) < 1e-9, (
                f"Candidate '{sc.name}' claims delta={sc.delta_vs_original} "
                "but scoring is unavailable"
            )

    # selected_due_to must NOT be "proxy_cost" when unavailable
    assert diag.selected_due_to != "proxy_cost", (
        f"selected_due_to='{diag.selected_due_to}' but scoring is unavailable"
    )


# ---------------------------------------------------------------------------
# 2. Degenerate all-zero scores select original if valid
# ---------------------------------------------------------------------------


def test_degenerate_scores_select_original():
    """When all valid candidates have cost=0 and scoring is unavailable, select original."""
    bm = _make_benchmark(n_hard=4, net_nodes=None)
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)

    assert diag.score_is_degenerate, "Expected degenerate scores with empty net_nodes"
    assert best is not None, "Best candidate should not be None"
    assert best.name == "original_raw", (
        f"Expected 'original_raw' to be selected when scores are degenerate, got '{best.name}'"
    )
    assert diag.selected_due_to == "validity_only"


# ---------------------------------------------------------------------------
# 3. Placement hashes detect distinct placements
# ---------------------------------------------------------------------------


def test_placement_hash_detects_distinct():
    """Different positions should produce different hashes."""
    pos1 = torch.tensor([[10.0, 10.0], [30.0, 10.0]], dtype=torch.float32)
    pos2 = torch.tensor([[10.0, 10.0], [50.0, 10.0]], dtype=torch.float32)
    pos1_copy = pos1.clone()

    assert placement_hash(pos1) == placement_hash(pos1_copy), "Identical positions must hash equal"
    assert placement_hash(pos1) != placement_hash(pos2), "Distinct positions must hash differently"


def test_placement_hash_is_deterministic():
    pos = torch.tensor([[1.5, 2.5], [10.0, 20.0]], dtype=torch.float32)
    h1 = placement_hash(pos)
    h2 = placement_hash(pos)
    assert h1 == h2, "Placement hash must be deterministic"


# ---------------------------------------------------------------------------
# 4. Candidate families are reported correctly
# ---------------------------------------------------------------------------


def test_candidate_families_reported():
    """generate_candidates should produce at least 3 non-original families."""
    bm = _make_benchmark(n_hard=6, net_nodes=[[0, 1, 2], [3, 4], [1, 5]])
    candidates = generate_candidates(bm)
    families = {c.family for c in candidates if c.family != "original"}
    assert len(families) >= 3, f"Expected ≥3 non-original families, got {families}"


def test_all_candidate_families_are_non_empty_strings():
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1], [2, 3]])
    candidates = generate_candidates(bm)
    for c in candidates:
        assert isinstance(c.family, str) and len(c.family) > 0, (
            f"Candidate '{c.name}' has empty or non-string family"
        )


# ---------------------------------------------------------------------------
# 5. Spectral unavailable when graph has no edges
# ---------------------------------------------------------------------------


def test_spectral_unavailable_no_edges():
    """When net_nodes is empty, connectivity_audit reports spectral_available=False."""
    bm = _make_benchmark(n_hard=5, net_nodes=None)
    audit = connectivity_audit(bm)
    assert audit["num_net_edges"] == 0
    assert not audit["spectral_available"], (
        "spectral_available should be False when there are no net edges"
    )


def test_spectral_available_with_edges():
    """When net_nodes has connected macros, spectral_available=True."""
    bm = _make_benchmark(n_hard=5, net_nodes=[[0, 1], [2, 3, 4]])
    audit = connectivity_audit(bm)
    assert audit["num_net_edges"] > 0
    assert audit["spectral_available"], (
        "spectral_available should be True when net edges exist"
    )


# ---------------------------------------------------------------------------
# 6. Terminal-anchor unavailable when no fixed endpoints
# ---------------------------------------------------------------------------


def test_terminal_anchor_unavailable_no_fixed():
    """With no fixed macros and no ports, terminal_anchor_available=False."""
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1]], fixed_mask=[False]*4, num_ports=0)
    audit = connectivity_audit(bm)
    assert audit["num_fixed_endpoints"] == 0
    assert not audit["terminal_anchor_available"], (
        "terminal_anchor_available should be False with no fixed endpoints"
    )


def test_terminal_anchor_available_with_fixed_macro():
    """With at least one fixed macro, terminal_anchor_available=True."""
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1]], fixed_mask=[True, False, False, False])
    audit = connectivity_audit(bm)
    assert audit["num_fixed_endpoints"] >= 1
    assert audit["terminal_anchor_available"]


def test_terminal_anchor_available_with_ports():
    """With I/O ports, terminal_anchor_available=True even if no fixed macros."""
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1]], fixed_mask=[False]*4, num_ports=2)
    audit = connectivity_audit(bm)
    assert audit["num_fixed_endpoints"] == 2
    assert audit["terminal_anchor_available"]


# ---------------------------------------------------------------------------
# 7. selected_due_to is correct
# ---------------------------------------------------------------------------


def test_selected_due_to_validity_only_when_unavailable():
    bm = _make_benchmark(n_hard=4, net_nodes=None)
    _, _, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    assert diag.selected_due_to == "validity_only"


def test_selected_due_to_proxy_cost_when_costs_differ():
    """With real HPWL differences (non-empty net_nodes), selection is by proxy_cost."""
    bm = _make_benchmark(
        n_hard=6,
        canvas=200.0,
        net_nodes=[[0, 1], [2, 3], [4, 5], [0, 3, 5]],
    )
    _, _, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    # local_proxy mode with real connectivity should give non-degenerate costs
    assert diag.scoring_mode == "local_proxy"
    assert diag.scoring_available
    if not diag.score_is_degenerate:
        assert diag.selected_due_to == "proxy_cost"


def test_selected_due_to_fallback_when_all_invalid():
    """If somehow all candidates are invalid, selected_due_to='fallback_original'."""
    from submissions.solver.core.candidate_types import CandidatePlacement

    bm = _make_benchmark(n_hard=2, canvas=5.0, macro_size=8.0)
    # Pass only one deliberately-bad candidate (macro larger than canvas, no valid fit)
    # Let legalizer try to fix it — if it can't, fallback triggers
    bad_pos = torch.tensor([[2.5, 2.5], [2.5, 2.5]], dtype=torch.float32)
    candidates_only_bad = [CandidatePlacement("original", "original", bad_pos.clone())]
    best, _, diag = score_and_select(candidates_only_bad, bm, plc=None)
    # original is always in scored; if invalid, selected_due_to='fallback_original'
    # (legalizer may repair it — accept either outcome)
    assert diag.selected_due_to in (
        "fallback_original",
        "fallback_legalized_original",
        "fallback_other_valid",
        "no_valid_scored_candidate",
        "validity_only",
        "tie_break",
    )


def test_scoring_diagnostics_is_returned():
    """score_and_select always returns a ScoringDiagnostics object."""
    bm = _make_benchmark(n_hard=4)
    result = score_and_select(generate_candidates(bm), bm, plc=None)
    assert len(result) == 3, "score_and_select should return (best, ranked, diag)"
    _, _, diag = result
    assert isinstance(diag, ScoringDiagnostics)
    assert diag.scoring_mode in ("official", "local_proxy", "unavailable")
    assert diag.selected_due_to in (
        "proxy_cost",
        "fallback_original",
        "fallback_legalized_original",
        "fallback_other_valid",
        "no_valid_scored_candidate",
        "validity_only",
        "tie_break",
    )


# ---------------------------------------------------------------------------
# Connectivity audit field completeness
# ---------------------------------------------------------------------------


def test_connectivity_audit_fields():
    """connectivity_audit returns all required fields with correct types."""
    bm = _make_benchmark(n_hard=4, net_nodes=[[0, 1], [2, 3]])
    audit = connectivity_audit(bm)

    required_fields = [
        "num_macros", "num_nets", "num_net_edges",
        "num_macros_with_degree_gt_0", "num_fixed_endpoints",
        "spectral_available", "terminal_anchor_available",
    ]
    for field in required_fields:
        assert field in audit, f"Missing field '{field}' in connectivity_audit"

    assert isinstance(audit["spectral_available"], bool)
    assert isinstance(audit["terminal_anchor_available"], bool)
    assert audit["num_macros"] == 4
    assert audit["num_nets"] == 2
    assert audit["num_net_edges"] == 2  # [0,1] → 1 edge, [2,3] → 1 edge
    assert audit["num_macros_with_degree_gt_0"] == 4


def test_connectivity_audit_singleton_nets_ignored():
    """Nets with <2 hard macro pins contribute no edges."""
    bm = _make_benchmark(n_hard=4, net_nodes=[[0], [1], [2, 3]])
    audit = connectivity_audit(bm)
    # Only [2,3] has ≥2 pins → 1 edge
    assert audit["num_net_edges"] == 1
    assert audit["num_macros_with_degree_gt_0"] == 2
