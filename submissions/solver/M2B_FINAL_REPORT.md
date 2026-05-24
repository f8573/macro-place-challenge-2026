# M2B Final Report

Generated: 2026-05-23

## Profile Configuration: `m2b-final`

| Parameter | Value |
|-----------|-------|
| `only_original_neighborhood` | `true` |
| `candidate_budget` | 80 |
| `neighborhood_macro_limit` | 20 |
| `neighborhood_step_profile` | medium |
| `refinement_around_winners` | `true` |
| `refinement_top_k` | 5 |
| `refinement_combo_size` | 2 |
| `refinement_seed_strategy` | **diverse** |
| `refinement_exploration_seeds` | 1 |
| `line_search_around_winners` | `true` |
| `line_search_top_k` | 3 |
| `line_search_max_scale` | 4.0 |
| `line_search_stop_after_worse` | 2 |
| `max_official_scores` | 60 |
| Persistent cache | Optional (not required for final quality) |

Profile is registered in:
- `submissions/solver/scripts/run_benchmarks.py` (`_PROFILES["m2b-final"]`)
- `submissions/solver/scripts/run_official_scoring_smoke.py` (`_SMOKE_PROFILES["m2b-final"]`)

---

## Files Changed

| File | Change |
|------|--------|
| `submissions/solver/scripts/run_benchmarks.py` | Added `m2b-final` profile; updated `official-smoke` description |
| `submissions/solver/scripts/run_official_scoring_smoke.py` | Added `--profile` flag; improved winner reporting (macro id, dx/dy, postlegal, legalizer extras, budget) |
| `submissions/solver/tests/test_candidates.py` | Added 6 m2b-final profile tests (236 total) |
| `submissions/solver/M2B_FINAL_REPORT.md` | This document |

---

## Test Results

### Full Test Suite

```
236 passed in 14.96s
```

### New M2B Final Profile Tests (6)

| Test | Status |
|------|--------|
| `test_m2b_final_profile_exists` | PASS |
| `test_m2b_final_uses_diverse_seed_strategy` | PASS |
| `test_m2b_final_preserves_original_raw_fallback` | PASS |
| `test_m2b_final_has_bounded_official_score_budget` | PASS |
| `test_m2b_final_does_not_require_persistent_cache` | PASS |
| `test_m2b_final_is_deterministic` | PASS |

---

## Official Benchmark Results (Smoke: ibm01/ibm02/ibm03)

### ibm02 — Confirmed cold rerun (2026-05-23, `--profile m2b-final --clear-score-cache`)

| Field | Value |
|-------|-------|
| raw_original_proxy_cost | 1.565849 |
| best_proxy_cost | **1.558417** |
| delta_vs_raw_original | **-0.007431** |
| winning_candidate | `original_refinement_m51_tiny0p5um_p0_p1` |
| winning_family | `original_refinement` |
| moved_macro_id | 51 |
| intended_dx / intended_dy | 0.0000 / **+0.5000** um |
| actual_postlegal_dx / dy | 0.0000 / **+0.5000** um |
| legalizer_moved_extra_macros | [] (none) |
| official_scores_used | 60 |
| skipped_by_budget | 111 |
| runtime | 565,718 ms (~9.4 min) |
| invariant | **OK** |

Matches prior cold run exactly. Determinism confirmed.

### Full Cold Run — All 3 Benchmarks (2026-05-23, `run_benchmarks --profile m2b-final --clear-score-cache`)

| Benchmark | raw_original | best_cost | delta | Winner | Winner Family | Scores | Skipped | Runtime | Invariant |
|-----------|-------------|-----------|-------|--------|---------------|--------|---------|---------|-----------|
| ibm01 | 1.0379 | **1.0378** | **-0.0001** | `original_line_search_m215_scale2p5x` | original_line_search | 55 | 70 | 211,434 ms | OK |
| ibm02 | 1.5658 | **1.5584** | **-0.0074** | `original_refinement_m51_tiny0p5um_p0_p1` | original_refinement | 60 | 111 | 576,049 ms | OK |
| ibm03 | 1.3255 | **1.3255** | **-0.0000** | `original_refinement_m289_tiny0p25um_m1_p0` | original_refinement | 54 | 101 | 364,418 ms | OK |

Total runtime: ~1,152 s (~19.2 min). All benchmarks: 0 cache hits (true cold run).

### Aggregate Summary

| Metric | Value |
|--------|-------|
| Benchmarks run | 3 |
| Improved | 3 |
| Unchanged | 0 |
| Regressed | 0 |
| Mean delta | ~-0.0025 |
| Median delta | ~-0.0001 |
| Best improvement | ibm02: **-0.007431** |
| Worst delta | ibm03: -2.9×10⁻⁵ (tiny but positive direction) |

ibm02 winner matches the standalone ibm02 rerun exactly: `original_refinement_m51_tiny0p5um_p0_p1`, cost 1.558417. Cross-run determinism confirmed.

---

## Cold Rerun Reproducibility

- **ibm01**: Run 1 and Run 2 produced matching results. ✓
- **ibm02**: Run 1 and Run 2 produced matching winners and costs (within floating-point). ✓
- **ibm03**: Run 1 and Run 2 produced matching results. ✓

The diverse seed strategy with `refinement_exploration_seeds=1` is deterministic: given the same benchmark and config, the ranked output, winner, and costs are identical across cold reruns.

---

## Cache Policy

- Persistent score cache is **optional** for development (speeds up repeated runs of the same benchmark).
- Persistent score cache is **not required** for final quality.
- All benchmark results cited in this report were obtained with `--clear-score-cache` (cold cache).
- The test `test_m2b_final_does_not_require_persistent_cache` verifies this programmatically.

---

## Known Risks

1. **ibm01/ibm02/ibm03 only**: The profile was validated on three small public benchmarks. Larger or denser private benchmarks may behave differently; the budget (max_official_scores=60) may be tighter relative to the search space.

2. **Legalizer cascades**: On dense benchmarks the greedy legalizer may move additional macros beyond the intended macro to resolve overlaps. The invariant (best ≤ raw_original) still holds, but the actual placement may differ from the intended move.

3. **Official scorer dependency**: The `m2b-final` profile requires `plc_client_os` (via `git submodule update --init external/MacroPlacement`) for official scoring. Without it, the profile falls back to local-proxy scoring only.

4. **No global candidate families**: `only_original_neighborhood=True` disables spectral, area_degree, and terminal_anchor families. If the neighborhood search misses a globally better placement, those alternatives are not tried.

---

## Commands to Reproduce

### Run full test suite
```bash
python -m pytest submissions/solver/tests
```

### Run m2b-final on smoke benchmarks (cold)
```bash
python -m submissions.solver.scripts.run_benchmarks \
  --profile m2b-final \
  --clear-score-cache
```

### Run ibm02 deterministic rerun
```bash
python -m submissions.solver.scripts.run_official_scoring_smoke \
  -b ibm02 \
  --profile m2b-final \
  --clear-score-cache
```

### Run all three benchmarks via official smoke script
```bash
python -m submissions.solver.scripts.run_official_scoring_smoke \
  --profile m2b-final \
  --clear-score-cache
```

---

## Submit-Readiness Assessment

**M2B is submit-ready** subject to the following conditions:

- [x] 236 tests pass (no regressions)
- [x] Profile is frozen and documented
- [x] ibm02 cold rerun deterministically confirmed (winner identical, cost 1.558417)
- [x] Full ibm01/ibm02/ibm03 cold run completed successfully (all improved, all invariants OK)
- [x] Persistent cache not required (0 cache hits in all benchmark runs)
- [x] Invariant (best ≤ raw_original) holds on all tested benchmarks
- [x] No benchmark regressed vs. original baseline
- [ ] Broader benchmark validation (beyond ibm01/02/03) pending
- [ ] `plc_client_os` submodule must be initialized for official scoring
