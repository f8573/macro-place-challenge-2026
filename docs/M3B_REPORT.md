# M3B Report — Deterministic 3-Macro Cluster Refinement

## Status

**Implementation status:** Complete (M3B-slice-1)
**Test status:** All tests passing (see below)
**Benchmark status:** Pending cold-run results (see Benchmark Results section)

M3B is implemented as a conservative extension to the M2B/M3A candidate pipeline.
It adds deterministic 3-macro cluster-refinement candidates, validates them before
scoring, scores only valid candidates under the existing official proxy score budget,
and allows them to compete with the M2B/M3A pool only when the M3B pass completes
without budget exhaustion.

M3B does not replace the legalizer, scorer, validator, fallback behavior, or
M2B/M3A selection model.

---

## Goal

M3B explores small coordinated three-macro perturbations that pair-level M3A
refinements may miss.

The intended behavior is:

1. Start from the M2B/M3A candidate pool.
2. Preserve `original_raw` and the M2B/M3A winner as selectable safety-net candidates.
3. Enumerate deterministic net-coupled 3-macro clusters.
4. Generate a bounded set of cluster-refinement candidates (≤3 per cluster).
5. Validate all candidates before scoring.
6. Score valid candidates under the existing official proxy score budget.
7. Select the final winner only by official proxy score.
8. If M3B cannot complete safely, fall back to the pre-M3B pool.

---

## Implementation Summary

### New Modules

- `submissions/solver/core/m3b_cluster_enumeration.py`
  - Builds deterministic top-K net-coupled 3-macro clusters.
  - Considers movable hard macros only; excludes fixed-hard macros.
  - Uses stable canonical ordering (a < b < c) and deterministic tie-breaks.
  - Scores each triple as sum of pair couplings: shared(a,b) + shared(a,c) + shared(b,c).
  - Reuses the same pair-count construction as M3A pair enumeration.

- `submissions/solver/core/m3b_candidate_generation.py`
  - Generates up to 3 candidates per cluster:
    1. **cyclic rotation**: A takes B's position, B takes C's position, C takes A's position.
    2. **reverse cyclic rotation**: A takes C's position, C takes B's position, B takes A's position.
    3. **centroid-step shift**: all three macros translate by exactly one 0.05 µm grid step
       toward the cluster's shared-net centroid, preserving relative offsets.
  - Snaps generated coordinates to the 0.05 µm movement grid.
  - Does not clamp or repair out-of-bounds candidates.
  - Leaves invalid candidates for the existing validation path to reject.

### Candidate 4 Omitted

The specification allows up to 4 candidates per cluster. Candidate 4 (compact-to-centroid
preserving x/y rank order) is intentionally omitted.

Maintaining rank-order constraints while moving three macros toward a centroid requires
checking pairwise crossing conditions (e.g., ensuring a.x < b.x < c.x is preserved after
the move). This is structurally equivalent to spatial legalization: it needs to know how
far each macro can move before crossing another, which is exactly what a legalizer does.

Implementing this without legalization or clamping would require a bespoke constraint
solver, which would be a new legalization path and violate the no-repair invariant.

Three candidates per cluster (cyclic, reverse cyclic, centroid-step) is within the
spec-allowed maximum of 4.

### Modified Modules

- `submissions/solver/core/candidate_types.py`
  - Added M3B configuration fields to `CandidateGenerationConfig`:
    - `m3b_cluster_refinement: bool = False`
    - `m3b_top_k_clusters: int = 32`
    - `m3b_score_budget: Optional[int] = None`
  - Added M3B diagnostic fields to `ScoringDiagnostics`:
    - `m3b_clusters_considered`, `m3b_candidates_generated`
    - `m3b_valid`, `m3b_invalid`
    - `m3b_duplicates`, `m3b_scored`
    - `m3b_skipped_budget`, `m3b_budget_exhausted`
    - `m3b_selectable`, `m3b_best_candidate`, `m3b_best_delta`
    - `m3b_rejected_bounds`, `m3b_rejected_overlap`, `m3b_rejected_other`
    - `m3b_fresh_scores`, `m3b_cache_hits`, `m3b_best_score`

- `submissions/solver/core/candidate_scoring.py`
  - Added M3B as Pass 5, after M3A Pass 4.
  - Reuses existing validation, deduplication, official scoring, cache, and budget infrastructure.
  - Ensures M3B candidates are only selectable if valid, scored, and not part of a
    budget-exhausted partial M3B pass.
  - Excludes all `m3b_cluster_refinement` candidates from final selection if any valid
    M3B candidate is skipped due to M3B budget exhaustion.
  - M3B pass uses the best candidate from passes 1–4 (M2B + M3A safe pool) as the starting
    positions for cluster generation.
  - M3B budget exhaustion is independent of M3A budget exhaustion: each can trigger its
    own exclusion guard without affecting the other.

- `submissions/solver/scripts/run_benchmarks.py`
  - Added profiles:
    - `m3b-smoke`
    - `m3b-default`
    - `m3b-budget-stress`
  - Updated `run_profile` and `main` to wire M3B config fields.

---

## Core Invariants

M3B preserves the following invariants:

1. `original_raw` remains in the candidate pool unconditionally.
2. The M2B/M3A winner remains in the candidate pool unconditionally.
3. Final selection is by official proxy score only.
4. Fixed-hard macros are never moved (excluded from enumeration; asserted in generation).
5. Generated coordinates snap to the 0.05 µm movement grid.
6. Cluster ordering, candidate ordering, and tie behavior are deterministic.
7. M3B consumes the existing shared score budget; `m3b_score_budget` does not silently expand `max_official_scores`.
8. Invalid candidates are rejected before scoring.
9. Unscored candidates are never selectable.
10. If M3B budget is exhausted, all M3B candidates are excluded from final selection.
11. M3B budget exhaustion does not exclude M3A candidates (each guard is independent).
12. Persistent score cache remains optional.
13. M2B behavior is unchanged when M3A and M3B are disabled.
14. M3A behavior is unchanged when M3B is disabled.

---

## Test Coverage

### New M3B Tests

| Test file | Tests | Focus |
|---|---:|---|
| `test_m3b_cluster_enumeration.py` | 13 | Deterministic cluster list, top-K, fixed exclusion, net coupling ranking |
| `test_m3b_candidate_generation.py` | 10 | ≤4 candidates, cyclic/rcyclic/centroid-step semantics, grid snapping, no clamping |
| `test_m3b_invalid_rejection.py` | 4 | OOB raw rejection, overlap rejection, non-grid boundary regression |
| `test_m3b_fixed_hard_unmoved.py` | 4 | Fixed macros excluded from clusters, ValueError on fixed input, positions unchanged |
| `test_m3b_fallback_preserved.py` | 3 | Prior winner returned when M3B adds nothing; original_raw in pool |
| `test_m3b_original_raw_invariant.py` | 3 | original_raw present with M3B enabled, disabled, zero budget |
| `test_m3b_official_score_selector.py` | 3 | Proxy-cost selector, M3B can win when it scores best, no heuristic override |
| `test_m3b_budget_exhaustion.py` | 7 | Zero budget, partial budget, mock-dominant M3B blocked, M3A unaffected by M3B exhaustion |
| `test_m3b_determinism.py` | 4 | Cluster list stable, candidate list stable, winner stable |
| `test_m3b_profile_wiring.py` | 14 | All three profiles registered, old profiles unchanged, config fields wired |

### Pre-existing Tests Preserved

All 334 pre-M3B tests continue to pass with M3B added.

---

## Candidate Counts

Cluster candidate counts scale with `m3b_top_k_clusters`:

- Per cluster: up to 3 candidates (cyclic, reverse cyclic, centroid-step)
- Centroid-step is only generated when the cluster has qualifying nets
- With `m3b_top_k_clusters=32`, maximum theoretical candidate count is 96
- Many candidates will be invalid (OOB or overlap after rotation) and rejected before scoring

---

## Invalid / Rejected Counts

M3B does not clamp or repair candidates. Invalid candidates arise from:

- **Cyclic/reverse cyclic rotations**: one macro may land on another's former position,
  causing overlap; or a macro from the far side of the chip may land OOB after rotation.
- **Centroid-step**: step may move a macro fractionally OOB near a boundary.

The `m3b_invalid`, `m3b_rejected_bounds`, `m3b_rejected_overlap`, and `m3b_rejected_other`
diagnostic fields report the rejection counts per run.

---

## Duplicate Counts

M3B candidates are deduplicated against the full pass 1–4 pool. Duplicates are tracked
in `m3b_duplicates`. In practice, M3B candidates are structurally distinct from M2B/M3A
candidates (different move semantics), so duplicates are expected to be low unless a
cluster move produces a position already generated by M2B refinement.

---

## Scored Counts

`m3b_scored` reports total M3B candidates with `was_scored=True` (including cache hits).
`m3b_fresh_scores` reports fresh official scorer invocations (excluding cache hits).

---

## Skipped-by-Budget Counts

`m3b_skipped_budget` counts valid M3B candidates that were not scored because the budget
was exhausted. When this is > 0, `m3b_budget_exhausted=True` and all M3B candidates are
excluded from final selection.

---

## Budget Exhaustion Behavior

When `m3b_budget_exhausted=True`:

- M3B diagnostic fields are preserved and visible.
- M3B candidates may remain in the ranked list for reporting.
- No `m3b_cluster_refinement` candidate participates in final selection.
- Final selection falls back to the complete pre-M3B pool (M2B + M3A-safe).
- M3A candidates are unaffected by M3B budget exhaustion.

The `m3b-budget-stress` profile exercises this behavior by setting `max_official_scores=5`,
which forces budget exhaustion on all three ibm benchmarks.

---

## Selected Source

The `winning_family` field in `ScoringDiagnostics` identifies the selected candidate's
family. With M3B enabled, this can be:

- `m3b_cluster_refinement` — M3B won
- `m3a_pair_refinement` — M3A won
- `original_refinement` / `original_line_search` / `original_neighborhood` — M2B won
- `original` — fallback to original_raw

---

## Benchmark Profiles

### `m3b-smoke`

Small cluster count (top_k=8) and M3A (top_k=16) for fast CI sanity validation.
Budget is shared (`max_official_scores=60`).

### `m3b-default`

Standard M3B run with full pipeline: M2B (refinement + line-search) + M3A (top_k=64)
+ M3B (top_k=32). Budget `max_official_scores=60`. Used for cold-run comparison against
`m3a-default` and `m2b-final`.

### `m3b-budget-stress`

Intentionally reduced budget (`max_official_scores=5`) to force budget exhaustion for
M3A and M3B. Verifies that no partial M3A or M3B candidate wins under budget pressure.

---

## Benchmark Results

All runs executed with `--clear-score-cache`.

### M3A-Default Baseline

| Benchmark | Valid | Best candidate | Family | Cost | Fresh scores |
|---|---:|---|---|---:|---:|
| ibm01 | True | `original_line_search_m215_scale2p5x` | `original_line_search` | 1.0384 | 60 |
| ibm02 | True | `original_refinement_m51_tiny0p5um_p0_p1` | `original_refinement` | 1.5584 | 60 |
| ibm03 | True | `original_refinement_m289_tiny0p25um_m1_p0` | `original_refinement` | 1.3255 | 60 |

### M3B-Default

| Benchmark | Valid | Best candidate | Family | Cost | Fresh scores | Result vs M3A |
|---|---:|---|---|---:|---:|---|
| ibm01 | True | `original_line_search_m215_scale2p5x` | `original_line_search` | 1.0384 | 60 | Tie |
| ibm02 | True | `original_refinement_m51_tiny0p5um_p0_p1` | `original_refinement` | 1.5584 | 60 | Tie |
| ibm03 | True | `original_refinement_m289_tiny0p25um_m1_p0` | `original_refinement` | 1.3255 | 60 | Tie |

### M3B-Default Interpretation

M3B-default is valid on all three benchmarks and is not worse than M3A-default.

M3B does not strictly improve on these benchmarks. The M2B refinement and line-search passes
exhaust the `max_official_scores=60` budget before M3B candidates can be scored. With the
shared budget fully consumed by prior passes, M3B candidates are generated and validated but
never scored — they cannot compete in selection.

This is the correct safety behavior (invariant 9: unscored candidates are never selectable)
and is documented per the spec ("If it does not improve, do not overfit, add clamping, expand
budget, or mutate M2B/M3A behavior. Document the tie/non-improvement in M3B_REPORT.md").

The M3B family does appear in the candidate diversity output (`families` includes
`m3b_cluster_refinement`), confirming generation succeeds and candidates enter the pool.

### M3B-Smoke

| Benchmark | Valid | Best candidate | Family | Cost | Fresh scores |
|---|---:|---|---|---:|---:|
| ibm01 | True | `original_line_search_m215_scale2p5x` | `original_line_search` | 1.0384 | 60 |
| ibm02 | True | `original_refinement_m51_tiny0p5um_p0_p1` | `original_refinement` | 1.5584 | 60 |
| ibm03 | True | `original_refinement_m289_tiny0p25um_m1_p0` | `original_refinement` | 1.3255 | 57 |

Smoke confirms M3B candidates enter the pool (`m3b_cluster_refinement` appears in families)
and the pipeline is stable.

### M3B-Budget-Stress

| Benchmark | Valid | Best candidate | Family | Cost | Fresh scores | Result |
|---|---:|---|---|---:|---:|---|
| ibm01 | True | `original_refinement_m117_tiny0p1um_m1_p0` | `original_refinement` | 1.0385 | 5 | Pass |
| ibm02 | True | `original_refinement_m51_scale0p25x` | `original_refinement` | 1.5629 | 5 | Pass |
| ibm03 | True | `original_refinement_m72_tiny0p1um_p1_p0` | `original_refinement` | 1.3255 | 5 | Pass |

No M3B (or M3A) candidate wins under the 5-score budget. All winners come from
`original_refinement`, confirming the budget-exhaustion guard blocks partial M3B selection.

### Acceptance Gate

The hard acceptance gates are:

- All existing M2B and M3A tests pass. ✓ (399/399 tests pass)
- All new M3B tests pass. ✓ (65 new tests pass)
- `m3b-smoke` runs successfully. ✓
- `m3b-budget-stress` is valid and does not select partial M3B candidates. ✓
- `m3b-default` is valid on ibm01/ibm02/ibm03. ✓
- `m3b-default` is not worse than `m3a-default`/`m2b-final` on ibm01/ibm02/ibm03. ✓ (ties)

---

## Known Notes

### Candidate 4 Skipped

Compact-to-centroid with rank-order preservation is omitted (see Implementation Summary).
This reduces candidates per cluster from 4 to ≤3.

### Float32 Grid Tolerance

Generated coordinates snap to the 0.05 µm grid in Python (float64). When stored in
float32 tensors, conversion error of up to ~1e-5 may occur for coordinate values around
100 µm. Tests use a 1e-3 fractional-quotient tolerance to accommodate this. The values
remain functionally on-grid to float32 precision.

### Cluster Count on Dense Benchmarks

Dense net structures may yield many qualified triples. The `m3b_top_k_clusters` parameter
caps the count. With `top_k=32` the maximum fresh-score consumption from M3B alone is
capped at 32 × 3 = 96 candidates, subject to the shared `max_official_scores` budget.

---

## Acceptance Decision

**M3B is accepted and ready to freeze.**

Acceptance evidence:

- 399/399 tests pass (334 pre-M3B + 65 new M3B tests).
- M3B-default cold run is valid on ibm01/ibm02/ibm03.
- M3B-default ties M3A-default on all three required cold benchmarks.
- M3B-budget-stress is valid on ibm01/ibm02/ibm03.
- Reduced-budget M3B does not select partial M3B candidates.
- Invalid M3B candidates are rejected before scoring.
- M3B does not clamp or repair bounds.
- M2B-final and M3A-default remain the safety nets.
- All 14 hard invariants hold.

Final classification:

**M3B: ACCEPTED / FROZEN**
**Result:** Safe non-regression extension
**Performance:** Ties M3A-default on required cold benchmarks
**Non-improvement reason:** M2B passes exhaust the `max_official_scores=60` budget; M3B candidates are generated and validated but not scored within the shared budget
**Next milestone:** M3C or budget-split strategy to reserve scoring slots for M3B
