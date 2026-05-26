# M4A Reduced Spec — Score-Loss Attribution (Artifact-Only)

**Milestone:** M4A  
**Theme:** Diagnose why M3D produced no meaningful score improvement.  
**Primary output:** Evidence-backed decision on what M4B should attack first.  
**Profile baseline:** `m3c-default`  
**Benchmarks:** `ibm01`, `ibm02`, `ibm03`  
**Status:** Implementation-ready. Replaces the bloated M4A spec for Codex implementation.

---

## 1. Milestone Summary

M4A is a **diagnostic milestone, not an optimization milestone**. It reads existing
M3D artifacts, computes structured diagnostics, and produces a clear recommendation
for what M4B should implement.

M4A must answer:

> What is blocking score improvement, and which of the following failure modes best
> explains the M3D result?

| Failure mode | M4B implication |
|---|---|
| **legality_bottleneck** | m3a/m3b families generate many invalid candidates; valid rate << baseline |
| **prefilter_evaluator_disagreement** | HPWL prefilter rank disagrees with evaluator rank; score budget is wasted |
| **candidate_diversity_collapse** | Scored candidates cluster around the same macros / placement hashes |
| **local_exhaustion_under_sampled_families** | Budget saturated, valid rate healthy, no clear other cause |
| **inconclusive** | Insufficient signal; instrument more before changing optimizer |

M4A does not improve scores. It explains why scores did not improve.

---

## 2. Non-Goals

M4A does **not**:

- change solver behavior or selection logic
- add, remove, or tune candidate families
- call the official scorer
- consume fresh official scores
- load or replay `.pt` geometry files
- compute region/density/hotspot attribution
- compute per-macro displacement from placement geometry
- compute per-net HPWL attribution from netlist
- compute top-macro or top-net contributor rankings
- prove topology lock from geometry
- emit empty-array placeholders for unsupported geometry metrics

---

## 3. Inputs

M4A reads **only** committed artifacts. No solver execution, no scorer invocation.

| Input | Path |
|---|---|
| Benchmark summary | `analysis/m3d/m3d_benchmark_summary.csv` |
| Family summary | `analysis/m3d/m3d_family_summary.csv` |
| Candidate effectiveness | `analysis/m3d/m3d_candidate_effectiveness.csv` |
| Runner JSON | `submissions/solver/artifacts/run_m3c-default.json` |

Benchmark `.pt` geometry files are **intentionally not required** for reduced M4A.

### 3.1 Actual field inventory

**`m3d_benchmark_summary.csv` columns used:**

```
benchmark, profile, selected_candidate, selected_family,
selected_cost, original_cost, classification,
late_stage_generated, late_stage_valid, late_stage_scored
```

Note: `classification` in this file is the M3D label (`near_local_optimum`); M4A
produces its own classification and does **not** inherit this value.

**`m3d_family_summary.csv` columns used:**

```
benchmark, profile, family,
generated_count, valid_count, invalid_count, duplicate_count,
admitted_count, not_admitted_count, scored_count, skipped_budget_count,
best_official_cost, best_official_delta_vs_final,
median_official_cost, median_official_delta_vs_final,
num_beating_final, num_near_tie, best_candidate_name
```

**`m3d_candidate_effectiveness.csv` columns used:**

```
benchmark, profile, candidate_name, family,
valid, duplicate, admitted, not_admitted, scored, skip_reason,
proxy_cost, approx_delta, is_selected, scored_pool_selectable,
placement_hash, source_stage
```

**`run_m3c-default.json` fields used (per benchmark result):**

```
benchmark, proxy_cost, raw_original_cost, delta_vs_raw_original,
official_scored_count, max_official_scores, fresh_official_scores,
duplicate_skipped_count, prefiltered_count, prefilter_mode,
selected_due_to
```

---

## 4. Data Model and Terminology

### 4.1 Terminology corrections (mandatory)

The following terminology must be used consistently throughout M4A output. These
corrections apply to all generated reports, JSON, and CSV files.

| M3D artifact field | M4A meaning | M4A term |
|---|---|---|
| `proxy_cost` | Evaluator cost returned by the official scorer | `evaluator_cost` |
| `approx_delta` / `approx_delta_hpwl` | HPWL-based prefilter signal (not official score) | `approx_delta` (retain name, clarify in docs) |
| M3D `classification = near_local_optimum` | Not inherited | _(see Section 6)_ |
| "proxy vs official mismatch" | Prefilter rank disagrees with evaluator rank | "approx-prefilter vs evaluator disagreement" |

Rules:

- **`proxy_cost` in M3D artifacts is treated as `evaluator_cost` in all M4A
  outputs.** It is not a separate proxy; it is the official evaluator output.
- **`approx_delta`** is the HPWL prefilter signal used to skip low-scoring
  candidates before evaluator calls. It is not an official score.
- **M4A does not inherit the M3D `near_local_optimum` classification.** That
  label is a catch-all; M4A replaces it with specific evidence-backed labels
  from Section 6.
- **Lower evaluator cost is assumed better.** All delta computations follow
  `delta = original_cost - selected_cost` (positive = improvement). This
  assumption must be guarded explicitly: if `original_cost < selected_cost`,
  emit a warning in the report caveats.

---

## 5. Required Diagnostics

### 5.1 Score Banding

Source: `m3d_benchmark_summary.csv` + `run_m3c-default.json`

Compute per benchmark:

```python
original_cost  = row["original_cost"]          # from benchmark_summary or runner JSON
selected_cost  = row["selected_cost"]           # proxy_cost of selected candidate
delta          = original_cost - selected_cost  # positive = improvement
relative_delta = delta / original_cost
```

Score band (parameter `official_epsilon = 1e-5` unless passed explicitly):

| Band | Condition |
|---|---|
| `meaningful_win` | `delta > 10 * official_epsilon` |
| `epsilon_win` | `0 < delta <= 10 * official_epsilon` |
| `flat` | `abs(delta) <= official_epsilon` |
| `regression` | `delta < -official_epsilon` |

Note: benchmarks classified as `meaningful_win` must not be described as "no
meaningful improvement" in any report text, even if further improvement is
desired.

---

### 5.2 Family Effectiveness

Source: `m3d_family_summary.csv`

Compute per (benchmark, family):

```python
valid_rate        = valid_count / generated_count  # 0.0 if generated_count == 0
score_rate        = scored_count / valid_count      # 0.0 if valid_count == 0
budget_share      = scored_count / total_scored_count_for_benchmark
best_eval_cost    = best_official_cost              # null if not present
best_eval_delta   = best_official_delta_vs_final    # null if not present
median_eval_cost  = median_official_cost            # null if not present
median_eval_delta = median_official_delta_vs_final  # null if not present
```

Note: `best_official_delta_vs_final` is relative to the selected final cost, not
to `original_cost`. Document this in the report.

Family classification helper (used by Section 6 rules):

- **local families**: families whose name starts with `m3a_` or `m3b_`
- **baseline families**: families whose name starts with `original`

---

### 5.3 Skip Reason Aggregation

Source: `m3d_candidate_effectiveness.csv`

Aggregate per (benchmark, family):

| Bucket | Source column / condition |
|---|---|
| `invalid_count` | `valid == False` (and not `duplicate`) |
| `duplicate_count` | `duplicate == True` |
| `budget_skip_count` | `scored == False and not_admitted == True` (skipped due to budget) |
| `prefilter_skip_count` | `scored == False and admitted == True` with prefilter-related `skip_reason` |
| `scored_count` | `scored == True` |
| `skip_reason_distribution` | Frequency count of non-null `skip_reason` values |

**Do not infer root causes from skip reasons that are not actually present** in the
`skip_reason` column. If the column is empty or contains only `"scored"`, emit a
caveat that skip reason granularity is unavailable.

---

### 5.4 Budget Use

Source: `run_m3c-default.json` (per benchmark result)

Compute per benchmark:

```python
official_scored_count     = result["official_scored_count"]
max_official_scores       = result["max_official_scores"]
budget_saturation         = official_scored_count / max_official_scores
duplicate_skipped_count   = result["duplicate_skipped_count"]
prefiltered_count         = result["prefiltered_count"]
```

Report per-family share of `scored_count` (from family summary) as fraction of
`official_scored_count` (from runner JSON).

---

### 5.5 Approx-Prefilter vs Evaluator Disagreement

Source: `m3d_candidate_effectiveness.csv`

Among scored candidates (`scored == True`) with non-null `approx_delta` and
non-null `proxy_cost` (treated as `evaluator_cost`):

```python
usable_candidates   = [c for c in scored if c.approx_delta is not None
                                         and c.proxy_cost is not None]
usable_count        = len(usable_candidates)
approx_coverage     = usable_count / total_scored_count
```

Guards — **do not compute or emit disagreement classification if**:
- `usable_count < 20`, OR
- `approx_coverage < 0.50`

When guards pass, compute:

- Spearman rank correlation between `approx_delta` (ascending = lower HPWL delta =
  prefilter-preferred) and `proxy_cost` (ascending = lower cost = evaluator-preferred)
- Top-5 and top-20 rank inversion indicators: for each candidate in the top-K by
  evaluator cost, report its rank by `approx_delta` (inversion = evaluator-top
  candidate is in the approx-bottom half)

Note: `approx_delta` values for m3a/m3b families are null in current M3D artifacts,
so `approx_coverage` will be low unless baseline families dominate the scored set.
The guards will suppress disagreement analysis in that case; emit a caveat explaining
why.

---

### 5.6 Candidate Name-Set Diversity

Source: `m3d_candidate_effectiveness.csv`

Using candidate names and `placement_hash`:

```python
# Parse macro IDs from candidate names where the pattern allows.
# e.g. "m3a_p16_5_154_swap"            -> macro IDs [5, 154]
# e.g. "original_refinement_m215_scale2x" -> macro ID [215]
# e.g. "m3b_c30_0_52_166_centroid_shift"  -> macro IDs [52, 166]
# Parsing is best-effort; unmatched names contribute 0 unique macros.

unique_macros_in_scored    = len(set of all parsed macro IDs across scored candidates)
unique_macro_ratio         = unique_macros_in_scored / total_scored_count
most_touched_macros        = top-5 macros by frequency across scored candidates

placement_hash_collisions  = scored_count - len(set of placement_hash values)
collision_ratio            = placement_hash_collisions / scored_count
  # null if placement_hash column is absent or all null
```

**Label this as weak diversity evidence, not geometry diversity.** Candidate
name-set diversity cannot substitute for geometry-based placement diversity.

---

### 5.7 Unsupported Diagnostics

The following diagnostics **cannot** be computed in reduced M4A because placement
geometry was not persisted in M3D artifacts. The report must list these explicitly
rather than emitting empty arrays or zero values.

Unsupported list (include verbatim in report and JSON):

```
- region_attribution          (die grid not available)
- density_hotspots            (macro coordinates not persisted)
- top_macro_contributor       (netlist + geometry not available)
- top_net_contributor         (netlist not available)
- topology_lock_proof         (geometry comparison not available)
- legalization_displacement   (pre/post legalizer positions not persisted)
```

---

## 6. Classification Rules

One primary classification per benchmark. Rules are evaluated in priority order;
first matching rule wins. All rules are deterministic.

### Notation

```
eps                     = official_epsilon (default 1e-5)
delta                   = original_cost - selected_cost
valid_rate_local        = weighted average valid_rate for local families
                          (m3a_*, m3b_*), weighted by generated_count
valid_rate_baseline     = weighted average valid_rate for baseline families
                          (original_*), weighted by generated_count
scored_count            = official_scored_count  (from runner JSON)
budget                  = max_official_scores    (from runner JSON)
spearman_rs             = Spearman rank corr(approx_delta, proxy_cost) over usable candidates
usable_prefilter_count  = count of scored candidates with non-null approx_delta + proxy_cost
approx_coverage         = usable_prefilter_count / scored_count
unique_macro_ratio      = unique_macros_in_scored / scored_count
collision_ratio         = placement_hash_collisions / scored_count  (null if hashes unavailable)
```

---

### Rule A — `legality_bottleneck`

Fires when **all** of:

1. `valid_rate_local < 0.20`
2. `valid_rate_baseline > 0.80`
3. `delta <= 10 * eps`

**Interpretation:** Local families generate mostly invalid candidates while baseline
families are healthy; invalidity is the primary reason good placements are not found.

**M4B:** legalization-aware regional repair

---

### Rule B — `prefilter_evaluator_disagreement`

Fires when **all** of:

1. `usable_prefilter_count >= 20`
2. `approx_coverage >= 0.50`
3. `spearman_rs < 0.30`
4. Top-5 rank inversions >= 3 (at least 3 of the top-5 evaluator candidates are in
   the bottom half by approx_delta rank)

**Interpretation:** The HPWL prefilter systematically disagrees with the evaluator;
score budget is being wasted on prefilter-preferred but evaluator-poor candidates.

**M4B:** correlation-aware ranking / prefilter repair

---

### Rule C — `candidate_diversity_collapse`

Fires when **all** of:

1. `unique_macro_ratio < 0.40` (or `collision_ratio > 0.20` if hashes available)
2. `delta <= 10 * eps`

**Interpretation:** Scored candidates repeatedly touch the same small set of macros
or share placement hashes; the search is stuck in a narrow basin.

**M4B:** beam search / elite pool / multi-start

---

### Rule D — `local_exhaustion_under_sampled_families`

Fires when **all** of:

1. `delta <= 10 * eps`
2. `scored_count >= 0.80 * budget`
3. Rules A, B, C do not fire

**Interpretation:** Budget was nearly exhausted with no legality, prefilter, or
diversity explanation; current families have been sampled thoroughly and are simply
not producing improvement.

**M4B:** regional destroy-and-repair

---

### Rule E — `inconclusive`

Fires otherwise (no prior rule matched).

**Interpretation:** Available signal is insufficient to attribute the failure mode.
Persist more instrumentation before implementing optimizer changes.

**M4B:** instrument more before optimizer changes

---

### Classification note

Benchmarks with `score_band == meaningful_win` still receive a classification based
on the above rules (they may be at a legality bottleneck or diversity collapse for
further improvement). Do not describe them as having "no meaningful improvement."

---

## 7. M4B Recommendation Mapping

### Per-benchmark

| Classification | M4B direction | Rationale |
|---|---|---|
| `legality_bottleneck` | Legalization-aware regional repair | Invalid local-family candidates waste generated count |
| `prefilter_evaluator_disagreement` | Correlation-aware ranking / prefilter repair | Score budget spent on prefilter-preferred but evaluator-rejected candidates |
| `candidate_diversity_collapse` | Beam search / elite pool / multi-start | Narrow basin; need broader exploration before scoring |
| `local_exhaustion_under_sampled_families` | Regional destroy-and-repair | Families thoroughly sampled; need larger structural moves |
| `inconclusive` | Instrument more before optimizer changes | Insufficient data to choose optimizer direction |

### Aggregate recommendation (ibm01 / ibm02 / ibm03)

| Outcome | Aggregate M4B |
|---|---|
| All three agree | Choose that M4B |
| Two agree, one `inconclusive` | Choose majority |
| Two agree, one differs (non-inconclusive) | Choose majority with scoped note for dissenter |
| Three-way split | Instrumentation-first before optimizer changes |

---

## 8. Required Outputs

All outputs go to `--output-dir` (default: `submissions/solver/reports/`).

| File | Description |
|---|---|
| `m4a_loss_attribution_report.md` | Human-readable report with all diagnostics and M4B recommendation |
| `m4a_loss_attribution.json` | Machine-readable summary (see Section 9) |
| `m4a_family_effectiveness.csv` | Per-(benchmark, family) effectiveness rows |
| `m4a_prefilter_vs_evaluator.csv` | Per scored candidate: approx_delta, evaluator_cost, approx_rank, evaluator_rank |

Every output must include:

- input artifact paths (as listed)
- `official_epsilon` value used
- profile name
- list of supported diagnostics computed
- list of unsupported diagnostics (from Section 5.7), not empty arrays
- per-benchmark: score band, classification, M4B recommendation
- caveats section (see Section 13)

---

## 9. JSON Schema Sketch

```json
{
  "profile": "m3c-default",
  "official_epsilon": 1e-5,
  "inputs": {
    "benchmark_summary": "analysis/m3d/m3d_benchmark_summary.csv",
    "family_summary": "analysis/m3d/m3d_family_summary.csv",
    "candidate_effectiveness": "analysis/m3d/m3d_candidate_effectiveness.csv",
    "runner_json": "submissions/solver/artifacts/run_m3c-default.json"
  },
  "supported_diagnostics": [
    "score_banding",
    "family_effectiveness",
    "skip_reason_aggregation",
    "budget_use",
    "approx_prefilter_vs_evaluator",
    "candidate_name_set_diversity"
  ],
  "unsupported_diagnostics": [
    "region_attribution",
    "density_hotspots",
    "top_macro_contributor",
    "top_net_contributor",
    "topology_lock_proof",
    "legalization_displacement"
  ],
  "benchmarks": {
    "ibm01": {
      "costs": {
        "original_cost": 1.038498,
        "selected_cost": 1.038421,
        "delta": 7.7e-5,
        "relative_delta_pct": 0.0074
      },
      "score_band": "epsilon_win",
      "family_effectiveness": [
        {
          "family": "m3a_pair_refinement",
          "generated_count": 384,
          "valid_count": 40,
          "valid_rate": 0.104,
          "scored_count": 5,
          "score_rate": 0.125,
          "selected_count": 0,
          "best_evaluator_cost": 1.038433,
          "best_evaluator_delta": 1.19e-5,
          "median_evaluator_cost": 1.038485,
          "budget_share": 0.089
        }
      ],
      "budget": {
        "official_scored_count": 56,
        "max_official_scores": 60,
        "budget_saturation": 0.933,
        "duplicate_skipped_count": 28,
        "prefiltered_count": 26
      },
      "prefilter_evaluator": {
        "usable_count": 18,
        "approx_coverage": 0.321,
        "guard_fired": true,
        "guard_reason": "approx_coverage < 0.50",
        "spearman_rs": null,
        "top5_inversions": null
      },
      "diversity": {
        "unique_macros_in_scored": 12,
        "unique_macro_ratio": 0.214,
        "most_touched_macros": ["m215", "m22", "m173"],
        "placement_hash_collisions": 0,
        "collision_ratio": 0.0,
        "diversity_note": "weak evidence — name-set only, not geometry"
      },
      "classification": "legality_bottleneck",
      "classification_reasons": [
        "valid_rate_local=0.076 < 0.20",
        "valid_rate_baseline=1.00 > 0.80",
        "delta=7.7e-5 <= 10*epsilon=1e-4"
      ],
      "m4b_recommendation": "legalization_aware_regional_repair",
      "caveats": []
    }
  },
  "aggregate_recommendation": {
    "classifications": {"ibm01": "...", "ibm02": "...", "ibm03": "..."},
    "outcome": "...",
    "m4b": "..."
  },
  "caveats": [
    "Geometry not persisted; region, density, per-macro displacement, and per-net HPWL attribution are unsupported by reduced M4A.",
    "proxy_cost in M3D artifacts is treated as evaluator cost in M4A, not as a separate proxy.",
    "M4A does not inherit M3D's near_local_optimum label.",
    "approx_delta is absent for m3a/m3b families; prefilter-evaluator disagreement analysis is suppressed when approx_coverage < 0.50.",
    "Candidate name-set diversity is weak evidence and is not a substitute for geometry diversity."
  ]
}
```

**Do not include** `top_hotspots`, `top_macros`, `top_nets`, or any
geometry-derived keys as empty arrays. Unsupported diagnostics appear only in
the `unsupported_diagnostics` list.

---

## 10. CLI Contract

```bash
python -m submissions.solver.m4a_loss_attribution \
  --profile m3c-default \
  --benchmarks ibm01 ibm02 ibm03 \
  --official-epsilon 1e-5 \
  --input-dir analysis/m3d \
  --runner-json submissions/solver/artifacts/run_m3c-default.json \
  --output-dir submissions/solver/reports
```

Notes:

- Profile is `m3c-default`, not `m3d-default`. The M3D analysis used the M3C run
  profile.
- Artifact-only mode only. The CLI must not have a flag to invoke the scorer or
  solver.
- `--official-epsilon` default is `1e-5`; passed through to all classification
  rules and score banding.
- Missing input files should produce a clear error. Partial inputs (e.g., missing
  runner JSON) should fall back gracefully with caveats rather than crashing, but
  must emit a prominent warning.

---

## 11. Implementation Constraints

### Allowed new files

```
submissions/solver/m4a_loss_attribution.py
submissions/solver/tests/test_m4a_loss_attribution.py
submissions/solver/reports/m4a_loss_attribution_report.md   (generated)
submissions/solver/reports/m4a_loss_attribution.json        (generated)
submissions/solver/reports/m4a_family_effectiveness.csv     (generated)
submissions/solver/reports/m4a_prefilter_vs_evaluator.csv   (generated)
```

### Forbidden modifications

```
submissions/solver/candidates/     (all candidate generators)
submissions/solver/scorer.py       (or any scorer module)
submissions/solver/selection*.py   (selection logic)
submissions/solver/score_cache*    (score cache)
submissions/solver/profiles/       (configs)
submissions/solver/runner.py       (runner)
analysis/m3d/                      (M3D artifacts — read only)
submissions/solver/artifacts/      (runner JSON artifacts — read only)
**/*.pt                            (benchmark geometry files)
```

### Hard constraints

1. **No scorer import.** `m4a_loss_attribution.py` must not import the scorer
   module or any module that calls the official scorer.
2. **No solver execution.** No subprocess calls to the solver or runner.
3. **No writes outside `--output-dir`.**
4. **No fresh official score consumption.**
5. **Unsupported diagnostics are omitted, not zero-filled.** Do not emit
   `"top_hotspots": []` or similar.

---

## 12. Required Tests

File: `submissions/solver/tests/test_m4a_loss_attribution.py`

| # | Test | Description |
|---|---|---|
| 1 | `test_score_banding` | All four bands (meaningful_win, epsilon_win, flat, regression) with synthetic costs and epsilon |
| 2 | `test_classification_rule_a` | legality_bottleneck fires with valid_rate_local<0.20, valid_rate_baseline>0.80, delta≤10*eps |
| 3 | `test_classification_rule_b` | prefilter_evaluator_disagreement fires / suppressed based on guard conditions |
| 4 | `test_classification_rule_c` | candidate_diversity_collapse fires based on unique_macro_ratio and collision_ratio |
| 5 | `test_classification_rule_d` | local_exhaustion fires when budget saturated and A/B/C do not fire |
| 6 | `test_classification_rule_e` | inconclusive fires when no other rule matches |
| 7 | `test_terminology_mapping` | Output JSON contains `evaluator_cost` not `proxy_cost`; `unsupported_diagnostics` contains geometry items |
| 8 | `test_unsupported_diagnostics_absent` | Output JSON does not contain `top_hotspots`, `top_macros`, `top_nets` keys anywhere |
| 9 | `test_no_near_local_optimum_label` | Output JSON does not use `"near_local_optimum"` as a classification value |
| 10 | `test_cli_integration` | CLI runs against committed M3D artifacts; all three benchmarks appear in output JSON |
| 11 | `test_no_scorer_import` | Importing `m4a_loss_attribution` does not trigger scorer import (check `sys.modules`) |
| 12 | `test_output_files_exist` | After CLI run, all four output files exist and are non-empty |
| 13 | `test_json_structural_regression` | Output JSON has required top-level keys: profile, official_epsilon, inputs, supported_diagnostics, unsupported_diagnostics, benchmarks, aggregate_recommendation, caveats |
| 14 | `test_prefilter_guard` | When `usable_count < 20` or `approx_coverage < 0.50`, `guard_fired == true` and `spearman_rs == null` |

---

## 13. Required Caveats

The following caveat text must appear verbatim (or substantively equivalent) in the
generated Markdown report and in the top-level `caveats` array of the JSON:

1. Geometry not persisted; region, density, per-macro displacement, and per-net
   HPWL attribution are unsupported by reduced M4A.
2. `proxy_cost` in M3D artifacts is treated as evaluator cost in M4A, not as a
   separate proxy.
3. M4A does not inherit M3D's `near_local_optimum` label; M4A classifications are
   derived independently from rules A–E.
4. Cache decomposition (cache_hits/cache_misses) is omitted if those fields are
   zero or inactive in the runner JSON.
5. Candidate name-set diversity is weak evidence and is not a substitute for
   geometry diversity.
6. `approx_delta` is absent for m3a/m3b family candidates in current M3D artifacts;
   prefilter-evaluator disagreement analysis is automatically suppressed when
   `approx_coverage < 0.50`.
7. `best_official_delta_vs_final` in M3D family summary is relative to the final
   selected cost, not to `original_cost`; do not interpret it as improvement over
   baseline.

---

## 14. Definition of Done

Reduced M4A is complete when:

- [ ] Tests pass (`pytest submissions/solver/tests/test_m4a_loss_attribution.py`)
- [ ] CLI runs without error on committed ibm01/ibm02/ibm03 artifacts
- [ ] All four output files are generated and non-empty
- [ ] Every benchmark has: `score_band`, `classification`, `m4b_recommendation`
- [ ] `unsupported_diagnostics` list is present and non-empty in JSON
- [ ] Output JSON contains no geometry-derived empty arrays
- [ ] Output JSON does not use `near_local_optimum` as a classification value
- [ ] No scorer or solver behavior was changed
- [ ] Aggregate recommendation is present and derived from per-benchmark results
- [ ] Report is sufficient to choose or defer M4B without additional analysis

---

## 15. Codex Implementation Prompt

Copy the following prompt verbatim to Codex:

---

```
Implement milestone M4A (reduced, artifact-only) for the MacroPlacement project.

## Goal

Produce a diagnostic tool that reads existing M3D CSV/JSON artifacts and outputs
structured score-loss attribution reports. This milestone does NOT improve scores.
It determines what M4B should implement.

## Effort

High. Follow the spec precisely. Do not add geometry features, do not call the
scorer, do not modify solver behavior.

## Inputs (read-only)

- analysis/m3d/m3d_benchmark_summary.csv
- analysis/m3d/m3d_family_summary.csv
- analysis/m3d/m3d_candidate_effectiveness.csv
- submissions/solver/artifacts/run_m3c-default.json

## New files you may create

- submissions/solver/m4a_loss_attribution.py        (main module)
- submissions/solver/tests/test_m4a_loss_attribution.py

## Generated outputs (written to --output-dir)

- m4a_loss_attribution_report.md
- m4a_loss_attribution.json
- m4a_family_effectiveness.csv
- m4a_prefilter_vs_evaluator.csv

## Forbidden modifications

Do NOT touch:
- Any file under submissions/solver/candidates/
- submissions/solver/scorer.py (or any scorer module)
- Any selection, runner, score_cache, or profile file
- Any file under analysis/m3d/ (read only)
- Any file under submissions/solver/artifacts/ (read only)
- Any *.pt benchmark file

## Hard constraints

1. m4a_loss_attribution.py must NOT import the scorer module.
2. No subprocess calls to the solver or runner.
3. No writes outside --output-dir.
4. No fresh official score consumption.
5. Do NOT emit empty arrays for unsupported geometry metrics. List them in
   unsupported_diagnostics only.
6. Do NOT classify any benchmark as "near_local_optimum". Use rules A-E only.

## Terminology

- M3D field `proxy_cost` = evaluator cost in M4A output. Label it `evaluator_cost`.
- `approx_delta` = HPWL prefilter signal, NOT an official score. Retain field name
  but document clearly.
- Attribution section heading is "approx-prefilter vs evaluator disagreement", not
  "proxy/official mismatch".

## CLI

python -m submissions.solver.m4a_loss_attribution \
  --profile m3c-default \
  --benchmarks ibm01 ibm02 ibm03 \
  --official-epsilon 1e-5 \
  --input-dir analysis/m3d \
  --runner-json submissions/solver/artifacts/run_m3c-default.json \
  --output-dir submissions/solver/reports

## Diagnostics to implement

1. Score banding (meaningful_win / epsilon_win / flat / regression)
2. Family effectiveness per (benchmark, family)
3. Skip reason aggregation
4. Budget use (saturation ratio, per-family share)
5. Approx-prefilter vs evaluator disagreement
   - Guard: skip if usable_count < 20 or approx_coverage < 0.50
   - When guard passes: Spearman rank correlation + top-5 inversion count
6. Candidate name-set diversity (parsed macro IDs + placement_hash collisions)
7. Unsupported diagnostics list (region, density, top_macro, top_net,
   topology_lock, legalization_displacement) - listed in output, NOT as empty arrays

## Classification rules (deterministic, evaluated in priority order)

A. legality_bottleneck:
   valid_rate_local < 0.20 AND valid_rate_baseline > 0.80 AND delta <= 10*eps
   where local = m3a_*/m3b_* families, baseline = original_* families

B. prefilter_evaluator_disagreement:
   usable_count >= 20 AND approx_coverage >= 0.50 AND spearman_rs < 0.30
   AND top5_inversions >= 3

C. candidate_diversity_collapse:
   unique_macro_ratio < 0.40 AND delta <= 10*eps
   (OR collision_ratio > 0.20 if placement_hash available)

D. local_exhaustion_under_sampled_families:
   delta <= 10*eps AND budget_saturation >= 0.80 AND A/B/C do not fire

E. inconclusive: otherwise

## Aggregate recommendation

- All agree -> that M4B
- Two agree, one inconclusive -> majority
- Two agree, one different -> majority + scoped note
- Three-way split -> instrument first

## Required tests (14 total, in test_m4a_loss_attribution.py)

1. test_score_banding - all four bands with synthetic data
2. test_classification_rule_a - legality_bottleneck fires and does not fire
3. test_classification_rule_b - prefilter disagreement fires / suppressed by guards
4. test_classification_rule_c - diversity collapse fires based on thresholds
5. test_classification_rule_d - local_exhaustion fires when budget saturated
6. test_classification_rule_e - inconclusive fires when no rule matches
7. test_terminology_mapping - evaluator_cost in output, proxy_cost absent
8. test_unsupported_diagnostics_absent - no geometry empty arrays in JSON
9. test_no_near_local_optimum_label - string absent as classification value
10. test_cli_integration - CLI runs, all three benchmarks in output JSON
11. test_no_scorer_import - importing module does not load scorer (sys.modules check)
12. test_output_files_exist - all four output files present and non-empty
13. test_json_structural_regression - required top-level keys present
14. test_prefilter_guard - guard_fired=true and spearman_rs=null when conditions unmet

## Required caveats in output (include in both .md and .json)

1. Geometry not persisted; region/density/macro/net attribution unsupported.
2. proxy_cost treated as evaluator_cost in M4A output.
3. near_local_optimum label not inherited from M3D.
4. cache_hits/cache_misses decomposition omitted if zero/inactive.
5. Name-set diversity is weak evidence only, not geometry diversity.
6. approx_delta absent for m3a/m3b families; prefilter analysis suppressed when
   approx_coverage < 0.50.
7. best_official_delta_vs_final is relative to selected cost, not original_cost.

## After implementation

Run:
  python -m pytest submissions/solver/tests/test_m4a_loss_attribution.py -v
  python -m submissions.solver.m4a_loss_attribution --profile m3c-default \
    --benchmarks ibm01 ibm02 ibm03 --official-epsilon 1e-5 \
    --input-dir analysis/m3d \
    --runner-json submissions/solver/artifacts/run_m3c-default.json \
    --output-dir submissions/solver/reports

Provide a final summary with:
- Changed/created files
- Test results (pass/fail counts)
- Per-benchmark classification and M4B recommendation from the generated JSON
- Any caveats that fired
```

---

*End of reduced M4A spec. This document replaces `docs/M4A_milestone.md` for
implementation purposes. The original spec is preserved for historical reference.*
