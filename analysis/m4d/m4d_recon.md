# Codex Reconnaissance Report — M4D

Reconnaissance only. No implementation performed.

## 1. Verdict

**Implementable with constraints**

Confidence: **0.73**

Why:
- The repo already has a clean place to insert budget-aware admission logic without touching scorer/cache/legalizer internals: [`submissions/solver/core/candidate_scoring.py:832`](submissions/solver/core/candidate_scoring.py:832) through [`submissions/solver/core/candidate_scoring.py:1717`](submissions/solver/core/candidate_scoring.py:1717).
- Read-only artifact/export plumbing is already in place for new ranking telemetry: [`submissions/solver/core/m3d_candidate_export.py:13`](submissions/solver/core/m3d_candidate_export.py:13) and [`submissions/solver/tests/test_m3d_candidate_export.py:107`](submissions/solver/tests/test_m3d_candidate_export.py:107).
- The observed mismatch is real, but it is not a single stable global ratio. It varies materially by benchmark and family, so M4D should stay simple: family-normalized score plus conservative quota floors, not a learned/global scalar.

Main constraint:
- The accepted M4C path already depends on the M4B reserved bucket. M4D should preserve M4C winners' selectability and avoid rewriting M4B generation or M4C within-family ranking.

## 2. Repository Map

| Area | Path | Relevant function/section | Purpose | M4D action |
|---|---|---|---|---|
| Score budget allocation across families | `submissions/solver/core/candidate_scoring.py` | `score_and_select`, M3C budget block [`832-898`](submissions/solver/core/candidate_scoring.py:832), M3A/M3B frontiers [`1281-1511`](submissions/solver/core/candidate_scoring.py:1281), M4B reserved block [`1549-1717`](submissions/solver/core/candidate_scoring.py:1549) | All budget slicing and frontier admission happen here | **Modify** |
| Candidate admission/sorting | `submissions/solver/core/candidate_scoring.py` | `_prefilter_score_set` [`245-297`](submissions/solver/core/candidate_scoring.py:245), pass-1 sort key [`929-946`](submissions/solver/core/candidate_scoring.py:929), refinement queue build [`1036-1096`](submissions/solver/core/candidate_scoring.py:1036), line-search scorer ordering [`375-470`](submissions/solver/core/candidate_scoring.py:375) | Current prefilter and score order | **Modify cautiously** |
| Current `approx_delta` use | `submissions/solver/core/original_neighborhood.py` | `_approx_delta_hpwl` [`113-125`](submissions/solver/core/original_neighborhood.py:113), candidate metadata write [`225-241`](submissions/solver/core/original_neighborhood.py:225) | Defines base approx delta signal | **Read** |
| Original prefilter logic | `submissions/solver/core/candidate_scoring.py` | `_prefilter_score_set` [`245-297`](submissions/solver/core/candidate_scoring.py:245) | Neighborhood-only positive approx pruning + exploration | **Modify** |
| Original refinement generation | `submissions/solver/core/original_refinement.py` | `_make_single_candidate` [`43-98`](submissions/solver/core/original_refinement.py:43), `_combo_candidates` [`187-277`](submissions/solver/core/original_refinement.py:187) | Emits approx-bearing refinement candidates | **Read; avoid generation changes** |
| Original line search generation | `submissions/solver/core/original_line_search.py` | `generate_original_line_search_candidates` [`48-155`](submissions/solver/core/original_line_search.py:48) | Emits approx-bearing line-search candidates and local HPWL cutoff | **Read; avoid generation changes** |
| M4B reserved bucket | `submissions/solver/core/m4b_region_repair.py` | `generate_m4b_region_repair_candidates` [`226-390`](submissions/solver/core/m4b_region_repair.py:226) | Generates legalized M4B audit/scoring pool and post-legal approx | **Read; avoid** |
| M4C within-family ranking | `submissions/solver/core/m4c_ranking.py` | `compute_rank_scores` [`33-51`](submissions/solver/core/m4c_ranking.py:33), `assign_buckets` [`54-169`](submissions/solver/core/m4c_ranking.py:54) | Family-internal M4B reserved-bucket ordering only | **Avoid changing for M4D v1** |
| Artifact export | `submissions/solver/core/m3d_candidate_export.py` | `_REQUIRED_FIELDS` [`13-47`](submissions/solver/core/m3d_candidate_export.py:13), `export_candidate_rows` [`50-146`](submissions/solver/core/m3d_candidate_export.py:50) | Stable candidate CSV schema | **Modify to add M4D fields if needed** |
| Family summary export | `submissions/solver/core/m3d_family_summary.py` | `summarize_candidate_families` [`12-182`](submissions/solver/core/m3d_family_summary.py:12) | Family-level aggregation | **Read; maybe no change** |
| Profile wiring | `submissions/solver/scripts/run_benchmarks.py` | `_PROFILES` M4B/M4C entries [`377-462`](submissions/solver/scripts/run_benchmarks.py:377), config wiring [`905-946`](submissions/solver/scripts/run_benchmarks.py:905) | Adds new profile and new config fields | **Modify** |
| M4A diagnostics | `submissions/solver/m4a_loss_attribution.py` | rank-column support referenced by tests [`submissions/solver/tests/test_m4a_loss_attribution.py:377`](submissions/solver/tests/test_m4a_loss_attribution.py:377) | Already supports alternate rank column without schema break | **Read; likely avoid code change** |
| M4A existing report | `submissions/solver/reports/m4a_on_m4c/m4a_loss_attribution_report.md` | report body | Confirms Rule B still fires on ibm02/ibm03 | **Read only** |
| Tests: M3C budget allocation | `submissions/solver/tests/test_m3c_budget_allocation.py` | profile and frontier tests [`68-620`](submissions/solver/tests/test_m3c_budget_allocation.py:68) | Existing budget/frontier invariants | **Extend** |
| Tests: M4B region repair | `submissions/solver/tests/test_m4b_region_repair.py` | metadata/admission tests [`65-260`](submissions/solver/tests/test_m4b_region_repair.py:65) | Guard M4B generation behavior | **Avoid behavior changes** |
| Tests: M4C ranking | `submissions/solver/tests/test_m4c_ranking.py` | reserved-bucket tests [`72-275`](submissions/solver/tests/test_m4c_ranking.py:72) | Guard accepted M4C behavior | **Avoid changing expectations for `m4c-default`** |
| Tests: candidate export | `submissions/solver/tests/test_m3d_candidate_export.py` | schema/selectability tests [`107-585`](submissions/solver/tests/test_m3d_candidate_export.py:107) | Export compatibility | **Extend if new fields are emitted** |
| Tests: M4A attribution | `submissions/solver/tests/test_m4a_loss_attribution.py` | input-prefix and rank-column tests [`348-443`](submissions/solver/tests/test_m4a_loss_attribution.py:348) | Confirms M4A compatibility | **Add narrow tests only if needed** |

## 3. Current Budget / Ranking Mechanics

### How original/pre-M3 candidates are admitted

Passes 1-3 are the pre-M3 pool:
- pass 1: original + neighborhood prepare/dedup/prefilter/score [`909-955`](submissions/solver/core/candidate_scoring.py:909)
- pass 2: original refinement [`984-1127`](submissions/solver/core/candidate_scoring.py:984)
- pass 3: original line search [`1129-1190`](submissions/solver/core/candidate_scoring.py:1129)

When M3C is enabled, pre-M3 total is explicitly capped by `m3c_pre_m3_budget` and defaults to `max_official_scores - m3a - m3b` [`841-868`](submissions/solver/core/candidate_scoring.py:841). In `m4c-default`, that remains `50` out of the total `80` score cap? No: M3C still slices only the non-M4B pool, and M4B gets a separate reserved `20`. The runner profile confirms:
- `max_official_scores = 80`
- `m3c_pre_m3_budget = 50`
- `m3c_m3a_reserved_budget = 5`
- `m3c_m3b_reserved_budget = 5`
- `m4b_reserved_scores = 20`
at [`submissions/solver/scripts/run_benchmarks.py:416-462`](submissions/solver/scripts/run_benchmarks.py:416).

### How `original_neighborhood` `approx_delta` affects prefilter/admission

`original_neighborhood` computes `approx_hpwl_delta` by summing incident-net HPWL changes only for the moved macro's nets [`submissions/solver/core/original_neighborhood.py:113-125`](submissions/solver/core/original_neighborhood.py:113), then stores it on each candidate [`225-241`](submissions/solver/core/original_neighborhood.py:225).

Prefiltering is currently **family-specific**:
- only `original_neighborhood` with finite approx participates in positive-delta pruning [`275-277`](submissions/solver/core/candidate_scoring.py:275)
- `approx <= 0` is admitted automatically [`279-283`](submissions/solver/core/candidate_scoring.py:279)
- `approx > 0` is sorted ascending and only the first `exploratory_score_count` survive [`287-295`](submissions/solver/core/candidate_scoring.py:287)
- all other families bypass this pass-1 prefilter entirely [`276-278`](submissions/solver/core/candidate_scoring.py:276)

After prefiltering, pass-1 scoring order is:
1. originals
2. non-neighborhood / non-approx families
3. improving neighborhood candidates by most-negative approx
4. exploratory positive neighborhood candidates
per [`934-944`](submissions/solver/core/candidate_scoring.py:934).

### How `original_refinement` is ranked/admitted

Refinement candidates are generated from selected neighborhood seeds [`995-1011`](submissions/solver/core/candidate_scoring.py:995), with their own approx deltas computed in generation [`submissions/solver/core/original_refinement.py:80`](submissions/solver/core/original_refinement.py:80) and for combo moves [`255-271`](submissions/solver/core/original_refinement.py:255).

Admission is not global-family-aware today. Instead, pass 2 builds per-seed queues:
- tier 0: prelegal-valid improving approx
- tier 1: prelegal-bad, still admitted because legalizer may fix them
- tier 2: no valid approx
- tier 3: exploratory positive approx
from [`1040-1084`](submissions/solver/core/candidate_scoring.py:1040)

Then it round-robins across seed queues so one seed cannot consume the full pass-2 budget [`1086-1096`](submissions/solver/core/candidate_scoring.py:1086). Budget comes from the remaining pre-M3 slice, optionally capped by `refinement_score_budget` [`1098-1112`](submissions/solver/core/candidate_scoring.py:1098).

### How `original_line_search` is ranked/admitted

Line search candidates are generated from top-K **officially scored** neighborhood seeds sorted by `proxy_cost`, not approx [`750-768`](submissions/solver/core/candidate_scoring.py:750). Generation computes new approx deltas and applies only a local HPWL headroom cutoff on overlap-free candidates [`submissions/solver/core/original_line_search.py:90-124`](submissions/solver/core/original_line_search.py:90).

Scoring order is per macro:
- priority scales `[1.5, 2.0, 2.5, 3.0, 1.25, 4.0, 0.75, 0.5, 0.25]` from [`submissions/solver/core/original_line_search.py:37-41`](submissions/solver/core/original_line_search.py:37)
- early stop after `line_search_stop_after_worse` consecutive worse official scores [`submissions/solver/core/candidate_scoring.py:390-399`](submissions/solver/core/candidate_scoring.py:390)
- line search uses the remaining pre-M3 budget after passes 1 and 2 [`1166-1184`](submissions/solver/core/candidate_scoring.py:1166)

### How M3A / M3B are admitted

M3A:
- generate/validate/dedup [`1240-1279`](submissions/solver/core/candidate_scoring.py:1240)
- with M3C enabled, only the first `_m3c_m3a_alloc` valid non-duplicates are admitted to the frontier [`1308-1317`](submissions/solver/core/candidate_scoring.py:1308)
- frontier candidates are scored with `max_scores=_m3c_m3a_alloc` [`1318-1331`](submissions/solver/core/candidate_scoring.py:1318)
- outside-frontier candidates get `skip_reason="m3c_not_admitted"` and do **not** count as budget exhaustion [`1310-1317`](submissions/solver/core/candidate_scoring.py:1310)

M3B:
- same structure, plus optional M3A rollover when enabled [`1458-1511`](submissions/solver/core/candidate_scoring.py:1458)
- frontier = first `_m3b_budget` valid non-duplicates [`1488-1497`](submissions/solver/core/candidate_scoring.py:1488)

### How M4B candidates are admitted under M4C

M4B generation is legalization-aware and writes both pre- and post-legal approx deltas [`submissions/solver/core/m4b_region_repair.py:293-366`](submissions/solver/core/m4b_region_repair.py:293).

Under `m4c-default`:
- valid non-duplicate M4B rows are gathered [`1622-1625`](submissions/solver/core/candidate_scoring.py:1622)
- they are converted into a stripped metadata list and ranked only within M4B by `assign_buckets` [`1626-1652`](submissions/solver/core/candidate_scoring.py:1626)
- `assign_buckets` writes `m4c_rank_score`, `m4c_rank_bucket`, `m4c_rank_reason`, `family_rank`, `family_normalized_approx_delta` back to candidate metadata [`1653-1661`](submissions/solver/core/candidate_scoring.py:1653)
- scored frontier = `ranked` bucket then `exploration` bucket [`1663-1686`](submissions/solver/core/candidate_scoring.py:1663)
- outside frontier gets `skip_reason="m4c_budget_exhausted"` [`1692-1695`](submissions/solver/core/candidate_scoring.py:1692)
- the reserved scoring budget is still hard-capped by `m4b_reserved_scores` [`1704-1710`](submissions/solver/core/candidate_scoring.py:1704)

### Where score budgets are enforced

The actual hard score cap is always `_score_batch`:
- cache hits are free
- fresh scores stop at `max_scores`
- over-budget candidates get `skip_reason="budget_exceeded"`
from [`300-372`](submissions/solver/core/candidate_scoring.py:300)

Line search has its own ordered wrapper but still enforces the same max score budget per frontier [`444-468`](submissions/solver/core/candidate_scoring.py:444).

### Safe insertion points for M4D

Safest places:
1. Pass-1 prefilter/sort key construction [`927-946`](submissions/solver/core/candidate_scoring.py:927)
2. Pass-2 refinement queue tiering or frontier slicing before `_score_batch` [`1045-1112`](submissions/solver/core/candidate_scoring.py:1045)
3. Optional M4B frontier slicing in `candidate_scoring.py`, **without** changing `m4b_region_repair.py` or `m4c_ranking.py` [`1622-1709`](submissions/solver/core/candidate_scoring.py:1622)
4. Profile/config wiring in `run_benchmarks.py` [`905-946`](submissions/solver/scripts/run_benchmarks.py:905)

Unsafe for M4D v1:
- changing M4B generation/legalization
- changing M4C within-family rank math
- changing scorer/cache/legalizer behavior

## 4. Approx_delta Scale Analysis

Source: `analysis/m4c/m4c_candidate_effectiveness.csv`

### Summary tables

`median_evaluator_delta_vs_selected = median(proxy_cost - selected_cost)` over scored rows.

#### ibm01

| Family | approx non-null | scored | median approx | median abs approx | IQR | median evaluator cost | median evaluator delta vs selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m4b_region_repair` | 130 | 20 | 8.6015 | 10.5719 | 28.7082 | 1.0385288 | 0.0005618 |
| `original_line_search` | 20 | 6 | -4.1314 | 4.1314 | 5.2240 | 1.0890301 | 0.0510631 |
| `original_neighborhood` | 78 | 25 | -0.8335 | 2.7540 | 5.2024 | 1.1407721 | 0.1028051 |
| `original_refinement` | 150 | 18 | -1.0637 | 2.3870 | 6.4615 | 1.1304364 | 0.0924695 |
| `m3a_pair_refinement` | 0 | 5 | n/a | n/a | n/a | 1.0384849 | 0.0005180 |
| `m3b_cluster_refinement` | 0 | 1 | n/a | n/a | n/a | 1.0384210 | 0.0004541 |

Top-5 raw approx / evaluator-rank highlights:
- `m4b_region_repair`: best raw approx is `m4b_r2_m19_m243_centroid_shift` at `-43.13`, but evaluator rank `20/20`; the selected M4B winner `m4b_r7_m7_m43_centroid_shift` has approx rank `14/20`.
- `original_refinement`: the three most-negative combo2 candidates all land evaluator ranks `16-18/18`; the best evaluator refinement `original_refinement_m215_scale2x` has approx rank `11/18`.
- `original_line_search`: milder mismatch; `original_line_search_m215_scale2p5x` is evaluator rank `1/6` and approx rank `2/6`.

#### ibm02

| Family | approx non-null | scored | median approx | median abs approx | IQR | median evaluator cost | median evaluator delta vs selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m4b_region_repair` | 139 | 20 | 5.2767 | 6.7168 | 13.6920 | 1.5580953 | 0.0055948 |
| `original_line_search` | 30 | 6 | -6.7675 | 7.6296 | 16.7039 | 1.6242345 | 0.0717340 |
| `original_neighborhood` | 78 | 25 | -1.4636 | 4.8353 | 18.6459 | 1.6356628 | 0.0831623 |
| `original_refinement` | 149 | 18 | -1.3000 | 6.5200 | 18.3538 | 1.6313661 | 0.0788656 |
| `m3a_pair_refinement` | 0 | 5 | n/a | n/a | n/a | 1.5584178 | 0.0059173 |
| `m3b_cluster_refinement` | 0 | 5 | n/a | n/a | n/a | 1.5582905 | 0.0057900 |

Top-5 raw approx / evaluator-rank highlights:
- `m4b_region_repair`: `m4b_r1_m4_m51_centroid_shift` raw approx rank `1/20` is evaluator rank `20/20`; `m4b_r1_m43_m51_spread` wins evaluator rank `1/20` with approx rank `10/20`.
- `original_neighborhood`: `m51_toward_centroid` raw approx rank `1/25` is evaluator rank `23/25`; `m122_right_s` is evaluator rank `1/25` with approx rank `23/25`.
- `original_refinement`: the three worst-magnitude combo2 approximations are evaluator ranks `15-17/18`; the best evaluator refinement `m51_tiny0p5um_p0_p1` is approx rank `15/18`.

#### ibm03

| Family | approx non-null | scored | median approx | median abs approx | IQR | median evaluator cost | median evaluator delta vs selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m4b_region_repair` | 128 | 20 | 5.3764 | 9.0887 | 13.5982 | 1.3269207 | 0.0014769 |
| `original_line_search` | 20 | 4 | -60.7805 | 60.7805 | 116.5663 | 1.3400910 | 0.0146472 |
| `original_neighborhood` | 78 | 25 | -2.1932 | 23.0800 | 54.1646 | 1.3584473 | 0.0330034 |
| `original_refinement` | 148 | 18 | -5.7914 | 13.2244 | 43.4196 | 1.3559412 | 0.0304974 |
| `m3a_pair_refinement` | 0 | 5 | n/a | n/a | n/a | 1.3261321 | 0.0006882 |
| `m3b_cluster_refinement` | 0 | 0 | n/a | n/a | n/a | n/a | n/a |

Top-5 raw approx / evaluator-rank highlights:
- `m4b_region_repair`: the three most-negative M4B approximations are all evaluator ranks `18-20/20`; the selected winner `m4b_r1_m7_m43_centroid_shift` is evaluator rank `1/20` with approx rank `16/20` and a **positive** approx delta (`9.66`).
- `original_line_search`: strict monotonic inversion; approx rank and evaluator rank are fully reversed over the scored subset.
- `original_refinement`: `combo2_m72_m240` is raw approx rank `1/18`, evaluator rank `15/18`; the best tiny-step winner `m289_tiny0p25um_m1_p0` is approx rank `12/18`.

### HPWL-scale ratio: original_* vs M4B

Using median absolute approx delta:

| Benchmark | `original_neighborhood / m4b` | `original_refinement / m4b` | `original_line_search / m4b` | pooled `original_* / m4b` |
|---|---:|---:|---:|---:|
| `ibm01` | 0.261x | 0.226x | 0.391x | 0.261x |
| `ibm02` | 0.720x | 0.971x | 1.136x | 0.971x |
| `ibm03` | 2.539x | 1.455x | 6.687x | 2.256x |

Answer:
- There is **no stable global ratio**.
- The mismatch is **benchmark-specific first**, then **family-specific**.
- It is **not primarily move-type-specific** inside M4B: M4B centroid vs spread medians are close on each benchmark.

### Raw-vs-normalized evidence

Cross-family Spearman on scored approx-bearing rows (`original_* + m4b_region_repair`):

| Benchmark | raw approx vs evaluator | family-normalized approx vs evaluator |
|---|---:|---:|
| `ibm01` | -0.0203 | 0.1257 |
| `ibm02` | -0.5742 | -0.0572 |
| `ibm03` | -0.8574 | -0.7467 |

Answer:
- **Family-normalized approx looks directionally better** on `ibm01` and `ibm02`.
- It is **not sufficient by itself** on `ibm03`.
- That argues for: normalized score **plus quota floors**, not normalized score alone.

## 5. Available Features for Calibration

Persisted per-candidate fields already available in M4C artifacts:

| Field | Source artifact | Coverage | Safe for ranking? | Leakage risk |
|---|---|---|---|---|
| `approx_delta` | `m4c_candidate_effectiveness.csv` | all approx-bearing families; blank for M3A/M3B | **Yes** | None |
| `family` | same | all rows | **Yes** | None |
| `source_stage` | same | all rows with metadata | **Yes** | None |
| `moved_macro_id` / `moved_macro_ids` | runner JSON + name parsing + candidate metadata | mixed; strongest for original/M4B | **Yes** | None |
| `region_id` | M4B rows only | M4B only | **Yes** | None |
| `move_type` | M4B explicit; some original types name-derived | partial | **Yes** | None |
| `pre_legalization_approx_delta` | M4B export | M4B only | **Maybe diagnostics only** | None |
| `post_legalization_approx_delta` | M4B export | M4B valid/admitted rows | **Yes** | None |
| `legalization_displacement_max` | M4B export | M4B only | **Yes, as tie-break/cap** | None |
| `legalization_displacement_mean` | M4B export | M4B only | **Yes, as tie-break/cap** | None |
| `m4c_rank_score` | M4B export | M4B only | **Diagnostics only for M4D unless intentionally reused** | None |
| `family_rank` | M4B export | M4B only | **Diagnostics / tie-break** | None |
| `family_normalized_approx_delta` | M4B export | M4B only | **Yes for M4B; not cross-family by itself** | None |
| `placement_hash` | all rows after prep/legalization | many rows | **No for ranking; use for dedup only** | None |
| `duplicate` | exported row | all rows | **No; filter only** | None |
| `valid` | exported row | all rows | **No; filter only** | None |
| `admitted` / `not_admitted` | exported row | all rows | **No; diagnostics only** | circular if reused |
| `scored` | exported row | all rows | **No; diagnostics only** | circular if reused |
| `proxy_cost` / `evaluator_cost` | candidate CSV / M4A derived report | scored rows only | **Forbidden for ranking calibration** | **High leakage** |

Notes:
- `proxy_cost` in the candidate CSV is treated as evaluator cost by M4A; it is explicitly post-score information and must stay offline only.
- M3A/M3B currently have no approx coverage in these artifacts, so any M4D calibration using approx must either exclude them or give them fixed-family handling.

## 6. Candidate M4D Mechanisms

### 1. Family-normalized prefilter score

How it works:
- For approx-bearing families, compute a per-family normalized rank score from that family's own approx distribution on the current benchmark/run.
- Use this normalized score instead of raw approx when building pass-1 and pass-2 scoring order.

Files touched:
- `submissions/solver/core/candidate_scoring.py`
- optionally `submissions/solver/core/m3d_candidate_export.py` if new telemetry is emitted

Expected impact:
- Reduces family-to-family scale mismatch in pass ordering.
- Best chance to help `ibm02`.

No-regression strategy:
- Apply only to families already using approx-bearing prefilter order.
- Keep originals and exact budget caps unchanged.

Risks:
- Alone, it does not fix `ibm03`.
- Could surface more mediocre M4B candidates if normalization flattens useful magnitude differences.

Tests needed:
- deterministic normalized ordering
- no evaluator leakage
- unchanged behavior when feature disabled

Select for M4D:
- **Yes**

### 2. Per-family budget quotas / reserved slices

How it works:
- Inside a fixed budgeted frontier, reserve minimum slots for specific families.
- Natural targets: pass-1/2/3 pre-M3 approx families, and possibly a conservative floor inside the M4B reserved slice.

Files touched:
- `submissions/solver/core/candidate_scoring.py`
- `submissions/solver/scripts/run_benchmarks.py`
- new tests near `test_m3c_budget_allocation.py`

Expected impact:
- Prevents one pathological approx family from dominating scored rows.
- Helps M4A explainability even if ranking is still imperfect.

No-regression strategy:
- Use small floors, not aggressive caps.
- Preserve total budgets and fallback behavior.

Risks:
- Starvation if the floor is too large for weak families.
- Can hide truly strong concentrated winners.

Tests needed:
- frontier count accounting
- no budget expansion
- outside-frontier rows marked `not_admitted`, not `budget_exceeded`

Select for M4D:
- **Yes**

### 3. Hybrid normalized score + quota floor

How it works:
- Order by normalized family score.
- Then enforce small quota floors so each important family gets at least some scored presence.

Files touched:
- same as above, mostly `candidate_scoring.py`

Expected impact:
- Best smallest-v1 balance.
- Improves cross-family fairness without a large redesign.

No-regression strategy:
- Floors only; keep top global slots free above the floor.
- Preserve all accepted M4C M4B mechanisms.

Risks:
- More moving pieces in budget accounting.
- Needs careful telemetry so M4A/read-only analysis stays comparable.

Tests needed:
- score-order determinism
- floor satisfaction
- preserved winner selectability

Select for M4D:
- **Primary recommendation**

### 4. Learned calibration

How it works:
- Fit a model from artifact features to predict evaluator ordering or family adjustments.

Expected impact:
- Could overfit public IBM trio very easily.

Risks:
- High leakage risk, hard to justify from three benchmarks.
- Not enough stable training data or feature coverage.
- Violates the "smallest viable" requirement.

Select for M4D:
- **Reject for now**

## 7. Recommended Smallest Viable M4D

### Must-have

1. Add a **family-normalized approx score** for approx-bearing pre-M3 families inside `candidate_scoring.py`.
2. Use that score to build a **cross-family pre-M3 frontier/order** for pass 1 and pass 2.
3. Add **small quota floors** across `original_neighborhood`, `original_refinement`, and `original_line_search` in the pre-M3 budgeted pool.
4. Emit M4D telemetry fields into candidate export so M4A can inspect the new rank column offline.

### Should-have

1. Add a **conservative M4B quota floor/ceiling policy only if needed**, but do not rewrite `m4c_ranking.py`.
2. Add a per-family admission summary to runner JSON or candidate metadata for explainability.
3. Add one diagnostics-only alternate M4A run using the M4D rank column.

### Defer

1. Learned calibrators
2. Per-benchmark hand-tuned constants
3. Any changes to M4B generation, legalizer behavior, or within-family M4C ranking
4. Any stronger search/repair/diversity optimizer

## 8. Budget/Profile Strategy

Recommendations:
- **Add `m4d-default`: yes**
- **Freeze `m4c-default`: yes**
- **Keep total `max_official_scores = 80`: yes**
- **Prefer reallocation inside the existing 60/pre-M3 slice first**: yes
- **Add a family quota floor**: yes, but small and only for approx-bearing pre-M3 families in v1

On the `60 + 20` split:
- My recommendation is to **keep the external 60 + 20 shape unchanged in the first M4D**.
- Reason: all three public benchmarks select an M4B winner under M4C, so shrinking the M4B reserve is a direct regression risk.
- If later evidence shows M4B over-allocation is the main reason Rule B survives, revisit with a floor/ceiling inside the `20`, not by rewriting M4B ranking.

How to ensure `selected_cost(m4d) <= selected_cost(m4c)`:
- keep M4B 20-slot reserved path intact
- preserve existing winner reachability
- apply only admission/ranking changes before scoring, not selection semantics
- run side-by-side winner-preservation tests for known M4C winners

How to keep existing winning candidates selectable:
- do not remove M4B frontier scoring
- do not change `m4c_known_winners`
- do not change selection code at [`1795-1860`](submissions/solver/core/candidate_scoring.py:1795)

## 9. M4A / Artifact Compatibility

Answers:
- Should M4D write `analysis/m4d/m4d_*.csv`? **Yes**
- Does M4A need a patch beyond `--input-prefix` / `--rank-column`? **Probably no**
- What rank column should M4A use for diagnostics? **A new M4D-specific rank column, e.g. `m4d_rank_score`**
- What fields should M4D emit? At minimum:
  - `m4d_rank_score`
  - `m4d_rank_bucket` or `m4d_admission_bucket`
  - `m4d_family_quota_bucket` or `m4d_family_floor_applied`
  - `m4d_family_normalized_approx_delta`
  - `m4d_cross_family_rank`
- How should M4D compare to M4C in reports?
  - canonical M4A on raw `approx_delta`
  - alternate M4A on `m4d_rank_score`
  - same family summary schema
  - same benchmark summary schema

Why I think M4A code need not change:
- rank-column alternate diagnostics are already covered by tests [`submissions/solver/tests/test_m4a_loss_attribution.py:377-443`](submissions/solver/tests/test_m4a_loss_attribution.py:377)
- input-prefix support is already present for `m4b` and `m4c`; `m4d` should fit the same pattern

## 10. Acceptance Metrics

Recommended exact M4D acceptance criteria:

1. `selected_cost(m4d) <= selected_cost(m4c)` on `ibm01`, `ibm02`, and `ibm03`
2. canonical M4A Rule B (`prefilter_evaluator_disagreement`) fires on **at most 1** benchmark
3. known M4C public winners remain **admissible and scoreable**
4. no family is starved to zero scored rows unless it generated zero admissible candidates
5. scored family shares are explainable from M4D quota settings and reported in artifacts
6. no changes to scorer/cache/legalizer
7. M4A rerun succeeds on M4D artifacts with:
   - default `rank_column=approx_delta`
   - alternate `rank_column=m4d_rank_score`
8. raw `approx_delta` may remain recorded unchanged
9. `m4c-default`, `m4b-default`, `m3c-default` remain behavior-frozen

Practical success bar:
- if M4D preserves all three M4C public costs and clears Rule B on one of `ibm02`/`ibm03`, it is likely worth the next adversarial pass

## 11. Risks and Guardrails

| Risk | Why it matters | How to detect | Mitigation | Block implementation? |
|---|---|---|---|---|
| Overfitting to public benchmarks | only 3 public targets, noisy ratios | M4D logic depends on benchmark names or hard-coded winner names beyond existing M4C list | ban per-benchmark conditionals; test source scan similar to M4C tests | No, if guarded |
| Family quotas starving true winners | floors/caps can suppress concentrated strong families | compare selected/scored winners vs M4C | floors only first; avoid aggressive ceilings | No |
| Calibrated score games M4A | optimizing only M4A could hurt true cost | selected cost regresses or winner set worsens | acceptance requires cost non-regression first | Yes if cost regresses |
| Selected-cost regression | direct failure | benchmark rerun | keep M4B reserve intact; preserve selection semantics | Yes |
| Scorer/cache/legalizer touched | disallowed scope | diff review | restrict edits to ranking/allocation/export/profile/test files | Yes |
| M4A schema drift | breaks diagnostics loop | M4A tests fail | extend export schema compatibly; keep existing columns | Yes if broken |
| Hidden benchmark generalization | public-only logic may collapse elsewhere | replay on larger suite when possible | keep simple normalized ranks, no learning | No |
| Budget accounting complexity | easy to create silent score leakage | diagnostics mismatch, tests fail | extend M3C-style invariant tests | No |
| Cross-family calibration accidentally uses evaluator cost | leakage invalidates result | source/code scan; tests | dedicated no-leakage tests | Yes |

## 12. Files to Modify in Implementation

### Add

- `analysis/m4d/m4d_*.csv` and reports at run time
- tests for M4D budget/rank wiring, likely a new `submissions/solver/tests/test_m4d_*.py`

### Modify

- `submissions/solver/core/candidate_scoring.py`
- `submissions/solver/core/m3d_candidate_export.py`
- `submissions/solver/scripts/run_benchmarks.py`
- possibly `submissions/solver/tests/test_m3d_candidate_export.py`
- possibly `submissions/solver/tests/test_m4a_loss_attribution.py`
- new tests adjacent to `test_m3c_budget_allocation.py` and `test_m4c_ranking.py`

### Do not touch

- `submissions/solver/core/m4b_region_repair.py`
- `submissions/solver/core/m4c_ranking.py`
- scorer / evaluator / cache / legalizer code
- `m3c-default`, `m4b-default`, `m4c-default`
- `analysis/m3d`, `analysis/m4b`, `analysis/m4c` artifacts
- existing M4A/M4B/M4C reports
- benchmark `.pt` files

## 13. Recommended Implementation Order

1. Add `m4d-default` profile wiring, feature-flagged off existing profiles.
2. Implement cross-family normalized telemetry in `candidate_scoring.py` for approx-bearing pre-M3 families.
3. Apply quota-floor admission in pre-M3 frontier/order only.
4. Export M4D telemetry fields through `m3d_candidate_export.py`.
5. Add deterministic no-leakage and budget-invariant tests.
6. Run M4A on M4D artifacts twice: canonical raw approx, then `m4d_rank_score`.
7. Only if still needed, consider a narrow M4B floor/ceiling adjustment inside the reserved 20, while leaving `m4c_ranking.py` untouched.

## 14. Final Recommendation

- **Yes, Opus should review the M4D direction next.**
- **Yes, Sonnet should draft a reduced M4D spec after Opus.**
- **Codex should wait for that reduced spec before implementation**, because the evidence supports the direction, but the exact quota shape should be locked before code is touched.

Bottom line:
- M4D is worth pursuing.
- The smallest credible path is **family-normalized cross-family admission plus small quota floors**.
- M4D should behave like a thin allocation layer around accepted M4C behavior, not a new optimizer.
