# M4A Loss Attribution Report

Profile: `m3c-default`
Official epsilon: `1e-05`

## Inputs

- `benchmark_summary`: `analysis\m3d\m3d_benchmark_summary.csv`
- `family_summary`: `analysis\m3d\m3d_family_summary.csv`
- `candidate_effectiveness`: `analysis\m3d\m3d_candidate_effectiveness.csv`
- `runner_json`: `submissions\solver\artifacts\run_m3c-default.json`

## Supported Diagnostics

- score_banding
- family_effectiveness
- skip_reason_aggregation
- budget_use
- approx_prefilter_vs_evaluator
- candidate_name_set_diversity

## Unsupported Diagnostics

- region_attribution
- density_hotspots
- top_macro_contributor
- top_net_contributor
- topology_lock_proof
- legalization_displacement

## Benchmark Classifications

| Benchmark | Score band | Delta | Classification | M4B recommendation |
|---|---|---:|---|---|
| ibm01 | epsilon_win | 7.66516e-05 | legality_bottleneck | legalization_aware_regional_repair |
| ibm02 | meaningful_win | 0.0082221 | prefilter_evaluator_disagreement | correlation_aware_ranking_or_prefilter_repair |
| ibm03 | epsilon_win | 3.02792e-05 | legality_bottleneck | legalization_aware_regional_repair |

## Per-Benchmark Diagnostics

### ibm01

- original_cost: `1.0385`
- selected_cost: `1.03842`
- delta: `7.66516e-05`
- relative_delta: `7.38101e-05`
- score_band: `epsilon_win`
- official_scored_count: `56`
- max_official_scores: `60`
- budget_saturation: `0.933333`
- duplicate_skipped_count: `28`
- prefiltered_count: `26`
- approx-prefilter vs evaluator usable_count: `49`
- approx-prefilter vs evaluator approx_coverage: `0.875`
- approx-prefilter vs evaluator spearman_rs: `-0.180376`
- approx-prefilter vs evaluator top5_inversions: `2`
- unique_macros_in_scored: `21`
- unique_macro_ratio: `0.375`
- placement_hash_collisions: `0`
- collision_ratio: `0`
- classification: `legality_bottleneck`
- M4B recommendation: `legalization_aware_regional_repair`

Classification reasons:
- valid_rate_local=0.0854167 < 0.20
- valid_rate_baseline=1 > 0.80
- delta=7.66516e-05 <= 10*epsilon=0.0001

### ibm02

- original_cost: `1.56585`
- selected_cost: `1.55763`
- delta: `0.0082221`
- relative_delta: `0.00525089`
- score_band: `meaningful_win`
- official_scored_count: `60`
- max_official_scores: `60`
- budget_saturation: `1`
- duplicate_skipped_count: `16`
- prefiltered_count: `23`
- approx-prefilter vs evaluator usable_count: `49`
- approx-prefilter vs evaluator approx_coverage: `0.816667`
- approx-prefilter vs evaluator spearman_rs: `-0.242347`
- approx-prefilter vs evaluator top5_inversions: `4`
- unique_macros_in_scored: `14`
- unique_macro_ratio: `0.233333`
- placement_hash_collisions: `0`
- collision_ratio: `0`
- classification: `prefilter_evaluator_disagreement`
- M4B recommendation: `correlation_aware_ranking_or_prefilter_repair`

Classification reasons:
- usable_count=49 >= 20
- approx_coverage=0.816667 >= 0.50
- spearman_rs=-0.242347 < 0.30
- top5_inversions=4 >= 3

### ibm03

- original_cost: `1.32549`
- selected_cost: `1.32546`
- delta: `3.02792e-05`
- relative_delta: `2.28438e-05`
- score_band: `epsilon_win`
- official_scored_count: `53`
- max_official_scores: `60`
- budget_saturation: `0.883333`
- duplicate_skipped_count: `21`
- prefiltered_count: `22`
- approx-prefilter vs evaluator usable_count: `47`
- approx-prefilter vs evaluator approx_coverage: `0.886792`
- approx-prefilter vs evaluator spearman_rs: `-0.737743`
- approx-prefilter vs evaluator top5_inversions: `5`
- unique_macros_in_scored: `15`
- unique_macro_ratio: `0.283019`
- placement_hash_collisions: `0`
- collision_ratio: `0`
- classification: `legality_bottleneck`
- M4B recommendation: `legalization_aware_regional_repair`

Classification reasons:
- valid_rate_local=0.131783 < 0.20
- valid_rate_baseline=1 > 0.80
- delta=3.02792e-05 <= 10*epsilon=0.0001

## Aggregate Recommendation

- outcome: `majority_with_scoped_dissenter`
- M4B: `legalization_aware_regional_repair`
- note: Majority classification is legality_bottleneck; scoped dissenters: ibm02.

## Caveats

- Cache decomposition is omitted if cache_hits/cache_misses are zero or inactive.
- Candidate name-set diversity is weak evidence and is not a substitute for geometry diversity.
- Geometry not persisted; region, density, per-macro displacement, and per-net HPWL attribution are unsupported by reduced M4A.
- M4A does not inherit M3D's near_local_optimum label; M4A classifications are derived independently from rules A-E.
- approx_delta is absent for m3a/m3b family candidates in current M3D artifacts; prefilter-evaluator disagreement analysis is suppressed when approx_coverage < 0.50.
- best_official_delta_vs_final is relative to the selected final cost, not to original_cost.
- ibm03: secondary signal - Rule B conditions also satisfied (spearman_rs=-0.737743, top5_inversions=5); prefilter disagreement is suppressed by Rule A priority.
- proxy_cost in M3D artifacts is treated as evaluator cost in M4A, not as a separate proxy.
