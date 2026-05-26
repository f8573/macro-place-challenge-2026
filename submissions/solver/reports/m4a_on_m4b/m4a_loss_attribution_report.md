# M4A Loss Attribution Report

Profile: `m4b-default`
Official epsilon: `1e-05`

## Inputs

- `benchmark_summary`: `analysis\m4b\m4b_benchmark_summary.csv`
- `family_summary`: `analysis\m4b\m4b_family_summary.csv`
- `candidate_effectiveness`: `analysis\m4b\m4b_candidate_effectiveness.csv`
- `runner_json`: `submissions\solver\artifacts\run_m4b-default.json`

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
| ibm01 | epsilon_win | 7.66516e-05 | local_exhaustion_under_sampled_families | regional_destroy_and_repair |
| ibm02 | meaningful_win | 0.0109816 | prefilter_evaluator_disagreement | correlation_aware_ranking_or_prefilter_repair |
| ibm03 | epsilon_win | 4.20809e-05 | prefilter_evaluator_disagreement | correlation_aware_ranking_or_prefilter_repair |

## Per-Benchmark Diagnostics

### ibm01

- original_cost: `1.0385`
- selected_cost: `1.03842`
- delta: `7.66516e-05`
- relative_delta: `7.38101e-05`
- score_band: `epsilon_win`
- official_scored_count: `76`
- max_official_scores: `80`
- budget_saturation: `0.95`
- duplicate_skipped_count: `28`
- prefiltered_count: `26`
- approx-prefilter vs evaluator usable_count: `69`
- approx-prefilter vs evaluator approx_coverage: `0.907895`
- approx-prefilter vs evaluator spearman_rs: `-0.398352`
- approx-prefilter vs evaluator top5_inversions: `0`
- unique_macros_in_scored: `32`
- unique_macro_ratio: `0.421053`
- placement_hash_collisions: `0`
- collision_ratio: `0`
- classification: `local_exhaustion_under_sampled_families`
- M4B recommendation: `regional_destroy_and_repair`

Classification reasons:
- delta=7.66516e-05 <= 10*epsilon=0.0001
- budget_saturation=0.95 >= 0.80
- Rules A/B/C did not fire

### ibm02

- original_cost: `1.56585`
- selected_cost: `1.55487`
- delta: `0.0109816`
- relative_delta: `0.00701317`
- score_band: `meaningful_win`
- official_scored_count: `80`
- max_official_scores: `80`
- budget_saturation: `1`
- duplicate_skipped_count: `16`
- prefiltered_count: `23`
- approx-prefilter vs evaluator usable_count: `69`
- approx-prefilter vs evaluator approx_coverage: `0.8625`
- approx-prefilter vs evaluator spearman_rs: `-0.658629`
- approx-prefilter vs evaluator top5_inversions: `5`
- unique_macros_in_scored: `25`
- unique_macro_ratio: `0.3125`
- placement_hash_collisions: `0`
- collision_ratio: `0`
- classification: `prefilter_evaluator_disagreement`
- M4B recommendation: `correlation_aware_ranking_or_prefilter_repair`

Classification reasons:
- usable_count=69 >= 20
- approx_coverage=0.8625 >= 0.50
- spearman_rs=-0.658629 < 0.30
- top5_inversions=5 >= 3

### ibm03

- original_cost: `1.32549`
- selected_cost: `1.32544`
- delta: `4.20809e-05`
- relative_delta: `3.17475e-05`
- score_band: `epsilon_win`
- official_scored_count: `73`
- max_official_scores: `80`
- budget_saturation: `0.9125`
- duplicate_skipped_count: `21`
- prefiltered_count: `22`
- approx-prefilter vs evaluator usable_count: `67`
- approx-prefilter vs evaluator approx_coverage: `0.917808`
- approx-prefilter vs evaluator spearman_rs: `-0.826391`
- approx-prefilter vs evaluator top5_inversions: `5`
- unique_macros_in_scored: `26`
- unique_macro_ratio: `0.356164`
- placement_hash_collisions: `0`
- collision_ratio: `0`
- classification: `prefilter_evaluator_disagreement`
- M4B recommendation: `correlation_aware_ranking_or_prefilter_repair`

Classification reasons:
- usable_count=67 >= 20
- approx_coverage=0.917808 >= 0.50
- spearman_rs=-0.826391 < 0.30
- top5_inversions=5 >= 3

## Aggregate Recommendation

- outcome: `majority_with_scoped_dissenter`
- M4B: `correlation_aware_ranking_or_prefilter_repair`
- note: Majority classification is prefilter_evaluator_disagreement; scoped dissenters: ibm01.

## Caveats

- Cache decomposition is omitted if cache_hits/cache_misses are zero or inactive.
- Candidate name-set diversity is weak evidence and is not a substitute for geometry diversity.
- Geometry not persisted; region, density, per-macro displacement, and per-net HPWL attribution are unsupported by reduced M4A.
- M4A does not inherit M3D's near_local_optimum label; M4A classifications are derived independently from rules A-E.
- approx_delta is absent for m3a/m3b family candidates in current M3D artifacts; prefilter-evaluator disagreement analysis is suppressed when approx_coverage < 0.50.
- best_official_delta_vs_final is relative to the selected final cost, not to original_cost.
- proxy_cost in M3D artifacts is treated as evaluator cost in M4A, not as a separate proxy.
