# M3D Effectiveness Analysis

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Profile | `m4c-default` |
| Benchmarks | ibm01, ibm02, ibm03 |
| Official epsilon | `1e-05` |
| Max official scores | `80` |

## Benchmark Summary

| Benchmark | Selected | Family | Cost | Orig Cost | Classification |
|-----------|----------|--------|------|-----------|----------------|
| ibm01 | m4b_r7_m7_m43_centroid_shift | m4b_region_repair | 1.037967 | 1.038498 | ranking_mismatch |
| ibm02 | m4b_r1_m43_m51_spread | m4b_region_repair | 1.552500 | 1.565849 | ranking_mismatch |
| ibm03 | m4b_r1_m7_m43_centroid_shift | m4b_region_repair | 1.325444 | 1.325486 | ranking_mismatch |

## Family Effectiveness

| Benchmark | Family | Generated | Valid | Scored | Beating Final | Near Tie | Best Cost |
|-----------|--------|-----------|-------|--------|---------------|----------|-----------|
| ibm01 | m3a_pair_refinement | 384 | 40 | 5 | 0 | 0 | 1.038433 |
| ibm01 | m3b_cluster_refinement | 96 | 1 | 1 | 0 | 0 | 1.038421 |
| ibm01 | m4b_region_repair | 130 | 130 | 20 | 0 | 2 | 1.037967 |
| ibm01 | original | 2 | 2 | 1 | 0 | 0 | 1.038498 |
| ibm01 | original_line_search | 20 | 20 | 6 | 0 | 0 | 1.038431 |
| ibm01 | original_neighborhood | 78 | 78 | 25 | 0 | 0 | 1.038459 |
| ibm01 | original_refinement | 150 | 150 | 18 | 0 | 0 | 1.038435 |
| ibm02 | m3a_pair_refinement | 384 | 37 | 5 | 0 | 0 | 1.558135 |
| ibm02 | m3b_cluster_refinement | 96 | 11 | 5 | 0 | 0 | 1.557626 |
| ibm02 | m4b_region_repair | 144 | 139 | 20 | 0 | 1 | 1.552500 |
| ibm02 | original | 2 | 2 | 1 | 0 | 0 | 1.565849 |
| ibm02 | original_line_search | 30 | 30 | 6 | 0 | 0 | 1.599192 |
| ibm02 | original_neighborhood | 78 | 78 | 25 | 0 | 0 | 1.607726 |
| ibm02 | original_refinement | 149 | 149 | 18 | 0 | 0 | 1.558418 |
| ibm03 | m3a_pair_refinement | 384 | 51 | 5 | 0 | 0 | 1.325456 |
| ibm03 | m3b_cluster_refinement | 3 | 0 | 0 | 0 | 0 | N/A |
| ibm03 | m4b_region_repair | 128 | 128 | 20 | 0 | 1 | 1.325444 |
| ibm03 | original | 2 | 2 | 1 | 0 | 0 | 1.325486 |
| ibm03 | original_line_search | 20 | 20 | 4 | 0 | 0 | 1.334186 |
| ibm03 | original_neighborhood | 78 | 78 | 25 | 0 | 0 | 1.328779 |
| ibm03 | original_refinement | 148 | 148 | 18 | 0 | 0 | 1.325457 |

## Failure Classification

| Benchmark | Classification | LS Generated | LS Scored | Beating Final | Next Step |
|-----------|----------------|--------------|----------|---------------|-----------|
| ibm01 | ranking_mismatch | 610 | 26 | 0 | redesign analytical prefilter/ranking |
| ibm02 | ranking_mismatch | 624 | 30 | 0 | redesign analytical prefilter/ranking |
| ibm03 | ranking_mismatch | 515 | 25 | 0 | redesign analytical prefilter/ranking |

## Top Late-Stage Candidates

| Benchmark | Candidate | Family | Cost | Selectable | Selected |
|-----------|-----------|--------|------|------------|----------|
| ibm01 | m4b_r7_m7_m43_centroid_shift | m4b_region_repair | 1.037967 | True | True |
| ibm01 | m4b_r7_m7_m35_centroid_shift | m4b_region_repair | 1.037969 | True | False |
| ibm01 | m4b_r7_m7_m55_centroid_shift | m4b_region_repair | 1.038102 | True | False |
| ibm01 | m4b_r8_m1_m31_spread | m4b_region_repair | 1.038141 | True | False |
| ibm01 | m4b_r7_m7_m21_spread | m4b_region_repair | 1.038344 | True | False |
| ibm01 | m4b_r7_m7_m35_spread | m4b_region_repair | 1.038362 | True | False |
| ibm01 | m4b_r8_m1_m8_spread | m4b_region_repair | 1.038372 | True | False |
| ibm01 | m4b_r8_m1_m49_spread | m4b_region_repair | 1.038383 | True | False |
| ibm01 | m4b_r8_m1_m53_spread | m4b_region_repair | 1.038389 | True | False |
| ibm01 | m3b_c30_0_52_166_centroid_shift | m3b_cluster_refinement | 1.038421 | True | False |

## Recommendations

- **redesign analytical prefilter/ranking** (ibm01, ibm02, ibm03)
