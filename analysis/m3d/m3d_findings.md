# M3D Effectiveness Analysis

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Profile | `m3c-default` |
| Benchmarks | ibm01, ibm02, ibm03 |
| Official epsilon | `1e-05` |
| Max official scores | `60` |

## Benchmark Summary

| Benchmark | Selected | Family | Cost | Orig Cost | Classification |
|-----------|----------|--------|------|-----------|----------------|
| ibm01 | m3b_c30_0_52_166_centroid_shift | m3b_cluster_refinement | 1.038421 | 1.038498 | near_local_optimum |
| ibm02 | m3b_c18_51_120_256_centroid_shift | m3b_cluster_refinement | 1.557626 | 1.565849 | near_local_optimum |
| ibm03 | m3a_p16_5_154_swap | m3a_pair_refinement | 1.325456 | 1.325486 | near_local_optimum |

## Family Effectiveness

| Benchmark | Family | Generated | Valid | Scored | Beating Final | Near Tie | Best Cost |
|-----------|--------|-----------|-------|--------|---------------|----------|-----------|
| ibm01 | m3a_pair_refinement | 384 | 40 | 5 | 0 | 0 | 1.038433 |
| ibm01 | m3b_cluster_refinement | 96 | 1 | 1 | 0 | 1 | 1.038421 |
| ibm01 | original | 2 | 2 | 1 | 0 | 0 | 1.038498 |
| ibm01 | original_line_search | 20 | 20 | 6 | 0 | 1 | 1.038431 |
| ibm01 | original_neighborhood | 78 | 78 | 25 | 0 | 0 | 1.038459 |
| ibm01 | original_refinement | 150 | 150 | 18 | 0 | 0 | 1.038435 |
| ibm02 | m3a_pair_refinement | 384 | 37 | 5 | 0 | 0 | 1.558135 |
| ibm02 | m3b_cluster_refinement | 96 | 11 | 5 | 0 | 1 | 1.557626 |
| ibm02 | original | 2 | 2 | 1 | 0 | 0 | 1.565849 |
| ibm02 | original_line_search | 30 | 30 | 6 | 0 | 0 | 1.599192 |
| ibm02 | original_neighborhood | 78 | 78 | 25 | 0 | 0 | 1.607726 |
| ibm02 | original_refinement | 149 | 149 | 18 | 0 | 0 | 1.558418 |
| ibm03 | m3a_pair_refinement | 384 | 51 | 5 | 0 | 1 | 1.325456 |
| ibm03 | m3b_cluster_refinement | 3 | 0 | 0 | 0 | 0 | N/A |
| ibm03 | original | 2 | 2 | 1 | 0 | 0 | 1.325486 |
| ibm03 | original_line_search | 20 | 20 | 4 | 0 | 0 | 1.334186 |
| ibm03 | original_neighborhood | 78 | 78 | 25 | 0 | 0 | 1.328779 |
| ibm03 | original_refinement | 148 | 148 | 18 | 0 | 1 | 1.325457 |

## Failure Classification

| Benchmark | Classification | LS Generated | LS Scored | Beating Final | Next Step |
|-----------|----------------|--------------|----------|---------------|-----------|
| ibm01 | near_local_optimum | 480 | 6 | 0 | try larger structural search |
| ibm02 | near_local_optimum | 480 | 10 | 0 | try larger structural search |
| ibm03 | near_local_optimum | 387 | 5 | 0 | try larger structural search |

## Top Late-Stage Candidates

| Benchmark | Candidate | Family | Cost | Selectable | Selected |
|-----------|-----------|--------|------|------------|----------|
| ibm01 | m3b_c30_0_52_166_centroid_shift | m3b_cluster_refinement | 1.038421 | True | True |
| ibm01 | m3a_p0_0_166_centroid_shift | m3a_pair_refinement | 1.038433 | True | False |
| ibm01 | m3a_p0_0_166_swap | m3a_pair_refinement | 1.038465 | True | False |
| ibm01 | m3a_p0_0_166_below | m3a_pair_refinement | 1.038485 | True | False |
| ibm01 | m3a_p6_10_63_swap | m3a_pair_refinement | 1.038789 | True | False |
| ibm01 | m3a_p5_9_42_right | m3a_pair_refinement | 1.041245 | True | False |
| ibm03 | m3a_p16_5_154_swap | m3a_pair_refinement | 1.325456 | True | True |
| ibm03 | m3a_p13_3_165_below | m3a_pair_refinement | 1.325471 | True | False |
| ibm03 | m3a_p16_5_154_below | m3a_pair_refinement | 1.326132 | True | False |
| ibm03 | m3a_p4_86_224_above | m3a_pair_refinement | 1.328703 | True | False |

## Recommendations

- **try larger structural search** (ibm01, ibm02, ibm03)
