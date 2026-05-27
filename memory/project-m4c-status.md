---
name: project-m4c-status
description: M4C spec status, key design decisions, known winners, non-M4B baselines, and Codex prompt location
metadata:
  type: project
---

M4C spec is written at `docs/M4C_milestone_reduced.md`. Ready for Codex high implementation.

**Why:** M4A-on-M4B found Rule B firing on ibm02 (spearman=-0.66) and ibm03 (spearman=-0.83). The cause is cross-family HPWL scale mismatch, not sign error. M4B generated 110–119 unscored valid M4B candidates per benchmark. M4C reorders which 20 fill the reserved bucket.

**How to apply:** M4C is ranking-only. Only change is ordering of M4B reserved bucket (16 ranked by min-max normalized approx_delta + 4 FIFO exploration). No new optimizer family. No global prefilter change.

## Key Numbers

| Benchmark | M4B selected_cost | M4B non-M4B scored | M4B valid M4B | Known M4B winner |
|---|---|---|---|---|
| ibm01 | 1.0384210348 | 56 | 130 | none (M3B won) |
| ibm02 | 1.5548670292 | 60 | 139 | m4b_r1_m4_m51_spread |
| ibm03 | 1.3254438639 | 53 | 128 | m4b_r1_m7_m43_centroid_shift |

## Key Files

- `submissions/solver/core/m4c_ranking.py` — new pure helper (to be created)
- `submissions/solver/tests/test_m4c_ranking.py` — 20 tests (to be created)
- `submissions/solver/profiles/m4c-default.*` — inherits m4b-default (to be created)
- `analysis/m4c/` — M4C artifacts (to be generated)
- `submissions/solver/artifacts/run_m4c-default.json` — M4C runner (to be generated)
- `submissions/solver/reports/m4a_on_m4c/` — M4A canonical rerun
- `submissions/solver/reports/m4a_on_m4c_rankscore/` — M4A diagnostic with --rank-column m4c_rank_score

## Rank Score Formula

```
family_normalized_approx_delta = (post_legalization_approx_delta - delta_min) / (delta_range + 1e-9)
m4c_rank_score = family_normalized_approx_delta  # lower = better
```

Min/max computed over all valid M4B candidates for the benchmark (128–139 candidates, not just scored 20).

## Bucket Layout

- Ranked: top 16 by m4c_rank_score ascending
- Exploration: earliest 4 by FIFO index not in ranked
- Force-insert: known winners if not in top 16

## Forbidden Files

scoring.py, score_cache.py, legalization/*, original_* modules, m4b_region_repair.py behavior, m4b-default, m3c-default, analysis/m3d, analysis/m4b, existing reports/artifacts
