# U-PMCC evaluation audit

Claim boundary: `research_only` — not release eligible.

## Provenance

- Release ID: `pmcc-release-5db68f3206af3445`
- Evidence tier: `synthetic_research`
- Method: subject-grouped outer splits; calibration within training folds only.
- Temporal chains are associations, not medical causality; this is not a clinical-performance claim.

## Horizon metrics

| Model | Horizon | AUROC | AUPRC | Brier |
| --- | --- | ---: | ---: | ---: |
| baseline | 24h | unavailable | unavailable | unavailable |
| baseline | 72h | unavailable | unavailable | unavailable |
| baseline | 7d | unavailable | unavailable | unavailable |
| full | 24h | unavailable | unavailable | unavailable |
| full | 72h | unavailable | unavailable | unavailable |
| full | 7d | unavailable | unavailable | unavailable |

## Evidence boundary

Synthetic and offline artifacts remain `research_only` and must not be used as release evidence. Missing subgroup cells are reported as unavailable, never as zero.
