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
| baseline | 24h | 1.0 | 1.0 | 0.20757884678200691 |
| baseline | 72h | 0.75 | 0.8333333333333333 | 0.38958459515570937 |
| baseline | 7d | 0.75 | 0.8333333333333333 | 0.3594967966166859 |
| full | 24h | 1.0 | 1.0 | 0.1977375576778162 |
| full | 72h | 0.75 | 0.8333333333333333 | 0.37112887584775084 |
| full | 7d | 0.75 | 0.8333333333333333 | 0.3484021042983468 |

## Evidence boundary

Synthetic and offline artifacts remain `research_only` and must not be used as release evidence. Missing subgroup cells are reported as unavailable, never as zero.
