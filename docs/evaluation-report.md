# Evaluation report

## Release gate: FAIL

This is an evidence report for a research and competition prototype. It does not
establish clinical performance and the wellbeing module is a screening/change
prompt only; it never produces a diagnosis.

### Global thresholds

| Metric | Observed | Threshold |
| --- | ---: | ---: |
| Fall F1 | 0.9078014184397163 | ≥ 0.90 |
| Fall recall | 0.9142857142857143 | ≥ 0.88 |
| End-to-end P95 latency (s) | unavailable | ≤ 2.0 |
| False alarms/hour | unavailable | ≤ 1.0 |

### Gate findings

- P95 latency evidence is missing
- false alarms/hour evidence is missing

### Classification evidence

| Metric | Observed |
| --- | ---: |
| Fall precision | 0.9014084507042254 |
| Fall F1 | 0.9078014184397163 |
| Fall recall | 0.9142857142857143 |
| Confusion matrix (TN, FP, FN, TP) | (74, 7, 6, 64) |
| Dataset | unavailable |
| Split strategy | leave-one-subject-out (reported source) |
| Decision threshold | unavailable |
| Random seed | unavailable |

### Performance evidence

| Metric | Observed |
| --- | ---: |
| P50 latency (s) | unavailable |
| P95 latency (s) | unavailable |
| Measurement scope | capture_decode_only |
| Hardware | {'platform': 'Windows-11-10.0.26200-SP0', 'machine': 'AMD64', 'processor': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel'} |
| Python | 3.12.10 |
| CUDA | unavailable |
| Model versions | {'opencv-python': '5.0.0.93', 'torch': 'unavailable', 'mmpose': 'unavailable', 'mmaction2': 'unavailable'} |

### Evidence files

- `experiments/outputs/losocv/summary.json`
- `outputs/benchmark.json`

## Interpretation limits

- Passing requires subject-level evaluation and a full end-to-end measurement;
  a capture/decode-only replay benchmark is explicitly not release eligible.
- No connected C6c or SDNL1 result is claimed in this report unless a separately
  recorded live-device evidence file says so. Unavailable capability remains
  unavailable, never demo data.
- A release-gate failure is intentional and must block performance claims until
  the missing or failing evidence is rerun.
