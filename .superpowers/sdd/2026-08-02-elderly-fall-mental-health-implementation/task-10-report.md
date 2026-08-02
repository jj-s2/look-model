# Task 10 report: low-burden wellbeing screening

## Delivered

- Offline Stanford GDS-15 configuration with the fixed 15-item order, Chinese and English
  content-review fields, yes/no options, risk answers, source URLs, access date, and
  version metadata. The result remains explicitly non-diagnostic.
- Seven-day personal trend baseline for activity, sleep, and voluntary wellbeing answers.
  A single abnormal day creates observation evidence only. Sustained change requires two
  adjacent natural days; missing days reset the sequence, same-day updates do not increment
  it, and out-of-order updates are rejected.
- A self-harm expression creates a highest-priority human-review event independently of a
  GDS result.
- Interaction policy enforces 28-day full-screening and 7-day short-check-in cooldowns for
  both completed check-ins and prior invitations, quiet hours from 21:00 through 08:00,
  one immediate confirmed-fall check, and user-initiated cooldown bypasses.

## Verification

`python -m pytest tests/mental -q` completed successfully: 28 passed.

`python -c "from mental.gds15 import GDS15; s=GDS15.from_json('configs/screening/gds15_zh.json'); print(s.version, len(s.items))"`
prints `stanford-gds15-zh-2026-08-02 15`.

`git diff --check` completed without output.

## Concerns

- This feature is offline and does not connect to devices or record audio. It is a screening
  and interaction-policy component, not a clinical assessment or emergency-response system.
- A self-harm expression is routed to human review and must not be replaced or downgraded by
  a GDS score.
