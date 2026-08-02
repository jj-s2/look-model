# Task 10 report: low-burden wellbeing screening

## Delivered

- Offline GDS-15 configuration with exactly 15 stable item IDs, Chinese yes/no prompts,
  risk answers, Stanford source URLs, access date, and version metadata.
- Non-diagnostic GDS-15 scoring output: score, screening risk band, completion time,
  version, and a fixed screening-not-diagnosis disclaimer.
- Seven-day personal trend baseline for activity, sleep, and voluntary wellbeing answers.
  One-day deviations remain observation evidence; sustained deviations may invite a short
  check-in.
- A self-harm expression creates a highest-priority, human-review event independently of
  any GDS score.
- Interaction policy enforces 28-day full-screening and 7-day short-check-in cooldowns,
  21:00–08:00 quiet hours, one immediate confirmed-fall check, and user-initiated cooldown
  bypasses.

## Verification

`python -m pytest tests/mental -q` completed successfully: 12 passed.

`python -c "from mental.gds15 import GDS15; s=GDS15.from_json('configs/screening/gds15_zh.json'); print(s.version, len(s.items))"`
printed `stanford-gds15-zh-2026-08-02 15`.

## Concerns

- This feature does not connect to devices and does not capture audio. It is screening and
  interaction-policy logic only; clinical assessment and emergency response remain human
  responsibilities.
- The Stanford Chinese source transcription is versioned in the configuration so future
  clinical/content review can update it without changing scoring code.
