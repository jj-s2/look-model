# Competition demo script

## Before presenting

1. Start with the device status panel visible. State the actual state aloud:
   this delivery has **no connected C6c/SDNL1 verification**. Every simulated
   source carries the on-screen `DEMO / OFFLINE REPLAY` watermark.
2. Use an approved public replay or a safeguarded healthy-adult simulation;
   never arrange a real fall by an older adult.
3. Confirm that device IDs, access tokens, device verification codes, and full
   serial numbers are absent from the screen and exported report.

## Seven-minute walkthrough

1. **Normal activity (1 min).** Show a walking/sitting/lying replay. Explain
   that no fall alert is created for ordinary movements.
2. **Pre-fall change (1 min).** Show the separate, explainable gait-risk card.
   It is a risk trend, not an event or a medical conclusion.
3. **Fall event (2 min).** Run a clearly labelled controlled fall replay. Show
   evidence, risk level, and the `vision_only` quality badge when radar/SDNL1
   is unavailable. The system sends a configured-contact alert; it does not
   contact emergency services automatically.
4. **One follow-up only (1 min).** After the event, present one “是否需要帮助？”
   prompt. Demonstrate that a refusal/no response ends the conversation instead
   of repeating questions. Do not infer mental state from the answer.
5. **Device degradation (1 min).** Show the offline capability report. State
   that `unavailable` means unknown/unvalidated—not a zero value and not demo
   data. Vision-only monitoring remains actionable but is labelled degraded.
6. **Wellbeing boundary (1 min).** Show a screening/change recommendation such
   as “建议由家属或专业人员进一步了解”. Explicitly state: the module supports
   screening, observed change, and human attention; it does not diagnose
   depression, anxiety, or any other condition.

## Closing checks

- Open `docs/evaluation-report.md` and read the gate state exactly as shown.
  Do not claim a full release pass if latency, false-alarm, or live-device
  evidence is unavailable.
- Explain the opt-in event clip window (10 seconds before, 20 seconds after)
  and seven-day retention. With replay disabled, no event clip is written.
- Keep the `DEMO / OFFLINE REPLAY` watermark on every replay frame throughout
  the presentation.
