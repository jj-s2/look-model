"""Tests for recall-constrained UR Fall threshold selection."""
from __future__ import annotations
import json
from pathlib import Path
from risk.urfall_threshold_selection import select_recall_constrained_threshold

def test_selects_highest_threshold_meeting_fall_recall_without_using_adl_to_choose(tmp_path: Path) -> None:
    fall=[{"label":1,"probability":.9},{"label":1,"probability":.3},{"label":0,"probability":.4},{"label":0,"probability":.1}]
    adl=[{"label":0,"probability":.8},{"label":0,"probability":.2}]
    fp,ap=tmp_path/"fall.jsonl",tmp_path/"adl.jsonl"
    fp.write_text("".join(json.dumps(x)+"\n" for x in fall)); ap.write_text("".join(json.dumps(x)+"\n" for x in adl))
    r=select_recall_constrained_threshold(fp,ap,recall_floor=.8)
    assert r["threshold"] == .3
    assert r["fall_recall"] == 1.0
    assert r["adl_false_positive_rate"] == .5
