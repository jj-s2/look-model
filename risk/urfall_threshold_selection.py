"""Select a fall-only recall-constrained threshold and audit it on held-out ADL."""
from __future__ import annotations
import json
from math import isfinite
from pathlib import Path
from typing import Any

def select_recall_constrained_threshold(fall_predictions_path: Path, adl_predictions_path: Path, *, recall_floor: float = .8) -> dict[str, float | str]:
    """Choose maximum threshold meeting fall recall; ADL is evaluated only afterwards."""
    if not isinstance(recall_floor, (int,float)) or isinstance(recall_floor,bool) or not 0 < recall_floor <= 1:
        raise ValueError("recall_floor must be in (0, 1]")
    fall=_read(Path(fall_predictions_path), expected_labels={0,1})
    adl=_read(Path(adl_predictions_path), expected_labels={0})
    if {int(row["label"]) for row in fall}!={0,1}: raise ValueError("fall predictions require both labels")
    candidates=sorted({float(row["probability"]) for row in fall}, reverse=True)
    valid=[]
    for threshold in candidates:
        tp=sum(row["label"]==1 and row["probability"]>=threshold for row in fall)
        fn=sum(row["label"]==1 and row["probability"]<threshold for row in fall)
        recall=tp/(tp+fn)
        if recall >= recall_floor: valid.append((threshold,recall))
    if not valid: raise ValueError("no threshold meets recall floor")
    threshold,recall=valid[0]
    tp=sum(row["label"]==1 and row["probability"]>=threshold for row in fall)
    fp=sum(row["label"]==0 and row["probability"]>=threshold for row in fall)
    precision=tp/(tp+fp) if tp+fp else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"selection_source":"fall_loso_predictions_only","threshold":threshold,"recall_floor":float(recall_floor),"fall_precision":precision,"fall_recall":recall,"fall_f1":f1,"adl_false_positive_rate":sum(row["probability"]>=threshold for row in adl)/len(adl)}

def _read(path: Path, *, expected_labels: set[int]) -> list[dict[str, Any]]:
    try: rows=[json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    except (OSError,UnicodeError,json.JSONDecodeError) as e: raise ValueError(f"cannot read predictions: {e}") from e
    if not rows: raise ValueError("prediction file is empty")
    for row in rows:
        if not isinstance(row,dict) or type(row.get("label")) is not int or row["label"] not in expected_labels or not isinstance(row.get("probability"),(int,float)) or isinstance(row["probability"],bool) or not isfinite(float(row["probability"])) or not 0<=float(row["probability"])<=1: raise ValueError("invalid prediction row")
    return rows
