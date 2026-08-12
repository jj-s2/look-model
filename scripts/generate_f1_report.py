"""Generate the human-readable F1-optimization handoff report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "superpowers" / "handoff" / "2026-08-04-f1-optimization-report.md"


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def generate_f1_report(experiment: Mapping[str, object]) -> str:
    """Render the optimization report from an experiment result dictionary."""

    candidate = dict(experiment.get("candidate", {}))
    inner = dict(experiment.get("inner_metrics", {}))
    confirmation = dict(experiment.get("confirmation_metrics", {}))
    baseline = dict(experiment.get("baseline", {}))
    promoted = bool(experiment.get("promoted", False))

    lines = [
        "# PA-DTSF F1 优化实验报告",
        "",
        "**日期**：2026-08-04  ",
        "**目标**：在保持 Recall ≥ 0.75、每个受试者 Recall ≥ 0.60 的前提下，提升跌倒事件 F1。",
        "",
        "## 1. 校准与阈值",
        "",
        f"| 项目 | 值 |",
        f"| --- | ---: |",
        f"| temperature | {float(experiment.get('temperature', 1.0)):.4f} |",
        f"| threshold | {float(experiment.get('threshold', 0.5)):.4f} |",
        f"| inner folds | {experiment.get('fold_thresholds', [])} |" if experiment.get("fold_thresholds") else "",
        "",
        "## 2. 验证集（inner）指标",
        "",
        f"| 指标 | 结果 |",
        f"| --- | ---: |",
        f"| mean F1 | {_pct(inner.get('inner_mean_f1', 0.0))} |",
        f"| mean Recall | {_pct(inner.get('inner_mean_recall', 0.0))} |",
        f"| worst subject Recall | {_pct(inner.get('worst_subject_recall', 0.0))} |",
        f"| subject macro F1 | {_pct(inner.get('subject_macro_f1', 0.0))} |",
        f"| ECE | {float(inner.get('ece', 0.0)):.4f} |",
        f"| Brier | {float(inner.get('brier', 0.0)):.4f} |",
        "",
        "## 3. 测试集（confirmation）指标",
        "",
        f"| 指标 | 结果 |",
        f"| --- | ---: |",
        f"| confirmation F1 | {_pct(confirmation.get('inner_mean_f1', 0.0))} |",
        f"| confirmation Recall | {_pct(confirmation.get('inner_mean_recall', 0.0))} |",
        f"| worst subject Recall | {_pct(confirmation.get('worst_subject_recall', 0.0))} |",
        f"| subject macro F1 | {_pct(confirmation.get('subject_macro_f1', 0.0))} |",
        f"| ECE | {float(confirmation.get('ece', 0.0)):.4f} |",
        f"| Brier | {float(confirmation.get('brier', 0.0)):.4f} |",
        "",
        "## 4. Promotion gate",
        "",
        f"**promoted**：{'是' if promoted else '否'}  ",
        "",
        "| 门控条件 | 候选值 | 基线/下限 | 通过 |",
        "| --- | --- | --- | --- |",
        f"| inner F1 增益 | {_pct(candidate.get('inner_mean_f1', 0.0))} | ≥ {_pct(baseline.get('inner_mean_f1', 0.0) + 0.03)} | {'是' if candidate.get('inner_mean_f1', 0.0) >= baseline.get('inner_mean_f1', 0.0) + 0.03 else '否'} |",
        f"| inner Recall | {_pct(candidate.get('inner_mean_recall', 0.0))} | ≥ 75.0% | {'是' if candidate.get('inner_mean_recall', 0.0) >= 0.75 else '否'} |",
        f"| worst subject Recall | {_pct(candidate.get('worst_subject_recall', 0.0))} | ≥ 60.0% | {'是' if candidate.get('worst_subject_recall', 0.0) >= 0.60 else '否'} |",
        f"| confirmation F1 | {_pct(candidate.get('confirmation_f1', 0.0))} | ≥ 80.0% | {'是' if candidate.get('confirmation_f1', 0.0) >= 0.80 else '否'} |",
        f"| confirmation Recall | {_pct(candidate.get('confirmation_recall', 0.0))} | ≥ 75.0% | {'是' if candidate.get('confirmation_recall', 0.0) >= 0.75 else '否'} |",
        f"| ECE | {float(candidate.get('ece', 0.0)):.4f} | ≤ 0.15 | {'是' if candidate.get('ece', 0.0) <= 0.15 else '否'} |",
        f"| Brier | {float(candidate.get('brier', 0.0)):.4f} | ≤ {float(baseline.get('brier', 0.215)):.4f} | {'是' if candidate.get('brier', 1.0) <= baseline.get('brier', 0.215) else '否'} |",
        "",
        "## 5. 证据边界",
        "",
        "- 所有阈值和温度参数仅在 validation split 上搜索，未在 test split 上调参。",
        "- 测试集结果为留出受试者（subject-level split）上的独立评估。",
        "- 当前结果基于 GMDCSA24 模拟青年受试者数据，不能直接外推至真实老年人群或临床场景。",
        "- 未进行心理健康标签验证，不生成心理健康“准确率”结论。",
        "",
        "## 6. 产物",
        "",
        "- `configs/skeleton/padtsf_v1_f1.py`：F1 优化配置。",
        "- `scripts/optimize_phase_threshold.py`：内层 3-fold 阈值搜索。",
        "- `scripts/run_phase_f1_experiment.py`：端到端实验与 promotion gate。",
        "- `scripts/plot_phase_results.py`：已更新以 overlay calibration threshold。",
        "- promoted release（若 gate 通过）：包含 `calibration.json` 的 release 目录。",
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path, help="Path to experiment.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    report = generate_f1_report(experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
