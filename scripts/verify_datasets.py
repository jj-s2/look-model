"""
数据集校验脚本

校验 datasets/raw/ 下所有已下载数据集的：
- 文件存在性
- 文件大小
- SHA256 / MD5

并与 datasets/manifest.json 中记录的校验值比对。
未通过校验的条目会输出到 experiments/logs/verification_report.json。

使用方式：
    python scripts/verify_datasets.py
    python scripts/verify_datasets.py --dataset GMDCSA24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "manifest.json"
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
REPORT_PATH = PROJECT_ROOT / "experiments" / "logs" / "verification_report.json"


def compute_hash(file_path: Path, algorithm: str = "sha256",
                 chunk_size: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_one(ds: dict) -> dict:
    """校验单个数据集条目。"""
    name = ds["name"]
    version = ds.get("version") or "unknown"
    expected_sha256 = ds.get("sha256")
    expected_md5 = ds.get("md5")
    expected_size = ds.get("expected_size_bytes") or ds.get("actual_size_bytes")

    # 推测文件路径
    candidates = list((RAW_DIR / name).glob("*")) if (RAW_DIR / name).exists() else []
    if not candidates:
        return {
            "name": name,
            "version": version,
            "status": "missing",
            "message": f"未找到 {RAW_DIR / name} 下的任何文件",
            "expected_sha256": expected_sha256,
            "actual_sha256": None,
        }

    target_file = candidates[0]
    actual_size = target_file.stat().st_size
    actual_sha256 = compute_hash(target_file, "sha256")
    actual_md5 = compute_hash(target_file, "md5")

    issues = []
    if expected_size and actual_size != expected_size:
        issues.append(f"size mismatch: expected {expected_size}, got {actual_size}")
    if expected_sha256 and actual_sha256 != expected_sha256:
        issues.append(f"sha256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    if expected_md5 and actual_md5 != expected_md5:
        issues.append(f"md5 mismatch: expected {expected_md5}, got {actual_md5}")

    return {
        "name": name,
        "version": version,
        "file": str(target_file.relative_to(PROJECT_ROOT)),
        "status": "ok" if not issues else "mismatch",
        "issues": issues,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="数据集校验脚本")
    parser.add_argument("--dataset", help="只校验指定数据集")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"[ERROR] 未找到 manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    reports = []
    for ds in manifest["datasets"]:
        if args.dataset and ds["name"] != args.dataset:
            continue
        print(f"[VERIFY] {ds['name']} ...", end=" ", flush=True)
        report = verify_one(ds)
        reports.append(report)
        print(report["status"].upper())
        if report["status"] != "ok":
            for issue in report.get("issues", [report.get("message", "")]):
                print(f"         - {issue}")

    # 保存报告
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "verified_at": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime()),
            "reports": reports,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[REPORT] {REPORT_PATH}")
    failed = [r for r in reports if r["status"] != "ok"]
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
