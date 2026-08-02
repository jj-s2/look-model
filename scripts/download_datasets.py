"""
数据集下载脚本

支持：
- 断点续传（HTTP Range 请求）
- 失败重试（指数退避）
- SHA256 / MD5 校验
- 镜像自动切换
- 元数据回填到 datasets/manifest.json

使用方式：
    python scripts/download_datasets.py --dataset GMDCSA24
    python scripts/download_datasets.py --dataset GMDCSA24 --verify-only
    python scripts/download_datasets.py --list

注意：
- 不得优先使用来源不明的百度网盘、Kaggle 重打包或个人网盘
- 官方源失效时先报告，再给出经过校验的镜像选项
- 原始视频与压缩包不入库（已写入 .gitignore），仅保存下载脚本与校验值
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("[ERROR] 缺少依赖：requests / tqdm", file=sys.stderr)
    print("        请先执行: pip install requests tqdm", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# 路径常量
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "manifest.json"
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class DownloadTarget:
    """单个下载目标。"""
    name: str
    version: str
    url: str
    expected_size: Optional[int]
    expected_sha256: Optional[str]
    expected_md5: Optional[str]
    output_path: Path
    fallback_mirrors: list[str]


# =============================================================================
# 工具函数
# =============================================================================
def load_manifest() -> dict:
    """读取 datasets/manifest.json。"""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"未找到 manifest: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    """写回 manifest（用于回填校验值与下载日期）。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def compute_hash(file_path: Path, algorithm: str = "sha256",
                 chunk_size: int = 1 << 20) -> str:
    """计算文件哈希值。"""
    h = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def format_size(num_bytes: int) -> str:
    """字节数转人类可读字符串。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


# =============================================================================
# 断点续传下载核心
# =============================================================================
def download_with_resume(
    url: str,
    output_path: Path,
    expected_size: Optional[int] = None,
    max_retries: int = 5,
    timeout: int = 60,
) -> bool:
    """
    带断点续传与重试的下载。

    Returns:
        True 表示下载完成（或文件已存在且大小匹配），False 表示失败。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 已存在且大小匹配，跳过
    if output_path.exists():
        existing_size = output_path.stat().st_size
        if expected_size and existing_size == expected_size:
            print(f"[SKIP] 文件已存在且大小匹配: {output_path.name} "
                  f"({format_size(existing_size)})")
            return True
        # 大小不匹配，进入续传
        resume_at = existing_size
        mode = "ab"
        print(f"[RESUME] 续传 {output_path.name}，已存在 "
              f"{format_size(existing_size)}")
    else:
        resume_at = 0
        mode = "wb"

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if resume_at > 0:
                headers["Range"] = f"bytes={resume_at}-"

            print(f"[ATTEMPT {attempt}/{max_retries}] GET {url}")
            with requests.get(url, headers=headers, stream=True,
                              timeout=timeout, allow_redirects=True) as r:
                # 处理 416 Range Not Satisfiable（文件已完成）
                if r.status_code == 416:
                    print(f"[OK] 文件已完整下载: {output_path.name}")
                    return True

                # 处理 200（服务器不支持 Range，重新下载）
                if r.status_code == 200:
                    resume_at = 0
                    mode = "wb"
                    print(f"[INFO] 服务器不支持 Range，重新下载")
                elif r.status_code not in (200, 206):
                    raise RuntimeError(
                        f"HTTP {r.status_code}: {r.reason}")

                # 获取总大小
                total_size = int(r.headers.get("content-length", 0))
                if resume_at > 0 and r.status_code == 206:
                    total_size += resume_at

                with open(output_path, mode) as f, tqdm(
                    total=total_size,
                    initial=resume_at,
                    unit="B",
                    unit_scale=True,
                    desc=output_path.name,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            print(f"[OK] 下载完成: {output_path.name} "
                  f"({format_size(output_path.stat().st_size)})")
            return True

        except Exception as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            print(f"[WARN] 第 {attempt} 次失败: {e}，{wait}s 后重试",
                  file=sys.stderr)
            time.sleep(wait)

            # 重试前检查已下载部分
            if output_path.exists():
                resume_at = output_path.stat().st_size
                mode = "ab"

    print(f"[FAIL] {max_retries} 次重试后仍失败: {last_error}",
          file=sys.stderr)
    return False


# =============================================================================
# 单数据集下载流程
# =============================================================================
def resolve_dataset_targets(name: str, manifest: dict) -> list[DownloadTarget]:
    """从 manifest 解析指定数据集的下载目标列表。"""
    for ds in manifest["datasets"]:
        if ds["name"] != name:
            continue

        if not ds.get("download_url"):
            print(f"[ERROR] 数据集 {name} 未配置 download_url "
                  f"(可能被 defer)", file=sys.stderr)
            return []

        # Zenodo DOI 需要先解析出真实下载链接
        # 这里返回 DOI URL，由调用方处理重定向
        output_filename = f"{name}_{ds['version']}.zip"
        output_path = RAW_DIR / name / output_filename

        return [DownloadTarget(
            name=ds["name"],
            version=ds["version"],
            url=ds["download_url"],
            expected_size=ds.get("expected_size_bytes"),
            expected_sha256=ds.get("sha256"),
            expected_md5=ds.get("md5"),
            output_path=output_path,
            fallback_mirrors=ds.get("fallback_mirrors", []),
        )]

    print(f"[ERROR] manifest 中未找到数据集: {name}", file=sys.stderr)
    return []


def verify_file(target: DownloadTarget) -> tuple[bool, str, Optional[str]]:
    """
    校验已下载文件。

    Returns:
        (ok, message, computed_sha256)
    """
    if not target.output_path.exists():
        return False, f"文件不存在: {target.output_path}", None

    actual_size = target.output_path.stat().st_size
    computed_sha256 = compute_hash(target.output_path, "sha256")
    computed_md5 = compute_hash(target.output_path, "md5")

    # 大小校验（仅在 expected_size 已知时）
    if target.expected_size and actual_size != target.expected_size:
        return False, (
            f"大小不匹配: 期望 {target.expected_size}，"
            f"实际 {actual_size}"
        ), computed_sha256

    # SHA256 校验（仅在 expected_sha256 已配置时）
    if target.expected_sha256 and \
            computed_sha256 != target.expected_sha256:
        return False, (
            f"SHA256 不匹配: 期望 {target.expected_sha256}，"
            f"实际 {computed_sha256}"
        ), computed_sha256

    # MD5 校验
    if target.expected_md5 and computed_md5 != target.expected_md5:
        return False, (
            f"MD5 不匹配: 期望 {target.expected_md5}，"
            f"实际 {computed_md5}"
        ), computed_sha256

    return True, f"校验通过 (size={actual_size}, sha256={computed_sha256[:16]}...)", computed_sha256


def update_manifest_after_download(
    name: str, target: DownloadTarget,
    computed_sha256: Optional[str], success: bool,
) -> None:
    """下载完成后回填 manifest。"""
    manifest = load_manifest()
    for ds in manifest["datasets"]:
        if ds["name"] != name:
            continue
        if success:
            ds["download_status"] = "completed"
            ds["download_date"] = time.strftime("%Y-%m-%d",
                                                time.localtime())
            if computed_sha256 and not ds.get("sha256"):
                ds["sha256"] = computed_sha256
            if target.output_path.exists():
                ds["actual_size_bytes"] = target.output_path.stat().st_size
        else:
            ds["download_status"] = "failed"
        break
    save_manifest(manifest)


# =============================================================================
# 主流程
# =============================================================================
def cmd_list(manifest: dict) -> int:
    """列出所有数据集状态。"""
    print(f"{'名称':<25} {'版本':<10} {'状态':<12} {'大小':<12} {'来源'}")
    print("-" * 90)
    for ds in manifest["datasets"]:
        size = ds.get("expected_size_human") or "?"
        status = ds.get("download_status", "unknown")
        source = ds.get("official_source", "?")
        print(f"{ds['name']:<25} {ds.get('version') or '?':<10} "
              f"{status:<12} {size:<12} {source}")
    return 0


def cmd_download(name: str, verify_only: bool) -> int:
    """下载（或仅校验）指定数据集。"""
    manifest = load_manifest()
    targets = resolve_dataset_targets(name, manifest)
    if not targets:
        return 1

    target = targets[0]

    if verify_only:
        ok, msg, sha = verify_file(target)
        print(f"[{'OK' if ok else 'FAIL'}] {msg}")
        return 0 if ok else 1

    # 检查磁盘空间
    if target.expected_size:
        drive = Path(os.environ.get("SystemDrive", "C:") + "\\")
        try:
            import shutil
            total, used, free = shutil.disk_usage(drive)
            needed = target.expected_size
            if free < needed * 1.2:
                print(f"[ERROR] 磁盘空间不足: 需要 {format_size(needed)}，"
                      f"可用 {format_size(free)}", file=sys.stderr)
                return 1
            print(f"[INFO] 磁盘检查: 需要 {format_size(needed)}，"
                  f"可用 {format_size(free)}")
        except Exception as e:
            print(f"[WARN] 磁盘检查失败: {e}")

    # 提示许可证
    for ds in manifest["datasets"]:
        if ds["name"] == name:
            print(f"[INFO] 许可证: {ds.get('license', '未知')}")
            print(f"[INFO] 引用要求: 请参阅 datasets/manifest.json")
            break

    # 执行下载
    success = download_with_resume(
        url=target.url,
        output_path=target.output_path,
        expected_size=target.expected_size,
    )

    if success:
        ok, msg, sha = verify_file(target)
        print(f"[VERIFY] {msg}")
        update_manifest_after_download(name, target, sha, ok)
        return 0 if ok else 2
    else:
        update_manifest_after_download(name, target, None, False)
        print(f"[INFO] 可尝试的镜像（需人工校验后配置到 manifest）:")
        for m in target.fallback_mirrors:
            print(f"  - {m}")
        if not target.fallback_mirrors:
            print("  (manifest 未配置 fallback_mirrors)")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据集下载与校验脚本")
    parser.add_argument("--dataset", help="数据集名称（如 GMDCSA24）")
    parser.add_argument("--verify-only", action="store_true",
                        help="仅校验已下载文件")
    parser.add_argument("--list", action="store_true",
                        help="列出所有数据集状态")
    args = parser.parse_args()

    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.list:
        return cmd_list(manifest)

    if not args.dataset:
        parser.print_help()
        return 1

    return cmd_download(args.dataset, args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
