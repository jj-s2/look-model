"""
测试数据集下载与校验脚本的基础功能。

不实际发起网络下载，只验证：
- manifest.json 能正确加载
- 脚本可被 import
- --list 命令可正常输出
- 哈希计算函数对已知内容返回正确值
"""

from __future__ import annotations

import json
import subprocess
import sys
import hashlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "manifest.json"

# 将 scripts 加入 import 路径
sys.path.insert(0, str(SCRIPTS_DIR))


class TestManifest:
    """测试 manifest.json 的结构与内容。"""

    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists(), f"未找到 {MANIFEST_PATH}"

    def test_manifest_schema(self):
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            m = json.load(f)
        assert "schema_version" in m
        assert "datasets" in m
        assert isinstance(m["datasets"], list)
        assert len(m["datasets"]) >= 1

    def test_gmdcsa24_entry(self):
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            m = json.load(f)
        names = [d["name"] for d in m["datasets"]]
        assert "GMDCSA24" in names, "GMDCSA24 必须在 manifest 中"

    def test_gmdcsa24_has_official_source(self):
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            m = json.load(f)
        for d in m["datasets"]:
            if d["name"] == "GMDCSA24":
                assert "zenodo" in d["official_source"].lower()
                assert d["expected_size_bytes"] > 0
                assert d["license"]
                assert d["citation"]
                return
        pytest.fail("未找到 GMDCSA24")

    def test_global_rules_present(self):
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            m = json.load(f)
        rules = m.get("global_rules", {})
        assert "no_random_split_by_video" in rules
        assert "no_unofficial_mirror" in rules
        assert "checkpoint_resume" in rules


class TestDownloadScript:
    """测试 download_datasets.py 的基础功能。"""

    def test_script_importable(self):
        import download_datasets
        assert hasattr(download_datasets, "download_with_resume")
        assert hasattr(download_datasets, "compute_hash")
        assert hasattr(download_datasets, "load_manifest")

    def test_list_command(self):
        """运行 python scripts/download_datasets.py --list 应返回 0。"""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "download_datasets.py"), "--list"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "GMDCSA24" in result.stdout

    def test_compute_hash_known_value(self):
        """对已知内容计算 SHA256 应返回正确值。"""
        import download_datasets as dd
        tmp = PROJECT_ROOT / "tests" / "_tmp_hash_test.txt"
        tmp.write_bytes(b"hello world")
        try:
            actual = dd.compute_hash(tmp, "sha256")
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert actual == expected
        finally:
            tmp.unlink(missing_ok=True)


class TestVerifyScript:
    """测试 verify_datasets.py 的基础功能。"""

    def test_script_importable(self):
        import verify_datasets
        assert hasattr(verify_datasets, "verify_one")
        assert hasattr(verify_datasets, "compute_hash")

    def test_verify_one_rejects_wrong_md5(self, tmp_path, monkeypatch):
        import verify_datasets as vd

        raw_dir = tmp_path / "raw"
        target_dir = raw_dir / "fixture"
        target_dir.mkdir(parents=True)
        monkeypatch.setattr(vd, "RAW_DIR", raw_dir)
        monkeypatch.setattr(vd, "PROJECT_ROOT", tmp_path)
        target = target_dir / "sample.bin"
        target.write_bytes(b"known content")
        result = vd.verify_one({
            "name": "fixture",
            "version": "1",
            "expected_size_bytes": target.stat().st_size,
            "md5": "00000000000000000000000000000000",
            "sha256": None,
        })

        assert result["status"] == "mismatch"
        assert any("md5 mismatch" in issue for issue in result["issues"])
