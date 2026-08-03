# PA-DTSF Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 建立可追溯的公开数据注册、统一相位标签、数据适配器和无泄漏受试者划分，为 PA-DTSF 训练提供稳定输入。

**Architecture:** 原始数据仍由现有下载脚本管理，新建 datasets/unified 包只负责元数据、统一 schema、标签映射和划分。所有适配器输出 UnifiedClip；训练脚本只能读取冻结的 dataset_lock.json 和 split_manifest.json。

**Tech Stack:** Python 3.10+、标准库 dataclasses/json/hashlib/pathlib、pytest；pandas 只允许在脚本边界使用。

## Global Constraints

- 原始视频、压缩包、关键点缓存和受限数据不得提交 Git。
- 所有拆分按 subject_id 和同步 event_group_id 分组。
- 许可不明的数据不能进入正式训练或发布包。
- 数据路径必须是项目相对路径，不能写入本机绝对路径。
- 未下载数据允许 sha256 为空；已下载完成的数据必须有真实 sha256。
- 现有 GMDCSA24 数据和 tests/test_scripts.py 行为必须保持兼容。

---

### Task 1: 修正数据注册表和校验缺陷

**Files:**
- Modify: datasets/manifest.json
- Create: datasets/unified/__init__.py
- Create: datasets/unified/registry.py
- Create: tests/datasets/test_registry.py
- Modify: scripts/verify_datasets.py
- Modify: tests/test_scripts.py

**Interfaces:**
- Produces: DatasetEntry、DatasetRegistry、load_registry(path: Path) -> DatasetRegistry
- Produces: DatasetEntry.can_train -> bool
- Consumes: datasets/manifest.json schema_version 2.0

- [ ] **Step 1: 写注册表失败测试**

    from pathlib import Path
    from datasets.unified.registry import load_registry

    ROOT = Path(__file__).resolve().parents[2]

    def test_required_dataset_sources_and_training_gate():
        registry = load_registry(ROOT / "datasets" / "manifest.json")
        assert registry["GMDCSA24"].expected_size_bytes == 1107545615
        assert registry["CAUCAFall"].official_source.endswith("/7w7fccy7ky/4")
        assert registry["Pre-VFall"].expected_size_bytes == 21096978752
        assert registry["UP-Fall-3D-Skeletons"].license_status == "needs_confirmation"
        assert registry["UP-Fall-3D-Skeletons"].can_train is False

    def test_downloaded_dataset_requires_sha256(tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            '{"schema_version":"2.0","datasets":[{"name":"x","version":"1",'
            '"official_source":"https://example.test","license":"CC BY 4.0",'
            '"license_status":"confirmed","download_status":"completed",'
            '"expected_size_bytes":1,"sha256":null,"subjects":1,'
            '"split_strategy":"subject_grouped","redistribution":"raw_data_not_committed"}]}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="completed dataset requires sha256"):
            load_registry(manifest)

- [ ] **Step 2: 运行测试并确认失败**

    python -m pytest tests/datasets/test_registry.py tests/test_scripts.py -q

Expected: FAIL，datasets.unified.registry 尚不存在，且旧清单的 GMDCSA24 expected_size_bytes 为 null。

- [ ] **Step 3: 更新 manifest 到 schema 2.0**

必须加入并锁定：

- GMDCSA24 expected_size_bytes=1107545615；
- Pre-VFall 版本 figshare-26488216-2025-04-28、大小 21096978752、CC BY 4.0、9 名受试者；
- CAUCAFall v4、官方源 7w7fccy7ky/4、CC BY 4.0、10 名受试者；
- UP-Fall-3D-Skeletons v1、官方源 12773013、license_status=needs_confirmation、training_allowed=false；
- OmniFall annotations、锁定数据卡 revision 字段、CC BY-NC 4.0、raw_video_license=upstream；
- NTU120 skeleton optional、academic_only=true、redistribution=forbidden；
- CHARLS optional、role=population_association_only。

- [ ] **Step 4: 实现严格注册表**

    @dataclass(frozen=True)
    class DatasetEntry:
        name: str
        version: str
        official_source: str
        license: str
        license_status: str
        download_status: str
        expected_size_bytes: int | None
        sha256: str | None
        subjects: int | None
        split_strategy: str
        redistribution: str
        training_allowed: bool = True

        @property
        def can_train(self) -> bool:
            return self.training_allowed and self.license_status == "confirmed"

    @dataclass(frozen=True)
    class DatasetRegistry:
        schema_version: str
        entries: tuple[DatasetEntry, ...]

        def __getitem__(self, name: str) -> DatasetEntry:
            return next(item for item in self.entries if item.name == name)

load_registry 必须拒绝重复 name、空版本、非 HTTPS 官方源、非正整数大小，以及 completed 但没有 sha256 的条目。GMDCSA24 已完成且已有真实 sha256，必须通过。

- [ ] **Step 5: 修复 MD5 校验比较错误**

将 scripts/verify_datasets.py 中：

    if expected_md5 and actual_md5 != actual_md5:

改为：

    if expected_md5 and actual_md5 != expected_md5:

在 tests/test_scripts.py 增加临时文件测试，传入错误 expected_md5 时 status 必须为 mismatch。

- [ ] **Step 6: 运行测试**

    python -m pytest tests/datasets/test_registry.py tests/test_scripts.py -q

Expected: PASS。

- [ ] **Step 7: 提交**

    git add datasets/manifest.json datasets/unified/__init__.py datasets/unified/registry.py tests/datasets/test_registry.py scripts/verify_datasets.py tests/test_scripts.py
    git commit -m "feat: add governed fall dataset registry"

---

### Task 2: 定义统一片段 schema 和稳定标识

**Files:**
- Create: datasets/unified/schema.py
- Create: tests/datasets/test_unified_schema.py

**Interfaces:**
- Produces: UnifiedClip、LabelSource、stable_clip_id
- Consumes: 项目相对 media_path 和来源标签

- [ ] **Step 1: 写失败测试**

    from datasets.unified.schema import UnifiedClip, stable_clip_id

    def test_clip_id_is_path_root_independent():
        first = stable_clip_id("GMDCSA24", "v2.1", "S1", "C1", "Fall/01.mp4", 1.0, 3.0)
        second = stable_clip_id("GMDCSA24", "v2.1", "S1", "C1", "Fall/01.mp4", 1.0, 3.0)
        assert first == second
        assert len(first) == 64

    def test_unknown_phase_requires_mask():
        with pytest.raises(ValueError, match="supervision_mask"):
            UnifiedClip(
                clip_id="x", dataset="d", dataset_version="1",
                subject_id="s", camera_id="c", event_group_id="e",
                media_path="relative.mp4", start_sec=0.0, end_sec=1.0,
                phase=None, coarse_event=None, hard_negative=None,
                supervision_mask=(), source_label="unknown", provenance={},
            )

    def test_absolute_media_path_is_rejected():
        with pytest.raises(ValueError, match="relative"):
            UnifiedClip(
                clip_id="x", dataset="d", dataset_version="1",
                subject_id="s", camera_id="c", event_group_id="e",
                media_path="F:/private/video.mp4",
                start_sec=0.0, end_sec=1.0, phase="normal_adl",
                coarse_event="adl", hard_negative=None,
                supervision_mask=("phase", "fall_event"),
                source_label="ADL", provenance={"source_file": "relative.mp4"},
            )

- [ ] **Step 2: 确认失败**

    python -m pytest tests/datasets/test_unified_schema.py -q

- [ ] **Step 3: 实现 schema**

UnifiedClip 字段必须与设计文档一致，并增加 event_group_id。phase 只允许六个值或 None；coarse_event 只允许 fall、adl 或 None；start_sec 必须小于 end_sec；subject_id、camera_id、source_label 不得为空。

stable_clip_id 使用 SHA-256：

    payload = "|".join([
        dataset, dataset_version, subject_id, camera_id,
        media_path.replace("\\", "/"), f"{start_sec:.6f}", f"{end_sec:.6f}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

- [ ] **Step 4: 增加 JSON 往返测试和实现**

    restored = UnifiedClip.from_dict(clip.to_dict())
    assert restored == clip
    assert restored.provenance["source_sha256"] == "abc"

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/datasets/test_unified_schema.py -q
    git add datasets/unified/schema.py tests/datasets/test_unified_schema.py
    git commit -m "feat: define unified fall clip schema"

---

### Task 3: 实现统一标签映射

**Files:**
- Create: datasets/unified/labels.py
- Create: tests/datasets/test_label_mapping.py

**Interfaces:**
- Produces: map_source_label(dataset: str, label: str, metadata: Mapping[str, object]) -> LabelMapping
- Produces: LabelMapping(phase, coarse_event, hard_negative, supervision_mask)

- [ ] **Step 1: 写表驱动失败测试**

    @pytest.mark.parametrize(
        "dataset,label,phase,coarse,hard_negative",
        [
            ("GMDCSA24", "ADL", "normal_adl", "adl", None),
            ("GMDCSA24", "Falling (FW)", None, "fall", None),
            ("Pre-VFall", "Abnormal", "prefall_abnormal", None, None),
            ("Pre-VFall", "Fall", None, "fall", None),
            ("CAUCAFall", "sitting", "normal_adl", "adl", "sit_down"),
            ("UP-Fall-3D-Skeletons", "impact", "impact", "fall", None),
            ("NTU120", "A42", "prefall_abnormal", None, None),
            ("NTU120", "A80", "normal_adl", "adl", "squat"),
        ],
    )
    def test_source_mapping(dataset, label, phase, coarse, hard_negative):
        mapped = map_source_label(dataset, label, {})
        assert (mapped.phase, mapped.coarse_event, mapped.hard_negative) == (
            phase, coarse, hard_negative
        )

- [ ] **Step 2: 确认失败**

    python -m pytest tests/datasets/test_label_mapping.py -q

- [ ] **Step 3: 实现显式映射表**

不得使用字符串包含关系猜测 impact 或 fallen。未知标签返回 phase=None、coarse_event=None、supervision_mask=("none",)，并记录 mapping_reason="unmapped_source_label"。

fall_coarse 的 supervision_mask 只能包含 fall_event，不包含 phase。单帧 Pre-VFall 不参与 phase_order。

- [ ] **Step 4: 加入非法精细标签测试**

    mapped = map_source_label("GMDCSA24", "Falling", {"impact_sec": None})
    assert mapped.phase is None
    assert "phase" not in mapped.supervision_mask

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/datasets/test_label_mapping.py -q
    git add datasets/unified/labels.py tests/datasets/test_label_mapping.py
    git commit -m "feat: map public fall labels conservatively"

---

### Task 4: 建立无泄漏受试者划分

**Files:**
- Create: datasets/unified/splits.py
- Create: tests/datasets/test_splits.py

**Interfaces:**
- Produces: grouped_split(clips, seed, train_ratio, val_ratio) -> SplitManifest
- Produces: leave_one_dataset_out(clips, held_out_dataset) -> SplitManifest
- Produces: assert_no_leakage(manifest, clips) -> None
- Produces: SplitManifest.assignments: Mapping[str, Literal["train", "validation", "test"]]

- [ ] **Step 1: 写同步机位和受试者泄漏失败测试**

    clips = [
        clip("a", subject="S1", camera="C1", event="E1"),
        clip("b", subject="S1", camera="C2", event="E1"),
        clip("c", subject="S2", camera="C1", event="E2"),
        clip("d", subject="S3", camera="C1", event="E3"),
    ]
    manifest = grouped_split(clips, seed=42, train_ratio=0.5, val_ratio=0.25)
    assert manifest.assignments["a"] == manifest.assignments["b"]
    assert_no_leakage(manifest, clips)

- [ ] **Step 2: 确认失败**

    python -m pytest tests/datasets/test_splits.py -q

- [ ] **Step 3: 实现分组键**

分组键固定为：

    group_key = f"{dataset}:{subject_id}"

同步事件 event_group_id 形成并查集合并约束。划分前按稳定哈希排序，随机种子只用于组级打乱。返回对象保存 schema_version、seed、clip_ids 和 group_ids。

- [ ] **Step 4: 实现 leave-one-dataset-out**

held_out_dataset 的全部片段只能进入 test。训练数据内部再按受试者生成 train/validation。归一化统计不得写入 split 模块。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/datasets/test_splits.py -q
    git add datasets/unified/splits.py tests/datasets/test_splits.py
    git commit -m "feat: add leakage-safe subject splits"

---

### Task 5: 实现五个数据适配器

**Files:**
- Create: datasets/unified/adapters/__init__.py
- Create: datasets/unified/adapters/base.py
- Create: datasets/unified/adapters/gmdcsa24.py
- Create: datasets/unified/adapters/prevfall.py
- Create: datasets/unified/adapters/caucafall.py
- Create: datasets/unified/adapters/upfall3d.py
- Create: datasets/unified/adapters/omnifall.py
- Create: tests/datasets/test_adapters.py
- Create: tests/fixtures/datasets/

**Interfaces:**
- Produces: DatasetAdapter.scan(root: Path) -> Sequence[UnifiedClip]
- Produces: get_adapter(name: str) -> DatasetAdapter
- Consumes: map_source_label、stable_clip_id

- [ ] **Step 1: 为每个适配器创建最小官方结构夹具**

夹具只包含 JSON/CSV/空媒体占位元数据，不复制真实视频。tests/fixtures/datasets 下分别模拟官方目录命名、受试者、机位和一条标签。

- [ ] **Step 2: 写失败测试**

    @pytest.mark.parametrize("adapter_name", [
        "gmdcsa24", "prevfall", "caucafall", "upfall3d", "omnifall"
    ])
    def test_adapter_emits_traceable_relative_clips(adapter_name, fixture_root):
        clips = get_adapter(adapter_name).scan(fixture_root / adapter_name)
        assert clips
        assert all(not Path(item.media_path).is_absolute() for item in clips)
        assert all(item.subject_id for item in clips)
        assert all("source_file" in item.provenance for item in clips)

- [ ] **Step 3: 确认失败**

    python -m pytest tests/datasets/test_adapters.py -q

- [ ] **Step 4: 实现适配器**

base.py 定义协议和公共路径检查。每个适配器只解析自己的官方格式；不能在 base.py 写数据集名称判断。

UP-Fall 适配器在注册表 can_train=false 时仍可生成 metadata_only 片段，但 supervision_mask 必须为空。OmniFall 适配器输出上游 dataset 字段，并把原视频许可状态写入 provenance。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/datasets/test_adapters.py -q
    git add datasets/unified/adapters tests/datasets/test_adapters.py tests/fixtures/datasets
    git commit -m "feat: add public fall dataset adapters"

---

### Task 6: 冻结 dataset lock 和 split manifest

**Files:**
- Create: scripts/prepare_unified_fall_data.py
- Create: tests/integration/test_prepare_unified_data.py
- Modify: .gitignore

**Interfaces:**
- Produces: prepare_unified_data(fixtures_root: Path, output_dir: Path, seed: int) -> Path
- Produces: build_dataset_lock(registry, clips, output_dir) -> Path
- Produces: build_split_manifest(clips, strategy, seed, output_dir) -> Path

- [ ] **Step 1: 写离线集成失败测试**

    def test_prepare_fixture_is_deterministic(tmp_path):
        first = prepare_unified_data(FIXTURES, tmp_path / "one", seed=42)
        second = prepare_unified_data(FIXTURES, tmp_path / "two", seed=42)
        assert json.loads((first / "dataset_lock.json").read_text("utf-8")) == json.loads(
            (second / "dataset_lock.json").read_text("utf-8")
        )
        assert json.loads((first / "split_manifest.json").read_text("utf-8")) == json.loads(
            (second / "split_manifest.json").read_text("utf-8")
        )

    def test_restricted_dataset_is_excluded_from_training(tmp_path):
        result = prepare_unified_data(FIXTURES, tmp_path, seed=42)
        split = json.loads((result / "split_manifest.json").read_text("utf-8"))
        restricted_ids = {
            item["clip_id"] for item in split["clips"]
            if item["dataset"] == "UP-Fall-3D-Skeletons"
        }
        assert not restricted_ids.intersection(split["partitions"]["train"])

- [ ] **Step 2: 确认失败**

    python -m pytest tests/integration/test_prepare_unified_data.py -q

- [ ] **Step 3: 实现命令**

命令接口固定：

    python scripts/prepare_unified_fall_data.py \
      --manifest datasets/manifest.json \
      --output datasets/annotations/unified/v1 \
      --strategy grouped \
      --seed 42

dataset_lock.json 记录注册表条目、适配器版本、样本计数、标签计数和源文件校验；split_manifest.json 记录 clip_id 到 partition 的映射。

- [ ] **Step 4: 更新忽略规则**

忽略 datasets/raw、datasets/processed 和大体积统一标注缓存；保留 registry、适配器、夹具和不含个人数据的 split schema 示例。

- [ ] **Step 5: 运行本计划验证**

    python -m pytest tests/datasets tests/integration/test_prepare_unified_data.py tests/test_scripts.py -q
    python scripts/prepare_unified_fall_data.py --fixtures tests/fixtures/datasets --output outputs/unified-fixture --strategy grouped --seed 42

Expected: 测试通过；fixture 产物明确标记 demo=true、release_eligible=false。

- [ ] **Step 6: 提交**

    git add scripts/prepare_unified_fall_data.py tests/integration/test_prepare_unified_data.py .gitignore
    git commit -m "feat: freeze reproducible fall dataset splits"

---

## Plan Completion Gate

运行：

    python -m pytest tests/datasets tests/integration/test_prepare_unified_data.py tests/test_scripts.py -q
    python -m pytest -q
    git status -sb

验收：

- 数据清单包含全部选定数据和正确官方来源；
- GMDCSA24 旧清单失败被真实大小修复；
- MD5 错误比较有回归测试；
- 所有片段有稳定 clip_id 和来源信息；
- 同一受试者、原视频和同步机位无跨折；
- 许可不明或禁止训练的数据被自动隔离；
- outputs 保持未跟踪；
- 完整回归中任何既有失败必须单独报告，不能删除测试掩盖。
