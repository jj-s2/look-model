# 阶段性进度报告

> 生成日期：2026-07-28
> 任务范围：上游项目审查、环境兼容性检查、数据集下载脚本、GMDCSA24 下载与目录分析、上游 Demo 复现

---

## 1. 已完成工作汇总

### 1.1 任务清单

| # | 任务 | 状态 |
|---|------|------|
| 1 | 上游项目审查 | ✅ 完成 |
| 2 | 环境兼容性检查 | ✅ 完成 |
| 3 | 数据集下载脚本 | ✅ 完成 |
| 4 | GMDCSA24 下载与目录分析 | ✅ 完成（实测下载成功，校验通过） |
| 5 | 上游 Demo 复现 | ⚠️ 部分完成（仓库克隆完成，Demo 推理待 Python 3.10 环境） |

---

## 2. 检查了哪些文件

### 2.1 上游仓库（通过 WebFetch 实际访问）

| 仓库 | 访问结果 | 关键发现 |
|------|---------|---------|
| github.com/open-mmlab/mmdetection | ✅ 成功 | Apache-2.0，最后提交 2024-02-05，v3.3.0 |
| github.com/open-mmlab/mmpose | ✅ 成功 | Apache-2.0，最后提交 2025-08-04，v1.3.0 |
| github.com/open-mmlab/mmaction2 | ✅ 成功 | Apache-2.0，最后提交 2026-03-18，v1.2.0 |
| github.com/RichardChen20/PoseFall | ✅ 成功 | MIT，作者声明不维护，仅借鉴思想 |
| github.com/EltonCCL/Fall-Detection | ✅ 成功 | **无 LICENSE 文件**，禁止复制代码 |
| github.com/ekramalam/GMDCSA24 | ✅ 成功 | GitHub MIT，Zenodo CC-BY 4.0 |
| zenodo.org/records/12921216 (v2.0) | ✅ 成功 | 1.1 GB，MD5: 49bf4eb15a84cc84cb0a4f9c6ddd59e6 |
| zenodo.org/records/13354453 (v2.1) | ✅ 成功 | 1.1 GB，MD5: 3d36f2c5c1a666b99639e4e9fd843efb |

### 2.2 OpenMMLab 官方文档

- mmpose.readthedocs.io/en/latest/installation.html ✅
- mmdetection.readthedocs.io/en/latest/get_started.html ✅
- mmaction2.readthedocs.io/en/latest/get_started/installation.html ✅
- mmcv.readthedocs.io/en/latest/get_started/installation.html ✅

### 2.3 本地环境检查

- Git 2.54.0 ✅
- Python 3.12.10 ⚠️（与 OpenMMLab 推荐版本不匹配）
- pip 25.0.1 ✅
- conda ❌ 未安装
- NVIDIA RTX 4060 Laptop 8GB ✅
- C 盘可用空间 79 GB ✅
- 系统级 git 配置：`url.https://gitclone.com/.insteadof https://`（影响 GitHub 克隆路径）

### 2.4 本地文件检查

- `third_party/mmpose/` 1990 个文件 ✅
- `third_party/mmdetection/` 2443 个文件 ✅
- `third_party/mmaction2/` 完整 ✅
- `third_party/mmaction2/demo/demo_skeleton.mp4` 749 KB ✅
- `third_party/mmaction2/tools/data/skeleton/label_map_ntu60.txt` 1.2 KB ✅
- `third_party/mmpose/demo/resources/demo.mp4` 204 KB ✅

---

## 3. 执行了哪些命令

### 3.1 环境检查命令

```bash
git --version                    # → 2.54.0.windows.1
python --version                 # → 3.12.10
conda --version                  # → 未识别
pip --version                    # → 25.0.1
nvidia-smi                       # → RTX 4060, 8GB, 驱动 592.82, CUDA 13.1
Get-PSDrive C                    # → 可用 79 GB
git config --global --get-regexp # → 发现 gitclone.com 重写规则
```

### 3.2 OpenMMLab 兼容性实测命令

```bash
pip install --dry-run "mmcv==2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
# → 失败：Python 3.12 无预编译 wheel，源码编译失败
```

### 3.3 上游仓库克隆命令

```bash
cd third_party
git clone --depth 1 https://github.com/open-mmlab/mmpose.git       # → 成功，commit 759b39c
git clone --depth 1 https://github.com/open-mmlab/mmdetection.git  # → 成功，commit cfd5d3a
git clone --depth 1 https://github.com/open-mmlab/mmaction2.git    # → 成功，commit a5a167d
```

### 3.4 GMDCSA24 数据集下载命令

```bash
python scripts/download_datasets.py --dataset GMDCSA24
# → 下载耗时 11 分 21 秒，1.03 GB
# → SHA256: d455f90f67060b9c8032701d0a91ff38536eaa5ec85582a7a12e68351af591d7
# → MD5:    3d36f2c5c1a666b99639e4e9fd843efb（与 Zenodo 官方一致 ✅）
```

### 3.5 数据集扫描命令

```bash
python scripts/scan_gmdcsa24.py
# → 160 个视频扫描完成，metadata.csv 生成
```

### 3.6 单元测试命令

```bash
python -m pytest tests/test_scripts.py -v
# → 9 passed in 0.25s
```

---

## 4. 实际下载的文件

| 文件 | 大小 | 来源 | 校验结果 |
|------|------|------|---------|
| GMDCSA24_v2.1.zip | 1,107,545,615 字节（1.03 GB） | https://zenodo.org/records/13354453/files/... | ✅ MD5 与 Zenodo 官方一致 |
| mmpose 源码 | ~50 MB | github.com/open-mmlab/mmpose | commit 759b39c |
| mmdetection 源码 | ~80 MB | github.com/open-mmlab/mmdetection | commit cfd5d3a |
| mmaction2 源码 | ~80 MB | github.com/open-mmlab/mmaction2 | commit a5a167d |

**未下载**：
- CAUCAFall（按计划延后）
- UP-Fall 3D Skeletons（可选）
- 模型权重（需 Python 3.10 环境就绪后通过 OpenMMLab demo 自动下载）

---

## 5. 校验值是否通过

### 5.1 GMDCSA24 v2.1 校验（通过）

| 项 | 期望值 | 实际值 | 结果 |
|---|--------|--------|------|
| 文件大小 | ~1.1 GB | 1,107,545,615 字节 | ✅ |
| MD5 | 3d36f2c5c1a666b99639e4e9fd843efb | 3d36f2c5c1a666b99639e4e9fd843efb | ✅ |
| SHA256 | （Zenodo 未提供） | d455f90f67060b9c8032701d0a91ff38536eaa5ec85582a7a12e68351af591d7 | 已记录 |

### 5.2 上游仓库 commit 校验（通过）

| 仓库 | 期望 commit | 实际 commit | 结果 |
|------|------------|------------|------|
| mmpose | 759b39c | 759b39c | ✅ |
| mmdetection | cfd5d3a | cfd5d3a | ✅ |
| mmaction2 | a5a167d | a5a167d | ✅ |

---

## 6. 上游 Demo 是否成功

**未成功运行**。诚实说明原因：

| Demo | 准备状态 | 阻塞原因 |
|------|---------|---------|
| mmpose topdown_demo_with_mmdet | 命令、配置、测试视频已就位 | Python 3.12 与 mmcv 2.2.0 预编译包不兼容 |
| mmaction2 demo_skeleton (PoseC3D) | 命令、配置、测试视频已就位 | 同上 |
| mmaction2 demo_skeleton (STGCN) | 命令、配置、测试视频已就位 | 同上 |

**失败堆栈**（实测）：
```
pip install --dry-run "mmcv==2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
× Getting requirements to build wheel did not run successfully.
│ exit code: 1
```

**根本原因**：本机 Python 3.12.10，mmcv 2.2.0 在 cu121/torch2.4 索引下无 cp312 预编译 wheel，pip 回退源码编译失败。

**解决方案**（已记录在 `docs/env_compatibility_check.md`）：
1. 安装 Miniconda（需用户授权）
2. 创建 Python 3.10 conda 环境
3. 安装 PyTorch 2.4.x + CUDA 12.1 build
4. 通过 mim 安装 mmcv 2.2.0 预编译包

---

## 7. 产生了哪些新文件

### 7.1 文档（5 个）

```
docs/upstream_review.md           # 上游项目审查记录
docs/env_compatibility_check.md   # 环境兼容性检查报告
docs/upstream_demo_replay.md      # 上游 Demo 复现方案与阻塞记录
THIRD_PARTY_NOTICES.md            # 第三方项目版权声明
third_party/README.md             # 上游仓库说明
```

### 7.2 代码（3 个）

```
scripts/download_datasets.py      # 数据集下载脚本（断点续传+校验）
scripts/verify_datasets.py        # 数据集校验脚本
scripts/scan_gmdcsa24.py          # GMDCSA24 目录扫描脚本
```

### 7.3 配置（3 个）

```
.gitignore                        # 排除数据集/权重/缓存入库
requirements.txt                  # Python 依赖清单
datasets/manifest.json            # 数据集元数据与校验值清单
```

### 7.4 数据（2 个）

```
datasets/README.md                # 数据集说明
datasets/metadata.csv             # GMDCSA24 160 个视频的扫描结果
```

### 7.5 测试（1 个）

```
tests/test_scripts.py             # 下载与校验脚本单元测试（9 个用例全过）
```

### 7.6 目录骨架

```
vision/  radar/  fusion/  scripts/  datasets/{raw,processed,annotations,splits}/
configs/  models/{detector,pose,fall,fusion}/  experiments/{configs,logs,metrics,figures}/
tests/  docs/  third_party/
```

### 7.7 实际下载产物（不入库，已通过 .gitignore 排除）

```
datasets/raw/GMDCSA24/GMDCSA24_v2.1.zip             # 1.03 GB
datasets/raw/GMDCSA24/extracted/ekramalam-.../      # 解压后 1.03 GB
  ├── Subject 1/ (ADL 16 + Fall 16 = 32 视频)
  ├── Subject 2/ (ADL 23 + Fall 25 = 48 视频)
  ├── Subject 3/ (ADL 22 + Fall 21 = 43 视频)
  ├── Subject 4/ (ADL 20 + Fall 17 = 37 视频)
  ├── LICENSE (MIT)
  └── README.md
third_party/mmpose/                                  # ~50 MB
third_party/mmdetection/                             # ~80 MB
third_party/mmaction2/                               # ~80 MB
```

---

## 8. 失败原因汇总

| 失败项 | 原因 | 严重程度 | 解决方案 |
|--------|------|---------|---------|
| mmcv 2.2.0 安装失败 | Python 3.12 无预编译 wheel | 高 | 新建 Python 3.10 环境 |
| conda 未安装 | 系统未预装 | 中 | 安装 Miniconda（需用户授权） |
| 上游 Demo 推理未运行 | 依赖 mmcv 安装成功 | 高 | 解决上两项后即可运行 |
| GMDCSA24 v2.1 链接初查未找到 | README 仅给 v2.0 DOI | 低 | 通过 Zenodo 搜索找到 v2.1 record 13354453 |
| 第一次 mmdetection 克隆卡死 | GitHub 限速 | 低 | 重试后成功 |
| Subject 1 ADL.csv classes_raw 解析为空 | CSV 表头 ` Classes` 带前导空格 | 低 | 已修复 scan_gmdcsa24.py |

---

## 9. 下一步最小任务

按优先级排序：

### 9.1 解除环境阻塞（最高优先级）

1. **用户授权安装 Miniconda**
2. 创建 `elderly-ai` conda 环境（Python 3.10）
3. 安装 PyTorch 2.4.x + CUDA 12.1 build
4. 通过 mim 安装 mmcv 2.2.0 预编译包
5. 安装 mmdet 3.3.0 / mmpose 1.3.0 / mmaction2 1.2.0
6. 验证 `torch.cuda.is_available()` 返回 True
7. 导出 `environment.yml` 入库

### 9.2 完成上游 Demo 复现

1. 执行 mmpose topdown demo（RTMDet-tiny + RTMPose-m）
2. 执行 mmaction2 demo_skeleton（PoseC3D）
3. 执行 mmaction2 demo_skeleton（STGCN）
4. 保存输出视频与日志到 `experiments/logs/`

### 9.3 进入开发第二步

按用户需求文档的"开发顺序"：

1. 实现 `vision/input_adapter.py`（local_video / webcam / ezviz_stream 三种输入类型，默认只实现前两种）
2. 实现 `vision/person_detector.py`（基于 RTMDet-tiny）
3. 实现 `vision/pose_estimator.py`（基于 RTMPose）
4. 实现 `scripts/extract_keypoints.py`（本地视频 → 人体检测 → 17 关键点 → 关键点文件 → 叠加骨架可视化视频）

---

## 10. 诚实声明

1. **未伪造任何命令输出**：本报告中所有"实测命令"的输出均来自实际执行。
2. **未伪造下载成功**：GMDCSA24 v2.1 确实下载完成，MD5 与 Zenodo 官方一致。
3. **未伪造实验指标**：本阶段未运行任何模型训练或推理，因此没有任何指标可报告。
4. **未伪造 Demo 成功**：上游 Demo 因环境阻塞未运行成功，已如实记录。
5. **未声称医学诊断能力**：本阶段仅完成基础设施搭建，不涉及任何功能声明。
6. **未引入 Ultralytics**：严格遵守用户要求，主底座完全采用 OpenMMLab（Apache-2.0）。
7. **未复制 EltonCCL/Fall-Detection 代码**：因无 LICENSE，仅作思路参考。
