# 上游 OpenMMLab Demo 复现记录

> 生成日期：2026-07-28
> 本文档记录本项目对 OpenMMLab 三个上游仓库 Demo 的复现方案、依赖检查、命令清单与当前阻塞项。
>
> **诚实声明**：截至本文档生成时，由于本机 Python 3.12 与 mmcv 2.2.0 预编译包不兼容（实测触发源码编译失败），Demo 实际推理**尚未运行成功**。本文档记录的是已完成的准备工作与待执行的复现命令，待 Python 3.10 环境就绪后即可执行。

---

## 1. 上游仓库克隆状态（已完成）

| 仓库 | 路径 | commit | 日期 | 文件数 | 状态 |
|------|------|--------|------|--------|------|
| mmpose | `third_party/mmpose/` | 759b39c | 2025-08-04 | 1990 | ✅ 完整克隆 |
| mmdetection | `third_party/mmdetection/` | cfd5d3a | 2024-02-05 | 2443 | ✅ 完整克隆 |
| mmaction2 | `third_party/mmaction2/` | a5a167d | 2026-03-18 | - | ✅ 完整克隆 |

校验方式：`git log -1 --format="%h %ci"`，commit hash 与 `docs/upstream_review.md` 一致。

---

## 2. Demo 入口与测试视频（已就位）

### 2.1 mmpose：topdown_demo_with_mmdet.py

- 路径：`third_party/mmpose/demo/topdown_demo_with_mmdet.py`
- 功能：mmdet 检测人体 → mmpose top-down 估计 17 关键点
- 测试视频：`third_party/mmpose/demo/resources/demo.mp4`（204 KB）
- 默认配置（来自 `demo/docs/zh_cn/2d_human_pose_demo.md`）：
  - 检测器：RTMDet-m + COCO person 权重
  - 姿态估计器：RTMPose-m + Body7 权重
- 可用 RTMDet 配置（在 `third_party/mmpose/demo/mmdetection_cfg/`）：
  - `rtmdet_tiny_8xb32-300e_coco.py`（**本项目首选**）
  - `rtmdet_m_640-8xb32_coco-person.py`
  - `rtmdet_nano_320-8xb32_coco-person.py`

### 2.2 mmaction2：demo_skeleton.py

- 路径：`third_party/mmaction2/demo/demo_skeleton.py`
- 功能：**端到端骨架动作识别**（人体检测 → 关键点 → 骨架时序分类）
- 测试视频：`third_party/mmaction2/demo/demo_skeleton.mp4`（749 KB）
- label map：`third_party/mmaction2/tools/data/skeleton/label_map_ntu60.txt`（1.2 KB）
- 官方提供两种骨架时序模型示例：
  - **PoseC3D**（SlowOnly-R50 × NTU60-XSub-Keypoint）
  - **STGCN**（STGCN × NTU60-XSub-Keypoint-2D）

---

## 3. 待执行的复现命令（依赖 Python 3.10 环境）

### 3.1 复现 mmpose topdown demo（RTMDet-tiny + RTMPose）

```bash
cd third_party/mmpose

python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py \
    https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input demo/resources/demo.mp4 \
    --output-root vis_results/ \
    --save-predictions \
    --device cuda:0
```

预期输出：
- `vis_results/<视频名>.mp4`（叠加骨架的可视化视频）
- `vis_results/<视频名>.json`（关键点 JSON）

### 3.2 复现 mmaction2 demo_skeleton（PoseC3D 端到端）

```bash
cd third_party/mmaction2

python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
```

预期输出：`demo/demo_skeleton_out.mp4`（含动作类别标签的可视化视频）

### 3.3 复现 mmaction2 demo_skeleton（STGCN 端到端）

```bash
cd third_party/mmaction2

python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out_stgcn.mp4 \
    --config configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --checkpoint https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
```

---

## 4. 当前阻塞项与失败原因（如实记录）

### 4.1 阻塞 1：Python 3.12 与 mmcv 2.2.0 预编译包不兼容

**实测命令**：
```bash
pip install --dry-run "mmcv==2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

**实测结果**：
```
× Getting requirements to build wheel did not run successfully.
│ exit code: 1
```

**原因**：mmcv 2.2.0 在 cu121/torch2.4 索引下没有 cp312（Python 3.12）的预编译 wheel，pip 回退到源码编译，源码编译失败。

**解决方案**：必须新建 Python 3.10 conda 环境（见 `docs/env_compatibility_check.md`）。需用户授权安装 Miniconda。

### 4.2 阻塞 2：conda 未安装

- `conda --version` 返回未识别
- 影响：无法按 OpenMMLab 官方推荐方式创建独立 Python 3.10 环境
- 解决方案：安装 Miniconda（需用户授权）

### 4.3 未阻塞项

- ✅ 三个 OpenMMLab 仓库已克隆到 `third_party/`
- ✅ Demo 测试视频与 label map 文件已就位
- ✅ Demo 命令已在本文档第 3 节完整记录
- ✅ GMDCSA24 v2.1 数据集已下载并解压（可用于后续链路打通）

---

## 5. Demo 复现成功标准

执行第 3 节命令后，需满足以下条件才算复现成功：

| Demo | 必须产出 | 验证方法 |
|------|---------|---------|
| mmpose topdown | `vis_results/*.mp4` 视频文件 | 视频中人体上叠加骨架关键点 |
| mmpose topdown | `vis_results/*.json` 关键点文件 | JSON 含 17 个关键点坐标与置信度 |
| mmaction2 PoseC3D | `demo_skeleton_out.mp4` | 视频含动作类别标签 |
| mmaction2 STGCN | `demo_skeleton_out_stgcn.mp4` | 视频含动作类别标签 |

复现成功后需保存：
- 实际执行命令的完整 stdout/stderr 日志
- 输出视频截图
- 关键点 JSON 样本
- 失败时的错误堆栈

---

## 6. 下一步最小任务

1. **用户授权安装 Miniconda**（或选择 venv + Python 3.10 安装包方案）
2. 创建 `elderly-ai` conda 环境（Python 3.10）
3. 安装 PyTorch 2.4.x + CUDA 12.1 build
4. 安装 mmcv 2.2.0 预编译包（cu121/torch2.4/cp310）
5. 安装 mmdet 3.3.0、mmpose 1.3.0、mmaction2 1.2.0
6. 验证 `torch.cuda.is_available()` 返回 True
7. 执行第 3.1 节命令，复现 mmpose topdown demo
8. 执行第 3.2 节命令，复现 mmaction2 PoseC3D demo
9. 将输出视频与日志归档到 `experiments/logs/`

---

## 7. 参考项目 Demo 复现说明

### 7.1 RichardChen20/PoseFall（MIT）

- **不实际克隆运行**：作者声明代码未清理、不持续维护
- 仅借鉴：300 帧 2D 姿态序列 → GCN 二分类的整体思路
- 复现方式：通过 mmaction2 的 STGCN demo（第 3.3 节）等价验证

### 7.2 EltonCCL/Fall-Detection（无 LICENSE）

- **不克隆、不运行、不复制代码**：仓库无明确 LICENSE
- 仅借鉴：mmdet → mmpose → mmaction2 工程串联顺序
- 复现方式：通过 mmaction2 的 demo_skeleton.py（第 3.2/3.3 节）等价验证

---

## 8. Demo 复现实际执行结果（2026-07-29 更新）

> 环境：`elderly-ai` conda 环境（Python 3.10 + PyTorch 2.1.0 + cu121 + mmcv 2.1.0）
> 三个上游 Demo 均已成功复现，链路 `本地视频 → 人体检测 → 17 关键点 → 骨架时序分类 → 可视化视频` 已打通。

### 8.1 mmpose topdown demo（RTMDet-m + RTMPose-m）

- **状态**：✅ 成功
- **包装脚本**：`scripts/run_mmpose_demo_wrapper.py`
- **日志**：`experiments/logs/mmpose_topdown_demo.log`
- **输出**：`third_party/mmpose/vis_results/demo.mp4`（叠加 17 关键点骨架的可视化视频）+ `demo.json`（关键点坐标与置信度）
- **关键修复**：PowerShell stdout 缓冲导致进程假死，wrapper 脚本通过 subprocess 实时读取输出解决
- **复现意义**：验证 RTMDet + RTMPose 链路可用，可作为 `vision/input_adapter.py` 的核心组件

### 8.2 mmaction2 PoseC3D demo（SlowOnly-R50 端到端）

- **状态**：✅ 成功
- **包装脚本**：`scripts/run_mmaction_posec3d_wrapper.py`
- **日志**：`experiments/logs/mmaction_posec3d_demo.log`
- **输出**：`third_party/mmaction2/demo/demo_skeleton_out.mp4`（含动作类别标签的可视化视频）
- **关键修复**：mmaction2 缺失 `drn` 模块，从 `third_party/mmaction2` 复制 drn 目录并创建 `__init__.py`；`importlib_metadata` 包缺失，pip 安装补齐
- **复现意义**：验证 PoseC3D 骨架时序模型可用，可作为 fall/ADL 二分类的候选模型

### 8.3 mmaction2 STGCN demo（STGCN 端到端）

- **状态**：✅ 成功
- **包装脚本**：`scripts/run_stgcn_demo_fixed.ps1`（注：扩展名为 .ps1 但内容为 Python 脚本，用 elderly-ai 环境的 python.exe 执行）
- **日志**：`experiments/logs/mmaction_stgcn_demo.log`
- **输出**：`third_party/mmaction2/demo/demo_skeleton_out_stgcn.mp4`（263017 字节，含动作类别标签的可视化视频）
- **总耗时**：32.4 秒（含模型加载、人体检测、姿态估计、骨架绘制、视频合成）
- **关键修复（核心难点）**：
  - `import mmaction` 卡住超过 2 分钟，通过 `importtime` 日志定位到 `yapf_third_party._ylib2to3.pgen2.driver` 模块
  - 根因：YAPF 默认缓存目录 `C:\Users\John\AppData\Local\Google\YAPF\0.43.0` 被 TRAE Sandbox 限制访问
  - 解决方案：Monkey-patch `platformdirs.user_cache_dir` 和 `platformdirs.windows.Windows.user_cache_dir`，将 YAPF 缓存重定向到项目内 `.cache/yapf` 目录；同时通过 `LOCALAPPDATA` 环境变量重定向到 `.cache/appdata`，确保 subprocess 子进程也使用项目内目录
- **复现意义**：验证 STGCN 骨架时序模型可用，与 PoseC3D 形成双模型候选，可作为跌倒识别的主力模型

### 8.4 复现成功后链路打通情况

| 链路环节 | 组件 | 状态 | 验证依据 |
|---------|------|------|---------|
| 本地视频输入 | `demo_skeleton.mp4` / `demo.mp4` | ✅ | 三个 demo 均成功读取 |
| 人体检测 | Faster R-CNN R50 FPN / RTMDet | ✅ | 72 帧均检测到人体 |
| 17 关键点提取 | HRNet-W32 / RTMPose-m | ✅ | JSON 含 17 关键点坐标 |
| 骨架时序分类 | PoseC3D / STGCN | ✅ | 输出动作类别标签 |
| 可视化视频合成 | moviepy | ✅ | 三个输出视频均可正常播放 |

**结论**：第一步"克隆并跑通上游 OpenMMLab 官方 Demo"任务完成，可进入第二步"实现 vision/input_adapter.py 等核心模块"阶段。
