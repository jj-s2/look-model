# 上游项目审查记录 (Upstream Project Review)

> 本文档记录本项目所有候选上游仓库的许可证、最后提交时间、依赖版本、可运行入口、可复用模块、不能直接采用的部分以及最终选择理由。
>
> 所有信息均通过实际访问 GitHub/Zenodo 页面获取，未伪造任何字段。访问日期：2026-07-28。

---

## 1. 候选仓库总览

| # | 仓库 | 用途定位 | LICENSE | 最后提交 | 是否采用 |
|---|------|---------|---------|---------|---------|
| 1 | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | 人体检测底座 | Apache-2.0 | 2024-02-05 | ✅ 采用 |
| 2 | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | 姿态估计底座 | Apache-2.0 | 2025-08-04 | ✅ 采用 |
| 3 | [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) | 骨架时序动作识别底座 | Apache-2.0 | 2026-03-18 | ✅ 采用 |
| 4 | [RichardChen20/PoseFall](https://github.com/RichardChen20/PoseFall) | GCN 跌倒检测基线参考 | MIT | 2023-02-11 | ⚠️ 仅借鉴思想 |
| 5 | [EltonCCL/Fall-Detection](https://github.com/EltonCCL/Fall-Detection) | 工程流程参考 | **无 LICENSE 文件** | 2025-06-05 | ⚠️ 仅思路参考，**不得复制代码** |

---

## 2. 详细审查记录

### 2.1 open-mmlab/mmdetection

| 字段 | 内容 |
|------|------|
| 仓库地址 | https://github.com/open-mmlab/mmdetection |
| LICENSE | Apache-2.0（仓库根目录有 LICENSE 文件，README 明确声明） |
| 最后提交时间 | 2024-02-05（commit cfd5d3a，"Release MM-GroundingDINO SwinB and SwinL Weights"） |
| 最新版本 | v3.3.0（2024-01-05） |
| 主要语言 | Python（基于 PyTorch） |
| 依赖 | PyTorch 1.8+，mmcv-full，mmengine |
| 可运行入口 | `demo/image_demo.py`、`demo/video_demo.py` |
| 可复用模块 | RTMDet 系列轻量检测器配置；COCO 预训练 person 检测权重 |
| 不能直接采用部分 | 默认配置覆盖 80 类 COCO，需精简为 person 单类 |
| 最终选择理由 | Apache-2.0 适合后续孵化；RTMDet-tiny 在 RTX 4060 上推理速度与精度平衡良好；与 mmpose/mmaction2 同生态无版本冲突 |

### 2.2 open-mmlab/mmpose

| 字段 | 内容 |
|------|------|
| 仓库地址 | https://github.com/open-mmlab/mmpose |
| LICENSE | Apache-2.0（页面顶部标签，README 明确声明） |
| 最后提交时间 | 2025-08-04（commit 759b39c） |
| 最新版本 | v1.3.2（2024-07-12） |
| 主要语言 | Python 83.6% |
| 依赖 | PyTorch 1.8+，mmcv，mmengine |
| 可运行入口 | `demo/topdown_demo_with_mmdet.py`、`demo/bottomup_demo.py` |
| 可复用模块 | RTMPose-s/m COCO 17 关键点预训练权重；top-down 推理 pipeline |
| 不能直接采用部分 | 默认配置覆盖多种姿态任务，需精简为 COCO 17 类 top-down |
| 最终选择理由 | Apache-2.0；RTMPose 是当前 2D 姿态 SOTA 轻量模型；17 关键点格式与 COCO/NTU/本项目的 feature_extractor 完全兼容；输出格式稳定可序列化为 .npz |

### 2.3 open-mmlab/mmaction2

| 字段 | 内容 |
|------|------|
| 仓库地址 | https://github.com/open-mmlab/mmaction2 |
| LICENSE | Apache-2.0 |
| 最后提交时间 | 2026-03-18（commit a5a167d） |
| 最新版本 | v1.2.0（2023-10-12） |
| 主要语言 | Python |
| 依赖 | PyTorch 1.8+，mmcv，mmengine，decord |
| 可运行入口 | `demo/demo_skeleton.py`、`demo/demo.py` |
| 可复用模块 | PoseC3D、STGCN++ 配置与训练 pipeline；骨架格式 `pose_data.pkl` 规范 |
| 不能直接采用部分 | 默认配置针对 NTU-60/120 与 Kinetics，需修改 num_classes=2（fall/adl）、窗口长度、数据加载器 |
| 最终选择理由 | Apache-2.0；与 mmdet/mmpose 同生态；骨架时序模型（PoseC3D/STGCN++）是当前主流方案；用户明确要求只选一个主时序模型，避免同时开发 GRU/TCN/ST-GCN/Transformer |

### 2.4 RichardChen20/PoseFall

| 字段 | 内容 |
|------|------|
| 仓库地址 | https://github.com/RichardChen20/PoseFall |
| LICENSE | MIT |
| 最后提交时间 | 2023-02-11（commit 4e759ed） |
| 主要语言 | Python 79.1%，CMake 20.9% |
| 作者声明 | README 明确表示不打算详细更新此仓库 |
| 可运行入口 | `main.py`（视频分支）、ROS 节点（单帧分支） |
| 可复用部分 | **仅借鉴数据流与实验思想**：取 300 帧 2D 姿态序列输入 GCN 做跌倒判定的整体思路 |
| 不能直接采用部分 | 作者声明代码未充分清理、不会持续维护；CMake 部分与 ROS 强耦合；不得作为最终工程直接使用 |
| 最终选择理由 | MIT 许可证宽松；其"300 帧 2D 姿态 → GCN 二分类"思路与本项目第五步方案高度一致；可参考其特征归一化与窗口划分策略，但代码层面不直接复制 |

### 2.5 EltonCCL/Fall-Detection

| 字段 | 内容 |
|------|------|
| 仓库地址 | https://github.com/EltonCCL/Fall-Detection |
| LICENSE | **未找到**（仓库文件列表无 LICENSE 文件，README 全文未提及许可证，页面侧栏无 license 标签） |
| 最后提交时间 | 2025-06-05（commit e07a6d4） |
| 主要语言 | Python 94.5% |
| 项目性质 | UROP 本科生研究项目 |
| 评估结论 | 在 UR Fall Detection 与 Le2i 数据集上评估了 ST-GCN、2s-AGCN、PoseC3D、STGCN++ 四种模型，PoseC3D F1 最高，STGCN++ 计算开销最低 |
| 可复用部分 | **仅工程流程思路参考**：mmdet → mmpose → mmaction2 的串联顺序；模型选型对比数据 |
| 不能直接采用部分 | **因无明确 LICENSE，不得复制任何代码**；其工程串联方式将通过自行编写适配器实现 |
| 最终选择理由 | 作为本项目工程流程的参考来源（验证 mmdet+mmpose+mmaction2 串联方案的可行性），但代码层面零采纳；本项目将基于 OpenMMLab 官方文档自行实现串联 |

---

## 3. 许可证风险汇总

| 仓库 | 许可证 | 风险等级 | 风险说明 |
|------|--------|---------|---------|
| mmdetection | Apache-2.0 | 低 | 允许商业使用与修改，仅需保留版权声明 |
| mmpose | Apache-2.0 | 低 | 同上 |
| mmaction2 | Apache-2.0 | 低 | 同上 |
| PoseFall | MIT | 低 | 宽松许可，但作者声明不维护，代码质量风险 |
| Fall-Detection | **无** | **高** | 法律风险最高，**禁止代码复制** |
| ~~Ultralytics YOLO~~ | AGPL-3.0 | **高** | 已排除，不得引入 |

**总体结论**：主技术底座完全采用 OpenMMLab 生态（Apache-2.0），适合后续项目孵化与发布。Ultralytics 已排除。EltonCCL/Fall-Detection 因无许可证仅作思路参考。

---

## 4. 数据集来源审查

### 4.1 GMDCSA24（第一批，强制下载）

| 字段 | 内容 |
|------|------|
| GitHub 仓库 | https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos |
| Zenodo DOI | 10.5281/zenodo.12921216（实际指向 v2.0） |
| GitHub Release v2.1 | 2024-08-21，release notes 说明 CSV 文件已修改，新增类别与起止时间字段 |
| v2.1 Zenodo 链接 | **未找到**——README 仅给出 v2.0 的 Zenodo DOI，v2.1 release 未提供新链接 |
| 文件大小（v2.0） | 约 1.1 GB |
| 受试者数量 | 4 名（Subject 1/2/3/4） |
| Zenodo LICENSE | CC-BY 4.0 |
| GitHub LICENSE | MIT |
| 论文 | Alam et al., 2024, "GMDCSA24: A Dataset for Human Fall Detection in Videos", Data in Brief (communicated) |
| 用途 | 第一批主训练数据，按受试者留一交叉验证 |

**v2.1 问题处置方案**：
1. 下载阶段优先尝试访问 Zenodo record 12921216 的版本列表，查找 v2.1 链接
2. 若 Zenodo 仅有 v2.0，则下载 v2.0 并在 `datasets/manifest.json` 中明确标注版本
3. 若 GitHub Release v2.1 提供了 source code zip，可考虑作为补充
4. 不擅自将 v2.0 当作 v2.1 标注

### 4.2 CAUCAFall（第二批，链路跑通后下载）

- 来源：Mendeley Data 官方版本
- 用途：遮挡、光照变化、家庭环境、多种跌倒角度的外部测试
- 下载时机：第一链路跑通后再下载

### 4.3 UP-Fall 3D Skeletons（第三批，可选）

- 来源：Zenodo
- 大小：约 4 MB
- 用途：骨架时序分类快速调试
- 限制：5 名受试者的 3D 骨架 CSV，不得与 2D RTMPose 关键点不经转换直接混合训练

---

## 5. 引用信息

```bibtex
@article{alam2024,
  title={GMDCSA24: A Dataset for Human Fall Detection in Videos},
  author={Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco and Hameed, I.A.},
  year={2024},
  journal={Data in Brief}
}

@dataset{ekram_alam_2024_12921216,
  author = {Ekram Alam},
  title = {{ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos: 2.0}},
  month = jul,
  year = 2024,
  publisher = {Zenodo},
  version = {v2.0},
  doi = {10.5281/zenodo.12921216},
  url = {https://doi.org/10.5281/zenodo.12921216}
}
```

OpenMMLab 项目引用请参见各自仓库 README 的 Citation 段落。

---

## 6. 审查结论与下一步

1. **技术底座确定**：mmdetection + mmpose + mmaction2，全部 Apache-2.0
2. **不引入 Ultralytics**：避免 AGPL-3.0 影响后续孵化
3. **EltonCCL/Fall-Detection 仅作思路参考**：无 LICENSE，零代码采纳
4. **PoseFall 仅借鉴实验思想**：MIT 但作者不维护，不作最终工程
5. **数据集优先级**：GMDCSA24 v2.0/v2.1 → CAUCAFall → UP-Fall 3D
6. **v2.1 链接待补**：下载阶段再次访问 Zenodo 确认
