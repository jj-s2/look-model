# Third-Party Notices

本项目使用以下开源上游项目。每个条目保留项目名称、许可证、引用方式与实际修改说明。

---

## 1. OpenMMLab 系列（Apache-2.0）

### 1.1 mmdetection
- 仓库：https://github.com/open-mmlab/mmdetection
- 许可证：Apache License 2.0
- 用途：人体检测（RTMDet-tiny）
- 引用：见仓库 README 的 Citation 段
- 实际修改：**未修改源码**，仅通过自定义 config 文件精简为 person 单类检测

### 1.2 mmpose
- 仓库：https://github.com/open-mmlab/mmpose
- 许可证：Apache License 2.0
- 用途：COCO 17 关键点姿态估计（RTMPose）
- 引用：见仓库 README 的 Citation 段
- 实际修改：**未修改源码**，仅通过自定义 config 与 `demo/topdown_demo_with_mmdet.py` 调用

### 1.3 mmaction2
- 仓库：https://github.com/open-mmlab/mmaction2
- 许可证：Apache License 2.0
- 用途：骨架时序动作识别（PoseC3D 或 STGCN++）
- 引用：见仓库 README 的 Citation 段
- 实际修改：**未修改源码**，通过自定义 config 修改 num_classes、窗口长度、数据加载器

---

## 2. 参考项目（仅借鉴思想，未复制代码）

### 2.1 RichardChen20/PoseFall
- 仓库：https://github.com/RichardChen20/PoseFall
- 许可证：MIT
- 用途：300 帧 2D 姿态 → GCN 跌倒检测的实验思想参考
- 实际采纳：仅参考其数据流设计与窗口划分策略，**未复制任何代码**

### 2.2 EltonCCL/Fall-Detection
- 仓库：https://github.com/EltonCCL/Fall-Detection
- 许可证：**无明确 LICENSE**
- 用途：mmdet → mmpose → mmaction2 工程串联流程参考
- 实际采纳：仅参考其工程串联顺序与模型选型对比结论，**因无许可证，未复制任何代码**

---

## 3. 数据集

### 3.1 GMDCSA24
- 来源：https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
- Zenodo：https://doi.org/10.5281/zenodo.12921216
- 许可证：CC-BY 4.0（Zenodo 数据集）
- 引用：见 `docs/upstream_review.md` 第 5 节

### 3.2 CAUCAFall（待下载）
- 来源：Mendeley Data 官方
- 用途：外部测试
- 许可证：待下载时确认

### 3.3 UP-Fall 3D Skeletons（可选）
- 来源：Zenodo
- 许可证：待下载时确认

---

## 4. 排除项

- **Ultralytics YOLO**：AGPL-3.0 许可证，可能影响后续项目孵化与发布方式，本项目不引入。
