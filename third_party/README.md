# third_party - 上游开源仓库

本目录保存本项目使用的上游 OpenMMLab 仓库源码，用于：
- 查阅 demo 入口与配置文件
- 离线调试与版本核对
- 作为修改参考（本项目通过自定义 config 与适配器扩展，不直接修改上游源码）

> 这些目录通过 `git clone --depth 1` 克隆，已通过 `.gitignore` 排除入库。本目录本身不入 Git，仅保留说明文档。

## 已克隆仓库

| 仓库 | 路径 | commit | 日期 | LICENSE |
|------|------|--------|------|---------|
| mmpose | `mmpose/` | 759b39c | 2025-08-04 | Apache-2.0 |
| mmdetection | `mmdetection/` | cfd5d3a | 2024-02-05 | Apache-2.0 |
| mmaction2 | `mmaction2/` | a5a167d | 2026-03-18 | Apache-2.0 |

## 重新克隆命令

```bash
cd third_party
git clone --depth 1 https://github.com/open-mmlab/mmpose.git
git clone --depth 1 https://github.com/open-mmlab/mmdetection.git
git clone --depth 1 https://github.com/open-mmlab/mmaction2.git
```

## 修改原则

- **不修改上游源码**
- 通过以下方式扩展：
  - 自定义 config 文件（放在本项目的 `configs/` 目录）
  - 通过 Python 继承与适配器模式扩展模块（放在本项目的 `vision/` 目录）
  - 独立项目模块（如 `vision/input_adapter.py`、`vision/feature_extractor.py`）

## 参考项目（未克隆，仅文档参考）

| 仓库 | LICENSE | 处理方式 |
|------|---------|---------|
| RichardChen20/PoseFall | MIT | 仅借鉴实验思想，不克隆 |
| EltonCCL/Fall-Detection | 无 | 无许可证，不克隆、不复制代码 |

详见 `docs/upstream_review.md` 与 `THIRD_PARTY_NOTICES.md`。
