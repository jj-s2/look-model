# 数据集说明

> 本目录保存数据集下载脚本、校验值与说明文档。
>
> **原始视频、压缩包、模型权重、关键点缓存一律不入库**（已写入 `.gitignore`），仅保存下载脚本、校验值、目录结构、少量无隐私测试样本和说明文档。

---

## 1. 数据集清单

| 名称 | 版本 | 大小 | 来源 | 许可证 | 用途 | 状态 |
|------|------|------|------|--------|------|------|
| GMDCSA24 | v2.0 | ~1.1 GB | [Zenodo 12921216](https://doi.org/10.5281/zenodo.12921216) | CC-BY 4.0 | 主训练（受试者留一） | pending |
| CAUCAFall | - | ~2 GB | Mendeley | 待确认 | 外部测试（链路跑通后下载） | deferred |
| UP-Fall 3D Skeletons | - | ~4 MB | Zenodo | 待确认 | 骨架时序调试（可选） | deferred |

完整元数据见 `manifest.json`。

---

## 2. 下载与校验

### 2.1 列出所有数据集状态

```bash
python scripts/download_datasets.py --list
```

### 2.2 下载 GMDCSA24

```bash
python scripts/download_datasets.py --dataset GMDCSA24
```

特性：
- 断点续传（HTTP Range）
- 失败重试（5 次，指数退避）
- 自动 SHA256 / MD5 校验
- 下载完成后回填 `manifest.json`

### 2.3 仅校验已下载文件

```bash
python scripts/download_datasets.py --dataset GMDCSA24 --verify-only
# 或
python scripts/verify_datasets.py
```

校验报告输出到 `experiments/logs/verification_report.json`。

---

## 3. 目录结构

```
datasets/
├── README.md                    # 本文件
├── manifest.json                # 数据集元数据与校验值清单
├── raw/                         # 原始视频与压缩包（不入库）
│   ├── GMDCSA24/
│   │   └── GMDCSA24_v2.0.zip
│   ├── CAUCAFall/
│   └── UP-Fall-3D-Skeletons/
├── processed/                   # 处理后的关键点 .npz（不入库）
├── annotations/                 # 标注文件（不入库）
├── splits/                      # 受试者级划分（入库 .csv）
└── metadata.csv                 # 数据集扫描结果（入库）
```

---

## 4. 数据集使用规则

### 4.1 受试者级划分（强制）

- **禁止**按视频随机划分训练/验证/测试集
- **必须**按受试者（subject_id）隔离
- GMDCSA24 受试者较少（4 名），使用**留一受试者交叉验证**（Leave-One-Subject-Out）
- CAUCAFall 采用**受试者级训练/验证/测试划分**
- 同一个人的相似视频不得同时进入训练集与测试集

### 4.2 标签统一

| 原始标签 | 统一标签 |
|---------|---------|
| fall / falling / fall_down | `fall` |
| walking / sitting / standing / bending / lying 等 | `adl` |
| near_fall / stumble / almost_fall | `near_fall`（可选） |

### 4.3 关键点序列格式

统一保存为 `.npz`，至少包含：

```python
{
    "keypoints": np.ndarray,    # shape: [T, 17, 2]，COCO 17 关键点
    "scores":    np.ndarray,    # shape: [T, 17]
    "bbox":      np.ndarray,    # shape: [T, 4]，x1y1x2y2
    "label":     str,           # "fall" / "adl" / "near_fall"
    "subject_id": str,
    "video_id":  str,
    "fps":       float,
    "source_dataset": str       # "GMDCSA24" / "CAUCAFall" / ...
}
```

### 4.4 跨数据集混合训练

- UP-Fall 3D 骨架 CSV **不得**与 2D RTMPose 关键点不经转换直接混合训练
- 若需混合，必须先做 3D→2D 投影或单独训练后做集成

---

## 5. 来源与镜像策略

### 5.1 优先官方源

- GMDCSA24：Zenodo 官方
- CAUCAFall：Mendeley 官方
- UP-Fall：Zenodo 官方

### 5.2 禁止使用的来源

- 来源不明的百度网盘
- Kaggle 重打包版本
- 个人网盘
- 未注明出处的 GitHub fork

### 5.3 官方源失效时的处置流程

1. **先报告**：在 `experiments/logs/verification_report.json` 中记录失效
2. **再提供镜像选项**：仅限以下两类
   - 数据集官方维护的镜像（如 Zenodo 的备份 DOI）
   - 论文作者明确授权的镜像
3. **校验镜像**：镜像必须能通过 `manifest.json` 中记录的 SHA256 / MD5 校验

---

## 6. GMDCSA24 v2.1 说明

- GitHub Release v2.1（2024-08-21）存在，release notes 说明 CSV 文件已修改
- README 中给出的 Zenodo DOI（10.5281/zenodo.12921216）实际指向 **v2.0**
- v2.1 的独立 Zenodo 链接**未找到**
- 处置方案：
  1. 下载阶段优先访问 Zenodo record 12921216 的版本列表
  2. 若 Zenodo 仅有 v2.0，则下载 v2.0 并在 `manifest.json` 中如实标注版本为 `v2.0`
  3. 不擅自将 v2.0 当作 v2.1 标注

---

## 7. 引用信息

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
