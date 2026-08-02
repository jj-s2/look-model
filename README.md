# Look Model

面向老年人居家场景的多模态 AI 研究原型。目前重点包括视频人体姿态分析、跌倒/日常活动识别、步态稳定性评估，以及萤石设备输入适配。

## 主要模块

- `vision/`：视频、摄像头与萤石流输入适配。
- `risk/`：步态稳定性和跌倒风险规则。
- `fusion/`：多模态融合模块预留目录。
- `radar/`：雷达与生理数据接入预留目录。
- `scripts/`：数据处理、训练、评估和端到端运行脚本。
- `configs/`：OpenMMLab、PoseC3D 等实验配置。
- `tests/`：基础自动化测试。
- `docs/`：方案、进度和兼容性文档。

## 环境

推荐使用 Conda：

```powershell
conda env create -f environment.yml
conda activate elderly-ai
```

也可根据 `requirements.txt` 在 Python 3.10 环境中安装依赖。

## 快速检查

```powershell
python scripts/test_cuda.py
python -m pytest tests
```

端到端入口为 `scripts/run_pipeline.py`。运行前请根据本机目录、模型权重和输入设备调整配置。

## 数据与模型

原始数据、处理后的序列化数据、模型权重、训练检查点和设备凭据不会提交到 Git。协作者应按照 `datasets/README.md`、`models/README.md` 及相关脚本自行准备。

## 说明

本仓库为研究与比赛原型。第三方组件及许可证信息见 `THIRD_PARTY_NOTICES.md`。
