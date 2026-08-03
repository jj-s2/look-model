# Look Model

面向老年人居家场景的多模态 AI 研究与比赛原型。它提供跌倒风险、跌倒事件和身心状态变化的分层提示；不是医疗器械，心理模块只用于筛查、变化提示和建议人工关注，**不作诊断**。

## 主要模块

- `vision/`：视频、摄像头与萤石流输入适配。
- `risk/`：步态稳定性和跌倒风险规则。
- `fusion/`：多模态融合模块预留目录。
- `radar/`：雷达与生理数据接入预留目录。
- `scripts/`：数据处理、训练、评估和端到端运行脚本。
- `configs/`：OpenMMLab、PoseC3D 等实验配置。
- `tests/`：基础自动化测试。
- `docs/`：方案、进度和兼容性文档。

## 最短验证路径

```powershell
conda env create -f environment.yml
conda activate elderly-ai
Copy-Item .env.example .env
python scripts/probe_ezviz_devices.py --offline-fixture --write-report docs/device-capability-report.md
python -m pytest tests/integration/test_monitoring_flow.py -q
python scripts/run_pipeline.py --input <local-video.mp4>
```

`Settings.from_env()` 会自动读取项目根目录的本地 `.env`（系统环境变量优先），因此已配置的萤石账号可以直接执行：

```powershell
python scripts/probe_ezviz_devices.py --write-report outputs/device-capability-live.md
```

直播地址接口支持设备验证码以及 EZOPEN/HLS/RTMP/FLV 协议参数；若设备开启码流加密，萤石平台可能返回 60019，此时使用 EZOPEN 播放器或在设备设置中关闭码流加密后再取 HLS 帧。

`--offline-fixture` 不访问网络，故意显示 `unavailable`；它不是 C6c、直播、对讲或 SDNL1 的真实验证。真实设备接通后才可运行不带该参数的探测，缺失能力仍必须显示 `unavailable`，不可用演示数据替代。

性能与门槛（任一失败即非零退出）：

```powershell
python scripts/benchmark_live_pipeline.py --input <controlled-local-replay.mp4> --duration-seconds 300 --output outputs/benchmark.json
python scripts/generate_evaluation_report.py --metrics experiments/outputs/losocv/summary.json outputs/benchmark.json --output docs/evaluation-report.md
```

完整部署、设备能力、评估边界和比赛演示步骤见 [部署说明](docs/deployment.md)、[设备能力报告](docs/device-capability-report.md)、[评估报告](docs/evaluation-report.md) 和 [演示脚本](docs/demo-script.md)。

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
