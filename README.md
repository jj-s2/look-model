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

## 已发布模型的实时监测入口

当前仓库已经把发布的 PA-DTSF 阶段模型接入统一实时服务：输入帧先经过
YOLO11-pose 提取 17 点人体骨架，再进入短/长时间窗、质量门控、阶段预测、
告警去重和本地看板。模型权重位于
`outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt`。

先用本地视频做不打开浏览器的 GPU/CPU 冒烟运行（运行时长由参数限定）：

```powershell
python scripts/run_live_monitor.py `
  --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt `
  --input <local-video.mp4> `
  --device auto `
  --smoke-seconds 10 `
  --no-browser
```

需要本地 Gradio 看板时，去掉 `--no-browser`；输入可以是视频文件、摄像头编号
（例如 `0`），或已由萤石平台授权返回的播放地址。播放地址只在本机进程中使用，
不要把带令牌的完整地址写入日志、截图或 Git。`--help` 不会加载 Torch、YOLO、
Gradio 或设备 SDK。

当前发布检查点训练并验证的是长时骨架分支；实时适配器会对短时分支传入质量为
零的占位向量，因此界面会保留质量标记，不会把未训练的短时特征伪装成已验证能力。
后续若补齐短时嵌入，应单独训练、按受试者划分验证，并更新 release ID 与指标。

## 生成算法效果图

使用冻结的 test split 重新推理并生成 ROC/PR、混淆矩阵、阈值权衡、校准和得分分布图：

```powershell
python scripts/plot_phase_results.py `
  --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt `
  --device auto `
  --output-dir docs/figures/padtfs-gmdcsa24-gpu-norm
```

本次留出测试结果为 32 个片段、ROC-AUC 0.836、Precision 0.800、Recall 0.750、
F1 0.774。完整指标和图表见
`docs/figures/padtfs-gmdcsa24-gpu-norm/`；这些是数据集留出测试证据，不等同于临床或现场设备效果。

`Settings.from_env()` 会自动读取项目根目录的本地 `.env`（系统环境变量优先），因此已配置的萤石账号可以直接执行：

```powershell
python scripts/probe_ezviz_devices.py --write-report outputs/device-capability-live.md
python scripts/probe_ezviz_stream.py --frames 3 --save-first-frame outputs/ezviz-first-frame.jpg
# 激活赛事设备套餐（每个设备通道只执行成功一次）
python scripts/activate_ezviz_package.py --package-code <package-code>
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

原始数据、处理后的序列化数据和设备凭据不会提交到 Git。协作者应按照
`datasets/README.md`、`models/README.md` 及相关脚本自行准备。

为保证比赛演示可复现，以下经过明确边界标记的模型权重随仓库发布：

- 跌倒风险相位模型：`outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt`；
- EATD 文本心理筛查研究基线：
  `outputs/releases/eatd-text-screening-research-only/eatd_text_baseline.joblib`。

第二项只用于非诊断性、低频且自愿的研究筛查。其验证 F1 为 0.300、AUC 为
0.670，未达到可部署标准，始终保持 `promoted=false`；不得据此作医疗诊断、
自动报警，或对摄像头/日常录音直接推断心理状态。模型指标与哈希见该目录中的
`metrics.json` 和 `README.md`。

## 说明

本仓库为研究与比赛原型。第三方组件及许可证信息见 `THIRD_PARTY_NOTICES.md`。
