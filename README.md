# Look Model：老年人多模态风险监测

面向居家与养老照护场景的可复现研究/比赛原型。仓库将骨架时序跌倒预判、可靠性门控、连续事件解码、萤石设备适配与低负担心理变化筛查组合为可审计链路。

> 安全边界：这不是医疗器械。跌倒发布权重仍需在真实场景完成阈值校准；心理模块仅用于自愿、低频、非诊断性变化提示，默认不触发外部告警。

## 仓库结构

| 目录 | 用途 |
| --- | --- |
| `risk/phase_model/` | RG-PCNet / PA-DTSF 骨架时序模型、校准、事件解码和连续评估 |
| `mental/` | PACE-WB 与文本心理筛查的研究性、人工复核链路 |
| `devices/`、`vision/` | 萤石流、摄像头、姿态与雷达适配 |
| `alerts/`、`pipeline/`、`core/` | 风险事件、可靠性门控、告警分发和运行编排 |
| `configs/` | 模型、训练和筛查配置 |
| `scripts/` | 数据准备、训练、评估、绘图和实时运行入口 |
| `tests/` | 单元、集成和连续评估回归测试 |
| `docs/` | 部署、演示、评估与提交材料 |

## 快速开始

建议使用 Python 3.10+ 和 Conda：

```powershell
conda env create -f environment.yml
conda activate elderly-ai
python -m pytest tests/risk/phase_model tests/integration -q
```

原始视频、设备令牌、抓帧、训练缓存和大部分运行输出都不会上传。只保留具有明确说明与版本边界的发布权重及其元数据。

## 跌倒预判与实时运行

已随仓库保留的研究发布包：

- `outputs/releases/padtfs-gmdcsa24-gpu-norm/`：骨架跌倒阶段模型、锁定数据划分与指标；
- 留出测试：32 个片段，ROC-AUC `0.836`、Precision `0.800`、Recall `0.750`、F1 `0.774`；
- `metrics.json` 明确标记 `promoted=false`，因此不应将这些数字称为真实家庭或临床效果。

本地视频或授权播放地址的实时冒烟：

```powershell
python scripts/run_live_monitor.py `
  --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt `
  --input <local-video.mp4-or-authorized-stream-url> `
  --device auto `
  --smoke-seconds 10 `
  --no-browser
```

从冻结测试划分生成 ROC/PR、校准、混淆矩阵和阈值图：

```powershell
python scripts/plot_phase_results.py `
  --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt `
  --device auto `
  --output-dir docs/figures/padtfs-gmdcsa24-gpu-norm
```

训练只接受已经提取好的姿态特征与冻结划分，不会把原始视频静默混入发布数据：

```powershell
python scripts/train_phase_model.py `
  --dataset-lock <dataset_lock.json> `
  --split-manifest <split_manifest.json> `
  --data-root <feature-root> `
  --output <run-output> `
  --release-id <release-id> `
  --epochs 10 --lr-scheduler cosine
```

## 心理变化筛查

心理通道只接受老人自愿填写的问答或明确同意的文本记录。它会输出趋势、质量状态与人工复核建议；`research_only`、`external_dispatch_allowed=false` 或可靠性不足时一律弃权，不会生成短信、电话或设备播报。

```powershell
python scripts/train_wellbeing_shadow.py --input <consented-checkins.jsonl> --output-dir outputs/mental/shadow-run
python scripts/evaluate_wellbeing_shadow.py --artifact <shadow_model.joblib> --input <held-out-checkins.jsonl> --output <evaluation.json>
```

`outputs/releases/eatd-text-screening-research-only/` 是文本研究基线：验证 AUC `0.670`、F1 `0.300`，仅供复现和后续改进，不可用于诊断或自动报警。

## 萤石设备接入

设备凭据必须写在本机 `.env` 或系统环境变量中，严禁提交：

```powershell
python scripts/probe_ezviz_devices.py --write-report outputs/device-capability-live.md
python scripts/probe_ezviz_stream.py --frames 3
python scripts/run_live_monitor.py --checkpoint <checkpoint.pt> --input <authorized-stream-url> --device auto
```

播放 URL 往往包含短期令牌，不能写进 README、日志截图或 Git 提交。设备能力不足、码流过期、无人像或模型窗口不足时，系统应输出质量原因或 `abstained`，而不是伪造跌倒告警。

## 证据、部署与开发

- 部署与演示：[部署说明](docs/deployment.md)、[演示脚本](docs/demo-script.md)
- 数据/模型边界：[数据集说明](datasets/README.md)、[模型说明](models/README.md)、[模型权重说明](docs/model-weights.md)
- 已提交材料：[算法验证摘要](docs/submission/algorithm_validation_summary.md)、[设备验证协议](docs/submission/device_validation_protocol.md)
- 测试：`python -m pytest tests -q`

如需网页端、FastAPI 服务和萤石桥接，请使用配套 `elderly-care` 应用工程；本仓库专注于算法、设备适配、评估与可复现发布。
