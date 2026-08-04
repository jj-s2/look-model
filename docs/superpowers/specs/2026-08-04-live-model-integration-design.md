# C6c 实时 PA-DTSF 推理闭环设计

## 目标

将已经训练并发布的 `padtfs-gmdcsa24-gpu-norm` 权重接入现有实时视觉链路，形成“萤石视频流 → 姿态窗口 → PA-DTSF 推理 → 时序平滑 → 风险决策 → 看板/告警”的可演示闭环，同时保持低质量弃权、隐私保护和心理健康模块的既有边界。

## 范围

本阶段只实现实时跌倒风险推理接入，不重新训练模型，不改变 SDNL1 数据接口，不实现语音问答，也不把筛查结果包装成医疗诊断。

当前发布权重的短时 512 维 embedding 尚未由真实 PoseC3D 特征生成，因此实时推理显式使用 `short_quality=0`、`long_quality=1` 的长时姿态降级模式；系统必须在模型元数据和日志中保留这一事实。

## 架构

```text
C6c EZVIZ playback URL
        │
        ▼
EzvizStreamAdapter / OpenCV
        │ BGR frames
        ▼
pose pipeline + tracker
        │ PoseObservation(timestamp, keypoints, scores, bbox)
        ▼
DualTimescaleBuffer
        │ short/long timestamp-aware windows
        ▼
TorchPhasePredictor
        │ normalized (64, 17, 3) long pose
        ▼
PhaseRiskService
        │ quality gate + phase smoother
        ▼
DecisionEngine
        ├── fall_event / prefall_warning / fall_forecast
        └── radar corroboration + wellbeing separation
        │
        ├── AlertDispatcher → local JSONL → optional EZVIZ/webhook delivery
        └── Gradio dashboard
```

## 组件与接口

### 1. TorchPhasePredictor

新增独立预测器模块，避免让 `PhaseRiskService` 依赖 PyTorch 细节。

建议接口：

```python
class TorchPhasePredictor:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "auto",
        model_version: str | None = None,
        embedding_version: str = "short_embedding_unavailable",
    ) -> None: ...

    def predict(self, window: DualWindow) -> PhaseModelOutput: ...
```

行为要求：

- `device="auto"` 优先选择 CUDA，CUDA 不可用时回退 CPU；显式设备不可用时抛出可读错误。
- 加载 checkpoint 时校验 `release_id`、模型维度和 state-dict 键；权重损坏或版本不匹配不得静默运行。
- 从 `window.long` 提取 64 个观察帧；每帧不足 17 个关键点时补零并降低质量。
- 使用既有 COCO-17 姿态归一化逻辑；不可用帧不伪造为高质量观察。
- 当前短分支使用全零 embedding 和 `short_quality=0`，长分支使用 `long_quality=1`。
- 将 logits 转换为合法概率，将最大阶段映射为 `Phase`，返回 `PhaseModelOutput`。
- 推理异常不在预测器内部生成跌倒事件；由上层服务的异常边界和质量门控处理。

### 2. 实时运行入口

新增一个明确的本地运行入口，接收：

- `--checkpoint`：模型权重路径；
- `--device`：`auto`、`cpu` 或 `cuda:0`；
- `--input`：本地视频、摄像头索引或 EZVIZ 播放地址；
- `--smoke-seconds`：限定运行时间，便于演示和测试；
- `--no-browser`：只运行服务，不启动 Gradio。

入口负责组装输入适配器、姿态管线、`DualTimescaleBuffer`、`TorchPhasePredictor`、`PhaseRiskService`、`LiveMonitoringService` 和 `AlertDispatcher`。设备凭据只能来自现有环境变量/配置，不写入日志或播放地址。

### 3. 错误与降级

- 视频流打不开、读取超时或重连失败：标记 camera degraded/offline，不生成确认跌倒。
- 姿态窗口不足：返回质量弃权事件，不调用模型或不触发升级告警。
- 模型加载失败：入口启动失败并给出检查点、设备和依赖提示；不能切换到随机权重。
- 单帧姿态异常：保留可审计的质量原因，继续后续帧处理。
- 告警发送失败：先保留本地 JSONL 记录，再记录 delivery failure，不阻塞监测循环。

## 测试策略

### 单元测试

- checkpoint 元数据和 state-dict 校验；
- CPU fallback 与显式 CUDA 不可用错误；
- 17 点姿态转 `(64,17,3)`、补零和归一化；
- logits 到 `PhaseModelOutput` 的概率和阶段映射；
- 窗口质量不足时的弃权行为；
- 短分支不可用时 `short_quality=0` 且不触发模型输入错误。

### 集成测试

- 使用小型伪 checkpoint 和固定姿态窗口，验证预测器输出能被 `PhaseRiskService` 消费；
- 使用离线视频/假输入适配器跑有限帧，验证完整事件链路；
- 验证 `DecisionEngine` 将预警、预测、确认跌倒和心理变化分开；
- 验证 `AlertDispatcher` 先写 JSONL、再调用可选 delivery，且冷却和恢复逻辑不回归。

### 验收标准

- `pytest -q` 全量通过；
- CPU 环境可执行离线 smoke test；
- GPU 环境可加载已发布 checkpoint 并输出 `PhaseModelOutput`；
- 离线视频至少能生成姿态、质量、风险决策和本地告警记录；
- 真实 C6c 验证只在设备在线、播放地址可用且用户主动运行时进行；没有真实设备证据时不得声称已完成现场验证。

## 非目标与后续工作

- 本阶段不实现 PoseC3D 短时 embedding；后续单独准备短分支特征、重新训练和评估。
- 本阶段不接管 SDNL1 的未公开健康数据 API；待字段映射和接口权限明确后再启用。
- 本阶段不加入持续语音询问；心理健康仍采用主动 GDS-15 和个人趋势基线。
- 本阶段不把公开模拟数据指标外推为真实老人或临床性能。
