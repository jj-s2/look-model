# Elderly Fall and Mental Health Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 PoseC3D、RTMDet、RTMPose 和步态分析代码上，交付一个可用萤石 C6c 与 SDNL1 运行、能区分跌倒事件/跌倒风险/心理健康变化且默认低打扰的本地比赛原型。

**Architecture:** 系统采用本地异步流水线和决策级晚融合。摄像头、雷达和问卷各自产生统一事件，视觉跌倒状态机、个体步态基线和心理趋势模块独立输出证据，融合层只负责风险分级、解释和告警去重；任何单一模态不可用时明确降低数据质量，而不是伪造数据或中断全系统。

**Tech Stack:** Python 3.10、PyTorch/OpenMMLab（RTMDet、RTMPose、PoseC3D）、OpenCV、NumPy/SciPy、scikit-learn、requests、Gradio、pytest、JSONL。

## Global Constraints

- 使用仓库现有 `.conda` Python 3.10/CUDA 环境；先做兼容性冒烟测试，确认必要前不升级 PyTorch/OpenMMLab。
- 所有萤石凭据、设备序列号和验证码只放在 `.env`，日志和异常中必须脱敏；仓库只提交 `.env.example`。
- C6c 的对讲能力通过运行时能力查询确认。连续视频可用于视觉分析，但不做连续录音；比赛版语音提示默认由电脑扬声器播放。
- 当前设备未连接；所有单元测试、集成测试和演示默认使用离线视频/JSONL 夹具，不得要求设备在线。真实 C6c/SDNL1 探测仅作为设备接通后的可选验收步骤，未接通时报告 `unavailable` 而不是失败或伪造在线状态。
- SDNL1 只消费实际返回且有单位、时间戳的字段。接口不可用时输出 `unavailable`，演示数据必须带 `demo=true`。
- 心理健康模块只输出筛查和变化风险，不做医学诊断。完整 GDS-15 的主动邀请间隔不少于 28 天，短问候不少于 7 天，21:00–08:00 不主动打扰；跌倒后只允许一次即时确认。
- 个体生理和步态基线需要至少 7 个有效自然日；异常日不回写基线。
- 视频默认只实时处理不落盘。用户明确开启事件回放后，保存跌倒前 10 秒和后 20 秒，默认保留 7 天。
- 验收下限：跌倒 F1 不低于 0.90、召回率不低于 0.88且不低于现有基线；端到端告警 P95 不高于 2 秒；日常活动误报目标低于 1 次/小时。
- 不让老年人执行真实跌倒。跌倒测试使用公开数据、年轻志愿者受控演示或离线回放。

---

### Task 1: 建立可迁移配置和基础测试入口

**Files:**
- Create: `core/__init__.py`
- Create: `core/settings.py`
- Create: `.env.example`
- Create: `pytest.ini`
- Create: `tests/test_settings.py`
- Modify: `paths.py`
- Modify: `scripts/run_pipeline.py`
- Modify: `scripts/test_input_adapter.py`
- Modify: `scripts/validate_pre_fall.py`

**Interfaces:**
- `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings`
- `Settings.redacted() -> dict[str, object]`
- `PROJECT_ROOT: pathlib.Path`

- [ ] **Step 1: 写出失败测试**

```python
# tests/test_settings.py
from pathlib import Path

from core.settings import Settings


def test_project_root_is_derived_from_repository():
    settings = Settings.from_env({})
    assert (settings.project_root / "paths.py").exists()


def test_secret_values_are_redacted():
    settings = Settings.from_env({
        "EZVIZ_APP_KEY": "app-key",
        "EZVIZ_APP_SECRET": "app-secret",
    })
    exported = str(settings.redacted())
    assert "app-key" not in exported
    assert "app-secret" not in exported
    assert "***" in exported


def test_explicit_cache_path_overrides_default(tmp_path: Path):
    settings = Settings.from_env({"MODEL_CACHE_DIR": str(tmp_path)})
    assert settings.model_cache_dir == tmp_path.resolve()
```

- [ ] **Step 2: 运行测试并确认失败原因正确**

Run: `.\.conda\python.exe -m pytest tests/test_settings.py -q`

Expected: FAIL，提示 `core.settings` 不存在。

- [ ] **Step 3: 实现最小配置对象并消除所有旧绝对路径**

```python
# core/settings.py
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_cache_dir: Path
    output_dir: Path
    ezviz_app_key: str | None
    ezviz_app_secret: str | None
    ezviz_device_serial: str | None
    ezviz_device_code: str | None
    ezviz_stream_url: str | None
    sdnl1_data_url: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        path = lambda key, default: Path(values.get(key, default)).expanduser().resolve()
        return cls(
            project_root=PROJECT_ROOT,
            model_cache_dir=path("MODEL_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "models")),
            output_dir=path("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")),
            ezviz_app_key=values.get("EZVIZ_APP_KEY"),
            ezviz_app_secret=values.get("EZVIZ_APP_SECRET"),
            ezviz_device_serial=values.get("EZVIZ_DEVICE_SERIAL"),
            ezviz_device_code=values.get("EZVIZ_DEVICE_CODE"),
            ezviz_stream_url=values.get("EZVIZ_STREAM_URL"),
            sdnl1_data_url=values.get("SDNL1_DATA_URL"),
        )

    def redacted(self) -> dict[str, object]:
        data = dict(self.__dict__)
        for key in ("ezviz_app_key", "ezviz_app_secret", "ezviz_device_serial", "ezviz_device_code"):
            if data[key]:
                data[key] = "***"
        return data
```

把 `paths.py` 改为从 `core.settings.PROJECT_ROOT` 派生；三个脚本都接受命令行参数并以仓库根目录为默认值，不再出现 `F:\look model` 或桌面旧路径。`.env.example` 只列变量名和安全说明，不写真实值。

- [ ] **Step 4: 验证配置测试与环境冒烟测试**

Run: `.\.conda\python.exe -m pytest tests/test_settings.py -q`

Expected: PASS。

Run: `.\.conda\python.exe -c "import torch,cv2,mmengine; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"`

Expected: 命令正常结束并记录版本；若 CUDA 为 `False`，计划仍继续但在运行说明中标为 CPU 降级模式。

- [ ] **Step 5: 提交本任务**

```powershell
git add core/__init__.py core/settings.py .env.example pytest.ini tests/test_settings.py paths.py scripts/run_pipeline.py scripts/test_input_adapter.py scripts/validate_pre_fall.py
git commit -m "chore: make project configuration portable"
```

---

### Task 2: 建立统一事件契约和本地事件存储

**Files:**
- Create: `core/events.py`
- Create: `core/event_store.py`
- Create: `tests/test_events.py`
- Create: `tests/test_event_store.py`

**Interfaces:**
- `SensorEvent.to_dict() -> dict[str, object]`
- `SensorEvent.from_dict(data: Mapping[str, object]) -> SensorEvent`
- `JsonlEventStore.append(event: SensorEvent) -> None`
- `JsonlEventStore.query(start: datetime, end: datetime, event_types: set[EventType] | None = None) -> list[SensorEvent]`

- [ ] **Step 1: 写序列化和查询失败测试**

```python
# tests/test_events.py
from datetime import datetime, timezone
from core.events import DataQuality, EventType, SensorEvent, Source


def test_event_round_trip_preserves_quality_and_payload():
    event = SensorEvent(
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        source=Source.VISION,
        event_type=EventType.FALL_EVENT,
        payload={"probability": 0.91},
        quality=DataQuality(available=True, confidence=0.88, demo=False),
    )
    assert SensorEvent.from_dict(event.to_dict()) == event
```

```python
# tests/test_event_store.py
from datetime import datetime, timedelta, timezone
from core.event_store import JsonlEventStore
from core.events import DataQuality, EventType, SensorEvent, Source


def test_query_filters_time_and_type(tmp_path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    now = datetime.now(timezone.utc)
    store.append(SensorEvent(now, Source.VISION, EventType.FALL_EVENT, {}, DataQuality(True, 1.0, False)))
    store.append(SensorEvent(now, Source.RADAR, EventType.PHYSIOLOGY, {}, DataQuality(False, 0.0, False)))
    result = store.query(now - timedelta(seconds=1), now + timedelta(seconds=1), {EventType.FALL_EVENT})
    assert [item.event_type for item in result] == [EventType.FALL_EVENT]
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/test_events.py tests/test_event_store.py -q`

Expected: FAIL，提示事件模块不存在。

- [ ] **Step 3: 实现带时区校验的事件模型**

```python
# core/events.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping


class Source(str, Enum):
    VISION = "vision"
    RADAR = "radar"
    SCREENING = "screening"
    SYSTEM = "system"


class EventType(str, Enum):
    POSE = "pose"
    FALL_EVENT = "fall_event"
    FALL_FORECAST = "fall_forecast"
    PHYSIOLOGY = "physiology"
    WELLBEING_CHANGE = "wellbeing_change"
    AVAILABILITY = "availability"


@dataclass(frozen=True)
class DataQuality:
    available: bool
    confidence: float
    demo: bool
    reason: str | None = None


@dataclass(frozen=True)
class SensorEvent:
    timestamp: datetime
    source: Source
    event_type: EventType
    payload: Mapping[str, object]
    quality: DataQuality

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not 0.0 <= self.quality.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
```

`JsonlEventStore` 使用 UTF-8、一行一个 JSON、写入后 `flush`；读取时跳过空行，但遇到结构损坏必须抛出包含行号且不含敏感值的异常。

- [ ] **Step 4: 运行测试**

Run: `.\.conda\python.exe -m pytest tests/test_events.py tests/test_event_store.py -q`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```powershell
git add core/events.py core/event_store.py tests/test_events.py tests/test_event_store.py
git commit -m "feat: add typed sensor event contract"
```

---

### Task 3: 接入萤石鉴权、设备探测和直播地址

**Files:**
- Create: `devices/__init__.py`
- Create: `devices/models.py`
- Create: `devices/ezviz_client.py`
- Create: `scripts/probe_ezviz_devices.py`
- Create: `tests/devices/test_ezviz_client.py`

**Interfaces:**
- `EzvizClient.get_access_token() -> AccessToken`
- `EzvizClient.list_devices() -> list[EzvizDevice]`
- `EzvizClient.get_live_address(device_serial: str, channel_no: int = 1) -> str`
- `EzvizDevice.talk_mode: Literal["none", "full_duplex", "half_duplex", "unknown"]`

- [ ] **Step 1: 用伪造 HTTP 会话写失败测试**

```python
# tests/devices/test_ezviz_client.py
from devices.ezviz_client import EzvizClient


def test_token_request_uses_form_body_and_never_repr_secret(fake_session):
    fake_session.queue({"code": "200", "data": {"accessToken": "token", "expireTime": 1999999999999}})
    client = EzvizClient("key", "secret", session=fake_session)
    token = client.get_access_token()
    assert token.value == "token"
    assert fake_session.last_call["data"] == {"appKey": "key", "appSecret": "secret"}
    assert "secret" not in repr(client)


def test_live_address_is_extracted_from_success_response(fake_session):
    fake_session.queue({"code": "200", "data": {"url": "ezopen://example/live"}})
    client = EzvizClient("key", "secret", session=fake_session, access_token="token")
    assert client.get_live_address("SERIAL", 1) == "ezopen://example/live"
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/devices/test_ezviz_client.py -q`

Expected: FAIL，提示 `devices.ezviz_client` 不存在。

- [ ] **Step 3: 实现客户端和脱敏异常**

固定基础地址 `https://open.ys7.com`，鉴权接口使用官方 `POST /api/lapp/token/get`，直播地址接口隔离为 `POST /api/lapp/v2/live/address/get`。所有响应先检查 `code == "200"`；错误对象只包含萤石错误码、错误消息和接口名，不包含请求体。

```python
class EzvizClient:
    def __init__(self, app_key, app_secret, session=None, access_token=None, timeout=10.0): ...
    def get_access_token(self) -> AccessToken: ...
    def list_devices(self) -> list[EzvizDevice]: ...
    def get_live_address(self, device_serial: str, channel_no: int = 1) -> str: ...
```

设备能力以 API 原始字段为准；`support_talk` 的已知值映射为 `0=none`、`1=full_duplex`、`3=half_duplex`，字段缺失时为 `unknown`。探测脚本只打印设备型号、在线状态、通道数、对讲模式和序列号末四位。

官方核对来源：

- 鉴权：`https://open.ys7.com/api/lapp/token/get`
- 设备能力字段：`https://open.ys7.com/doc/zh/pc/group__data.html`
- 对讲模式定义：`https://open.ys7.com/doc/zh/ios/interface_e_z_device_info.html`

- [ ] **Step 4: 运行单测和无凭据安全分支**

Run: `.\.conda\python.exe -m pytest tests/devices/test_ezviz_client.py -q`

Expected: PASS。

Run: `.\.conda\python.exe scripts/probe_ezviz_devices.py`

Expected: 未配置 `.env` 时给出缺失变量名并以退出码 2 结束，不打印任何环境变量值；设备未连接时允许使用 `--offline-fixture` 输出明确 `unavailable` 状态。

- [ ] **Step 5: 提交本任务**

```powershell
git add devices scripts/probe_ezviz_devices.py tests/devices/test_ezviz_client.py
git commit -m "feat: add secure EZVIZ device client"
```

---

### Task 4: 完成可重连的 C6c 视频输入适配器

**Files:**
- Modify: `vision/input_adapter.py`
- Create: `vision/stream_health.py`
- Create: `tests/vision/test_input_adapter.py`

**Interfaces:**
- `EzvizStreamAdapter(url_provider: Callable[[], str], capture_factory: Callable[[str], VideoCapture] = cv2.VideoCapture, max_retries: int = 5)`
- `EzvizStreamAdapter.read() -> tuple[bool, np.ndarray | None]`
- `EzvizStreamAdapter.health -> StreamHealth`

- [ ] **Step 1: 写断流重连和释放资源测试**

```python
def test_ezviz_adapter_refreshes_url_after_read_failure(fake_capture_factory):
    urls = iter(["url-1", "url-2"])
    adapter = EzvizStreamAdapter(
        url_provider=lambda: next(urls),
        capture_factory=fake_capture_factory.with_sequences([
            [(False, None)],
            [(True, FRAME)],
        ]),
        backoff_seconds=(0.0,),
    )
    ok, frame = adapter.read()
    assert ok is True
    assert frame is FRAME
    assert fake_capture_factory.opened_urls == ["url-1", "url-2"]


def test_release_is_idempotent(fake_capture_factory):
    adapter = EzvizStreamAdapter(lambda: "url", fake_capture_factory)
    adapter.release()
    adapter.release()
    assert fake_capture_factory.release_count == 1
```

- [ ] **Step 2: 运行并确认现有 `NotImplementedError`**

Run: `.\.conda\python.exe -m pytest tests/vision/test_input_adapter.py -q`

Expected: FAIL，原因指向现有 `EzvizStreamAdapter` 未实现。

- [ ] **Step 3: 实现健康状态和有上限的指数退避**

`StreamHealth` 包含 `state`（`connecting/healthy/degraded/offline/closed`）、连续失败数、最后成功时间和脱敏原因。读取失败时释放旧句柄、刷新直播地址后重连；退避序列为 0.5、1、2、4、8 秒并封顶，超过次数返回 `(False, None)`，不无限阻塞主循环。

- [ ] **Step 4: 运行测试**

Run: `.\.conda\python.exe -m pytest tests/vision/test_input_adapter.py -q`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```powershell
git add vision/input_adapter.py vision/stream_health.py tests/vision/test_input_adapter.py
git commit -m "feat: implement resilient EZVIZ stream adapter"
```

---

### Task 5: 将现有视觉推理脚本拆成可测试流水线

**Files:**
- Create: `vision/pose_pipeline.py`
- Create: `vision/pose_buffer.py`
- Create: `tests/vision/test_pose_pipeline.py`
- Modify: `scripts/run_pipeline.py`

**Interfaces:**
- `PoseFrameResult(frame_index, timestamp, bboxes, keypoints, keypoint_scores)`
- `PosePipeline.process(frame: np.ndarray, timestamp: datetime) -> PoseFrameResult`
- `PoseSequenceBuffer.append(result: PoseFrameResult) -> list[PoseFrameResult] | None`

- [ ] **Step 1: 用假检测器和假姿态器写失败测试**

```python
def test_pipeline_passes_only_person_boxes_to_pose_estimator():
    detector = FakeDetector([
        Detection("person", 0.9, [1, 2, 30, 40]),
        Detection("chair", 0.8, [5, 6, 20, 25]),
    ])
    pose = FakePoseEstimator()
    result = PosePipeline(detector, pose, person_threshold=0.5).process(FRAME, NOW)
    assert pose.last_boxes == [[1, 2, 30, 40]]
    assert result.timestamp == NOW


def test_sequence_buffer_emits_fixed_length_overlapping_window():
    buffer = PoseSequenceBuffer(window_size=4, stride=2)
    emitted = [buffer.append(frame(i)) for i in range(6)]
    assert [len(x) for x in emitted if x is not None] == [4, 4]
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/vision/test_pose_pipeline.py -q`

Expected: FAIL，提示新模块不存在。

- [ ] **Step 3: 提取协议并延迟导入重型依赖**

检测器和姿态器通过 `Protocol` 注入，单测不加载模型；OpenMMLab 的初始化只在工厂函数中发生。`run_pipeline.py` 复用 `PosePipeline`，保留现有 JSON 和标注视频输出格式，并补充 UTC 时间戳、模型版本和输入源质量。

- [ ] **Step 4: 单测与离线视频冒烟测试**

Run: `.\.conda\python.exe -m pytest tests/vision/test_pose_pipeline.py tests/test_scripts.py -q`

Expected: PASS。

Run: `.\.conda\python.exe scripts/run_pipeline.py --help`

Expected: 显示 `--input`、`--output-dir`、`--device`，不加载模型、不访问网络。

- [ ] **Step 5: 提交本任务**

```powershell
git add vision/pose_pipeline.py vision/pose_buffer.py tests/vision/test_pose_pipeline.py scripts/run_pipeline.py
git commit -m "refactor: extract testable pose inference pipeline"
```

---

### Task 6: 实现带迟滞的跌倒事件状态机

**Files:**
- Create: `risk/fall_state_machine.py`
- Create: `tests/risk/test_fall_state_machine.py`

**Interfaces:**
- `FallObservation(timestamp, fall_probability, gait_risk, torso_angle_deg, vertical_speed, ground_ratio, keypoint_quality)`
- `FallStateMachine.update(observation: FallObservation) -> FallDecision`
- 状态：`normal`、`unstable`、`descending`、`on_ground`、`recovered`

- [ ] **Step 1: 写正常弯腰、真实跌倒序列和低质量关键点测试**

```python
def test_brief_bending_does_not_emit_fall_event(machine):
    decisions = [machine.update(obs(t, p=0.2, angle=70, speed=0.1, ground=0.2)) for t in range(6)]
    assert not any(item.confirmed_fall for item in decisions)


def test_descent_followed_by_ground_persistence_emits_once(machine):
    sequence = [
        obs(0, p=0.25, angle=10, speed=0.1, ground=0.1),
        obs(1, p=0.80, angle=45, speed=1.1, ground=0.3),
        obs(2, p=0.92, angle=80, speed=1.4, ground=0.8),
        obs(3, p=0.90, angle=85, speed=0.1, ground=0.9),
        obs(4, p=0.88, angle=86, speed=0.0, ground=0.9),
    ]
    decisions = [machine.update(item) for item in sequence]
    assert sum(item.confirmed_fall for item in decisions) == 1
    assert "ground_persistence" in decisions[-1].reasons


def test_low_keypoint_quality_cannot_confirm_fall(machine):
    decision = machine.update(obs(0, p=0.99, angle=90, speed=2.0, ground=1.0, quality=0.1))
    assert decision.confirmed_fall is False
    assert decision.data_quality == "degraded"
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/risk/test_fall_state_machine.py -q`

Expected: FAIL，提示状态机模块不存在。

- [ ] **Step 3: 实现阈值配置、持续时间和单次事件锁存**

视觉模型概率阈值从现有验证结果的 0.75 起步，但所有阈值写入 `FallThresholds`，不散落在代码中。进入高风险状态需连续证据；确认跌倒需下降证据后在地持续；恢复需连续站立证据。每次状态转换返回原因码和用于复现实验的证据值。

- [ ] **Step 4: 运行测试**

Run: `.\.conda\python.exe -m pytest tests/risk/test_fall_state_machine.py -q`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```powershell
git add risk/fall_state_machine.py tests/risk/test_fall_state_machine.py
git commit -m "feat: add explainable fall event state machine"
```

---

### Task 7: 改造尺度无关步态特征和七日个体基线

**Files:**
- Modify: `risk/gait_stability.py`
- Create: `risk/personal_baseline.py`
- Create: `tests/risk/test_gait_stability.py`
- Create: `tests/risk/test_personal_baseline.py`

**Interfaces:**
- `GaitStabilityAnalyzer.extract_features(sequence, frame_size) -> GaitFeatures`
- `RobustPersonalBaseline.add_day(day: date, features: Mapping[str, float], valid: bool) -> None`
- `RobustPersonalBaseline.score(features: Mapping[str, float]) -> BaselineScore`

- [ ] **Step 1: 写尺度不变性、冷启动和异常日隔离测试**

```python
def test_gait_features_are_invariant_to_uniform_pixel_scaling():
    original = analyzer.extract_features(KEYPOINTS, (640, 480))
    scaled = analyzer.extract_features(KEYPOINTS * 2.0, (1280, 960))
    assert scaled.sway == pytest.approx(original.sway, rel=0.05)
    assert scaled.step_variability == pytest.approx(original.step_variability, rel=0.05)


def test_baseline_is_not_ready_before_seven_valid_days():
    baseline = RobustPersonalBaseline(min_valid_days=7)
    for day in DAYS[:6]:
        baseline.add_day(day, {"sway": 0.2}, valid=True)
    assert baseline.ready is False


def test_anomalous_day_does_not_shift_reference_distribution():
    baseline = ready_baseline()
    before = baseline.reference["sway"]
    baseline.add_day(DAYS[7], {"sway": 99.0}, valid=False)
    assert baseline.reference["sway"] == before
```

- [ ] **Step 2: 运行并确认现有像素尺度逻辑失败**

Run: `.\.conda\python.exe -m pytest tests/risk/test_gait_stability.py tests/risk/test_personal_baseline.py -q`

Expected: FAIL；尺度测试暴露现有原始像素特征差异，基线模块不存在。

- [ ] **Step 3: 使用躯干长度和画面尺寸归一化**

特征至少包含身体中心晃动、步宽、步频稳定性、左右对称性、躯干角度变化和关键点质量。个体参考采用中位数与 IQR；IQR 过小时设置数值下限。冷启动阶段只显示群体规则风险并标记 `baseline_ready=false`。

- [ ] **Step 4: 运行测试和现有预跌倒验证脚本**

Run: `.\.conda\python.exe -m pytest tests/risk/test_gait_stability.py tests/risk/test_personal_baseline.py -q`

Expected: PASS。

Run: `.\.conda\python.exe scripts/validate_pre_fall.py --help`

Expected: 正常显示参数，不读取数据集。

- [ ] **Step 5: 提交本任务**

```powershell
git add risk/gait_stability.py risk/personal_baseline.py tests/risk/test_gait_stability.py tests/risk/test_personal_baseline.py
git commit -m "feat: add scale-invariant gait baseline"
```

---

### Task 8: 建立可复现的预跌倒学习模型与受试者级评估

**Files:**
- Create: `risk/prefall_model.py`
- Create: `risk/prefall_evaluation.py`
- Create: `scripts/train_prefall_model.py`
- Create: `scripts/evaluate_prefall.py`
- Create: `tests/risk/test_prefall_model.py`
- Create: `tests/risk/test_prefall_evaluation.py`

**Interfaces:**
- `PrefallModel.fit(features: DataFrame, labels: Series) -> PrefallModel`
- `PrefallModel.predict_proba(features: DataFrame) -> np.ndarray`
- `evaluate_subject_wise(features, labels, subject_ids) -> EvaluationReport`

- [ ] **Step 1: 写列顺序、数据泄漏和模型晋级规则测试**

```python
def test_model_rejects_missing_or_reordered_features(trained_model):
    with pytest.raises(ValueError, match="feature schema"):
        trained_model.predict_proba(FRAME.drop(columns=["sway"]))


def test_subjects_never_cross_train_and_validation_folds():
    folds = make_subject_folds(SUBJECT_IDS)
    for train_idx, valid_idx in folds:
        assert set(SUBJECT_IDS[train_idx]).isdisjoint(SUBJECT_IDS[valid_idx])


def test_candidate_is_not_promoted_when_recall_regresses():
    assert should_promote(candidate={"f1": .92, "recall": .84}, baseline={"f1": .90, "recall": .88}) is False
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/risk/test_prefall_model.py tests/risk/test_prefall_evaluation.py -q`

Expected: FAIL，提示模型和评估模块不存在。

- [ ] **Step 3: 实现轻量模型、元数据和晋级门槛**

第一版使用 `StandardScaler + LogisticRegression(class_weight="balanced")`，模型文件同时保存特征顺序、训练数据摘要、随机种子、阈值和指标。评估使用 `GroupKFold`；数据集受试者少于折数时自动降低折数但不少于 2。只有验证集 F1、召回率均不低于规则基线且满足全局验收线时，脚本才把模型标为 `promoted=true`。

- [ ] **Step 4: 运行测试与小型合成训练**

Run: `.\.conda\python.exe -m pytest tests/risk/test_prefall_model.py tests/risk/test_prefall_evaluation.py -q`

Expected: PASS。

Run: `.\.conda\python.exe scripts/train_prefall_model.py --synthetic-smoke-test --output-dir outputs/smoke/prefall`

Expected: 生成模型、`metrics.json` 和 `model_card.json`，不覆盖正式模型。

- [ ] **Step 5: 提交本任务**

```powershell
git add risk/prefall_model.py risk/prefall_evaluation.py scripts/train_prefall_model.py scripts/evaluate_prefall.py tests/risk/test_prefall_model.py tests/risk/test_prefall_evaluation.py
git commit -m "feat: add leakage-safe prefall model evaluation"
```

---

### Task 9: 接入 SDNL1 生理数据并明确真实/演示/不可用状态

**Files:**
- Create: `radar/base.py`
- Create: `radar/ezviz_source.py`
- Create: `radar/demo_jsonl.py`
- Create: `radar/features.py`
- Create: `tests/radar/test_sources.py`
- Create: `tests/radar/test_features.py`
- Modify: `radar/__init__.py`

**Interfaces:**
- `PhysiologySource.poll(start: datetime, end: datetime) -> PhysiologyBatch`
- `PhysiologyRecord(timestamp, heart_rate, respiratory_rate, sleep_stage, source_fields)`
- `summarize_daily(records: Sequence[PhysiologyRecord]) -> DailyPhysiologySummary`

- [ ] **Step 1: 写缺失接口、演示标记和非法单位测试**

```python
def test_real_source_reports_unavailable_without_configured_endpoint():
    batch = EzvizPhysiologySource(data_url=None, client=CLIENT).poll(START, END)
    assert batch.quality.available is False
    assert batch.quality.demo is False
    assert batch.quality.reason == "sdnl1_api_not_configured"


def test_demo_source_marks_every_batch_as_demo(tmp_path):
    source = DemoJsonlPhysiologySource(write_fixture(tmp_path))
    assert source.poll(START, END).quality.demo is True


def test_out_of_range_measurement_is_excluded_and_counted():
    summary = summarize_daily([record(heart_rate=400), record(heart_rate=72)])
    assert summary.valid_heart_rate_count == 1
    assert summary.invalid_measurement_count == 1
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/radar/test_sources.py tests/radar/test_features.py -q`

Expected: FAIL，提示雷达模块不存在。

- [ ] **Step 3: 实现能力优先的数据源协议**

真实源只解析配置映射中明确声明的字段名、单位和时间戳格式；未知字段保留在 `source_fields` 供审计但不参与风险计算。日摘要包含静息心率中位数、呼吸率中位数、睡眠时长、夜间觉醒次数、有效覆盖率和质量原因。缺少官方健康数据接口时，UI 明确显示“设备在线但健康数据接口未接通”。

- [ ] **Step 4: 运行测试**

Run: `.\.conda\python.exe -m pytest tests/radar/test_sources.py tests/radar/test_features.py -q`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```powershell
git add radar tests/radar
git commit -m "feat: add auditable SDNL1 physiology sources"
```

---

### Task 10: 实现 GDS-15、心理趋势和低打扰交互策略

**Files:**
- Create: `mental/__init__.py`
- Create: `mental/gds15.py`
- Create: `mental/trend.py`
- Create: `mental/interaction_policy.py`
- Create: `configs/screening/gds15_zh.json`
- Create: `tests/mental/test_gds15.py`
- Create: `tests/mental/test_trend.py`
- Create: `tests/mental/test_interaction_policy.py`

**Interfaces:**
- `GDS15.score(answers: Mapping[str, bool]) -> GDS15Result`
- `WellbeingTrendAnalyzer.update(day: date, physiology: DailyPhysiologySummary | None, activity: ActivitySummary | None, checkin: CheckinResult | None) -> TrendResult`
- `InteractionPolicy.evaluate(context: InteractionContext) -> InteractionDecision`

- [ ] **Step 1: 写计分、趋势和频率上限测试**

```python
def test_gds_requires_exactly_fifteen_answers(scale):
    with pytest.raises(ValueError, match="15 answers"):
        scale.score({"q01": True})


def test_gds_score_uses_each_items_risk_answer(scale):
    answers = {item.id: item.risk_answer for item in scale.items}
    assert scale.score(answers).score == 15
    assert scale.score(answers).is_diagnosis is False


def test_full_screening_is_not_invited_twice_within_28_days(policy):
    context = context_at("2026-08-02T10:00:00+08:00", last_full_screening="2026-07-10T10:00:00+08:00")
    assert policy.evaluate(context).invite_full_gds is False


def test_quiet_hours_suppress_non_emergency_prompt(policy):
    context = context_at("2026-08-02T22:00:00+08:00", sustained_change=True)
    decision = policy.evaluate(context)
    assert decision.prompt is None
    assert decision.reason == "quiet_hours"


def test_confirmed_fall_allows_one_immediate_check(policy):
    first = policy.evaluate(context_at("2026-08-02T22:00:00+08:00", confirmed_fall=True))
    second = policy.evaluate(context_at("2026-08-02T22:01:00+08:00", confirmed_fall=True, fall_check_already_sent=True))
    assert first.prompt == "fall_confirmation"
    assert second.prompt is None
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/mental -q`

Expected: FAIL，提示心理模块不存在。

- [ ] **Step 3: 导入公版量表并实现非诊断性输出**

`configs/screening/gds15_zh.json` 必须包含 15 个条目的稳定 ID、中文题目、是/否选项、风险答案和来源元数据。逐条转录并双人核对 Stanford 列出的中文版；文件记录来源 URL 和访问日期：

- `https://web.stanford.edu/~yesavage/GDS`
- `https://web.stanford.edu/~yesavage/Chinese.html`
- `https://web.stanford.edu/~yesavage/Australian%20Chinese.pdf`

量表结果只返回分数、风险区间、完成时间、版本和“筛查不是诊断”的固定说明。趋势模块用至少 7 日基线比较活动量、睡眠规律和用户自愿回答；单日偏差只能产生观察证据，连续变化才允许邀请短问候。

`InteractionPolicy` 固定执行：完整量表 28 天冷却、短问候 7 天冷却、21:00–08:00 安静时段、跌倒后一次确认、用户主动发起不受主动邀请冷却限制。任何用户自发表达的自伤/轻生意图直接生成人工复核最高优先级事件，不由 GDS 分数替代。

- [ ] **Step 4: 运行测试和量表资源完整性校验**

Run: `.\.conda\python.exe -m pytest tests/mental -q`

Expected: PASS。

Run: `.\.conda\python.exe -c "from mental.gds15 import GDS15; s=GDS15.from_json('configs/screening/gds15_zh.json'); print(s.version, len(s.items))"`

Expected: 打印版本和 `15`。

- [ ] **Step 5: 提交本任务**

```powershell
git add mental configs/screening/gds15_zh.json tests/mental
git commit -m "feat: add low-burden wellbeing screening"
```

---

### Task 11: 实现多模态晚融合、分级告警和去重

**Files:**
- Create: `fusion/decision_engine.py`
- Create: `alerts/__init__.py`
- Create: `alerts/models.py`
- Create: `alerts/dispatcher.py`
- Create: `tests/fusion/test_decision_engine.py`
- Create: `tests/alerts/test_dispatcher.py`
- Modify: `fusion/__init__.py`

**Interfaces:**
- `DecisionEngine.evaluate(events: Sequence[SensorEvent], now: datetime) -> list[RiskDecision]`
- `RiskDecision(kind, level, score, reasons, quality, recommended_action)`
- `AlertDispatcher.dispatch(decision: RiskDecision) -> DispatchResult`

- [ ] **Step 1: 写独立输出、缺失模态和告警去重测试**

```python
def test_fall_event_and_wellbeing_change_remain_separate(engine):
    decisions = engine.evaluate([CONFIRMED_FALL, SUSTAINED_WELLBEING_CHANGE], NOW)
    assert {item.kind for item in decisions} == {"fall_event", "wellbeing_change"}


def test_missing_radar_lowers_quality_but_does_not_block_fall(engine):
    decisions = engine.evaluate([CONFIRMED_FALL, RADAR_UNAVAILABLE], NOW)
    fall = next(item for item in decisions if item.kind == "fall_event")
    assert fall.level == "critical"
    assert fall.quality == "vision_only"


def test_same_fall_is_dispatched_only_once_within_cooldown(dispatcher):
    assert dispatcher.dispatch(FALL_DECISION).sent is True
    assert dispatcher.dispatch(FALL_DECISION).sent is False
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/fusion tests/alerts -q`

Expected: FAIL，提示融合和告警模块不存在。

- [ ] **Step 3: 实现四级风险和证据解释**

风险级别为 `info/watch/warning/critical`。确认跌倒可直接为 `critical`；跌倒趋势与心理变化始终是不同决策。生理异常只能增强心理变化证据，不能单独给出心理诊断。每项决策至少包含两个可读原因字段：触发证据和缺失/降级证据。

第一版 dispatcher 写入本地 JSONL 并在 UI 显示；可选 webhook 只有在用户配置地址后才启用。去重键由事件类型、跟踪对象和时间窗口构成，确认恢复后才能开启下一次跌倒事件。

- [ ] **Step 4: 运行测试**

Run: `.\.conda\python.exe -m pytest tests/fusion tests/alerts -q`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```powershell
git add fusion alerts tests/fusion tests/alerts
git commit -m "feat: add explainable late fusion alerts"
```

---

### Task 12: 组装实时服务、隐私保留和本地演示界面

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/live_service.py`
- Create: `storage/__init__.py`
- Create: `storage/clip_buffer.py`
- Create: `storage/retention.py`
- Create: `ui/__init__.py`
- Create: `ui/dashboard.py`
- Create: `app.py`
- Create: `tests/pipeline/test_live_service.py`
- Create: `tests/storage/test_retention.py`
- Create: `tests/ui/test_dashboard_view_model.py`
- Modify: `requirements.txt`

**Interfaces:**
- `LiveMonitoringService.step() -> ServiceSnapshot`
- `CircularClipBuffer.on_frame(frame, timestamp) -> None`
- `CircularClipBuffer.confirm_event(event_id, post_seconds=20) -> Path | None`
- `RetentionPolicy.prune(now: datetime) -> list[Path]`
- `build_dashboard(service: LiveMonitoringService) -> gradio.Blocks`

- [ ] **Step 1: 写端到端单步、回放开关和保留期测试**

```python
def test_live_service_emits_decision_from_fake_sources(service):
    snapshot = service.step()
    assert snapshot.camera_health == "healthy"
    assert any(item.kind == "fall_event" for item in snapshot.decisions)


def test_clip_is_not_written_when_recording_opt_in_is_false(tmp_path):
    buffer = CircularClipBuffer(tmp_path, recording_opt_in=False, pre_seconds=10)
    buffer.on_frame(FRAME, NOW)
    assert buffer.confirm_event("event-1") is None
    assert list(tmp_path.iterdir()) == []


def test_retention_removes_only_expired_event_clips(tmp_path):
    expired, current = create_dated_clips(tmp_path)
    removed = RetentionPolicy(tmp_path, keep_days=7).prune(NOW)
    assert removed == [expired]
    assert current.exists()
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/pipeline tests/storage tests/ui -q`

Expected: FAIL，提示服务、存储和 UI 模块不存在。

- [ ] **Step 3: 实现非阻塞服务和比赛界面**

服务每次迭代有明确超时；摄像头、雷达或 UI 单个异常不能终止其他模块。界面只展示：设备在线/质量、实时画面、跌倒事件、跌倒趋势、心理健康变化趋势、证据说明、告警历史和用户主动发起的 GDS-15。心理区域固定显示“风险筛查，不构成医学诊断”。

在 `requirements.txt` 添加与 Python 3.10 兼容的 `gradio>=5,<7`。`app.py --demo-fixtures` 使用显著水印和 `demo=true` 事件；默认模式不自动回退到演示数据。

- [ ] **Step 4: 运行测试与无设备 UI 冒烟测试**

Run: `.\.conda\python.exe -m pytest tests/pipeline tests/storage tests/ui -q`

Expected: PASS。

Run: `.\.conda\python.exe app.py --demo-fixtures --no-browser --smoke-seconds 5`

Expected: 服务启动、处理演示事件、5 秒后干净退出，控制台不出现凭据或完整设备序列号。

- [ ] **Step 5: 提交本任务**

```powershell
git add pipeline storage ui app.py tests/pipeline tests/storage tests/ui requirements.txt
git commit -m "feat: assemble privacy-aware monitoring demo"
```

---

### Task 13: 完成真实设备试运行、性能评估和比赛交付文档

**Files:**
- Create: `tests/integration/test_monitoring_flow.py`
- Create: `scripts/benchmark_live_pipeline.py`
- Create: `scripts/generate_evaluation_report.py`
- Create: `docs/deployment.md`
- Create: `docs/device-capability-report.md`
- Create: `docs/evaluation-report.md`
- Create: `docs/demo-script.md`
- Modify: `README.md`

**Interfaces:**
- `benchmark_live_pipeline(input_source, duration_seconds) -> BenchmarkReport`
- `generate_evaluation_report(metrics_files: Sequence[Path]) -> str`

- [ ] **Step 1: 写完整降级流和报告门槛测试**

```python
def test_monitoring_flow_survives_radar_unavailable(fake_system):
    result = fake_system.run(events=[CAMERA_FALL_SEQUENCE, RADAR_UNAVAILABLE])
    assert result.exit_code == 0
    assert result.last_fall_decision.quality == "vision_only"


def test_evaluation_report_fails_release_gate_below_recall_threshold():
    with pytest.raises(ReleaseGateError, match="recall"):
        build_report({"fall_f1": 0.92, "fall_recall": 0.87, "p95_latency_seconds": 1.2})
```

- [ ] **Step 2: 运行并确认失败**

Run: `.\.conda\python.exe -m pytest tests/integration/test_monitoring_flow.py -q`

Expected: FAIL，提示集成夹具或报告生成器不存在。

- [ ] **Step 3: 实现基准脚本和可审计报告**

性能报告记录硬件、Python/CUDA/模型版本、数据集、阈值、随机种子、每类混淆矩阵、F1、精确率、召回率、误报/小时和端到端 P50/P95。真实设备能力报告记录 C6c 在线、直播、对讲模式和 SDNL1 数据字段是否可用，但序列号只保留末四位。

`README.md` 提供从复制 `.env.example`、设备探测、离线验证、实时启动到比赛演示的最短路径。`docs/demo-script.md` 明确演示数据水印、真实设备状态、跌倒后一次问询以及心理模块非诊断边界。

- [ ] **Step 4: 运行全部测试和发布门槛**

Run: `.\.conda\python.exe -m pytest -q`

Expected: PASS，无跳过的核心单元测试；需要真实设备的测试使用显式 `--run-device-tests` 才执行。

Run: `.\.conda\python.exe scripts/benchmark_live_pipeline.py --input datasets/evaluation/controlled_fall.mp4 --duration-seconds 300 --output outputs/benchmark.json`

Expected: 生成性能 JSON，P95 延迟不高于 2 秒；如果未达到，报告标红且不声称通过。

Run: `.\.conda\python.exe scripts/generate_evaluation_report.py --metrics outputs --output docs/evaluation-report.md`

Expected: 只有在 F1、召回率、延迟和误报率达到全局门槛时退出码为 0。

- [ ] **Step 5: 进行真实设备能力核验**

Run: `.\.conda\python.exe scripts/probe_ezviz_devices.py --offline-fixture --write-report docs/device-capability-report.md`

Expected: 在设备未连接时生成含 `unavailable` 状态的能力报告并正常退出；设备接通后再运行不带 `--offline-fixture` 的命令，C6c 的直播和对讲能力、SDNL1 数据接口状态才来自真实响应，缺失能力写明 `unavailable`，不以演示数据冒充。

- [ ] **Step 6: 提交本任务**

```powershell
git add tests/integration scripts/benchmark_live_pipeline.py scripts/generate_evaluation_report.py docs/deployment.md docs/device-capability-report.md docs/evaluation-report.md docs/demo-script.md README.md
git commit -m "docs: add verified deployment and evaluation workflow"
```

---

## Final Verification Gate

- [ ] 运行 `.\.conda\python.exe -m pytest -q` 并保存完整输出。
- [ ] 设备接通后再运行真实 C6c 直播至少 30 分钟，记录断流次数、自动恢复时间、平均 FPS 和 P95 延迟；设备未接通时使用离线回放并在报告中标记真实设备验证跳过。
- [ ] 用公开跌倒/日常活动数据执行受试者级评估，确认 F1、召回和误报率不低于全局门槛。
- [ ] 核对所有心理健康文案均使用“筛查、变化、建议人工关注”，不出现诊断结论。
- [ ] 搜索仓库确认没有真实 `appSecret`、access token、设备验证码、完整设备序列号或旧绝对路径。
- [ ] 在 SDNL1 真实接口不可用时，确认 UI、报告和事件都明确显示不可用；演示数据始终显示演示标记。
- [ ] 开启和关闭事件回放各测试一次，确认关闭时不落盘、开启时只保存 10 秒前和 20 秒后的事件片段并按 7 天清理。
- [ ] 按 `docs/demo-script.md` 完整彩排，禁止安排老年人真实跌倒。

## Fresh Reviewer Gate

每个任务提交后，先由独立代码审查检查需求符合性，再检查实现质量；发现问题时在进入下一任务前修复并重新运行该任务测试。最终审查同时对照设计规格 `docs/superpowers/specs/2026-08-02-elderly-fall-mental-health-design.md`、本计划和评估报告。
