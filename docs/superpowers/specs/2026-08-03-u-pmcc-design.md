# U-PMCC 个体化多模态时序跌倒风险设计

- 日期：2026-08-03
- 状态：设计已确认，待书面规格复核
- 适用仓库：`jj-s2/look-model`
- 目标分支：`codex/elderly-monitoring`
- 核心名称：U-PMCC（Uncertainty-aware Personalized Multimodal Change Chain）
- 中文名称：不确定性感知的个体化多模态时序风险链

## 1. 文档目的

本文是可直接交给编码模型执行的技术规格。它定义 U-PMCC 的研究边界、算法、数据契约、模块接口、训练评估规则、测试用例和验收门槛。

本阶段只实现算法、离线数据流程、评估和可审计输出，不以比赛大屏或现场展示为目标。C6c 和 SDNL1 尚未连接，因此真实设备验证不属于本阶段完成条件；系统必须保留以后接入真实设备的接口。

## 2. 背景与差异化

成熟养老监测方案已经覆盖多项常见能力：

- CarePredict 使用穿戴设备和环境位置数据学习个人日常活动模式，并对睡眠、活动、进食等变化提供提醒。
- Nobi 将跌倒检测与离床提醒、照明干预、双向沟通和跌倒原因分析结合。
- Vayyar Care 使用无接触毫米波感知，在暗光和潮湿环境中进行活动与跌倒监测，并强调隐私与护理系统集成。
- SafelyYou 将 AI 检测、事件视频复核、人工确认和跌倒原因分析组成闭环。

因此，下列内容不能单独作为本项目的核心创新：个人基线、被动活动监测、多模态加权、轻量 TCN、分级告警或隐私保护。它们是成熟方案已经采用或产品化的能力。

U-PMCC 的差异化聚焦于四点：

1. 将长期身心状态变化与短期步态退化建模为有时间顺序、可回溯的跨模态证据链。
2. 使用离散时间生存模型统一输出未来 24 小时、72 小时和 7 天累计风险，使不同预测窗口天然单调。
3. 同时输出风险、不确定区间、有效证据覆盖率和拒绝结论；数据不足时不制造高置信度判断。
4. 使用人工确认与干预结果形成审计闭环，但不允许未经确认的告警自动成为训练标签。

市场参考仅用于产品能力对照，不作为独立临床性能证据：

- CarePredict：https://shop.carepredict.com/
- Nobi：https://www.nobi.life/en_GB/product
- Vayyar Care：https://vayyar.com/wp-content/uploads/2023/10/Vayyar_Care_Product_Specs.pdf
- SafelyYou：https://www.safely-you.com/

## 3. 目标与非目标

### 3.1 目标

U-PMCC 应当：

1. 从视觉步态、睡眠、生理、活动和近跌倒事件生成统一日级观察记录。
2. 使用人群先验与个人稳健基线识别相对变化。
3. 识别预先声明的跨模态时间关联链，但不声称医学因果关系。
4. 输出 24 小时、72 小时和 7 天累计跌倒风险。
5. 输出置信区间、数据质量、主要证据、降级原因和模型来源。
6. 在缺少 PyTorch、缺少生理数据、设备离线或模型加载失败时安全退化。
7. 支持规则基线、CPU 默认模型和可选轻量 TCN 的独立评估。
8. 严格隔离真实数据、公开数据、合成数据和演示夹具的指标与模型晋级资格。

### 3.2 非目标

本阶段不做：

- 不进行抑郁症、焦虑症、认知障碍或其他疾病诊断。
- 不把睡眠或心理变化写成跌倒的确定原因。
- 不使用合成 SDNL1 数据声称真实预测性能或临床有效性。
- 不自动联系急救机构，不给出医疗处置指令。
- 不训练语音情绪识别或持续录音模型。
- 不要求老人或其他受试者实施真实危险跌倒。
- 不把比赛界面、动画或大屏作为算法验收条件。
- 不在没有配对纵向真实数据时把 TCN 晋级为默认发布模型。

## 4. 系统边界与数据流

系统保留现有三个独立输出：

- `fall_event`：已经发生或正在发生的跌倒事件，由现有实时状态机负责。
- `fall_forecast`：U-PMCC 输出的未来跌倒风险。
- `wellbeing_change`：睡眠、活动或量表变化提示，不作诊断。

U-PMCC 只负责 `fall_forecast`。它不得改变确认跌倒事件的紧急告警规则，也不得根据心理量表直接产生跌倒紧急告警。

```text
视觉骨架/步态窗口 ─> 日级步态摘要 ──────────┐
SDNL1/离线夹具 ────> 睡眠与生理日摘要 ─────┤
活动/近跌倒事件 ───> 行为日摘要 ───────────┼─> 日级观察校验
                                            │
                                            ├─> 人群先验+个人基线
                                            ├─> 变化事件
                                            ├─> 时间关联链
                                            ├─> 生存风险模型
                                            ├─> 不确定性与拒绝门控
                                            └─> fall_forecast + 证据账本
```

日级模型采用最近 14 个自然日作为输入窗口，预测未来第 1 至第 7 天的离散风险率。跨模态时间链默认窗口为 72 小时。

## 5. 数据来源与证据等级

每个数据记录和模型产物必须声明 `evidence_tier`：

| 等级 | 值 | 用途 | 可用于发布晋级 |
| --- | --- | --- | --- |
| 真实纵向设备数据 | `real_device_longitudinal` | 个体校准、真实预测评估 | 是 |
| 公开真实数据 | `real_public` | 视觉特征、跌倒/ADL 和可用的纵向任务评估 | 视任务匹配程度决定 |
| 合成研究数据 | `synthetic_research` | 契约、算法逻辑、压力和缺失测试 | 否 |
| 离线演示夹具 | `offline_fixture` | 冒烟、设备离线和流程测试 | 否 |

不同证据等级不得合并生成一个发布指标。所有评估 JSON 必须含相同的 `release_id`、`dataset_id`、`evidence_tier`、`subject_split_id` 和 `schema_version`，报告生成器发现不一致时必须失败。

合成数据可以运行训练代码和验证损失下降，但生成的模型必须写入：

```json
{
  "promoted": false,
  "synthetic": true,
  "not_for_clinical_performance": true
}
```

## 6. 核心数据契约

### 6.1 日级观察

建议在 `risk/pmcc/schema.py` 定义不可变数据类 `DailyObservation`：

```json
{
  "schema_version": "1.0",
  "subject_id": "anonymous-001",
  "date": "2026-08-03",
  "timezone": "Asia/Shanghai",
  "features": {
    "sleep_duration_minutes": 370.0,
    "night_awakenings": 4.0,
    "bed_exits": 3.0,
    "activity_minutes": 92.0,
    "sedentary_ratio": 0.64,
    "sit_to_stand_seconds": 4.8,
    "trunk_sway": 0.31,
    "gait_asymmetry": 0.22,
    "near_fall_count": 1.0,
    "heart_rate_median": 72.0,
    "respiration_rate_median": 16.0
  },
  "available": {
    "vision": true,
    "sleep": true,
    "physiology": false,
    "activity": true
  },
  "quality": {
    "vision": 0.90,
    "sleep": 0.85,
    "physiology": 0.0,
    "activity": 0.80
  },
  "provenance": {
    "evidence_tier": "synthetic_research",
    "dataset_id": "pmcc-synthetic-v1",
    "synthetic": true,
    "demo": false
  }
}
```

规则：

- `subject_id` 必须匿名化且非空。
- `date` 按记录时区解释，禁止使用无时区时间戳聚合日级数据。
- 缺失特征使用 `null` 或不提供字段，禁止填充为零。
- `available=false` 时相应质量必须为 `0.0`。
- 质量必须位于 `[0, 1]`，布尔值不得被当作数值。
- 未知特征可以保留在原始数据，但模型只读取版本化特征清单。

### 6.2 变化事件

`ChangeEvent` 至少包含：

```json
{
  "subject_id": "anonymous-001",
  "feature": "sleep_duration_minutes",
  "event_name": "sleep_duration_decreased",
  "start_date": "2026-08-01",
  "last_observed_date": "2026-08-03",
  "directional_z": 2.1,
  "persistence": 0.83,
  "quality": 0.85,
  "baseline_state": "personal",
  "evidence_ids": ["obs-001", "obs-002"]
}
```

### 6.3 时间关联链

`TemporalChain` 必须记录每个节点及时间差，不得只保留最终分数：

```json
{
  "chain_type": "sleep_to_activity_to_gait",
  "nodes": [
    "sleep_duration_decreased",
    "activity_minutes_decreased",
    "trunk_sway_increased"
  ],
  "gaps_hours": [22.0, 25.0],
  "score": 0.72,
  "quality": 0.81,
  "association_only": true
}
```

### 6.4 风险输出

`PMCCForecast` 必须包含：

```json
{
  "schema_version": "1.0",
  "subject_id": "anonymous-001",
  "generated_at": "2026-08-03T12:00:00+08:00",
  "cumulative_risk": {
    "24h": 0.24,
    "72h": 0.46,
    "7d": 0.61
  },
  "uncertainty": {
    "method": "bootstrap",
    "lower_72h": 0.31,
    "upper_72h": 0.58,
    "width_72h": 0.27
  },
  "decision": "watch",
  "forecast_band": "elevated",
  "abstained": false,
  "quality": 0.78,
  "coverage": 0.75,
  "baseline_state": "personal",
  "model_kind": "rule_survival_calibrator",
  "promoted": false,
  "reasons": [
    "近3天睡眠时长较个人基线下降",
    "睡眠变化后活动量下降",
    "躯干摆动增加"
  ],
  "chain_ids": ["chain-001"],
  "provenance": {
    "release_id": "pmcc-2026-08-03",
    "evidence_tier": "synthetic_research"
  }
}
```

风险必须满足：`24h <= 72h <= 7d`。输出理由必须能够追溯到观察记录或变化事件。

## 7. 个体化基线

### 7.1 冷启动状态

基线采用三级状态，不在第 7 天突然切换：

| 有效日数 | 状态 | 行为 |
| ---: | --- | --- |
| 0–6 | `population_only` | 只使用人群先验；长期预测标记低置信度 |
| 7–13 | `blended` | 人群先验与个人统计量按有效日数平滑混合 |
| 14 及以上 | `personal` | 个人稳健基线为主，人群先验只用于尺度下限 |

每个特征单独计算有效日数。不能因为睡眠有 14 天数据，就声称步态基线也已就绪。

### 7.2 稳健标准化

对特征 `i`：

```text
center_i = median(valid_baseline_values_i)
scale_i = max(1.4826 * MAD(valid_baseline_values_i), feature_floor_i)
directional_z_i,t = clip(risk_direction_i * (x_i,t - center_i) / scale_i, -5, 5)
```

`risk_direction` 由版本化配置给出，例如睡眠时长减少和离床次数增加都映射为正风险方向。没有方向定义的特征不得进入风险模型。

### 7.3 基线更新保护

以下日期不进入基线更新：

- 已确认跌倒或近跌倒日及其后 3 天；
- 数据质量低于 `0.5` 的日期；
- 设备离线导致模态不完整的日期；
- 被明确标记为住院、旅行或环境改变的日期；
- 综合异常分数超过训练配置阈值的日期。

基线更新使用固定长度 30 个有效日窗口。所有排除原因必须记录，不得静默丢弃。

## 8. 变化事件与时间关联链

### 8.1 变化事件规则

默认在最近 3 个有效日中至少 2 日满足 `directional_z >= 1.5` 才生成变化事件。严重变化阈值默认为 `2.5`。阈值必须放入配置和模型卡，不得散落在代码中。

持续性定义为最近 3 个有效日中达到阈值的质量加权比例。缺失日不视为正常日。

### 8.2 允许的关联边

第一版只支持以下预注册边：

```text
sleep_disruption -> activity_decrease
sleep_disruption -> gait_instability
activity_decrease -> sit_to_stand_decline
sit_to_stand_decline -> near_fall_increase
gait_instability -> near_fall_increase
physiology_change -> sleep_disruption
```

这些边表示研究假设和时间关联，不表示因果。不得由模型自动生成带医学含义的新边。

### 8.3 链分数

对于时间顺序正确且间隔不超过 72 小时的边 `(a, b)`：

```text
edge_score = w_ab * persistence_a * persistence_b
             * quality_a * quality_b * exp(-delta_hours / tau)
```

默认 `tau=48` 小时。完整链分数为边分数的有界组合，必须裁剪到 `[0,1]`。`w_ab` 只能来自版本化配置或训练产物，不允许在运行时根据预期结果手工修改。

## 9. 离散时间生存模型

### 9.1 预测目标

模型以一天为时间步，输出未来 7 天条件风险率：

```text
h_k = P(T = k | T >= k, history), k = 1..7
```

累计风险：

```text
P(T <= H) = 1 - product(1 - h_k), k=1..H
```

由此定义：

- `24h = P(T <= 1)`
- `72h = P(T <= 3)`
- `7d = P(T <= 7)`

该结构天然保证累计风险单调。测试必须验证浮点容差内的单调性。

### 9.2 输入张量

最近 14 日输入包含：

1. 个体化 `directional_z`；
2. 原始特征的有界变换；
3. 每个特征的缺失掩码；
4. 每个模态的数据质量；
5. 变化事件持续性；
6. 时间链分数；
7. 即时步态风险的日级统计；
8. 基线状态编码。

禁止把缺失值直接填成正常值。数值填充只用于构造固定形状张量，模型必须同时接收缺失掩码。

### 9.3 默认模型与可选 TCN

默认 CPU 路径为规则风险特征加离散时间逻辑校准器，依赖 NumPy 和 scikit-learn。

可选 TCN 使用三层单向时序（causal）膨胀一维卷积；这里的 causal 是卷积不读取未来时间步，不表示医学因果：

- kernel size：3；
- dilation：1、2、4；
- channel：16、32、32；
- dropout：0.1；
- 输出：7 个条件风险率 logits。

PyTorch 必须是可选依赖。没有 PyTorch 或模型加载失败时，服务自动退化到默认 CPU 路径，并在输出中记录 `model_kind` 和降级原因。

### 9.4 删失与标签

训练样本必须包含 `event_day` 或 `censor_day`。观察期内没有跌倒不等于永久负样本；损失函数只计算到事件或删失时点。

真实跌倒标签来源必须可审计：公开数据标签、真实设备人工确认或研究人员复核。自动告警不能反过来作为自身的真值标签。

## 10. 不确定性、覆盖率与拒绝机制

### 10.1 默认不确定性方法

默认 CPU 模型使用按受试者重采样的 bootstrap 集成，至少 5 个成员；输出风险中位数及 10%–90% 区间。TCN 可以使用 bootstrap 或 MC dropout，但必须在模型卡中声明。

### 10.2 证据覆盖率

覆盖率按版本化必需特征组计算，不按字段总数简单平均：

```text
coverage = available_weight / expected_weight
quality = sum(group_weight * group_quality) / available_weight
```

视觉步态组始终是 `fall_forecast` 的关键组。只有睡眠或心理变化而没有任何步态/活动证据时，不得发布高等级跌倒预测。

### 10.3 拒绝条件

满足任一条件时 `abstained=true`：

- 有效证据覆盖率低于 `0.5`；
- 综合质量低于 `0.5`；
- 72 小时风险区间宽度大于 `0.35`；
- 输入时间戳乱序且无法无损排序；
- 模型、特征清单和输入 schema 版本不兼容；
- 所有风险证据都来自 `synthetic_research` 或 `offline_fixture`，但调用方请求真实发布结论。

拒绝时仍返回结构化结果和原因，但不得输出高风险或紧急处置结论。

### 10.4 风险等级

第一版默认阈值：

| 72 小时累计风险 | `decision` | `forecast_band` | 动作 |
| ---: | --- | --- | --- |
| `<0.20` | `info` | `low` | 继续观察 |
| `0.20–0.45` | `watch` | `elevated` | 记录变化并建议检查环境 |
| `0.45–0.70` | `warning` | `high` | 建议家属或照护人员关注 |
| `>=0.70` | `warning` | `very_high` | 建议及时人工评估和风险干预 |

`forecast_band=very_high` 仍不是实时红色跌倒紧急告警。U-PMCC 不产生 `critical`；只有现有 `fall_event` 模块确认或高度疑似跌倒时才进入紧急告警流程。这样可以保持与现有 `RiskLevel` 契约兼容。

当处于 `population_only`、只存在单一证据组或 `promoted=false` 时，预测最多显示 `watch`，除非调用方处于明确的研究分析模式。

## 11. 人工确认与干预反馈

### 11.1 反馈结构

每次预测或告警允许追加 `OutcomeFeedback`：

```json
{
  "forecast_id": "forecast-001",
  "reviewed_at": "2026-08-03T18:00:00+08:00",
  "reviewer_role": "family",
  "outcome": "near_fall",
  "interventions": ["clear_path", "check_night_light"],
  "outcome_24h": "no_fall",
  "outcome_72h": "no_fall",
  "notes_present": false
}
```

`outcome` 枚举固定为：

```text
confirmed_fall
near_fall
normal_adl
false_alarm
unknown
```

自由文本备注不进入模型。真实身份、联系方式和备注与研究特征分库存储。

### 11.2 标签使用规则

- `confirmed_fall` 必须有人工确认或公开真值来源。
- `false_alarm` 需要人工复核，不得由系统自动判定。
- `unknown` 不进入监督标签，但用于覆盖率和运营统计。
- 合成和演示反馈不得进入真实模型校准。
- 模型更新只能离线执行，并生成新的版本、评估报告和晋级记录；运行时不得在线自学习。

### 11.3 非医疗干预建议

第一版只允许从审核过的建议表中选择：

```text
check_environment_path
check_night_light
check_bedside_support
invite_mobility_review
contact_family
verify_device_status
continue_observation
```

建议必须显示触发证据，并明确系统不替代医生或照护人员。不得自动推荐药物调整或疾病治疗。

## 12. 心理健康模块边界

心理健康数据只通过以下方式参与系统：

1. `wellbeing_change` 独立输出睡眠、活动和主动量表变化。
2. 睡眠或活动变化可以成为 U-PMCC 的背景证据，但必须同时存在步态或活动能力证据，才能提高跌倒预测等级。
3. GDS-15 分数不直接作为生存模型数值输入；第一版只使用“筛查已完成”和“建议人工关注”等非诊断状态作为可选上下文。
4. 自伤相关回答继续走独立人工关注流程，不进入跌倒风险计算。

## 13. 代码结构与稳定接口

新增独立包，避免继续扩大现有文件：

```text
risk/pmcc/
├── __init__.py
├── schema.py              # 数据类、枚举、严格校验和序列化
├── baseline.py            # 人群先验、个人基线和更新保护
├── changes.py             # 方向标准化、变化事件和持续性
├── chains.py              # 预注册关联边和时间链
├── features.py            # 14日张量、掩码和特征清单
├── survival.py            # 累计风险数学、CPU校准器、可选TCN接口
├── uncertainty.py         # bootstrap区间、覆盖率和拒绝规则
├── decisions.py           # 风险等级、理由和干预建议
├── feedback.py            # 人工确认和结果存储契约
└── service.py             # 编排入口，不包含模型内部细节
```

脚本：

```text
scripts/generate_pmcc_fixture.py
scripts/build_pmcc_dataset.py
scripts/train_pmcc.py
scripts/evaluate_pmcc.py
scripts/generate_pmcc_report.py
```

测试：

```text
tests/risk/pmcc/test_schema.py
tests/risk/pmcc/test_baseline.py
tests/risk/pmcc/test_changes.py
tests/risk/pmcc/test_chains.py
tests/risk/pmcc/test_survival.py
tests/risk/pmcc/test_uncertainty.py
tests/risk/pmcc/test_decisions.py
tests/risk/pmcc/test_feedback.py
tests/risk/pmcc/test_service.py
tests/integration/test_pmcc_offline_flow.py
```

已有 `risk/personal_baseline.py`、`risk/gait_stability.py`、`fusion/decision_engine.py` 和 `core/events.py` 的公开行为必须保持兼容。新包通过适配器读取其结果；不得在首轮实现中重写现有实时跌倒状态机。

### 13.1 服务接口

```python
class PMCCService:
    def observe(self, observation: DailyObservation) -> None: ...
    def forecast(self, subject_id: str, as_of: date) -> PMCCForecast: ...
    def record_feedback(self, feedback: OutcomeFeedback) -> None: ...
```

`forecast` 必须是确定性的：相同模型版本、相同输入和相同随机种子得到相同结果。

模型接口：

```python
class SurvivalRiskModel(Protocol):
    def predict_hazards(self, features: FeatureWindow) -> Sequence[float]: ...
```

规则模型、逻辑校准器和 TCN 都实现该协议。

## 14. 训练流程

### 14.1 数据准备

1. 校验 schema、时区、重复记录和证据等级。
2. 按受试者和日期排序；同一受试者同日重复记录必须显式合并或报错。
3. 在训练折内部计算人群先验和个人基线，禁止使用测试折信息。
4. 构造 14 日输入窗口、未来 7 日事件/删失标签和缺失掩码。
5. 保存 `subject_split_id` 和每个样本的来源。

### 14.2 划分策略

- 真实数据优先使用按受试者分组的外层交叉验证。
- 同一人的相邻窗口不得同时出现在训练和测试中。
- 调参只使用内层训练折。
- 最终阈值和校准器不得查看外层测试折。
- 数据量不足时报告不可用，不用普通随机窗口划分代替。

### 14.3 模型晋级顺序

比较：

1. 固定人群阈值；
2. 个人稳健基线规则；
3. 个人基线 + 时间链；
4. CPU 离散生存校准器；
5. 可选 TCN 生存模型；
6. 完整 U-PMCC（最佳合格生存模型 + 不确定性 + 拒绝）。

新模型只有在真实、任务匹配的受试者级评估中优于当前发布基线才可 `promoted=true`。没有合格纵向数据时，所有学习模型保持 `research_only`，默认运行个人基线与时间链规则。

## 15. 评估与消融

### 15.1 指标

跌倒前预测至少报告：

- 24 小时、72 小时和 7 天的 time-dependent AUC；
- AUPRC，作为稀有事件的主要排序指标；
- Brier 分数和校准曲线；
- 固定告警预算下的召回率和精确率；
- 每受试者日误报数；
- 中位提前量及四分位范围；
- 拒绝覆盖率、拒绝样本性能和非拒绝样本性能；
- 各证据等级、模态可用状态和基线状态下的分层指标。

现有跌倒事件识别的 F1、召回率和每小时误报指标继续单独报告，不与 U-PMCC 的未来风险指标合并。

### 15.2 必需消融

逐项移除：

- 个人基线；
- 时间关联链；
- 睡眠/生理特征；
- 即时步态风险；
- 缺失掩码；
- 数据质量门控；
- 不确定性拒绝；
- TCN，仅保留 CPU 校准器。

### 15.3 晋级门槛

必须同时满足：

1. 外层受试者级评估的 AUPRC 相对当前基线提高至少 5%，或在 AUPRC 持平时 Brier 分数相对改善至少 5%。
2. 在相同告警预算下，召回率不得低于当前基线。
3. 每受试者日误报数不得比当前基线恶化超过 10%。
4. 24 小时、72 小时和 7 天累计风险全部满足单调性。
5. 非拒绝样本覆盖率至少 70%；若达不到，只能保持研究状态。
6. 所有指标来自同一 `release_id`、数据集版本和受试者划分。
7. 合成数据、演示夹具或跨任务公开数据不能单独触发晋级。

这些是工程发布门槛，不代表临床有效性。临床相关主张需要独立前瞻性研究。

## 16. 错误处理与降级

- 少于 14 日输入：允许预测，但按冷启动规则降低等级并说明依据。
- 缺少视觉步态组：返回 `abstained` 或最多 `watch`，不得输出高等级跌倒预测。
- SDNL1 不可用：保留视觉、活动和近跌倒证据，标记 `vision_only` 或等价质量状态。
- 时间戳重复：如果内容完全相同则去重；内容冲突则报数据错误。
- 时间戳乱序：能无损排序时排序并记录；跨时区或日期归属不明确时拒绝。
- 模型文件缺失或校验失败：退化到规则模型，不使用随机权重。
- 特征版本不兼容：拒绝预测并指出版本差异。
- 不确定性计算失败：不得退回单点高置信度输出；使用规则路径并标记区间不可用。
- 反馈存储失败：不影响实时 `fall_event` 告警，但必须记录本地可观测错误。

## 17. 隐私、安全与审计

- U-PMCC 默认只需要骨架派生特征和日级摘要，不需要连续保存原始视频。
- 设备密钥、联系人和身份映射不得进入训练数据或模型产物。
- 每个预测保存模型版本、特征版本、证据 ID、拒绝原因和来源等级。
- 事件片段继续遵守现有授权开关和保留期限。
- 训练、评估和报告不得输出完整设备序列号、访问令牌或真实姓名。
- 人工反馈修改必须追加记录，不覆盖原始预测。

## 18. 测试规格

实现必须先写失败测试，再写代码。至少覆盖以下行为：

### 18.1 Schema

- 拒绝无时区时间、布尔型置信度、范围外质量和空受试者 ID。
- 缺失特征不会变成零。
- `available=false` 与非零质量冲突时拒绝。
- JSON 往返不丢失 provenance。

### 18.2 基线

- 0、6、7、13、14 个有效日对应正确状态。
- 每个特征独立计算有效日。
- MAD 为零时使用特征尺度下限。
- 已确认跌倒、设备离线和异常日不污染基线。
- 测试折数据不进入训练折基线。

### 18.3 变化与时间链

- 3 日中 2 日异常才生成变化事件。
- 缺失日不算正常日。
- 反向时间顺序不生成链。
- 超过 72 小时不生成链。
- 低质量节点降低链分数。
- 输出节点、间隔和关联声明完整。

### 18.4 生存风险

- 7 个风险率均在 `[0,1]`。
- 累计风险严格非递减。
- 删失后的时间步不计入损失。
- 同输入和种子得到同输出。
- PyTorch 不可用时 CPU 路径正常工作。

### 18.5 不确定性与拒绝

- 覆盖率或质量低于阈值时拒绝。
- 区间宽度过大时拒绝。
- 单一睡眠异常不能产生高等级跌倒预测。
- 缺少 SDNL1 时能视觉降级。
- `synthetic_research` 不能生成真实发布通过结论。

### 18.6 反馈

- 未确认自动告警不能成为真值标签。
- `unknown` 不进入监督训练。
- 合成反馈不能进入真实校准。
- 反馈追加不改变原预测记录。

### 18.7 端到端离线流程

使用固定夹具完成：

```text
14日正常基线
→ 2日睡眠异常
→ 活动下降
→ 步态摆动增加
→ 时间链形成
→ 风险上升并列出证据
→ 移除生理模态
→ 视觉降级但不崩溃
→ 输入低质量数据
→ 拒绝高置信度结论
```

## 19. 编码任务顺序

交给其他模型时按以下顺序实施，每步单独提交并运行局部测试：

1. 建立 `risk/pmcc` 包、schema 和严格序列化测试。
2. 实现三级冷启动、稳健个人基线和更新保护。
3. 实现方向变化量、持续性和变化事件。
4. 实现预注册时间关联边、72 小时窗口和证据链。
5. 实现离散风险率到累计风险的纯函数及删失损失接口。
6. 实现 CPU 规则/逻辑生存模型和版本化模型卡。
7. 实现 bootstrap 不确定区间、覆盖率和拒绝机制。
8. 实现风险等级、证据解释和审核过的干预建议。
9. 实现人工确认反馈契约和追加式存储。
10. 实现 `PMCCService`，适配已有步态、生理和事件模块。
11. 实现合成夹具生成器；所有产物明确标记不可晋级。
12. 实现受试者级训练、评估、消融和报告脚本。
13. 可选实现 PyTorch TCN；无 PyTorch 路径必须继续通过。
14. 运行端到端离线流程和完整回归测试。

编码模型不得在一个提交中同时重写现有跌倒状态机、UI、设备客户端和 U-PMCC。发现现有接口不足时，先增加兼容适配器和测试。

## 20. 完成标准

本阶段完成必须满足：

- `risk/pmcc` 模块和列出的测试存在且通过。
- 默认 CPU 路径不需要真实设备或 PyTorch。
- 离线完整变化链可重复运行。
- 合成数据、公开数据和真实数据指标严格隔离。
- 输出包含多时间窗风险、不确定区间、覆盖率、证据和拒绝原因。
- 缺失模态、低质量数据和模型加载失败均安全降级。
- 不出现心理诊断、医学因果或合成临床性能声明。
- 现有跌倒事件、心理筛查、告警和设备离线测试保持兼容。
- 代码、模型卡和报告不包含密钥、完整序列号或旧绝对路径。

真实纵向配对数据尚不可用时，学习模型最终状态应为 `research_only`，默认发布路径为个人基线规则、时间关联链和不确定性拒绝。该限制不是实现失败，而是证据边界。

## 21. 研究依据

以下资料支持睡眠、日常活动、步态与跌倒风险之间存在值得研究的关联，但不证明 U-PMCC 的因果链或临床效果：

- Sleep Health and Falls Risk for Older Adults Living in Residential Aged Care and in Community Dwelling Settings：https://pmc.ncbi.nlm.nih.gov/articles/PMC11650497/
- Autonomous fall risk assessment in Australian Residential Aged Care Facilities using passive sensors：https://pubmed.ncbi.nlm.nih.gov/41443324/
- Prediction of injurious falls in older adults using digital gait biomarkers extracted from large-scale wrist sensor data：https://www.ukbiobank.ac.uk/publications/prediction-of-injurious-falls-in-older-adults-using-digital-gait-biomarkers-extracted-from-large-scale-wrist-sensor-data/
- An Interpretable Machine Learning Approach to Predict Fall Risk Among Community-Dwelling Older Adults：https://pubmed.ncbi.nlm.nih.gov/35112279/

## 22. 交给编码模型的启动指令

可以把以下内容与本文档路径一起交给编码模型：

```text
请完整阅读 docs/superpowers/specs/2026-08-03-u-pmcc-design.md，严格按“编码任务顺序”实施。

要求：
1. 每项功能先写失败测试，再写最小实现。
2. 保持现有 fall_event、心理筛查、告警、设备客户端和 gait_stability 的公开行为兼容。
3. 默认路径只能依赖 NumPy 和 scikit-learn；PyTorch TCN 必须可选。
4. synthetic_research 和 offline_fixture 产物必须 promoted=false，不能进入真实发布指标。
5. 不得把时间关联写成医学因果，不得输出心理诊断或自动医疗建议。
6. 缺失模态、低质量数据、版本不兼容和模型加载失败必须安全降级或拒绝。
7. 每个任务独立提交，并在提交前运行该模块测试；最终运行完整回归测试。
8. 不要提交 outputs、密钥、令牌、设备验证码、完整序列号或本机绝对路径。

先检查当前分支和既有测试，再从任务1“schema和严格序列化测试”开始。不要在同一提交中实现全部模块。
```
