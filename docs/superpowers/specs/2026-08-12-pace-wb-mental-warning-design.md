# PACE-WB：低打扰、个体化、可弃权的老年身心状态筛查设计

## 1. 结论与边界

本项目不把摄像头表情、日常谈话或声学特征直接解释为抑郁症诊断。正式运行链只使用经过同意的日级睡眠/活动聚合趋势，并在持续变化、个人基线充分、问询预算允许时，邀请一次自愿的短问候。

音频和文本只在老人明确同意的短问候中临时处理；GDS-15 只在老人主动完成或明确同意后使用，最多每 28 天邀请一次。任何心理结果都带有 `is_diagnosis=false`，不自动呼叫急救，不自动向萤石、Webhook、短信或家属通道发送普通心理事件。

明确表达的自伤疑似内容进入人工复核队列，不参与普通心理分数。低置信 ASR 只能标记为待复核或进行一次中性澄清，不能自动排除风险。

## 2. 借鉴与拒绝

### 2.1 借鉴

- EATD 官方项目的受试者级划分、文本/音频单模态基线和晚期融合消融；EATD 仅有 162 名志愿者，不足以证明老年居家泛化。
- MacBERT-base 作为自愿中文短回答的研究性文本编码器；先冻结主体或使用轻量适配器。
- openSMILE eGeMAPSv02 作为低维、可解释的研究性声学特征；版本、配置和许可证固定。
- GLOBEM 等纵向行为建模工作提供个人基线、跨时间评估和缺失数据压力测试思路，但不把青年/可穿戴人群标签迁移为老年心理标签。
- DAIC-WOZ 仅在获得许可后用于跨域研究，不进入默认运行链。

### 2.2 拒绝

- 连续监听或从摄像头表情推断心理疾病。
- 把当前 EATD 文本基线（验证 F1=0.30、AUC=0.67）接入实时心理告警。
- 把同一次回答的音频和 ASR 文本计为两个独立证据域。
- 缺失模态补零后继续输出高置信度结果。
- 让通用 LLM 生成心理风险等级或自伤结论。
- 仅由冷却期到期触发完整 GDS-15。

## 3. 整体架构

```text
日级睡眠/活动聚合
  → PACE-Behavior：严格校验 → 每特征有效日 → median/MAD → EWMA+CUSUM
  → invite_candidate → InteractionPolicy（7日短问候、28日GDS、静默、拒绝）
  → 本地短问候（用户自愿）
  → PACE-Voluntary research_shadow：MacBERT文本 + eGeMAPSv02/GRU声学
  → PACE-Safety Gate：质量、缺失、反事实稳定性、选择性弃权
  → WellbeingAssessmentEvent（local_only / human_review_only）
```

跌倒事件继续由现有 `DecisionEngine` 和 `AlertDispatcher` 独立处理，心理链路永远不能产生 `critical`，也不能改变跌倒分数。

## 4. 算法创新：PACE-WB

PACE-WB（Personalized Abstaining Consent-gated Evidence fusion for Wellbeing）是一项面向本项目场景的系统级算法创新，主张限定为“低打扰居家筛查的个人变化、可靠性门控和权限隔离”，不声称单个统计公式首次提出。

### 4.1 个体变化

```text
z_m(t) = (x_m(t) - median_m) / (1.4826 * MAD_m + epsilon)
```

个人基线只使用有效观测日。0–6 日为 `cold_start`，7–13 日为 `provisional`，至少 14 个有效日且覆盖工作日/周末后才为 `operational`。旅行、住院、急性身体不适、设备切换、低质量和非法值不更新基线。同一特征或同一证据域最近三个有效日中至少两日异常，才可形成持续变化。

### 4.2 可靠性与弃权

```text
reliability = availability * source_quality * completeness * in_domain_gate
```

覆盖率不足、模态冲突、ASR 置信度不足、输入过短、模型版本/配置不一致时输出 `abstained`。影子评估可比较 `p_all` 与删除一个模态后的 `p_without_m`；若删除低质量模态导致跨阈值翻转，只降低可靠度或弃权，不把差值解释为因果贡献。EATD 上的 Conformal/风险-覆盖率结果只作为同分布研究记录，没有独立老年校准集前不宣称覆盖保证。

### 4.3 问询预算

模型只产生 `invite_candidate`，不能绕过 `InteractionPolicy`。短问候每 7 天最多一次，完整 GDS-15 每 28 天最多一次；21:00–08:00 静默；拒绝、无响应和停止提醒都持久化并立即结束本次问询。

## 5. 事件与权限契约

心理链路使用独立结构化事件，必要时再转换为兼容的 `EventType.WELLBEING_CHANGE`：

```text
WellbeingAssessmentEvent
- event_id, subject_alias, timestamp
- state: baseline_forming | observe | invite_candidate | screening_concern | abstained | human_review_required
- action: local_record_only | local_invite_short_checkin | local_offer_gds | enqueue_human_review
- delivery_scope: external_forbidden | local_only | internal_human_review_only
- evidence_domains, evidence_codes, coverage, quality, uncertainty
- baseline_state, abstention_reasons, consent_scope, model_version
- is_diagnosis=false, fall_critical_eligible=false, demo
```

普通心理事件禁止 Webhook、EZVIZ、短信、邮件、家属和急救通道；人工复核事件只创建内部待办，外部联系必须由人工授权事件完成；`research_shadow` 只能记录影子结果；不保存原始音频/视频，除非另行同意、加密、审计并设短 TTL。

## 6. 训练、评估与晋级

先保留 TF-IDF+Logistic Regression 基线，按 EATD 受试者级划分复现 audio-GRU/text-BiLSTM、MacBERT、eGeMAPS 和晚期融合；使用模态掩码、ASR 低置信、回答截短、噪声和缺失增强训练影子门控。老年数据到位前所有学习分支为 `research_only=true, promoted=false`。

报告 AUPRC、ROC-AUC、macro-F1、Brier、ECE、风险-覆盖率、缺失率 25/50/75%、方言/设备分层、每人每月误邀请数、接受/拒绝/停止率、静默违规数和普通心理外部投递次数（目标为 0）。

只有独立老年受试者外测在相同问询预算下相对 `rules_v1` 有预注册的 AUPRC/Brier 净收益，且缺失、校准和亚组测试通过，才允许晋级到 `promoted_local_invitation`。

## 7. 实施优先级与回滚

P0 修复趋势、交互、事件和外部投递合同；P1 实现 PACE-Behavior 和本地短问候编排；P2 实现 EATD 影子训练、MacBERT/eGeMAPS 研究评估；P3 收集老年纵向配对数据并完成外测和人工复核演练。

配置保留 `rules_v1 | research_shadow | promoted_local_invitation`；模型缺失、质量不足、覆盖率不足、版本不一致或审计失败时自动回退到稳健趋势＋自愿 GDS-15。

## 8. 参考资料

- EATD：https://arxiv.org/abs/2202.08210 、https://github.com/speechandlanguageprocessing/ICASSP2022-Depression
- MacBERT：https://github.com/ymcui/MacBERT 、https://arxiv.org/abs/2004.13922
- openSMILE：https://audeering.github.io/opensmile/about.html
- DAIC-WOZ：https://dcapswoz.ict.usc.edu/
- GLOBEM：https://github.com/UW-EXP/GLOBEM
- WHO AI 健康伦理：https://www.who.int/publications-detail-redirect/9789240037403
- USPSTF：https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-depression-suicide-risk-adults
