# PACE-WB 心理变化预警验证报告

## 结论

当前版本可以用于比赛演示和研究复现，但心理模块保持 `research_shadow` / `promoted=false`。它能做的是：从日级睡眠、活动和自愿短问答中识别“相对个人基线的持续变化”，在证据不足时弃权，并提供本地短问候候选；它不会诊断抑郁症，不会连续监听摄像头或麦克风，也不会把普通心理事件发送到萤石、Webhook、短信、邮件或家属通道。

## 已验证的实现

1. `PACE-Behavior`：0–6 天冷启动、7–13 天预备、至少 14 个有效日运行；每特征有效日独立计数；同一特征最近 3 个有效日中至少 2 次异常才形成候选；旅行、低质量和非法值弃权；异常日不写入个人基线。
2. `PACE-Voluntary`：自愿同意后才处理 2–3 个短回答；支持注入 MacBERT 文本和 eGeMAPSv02/GRU 音频预测器；缺少权重、版本不符、低 ASR 置信度、音频质量不足、模态冲突或反事实删除不稳定时弃权。
3. `PACE-Safety Gate`：研究影子结果不晋级；高分只生成筛查关注或人工复核记录；自伤候选要求人工复核，外发恒为禁用。
4. 交付防火墙：融合层、实时服务层、`AlertDispatcher` 三处都拒绝普通 wellbeing 外发；跌倒事件仍走原有风险和告警路径。
5. 交互与隐私：短问候一周最多一次，GDS-15 四周最多一次；GDS 前端只返回题目和选项，不暴露 `risk_answer`；拒绝/停止状态可持久化。

## 软件回归证据

在 2026-08-12 的工作树中运行：

```powershell
python -m pytest tests/mental tests/fusion tests/pipeline tests/ui tests/integration -q
python -m pytest tests/alerts -q
python -m compileall mental fusion pipeline alerts ui scripts
```

心理、融合、实时管线、UI 和集成回归共 260 项通过；告警回归 13 项通过。完整机器可读记录见 [`metrics.json`](../../outputs/mental/pace-wb-validation/metrics.json)。

## 研究基线与边界

EATD 文本基线沿用官方 83/79 受试者划分，验证集 79 人、11 个高风险标签，F1=0.300、ROC-AUC=0.670。该结果只说明研究基线可复现，不代表老年人效果，也没有进入默认实时心理告警。老年外部纵向配对数据当前为 0 人，因此发布门控必然保持 `promoted=false`。

## 复现命令

```powershell
# 研究文本基线：只保存模型和审计清单，不保存原始音频
python scripts/train_wellbeing_shadow.py --input <consented-checkins.jsonl> --output-dir outputs/mental/shadow-run
python scripts/evaluate_wellbeing_shadow.py --artifact outputs/mental/shadow-run/shadow_model.joblib --input <held-out-checkins.jsonl> --output outputs/mental/shadow-run/evaluation.json

# 发布门控：没有独立老年外测时明确拒绝晋级
python scripts/evaluate_wellbeing_release.py --config configs/screening/pace_wb_v1.json --evidence <evidence.json> --output-dir outputs/mental/release-audit
```

## 下一阶段所需证据

- 受试者级、独立于训练集的老年纵向数据，至少覆盖工作日/周末、缺失和设备切换；
- 自愿短问答的人工复核标签与自伤候选人工处置演练；
- 在相同问询预算下，相对 `rules_v1` 的 AUPRC、Brier、ECE、覆盖率-风险和每人每月误邀请数净收益；
- 只有上述证据通过且有人值守的人工复核队列建立后，才考虑 `promoted_local_invitation`，仍不授予心理事件外发权限。
