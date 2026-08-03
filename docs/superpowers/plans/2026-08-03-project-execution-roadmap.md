# 老年跌倒与心理健康项目实施路线图

> 本路线图只负责执行顺序。技术设计以 docs/superpowers/specs/2026-08-03-multidataset-fall-wellbeing-optimization-design.md 为准。

## 执行顺序

1. 2026-08-03-padtfs-data-foundation.md
2. 2026-08-03-padtfs-core-model.md
3. 2026-08-03-c6c-live-alert-closure.md
4. 2026-08-03-wellbeing-pmcc-evidence.md

不得并行执行会修改同一公共接口的任务。每份计划完成后必须：

- 运行该计划列出的局部测试；
- 运行完整回归测试；
- 检查 Git 差异和未跟踪文件；
- 生成单独提交；
- 由另一个审查上下文核对设计符合性；
- 达到验收门槛后才进入下一份计划。

## 里程碑

| 里程碑 | 可独立验收的结果 |
| --- | --- |
| M1 数据底座 | 数据来源、许可、统一标签、受试者划分和 dataset lock 可复现 |
| M2 核心模型 | 双时间尺度、相位约束、质量门控和评估脚本可离线运行 |
| M3 实时闭环 | C6c 流进入完整推理链，告警可审计，延迟和误报有真实证据 |
| M4 心理与长期风险 | 低打扰筛查、日级摘要和 U-PMCC 分流可演示且不作诊断 |

## 设计覆盖索引

| 设计要求 | 实施位置 |
| --- | --- |
| 数据注册、许可和来源 | 数据底座 Task 1 |
| UnifiedClip 与稳定标识 | 数据底座 Task 2 |
| 六相位和困难负样本映射 | 数据底座 Task 3 |
| 受试者与同步机位无泄漏划分 | 数据底座 Task 4 |
| 五个公开数据适配器 | 数据底座 Task 5 |
| dataset lock 和 split manifest | 数据底座 Task 6 |
| 双时间尺度窗口 | 核心模型 Task 2 |
| 归一化与质量门控 | 核心模型 Task 3 |
| 相位顺序和状态平滑 | 核心模型 Task 4 |
| TCN、融合头和多任务损失 | 核心模型 Task 5 |
| 四类风险分流与 critical 边界 | 核心模型 Task 6 |
| 跨数据集评估和晋级 | 核心模型 Task 7 |
| 训练、校准、消融和模型卡 | 核心模型 Task 8 |
| C6c 完整推理 | 实时闭环 Task 1 |
| 本地、Webhook 和萤石交付 | 实时闭环 Task 2–3 |
| P50/P95 和每小时误报 | 实时闭环 Task 4 |
| 30 分钟真实设备验证 | 实时闭环 Task 5 |
| 低打扰心理筛查 | 心理与长期风险 Task 1 |
| 隐私日级活动摘要 | 心理与长期风险 Task 2 |
| U-PMCC 分流 | 心理与长期风险 Task 3 |
| CHARLS 群体关联和 UI 分屏 | 心理与长期风险 Task 4 |

## 全项目禁止项

- 不提交 outputs、原始视频、压缩包、关键点缓存或受限数据。
- 不提交 AppKey、Secret、AccessToken、验证码或完整设备序列号。
- 不按帧或随机窗口拆分同一受试者数据。
- 不让 fall_forecast、prefall_warning 或 wellbeing_change 产生 critical。
- 不把模拟、合成或公开数据指标描述成真实老年临床效果。
- 不在 SDNL1 接口未验证时生成虚构生理数据。
- 不要求老人模拟跌倒。

## 最终验证

    python -m pytest -q
    python scripts/verify_datasets.py
    python scripts/generate_evaluation_report.py
    python scripts/generate_pmcc_report.py

设备验证只在用户明确连接 C6c 且本机密钥配置完整时运行。离线测试不得依赖真实设备或互联网。
