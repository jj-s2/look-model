# PA-DTSF 留出测试集效果图

本报告由 `scripts/plot_phase_results.py` 根据冻结的 GMDCSA24 v2.1 受试者划分自动生成。所有图表均来自未参与训练的 test split，固定阈值为 `0.50`，没有在测试集上调阈值。

## 核心结果

| 指标 | 结果 |
| --- | ---: |
| 测试片段 | 32 |
| 阳性（跌倒） | 16 |
| Precision | 80.0% |
| Recall | 75.0% |
| F1 | 77.4% |
| ROC-AUC | 0.836 |
| Average Precision | 0.890 |
| Brier score | 0.215 |

## 证据边界

- 数据集：GMDCSA24 v2.1；按 subject 分组，当前 test split 为留出的受试者集合。
- 推理路径：归一化 64 帧骨架、长时 PA-DTSF 分支、`short_quality=0`，与实时入口一致。
- 这些结果证明该冻结测试划分上的算法区分能力；不能直接外推为真实老人群体、临床诊断或萤石设备现场效果。
- 本 release 没有经过心理健康标签验证，因此不生成心理健康“准确率”图，避免把筛查功能误报成诊断模型。

## 生成信息

- release_id：`padtfs-gmdcsa24-gpu-norm`
- checkpoint_sha256：`6625d0a8482eac96e1e52f3efabddb8f9b7e3b2f00a1dbf40763d5d4f85d9ed5`
- device：`cuda`

图文件：ROC/PR、混淆矩阵、阈值权衡、校准图和得分分布图。
