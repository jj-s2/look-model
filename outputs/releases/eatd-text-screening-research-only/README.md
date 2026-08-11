# EATD 文本心理筛查研究基线

本目录保存一个可复现的研究模型，不是临床模型、心理诊断工具或自动报警器。

- 模型文件：`eatd_text_baseline.joblib`
- 模型：仅官方训练分区拟合的中文字符 TF-IDF 加类别平衡 Logistic Regression
- 超参数选择：训练集 5 折分层 OOF；验证分区不参与词表、模型或阈值拟合
- 验证集：79 个官方验证样本，其中 11 个为量表高风险标签
- 指标：AUC 0.670，F1 0.300，Recall 0.818，Precision 0.184
- 决策阈值：0.23912689936524265
- SHA-256：`F731CE42FE207BFC43259FB9DC967AE60291B3F1BED917ED4516A23C77968BA1`
- 发布状态：`promoted=false`

训练与复现：

```powershell
python scripts/train_eatd_baseline.py --data F:/datasets/mental_health/EATD-Corpus-ready/EATD-Corpus --out F:/datasets/mental_health/eatd_text_oof_run
```

仅可在获得自愿同意、最小化数据使用和人工复核的前提下，用于提示“建议由家属或
专业人员进一步了解”。不得用于医疗结论、紧急救援触发或从日常摄像头/音频持续
监测心理状态。
