# UP-Fall 时序预跌倒基线设计

## 目标

在 UP-Fall 适配器生成的 33 关节帧窗口上训练轻量时序分类器，区分“首次冲击前的近端窗口”和“同一次记录中更早的安全窗口”。结果只作为受控的 UP-Fall 实验基线，不宣称为老年人临床预警模型，也不替代 GMDCSA24 的验证结果。

## 模型与输入

输入为 float32 姿态张量，形状为 (T, 33, 3)，其中 T 等于 pre_frames。每个样本按自身帧内关节质心和尺度规范化；同时保留相对初始质心的平移与一阶速度，以保留姿态和重心变化。轻量 TCN 使用两层一维卷积、GroupNorm、GELU、dropout 和全局平均池化，输出一个预冲击 logit。

## 训练与评估

使用 Leave-One-Subject-Out：每折仅用其余主体训练，固定 0.5 判定阈值，不基于留出主体选择阈值或早停轮次。每折使用确定性随机种子、BCEWithLogitsLoss 和训练主体统计得到的类别权重。报告每折和主体宏平均 Precision、Recall、F1。

## 产物与安全界限

训练脚本输出 metrics.json、逐折预测 JSONL、模型状态字典和模型卡。模型卡必须含 experimental_only=true、not_for_clinical_performance=true、window_unit=frames、数据摘要和主体数。任何分数不得自动提升为可发布跌倒预警模型。

## 验证

测试验证 33 关节张量加载、主体留出不重叠、归一化与网络输出形状、CPU 上的小型 LOSO 训练可复现、实验产物禁止晋级。真实训练使用 GPU（若可用），并保存结果到 F:/datasets/fall_prediction/upfall_3d_skeletons/temporal_tcn/。
