# 算法验证摘要（截至 2026-08-12）

## 可复现的跌倒风险外部实验

| 训练/评估方案 | 跌倒 F1 | Recall | ADL 假警率 | 结论 |
|---|---:|---:|---:|---|
| 仅 UR Fall 跌倒窗口训练，阈值 0.50 | 0.767（序列宏平均） | 0.800 | 0.316 | 误报偏高，不部署 |
| UR Fall 跨域 ADL 训练，阈值 0.50 | 0.667 | 0.700 | 0.044 | 过度保守，不部署 |
| 跨域模型；跌倒 OOF Recall >= 0.80 取最高阈值 | 0.842 | 0.800 | 0.089 | 最佳外部实验候选，仍不部署 |

阈值 0.24407608807086945 仅由跌倒序列的留一预测选择；ADL 仅用于独立假警审计，未参与阈值选择。

## 数据与质量控制

- UR Fall：30 段跌倒 RGB 与 40 段 ADL RGB，均为 cam0 视角。
- 跌倒窗口由离线加速度峰值锚定，窗口严格结束于冲击前 5 帧。
- 20 段跌倒序列通过完整关键点成对校验，构成 40 个正/早期安全窗口。
- 40 段 ADL 生成 80 个窗口；MediaPipe Pose Landmarker 质量校验后保留 48 个完整窗口。
- 未检出人体关键点、帧缺失、哈希不一致或同步表异常的样本全部排除，不补零。

## 运行产物

- 跌倒窗口：F:\datasets\fall_prediction\urfall\prefall_windows
- 跌倒姿态：F:\datasets\fall_prediction\urfall\pose_windows_paired
- ADL 姿态：F:\datasets\fall_prediction\urfall\adl_pose_windows
- 跨域训练结果：F:\datasets\fall_prediction\urfall\crossdomain_tcn
- 阈值审计：F:\datasets\fall_prediction\urfall\crossdomain_tcn\threshold_selection.json

## 运行命令

~~~powershell
python scripts/train_urfall_crossdomain_tcn.py --fall-manifest F:/datasets/fall_prediction/urfall/pose_windows_paired/pose_manifest.jsonl --fall-root F:/datasets/fall_prediction/urfall/pose_windows_paired --adl-manifest F:/datasets/fall_prediction/urfall/adl_pose_windows/pose_manifest.jsonl --adl-root F:/datasets/fall_prediction/urfall/adl_pose_windows --output-dir F:/datasets/fall_prediction/urfall/crossdomain_tcn --epochs 80 --device cuda

python scripts/select_urfall_threshold.py --fall-predictions F:/datasets/fall_prediction/urfall/crossdomain_tcn/fall_predictions.jsonl --adl-predictions F:/datasets/fall_prediction/urfall/crossdomain_tcn/adl_predictions.jsonl --output F:/datasets/fall_prediction/urfall/crossdomain_tcn/threshold_selection.json --recall-floor 0.8
~~~

## 使用边界

这些指标来自公开数据集的离线实验，不代表萤石摄像头真实场景性能。部署前必须以设备实拍视频做独立验证，并保留人工确认和报警冷却机制。所有 UR Fall 训练产物均为 promoted=false。
