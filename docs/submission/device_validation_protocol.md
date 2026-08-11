# 萤石设备实拍验证操作卡

## 目的

验证已发布 PA-DTSF 相位模型在本地视频、摄像头或萤石流上的可运行性。该操作只生成本地报警日志，不会把任何结果自动发布为生产模型。

## 已验证的本地输入

- 检查点：outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt
- 运行入口：scripts/run_live_monitor.py
- 输出：指定目录中的 local_alerts.jsonl
- 当前离线外部实验阈值：UR Fall 候选 0.244；它不能直接替换 PA-DTSF 服务阈值，必须先实拍校准。

## 已完成的离线端到端烟测

2026-08-12 已使用 UR Fall 的录制室内视频完成一次本地端到端烟测：本地视频
PTS、YOLO11-pose 姿态提取、PA-DTSF 检查点、双时间尺度缓冲和本地告警服务均
参与执行。最终状态为 `camera_health=healthy`、`source_error_count=0`；该片段未
产生预测告警，`decision_count=0`，符合“无已确认跌倒时不告警”的安全预期。

录像回放使用媒体时间戳而不是解码速度；短时间尺度允许达到配置覆盖率的真实
姿态用于质量评估，长时间尺度仍只接受完整且真实的模型输入，不插值、不复制。

## 安全测试顺序

1. 先使用 30 至 60 秒的已录制室内视频，确认脚本能输出 camera_health、decision_count 和本地日志。
2. 再使用萤石流做 60 秒 smoke run，只观察视频、姿态获取和报警链路健康；不得做真实跌倒危险动作。
3. 由成年人在安全垫和陪护条件下演示缓慢坐下、行走、弯腰等 ADL，记录任何误报。
4. 实拍数据单独存放，并由人工复核事件；不得把隐私视频上传到公开仓库。

## 本地视频运行

~~~powershell
python scripts/run_live_monitor.py --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt --input D:/test/room_adl.mp4 --model yolo11n-pose.pt --device cuda --output-dir outputs/live-test --smoke-seconds 60 --no-browser
~~~

## 萤石流运行

将 input 替换为已由用户在本机验证可访问的 RTSP/设备流地址。流地址不得提交到 Git、文档或日志。

~~~powershell
python scripts/run_live_monitor.py --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt --input "RTSP_ADDRESS_REDACTED" --model yolo11n-pose.pt --device cuda --output-dir outputs/live-ezviz-smoke --smoke-seconds 60 --no-browser
~~~

## 通过标准

- 流可持续读取且 camera_health 正常。
- 无源错误导致服务退出。
- ADL 演示期间的报警均被本地日志记录且可人工解释。
- 所有实拍指标都与公开数据集指标分开报告。

未满足任一条件时，记录失败原因并保持 promoted=false。
