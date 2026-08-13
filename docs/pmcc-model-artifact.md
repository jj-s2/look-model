# PMCC 模型工件加载

`scripts/train_pmcc.py` 生成的 `model.json` 是只包含 JSON 数字、系数和
schema 元数据的可移植工件，不使用 pickle，也不携带设备序列号或本机路径。
它明确记录 `evidence_tier`、`promoted` 和完整的 PMCC
`directional_z/raw/derived` 特征 schema。

可以直接交给本地服务加载：

```python
from risk.pmcc.service import PMCCService

service = PMCCService(model="outputs/pmcc-model/model.json")
```

或者显式加载并检查模型元数据：

```python
from risk.pmcc.calibrator import RuleSurvivalCalibrator
from risk.pmcc.service import PMCCService

model = RuleSurvivalCalibrator.load_artifact("outputs/pmcc-model/model.json")
service = PMCCService(model=model)
```

`RuleSurvivalCalibrator.from_artifact(mapping)` 适合从对象存储读取已解析
的 JSON。加载器会校验七个 horizon estimator、系数维度、有限数值和特征
schema；不支持的或损坏的工件会拒绝加载。合成/离线证据工件始终保持
`promoted=false`，不能提供发布指标。
