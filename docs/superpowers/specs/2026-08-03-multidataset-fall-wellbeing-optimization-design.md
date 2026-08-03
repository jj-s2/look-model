# 多数据集相位感知跌倒预警与低负担心理健康模块优化设计

> 状态：已确认，可作为后续实施的唯一设计依据
> 日期：2026-08-03
> 适用仓库：jj-s2/look-model
> 目标读者：接手实现、训练、评估和演示的编码 AI 或开发者

## 1. 文档目的

本设计把现有“跌倒/日常活动二分类”升级为一套可解释、可评估、可在萤石 C6c 实时演示的连续防护系统，同时保留低负担的心理健康筛查能力。

项目只做一个核心算法创新：

**PA-DTSF：相位感知双时间尺度质量门控骨架融合网络**

它同时识别：

1. 正常日常活动；
2. 跌倒前异常；
3. 快速下降；
4. 落地冲击；
5. 倒地；
6. 恢复。

心理健康不做摄像头诊断，也不持续监听语音。它通过自愿量表、低频短问答和个人趋势变化给出“建议关注”，与实时跌倒事件保持独立。

本文档不是临床方案。公开数据中的跌倒大多由健康年轻人模拟，所有指标只能表述为研究或工程验证结果。

## 2. 已有基础与必须修复的问题

### 2.1 可复用能力

- C6c 已能接入萤石开放平台并获取实时画面。
- RTMDet、RTMPose 和 PoseC3D 已有配置与封装基础。
- 已发布的 PoseC3D 权重位于 experiments/outputs/posec3d_gmdcsa24/best_acc_top1_epoch_45.pth。
- GMDCSA24 按受试者留一评估的已有结果：
  - Accuracy：0.9139
  - Fall F1：0.9078
  - Fall Recall：0.9143
  - 混淆矩阵：TP=64，FP=7，FN=6，TN=74
- 现有模块已覆盖：
  - risk/gait_stability.py：骨架步态特征；
  - risk/fall_state_machine.py：实时跌倒状态机；
  - risk/pmcc：24 小时、72 小时和 7 天研究型风险链；
  - mental/gds15.py、mental/trend.py、mental/interaction_policy.py：筛查和低打扰策略；
  - fusion/decision_engine.py：事件、预测和心理变化的分流；
  - pipeline/live_service.py：有超时保护的实时编排；
  - alerts/dispatcher.py：本地 JSONL 告警记录和去重。

### 2.2 当前证据缺口

- 跌倒前规则在现有小样本上的 ROC-AUC 为 0.8456，但当召回率约为 90% 时，误报率约为 33.3%，误报是首要瓶颈。
- 当前发布报告缺少完整推理链 P95 延迟和正常生活视频上的每小时误报数。
- 现有训练主要依赖 4 名受试者的 GMDCSA24，跨家庭、遮挡、夜间和跨数据集泛化证据不足。
- fusion/decision_engine.py 当前允许 fall_forecast 在分数大于等于 0.9 时成为 critical。此行为与“预测不等于已发生跌倒”的安全边界冲突，后续实现必须改为：
  - fall_event 才能产生 critical；
  - fall_forecast 最高只能是 warning；
  - wellbeing_change 最高只能是 warning。
- alerts/dispatcher.py 只有本地写入，没有真实网络推送适配器。
- SDNL1 没有已验证的公开健康数据接口和字段映射，不能把它写成已完成的生理融合能力。
- U-PMCC 现有训练样本属于合成研究数据，promoted=false，不能作为真实性能证据。

## 3. 总体设计原则

1. **事件、短期征兆、长期风险、心理变化分开输出。** 不再用一个 final_risk 混合所有任务。
2. **骨架优先。** 训练和长期存储尽量使用关键点、置信度和派生特征，不长期保存家庭原始视频。
3. **相位而非二分类。** 模型学习跌倒过程顺序，减少坐下、下蹲、捡东西和主动躺下造成的误报。
4. **双时间尺度。** 短窗口负责快速下降和落地，长窗口负责步态不稳、眩晕样动作和恢复。
5. **质量门控。** 遮挡、低照度、多人混淆和关键点缺失时降低置信度或拒绝高等级结论。
6. **跨数据集验证优先于同数据集高分。** 主结果必须包含留一数据集外测。
7. **个人基线只做增量修正。** 冷启动阶段不得用个人化规则覆盖通用模型。
8. **人机协同。** 预测只建议关注；紧急告警需要跌倒事件证据或人工确认。
9. **可演示但不伪造。** 演示数据、公开数据、真实设备数据和合成数据必须分别标识。

## 4. 系统架构

整体数据流如下：

    萤石 C6c 视频
        ↓
    解码、时间戳、流健康检查
        ↓
    人体检测、跟踪、COCO-17 关键点与置信度
        ↓
    ┌──────────────────────────┬──────────────────────────┐
    │ 短窗口：48 帧，10 fps     │ 长窗口：64 帧，2 fps      │
    │ 覆盖 4.8 秒              │ 覆盖 32 秒               │
    │ PoseC3D 事件表征          │ TCN/骨架特征表征          │
    └─────────────┬────────────┴─────────────┬────────────┘
                  ↓                          ↓
              相位分类头                 跌倒前异常头
                  └───────────┬──────────────┘
                              ↓
                    质量门控与相位约束
                              ↓
             fall_event / prefall_warning / recovery
                              ↓
                 去重、人工确认、告警适配器

与它并行运行：

    日级活动与睡眠摘要 + 自愿 GDS-15 / 短问答
                              ↓
                   wellbeing_change
                              ↓
                   低频邀请或人工关注

U-PMCC 使用日级摘要生成 24 小时、72 小时和 7 天风险研究输出，但不得改变实时跌倒事件结论。

## 5. 公开数据集组合

### 5.1 必选数据

| 数据集 | 本项目用途 | 已知规模与特点 | 许可与处理要求 |
| --- | --- | --- | --- |
| GMDCSA24 v2.1 | 保留现有基线；训练跌倒/ADL 和粗粒度下降阶段 | 4 名受试者；已有视频和时间段标注 | Zenodo 数据为 CC BY 4.0；原始数据不提交 Git |
| Pre-VFall | 核心跌倒前异常监督 | 约 2.2 万张图像，9 名健康年轻参与者，正常/异常/跌倒，45°和90°双视角；压缩包 21,096,978,752 字节 | Figshare 元数据为 CC BY 4.0；下载后固定版本和校验值；不得描述为真实老年跌倒 |
| CAUCAFall v4 | 家庭场景、遮挡、光照、夜间和不同视角泛化 | 10 名受试者，5 类跌倒、5 类 ADL，每名受试者 10 个视频；HIKVISION 红外摄像机 | Mendeley Data 标记 CC BY 4.0；官方源为 https://data.mendeley.com/datasets/7w7fccy7ky/4 |
| 3D skeletons UP-Fall v1 | 冲击时刻辅助监督和快速骨架调试 | 5 名受试者；CSV 含 3D 关节和 impact/non-impact 标签；文件合计约 4 MB | 官方源为 https://zenodo.org/records/12773013；页面未明确显示许可，确认授权前不得重新分发或进入正式训练发布 |
| OmniFall 标注与划分 | 统一时序标签和跨数据集评估协议 | 当前数据卡约 5.26 万条片段标注、9 个来源值、16 类标签；论文描述 8 个公开数据集、101 名受试者、29 个机位 | 当前数据卡标记 CC BY-NC 4.0；原视频仍遵守各上游数据集许可，仓库只保存可再分发的标注和下载说明 |

Pre-VFall 官方源：
https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216

OmniFall 数据卡：
https://huggingface.co/datasets/simplexsigil2/omnifall

OmniFall 论文：
https://arxiv.org/abs/2505.19889

### 5.2 可选数据

| 数据集 | 何时使用 | 限制 |
| --- | --- | --- |
| NTU RGB+D 120 | 只下载 3D skeleton，用 A42 staggering、A43 falling down 及 A5/A6/A8/A9/A80 等困难负样本做骨架预训练 | 仅限学术研究，需要注册并接受协议；禁止重新分发或生成并发布衍生数据集；官方源 https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| CHARLS | 对睡眠、活动、抑郁量表和跌倒之间的群体关联做统计研究或背景论证 | 它是 45 岁及以上中国人群的纵向调查，不是视频/音频模型训练集，不得用来声称摄像头可诊断心理疾病；官方源 https://charls.pku.edu.cn/ |

### 5.3 明确不采用的训练路线

- 不使用 DAIC-WOZ 或 MODMA 训练“老年抑郁识别模型”。人群、采集任务和家庭被动监测场景均不匹配。
- 不把 SisFall、UNIVRFall 等 IMU 数据直接与 C6c 视频特征拼接。没有同一受试者、同一时刻的配对数据时，这种融合会制造虚假对应关系。
- 不下载完整 850 GB 以上的 UP-Fall 多模态原始数据作为第一阶段必需项，投入大而对当前 C6c 演示的边际收益低。
- 不把 OmniFall 的元数据许可当成所有原视频的统一许可。
- 不在 GitHub 提交原始视频、压缩包、关键点缓存、访问令牌、设备验证码或完整设备序列号。

## 6. 数据注册与可追溯性

datasets/manifest.json 必须升级到统一数据注册表。每个数据集条目至少包含：

    {
      "name": "Pre-VFall",
      "version": "figshare-26488216-2025-04-28",
      "official_source": "https://figshare.com/articles/dataset/26488216",
      "license": "CC BY 4.0",
      "expected_size_bytes": 21096978752,
      "sha256": null,
      "download_status": "not_downloaded",
      "subjects": 9,
      "split_strategy": "subject_grouped",
      "redistribution": "raw_data_not_committed",
      "role": ["prefall_supervision", "cross_view"]
    }

实际下载后必须生成 SHA-256，并把 null 替换为真实值。未下载时允许校验值为空，但验证脚本必须区分“尚未下载”和“下载后缺失校验值”。

现有清单中的两处来源需要修正：

- CAUCAFall 从错误或旧地址改为 Mendeley v4：7w7fccy7ky/4。
- 3D skeletons UP-Fall 改为 Zenodo 12773013，并把许可状态标记为 needs_license_confirmation。

原始数据目录固定为：

    datasets/raw/<dataset>/<version>/

统一标注目录固定为：

    datasets/annotations/unified/<schema_version>/<dataset>/

派生骨架目录固定为：

    datasets/processed/pose17/<extractor_version>/<dataset>/

所有派生产物保存：

- 原始文件相对路径；
- 数据集版本；
- subject_id；
- camera_id；
- 开始和结束时间；
- 原标签；
- 统一标签；
- 标签来源；
- 提取器版本；
- 原始文件 SHA-256；
- 是否可用于训练、验证、演示或只作研究。

## 7. 统一标签规范

### 7.1 训练相位

训练目标固定为六个相位：

| phase_id | 名称 | 定义 |
| ---: | --- | --- |
| 0 | normal_adl | 正常站立、行走、坐下、起身、下蹲、捡物、主动躺下等 |
| 1 | prefall_abnormal | 异常摇摆、无支撑失稳、眩晕样动作、明显虚弱、混乱步态等跌倒前征兆 |
| 2 | descending | 身体重心快速不可控下降，尚未确认接触地面 |
| 3 | impact | 身体与地面或低位物体发生主要冲击的短时阶段 |
| 4 | fallen | 跌倒后低位姿态持续，尚未恢复 |
| 5 | recovering | 从跌倒或疑似跌倒状态主动恢复至坐、跪或站 |

unknown、遮挡严重和无法对齐的片段不作为第七类训练，而是通过 supervision_mask 排除对应损失。

### 7.2 困难负样本

以下行为统一标成 normal_adl，但 hard_negative 必须记录具体类型：

- sit_down；
- stand_up；
- pick_up；
- squat；
- kneel；
- intentional_lie_down；
- exercise_bend；
- enter_or_leave_frame；
- pet_or_object_motion；
- partial_occlusion。

评估报告必须单独给出困难负样本误报率。

### 7.3 不同数据集映射

- GMDCSA24：
  - ADL → normal_adl；
  - Falling 时间段 → fall_coarse，只监督 descending、impact、fallen 三者概率之和；
  - 没有精确冲击标签时不得强行生成 impact 标签。
- Pre-VFall：
  - Normal → normal_adl；
  - Abnormal → prefall_abnormal；
  - Fall → fall_coarse；
  - 单帧图像只监督对应分类头，不参与相位顺序损失。
- CAUCAFall：
  - walking、hopping、picking、sitting、kneeling → normal_adl 并记录 hard_negative；
  - 原始 fall/no-fall 可监督事件头；
  - 精细相位仅使用可验证的时序标注，不根据文件名臆造。
- 3D skeletons UP-Fall：
  - LABEL=1 → impact；
  - LABEL=0 不能自动当成 normal_adl，必须结合活动和时间位置建立 mask。
- OmniFall：
  - 使用锁定版本的数据卡标签映射；
  - 数据卡当前 16 类与论文 10 类的差异必须记录在 model card；
  - 它提供统一评估标注，不自动赋予上游原视频再分发权。
- NTU RGB+D 120：
  - A42 staggering → prefall_abnormal 的辅助预训练标签；
  - A43 falling down → fall_coarse；
  - A5、A6、A8、A9、A80 → normal_adl 困难负样本；
  - 只用于预训练或消融，不进入主结果时必须明确标注。

### 7.4 标注记录

统一样本记录接口：

    @dataclass(frozen=True)
    class UnifiedClip:
        clip_id: str
        dataset: str
        dataset_version: str
        subject_id: str
        camera_id: str
        media_path: str
        start_sec: float
        end_sec: float
        phase: str | None
        coarse_event: str | None
        hard_negative: str | None
        supervision_mask: tuple[str, ...]
        source_label: str
        provenance: dict[str, str]

同一 clip_id 必须由数据集版本、受试者、机位、文件和时间段稳定生成。路径变化不能改变 clip_id。

## 8. 核心算法：PA-DTSF

### 8.1 输入

每个跟踪对象产生：

- COCO-17 二维关键点；
- 每个关键点置信度；
- 人体框；
- 帧时间戳；
- 画面尺寸；
- 跟踪连续性；
- 流是否延迟、丢帧或断开。

关键点先按躯干长度归一化，以髋中心平移到原点。缺失点保留 mask，不得填成零后假装为真实坐标。

### 8.2 双时间尺度

短分支：

- 48 帧，目标采样率 10 fps，覆盖约 4.8 秒；
- 复用 PoseC3D 的事件表征；
- 重点识别 descending、impact、fallen；
- 滑动步长 5 帧，理论更新间隔约 0.5 秒。

长分支：

- 64 帧，目标采样率 2 fps，覆盖约 32 秒；
- 输入归一化骨架、置信度 mask 和步态派生特征；
- 使用轻量 TCN，默认 CPU 可运行；
- 重点识别 prefall_abnormal、持续失稳和 recovery。

现有方案中的“1 fps、48 秒 PoseC3D 窗口”不再作为实时事件默认值，因为它会稀释冲击运动并增加延迟。1 fps 只可用于日级活动摘要。

### 8.3 多任务输出

模型输出：

    PhaseModelOutput(
        phase_probs: tuple[float, float, float, float, float, float],
        fall_event_prob: float,
        prefall_prob: float,
        recovery_prob: float,
        quality_score: float,
        embedding_version: str,
        model_version: str,
    )

训练损失固定为：

    total_loss =
        1.0 * masked_phase_cross_entropy
        + 0.8 * prefall_focal_loss
        + 0.7 * fall_event_binary_loss
        + 0.3 * recovery_binary_loss
        + 0.2 * phase_order_penalty

没有某类标签的样本通过 supervision_mask 跳过对应损失。不得用伪造标签补齐。

### 8.4 相位顺序约束

允许的主要迁移：

    normal_adl → prefall_abnormal
    normal_adl → descending
    prefall_abnormal → normal_adl
    prefall_abnormal → descending
    descending → impact
    descending → fallen
    impact → fallen
    fallen → recovering
    recovering → normal_adl

模型可以在证据不足时停留或回到 normal_adl。禁止由单帧噪声直接从 normal_adl 跳到 fallen 并产生确认跌倒。

phase_order_penalty 对高概率非法迁移施加惩罚。推理阶段再用现有 FallStateMachine 的持久性规则平滑，神经网络不替代状态机。

### 8.5 质量门控

quality_score 由以下证据计算：

- 可用关键点比例；
- 关键点平均置信度；
- 躯干和髋部关键点完整性；
- 跟踪 ID 连续性；
- 遮挡比例；
- 有效帧比例；
- 流时间戳新鲜度。

门控规则：

- quality_score 大于等于 0.70：正常输出；
- 0.50 至 0.70：输出降级，预测最高 warning，原因中显示质量不足；
- 小于 0.50：拒绝高置信度预测，只提示检查设备或人工确认；
- 连续 2 秒无可靠人体骨架：不得把“人消失”判断为跌倒。

模型融合采用质量加权：

    gated_embedding =
        short_quality * short_embedding
        + long_quality * long_embedding

权重归一化后再进入多任务头。某个分支不可用时，另一分支可独立工作并标记 degraded。

### 8.6 个人基线

个人化只作用于 prefall_prob，不修改 fall_event_prob。

- 0 至 6 个有效日：population_only；
- 7 至 13 个有效日：warming_up；
- 14 个及以上有效日：personalized；
- 已确认跌倒日、设备故障日、关键点质量不足日不得更新基线；
- 个人修正幅度限制为基础概率的正负 0.15；
- 个人数据不足时回退到总体模型，不得在线训练神经网络。

## 9. 训练策略

### 9.1 阶段顺序

1. 建立统一数据注册、适配器、标签映射和可复现受试者划分。
2. 用 GMDCSA24 和 CAUCAFall 训练事件分支，优先解决困难负样本。
3. 用 Pre-VFall 训练跌倒前异常头。
4. 用 3D skeletons UP-Fall 的 impact 标签辅助冲击头；若许可未确认则跳过此阶段并记录原因。
5. 使用 OmniFall 的锁定标注版本统一相位和进行跨数据集评估。
6. 可选使用 NTU skeleton 子集做预训练，再在主数据上微调。
7. 联合训练双分支和相位约束。
8. 在验证折上做温度缩放校准，测试折只使用冻结阈值。
9. 用 C6c 正常生活录像做域校准和误报评估，不要求真人模拟危险跌倒。

### 9.2 数据划分

- 任何同一受试者的片段不得跨训练、验证和测试。
- 同一原视频切出的窗口必须全部属于同一折。
- 同步多机位属于同一事件组，不能分到不同折。
- 主实验必须包含 leave-one-dataset-out：
  - 每轮用若干公开数据训练；
  - 完整保留一个数据集作为外部测试；
  - 测试数据集不得参与阈值、归一化统计或早停。
- Pre-VFall 只有 9 名健康年轻人，必须同时报告按受试者留一结果，不能按图像随机划分。
- C6c 本地视频只用于域验证或经明确同意后的微调；同一场景不能同时用于模型选择和最终演示评估。

### 9.3 采样与增强

- 每个 batch 按数据集和类别平衡采样，防止最大数据集主导训练。
- 必须加入：
  - 水平翻转；
  - 随机关键点丢失；
  - 置信度噪声；
  - 时间缩放 0.8 至 1.2；
  - 随机裁剪；
  - 轻度旋转；
  - 多人轨迹干扰模拟。
- 不使用会改变动作语义的垂直翻转。
- 夜间鲁棒性通过 CAUCAFall 原视频和关键点缺失增强验证，不通过人为调亮测试图像来掩盖问题。

### 9.4 基线与消融

所有新模型必须与以下基线在相同划分上比较：

1. 现有 gait_stability 固定规则；
2. 已发布 PoseC3D 二分类权重；
3. 仅短分支；
4. 仅长分支；
5. 双分支但无相位顺序约束；
6. 双分支但无质量门控；
7. 完整 PA-DTSF；
8. 可选 NTU 预训练版本。

只有完整模型在主指标或误报指标上稳定优于基线，才能把“相位感知双时间尺度质量门控”写成有效创新。

## 10. 推理与决策契约

### 10.1 事件类型

系统保留四种独立结果：

| kind | 含义 | 最高等级 |
| --- | --- | --- |
| fall_event | 已发生或高度疑似正在发生跌倒 | critical |
| prefall_warning | 秒级跌倒前异常 | warning |
| fall_forecast | 24 小时、72 小时、7 天长期风险 | warning |
| wellbeing_change | 心理或生活规律变化筛查 | warning |

### 10.2 实时决策

建议默认阈值只作为起始值，最终必须由验证折冻结：

- fall_event_prob 小于 0.60：不触发跌倒事件；
- 0.60 至 0.85：warning，要求人工查看；
- 大于等于 0.85，且 descending/impact/fallen 连续证据满足状态机：critical；
- prefall_prob 大于等于验证阈值并持续至少两个更新周期：warning；
- prefall_warning 永远不能单独成为 critical；
- 恢复必须由 recovering → normal_adl 的连续证据或人工确认解除活动跌倒告警。

每个结果必须包含：

- subject_id；
- 时间戳；
- 分数和校准版本；
- 当前相位；
- 证据质量；
- 主要理由；
- 是否降级；
- 推荐动作；
- 模型和数据 schema 版本。

### 10.3 多人场景

- 每个 tracking_id 独立维护窗口和状态机。
- 跟踪 ID 切换时不得继承活动跌倒状态，除非轨迹重识别通过位置、时间和骨架连续性验证。
- 画面中无人时输出 availability 或 no_person，不输出 normal。
- 若两人重叠导致髋部和躯干关键点质量下降，进入 degraded，不猜测目标身份。

## 11. 心理健康模块

### 11.1 保留的方案

- GDS-15：完整筛查邀请冷却 28 天；用户可主动发起。
- 短问答：持续趋势变化后才邀请，冷却 7 天。
- 安静时段：21:00 至次日 08:00 不主动提问，跌倒确认除外。
- 趋势基线：至少 7 个有效日后比较活动、睡眠和自评变化。
- 自伤相关回答：生成独立人工关注事件，不参与跌倒概率计算。

### 11.2 明确边界

- 不持续录音；
- 不从人脸表情、语调或摄像头画面诊断抑郁；
- 不把 GDS-15 表述为医学诊断；
- 不因为单日少活动就主动问答；
- 不因为 fall_forecast 上升而频繁追问；
- 心理模块不向实时跌倒事件提供 critical 证据；
- CHARLS 只能用于群体关联分析和研究背景，不训练家庭摄像头心理分类器。

### 11.3 用户可见语言

允许：

- “最近一周活动规律与个人基线相比有变化，是否愿意做一个简短状态问答？”
- “本结果仅用于筛查和趋势提醒，不构成诊断。”

禁止：

- “系统判断您患有抑郁症。”
- “摄像头已诊断心理疾病。”
- “您将在 72 小时内跌倒。”

## 12. 萤石设备和告警闭环

### 12.1 职责边界

萤石平台负责设备连接、视频地址、设备状态和平台允许的消息能力；本项目服务器负责解码、骨架提取、模型推理和风险决策。不得声称模型在萤石云端运行，除非后续获得明确的算法托管产品与合同能力。

### 12.2 告警适配器

保留 AlertDispatcher 作为本地审计核心，在它之后增加独立 DeliveryAdapter：

    class DeliveryAdapter(Protocol):
        def send(self, decision: RiskDecision) -> DeliveryResult: ...

实现顺序：

1. LocalJsonlDelivery：现有行为；
2. DashboardDelivery：本地界面；
3. WebhookDelivery：向已配置的服务端地址推送；
4. EzvizDelivery：只调用官方文档明确支持且账户已开通的接口；
5. ContactDelivery：短信或电话需独立服务商和用户授权。

摄像头扬声器播报不是默认承诺。只有在 C6c 型号、开放平台套餐和 SDK 实测支持双向语音或语音下发时才启用；否则使用萤石 App 消息、本地页面声音或照护者通知。

### 12.3 密钥

- AppKey、Secret、AccessToken、验证码和完整序列号只放在本机环境变量或未跟踪配置文件。
- 日志、测试夹具、截图、文档和 Git 历史不得包含真实密钥。
- 即使用户允许公开，也不应把可用于设备控制的凭据提交 GitHub。

## 13. 错误处理与安全降级

| 情况 | 系统行为 |
| --- | --- |
| 视频断流 | 标记 camera offline，尝试有上限的重连，不产生新跌倒判断 |
| 帧延迟过大 | 标记 stale，不把旧帧当实时告警 |
| 无人体 | 输出 no_person，不输出 normal |
| 关键点质量不足 | degraded 或 abstained，最高不超过 warning |
| 模型权重缺失或校验失败 | 回退到经过测试的规则路径，明确 model_unavailable |
| SDNL1 无接口 | radar offline，不生成虚构生理数据 |
| 网络推送失败 | 本地告警仍落盘，记录重试状态，不阻塞实时推理 |
| 心理问答存储失败 | 不重复轰炸用户，默认进入冷却并提示系统维护 |
| 多模型版本不兼容 | 拒绝加载，不能静默使用随机权重 |
| 公开数据许可不明确 | 可下载元数据做调查，但不得训练、发布或再分发原始数据 |

## 14. 评估方案

### 14.1 实时跌倒事件

必报：

- Precision、Recall、F1；
- 按受试者和按数据集的混淆矩阵；
- 困难负样本误报率；
- 每小时误报数；
- 事件检测延迟；
- 完整推理 P50 和 P95 延迟；
- 质量拒绝覆盖率。

发布门槛：

- Fall F1 大于等于 0.90；
- Fall Recall 大于等于 0.88；
- 正常生活每小时误报数小于等于 1.0；
- C6c 完整链路 P95 小于等于 2.0 秒；
- 所有指标来自同一 release_id、同一模型和冻结阈值。

### 14.2 跌倒前异常

主指标不是 Accuracy，而是：

- AUPRC；
- ROC-AUC；
- 在 90% 召回率附近的 FPR；
- 从首次有效预警到下降开始的提前量；
- 每小时错误预警数；
- 预警持续性和撤销次数。

晋级条件：

- 相对现有 gait_stability 规则，AUPRC 至少相对提高 5%；或
- 在召回率不低于现有规则时，FPR 相对下降至少 20%；
- 中位有效提前量必须大于 0 秒；
- leave-one-dataset-out 中至少三轮不劣于二分类基线；
- 没有满足条件时保留 research_only，不替换默认规则。

### 14.3 跨数据集泛化

至少报告：

- 每个留出数据集上的 macro F1、fall recall 和 false positives；
- 同数据集与跨数据集性能差；
- 白天、夜间、遮挡、不同机位分层结果；
- 模型是否使用 NTU 预训练；
- 每个上游数据集的版本和许可状态。

不能只挑最好的一折作为最终成绩。

### 14.4 心理健康

心理模块不报告“抑郁识别准确率”。只报告：

- 邀请次数；
- 用户接受率；
- 7 天和 28 天冷却遵守率；
- 安静时段误触发数；
- 趋势变化解释完整率；
- 自伤回答转人工关注的流程测试；
- 缺失数据和退出授权后的停止率。

### 14.5 真实设备证据

至少采集：

1. C6c 连续 30 分钟正常生活视频：
   - 走动；
   - 坐下、起身；
   - 捡物；
   - 下蹲；
   - 短时遮挡；
   - 离开画面再进入。
2. 安全的非危险演示：
   - 可用公开跌倒视频回放到系统；
   - 或由工作人员在软垫和保护人员在场时演示，但这不是必需条件。
3. 记录：
   - 视频来源；
   - 时间段；
   - 模型版本；
   - 阈值；
   - P50/P95；
   - 每小时误报数；
   - 设备断流和重连次数。

不得要求老人真实模拟跌倒。

## 15. 报告和模型卡

每次训练输出独立 release_id，目录包含：

    experiments/releases/<release_id>/
        dataset_lock.json
        split_manifest.json
        config.py
        metrics.json
        per_subject_metrics.csv
        per_dataset_metrics.csv
        confusion_matrices.json
        calibration.json
        ablation.json
        benchmark.json
        model_card.md
        checksums.sha256

model_card.md 必须写明：

- 模型用途和禁止用途；
- 训练数据版本；
- 模拟年轻人数据限制；
- 标签映射版本；
- 划分策略；
- 阈值来源；
- 关键指标及置信区间；
- C6c 实测是否完成；
- SDNL1 是否真实可用；
- 许可与再分发限制；
- promoted 为 true 或 false；
- 失败模式和降级行为。

任何缺失指标显示 unavailable，不得写成 0，也不得从不同 release_id 拼接成一次通过。

## 16. 代码边界

建议新增以下独立模块，避免继续扩大已有文件：

    datasets/unified/
        schema.py
        registry.py
        splits.py
        labels.py
        adapters/
            gmdcsa24.py
            prevfall.py
            caucafall.py
            upfall3d.py
            omnifall.py
            ntu120.py

    risk/phase_model/
        schema.py
        normalization.py
        windows.py
        quality.py
        transitions.py
        model.py
        losses.py
        calibration.py
        service.py

    alerts/delivery/
        base.py
        local_jsonl.py
        dashboard.py
        webhook.py
        ezviz.py

    scripts/
        prepare_unified_fall_data.py
        train_phase_model.py
        evaluate_phase_model.py
        benchmark_c6c_pipeline.py
        generate_release_report.py

每个文件只承担一种职责。risk/phase_model/service.py 只做编排，不包含训练循环；datasets/unified/adapters 只解释源数据，不执行模型推理。

必须保持兼容：

- vision.pose_pipeline.PoseFrameResult；
- risk.gait_stability.GaitStabilityAnalyzer；
- risk.fall_state_machine 的现有公开行为；
- core.events.SensorEvent；
- fusion.decision_engine.RiskDecision；
- pipeline.live_service.LiveMonitoringService；
- mental 包的筛查和冷却规则。

如果新接口需要改变这些类型，先增加适配器和回归测试，不得一次性重写整个系统。

## 17. 测试规格

### 17.1 数据测试

- 每个适配器能把一个官方样例转换为 UnifiedClip。
- 缺少 subject_id 的样本拒绝进入受试者级评估。
- 同一受试者和同步机位不会跨折。
- 原视频切片不会跨折。
- fall_coarse 不会被错误当作 impact。
- unknown 和低质量标签正确设置 supervision_mask。
- 数据注册表来源、版本、大小和许可字段通过校验。
- 非商业或禁止再分发的数据不会进入发布包。

### 17.2 模型测试

- 两个窗口使用真实时间戳采样，不假设输入帧率恒定。
- 关键点缺失保留 mask。
- 无人体时不调用姿态模型。
- 单帧噪声不能直接触发 confirmed fall。
- prefall_warning 不能产生 critical。
- quality_score 小于 0.5 时拒绝高置信度结论。
- 非法相位跳转被惩罚或状态机拦截。
- 相同权重、输入、配置和随机种子得到相同输出。
- 短分支、长分支任一不可用时可以安全降级。

### 17.3 心理健康测试

- 21:00 至 08:00 不主动邀请。
- 完整 GDS-15 邀请遵守 28 天冷却。
- 短问答遵守 7 天冷却。
- 用户主动发起不受被动邀请冷却阻止。
- 单日活动下降不触发持续变化。
- 自伤相关回答只进入人工关注流程。
- 输出中没有诊断性语言。

### 17.4 端到端测试

固定流程：

    正常走动
    → 坐下和捡物困难负样本
    → 跌倒前异常
    → 快速下降
    → 冲击
    → 倒地
    → 恢复
    → 告警解除

同时验证：

- 事件顺序；
- 告警去重；
- 恢复后重新布防；
- 本地记录；
- 推送失败不阻塞；
- 断流降级；
- 质量原因可见；
- 演示和真实证据标识不混淆。

## 18. 实施批次

### 批次 A：数据底座

- 修正 datasets/manifest.json；
- 增加统一 schema、标签映射、适配器和受试者级划分；
- 下载顺序：3D skeletons UP-Fall 元数据核验 → CAUCAFall → Pre-VFall；
- OmniFall 先下载标注和划分，不自动下载无权再分发的上游视频；
- 输出 dataset_lock.json 和统计报告。

完成标志：所有必选数据都能生成统一标注；许可不明的数据被自动阻止进入正式训练。

### 批次 B：强基线

- 在 GMDCSA24 + CAUCAFall 上重训 PoseC3D 事件分支；
- 增加困难负样本；
- 复现旧权重结果并在同划分比较；
- 固定阈值和模型卡。

完成标志：同数据集指标不退化，且困难负样本误报下降。

### 批次 C：核心创新

- 实现双时间窗口；
- 实现长时 TCN、相位头、质量门控和相位顺序损失；
- 接入 Pre-VFall；
- 许可允许时接入 impact 数据；
- 完成消融。

完成标志：达到第 14.2 节晋级条件，否则保持 research_only。

### 批次 D：跨域评估

- 锁定 OmniFall 标签版本；
- 运行 leave-one-dataset-out；
- 输出逐受试者、逐数据集、逐场景报告；
- 进行概率校准。

完成标志：所有结果来自冻结配置，报告不挑选最佳折。

### 批次 E：C6c 实时闭环

- 以 10 fps 短窗口和 2 fps 长窗口接入实时流；
- 测量完整链路 P50/P95；
- 采集 30 分钟正常活动；
- 统计每小时误报；
- 接入本地页面和至少一种真实可验证的通知方式。

完成标志：第 14.1 节全部发布门槛有真实证据。

### 批次 F：心理健康和长期风险

- 保留 GDS-15、趋势和低打扰策略；
- 将日级活动摘要送入 U-PMCC；
- 所有合成 U-PMCC 结果继续 promoted=false；
- 可选使用 CHARLS 做独立统计研究，不与视频模型联合训练。

完成标志：心理流程可演示、低打扰、无诊断性结论，长期风险与实时事件完全分流。

## 19. 完成定义

项目只有同时满足以下条件才可称为“算法和演示完整”：

- 数据来源、版本、许可、校验和和受试者划分可追溯；
- PA-DTSF 和所有基线使用相同划分完成评估；
- 主结果包含留一数据集外测；
- 跌倒事件 F1、召回率、每小时误报和端到端延迟达到发布门槛；
- 跌倒前分支达到相对规则基线的晋级条件；
- C6c 完成至少 30 分钟正常生活实测；
- 预测、事件、心理变化和长期风险分开显示；
- 只有 fall_event 可以产生 critical；
- 网络或模型失败时能够安全降级；
- 心理健康模块不持续监听、不诊断、不频繁提问；
- SDNL1 未获得真实接口前始终明确显示 unavailable；
- 所有演示、合成和真实结果有明显来源标签；
- GitHub 中不包含原始数据和任何真实设备密钥。

## 20. 交给其他 AI 的启动指令

将下面文字连同本文档路径交给编码 AI：

    请完整阅读：
    docs/superpowers/specs/2026-08-03-multidataset-fall-wellbeing-optimization-design.md

    本文档是当前项目的唯一设计依据。先检查当前分支、已有代码、测试、
    数据清单和未跟踪文件，不要覆盖用户的 outputs。

    实施规则：
    1. 先制定逐任务实施计划，再修改代码。
    2. 每项功能先写失败测试，再写最小实现。
    3. 每个任务只做一个可独立评审、可独立测试的改动。
    4. 不改变已有公开接口；需要变化时先增加兼容适配器。
    5. 原始视频、压缩包、关键点缓存和受限数据不提交 Git。
    6. 不提交 AppKey、Secret、AccessToken、验证码或完整设备序列号。
    7. 不把不同受试者划分策略或不同 release_id 的指标拼接。
    8. 模拟、合成、公开和真实设备证据必须分别标识。
    9. 未达到晋级门槛的模型保持 research_only 和 promoted=false。
    10. 只有 fall_event 可以产生 critical。
    11. 不对心理健康作诊断，不要求老人模拟跌倒。
    12. 每个任务完成后运行局部测试；每批次结束运行完整回归测试。
    13. 每次提交前检查 git diff，只提交本任务文件。
    14. 任何公开数据源或许可发生变化时停止下载并记录证据，不猜测。

    实施顺序严格按本文档第18节，从批次A开始。
    第一项工作是修正数据注册表并为统一数据 schema 写失败测试。

## 21. 参考来源

- GMDCSA24：https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
- Pre-VFall：https://figshare.com/articles/dataset/26488216
- CAUCAFall v4：https://data.mendeley.com/datasets/7w7fccy7ky/4
- 3D skeletons UP-Fall：https://zenodo.org/records/12773013
- OmniFall 数据卡：https://huggingface.co/datasets/simplexsigil2/omnifall
- OmniFall 论文：https://arxiv.org/abs/2505.19889
- NTU RGB+D / 120：https://rose1.ntu.edu.sg/dataset/actionRecognition/
- CHARLS：https://charls.pku.edu.cn/

## 22. 设计结论

推荐路线不是简单堆叠更多模型，而是用公开数据补齐三个证据缺口：

- Pre-VFall 补跌倒前异常；
- CAUCAFall 补家庭环境、遮挡和夜间泛化；
- UP-Fall impact 标签补落地冲击；
- OmniFall 补统一标签和跨数据集评估。

在工程上，用“短时事件分支 + 长时征兆分支 + 相位顺序 + 质量门控”形成一个可讲清、可做消融、可在 C6c 上运行的核心创新。心理健康则坚持低负担筛查和个人趋势，不追求不可信的摄像头心理诊断。最终评分提升依赖可复现外测、低误报、真实设备延迟和完整演示闭环，而不是只提高单一准确率。
