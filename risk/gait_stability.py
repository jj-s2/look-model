"""跌倒前预判核心模块：基于骨架时序的步态稳定性评分。

区别于 PoseC3D 的"跌倒后二分类"，本模块从连续骨架序列中提取步态稳定性指标，
输出连续风险评分 [0,1]，用于跌倒发生前的风险预判。

COCO 17 关键点索引：
  0: nose            1: l_eye        2: r_eye
  3: l_ear           4: r_ear
  5: l_shoulder      6: r_shoulder
  7: l_elbow         8: r_elbow      9: l_wrist   10: r_wrist
  11: l_hip          12: r_hip
  13: l_knee         14: r_knee
  15: l_ankle        16: r_ankle
"""
import numpy as np

# 关键点索引常量
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


class GaitStabilityAnalyzer:
    """从骨架序列提取步态稳定性指标并输出风险评分。

    用法：
        analyzer = GaitStabilityAnalyzer()
        result = analyzer.analyze(keypoints, keypoint_scores=None)
        # result: dict, 含各指标值与 risk_score
    """

    def __init__(self, fps=29.7, window_sec=3.0):
        """
        Args:
            fps: 视频帧率（GMDCSA24 默认 ~29.7）
            window_sec: 滑动窗口长度（秒），用于计算时序统计量
        """
        self.fps = fps
        self.window = max(int(fps * window_sec), 1)

    def analyze(self, keypoints, keypoint_scores=None):
        """分析骨架序列，返回步态稳定性指标与风险评分。

        Args:
            keypoints: np.ndarray, shape (T, 17, 2) 或 (N, T, 17, 2)
                      单人场景取第一人
            keypoint_scores: np.ndarray, shape (T, 17) 或 (N, T, 17)，可选

        Returns:
            dict: {
                'activity_level': float,        # 活动量（像素/帧）
                'activity_trend': float,        # 活动量变化率（后段-前段）
                'com_height': float,            # 重心垂直高度（像素）
                'com_vertical_drop': float,     # 重心垂直下降量（像素）
                'com_sway': float,              # 重心横向摆动标准差（像素）
                'body_lean_angle': float,       # 身体倾斜角（度）
                'body_lean_var': float,         # 倾斜角方差
                'gait_jitter': float,           # 步态抖动度（脚踝速度方差）
                'confidence': float,            # 关键点平均置信度
                'risk_score': float,            # 综合风险评分 [0,1]
            }
        """
        kp = self._normalize_keypoints(keypoints)
        scores = self._normalize_scores(keypoint_scores, kp.shape[0])
        T = kp.shape[0]

        if T < 3:
            return self._empty_result()

        # 1. 活动量：所有关键点帧间位移均值
        activity = self._calc_activity(kp)  # (T-1,)

        # 2. 重心（髋部中点）
        com = (kp[:, L_HIP] + kp[:, R_HIP]) / 2.0  # (T, 2)
        com_x, com_y = com[:, 0], com[:, 1]

        # 3. 重心横向摆动
        com_sway = float(np.std(com_x)) if T > 1 else 0.0

        # 4. 重心垂直下降（后段均值 - 前段均值，y增大=下降）
        half = T // 2
        if half > 0:
            com_vertical_drop = float(np.mean(com_y[half:]) - np.mean(com_y[:half]))
        else:
            com_vertical_drop = 0.0

        # 5. 身体倾斜角（肩-髋向量与垂直方向夹角）
        shoulder_mid = (kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2.0
        torso_vec = shoulder_mid - com  # (T, 2), dx, dy
        # 与垂直向下方向 (0,1) 的夹角（度）
        # angle = atan2(|dx|, |dy|) * 180/pi
        dx, dy = torso_vec[:, 0], torso_vec[:, 1]
        angles = np.degrees(np.arctan2(np.abs(dx), np.maximum(np.abs(dy), 1e-6)))
        body_lean_angle = float(np.mean(angles))
        body_lean_var = float(np.var(angles))

        # 6. 步态抖动度：脚踝速度方差
        ankles = kp[:, [L_ANKLE, R_ANKLE]]  # (T, 2, 2)
        if T > 2:
            ankle_vel = np.diff(ankles, axis=0)  # (T-1, 2, 2)
            ankle_speed = np.linalg.norm(ankle_vel, axis=2)  # (T-1, 2)
            gait_jitter = float(np.var(ankle_speed))
        else:
            gait_jitter = 0.0

        # 7. 活动量趋势（后段均值 - 前段均值）
        if len(activity) >= 2:
            ah = len(activity) // 2
            activity_trend = float(np.mean(activity[ah:]) - np.mean(activity[:ah]))
        else:
            activity_trend = 0.0

        # 8. 置信度
        confidence = float(np.mean(scores)) if scores is not None else 1.0

        # 9. 末段变化率指标（核心：跌倒前兆是"最后几帧的突变"）
        last_n = min(int(self.fps * 1.0), T)  # 最后 1 秒
        if last_n < 3:
            last_n = min(3, T)
        # 末段重心垂直速度（正=下降，图像 y 增大）
        if T > last_n:
            com_vel_y = float(np.mean(np.diff(com_y[-last_n:])))
        else:
            com_vel_y = 0.0
        # 末段活动量突增比（末段 / 前段）
        if len(activity) > last_n and np.mean(activity[:-last_n]) > 1e-3:
            activity_burst = float(np.mean(activity[-last_n:]) /
                                   max(np.mean(activity[:-last_n]), 1e-3))
        else:
            activity_burst = float(np.mean(activity[-last_n:]) if len(activity) > 0 else 0)
        # 末段倾斜角增大
        if T > last_n:
            lean_trend = float(np.mean(angles[-last_n:]) - np.mean(angles[:-last_n]))
        else:
            lean_trend = 0.0

        # 10. 综合风险评分（规则版，聚焦末段突变）
        risk_score = self._rule_based_risk(
            com_vertical_drop=com_vertical_drop,
            com_vel_y=com_vel_y,
            activity_burst=activity_burst,
            lean_trend=lean_trend,
            body_lean_angle=body_lean_angle,
            com_sway=com_sway,
            gait_jitter=gait_jitter,
            activity_trend=activity_trend,
            kp=kp,
        )

        return {
            'activity_level': float(np.mean(activity)),
            'activity_trend': activity_trend,
            'com_height': float(np.mean(com_y)),
            'com_vertical_drop': com_vertical_drop,
            'com_vel_y': com_vel_y,
            'activity_burst': activity_burst,
            'com_sway': com_sway,
            'body_lean_angle': body_lean_angle,
            'body_lean_var': body_lean_var,
            'lean_trend': lean_trend,
            'gait_jitter': gait_jitter,
            'confidence': confidence,
            'risk_score': float(risk_score),
        }

    def analyze_windowed(self, keypoints, keypoint_scores=None, stride=None):
        """滑动窗口分析，返回每窗的风险评分序列。

        Args:
            keypoints: (T, 17, 2)
            stride: 窗口步长（帧），默认 = window/2（50% 重叠）

        Returns:
            list[dict]: 每个窗口的分析结果，含 'frame_start', 'frame_end'
        """
        kp = self._normalize_keypoints(keypoints)
        scores = self._normalize_scores(keypoint_scores, kp.shape[0])
        T = kp.shape[0]
        stride = stride or max(self.window // 2, 1)

        results = []
        for start in range(0, max(T - self.window + 1, 1), stride):
            end = start + self.window
            if end > T:
                end = T
            seg_kp = kp[start:end]
            seg_sc = scores[start:end] if scores is not None else None
            r = self.analyze(seg_kp, seg_sc)
            r['frame_start'] = start
            r['frame_end'] = end
            results.append(r)
        return results

    # ---------- 内部方法 ----------

    def _normalize_keypoints(self, keypoints):
        """统一为 (T, 17, 2)。"""
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim == 4:
            kp = kp[0]  # 取第一人
        elif kp.ndim == 2:
            kp = kp[None]  # 单帧
        return kp

    def _normalize_scores(self, scores, T):
        """统一为 (T, 17) 或 None。"""
        if scores is None:
            return None
        sc = np.asarray(scores, dtype=np.float32)
        if sc.ndim == 3:
            sc = sc[0]
        elif sc.ndim == 1:
            sc = sc[None]
        return sc

    def _calc_activity(self, kp):
        """计算帧间活动量（所有关键点位移均值）。"""
        if kp.shape[0] < 2:
            return np.zeros(1, dtype=np.float32)
        diff = np.diff(kp, axis=0)  # (T-1, 17, 2)
        return np.mean(np.linalg.norm(diff, axis=2), axis=1)  # (T-1,)

    def _rule_based_risk(self, com_vertical_drop, com_vel_y, activity_burst,
                         lean_trend, body_lean_angle, com_sway, gait_jitter,
                         activity_trend, kp):
        """规则版风险评分（聚焦末段突变，后续用训练数据替换为学习版）。

        核心征兆（基于 GMDCSA24 数据观察修正）：
          - 重心末段下降速度（跌倒最直接前兆）
          - 末段活动量突增（从静坐到异常动作的过渡）
          - 倾斜角末段增大趋势
          - 整窗重心下降量（辅助）
        注意：GMDCSA24 Fall 多为"坐着→跌倒"，整窗绝对活动量反而低于 ADL，
        因此聚焦"末段变化率"而非"整窗统计量"。
        """
        # 末段重心下降速度（像素/帧），正=下降，>1 视为高风险
        r_com_vel = self._sigmoid_risk(com_vel_y, threshold=1.0, scale=0.8)
        # 末段活动量突增比，>1.5 视为高风险（末段活动量是前段 1.5 倍）
        r_act_burst = self._sigmoid_risk(activity_burst, threshold=1.5, scale=0.8)
        # 倾斜角末段增大，>3 度视为高风险
        r_lean_trend = self._sigmoid_risk(lean_trend, threshold=3.0, scale=3.0)
        # 整窗重心下降量，>15 视为高风险
        r_drop = self._sigmoid_risk(com_vertical_drop, threshold=15.0, scale=8.0)

        # 加权融合（基于 GMDCSA24 数据区分度校准）
        # activity_burst 区分度 0.75 > lean_trend 0.67 > com_vel 0.22 > drop 0.08
        weights = {
            'act_burst': 0.40,    # 活动突增（区分度最高）
            'lean_trend': 0.35,   # 倾斜增大趋势
            'com_vel': 0.15,      # 末段重心下降速度
            'drop': 0.10,         # 整窗下降量
        }
        risk = (weights['act_burst'] * r_act_burst +
                weights['lean_trend'] * r_lean_trend +
                weights['com_vel'] * r_com_vel +
                weights['drop'] * r_drop)
        return float(np.clip(risk, 0.0, 1.0))

    def _sigmoid_risk(self, value, threshold, scale):
        """将指标值映射到 [0,1] 风险贡献。value 越大风险越高。"""
        return float(1.0 / (1.0 + np.exp(-(value - threshold) / scale)))

    def _empty_result(self):
        return {
            'activity_level': 0.0, 'activity_trend': 0.0,
            'com_height': 0.0, 'com_vertical_drop': 0.0,
            'com_vel_y': 0.0, 'activity_burst': 0.0,
            'com_sway': 0.0, 'body_lean_angle': 0.0, 'body_lean_var': 0.0,
            'lean_trend': 0.0,
            'gait_jitter': 0.0, 'confidence': 0.0, 'risk_score': 0.0,
        }
