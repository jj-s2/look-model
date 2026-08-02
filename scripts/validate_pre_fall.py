"""验证跌倒前预判有效性。

流程：
1. 解析 metadata.classes_raw，定位 Fall 视频中跌倒起始时间
2. 截取"跌倒前 N 秒"片段作为高风险样本
3. 截取 ADL 视频等长片段作为低风险样本
4. 用 GaitStabilityAnalyzer 计算指标
5. 对比 Fall 前 vs ADL 的指标分布与风险评分区分度

输出：
  - 每类的指标均值/标准差对比表
  - 风险评分分布直方图
  - ROC-AUC（区分 Fall 前 vs ADL 的能力）
  - 不同提前量（跌倒前 3/5/10 秒）的预判召回率
"""
import argparse
import os
import re
import csv
import pickle
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Validate pre-fall risk prediction.")
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
project_dir = str(args.project_root.expanduser().resolve())
POSE_DIR = os.path.join(project_dir, "datasets", "processed", "gmdcsa24_pose")
METADATA_CSV = os.path.join(project_dir, "datasets", "metadata.csv")

import sys
sys.path.insert(0, project_dir)
from risk.gait_stability import GaitStabilityAnalyzer


def parse_fall_onset(classes_raw):
    """从 classes_raw 解析跌倒起始时间（秒）。

    示例：
      "Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]" -> 3.4
      "Falling[1.9 to 6]; ..." -> 1.9

    Returns:
        float or None: 跌倒起始时间（秒），无 Falling 段则 None
    """
    if not classes_raw:
        return None
    # 匹配 Fallingxxx[start to end]
    m = re.search(r'Falling[^\[]*\[(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\]', classes_raw)
    if m:
        return float(m.group(1))
    return None


def load_pose(frame_dir):
    """加载单视频的 pose pickle。"""
    path = os.path.join(POSE_DIR, f"{frame_dir}.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_pre_fall_segment(anno, fall_onset_sec, fps, pre_sec=5.0):
    """截取跌倒前 pre_sec 秒的骨架片段。

    Args:
        anno: pose pickle dict
        fall_onset_sec: 跌倒起始时间（秒）
        fps: 视频帧率
        pre_sec: 提前量（秒）

    Returns:
        (keypoint_segment, keypoint_score_segment) 或 None
    """
    kp = anno['keypoint']  # (1, T, 17, 2)
    sc = anno.get('keypoint_score', None)
    T = kp.shape[1]

    fall_onset_frame = int(fall_onset_sec * fps)
    start = max(0, fall_onset_frame - int(pre_sec * fps))
    end = fall_onset_frame

    if end - start < 10:  # 片段太短
        return None

    seg_kp = kp[0, start:end]  # (T_seg, 17, 2)
    seg_sc = sc[0, start:end] if sc is not None else None
    return seg_kp, seg_sc


def extract_adl_segment(anno, fps, duration_sec=5.0):
    """从 ADL 视频截取等长片段（取中段，避免开头结尾噪声）。"""
    kp = anno['keypoint']
    sc = anno.get('keypoint_score', None)
    T = kp.shape[1]
    seg_len = int(duration_sec * fps)
    if T < seg_len:
        seg_kp = kp[0]
        seg_sc = sc[0] if sc is not None else None
    else:
        start = (T - seg_len) // 2
        seg_kp = kp[0, start:start + seg_len]
        seg_sc = sc[0, start:start + seg_len] if sc is not None else None
    return seg_kp, seg_sc


def main():
    print("=== 跌倒前预判有效性验证 ===\n")

    # 读取 metadata
    with open(METADATA_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    analyzer = GaitStabilityAnalyzer(fps=29.7, window_sec=3.0)

    # 收集样本
    pre_fall_results = []  # Fall 前片段
    adl_results = []       # ADL 片段

    fall_count = 0
    fall_parsed = 0
    fall_seg_ok = 0
    adl_count = 0
    adl_seg_ok = 0

    for row in rows:
        subject = row['subject_id']
        category = row['category']
        video_id = row['video_id']
        label = row['label']
        classes_raw = row.get('classes_raw', '')
        fps = float(row['fps'])
        frame_dir = f"S{subject}_{category}{video_id}"

        anno = load_pose(frame_dir)
        if anno is None:
            continue

        if label.lower() == 'fall':
            fall_count += 1
            onset = parse_fall_onset(classes_raw)
            if onset is None:
                continue
            fall_parsed += 1

            # 截取跌倒前 5 秒
            seg = extract_pre_fall_segment(anno, onset, fps, pre_sec=5.0)
            if seg is None:
                continue
            fall_seg_ok += 1
            seg_kp, seg_sc = seg
            result = analyzer.analyze(seg_kp, seg_sc)
            result['frame_dir'] = frame_dir
            result['subject'] = subject
            result['fall_onset'] = onset
            pre_fall_results.append(result)
        else:
            adl_count += 1
            seg = extract_adl_segment(anno, fps, duration_sec=5.0)
            if seg is None:
                continue
            adl_seg_ok += 1
            seg_kp, seg_sc = seg
            result = analyzer.analyze(seg_kp, seg_sc)
            result['frame_dir'] = frame_dir
            result['subject'] = subject
            adl_results.append(result)

    print(f"Fall 视频: {fall_count} 个, 解析到跌倒时间: {fall_parsed}, "
          f"成功截取前片段: {fall_seg_ok}")
    print(f"ADL 视频: {adl_count} 个, 成功截取片段: {adl_seg_ok}\n")

    if not pre_fall_results or not adl_results:
        print("样本不足，无法验证")
        return

    # 指标对比
    indicators = ['activity_level', 'activity_trend', 'com_vertical_drop',
                  'com_vel_y', 'activity_burst', 'lean_trend',
                  'com_sway', 'body_lean_angle', 'body_lean_var',
                  'gait_jitter', 'confidence', 'risk_score']

    print("=== 指标对比（Fall 前 5s vs ADL）===\n")
    print(f"{'指标':<22} {'Fall前均值':<12} {'Fall前std':<12} "
          f"{'ADL均值':<12} {'ADL std':<12} {'区分度':<10}")
    print("-" * 90)

    fall_vals = {}
    adl_vals = {}
    for ind in indicators:
        fv = [r[ind] for r in pre_fall_results]
        av = [r[ind] for r in adl_results]
        fall_vals[ind] = np.array(fv)
        adl_vals[ind] = np.array(av)
        fm, fs = np.mean(fv), np.std(fv)
        am, as_ = np.mean(av), np.std(av)
        # 区分度：均值差 / 合并 std
        pooled_std = np.sqrt((fs**2 + as_**2) / 2) if (fs + as_) > 0 else 1
        sep = abs(fm - am) / pooled_std if pooled_std > 0 else 0
        print(f"{ind:<22} {fm:<12.4f} {fs:<12.4f} {am:<12.4f} {as_:<12.4f} {sep:<10.4f}")

    # 风险评分 ROC-AUC（手动实现，避免 sklearn 依赖）
    print("\n=== 风险评分区分能力 ===\n")
    risk_fall = fall_vals['risk_score']
    risk_adl = adl_vals['risk_score']

    # 手动计算 AUC：正例=Fall前，负例=ADL
    # AUC = P(score_pos > score_neg)
    n_pos, n_neg = len(risk_fall), len(risk_adl)
    if n_pos > 0 and n_neg > 0:
        correct = 0
        for fp in risk_fall:
            for fn in risk_adl:
                if fp > fn:
                    correct += 1
                elif fp == fn:
                    correct += 0.5
        auc = correct / (n_pos * n_neg)
    else:
        auc = 0.5
    print(f"风险评分 ROC-AUC: {auc:.4f}")

    # 不同阈值下的表现
    print(f"\n{'阈值':<8} {'TPR(Fall前检出)':<18} {'FPR(ADL误报)':<18} {'F1':<8}")
    print("-" * 60)
    for thresh in np.arange(0.1, 0.9, 0.1):
        tp = np.sum(risk_fall >= thresh)
        fp = np.sum(risk_adl >= thresh)
        fn = np.sum(risk_fall < thresh)
        tn = np.sum(risk_adl < thresh)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0
        print(f"{thresh:<8.2f} {tpr:<18.4f} {fpr:<18.4f} {f1:<8.4f}")

    # 不同提前量的预判能力
    print("\n=== 不同提前量（跌倒前 N 秒）的预判能力 ===\n")
    for pre_sec in [3.0, 5.0, 10.0]:
        pre_results = []
        for row in rows:
            if row['label'].lower() != 'fall':
                continue
            frame_dir = f"S{row['subject_id']}_{row['category']}{row['video_id']}"
            anno = load_pose(frame_dir)
            if anno is None:
                continue
            onset = parse_fall_onset(row.get('classes_raw', ''))
            if onset is None:
                continue
            fps = float(row['fps'])
            seg = extract_pre_fall_segment(anno, onset, fps, pre_sec=pre_sec)
            if seg is None:
                continue
            seg_kp, seg_sc = seg
            r = analyzer.analyze(seg_kp, seg_sc)
            pre_results.append(r['risk_score'])

        if not pre_results:
            continue
        pre_arr = np.array(pre_results)
        # 阈值 0.5 时的检出率
        detect_rate = np.mean(pre_arr >= 0.5)
        print(f"跌倒前 {pre_sec:.0f}s: 样本={len(pre_arr)}, "
              f"风险评分均值={np.mean(pre_arr):.4f}, "
              f"阈值0.5检出率={detect_rate:.4f}")

    # 按受试者分组看 Fall 前风险评分
    print("\n=== 按受试者分组的 Fall 前风险评分 ===\n")
    print(f"{'受试者':<8} {'样本数':<8} {'均值':<10} {'std':<10} {'检出率(>=0.5)':<12}")
    print("-" * 50)
    for sub in ['1', '2', '3', '4']:
        sub_risks = [r['risk_score'] for r in pre_fall_results if r['subject'] == sub]
        if not sub_risks:
            continue
        arr = np.array(sub_risks)
        det = np.mean(arr >= 0.5)
        print(f"S{sub:<7} {len(arr):<8} {np.mean(arr):<10.4f} "
              f"{np.std(arr):<10.4f} {det:<12.4f}")

    # 保存结果
    out_path = os.path.join(project_dir, "experiments", "outputs",
                            "gait_stability_validation.json")
    summary = {
        'fall_pre_samples': len(pre_fall_results),
        'adl_samples': len(adl_results),
        'roc_auc': float(auc),
        'fall_pre_risk_mean': float(np.mean(risk_fall)),
        'adl_risk_mean': float(np.mean(risk_adl)),
        'indicators': {
            ind: {
                'fall_mean': float(np.mean(fall_vals[ind])),
                'fall_std': float(np.std(fall_vals[ind])),
                'adl_mean': float(np.mean(adl_vals[ind])),
                'adl_std': float(np.std(adl_vals[ind])),
            }
            for ind in indicators
        }
    }
    import json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
