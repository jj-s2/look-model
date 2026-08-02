"""阈值敏感度分析：在不同 Fall 阈值下评估测试集表现。

模型默认阈值 0.5 导致所有 Fall 漏检。
本脚本扫描阈值 0.05~0.5，找出最优阈值。
"""
import os
import pickle
import numpy as np

project_dir = r"C:\Users\John\Desktop\look model"

# 读取预测结果
results_path = os.path.join(project_dir, "experiments", "outputs",
                            "posec3d_gmdcsa24", "test_results.pkl")
with open(results_path, 'rb') as f:
    preds = pickle.load(f)

# 读取 GT
gt_path = os.path.join(project_dir, "datasets", "annotations", "gmdcsa24_pose_test.pkl")
with open(gt_path, 'rb') as f:
    gt_data = pickle.load(f)

# 提取 Fall 类概率
fall_probs = []  # 每个样本的 Fall 类概率
for p in preds:
    if isinstance(p, dict):
        score = p['pred_score']
        if hasattr(score, 'numpy'):
            score = score.numpy()
        fall_probs.append(float(score[1]))  # Fall 类概率
    else:
        fall_probs.append(float(p[1]))

gt_labels = [a['label'] for a in gt_data]
frame_dirs = [a['frame_dir'] for a in gt_data]

print("=== 各样本 Fall 概率 ===")
print(f"{'样本':<15} {'GT':<6} {'Fall概率':<10} {'正确?':<6}")
for fd, gt, prob in zip(frame_dirs, gt_labels, fall_probs):
    gt_name = 'Fall' if gt == 1 else 'ADL'
    # 默认阈值 0.5 下的预测
    pred = 1 if prob > 0.5 else 0
    correct = '✓' if pred == gt else '✗'
    print(f"{fd:<15} {gt_name:<6} {prob:.4f}    {correct}")

# 阈值扫描
print(f"\n=== 阈值扫描 ===")
print(f"{'阈值':<8} {'Acc':<8} {'Fall_P':<8} {'Fall_R':<8} {'Fall_F1':<8} "
      f"{'ADL_P':<8} {'ADL_R':<8} {'TP':<4} {'FP':<4} {'FN':<4}")

best_f1 = 0
best_thresh = 0.5
best_metrics = None

for thresh in np.arange(0.05, 0.55, 0.05):
    pred_labels = [1 if p > thresh else 0 for p in fall_probs]

    # 混淆矩阵
    tp = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 1 and p == 1)
    fp = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 0 and p == 1)
    fn = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 1 and p == 0)
    tn = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 0 and p == 0)

    acc = (tp + tn) / len(gt_labels)
    fall_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    fall_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    fall_f1 = 2 * fall_p * fall_r / (fall_p + fall_r) if (fall_p + fall_r) > 0 else 0
    adl_p = tn / (tn + fn) if (tn + fn) > 0 else 0
    adl_r = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"{thresh:<8.2f} {acc:<8.4f} {fall_p:<8.4f} {fall_r:<8.4f} {fall_f1:<8.4f} "
          f"{adl_p:<8.4f} {adl_r:<8.4f} {tp:<4} {fp:<4} {fn:<4}")

    if fall_f1 > best_f1:
        best_f1 = fall_f1
        best_thresh = thresh
        best_metrics = (acc, fall_p, fall_r, fall_f1, adl_p, adl_r, tp, fp, fn)

print(f"\n=== 最优阈值（按 Fall F1）===")
print(f"阈值: {best_thresh:.2f}")
print(f"  Acc: {best_metrics[0]:.4f}")
print(f"  Fall: P={best_metrics[1]:.4f}, R={best_metrics[2]:.4f}, F1={best_metrics[3]:.4f}")
print(f"  ADL:  P={best_metrics[4]:.4f}, R={best_metrics[5]:.4f}")
print(f"  TP={best_metrics[6]}, FP={best_metrics[7]}, FN={best_metrics[8]}")
