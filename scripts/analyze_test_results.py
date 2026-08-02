"""分析测试集预测结果，生成混淆矩阵和每类 P/R/F1。"""
import os
import pickle
import numpy as np

project_dir = r"C:\Users\John\Desktop\look model"

# 读取测试结果
results_path = os.path.join(project_dir, "experiments", "outputs",
                            "posec3d_gmdcsa24", "test_results.pkl")
with open(results_path, 'rb') as f:
    preds = pickle.load(f)

# 读取测试集 ground truth
gt_path = os.path.join(project_dir, "datasets", "annotations", "gmdcsa24_pose_test.pkl")
with open(gt_path, 'rb') as f:
    gt_data = pickle.load(f)

print(f"预测结果数: {len(preds)}")
print(f"GT 样本数: {len(gt_data)}")

# 检查 preds 结构
print(f"preds[0] type: {type(preds[0]).__name__}")
if isinstance(preds[0], dict):
    print(f"preds[0] keys: {list(preds[0].keys())}")
    # 取第一个元素查看
    for k, v in preds[0].items():
        print(f"  {k}: type={type(v).__name__}, val={v if not hasattr(v, 'shape') else v.shape}")
elif isinstance(preds[0], np.ndarray):
    print(f"preds[0] shape: {preds[0].shape}")
print()

# preds 可能是 list[dict]，需提取 pred_label 字段
pred_labels = []
for p in preds:
    if isinstance(p, dict):
        # 尝试常见字段名
        if 'pred_label' in p:
            pl = p['pred_label']
            if hasattr(pl, 'item'):
                pred_labels.append(int(pl.item()) if pl.ndim == 0 else int(np.argmax(pl)))
            else:
                pred_labels.append(int(pl))
        elif 'pred' in p:
            pred_labels.append(int(p['pred']))
        else:
            # 尝试从 pred_scores 取 argmax
            scores = p.get('pred_scores') or p.get('scores')
            if scores is not None:
                if hasattr(scores, 'numpy'):
                    scores = scores.numpy()
                pred_labels.append(int(np.argmax(scores)))
            else:
                pred_labels.append(-1)
    elif isinstance(p, np.ndarray):
        if p.ndim == 0:
            pred_labels.append(int(p.item()))
        elif p.ndim == 1:
            pred_labels.append(int(np.argmax(p)))
        else:  # (num_clips, num_classes)
            pred_labels.append(int(np.argmax(p.mean(axis=0))))
    else:
        pred_labels.append(int(p))

gt_labels = [a['label'] for a in gt_data]
frame_dirs = [a['frame_dir'] for a in gt_data]

print(f"\n=== 预测标签分布 ===")
print(f"  ADL (0): {pred_labels.count(0)}")
print(f"  Fall (1): {pred_labels.count(1)}")

print(f"\n=== 真实标签分布 ===")
print(f"  ADL (0): {gt_labels.count(0)}")
print(f"  Fall (1): {gt_labels.count(1)}")

# 混淆矩阵
cm = np.zeros((2, 2), dtype=int)
for gt, pred in zip(gt_labels, pred_labels):
    cm[gt][pred] += 1

print(f"\n=== 混淆矩阵 ===")
print(f"           预测ADL  预测Fall")
print(f"  真实ADL   {cm[0][0]:>5}    {cm[0][1]:>5}")
print(f"  真实Fall  {cm[1][0]:>5}    {cm[1][1]:>5}")

# 每类 P/R/F1
print(f"\n=== 每类指标 ===")
classes = ['ADL', 'Fall']
for i, cls in enumerate(classes):
    tp = cm[i][i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  {cls}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn})")

# 整体准确率
acc = (cm[0][0] + cm[1][1]) / cm.sum()
print(f"\n=== 整体准确率 ===")
print(f"  Accuracy: {acc:.4f} ({cm[0][0]+cm[1][1]}/{cm.sum()})")

# 错误样本
print(f"\n=== 错误样本 ===")
for i, (gt, pred, fd) in enumerate(zip(gt_labels, pred_labels, frame_dirs)):
    if gt != pred:
        gt_name = classes[gt]
        pred_name = classes[pred]
        print(f"  {fd}: GT={gt_name}, Pred={pred_name}")
