"""诊断 LOSOCV test_results.pkl 的实际结构，定位 analyze_fold 解析 bug。

mmaction2 test.log 报告 acc=0.9375，但 analyze_fold 报告 acc=0.4792、Fall F1=0。
说明 pred_label/pred_score 提取逻辑与 pkl 实际结构不匹配。
"""
import os
import pickle
import numpy as np

project_dir = r"C:\Users\John\Desktop\look model"
OUT_DIR = os.path.join(project_dir, "experiments", "outputs", "losocv")

for fold in ["fold_S1", "fold_S2", "fold_S3", "fold_S4"]:
    fold_dir = os.path.join(OUT_DIR, fold)
    dump_path = os.path.join(fold_dir, "test_results.pkl")
    test_path = os.path.join(fold_dir, "test.pkl")

    if not os.path.isfile(dump_path):
        print(f"\n[{fold}] 缺 test_results.pkl，跳过")
        continue

    with open(dump_path, 'rb') as f:
        preds = pickle.load(f)
    with open(test_path, 'rb') as f:
        gt_data = pickle.load(f)

    print(f"\n=== {fold} ===")
    print(f"preds 类型: {type(preds)}, 长度: {len(preds)}")
    print(f"gt_data 类型: {type(gt_data)}, 长度: {len(gt_data)}")

    # 看第一个 pred 元素的结构
    p0 = preds[0]
    print(f"\npreds[0] 类型: {type(p0)}")
    if isinstance(p0, dict):
        print(f"preds[0] keys: {list(p0.keys())}")
        for k, v in p0.items():
            print(f"  {k}: type={type(v).__name__}", end="")
            if hasattr(v, 'shape'):
                print(f" shape={v.shape} dtype={v.dtype}", end="")
            elif isinstance(v, (list, tuple)):
                print(f" len={len(v)}", end="")
            print(f"  value={v}" if not hasattr(v, 'shape') else "")
    else:
        print(f"preds[0] value: {p0}")

    # 看第一个 gt 元素
    g0 = gt_data[0]
    print(f"\ngt_data[0] 类型: {type(g0)}")
    if isinstance(g0, dict):
        print(f"gt_data[0] keys: {list(g0.keys())}")
        print(f"  label: {g0.get('label')}")
        print(f"  frame_dir: {g0.get('frame_dir')}")

    # 尝试正确解析
    print(f"\n--- 尝试解析 ---")
    gt_labels = [a['label'] for a in gt_data]
    n_fall_gt = sum(1 for l in gt_labels if l == 1)
    n_adl_gt = sum(1 for l in gt_labels if l == 0)
    print(f"GT: {len(gt_labels)} 样本, ADL={n_adl_gt}, Fall={n_fall_gt}")

    # 尝试不同方式提取 pred_label
    pred_labels_v1 = []  # 原始方式
    pred_labels_v2 = []  # argmax 方式
    fall_probs = []
    for p in preds:
        if isinstance(p, dict):
            pl = p.get('pred_label')
            score = p.get('pred_score')
            # pred_label 处理
            if hasattr(pl, 'item') and pl.ndim == 0:
                pred_labels_v1.append(int(pl.item()))
            elif hasattr(pl, 'numpy'):
                arr = pl.numpy() if hasattr(pl, 'numpy') else np.array(pl)
                pred_labels_v1.append(int(arr) if arr.ndim == 0 else int(np.argmax(arr)))
            elif isinstance(pl, (list, tuple, np.ndarray)):
                arr = np.array(pl)
                pred_labels_v1.append(int(arr) if arr.ndim == 0 else int(np.argmax(arr)))
            else:
                pred_labels_v1.append(int(pl))

            # pred_score 处理
            if hasattr(score, 'numpy'):
                score = score.numpy()
            score_arr = np.array(score).flatten()
            pred_labels_v2.append(int(np.argmax(score_arr)))
            fall_probs.append(float(score_arr[1]) if len(score_arr) > 1 else float(score_arr[0]))
        else:
            pred_labels_v1.append(int(p))
            pred_labels_v2.append(int(p))
            fall_probs.append(0.5)

    # v1 混淆
    tp1 = sum(1 for gt, p in zip(gt_labels, pred_labels_v1) if gt == 1 and p == 1)
    fp1 = sum(1 for gt, p in zip(gt_labels, pred_labels_v1) if gt == 0 and p == 1)
    fn1 = sum(1 for gt, p in zip(gt_labels, pred_labels_v1) if gt == 1 and p == 0)
    tn1 = sum(1 for gt, p in zip(gt_labels, pred_labels_v1) if gt == 0 and p == 0)
    acc1 = (tp1 + tn1) / len(gt_labels)

    # v2 混淆 (argmax of score)
    tp2 = sum(1 for gt, p in zip(gt_labels, pred_labels_v2) if gt == 1 and p == 1)
    fp2 = sum(1 for gt, p in zip(gt_labels, pred_labels_v2) if gt == 0 and p == 1)
    fn2 = sum(1 for gt, p in zip(gt_labels, pred_labels_v2) if gt == 1 and p == 0)
    tn2 = sum(1 for gt, p in zip(gt_labels, pred_labels_v2) if gt == 0 and p == 0)
    acc2 = (tp2 + tn2) / len(gt_labels)

    print(f"\nv1 (pred_label 直接取): acc={acc1:.4f} TP={tp1} FP={fp1} FN={fn1} TN={tn1}")
    print(f"v2 (argmax pred_score): acc={acc2:.4f} TP={tp2} FP={fp2} FN={fn2} TN={tn2}")

    # 概率分布
    fall_probs_arr = np.array(fall_probs)
    fall_gt_probs = [p for gt, p in zip(gt_labels, fall_probs) if gt == 1]
    adl_gt_probs = [p for gt, p in zip(gt_labels, fall_probs) if gt == 0]
    print(f"\nFall 类样本的 pred_score[1] 分布: min={min(fall_gt_probs):.4f} "
          f"max={max(fall_gt_probs):.4f} mean={np.mean(fall_gt_probs):.4f}")
    print(f"ADL 类样本的 pred_score[1] 分布: min={min(adl_gt_probs):.4f} "
          f"max={max(adl_gt_probs):.4f} mean={np.mean(adl_gt_probs):.4f}")

    # 阈值扫描
    print(f"\n阈值扫描 (基于 pred_score[1]):")
    print(f"{'阈值':<8} {'Acc':<8} {'Fall_P':<8} {'Fall_R':<8} {'Fall_F1':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0.05, 0.95, 0.05):
        pred_t = [1 if p > thresh else 0 for p in fall_probs]
        tp = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 1 and p == 1)
        fp = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 0 and p == 1)
        fn = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 1 and p == 0)
        tn = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 0 and p == 0)
        acc = (tp + tn) / len(gt_labels)
        fall_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        fall_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        fall_f1 = 2 * fall_p * fall_r / (fall_p + fall_r) if (fall_p + fall_r) > 0 else 0
        print(f"{thresh:<8.2f} {acc:<8.4f} {fall_p:<8.4f} {fall_r:<8.4f} {fall_f1:<8.4f} {tp:<4} {fp:<4} {fn:<4}")
        if fall_f1 > best_f1:
            best_f1 = fall_f1
            best_thresh = thresh
    print(f"最佳阈值: {best_thresh:.2f}, Fall F1: {best_f1:.4f}")
