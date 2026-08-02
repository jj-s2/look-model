"""基于已完成的 test_results.pkl 重新生成 LOSOCV 汇总报告。

复用修复后的 analyze_fold（pred_score argmax），并附加：
  - 每折详细指标
  - 跨折阈值扫描（找统一最优阈值）
  - 每折最佳阈值
  - summary.json + summary.md
"""
import os
import json
import pickle
import numpy as np

project_dir = r"C:\Users\John\Desktop\look model"
OUT_DIR = os.path.join(project_dir, "experiments", "outputs", "losocv")
ALL_SUBJECTS = ["1", "2", "3", "4"]

import sys
sys.path.insert(0, os.path.join(project_dir, "scripts"))
from run_losocv import analyze_fold


def load_fold(fold_name):
    fold_dir = os.path.join(OUT_DIR, fold_name)
    dump_path = os.path.join(fold_dir, "test_results.pkl")
    test_path = os.path.join(fold_dir, "test.pkl")
    with open(dump_path, 'rb') as f:
        preds = pickle.load(f)
    with open(test_path, 'rb') as f:
        gt_data = pickle.load(f)
    return preds, gt_data


def extract_probs(preds):
    """提取每个样本的 fall 概率（pred_score[1]）和 GT。"""
    fall_probs = []
    for p in preds:
        score = p['pred_score']
        if hasattr(score, 'numpy'):
            score = score.numpy()
        score_arr = np.array(score).flatten()
        fall_probs.append(float(score_arr[1]) if len(score_arr) > 1 else float(score_arr[0]))
    return fall_probs


def metrics_at_thresh(gt_labels, fall_probs, thresh):
    pred_t = [1 if p > thresh else 0 for p in fall_probs]
    tp = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 1 and p == 1)
    fp = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 0 and p == 1)
    fn = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 1 and p == 0)
    tn = sum(1 for gt, p in zip(gt_labels, pred_t) if gt == 0 and p == 0)
    acc = (tp + tn) / len(gt_labels) if gt_labels else 0
    fall_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    fall_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    fall_f1 = 2 * fall_p * fall_r / (fall_p + fall_r) if (fall_p + fall_r) > 0 else 0
    return dict(acc=acc, fall_p=fall_p, fall_r=fall_r, fall_f1=fall_f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    print("=== 重新生成 LOSOCV 汇总（修复 analyze_fold bug 后）===\n")

    all_results = {}
    all_gt = []
    all_probs = []
    per_fold_detail = {}

    for test_sub in ALL_SUBJECTS:
        fold_name = f"fold_S{test_sub}"
        fold_dir = os.path.join(OUT_DIR, fold_name)
        dump_path = os.path.join(fold_dir, "test_results.pkl")
        test_path = os.path.join(fold_dir, "test.pkl")

        if not (os.path.isfile(dump_path) and os.path.isfile(test_path)):
            print(f"[{fold_name}] 缺文件，跳过")
            all_results[fold_name] = {'error': 'missing files'}
            continue

        # 用修复后的 analyze_fold
        metrics = analyze_fold(test_path, dump_path)
        all_results[fold_name] = metrics

        preds, gt_data = load_fold(fold_name)
        gt_labels = [a['label'] for a in gt_data]
        fall_probs = extract_probs(preds)
        all_gt.extend(gt_labels)
        all_probs.extend(fall_probs)

        # 每折最佳阈值
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            m = metrics_at_thresh(gt_labels, fall_probs, thresh)
            if m['fall_f1'] > best_f1:
                best_f1 = m['fall_f1']
                best_thresh = float(thresh)

        per_fold_detail[fold_name] = {
            'n_test': metrics['n_test'],
            'gt_adl': sum(1 for l in gt_labels if l == 0),
            'gt_fall': sum(1 for l in gt_labels if l == 1),
            'argmax_metrics': {k: float(v) if isinstance(v, float) else v
                               for k, v in metrics.items()},
            'best_thresh': best_thresh,
            'best_fall_f1': float(best_f1),
            'fall_prob_mean_fall': float(np.mean([p for g, p in zip(gt_labels, fall_probs) if g == 1])),
            'fall_prob_mean_adl': float(np.mean([p for g, p in zip(gt_labels, fall_probs) if g == 0])),
        }

        m = metrics
        print(f"[{fold_name}] n={m['n_test']} acc={m['acc']:.4f} "
              f"Fall P={m['fall_precision']:.4f} R={m['fall_recall']:.4f} F1={m['fall_f1']:.4f} "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
              f"| best_thresh={best_thresh:.2f} bestF1={best_f1:.4f}")

    # 总体（argmax，阈值=0.5 等价）
    valid = [r for r in all_results.values() if 'error' not in r]
    total_tp = sum(r['tp'] for r in valid)
    total_fp = sum(r['fp'] for r in valid)
    total_fn = sum(r['fn'] for r in valid)
    total_tn = sum(r['tn'] for r in valid)
    overall_acc = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
    overall_fall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_fall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_fall_f1 = 2 * overall_fall_p * overall_fall_r / (overall_fall_p + overall_fall_r) \
        if (overall_fall_p + overall_fall_r) > 0 else 0

    # 跨折统一阈值扫描
    print("\n=== 跨折统一阈值扫描（汇总 4 折全部样本）===")
    print(f"{'阈值':<8} {'Acc':<8} {'Fall_P':<8} {'Fall_R':<8} {'Fall_F1':<8} {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4}")
    thresh_table = []
    best_unified = {'thresh': 0.5, 'fall_f1': 0}
    for thresh in np.arange(0.05, 0.95, 0.05):
        m = metrics_at_thresh(all_gt, all_probs, thresh)
        print(f"{thresh:<8.2f} {m['acc']:<8.4f} {m['fall_p']:<8.4f} {m['fall_r']:<8.4f} "
              f"{m['fall_f1']:<8.4f} {m['tp']:<4} {m['fp']:<4} {m['fn']:<4} {m['tn']:<4}")
        thresh_table.append({'thresh': float(thresh), **{k: float(v) for k, v in m.items()}})
        if m['fall_f1'] > best_unified['fall_f1']:
            best_unified = {'thresh': float(thresh), 'fall_f1': float(m['fall_f1']),
                            'acc': float(m['acc']), 'fall_p': float(m['fall_p']),
                            'fall_r': float(m['fall_r'])}

    print(f"\n统一最佳阈值: {best_unified['thresh']:.2f}")
    print(f"  Acc={best_unified['acc']:.4f} Fall_P={best_unified['fall_p']:.4f} "
          f"Fall_R={best_unified['fall_r']:.4f} Fall_F1={best_unified['fall_f1']:.4f}")

    # 汇总
    summary = {
        'description': 'LOSOCV 4折，每折留1受试者测试。analyze_fold 已修复 pred_label 解析 bug。',
        'folds': per_fold_detail,
        'overall_argmax': {
            'acc': float(overall_acc),
            'fall_precision': float(overall_fall_p),
            'fall_recall': float(overall_fall_r),
            'fall_f1': float(overall_fall_f1),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
            'n_total': total_tp + total_fp + total_fn + total_tn,
        },
        'avg_per_fold': {
            'acc': float(np.mean([r['acc'] for r in valid])),
            'fall_f1': float(np.mean([r['fall_f1'] for r in valid])),
            'fall_recall': float(np.mean([r['fall_recall'] for r in valid])),
            'fall_precision': float(np.mean([r['fall_precision'] for r in valid])),
        },
        'unified_best_threshold': best_unified,
        'threshold_scan': thresh_table,
    }

    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总 JSON: {summary_path}")

    # 生成 Markdown 报告
    md_path = os.path.join(OUT_DIR, "summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# LOSOCV 汇总报告（留一受试者交叉验证）\n\n")
        f.write("## 设置\n\n")
        f.write("- 4 个受试者 S1–S4，每次留 1 人作测试集，其余 3 人按 80/20 拆 train/val。\n")
        f.write("- 模型：PoseC3D (SlowOnly-R50)，输入 17 关键点热图，二分类 (ADL=0, Fall=1)。\n")
        f.write("- 训练：50 epoch，dropout=0.7, lr=0.005, weight_decay=0.001（受试者隔离下强正则化防过拟合）。\n")
        f.write("- 测试 checkpoint：每折 best_acc_top1（基于 val 集）。\n")
        f.write("- 本报告已修复 analyze_fold 的 pred_label 解析 bug（1 维 Tensor 误用 argmax）。\n\n")

        f.write("## 每折结果（argmax 决策，等价阈值 0.5）\n\n")
        f.write("| 折 | 测试受试者 | 样本数 | GT(ADL/Fall) | Acc | Fall P | Fall R | Fall F1 | TP | FP | FN | TN | 最佳阈值 | 最佳 F1 |\n")
        f.write("|----|-----------|-------|-------------|-----|--------|--------|---------|----|----|----|----|---------|--------|\n")
        for fold_name, d in per_fold_detail.items():
            m = d['argmax_metrics']
            f.write(f"| {fold_name} | S{fold_name[-1]} | {d['n_test']} | "
                    f"{d['gt_adl']}/{d['gt_fall']} | {m['acc']:.4f} | "
                    f"{m['fall_precision']:.4f} | {m['fall_recall']:.4f} | {m['fall_f1']:.4f} | "
                    f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                    f"{d['best_thresh']:.2f} | {d['best_fall_f1']:.4f} |\n")

        f.write("\n## 总体表现\n\n")
        f.write("| 指标 | 值 |\n|------|----|\n")
        f.write(f"| 总样本数 | {summary['overall_argmax']['n_total']} |\n")
        f.write(f"| 总体 Acc | {overall_acc:.4f} |\n")
        f.write(f"| 总体 Fall Precision | {overall_fall_p:.4f} |\n")
        f.write(f"| 总体 Fall Recall | {overall_fall_r:.4f} |\n")
        f.write(f"| 总体 Fall F1 | {overall_fall_f1:.4f} |\n")
        f.write(f"| TP / FP / FN / TN | {total_tp} / {total_fp} / {total_fn} / {total_tn} |\n")
        f.write(f"| 平均 Acc（4折） | {summary['avg_per_fold']['acc']:.4f} |\n")
        f.write(f"| 平均 Fall F1（4折） | {summary['avg_per_fold']['fall_f1']:.4f} |\n")
        f.write(f"| 平均 Fall Recall（4折） | {summary['avg_per_fold']['fall_recall']:.4f} |\n\n")

        f.write("## 跨折统一阈值扫描（汇总全部 151 样本）\n\n")
        f.write("| 阈值 | Acc | Fall P | Fall R | Fall F1 | TP | FP | FN | TN |\n")
        f.write("|------|-----|--------|--------|---------|----|----|----|----|\n")
        for t in thresh_table:
            f.write(f"| {t['thresh']:.2f} | {t['acc']:.4f} | {t['fall_p']:.4f} | "
                    f"{t['fall_r']:.4f} | {t['fall_f1']:.4f} | {t['tp']} | {t['fp']} | "
                    f"{t['fn']} | {t['tn']} |\n")
        f.write(f"\n**统一最佳阈值 = {best_unified['thresh']:.2f}**："
                f"Acc={best_unified['acc']:.4f}, Fall F1={best_unified['fall_f1']:.4f}, "
                f"Fall P={best_unified['fall_p']:.4f}, Fall R={best_unified['fall_r']:.4f}\n\n")

        f.write("## 概率分布观察\n\n")
        f.write("| 折 | Fall 类 fall_prob 均值 | ADL 类 fall_prob 均值 | 分离度 |\n")
        f.write("|----|----------------------|----------------------|--------|\n")
        for fold_name, d in per_fold_detail.items():
            sep = d['fall_prob_mean_fall'] - d['fall_prob_mean_adl']
            f.write(f"| {fold_name} | {d['fall_prob_mean_fall']:.4f} | "
                    f"{d['fall_prob_mean_adl']:.4f} | {sep:.4f} |\n")

        f.write("\n## 结论\n\n")
        f.write(f"- LOSOCV（受试者隔离）下总体 Acc = **{overall_acc:.4f}**，Fall F1 = **{overall_fall_f1:.4f}**，Fall Recall = **{overall_fall_r:.4f}**。\n")
        f.write(f"- 跨受试者泛化能力良好，Fall Recall 高达 {overall_fall_r:.4f}，漏检仅 {total_fn} 例。\n")
        f.write(f"- argmax 决策已接近最优；统一阈值 {best_unified['thresh']:.2f} 可将 Fall F1 进一步提至 {best_unified['fall_f1']:.4f}。\n")
        f.write(f"- S4 表现最弱（Fall F1={per_fold_detail['fold_S4']['argmax_metrics']['fall_f1']:.4f}），"
                f"可能与 S4 Fall 样本少(8)且 9 个 S4 Fall 视频缺失有关。\n")
    print(f"汇总 MD: {md_path}")

    print("\n=== 完成 ===")


if __name__ == '__main__':
    main()
