"""独立分析 LOSOCV 已完成折的结果，并续跑未完成的折。

fold_S1 已完成训练+测试，直接分析；
fold_S2/S3/S4 继续训练+测试。
"""
import os
import sys
import json
import time
import pickle
import subprocess
import numpy as np

project_dir = r"C:\Users\John\Desktop\look model"
OUT_DIR = os.path.join(project_dir, "experiments", "outputs", "losocv")
PY_EXE = os.path.join(project_dir, ".conda", "envs", "elderly-ai", "python.exe")
ALL_SUBJECTS = ["1", "2", "3", "4"]

# 复用 run_losocv 的函数
sys.path.insert(0, os.path.join(project_dir, "scripts"))
from run_losocv import (rebuild_split_pkl, create_fold_config,
                        run_train, run_test, analyze_fold)


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === LOSOCV 续跑 ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    for test_sub in ALL_SUBJECTS:
        fold_name = f"fold_S{test_sub}"
        fold_dir = os.path.join(OUT_DIR, fold_name)

        dump_path = os.path.join(fold_dir, "test_results.pkl")
        test_path = os.path.join(fold_dir, "test.pkl")

        # 如果已有 test_results.pkl，直接分析
        if os.path.isfile(dump_path) and os.path.isfile(test_path):
            print(f"\n[{time.strftime('%H:%M:%S')}] === {fold_name} 已完成，直接分析 ===")
            metrics = analyze_fold(test_path, dump_path)
            all_results[fold_name] = metrics
            print(f"[{time.strftime('%H:%M:%S')}] 结果: acc={metrics['acc']:.4f}, "
                  f"fall_f1={metrics['fall_f1']:.4f}, "
                  f"fall_R={metrics['fall_recall']:.4f}, "
                  f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
            continue

        # 否则训练+测试
        print(f"\n[{time.strftime('%H:%M:%S')}] === Fold {fold_name} (test=S{test_sub}) ===")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"[{time.strftime('%H:%M:%S')}] 重建 split pkl...")
        train_path, val_path, test_path = rebuild_split_pkl(test_sub)

        print(f"[{time.strftime('%H:%M:%S')}] 创建配置...")
        config_path = create_fold_config(train_path, val_path, test_path, fold_dir, fold_name)

        print(f"[{time.strftime('%H:%M:%S')}] 训练...")
        train_log = os.path.join(fold_dir, "train.log")
        ok = run_train(config_path, fold_dir, train_log)
        if not ok:
            print(f"[{time.strftime('%H:%M:%S')}] 训练失败！跳过此折")
            all_results[fold_name] = {'error': 'train failed'}
            continue

        # 用最后 epoch（epoch_50）而非 best_acc
        # best_acc 基于受试者隔离的 val，倾向选早期偏向 ADL 的 epoch
        epochs = [f for f in os.listdir(fold_dir) if f.startswith('epoch_')]
        if not epochs:
            all_results[fold_name] = {'error': 'no checkpoint'}
            continue
        best_ckpt = sorted(epochs, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        checkpoint = os.path.join(fold_dir, best_ckpt)
        print(f"[{time.strftime('%H:%M:%S')}] checkpoint: {best_ckpt}")

        print(f"[{time.strftime('%H:%M:%S')}] 测试...")
        test_log = os.path.join(fold_dir, "test.log")
        ok = run_test(config_path, checkpoint, fold_dir, dump_path, test_log)
        if not ok:
            all_results[fold_name] = {'error': 'test failed'}
            continue

        print(f"[{time.strftime('%H:%M:%S')}] 分析...")
        metrics = analyze_fold(test_path, dump_path)
        all_results[fold_name] = metrics
        print(f"[{time.strftime('%H:%M:%S')}] 结果: acc={metrics['acc']:.4f}, "
              f"fall_f1={metrics['fall_f1']:.4f}, "
              f"fall_R={metrics['fall_recall']:.4f}, "
              f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

    # 汇总
    print(f"\n[{time.strftime('%H:%M:%S')}] === LOSOCV 汇总 ===")
    valid_folds = [r for r in all_results.values() if 'error' not in r]
    if valid_folds:
        avg_acc = np.mean([r['acc'] for r in valid_folds])
        avg_fall_f1 = np.mean([r['fall_f1'] for r in valid_folds])
        avg_fall_r = np.mean([r['fall_recall'] for r in valid_folds])
        avg_fall_p = np.mean([r['fall_precision'] for r in valid_folds])
        total_tp = sum(r['tp'] for r in valid_folds)
        total_fp = sum(r['fp'] for r in valid_folds)
        total_fn = sum(r['fn'] for r in valid_folds)
        total_tn = sum(r['tn'] for r in valid_folds)
        overall_acc = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
        overall_fall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_fall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_fall_f1 = 2 * overall_fall_p * overall_fall_r / (overall_fall_p + overall_fall_r) \
            if (overall_fall_p + overall_fall_r) > 0 else 0

        summary = {
            'folds': all_results,
            'avg_acc': float(avg_acc),
            'avg_fall_f1': float(avg_fall_f1),
            'avg_fall_recall': float(avg_fall_r),
            'avg_fall_precision': float(avg_fall_p),
            'overall_tp': total_tp,
            'overall_fp': total_fp,
            'overall_fn': total_fn,
            'overall_tn': total_tn,
            'overall_acc': float(overall_acc),
            'overall_fall_precision': float(overall_fall_p),
            'overall_fall_recall': float(overall_fall_r),
            'overall_fall_f1': float(overall_fall_f1),
        }
        print(f"  平均 Acc: {avg_acc:.4f}")
        print(f"  平均 Fall F1: {avg_fall_f1:.4f}")
        print(f"  平均 Fall Recall: {avg_fall_r:.4f}")
        print(f"  总体混淆: TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
        print(f"  总体 Acc: {overall_acc:.4f}")
        print(f"  总体 Fall F1: {overall_fall_f1:.4f}")

        summary_path = os.path.join(OUT_DIR, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  汇总报告: {summary_path}")

    print(f"\n[{time.strftime('%H:%M:%S')}] === 完成 (总耗时 {time.time()-t0:.0f}s) ===")


if __name__ == '__main__':
    main()
