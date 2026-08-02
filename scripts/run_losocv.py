"""留一受试者交叉验证（LOSOCV）。

4 个受试者 S1-S4，每次留 1 人作测试集，其余 3 人按 80/20 拆分为 train/val。
共训练 4 个模型，汇总 4 折结果给出总体表现。

用法：
  python scripts/run_losocv.py

输出：
  experiments/outputs/losocv/fold_S{X}/  - 每折的 checkpoint 和测试结果
  experiments/outputs/losocv/summary.json - 汇总报告
"""
import os
import sys
import json
import time
import pickle
import subprocess
import numpy as np

project_dir = r"C:\Users\John\Desktop\look model"
sys.path.insert(0, project_dir)
from paths import LOSOCV_DIR  # checkpoint 已迁移至 F 盘
ANNO_DIR = os.path.join(project_dir, "datasets", "annotations")
OUT_DIR = LOSOCV_DIR
PY_EXE = os.path.join(project_dir, ".conda", "envs", "elderly-ai", "python.exe")

# 4 个受试者
ALL_SUBJECTS = ["1", "2", "3", "4"]


def rebuild_split_pkl(test_subject):
    """根据测试受试者重建 train/val/test pkl。

    test_subject: 留作测试的受试者 ID
    其余 3 个受试者：80% train, 20% val（按受试者内随机抽样）
    """
    import csv
    import random

    METADATA_CSV = os.path.join(project_dir, "datasets", "metadata.csv")
    POSE_DIR = os.path.join(project_dir, "datasets", "processed", "gmdcsa24_pose")
    LABEL_MAP = {"adl": 0, "fall": 1}

    with open(METADATA_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # 建立所有样本的缓存路径与元信息
    samples = []  # list of (frame_dir, subject, label, cache_path)
    for row in rows:
        subject = row['subject_id']
        video_id = row['video_id']
        category = row['category']
        label_str = row['label']
        frame_dir = f"S{subject}_{category}{video_id}"
        label = LABEL_MAP.get(label_str.lower())
        if label is None:
            continue
        cache_path = os.path.join(POSE_DIR, f"{frame_dir}.pkl")
        if not os.path.isfile(cache_path):
            continue  # 跳过缺失（9 个 S4 Fall）
        samples.append((frame_dir, subject, label, cache_path))

    # 划分
    test_samples = [s for s in samples if s[1] == test_subject]
    other_samples = [s for s in samples if s[1] != test_subject]

    # 其余受试者按 80/20 拆分 train/val（固定随机种子保证可复现）
    random.seed(42)
    random.shuffle(other_samples)
    n_total = len(other_samples)
    n_train = int(n_total * 0.8)
    train_samples = other_samples[:n_train]
    val_samples = other_samples[n_train:]

    # 生成 pkl
    def make_pkl(samples_list, out_path):
        annos = []
        for frame_dir, subject, label, cache_path in samples_list:
            try:
                with open(cache_path, 'rb') as f:
                    anno = pickle.load(f)
                anno['label'] = label
                annos.append(anno)
            except Exception as e:
                print(f"  跳过 {frame_dir}: {e}")
        with open(out_path, 'wb') as f:
            pickle.dump(annos, f)
        n_adl = sum(1 for a in annos if a['label'] == 0)
        n_fall = sum(1 for a in annos if a['label'] == 1)
        return len(annos), n_adl, n_fall

    fold_dir = os.path.join(OUT_DIR, f"fold_S{test_subject}")
    os.makedirs(fold_dir, exist_ok=True)

    train_path = os.path.join(fold_dir, "train.pkl")
    val_path = os.path.join(fold_dir, "val.pkl")
    test_path = os.path.join(fold_dir, "test.pkl")

    n_tr, tr_adl, tr_fall = make_pkl(train_samples, train_path)
    n_va, va_adl, va_fall = make_pkl(val_samples, val_path)
    n_te, te_adl, te_fall = make_pkl(test_samples, test_path)

    print(f"  Fold S{test_subject}: train={n_tr}({tr_adl}a/{tr_fall}f) "
          f"val={n_va}({va_adl}a/{va_fall}f) test={n_te}({te_adl}a/{te_fall}f)")

    return train_path, val_path, test_path


def create_fold_config(train_path, val_path, test_path, work_dir, fold_name):
    """为当前 fold 创建配置文件。

    注意：之前用 dropout=0.7/lr=0.005/weight_decay=0.001 导致 Fall 类完全没学到。
    现回退到原始参数（dropout=0.5, lr=0.02, weight_decay=0.0003），
    仅依赖受试者隔离本身来评估泛化能力。
    """
    config_path = os.path.join(work_dir, "config.py")
    # 读取基础配置并替换路径
    base_config = os.path.join(project_dir, "configs", "skeleton",
                               "posec3d_slowonly_r50_gmdcsa24_fall.py")
    with open(base_config, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换 ann_file 路径
    content = content.replace(
        "datasets/annotations/gmdcsa24_pose_train.pkl",
        train_path.replace("\\", "/"))
    content = content.replace(
        "datasets/annotations/gmdcsa24_pose_val.pkl",
        val_path.replace("\\", "/"))
    content = content.replace(
        "datasets/annotations/gmdcsa24_pose_test.pkl",
        test_path.replace("\\", "/"))

    # 保持原始超参数，不做过度正则化
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return config_path


def run_train(config_path, work_dir, log_path):
    """运行训练（Popen 实时写日志，无 timeout）。"""
    train_script = os.path.join(project_dir, "third_party", "mmaction2",
                                "tools", "train.py")
    cmd = [PY_EXE, "-u", train_script, config_path, "--work-dir", work_dir]

    with open(log_path, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=project_dir, text=True, bufsize=1)
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
        proc.wait()
    return proc.returncode == 0


def run_test(config_path, checkpoint, work_dir, dump_path, log_path):
    """运行测试（Popen 实时写日志，无 timeout）。"""
    test_script = os.path.join(project_dir, "third_party", "mmaction2",
                               "tools", "test.py")
    cmd = [PY_EXE, "-u", test_script, config_path, checkpoint, "--dump", dump_path]

    with open(log_path, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=project_dir, text=True, bufsize=1)
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
        proc.wait()
    return proc.returncode == 0


def analyze_fold(test_path, dump_path):
    """分析单折结果，返回指标字典。"""
    with open(dump_path, 'rb') as f:
        preds = pickle.load(f)
    with open(test_path, 'rb') as f:
        gt_data = pickle.load(f)

    # 提取预测标签和概率
    # 注意：pred_label 是 shape=[1] 的 1 维 Tensor（如 tensor([0])），
    # 不能用 pl.ndim==0 判断；item() 对单元素张量无论几维都有效。
    # 为稳健起见，优先用 pred_score 的 argmax 作为 pred_label。
    pred_labels = []
    fall_probs = []
    for p in preds:
        if isinstance(p, dict):
            score = p['pred_score']
            if hasattr(score, 'numpy'):
                score = score.numpy()
            score_arr = np.array(score).flatten()
            # pred_score 形如 [p_adl, p_fall]，argmax 即预测类别
            pred_labels.append(int(np.argmax(score_arr)))
            fall_probs.append(float(score_arr[1]) if len(score_arr) > 1 else float(score_arr[0]))
        else:
            pred_labels.append(int(p))
            fall_probs.append(0.5)

    gt_labels = [a['label'] for a in gt_data]

    # 混淆矩阵
    tp = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 1 and p == 1)
    fp = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 0 and p == 1)
    fn = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 1 and p == 0)
    tn = sum(1 for gt, p in zip(gt_labels, pred_labels) if gt == 0 and p == 0)

    acc = (tp + tn) / len(gt_labels) if gt_labels else 0
    fall_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    fall_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    fall_f1 = 2 * fall_p * fall_r / (fall_p + fall_r) if (fall_p + fall_r) > 0 else 0
    adl_p = tn / (tn + fn) if (tn + fn) > 0 else 0
    adl_r = tn / (tn + fp) if (tn + fp) > 0 else 0
    adl_f1 = 2 * adl_p * adl_r / (adl_p + adl_r) if (adl_p + adl_r) > 0 else 0

    return {
        'n_test': len(gt_labels),
        'acc': acc,
        'fall_precision': fall_p,
        'fall_recall': fall_r,
        'fall_f1': fall_f1,
        'adl_precision': adl_p,
        'adl_recall': adl_r,
        'adl_f1': adl_f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def main():
    import numpy as np  # analyze_fold 用到

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === LOSOCV 启动 ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    for test_sub in ALL_SUBJECTS:
        fold_name = f"fold_S{test_sub}"
        fold_dir = os.path.join(OUT_DIR, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n[{time.strftime('%H:%M:%S')}] === Fold {fold_name} (test=S{test_sub}) ===")

        # 1. 重建 split pkl
        print(f"[{time.strftime('%H:%M:%S')}] 重建 split pkl...")
        train_path, val_path, test_path = rebuild_split_pkl(test_sub)

        # 2. 创建配置
        print(f"[{time.strftime('%H:%M:%S')}] 创建配置...")
        config_path = create_fold_config(train_path, val_path, test_path, fold_dir, fold_name)

        # 3. 训练
        print(f"[{time.strftime('%H:%M:%S')}] 训练...")
        train_log = os.path.join(fold_dir, "train.log")
        ok = run_train(config_path, fold_dir, train_log)
        if not ok:
            print(f"[{time.strftime('%H:%M:%S')}] 训练失败！跳过此折")
            all_results[fold_name] = {'error': 'train failed'}
            continue

        # 4. 找最佳 checkpoint
        best_ckpts = [f for f in os.listdir(fold_dir) if f.startswith('best_acc')]
        if not best_ckpts:
            # 用最后一个 epoch
            epochs = [f for f in os.listdir(fold_dir) if f.startswith('epoch_')]
            if not epochs:
                print(f"[{time.strftime('%H:%M:%S')}] 无 checkpoint！跳过此折")
                all_results[fold_name] = {'error': 'no checkpoint'}
                continue
            best_ckpt = sorted(epochs, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        else:
            best_ckpt = best_ckpts[0]
        checkpoint = os.path.join(fold_dir, best_ckpt)
        print(f"[{time.strftime('%H:%M:%S')}] 使用 checkpoint: {best_ckpt}")

        # 5. 测试
        print(f"[{time.strftime('%H:%M:%S')}] 测试...")
        dump_path = os.path.join(fold_dir, "test_results.pkl")
        test_log = os.path.join(fold_dir, "test.log")
        ok = run_test(config_path, checkpoint, fold_dir, dump_path, test_log)
        if not ok:
            print(f"[{time.strftime('%H:%M:%S')}] 测试失败！跳过此折")
            all_results[fold_name] = {'error': 'test failed'}
            continue

        # 6. 分析
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

    print(f"\n[{time.strftime('%H:%M:%S')}] === LOSOCV 完成 (总耗时 {time.time()-t0:.0f}s) ===")


if __name__ == '__main__':
    main()
