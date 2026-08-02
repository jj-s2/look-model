"""监控姿态提取进度，完成后自动启动 PoseC3D 训练。

逻辑：
1. 轮询检查 extract_gmdcsa24_pose.py 进程是否存活
2. 进程结束后，检查最终 pkl 文件是否生成
3. 验证 train/val/test pkl 内容
4. 自动启动训练脚本
"""
import os
import sys
import time
import subprocess
import pickle

project_dir = r"C:\Users\John\Desktop\look model"
POSE_DIR = os.path.join(project_dir, "datasets", "processed", "gmdcsa24_pose")
ANNO_DIR = os.path.join(project_dir, "datasets", "annotations")
LOG_FILE = os.path.join(project_dir, "experiments", "logs", "extract_gmdcsa24_pose.log")

def get_pkl_count():
    """获取已生成的单视频 pkl 数量。"""
    if not os.path.isdir(POSE_DIR):
        return 0
    return len([f for f in os.listdir(POSE_DIR) if f.endswith('.pkl')])

def is_extract_running():
    """检查 extract_gmdcsa24_pose.py 是否还在运行。"""
    try:
        # 通过 tasklist 查找 python 进程
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True, timeout=10)
        # 简单判断：有 python.exe 进程在跑即认为可能还在提取
        # 更精确的方式是检查命令行参数
        return 'python.exe' in result.stdout
    except Exception:
        return False

def is_extract_running_precise():
    """精确检查 extract 脚本是否在运行（通过 wmic 查命令行）。"""
    try:
        result = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'commandline'],
            capture_output=True, text=True, timeout=10)
        return 'extract_gmdcsa24_pose' in result.stdout
    except Exception:
        # wmic 失败时回退到简单检查
        return is_extract_running()

def verify_final_pkls():
    """验证最终生成的 train/val/test pkl 文件。"""
    print(f"\n[{time.strftime('%H:%M:%S')}] === 验证最终 pkl 文件 ===")
    all_ok = True
    for split in ['train', 'val', 'test']:
        pkl_path = os.path.join(ANNO_DIR, f"gmdcsa24_pose_{split}.pkl")
        if not os.path.isfile(pkl_path):
            print(f"  [FAIL] {split}: 文件不存在")
            all_ok = False
            continue
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            n_adl = sum(1 for a in data if a['label'] == 0)
            n_fall = sum(1 for a in data if a['label'] == 1)
            print(f"  [OK] {split}: {len(data)} samples (adl={n_adl}, fall={n_fall})")
            # 抽查第一个样本格式
            if data:
                a = data[0]
                kp_shape = a['keypoint'].shape
                print(f"       样本0: frame_dir={a['frame_dir']}, "
                      f"keypoint={kp_shape}, total_frames={a['total_frames']}")
        except Exception as e:
            print(f"  [FAIL] {split}: 读取失败 - {e}")
            all_ok = False
    return all_ok

def launch_training():
    """启动 PoseC3D 训练。"""
    print(f"\n[{time.strftime('%H:%M:%S')}] === 启动 PoseC3D 训练 ===")
    train_script = os.path.join(project_dir, "scripts", "train_posec3d_gmdcsa24.py")
    py_exe = os.path.join(project_dir, ".conda", "envs", "elderly-ai", "python.exe")

    cmd = [py_exe, "-u", train_script]
    print(f"[{time.strftime('%H:%M:%S')}] command: {' '.join(cmd)}")

    # 前台运行训练，实时输出日志
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=project_dir,
        text=True,
        bufsize=1,
    )

    train_log = os.path.join(project_dir, "experiments", "logs",
                             "train_posec3d_gmdcsa24.log")
    os.makedirs(os.path.dirname(train_log), exist_ok=True)

    with open(train_log, 'w', encoding='utf-8') as log_f:
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
            log_f.write(line)
            log_f.flush()

    proc.wait()
    print(f"\n[{time.strftime('%H:%M:%S')}] 训练退出码: {proc.returncode}")
    return proc.returncode

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === 监控启动 ===")
    print(f"目标: 等待姿态提取完成后自动启动训练")

    # 轮询监控
    check_interval = 60  # 每 60 秒检查一次
    max_wait = 7200      # 最长等待 2 小时
    waited = 0

    while waited < max_wait:
        count = get_pkl_count()
        running = is_extract_running_precise()

        print(f"[{time.strftime('%H:%M:%S')}] 已生成 {count}/160 pkl, "
              f"extract 进程运行中: {running}")

        if not running:
            # 进程结束，检查是否生成足够 pkl
            if count >= 150:  # 允许少量失败
                print(f"\n[{time.strftime('%H:%M:%S')}] 提取进程已结束，"
                      f"生成 {count} 个 pkl，开始验证")
                break
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 警告: 进程结束但仅生成 {count} pkl")
                # 进程已结束但数量不足，也继续验证
                break

        time.sleep(check_interval)
        waited += check_interval

    # 等待 5 秒确保 pkl 文件写入完成
    time.sleep(5)

    # 验证最终 pkl
    if not verify_final_pkls():
        print(f"\n[{time.strftime('%H:%M:%S')}] pkl 验证失败，不启动训练")
        print("请手动检查 datasets/annotations/ 目录")
        return 1

    # 启动训练
    rc = launch_training()
    print(f"\n[{time.strftime('%H:%M:%S')}] 全部流程完成 (总耗时 {time.time()-t0:.0f}s)")
    return rc

if __name__ == '__main__':
    sys.exit(main())
