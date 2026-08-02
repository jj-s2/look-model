"""在测试集上评估最佳 PoseC3D 模型。

使用 best_acc_top1_epoch_45.pth 评估 test 集（28 样本），
输出准确率、混淆矩阵、每类精确率/召回率/F1。
"""
import os
import sys
import time
import subprocess

project_dir = r"C:\Users\John\Desktop\look model"

# usercustomize.py 已自动处理 YAPF 缓存
# 通过 subprocess 调用 mmaction2/tools/test.py
py_exe = os.path.join(project_dir, ".conda", "envs", "elderly-ai", "python.exe")
test_script = os.path.join(project_dir, "third_party", "mmaction2", "tools", "test.py")

config = os.path.join(project_dir, "configs", "skeleton",
                      "posec3d_slowonly_r50_gmdcsa24_fall.py")
# checkpoint 已迁移至 F 盘
sys.path.insert(0, project_dir)
from paths import POSEC3D_GMDCSA24_DIR
checkpoint = os.path.join(POSEC3D_GMDCSA24_DIR, "best_acc_top1_epoch_45.pth")
out_pkl = os.path.join(project_dir, "experiments", "outputs",
                       "posec3d_gmdcsa24", "test_results.pkl")

cmd = [
    py_exe, "-u", test_script,
    config, checkpoint,
    "--dump", out_pkl,
]

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] === PoseC3D 测试集评估 ===")
print(f"[{time.strftime('%H:%M:%S')}] checkpoint: {checkpoint}")
print(f"[{time.strftime('%H:%M:%S')}] command: {' '.join(cmd)}")
print()

env = {**os.environ, "PYTHONUNBUFFERED": "1"}
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=project_dir,
    env=env,
    text=True,
    bufsize=1,
)

for line in proc.stdout:
    print(f"[{time.time()-t0:.1f}s] {line.rstrip()}", flush=True)

proc.wait()
print(f"\n[{time.time()-t0:.1f}s] 测试退出码: {proc.returncode}")

if proc.returncode == 0:
    print(f"[{time.time()-t0:.1f}s] 测试结果保存至: {out_pkl}")
