"""PoseC3D 微调训练包装脚本：GMDCSA24 fall vs ADL 二分类。

功能：
1. Monkey-patch platformdirs 解决 YAPF 缓存 sandbox 限制
2. 设置 sys.path 引入 mmaction2
3. 调用 mmaction2/tools/train.py 进行训练

运行：
  python scripts/train_posec3d_gmdcsa24.py
"""
import os
import sys
import time
import subprocess

project_dir = r"C:\Users\John\Desktop\look model"

# ---- 0. monkey-patch platformdirs（必须在 import mmcv 之前）----
yapf_cache_dir = os.path.join(project_dir, ".cache", "yapf")
os.makedirs(yapf_cache_dir, exist_ok=True)
appdata_dir = os.path.join(project_dir, ".cache", "appdata")
os.makedirs(appdata_dir, exist_ok=True)
os.environ["LOCALAPPDATA"] = appdata_dir

# 训练通过 subprocess 调用，子进程需要 LOCALAPPDATA 环境变量
# 这里设置当前进程的环境变量，subprocess 会继承
env = {**os.environ, "PYTHONUNBUFFERED": "1", "LOCALAPPDATA": appdata_dir}

# ---- 1. 配置 ----
config = os.path.join(project_dir, "configs", "skeleton",
                      "posec3d_slowonly_r50_gmdcsa24_fall.py")
work_dir = os.path.join(project_dir, "experiments", "outputs", "posec3d_gmdcsa24")
os.makedirs(work_dir, exist_ok=True)

py_exe = sys.executable
train_script = os.path.join(project_dir, "third_party", "mmaction2",
                            "tools", "train.py")

cmd = [
    py_exe, "-u", train_script,
    config,
    "--work-dir", work_dir,
]

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] === PoseC3D GMDCSA24 训练启动 ===")
print(f"[{time.strftime('%H:%M:%S')}] config: {config}")
print(f"[{time.strftime('%H:%M:%S')}] work_dir: {work_dir}")
print(f"[{time.strftime('%H:%M:%S')}] command: {' '.join(cmd)}")
print(f"[{time.strftime('%H:%M:%S')}] cwd: {project_dir}")
print()

# ---- 2. 运行训练 ----
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=project_dir,  # cwd 为项目根，使 ann_file 相对路径生效
    env=env,
    text=True,
    bufsize=1,
)

for line in proc.stdout:
    print(f"[{time.time()-t0:.1f}s] {line.rstrip()}", flush=True)

proc.wait()
print(f"\n[{time.time()-t0:.1f}s] 训练退出码: {proc.returncode}")

if proc.returncode == 0:
    print(f"[{time.time()-t0:.1f}s] 训练完成！模型保存在: {work_dir}")
else:
    print(f"[{time.time()-t0:.1f}s] 训练失败，请检查日志")
