"""STGCN demo wrapper - 修复 yapf 缓存目录被 sandbox 限制的问题"""
import sys
import os
import time
from pathlib import Path

# 设置 unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

log_path = Path(r"C:\Users\John\Desktop\look model\experiments\logs\mmaction_stgcn_demo.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, data):
        for w in self.writers:
            try:
                w.write(data)
                w.flush()
            except:
                pass
    def flush(self):
        for w in self.writers:
            try:
                w.flush()
            except:
                pass

log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
tee = TeeWriter(sys.stdout, log_file)
sys.stdout = tee

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] === mmaction2 demo_skeleton (STGCN) start ===")

# 关键修复：monkey-patch platformdirs.user_cache_dir，重定向 YAPF 缓存到项目内
project_dir = r"C:\Users\John\Desktop\look model"
yapf_cache_dir = os.path.join(project_dir, ".cache", "yapf")
os.makedirs(yapf_cache_dir, exist_ok=True)

import platformdirs
_original_user_cache_dir = platformdirs.user_cache_dir
def _patched_user_cache_dir(appname=None, appauthor=None, version=None, opinion=True, ensure_exists=False):
    """重定向 user_cache_dir 到项目内目录，避免 sandbox 限制"""
    path = yapf_cache_dir
    params = []
    if appauthor is not False:
        author = appauthor or appname or ""
        if author:
            params.append(author)
    if appname:
        params.append(appname)
    if opinion:
        params.append("Cache")
    if version:
        params.append(version)
    if params:
        path = os.path.join(path, *params)
    os.makedirs(path, exist_ok=True)
    return path

platformdirs.user_cache_dir = _patched_user_cache_dir
# 同时 patch platformdirs.windows 模块中的引用
import platformdirs.windows
import platformdirs.api
# patch 实例方法
_original_Windows_user_cache_dir = platformdirs.windows.Windows.user_cache_dir
@property
def _patched_Windows_user_cache_dir(self):
    path = yapf_cache_dir
    return self._append_parts(path, opinion_value="Cache")
platformdirs.windows.Windows.user_cache_dir = _patched_Windows_user_cache_dir

print(f"[{time.time()-t0:.1f}s] patched platformdirs.user_cache_dir -> {yapf_cache_dir}")

mmaction_dir = r"C:\Users\John\Desktop\look model\third_party\mmaction2"
os.chdir(mmaction_dir)
print(f"[{time.time()-t0:.1f}s] cwd: {os.getcwd()}")

# 运行 demo
import subprocess

py_exe = sys.executable
demo_script = r"demo\demo_skeleton.py"
input_video = r"demo\demo_skeleton.mp4"
output_video = r"demo\demo_skeleton_out_stgcn.mp4"

config = r"configs\skeleton\stgcn\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint = r"C:\Users\John\Desktop\look model\models\pretrained\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.pth"
det_config = r"demo\demo_configs\faster-rcnn_r50_fpn_2x_coco_infer.py"
det_checkpoint = r"C:\Users\John\Desktop\look model\models\pretrained\faster_rcnn_r50_fpn_2x_coco.pth"
pose_config = r"demo\demo_configs\td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py"
pose_checkpoint = r"C:\Users\John\Desktop\look model\models\pretrained\hrnet_w32_coco_256x192-c78dce93_20200708.pth"
label_map = r"tools\data\skeleton\label_map_ntu60.txt"

cmd = [
    py_exe, "-u", demo_script,
    input_video, output_video,
    "--config", config,
    "--checkpoint", checkpoint,
    "--det-config", det_config,
    "--det-checkpoint", det_checkpoint,
    "--det-score-thr", "0.9",
    "--det-cat-id", "0",
    "--pose-config", pose_config,
    "--pose-checkpoint", pose_checkpoint,
    "--label-map", label_map,
]

print(f"[{time.time()-t0:.1f}s] command: {' '.join(cmd)}")

# 设置环境变量传递 cache 目录给子进程
# 通过 LOCALAPPDATA 重定向，使子进程的 platformdirs 也指向项目内目录
appdata_dir = os.path.join(project_dir, ".cache", "appdata")
os.makedirs(appdata_dir, exist_ok=True)
env = {**os.environ, "PYTHONUNBUFFERED": "1", "LOCALAPPDATA": appdata_dir}
print(f"[{time.time()-t0:.1f}s] LOCALAPPDATA -> {appdata_dir}")

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=mmaction_dir,
    env=env,
    text=True,
)

for line in proc.stdout:
    print(f"[{time.time()-t0:.1f}s] {line.rstrip()}")

proc.wait()
print(f"[{time.time()-t0:.1f}s] demo exited with code: {proc.returncode}")

# 检查输出
out_path = Path(mmaction_dir) / output_video
if out_path.exists():
    print(f"[{time.time()-t0:.1f}s] output: {out_path.name} ({out_path.stat().st_size} bytes)")
else:
    print(f"[{time.time()-t0:.1f}s] output file not found: {out_path}")

print(f"[{time.time()-t0:.1f}s] === DONE ===")
log_file.close()
