"""包装 mmpose topdown demo，将输出同时写入文件和 stdout"""
import sys
import os
import time
from pathlib import Path

# 设置 unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

log_path = Path(r"C:\Users\John\Desktop\look model\experiments\logs\mmpose_demo.log")
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
print(f"[{time.time()-t0:.1f}s] === mmpose topdown demo start ===")

# 切换工作目录
os.chdir(r"C:\Users\John\Desktop\look model\third_party\mmpose")
print(f"[{time.time()-t0:.1f}s] cwd: {os.getcwd()}")

# 导入
print(f"[{time.time()-t0:.1f}s] importing torch...")
import torch
print(f"[{time.time()-t0:.1f}s] torch ok: {torch.__version__}")

print(f"[{time.time()-t0:.1f}s] importing mmcv...")
import mmcv
print(f"[{time.time()-t0:.1f}s] mmcv ok: {mmcv.__version__}")

print(f"[{time.time()-t0:.1f}s] importing mmdet...")
import mmdet
print(f"[{time.time()-t0:.1f}s] mmdet ok: {mmdet.__version__}")

print(f"[{time.time()-t0:.1f}s] importing mmpose...")
import mmpose
print(f"[{time.time()-t0:.1f}s] mmpose ok: {mmpose.__version__}")

print(f"[{time.time()-t0:.1f}s] cuda available: {torch.cuda.is_available()}")

# 运行 demo
print(f"[{time.time()-t0:.1f}s] running demo...")

# 使用 subprocess 运行 demo 脚本
import subprocess

py_exe = sys.executable
demo_script = r"demo\topdown_demo_with_mmdet.py"
det_config = r"demo\mmdetection_cfg\rtmdet_tiny_8xb32-300e_coco.py"
det_checkpoint = r"F:\look model\models\pretrained\rtmdet_tiny_8xb32-300e_coco.pth"
pose_config = r"configs\body_2d_keypoint\rtmpose\coco\rtmpose-m_8xb256-420e_coco-256x192.py"
pose_checkpoint = r"F:\look model\models\pretrained\rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
input_video = r"demo\resources\demo.mp4"
output_root = r"vis_results"

cmd = [
    py_exe, "-u", demo_script,
    det_config, det_checkpoint,
    pose_config, pose_checkpoint,
    "--input", input_video,
    "--output-root", output_root,
    "--save-predictions",
    "--device", "cuda:0",
]

print(f"[{time.time()-t0:.1f}s] command: {' '.join(cmd)}")

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=r"C:\Users\John\Desktop\look model\third_party\mmpose",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    text=True,
)

for line in proc.stdout:
    print(f"[{time.time()-t0:.1f}s] {line.rstrip()}")

proc.wait()
print(f"[{time.time()-t0:.1f}s] demo exited with code: {proc.returncode}")

# 检查输出
vis_dir = Path(r"C:\Users\John\Desktop\look model\third_party\mmpose\vis_results")
if vis_dir.exists():
    for f in vis_dir.iterdir():
        print(f"[{time.time()-t0:.1f}s] output: {f.name} ({f.stat().st_size} bytes)")
else:
    print(f"[{time.time()-t0:.1f}s] vis_results dir not found")

print(f"[{time.time()-t0:.1f}s] === DONE ===")
log_file.close()
