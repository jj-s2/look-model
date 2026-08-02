"""包装 mmaction2 demo_skeleton（STGCN），将输出同时写入文件和 stdout"""
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

mmaction_dir = r"C:\Users\John\Desktop\look model\third_party\mmaction2"
os.chdir(mmaction_dir)
print(f"[{time.time()-t0:.1f}s] cwd: {os.getcwd()}")

# 直接通过 subprocess 调用 demo_skeleton.py，避免 wrapper 自身 import 卡住
import subprocess

py_exe = sys.executable
demo_script = r"demo\demo_skeleton.py"
input_video = r"demo\demo_skeleton.mp4"
output_video = r"demo\demo_skeleton_out_stgcn.mp4"

config = r"configs\skeleton\stgcn\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint = r"F:\look model\models\pretrained\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.pth"
det_config = r"demo\demo_configs\faster-rcnn_r50_fpn_2x_coco_infer.py"
det_checkpoint = r"F:\look model\models\pretrained\faster_rcnn_r50_fpn_2x_coco.pth"
pose_config = r"demo\demo_configs\td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py"
pose_checkpoint = r"F:\look model\models\pretrained\hrnet_w32_coco_256x192-c78dce93_20200708.pth"
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

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=mmaction_dir,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
