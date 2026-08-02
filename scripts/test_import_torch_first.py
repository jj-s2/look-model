"""先 import torch 再 import mmcv（复现 PoseC3D wrapper 的顺序）"""
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing torch FIRST...")
import torch
log(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()} ({time.time()-t0:.1f}s)")

log("importing mmcv (after torch)...")
import mmcv
log(f"mmcv: {mmcv.__version__} ({time.time()-t0:.1f}s)")

log("importing mmengine (after torch)...")
import mmengine
log(f"mmengine: {mmengine.__version__} ({time.time()-t0:.1f}s)")

log("importing mmaction (after torch)...")
import mmaction
log(f"mmaction: {mmaction.__version__} ({time.time()-t0:.1f}s)")

log("DONE")
