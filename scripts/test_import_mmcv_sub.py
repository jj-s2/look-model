"""测试 mmcv 各子模块 - 直接 import 避免触发 mmcv/__init__.py"""
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing numpy...")
import numpy as np
log(f"numpy: {np.__version__} ({time.time()-t0:.1f}s)")

log("importing mmcv.arraymisc.quantization (bypass __init__)...")
import importlib
mod = importlib.import_module('mmcv.arraymisc.quantization')
log(f"quantization ok ({time.time()-t0:.1f}s)")

log("importing mmcv.image (bypass __init__)...")
import mmcv.image
log(f"image ok ({time.time()-t0:.1f}s)")

log("importing mmcv.transforms (bypass __init__)...")
import mmcv.transforms
log(f"transforms ok ({time.time()-t0:.1f}s)")

log("importing mmcv.video (bypass __init__)...")
import mmcv.video
log(f"video ok ({time.time()-t0:.1f}s)")

log("importing mmcv.visualization (bypass __init__)...")
import mmcv.visualization
log(f"visualization ok ({time.time()-t0:.1f}s)")

log("importing mmcv (full)...")
import mmcv
log(f"mmcv: {mmcv.__version__} ({time.time()-t0:.1f}s)")

log("DONE")
