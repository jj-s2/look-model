"""测试 mmengine 各子模块"""
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing numpy...")
import numpy as np
log(f"numpy ok ({time.time()-t0:.1f}s)")

log("importing cv2...")
import cv2
log(f"cv2 ok ({time.time()-t0:.1f}s)")

log("importing mmengine.version...")
from mmengine.version import __version__ as mmengine_ver
log(f"version: {mmengine_ver} ({time.time()-t0:.1f}s)")

log("importing mmengine.logging...")
import mmengine.logging
log(f"logging ok ({time.time()-t0:.1f}s)")

log("importing mmengine.config...")
import mmengine.config
log(f"config ok ({time.time()-t0:.1f}s)")

log("importing mmengine.fileio...")
import mmengine.fileio
log(f"fileio ok ({time.time()-t0:.1f}s)")

log("importing mmengine.registry...")
import mmengine.registry
log(f"registry ok ({time.time()-t0:.1f}s)")

log("importing mmengine.utils...")
import mmengine.utils
log(f"utils ok ({time.time()-t0:.1f}s)")

log("importing mmengine (full)...")
import mmengine
log(f"mmengine: {mmengine.__version__} ({time.time()-t0:.1f}s)")

log("DONE")
