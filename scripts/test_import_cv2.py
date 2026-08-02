"""测试 import cv2 和 mmengine"""
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing numpy...")
import numpy as np
log(f"numpy: {np.__version__} ({time.time()-t0:.1f}s)")

log("importing cv2...")
import cv2
log(f"cv2: {cv2.__version__} ({time.time()-t0:.1f}s)")

log("importing mmengine...")
import mmengine
log(f"mmengine: {mmengine.__version__} ({time.time()-t0:.1f}s)")

log("importing mmengine.fileio...")
import mmengine.fileio
log(f"mmengine.fileio ok ({time.time()-t0:.1f}s)")

log("DONE")
