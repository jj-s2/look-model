"""测试 mmcv 各子模块 import"""
import sys
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing mmcv.arraymisc...")
try:
    from mmcv import arraymisc
    log(f"arraymisc ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"arraymisc FAIL: {e}")

log("importing mmcv.image...")
try:
    from mmcv import image
    log(f"image ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"image FAIL: {e}")

log("importing mmcv.transforms...")
try:
    from mmcv import transforms
    log(f"transforms ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"transforms FAIL: {e}")

log("importing mmcv.version...")
try:
    from mmcv import version
    log(f"version ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"version FAIL: {e}")

log("importing mmcv.video...")
try:
    from mmcv import video
    log(f"video ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"video FAIL: {e}")

log("importing mmcv.visualization...")
try:
    from mmcv import visualization
    log(f"visualization ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"visualization FAIL: {e}")

log("DONE")
