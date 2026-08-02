"""绕过 __init__.py 直接加载 mmengine 子模块"""
import importlib.util
import sys
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

# 直接用 importlib 加载 version.py，绕过 mmengine/__init__.py
log("loading mmengine/version.py directly...")
spec = importlib.util.spec_from_file_location(
    "mmengine_version_standalone",
    r"C:\Users\John\Desktop\look model\.conda\envs\elderly-ai\lib\site-packages\mmengine\version.py"
)
version_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_mod)
log(f"version: {version_mod.__version__} ({time.time()-t0:.1f}s)")

# 现在 try mmengine 子模块，一个一个来
log("trying mmengine.logging...")
try:
    import mmengine.logging
    log(f"logging ok ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"logging FAIL: {e}")

log("DONE")
