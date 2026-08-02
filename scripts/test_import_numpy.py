"""测试 import numpy"""
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing numpy...")
import numpy as np
log(f"numpy: {np.__version__} ({time.time()-t0:.1f}s)")

log("DONE")
