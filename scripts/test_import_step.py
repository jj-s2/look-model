"""分步测试 import 定位卡住位置"""
import sys
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("start")
t0 = time.time()

log("importing mmcv...")
import mmcv
log(f"mmcv: {mmcv.__version__} ({time.time()-t0:.1f}s)")

log("importing mmengine...")
import mmengine
log(f"mmengine: {mmengine.__version__} ({time.time()-t0:.1f}s)")

log("importing torch...")
import torch
log(f"torch: {torch.__version__} ({time.time()-t0:.1f}s)")

log("from mmengine.utils import digit_version...")
from mmengine.utils import digit_version
log(f"digit_version ok ({time.time()-t0:.1f}s)")

log("importing mmaction.version...")
from mmaction.version import __version__ as mmaction_ver
log(f"mmaction.version: {mmaction_ver} ({time.time()-t0:.1f}s)")

log("importing mmaction (full)...")
import mmaction
log(f"mmaction: {mmaction.__version__} ({time.time()-t0:.1f}s)")

log("DONE")
