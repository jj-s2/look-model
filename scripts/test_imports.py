"""测试 mmcv 单独导入（不先导入 torch）"""
import time

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] start importing mmcv (no torch first)", flush=True)
import mmcv
print(f"[{time.time()-t0:.1f}s] mmcv ok: {mmcv.__version__}", flush=True)

print(f"[{time.time()-t0:.1f}s] start importing mmdet", flush=True)
import mmdet
print(f"[{time.time()-t0:.1f}s] mmdet ok: {mmdet.__version__}", flush=True)

print(f"[{time.time()-t0:.1f}s] start importing mmpose", flush=True)
import mmpose
print(f"[{time.time()-t0:.1f}s] mmpose ok: {mmpose.__version__}", flush=True)

print(f"[{time.time()-t0:.1f}s] start importing mmaction", flush=True)
import mmaction
print(f"[{time.time()-t0:.1f}s] mmaction ok: {mmaction.__version__}", flush=True)

print(f"[{time.time()-t0:.1f}s] ALL DONE", flush=True)
