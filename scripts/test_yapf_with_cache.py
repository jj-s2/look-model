"""测试设置 LOCALAPPDATA 到项目内目录后 import yapf"""
import os
import sys
import time

# 重定向 LOCALAPPDATA 到项目内目录，避免 sandbox 限制
project_dir = r"C:\Users\John\Desktop\look model"
cache_dir = os.path.join(project_dir, ".cache", "appdata")
os.makedirs(cache_dir, exist_ok=True)
os.environ["LOCALAPPDATA"] = cache_dir

print(f"[{time.strftime('%H:%M:%S')}] LOCALAPPDATA redirected to: {cache_dir}", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] importing yapf...", flush=True)

t0 = time.time()
import yapf
print(f"[{time.strftime('%H:%M:%S')}] yapf: {yapf.__version__} ({time.time()-t0:.1f}s)", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] importing mmengine...", flush=True)
import mmengine
print(f"[{time.strftime('%H:%M:%S')}] mmengine: {mmengine.__version__} ({time.time()-t0:.1f}s)", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] importing mmaction...", flush=True)
import mmaction
print(f"[{time.strftime('%H:%M:%S')}] mmaction: {mmaction.__version__} ({time.time()-t0:.1f}s)", flush=True)

print("DONE", flush=True)
