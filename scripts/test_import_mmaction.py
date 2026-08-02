"""测试 import mmaction 是否卡住 - 用脚本文件避免命令行转义"""
import sys
import time

print(f"start at {time.strftime('%H:%M:%S')}", flush=True)
t0 = time.time()

print("importing torch...", flush=True)
import torch
print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}", flush=True)

print("importing mmaction...", flush=True)
import mmaction
print(f"mmaction: {mmaction.__version__}", flush=True)

print("importing mmaction.apis...", flush=True)
from mmaction.apis import detection_inference, inference_skeleton, init_recognizer, pose_inference
print("mmaction.apis ok", flush=True)

print(f"DONE at {time.strftime('%H:%M:%S')}, elapsed {time.time()-t0:.1f}s", flush=True)
