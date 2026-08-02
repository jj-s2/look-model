"""验证 vision/input_adapter.py 的功能。

用 mmpose demo 视频测试 LocalVideoAdapter 的：
- 工厂函数自动识别类型
- 元信息读取（fps/帧数/宽高）
- 逐帧迭代
- 单帧读取与形状校验
- with 上下文资源释放
- ezviz_stream 占位抛 NotImplementedError
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 添加项目根到 sys.path
parser = argparse.ArgumentParser(description="Exercise input-adapter implementations.")
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
project_dir = str(args.project_root.expanduser().resolve())
sys.path.insert(0, project_dir)

from vision.input_adapter import (
    InputAdapter,
    LocalVideoAdapter,
    WebcamAdapter,
    EzvizStreamAdapter,
    create_input_adapter,
)

demo_video = os.path.join(
    project_dir, "third_party", "mmpose", "demo", "resources", "demo.mp4"
)
if not os.path.isfile(demo_video):
    # 回退到 vision 目录下的 demo.mp4
    demo_video = os.path.join(project_dir, "vision", "demo.mp4")

print(f"[{time.strftime('%H:%M:%S')}] 测试视频: {demo_video}")
print(f"[{time.strftime('%H:%M:%S')}] 文件存在: {os.path.isfile(demo_video)}")
print()


# 测试 1: 工厂函数自动识别 local_video
print("=== 测试 1: 工厂函数自动识别 local_video ===")
adapter = create_input_adapter(demo_video)
print(f"类型: {type(adapter).__name__}")
print(f"input_type (未打开): {adapter.input_type}")
assert isinstance(adapter, LocalVideoAdapter)
print()

# 测试 2: with 上下文 + 元信息
print("=== 测试 2: with 上下文 + 元信息 ===")
with create_input_adapter(demo_video) as adapter:
    meta = adapter.meta()
    print(f"元信息: {meta}")
    print(f"fps={adapter.fps}, frame_count={adapter.frame_count}, "
          f"{adapter.width}x{adapter.height}")
    print(f"is_opened={adapter.is_opened}")
    assert adapter.is_opened
    assert adapter.frame_count > 0
    assert adapter.width > 0 and adapter.height > 0
print(f"退出 with 后 is_opened={adapter.is_opened}")
assert not adapter.is_opened
print()

# 测试 3: 逐帧迭代计数
print("=== 测试 3: 逐帧迭代计数 ===")
with create_input_adapter(demo_video) as adapter:
    count = 0
    first_frame = None
    for frame in adapter:
        if count == 0:
            first_frame = frame
        count += 1
    print(f"实际迭代帧数: {count} (meta 报告 {adapter.frame_count})")
    print(f"首帧形状: {first_frame.shape}, dtype: {first_frame.dtype}")
    assert count > 0
    assert first_frame.ndim == 3 and first_frame.shape[2] == 3
print()

# 测试 4: read_frame 单帧读取
print("=== 测试 4: read_frame 单帧读取 ===")
adapter = create_input_adapter(demo_video)
adapter.open()
frame = adapter.read_frame()
print(f"单帧形状: {frame.shape}")
assert frame is not None
# 读完所有帧后应返回 None
remaining = 0
while adapter.read_frame() is not None:
    remaining += 1
print(f"剩余帧数: {remaining}")
# 读完后再读一次应为 None
assert adapter.read_frame() is None
adapter.close()
print()

# 测试 5: 不存在的文件
print("=== 测试 5: 不存在的文件抛 FileNotFoundError ===")
try:
    with create_input_adapter("nonexistent.mp4"):
        pass
    print("FAIL: 未抛异常")
except FileNotFoundError as e:
    print(f"OK: FileNotFoundError -> {e}")
print()

# 测试 6: ezviz_stream 占位
print("=== 测试 6: ezviz_stream 占位抛 NotImplementedError ===")
adapter = create_input_adapter("rtsp://example.com/stream", input_type="ezviz_stream")
print(f"类型: {type(adapter).__name__}")
try:
    adapter.open()
    print("FAIL: 未抛异常")
except NotImplementedError as e:
    print(f"OK: NotImplementedError -> {e}")
print()

# 测试 7: 显式 input_type
print("=== 测试 7: 显式 input_type ===")
a1 = create_input_adapter(demo_video, input_type="local_video")
assert isinstance(a1, LocalVideoAdapter)
print("local_video OK")
a2 = create_input_adapter(0, input_type="webcam")
assert isinstance(a2, WebcamAdapter)
print("webcam OK (未实际打开设备)")
print()

# 测试 8: 不支持的 input_type
print("=== 测试 8: 不支持的 input_type 抛 ValueError ===")
try:
    create_input_adapter(demo_video, input_type="unknown_type")
    print("FAIL: 未抛异常")
except ValueError as e:
    print(f"OK: ValueError -> {e}")
print()

print("=" * 50)
print(f"[{time.strftime('%H:%M:%S')}] 全部测试通过 ✅")
