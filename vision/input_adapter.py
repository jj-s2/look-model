"""输入适配模块。

统一封装本地视频、电脑摄像头、海康萤石云流三种输入源，
向下游（人体检测 / 关键点提取 / 骨架时序分类）提供逐帧 BGR ndarray 接口。

支持类型：
- local_video  : 本地视频文件（mp4/avi/mov 等 OpenCV 可读格式）
- webcam       : 本机摄像头（通过设备索引指定）
- ezviz_stream : 萤石云 RTSP/RTMP 流（本期仅预留接口，不实现）

设计原则：
- 仅依赖 numpy + cv2，避免在输入层引入 OpenMMLab 重型依赖
- 统一接口：open / close / read_frame / __iter__ / 元信息属性
- 资源安全：支持 with 上下文管理，异常时自动释放 VideoCapture
"""
from __future__ import annotations

import abc
import os
from typing import Iterator, Optional

import cv2
import numpy as np


class InputAdapter(abc.ABC):
    """输入适配器抽象基类。

    子类需实现 `_open_capture` 返回 cv2.VideoCapture，
    并可选覆盖 `_build_meta` 补充输入源特有元信息。
    """

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self._cap: Optional[cv2.VideoCapture] = None
        self._meta: dict = {}

    # ---- 生命周期 ----
    def open(self) -> "InputAdapter":
        """打开输入源，失败抛 RuntimeError。"""
        if self._cap is not None:
            return self
        cap = self._open_capture()
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"无法打开输入源: {self.source!r}")
        self._cap = cap
        self._meta = self._build_meta()
        return self

    def close(self) -> None:
        """释放输入源资源。"""
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __enter__(self) -> "InputAdapter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---- 核心读取 ----
    def read_frame(self) -> Optional[np.ndarray]:
        """读取下一帧，返回 BGR HWC ndarray；流结束或失败返回 None。"""
        if self._cap is None:
            raise RuntimeError("输入源未打开，请先调用 open() 或使用 with 语句")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def __iter__(self) -> Iterator[np.ndarray]:
        """逐帧迭代，直到流结束。"""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    # ---- 元信息 ----
    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return float(self._meta.get("fps", 0.0))

    @property
    def frame_count(self) -> int:
        return int(self._meta.get("frame_count", 0))

    @property
    def width(self) -> int:
        return int(self._meta.get("width", 0))

    @property
    def height(self) -> int:
        return int(self._meta.get("height", 0))

    @property
    def input_type(self) -> str:
        return self._meta.get("input_type", "unknown")

    def meta(self) -> dict:
        """返回输入源元信息字典（副本）。"""
        return dict(self._meta)

    # ---- 子类实现 ----
    @abc.abstractmethod
    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """子类实现具体的 VideoCapture 创建逻辑。"""

    def _build_meta(self) -> dict:
        """从 VideoCapture 读取通用元信息，子类可覆盖补充。"""
        cap = self._cap
        meta = {
            "input_type": "unknown",
            "source": str(self.source),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        return meta


class LocalVideoAdapter(InputAdapter):
    """本地视频文件适配器。

    source: 视频文件路径（mp4/avi/mov 等）
    """

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        path = str(self.source)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"视频文件不存在: {path}")
        return cv2.VideoCapture(path)

    def _build_meta(self) -> dict:
        meta = super()._build_meta()
        meta["input_type"] = "local_video"
        meta["path"] = os.path.abspath(str(self.source))
        meta["file_size"] = os.path.getsize(meta["path"])
        return meta


class WebcamAdapter(InputAdapter):
    """本机摄像头适配器。

    source: 设备索引（0 表示默认摄像头）或设备路径字符串
    kwargs:
        width  : 期望采集宽度（可选，由驱动协商）
        height : 期望采集高度（可选）
        fps    : 期望采集帧率（可选）
    """

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        # 协商期望参数
        if "width" in self.kwargs:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.kwargs["width"])
        if "height" in self.kwargs:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.kwargs["height"])
        if "fps" in self.kwargs:
            cap.set(cv2.CAP_PROP_FPS, self.kwargs["fps"])
        return cap

    def _build_meta(self) -> dict:
        meta = super()._build_meta()
        meta["input_type"] = "webcam"
        meta["device"] = self.source
        # 摄像头通常无法预知总帧数
        meta["frame_count"] = -1
        return meta


class EzvizStreamAdapter(InputAdapter):
    """萤石云流适配器（预留）。

    本期不实现，调用即抛 NotImplementedError。
    预留接入点：RTSP/RTMP URL 或萤石云 OpenAPI 返回的播放地址。
    """

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        raise NotImplementedError(
            "ezviz_stream 输入类型本期未实现，"
            "请使用 local_video 或 webcam。后续可通过 RTSP URL 接入。"
        )


# ---- 输入源类型识别 ----
def _detect_input_type(source) -> str:
    """根据 source 形态推断输入类型。

    - int                            -> webcam
    - str 以 rtsp/rtmp/http 开头      -> ezviz_stream（预留）
    - str 全数字                      -> webcam
    - str 看起来像文件路径（含扩展名） -> local_video
      （文件是否存在由 LocalVideoAdapter 在 open 时检查并抛 FileNotFoundError）
    """
    if isinstance(source, int):
        return "webcam"
    if isinstance(source, str):
        lower = source.lower().strip()
        if lower.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "ezviz_stream"
        if source.isdigit():
            return "webcam"
        # 含路径分隔符或文件扩展名的字符串视为本地视频路径
        if os.path.sep in source or "/" in source or "\\" in source:
            return "local_video"
        _, ext = os.path.splitext(source)
        if ext:  # 有扩展名，视为视频文件路径
            return "local_video"
    raise ValueError(f"无法识别输入源类型: {source!r}")


def create_input_adapter(source, input_type: Optional[str] = None, **kwargs) -> InputAdapter:
    """输入适配器工厂函数。

    参数:
        source      : 输入源（文件路径 / 设备索引 / 流地址）
        input_type  : 显式指定类型，可选值 local_video / webcam / ezviz_stream
                      未指定时按 source 形态自动推断
        **kwargs    : 传递给具体 Adapter 的额外参数（如 webcam 的 width/height/fps）

    返回:
        InputAdapter 实例（未打开，需调用 open() 或使用 with 语句）

    示例:
        >>> with create_input_adapter("demo.mp4") as adapter:
        ...     for frame in adapter:
        ...         process(frame)
        >>> with create_input_adapter(0, width=640, height=480) as cam:
        ...     frame = cam.read_frame()
    """
    if input_type is None:
        input_type = _detect_input_type(source)

    registry = {
        "local_video": LocalVideoAdapter,
        "webcam": WebcamAdapter,
        "ezviz_stream": EzvizStreamAdapter,
    }
    if input_type not in registry:
        raise ValueError(
            f"不支持的 input_type: {input_type!r}，"
            f"可选值: {list(registry.keys())}"
        )
    return registry[input_type](source, **kwargs)


__all__ = [
    "InputAdapter",
    "LocalVideoAdapter",
    "WebcamAdapter",
    "EzvizStreamAdapter",
    "create_input_adapter",
]
