"""Stream exact original exposures; no re-timing, nearest-frame fallback or old reader."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator
import av
import numpy as np


class SourceFrameError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NativeImage:
    source_frame_id: str
    native_pts: int
    time_base: Fraction
    rgb: np.ndarray

    @property
    def source_seconds(self):
        return float(self.native_pts * self.time_base)


class NativeVideoReader:
    """A streaming current-source consumer, not a production/qualification chain.

    It checks the current clock, source dimensions and requested PTS. It never
    hashes or regenerates historical evidence. Decoded images do not become fit
    observations until a separately identified material feature is admitted.
    """
    def __init__(self, path, data_root, clock, width, height, fps_num, fps_den):
        self.path = Path(path).resolve()
        root = Path(data_root).resolve()
        try:
            self.path.relative_to(root)
        except ValueError as exc:
            raise SourceFrameError("原片必须位于配置的Data根内") from exc
        self.clock = clock
        self.time_base = Fraction(clock.time_base_num, clock.time_base_den)
        self.width, self.height = width, height
        self.fps = Fraction(fps_num, fps_den)
        if not self.path.is_file():
            raise SourceFrameError(f"当前原片不存在: {self.path}")

    def frames(self, source_frames) -> Iterator[NativeImage]:
        requested = tuple(source_frames)
        if not requested:
            return
        for frame in requested:
            if frame.clock_id != self.clock.id or (frame.width,frame.height) != (self.width,self.height):
                raise SourceFrameError(f"{frame.id}: 不是当前源时钟/原生尺寸")
            if type(frame.native_pts) is not int:
                raise SourceFrameError(f"{frame.id}: PTS必须为原始整数")
        if any(b.native_pts <= a.native_pts for a,b in zip(requested,requested[1:])):
            raise SourceFrameError("源帧请求须原生PTS严格递增且不重复；不得按导出序号补帧")
        with av.open(str(self.path), mode="r") as container:
            if not container.streams.video:
                raise SourceFrameError("当前源没有视频轨")
            stream = container.streams.video[0]
            if Fraction(stream.time_base) != self.time_base:
                raise SourceFrameError(f"源time_base {stream.time_base} 与声明 {self.time_base} 不一致")
            if (stream.width,stream.height) != (self.width,self.height) or Fraction(stream.average_rate) != self.fps:
                raise SourceFrameError("当前源的原生尺寸或帧率与节目绑定不一致")
            stream.thread_type = "AUTO"
            container.seek(requested[0].native_pts, stream=stream, backward=True, any_frame=False)
            next_index=0
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                target=requested[next_index]
                if decoded.pts < target.native_pts:
                    continue
                if decoded.pts > target.native_pts:
                    raise SourceFrameError(f"{target.id}: 原片缺少精确PTS {target.native_pts}，不能替换为 {decoded.pts}")
                image=decoded.to_ndarray(format="rgb24")
                if image.shape != (self.height,self.width,3):
                    raise SourceFrameError(f"{target.id}: 解码图像尺寸不一致")
                image.flags.writeable=False
                yield NativeImage(target.id,int(decoded.pts),self.time_base,image)
                next_index += 1
                if next_index == len(requested):
                    return
            raise SourceFrameError(f"原片结束仍缺少PTS {requested[next_index].native_pts}")

    def frame(self, source_frame):
        return next(self.frames((source_frame,)))
