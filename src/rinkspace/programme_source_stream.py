"""Read every saved native programme frame once, without a new image cache.

The frozen clock owns exposure aliases. Camera projection is available only
for navigating source searches; no projected coordinate is a measurement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FrozenProgrammeSource:
    def __init__(self, state_identity_path, source_path, shot_manifest_path):
        self.identity_path = Path(state_identity_path).resolve()
        self.identity = json.loads(self.identity_path.read_text(encoding='utf8'))
        camera_path = Path(self.identity['camera_path'])
        scene_path = self.identity_path.with_name('scene_UNACCEPTED.json')
        if (file_sha(camera_path) != self.identity['camera_sha256']
                or file_sha(scene_path) != self.identity['scene_sha256']):
            raise ValueError('Frozen programme camera/scene changed')
        self.source_path = Path(source_path).resolve()
        with np.load(camera_path, allow_pickle=False) as saved:
            self.native_pts = saved['pts'].copy()
            self.actual_pts = saved['exposure_representative_pts'].copy()
            self.cameras = saved['camera25'].copy()
            self.installations = saved['installation_id'].copy()
            self.metadata = json.loads(str(saved['metadata_json'].item()))
        self.scene = json.loads(scene_path.read_text(encoding='utf8'))
        self.source_sha256 = self.metadata['source_sha256']
        self.time_base = self.metadata['source_time_base']
        if file_sha(self.source_path) != self.source_sha256:
            raise ValueError('Original programme source bytes changed')
        n = len(self.native_pts)
        if (self.cameras.shape != (n, 25) or self.actual_pts.shape != (n,)
                or self.installations.shape != (n,) or np.any(np.diff(self.native_pts) <= 0)
                or not set(map(int, self.actual_pts)).issubset(set(map(int, self.native_pts)))):
            raise ValueError('Complete original frame/camera/exposure clock required')
        self.lookup = {int(p): i for i, p in enumerate(self.native_pts)}
        self.actual_inventory = np.unique(self.actual_pts)
        shots = json.loads(Path(shot_manifest_path).read_text(encoding='utf8'))
        if shots['source_sha256_from_timeline'] != self.source_sha256:
            raise ValueError('Source shots belong to another original video')
        self.shots = shots['shots']
        self.source_binding = dict(source_sha256=self.source_sha256,
            time_base=self.time_base, source_path=str(self.source_path),
            source_resolution=[1920, 1080], native_hw=[1080, 1920],
            image_domain='native_brown_pixels')

    def frames(self, *, start_pts=None, stop_pts=None, cpu_threads=1):
        """Yield each exact native PTS in order from one original-video scan."""
        wanted = self.native_pts
        if start_pts is not None:
            wanted = wanted[wanted >= int(start_pts)]
        if stop_pts is not None:
            wanted = wanted[wanted < int(stop_pts)]
        if not len(wanted):
            return
        ordinal = 0
        with av.open(str(self.source_path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = int(cpu_threads)
            if str(stream.time_base) != self.time_base:
                raise ValueError('Original decoder time base changed')
            container.seek(int(wanted[0]), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts < wanted[0]:
                    continue
                if ordinal == len(wanted):
                    break
                target = int(wanted[ordinal])
                if frame.pts < target:
                    continue
                if frame.pts != target:
                    raise ValueError(f'Original frame omitted: {target}')
                pixels = frame.to_ndarray(format='bgr24')
                if pixels.shape != (1080, 1920, 3) or pixels.dtype != np.uint8:
                    raise ValueError('Original native pixel domain changed')
                i = self.lookup[target]
                shot = [s for s in self.shots if int(s['start_pts']) <= target < int(s['end_pts'])]
                if len(shot) != 1:
                    raise ValueError(f'Original frame has no unique source shot: {target}')
                yield dict(native_pts=target, actual_pts=int(self.actual_pts[i]),
                    representative=target == int(self.actual_pts[i]),
                    camera25=self.cameras[i], installation_id=str(self.installations[i]),
                    raw_bgr=pixels, source_binding={**self.source_binding,
                        'source_pts': target, 'actual_pts': int(self.actual_pts[i]),
                        'shot_id': shot[0]['id'],
                        'decoded_native_BGR_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
                ordinal += 1
        if ordinal != len(wanted):
            raise ValueError(f'Original frame coverage incomplete: {ordinal}/{len(wanted)}')
