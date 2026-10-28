"""Read-only sources, exact native timestamps, and explicit fresh output folders."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile
import hashlib, json, shutil, subprocess
import numpy as np
import cv2

BUNDLE_MEMBERS = ('media/source_s01.mp4', 'media/source_s01_clock.json',
                  'current/camera_trajectory.npz', 'diagnostics/current_camera_physical.json')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj) -> None:
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert,
                               allow_nan=False), encoding='utf-8')

def fresh_output(path: Path, inputs: list[Path]) -> Path:
    p = path.expanduser().resolve()
    for source in inputs:
        s = source.expanduser().resolve()
        if p == s or (s.is_dir() and p.is_relative_to(s)):
            raise ValueError('Output must not be inside an input directory.')
    if p.exists(): raise FileExistsError(f'Refusing to overwrite output: {p}')
    p.mkdir(parents=True)
    return p

def prepare_bundle(source: Path, out: Path) -> tuple[Path, dict, Path | None, dict]:
    source = source.resolve(strict=True)
    hashes = {}
    if source.is_file():
        hashes[str(source)] = sha256(source)
        root = out / '_input_cache'
        with ZipFile(source) as z:
            names = set(z.namelist())
            for name in BUNDLE_MEMBERS:
                if name not in names:
                    if name.startswith('media/'): raise ValueError(f'Missing {name}')
                    continue
                info = z.getinfo(name)
                if info.file_size > 256_000_000: raise ValueError('Unexpectedly large bundle member')
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as a, target.open('xb') as b: shutil.copyfileobj(a, b)
    else:
        root = source
        for name in BUNDLE_MEMBERS:
            p = root / name
            if p.is_file(): hashes[str(p)] = sha256(p)
    clock = json.loads((root / BUNDLE_MEMBERS[1]).read_text(encoding='utf-8'))
    video = root / BUNDLE_MEMBERS[0]
    if sha256(video) != clock['clip_sha256']: raise ValueError('Source video SHA256 mismatch')
    camera = root / BUNDLE_MEMBERS[2]
    return video, clock, camera if camera.is_file() else None, hashes

def assert_unchanged(hashes: dict[str, str]) -> None:
    for p, h in hashes.items():
        if sha256(Path(p)) != h: raise RuntimeError(f'Input changed during run: {p}')

def probe_video(video: Path, ffprobe='ffprobe') -> dict:
    cmd = [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'stream=width,height,time_base,sample_aspect_ratio:frame=pts,width,height', '-of', 'json', str(video)]
    try: obj = json.loads(subprocess.check_output(cmd, stderr=subprocess.PIPE))
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError('ffprobe is required; install FFmpeg or supply --ffprobe.') from e
    if len(obj.get('streams', [])) != 1: raise ValueError('Expected one selected video stream')
    stream = obj['streams'][0]
    frames = obj.get('frames', [])
    if not frames or any('pts' not in f for f in frames): raise ValueError('Native frame PTS unavailable')
    pts = [int(f['pts']) for f in frames]
    if any(b <= a for a, b in zip(pts, pts[1:])): raise ValueError('Frame PTS must be strictly increasing')
    size = (int(stream['width']), int(stream['height']))
    if any((f.get('width', size[0]), f.get('height', size[1])) != size for f in frames):
        raise ValueError('Resolution changes require separate segments')
    return dict(width=size[0], height=size[1], time_base=stream['time_base'],
                sample_aspect_ratio=stream.get('sample_aspect_ratio', 'unknown'), pts=pts)

def select_frames(probe: dict, start: int, end: int, stride_s: float, max_frames: int) -> list[int]:
    if stride_s <= 0 or max_frames < 3 or start > end: raise ValueError('Invalid sampling limits')
    pts = np.asarray(probe['pts'], np.int64)
    valid = np.flatnonzero((pts >= start) & (pts <= end))
    if len(valid) < 3: raise ValueError('Fewer than 3 native frames in interval')
    tb = float(Fraction(probe['time_base']))
    targets = np.arange(pts[valid[0]], pts[valid[-1]] + 0.1, stride_s / tb)
    targets = np.r_[targets, pts[valid[-1]]]
    if len(targets) > max_frames: targets = np.linspace(pts[valid[0]], pts[valid[-1]], max_frames)
    chosen = sorted({int(valid[np.argmin(np.abs(pts[valid] - t))]) for t in targets})
    return chosen

def decode_selected(video: Path, probe: dict, indices: list[int], out: Path,
                    clock: dict | None = None, ffmpeg='ffmpeg') -> list[dict]:
    """Decode display-order indices from ffprobe, never derive source time from FPS."""
    out.mkdir(parents=True, exist_ok=True)
    terms = [f'eq(n\\,{i})' for i in indices]
    selected = '+'.join('(' + '+'.join(terms[k:k+20]) + ')' for k in range(0, len(terms), 20))
    cmd = [ffmpeg, '-v', 'error', '-threads', '2', '-i', str(video), '-map', '0:v:0',
           '-vf', f'select={selected}', '-vsync', '0', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    expected = {int(f['clip_pts']): f for f in (clock or {}).get('frames', [])}
    offset = int((clock or {}).get('source_pts_equals_clip_pts_plus', 0))
    byte_count = probe['width'] * probe['height'] * 3
    stderr_path = out / 'ffmpeg_stderr.txt'
    result = []
    with stderr_path.open('wb') as stderr:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr)
        try:
            for index in indices:
                raw = bytearray()
                while len(raw) < byte_count:
                    part = proc.stdout.read(byte_count - len(raw))
                    if not part: break
                    raw.extend(part)
                if len(raw) != byte_count: raise RuntimeError(f'Decode failed; see {stderr_path}')
                clip_pts = probe['pts'][index]
                digest = hashlib.sha256(raw).hexdigest()
                verified = None
                if expected:
                    if clip_pts not in expected: raise ValueError('Selected PTS absent from supplied clock')
                    verified = digest == expected[clip_pts]['decoded_BGR_sha256']
                    if not verified: raise ValueError(f'Decoded BGR SHA mismatch at PTS {clip_pts}')
                frame = np.frombuffer(raw, np.uint8).reshape(probe['height'], probe['width'], 3)
                filename = f'frame_{clip_pts + offset}.png'
                if not cv2.imwrite(str(out / filename), frame): raise OSError('Could not save frame')
                result.append(dict(pts=clip_pts + offset, clip_pts=clip_pts, decoded_index=index,
                    time_base=(clock or {}).get('source_time_base', probe['time_base']),
                    image=str(out / filename), decoded_bgr_sha256=digest, source_hash_verified=verified))
            if proc.stdout.read(1): raise RuntimeError('Unexpected extra decoded frames')
            if proc.wait() != 0: raise RuntimeError(f'ffmpeg failed; see {stderr_path}')
        finally:
            if proc.poll() is None: proc.kill()
            proc.wait()
    return result
