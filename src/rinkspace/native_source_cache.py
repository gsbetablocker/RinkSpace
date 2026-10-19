"""Read authenticated existing native source PNGs without creating a cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _file_identity(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def authenticate_missing_native_png(path, *, resolve_source=Path, authenticate_file=_sha):
    """Reproduce a missing historical PNG's SHA entirely in memory.

    Only the known complete secondary lossless-cache schema is supported.
    Every original video byte is authenticated through ``authenticate_file``;
    a construction scope may deduplicate that real file hash and revalidate it
    at exit. No missing original PNG, cache or manifest is written.
    """
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Missing-PNG authentication only applies to absent files')
    manifest = path.parent / 'complete_original_native_cache_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(path)
    manifest_identity = _file_identity(manifest)
    encoded = manifest.read_bytes()
    cache = json.loads(encoded)
    if cache.get('schema') != 'rinkspace.complete-secondary-lossless-native-source-cache.v1':
        raise ValueError('Missing PNG has an unsupported cache manifest')
    manifest_sha = authenticate_file(manifest)
    if manifest_sha != hashlib.sha256(encoded).hexdigest():
        raise ValueError('Missing-PNG manifest changed while reading')
    rows = [r for r in cache['frames'] if Path(r['raw_png']).name == path.name]
    if len(rows) != 1:
        raise ValueError('Missing PNG does not have one exact original cache row')
    row = rows[0]
    pts = int(row['native_pts'])
    if sum(int(r['native_pts']) == pts for r in cache['frames']) != 1:
        raise ValueError('Missing-PNG manifest repeats a native source presentation')
    if cache['native_timebase'] != '1/15360':
        raise ValueError('Missing-PNG original source time base changed')
    source = Path(resolve_source(cache['source']['path'])).resolve()
    source_identity = _file_identity(source)
    source_sha = authenticate_file(source)
    if source_sha != cache['source']['sha256']:
        raise ValueError('Missing-PNG original video SHA changed')
    import av
    import cv2
    pixels = None
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        if str(stream.time_base) != cache['native_timebase']:
            raise ValueError('Decoded original video time base changed')
        container.seek(pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts > pts:
                break
            if frame.pts == pts:
                pixels = frame.to_ndarray(format='bgr24')
                break
    if pixels is None or pixels.shape != (1080, 1920, 3):
        raise ValueError('Exact original native presentation is unavailable: ' + str(pts))
    bgr_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    if bgr_sha != row['source_BGR_sha256']:
        raise ValueError('Missing-PNG original native BGR differs: ' + str(pts))
    # Historical producer review_frozen_secondary_complete_cuts_cached.py
    # writes this exact lossless encoding; defaults produce different bytes.
    ok, png = cv2.imencode('.png', pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError('Original PNG memory encoding failed')
    png_sha = hashlib.sha256(png.tobytes()).hexdigest()
    if png_sha != row['raw_png_sha256']:
        raise ValueError('Missing-PNG memory reproduction differs from original PNG SHA: ' + str(pts))
    if (_file_identity(manifest) != manifest_identity or
            _file_identity(source) != source_identity):
        raise ValueError('Missing-PNG authenticated inputs changed during memory reproduction')
    receipt = dict(schema='rinkspace.missing-original-native-PNG-memory-authentication.v1',
        helper_path=str(Path(__file__).resolve()), helper_sha256=_sha(__file__),
        requested_path=str(path), native_pts=pts, actual_pts=int(row['actual_pts']),
        manifest_path=str(manifest), manifest_sha256=manifest_sha, manifest_identity=list(manifest_identity),
        source_path=str(source), source_sha256=source_sha, source_identity=list(source_identity),
        source_time_base=cache['native_timebase'], native_BGR_sha256=bgr_sha,
        reproduced_original_PNG_sha256=png_sha, PNG_compression=1,
        encoded_PNG_bytes=len(png), original_video_file_SHA_authenticated=True,
        original_native_presentation_not_replaced_by_exposure_alias=True,
        original_manifest_modified=False, new_source_images_written=0)
    return png_sha, receipt


class AuthenticatedNativeSourceCache:
    """Bind recorded original pixels to a camera's original exposure clock.

    The caller supplies the already authenticated camera's source identity and
    native-to-actual aliases. Filenames never establish source identity or PTS.
    Each selected PNG and decoded BGR buffer is checked on every read.
    """

    def __init__(self, manifests, *, source_path, source_sha256,
                 source_time_base, aliases, resolve_source=Path):
        self.aliases = {int(p): int(a) for p, a in aliases.items()}
        self.manifests = []
        self.rows = {}
        source_path = Path(resolve_source(source_path)).resolve()
        for manifest in manifests:
            manifest = Path(resolve_source(manifest)).resolve()
            encoded = manifest.read_bytes()
            cache = json.loads(encoded)
            self.manifests.append(dict(path=str(manifest),
                sha256=hashlib.sha256(encoded).hexdigest()))
            complete = cache.get('schema') == 'rinkspace.complete-secondary-lossless-native-source-cache.v1'
            if complete:
                source = cache['source']
                path, digest, clock = source['path'], source['sha256'], cache['native_timebase']
            elif isinstance(cache.get('source_video'), dict):
                source = cache['source_video']
                path = source['path']
                digest = source.get('sha256', source.get('recorded_sha256'))
                clock = source.get('time_base')
            elif 'source_sha256_from_timeline' in cache:
                path, digest = cache['source_path'], cache['source_sha256_from_timeline']
                clock = cache.get('source_time_base', cache.get('native_timebase'))
            else:
                raise ValueError('Cache manifest has no recorded original source identity: ' + str(manifest))
            if (Path(resolve_source(path)).resolve() != source_path or
                    digest != source_sha256 or clock != source_time_base):
                raise ValueError('Cache original source identity/clock differs: ' + str(manifest))
            seen = set()
            for item in cache['frames']:
                native = int(item.get('native_pts', item.get('pts')))
                if native in seen:
                    raise ValueError('Cache repeats a native presentation: ' + str(native))
                seen.add(native)
                if native not in self.aliases:
                    raise ValueError('Cache PTS lies outside authenticated camera clock: ' + str(native))
                actual = self.aliases[native]
                if 'actual_pts' in item and int(item['actual_pts']) != actual:
                    raise ValueError('Cache actual exposure alias differs: ' + str(native))
                original = Path(item['raw_png'] if complete else item['path'])
                image_path = Path(resolve_source(original)).resolve()
                if complete and not image_path.is_file():
                    # Historical relocation is authenticated by both hashes.
                    image_path = manifest.parent / original.name
                png_sha = item['raw_png_sha256'] if complete else item.get('sha256', item.get('png_sha256'))
                bgr_sha = (item['source_BGR_sha256'] if complete else
                    item.get('native_bgr24_sha256', item.get('native_sha256')))
                if not png_sha or not bgr_sha:
                    raise ValueError('Cache row needs original PNG and decoded BGR hashes: ' + str(native))
                row = dict(cache_native_pts=native, actual_exposure_pts=actual,
                    path=str(image_path), png_sha256=png_sha, native_bgr24_sha256=bgr_sha,
                    manifest=self.manifests[-1])
                previous = self.rows.get(native)
                if previous and previous['native_bgr24_sha256'] != bgr_sha:
                    raise ValueError('Overlapping cache pixels disagree for native presentation: ' + str(native))
                self.rows.setdefault(native, row)

    def row_for_pts(self, native):
        native = int(native)
        if native not in self.aliases:
            raise ValueError('Requested PTS lies outside authenticated camera clock: ' + str(native))
        # Different encoded presentations can represent one actual exposure
        # while retaining different decoded RGB bytes. Never replace their
        # pixels with the representative's pixels through an alias.
        row = self.rows.get(native)
        if row is None:
            raise ValueError('Existing original cache omits requested exposure: ' + str(native))
        return dict(row, requested_native_pts=native)

    def read(self, native):
        import cv2
        row = self.row_for_pts(native)
        path = Path(row['path'])
        if not path.is_file() or _sha(path) != row['png_sha256']:
            raise ValueError('Cached original PNG changed: ' + str(native))
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if (pixels is None or pixels.shape != (1080, 1920, 3) or
                hashlib.sha256(pixels.tobytes()).hexdigest() != row['native_bgr24_sha256']):
            raise ValueError('Cached original native BGR changed: ' + str(native))
        return pixels, row

    def assert_unchanged(self):
        for manifest in self.manifests:
            if _sha(manifest['path']) != manifest['sha256']:
                raise ValueError('Original cache manifest changed during review: ' + manifest['path'])
