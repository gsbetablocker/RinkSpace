"""Current four-programme inputs; no historical production-chain imports."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from rinkspace.storage import configured_data_root
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import ParameterState, SceneParameterModel
from rinkspace.session.gpu_geometry import NativeSceneGPU

DATA = configured_data_root(ROOT)
EVENT = ROOT / 'events/ankara_2026'
OUT = DATA / 'events/ankara_2026/scene/joint_refine_1117'
# Current repair uses the complete source packets. Historical sparse runs must
# select their input explicitly; an omitted environment variable must not undo it.
PREPARED = Path(os.environ.get('RINKSPACE_JOINT_PREPARED', str(OUT/'repair/prepared_v7'))).resolve()
LIBRARY = Path(os.environ.get('RINKSPACE_JOINT_LIBRARY', str(DATA / 'build/ankara_dense_1118/rinkspace_core.dll'))).resolve()
OLD_LIBRARY = DATA / 'build/jin_lane_goal_1110/rinkspace_core.dll'
FIELD = EVENT / 'scene/field/free_joint_1117.json'
PROGRAMMES = {'jin': 'jin_shuxian_sp', 'okada': 'okada_mei_sp',
              'chen': 'chen_xi_jin_sp', 'nayeon': 'nayeon_ko_sp'}
REGISTRATIONS = {k: EVENT / 'programmes' / v / (
    'fit_v11/provisional_final_1117.json' if k == 'okada' else
    'whole_quality_1117.json' if k == 'nayeon' else 'provisional_final_1117.json')
    for k, v in PROGRAMMES.items()}


def read(path):
    return json.loads(Path(path).read_bytes())


def safe_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        if len(str(path).encode('utf-16-le')) // 2 >= 260 or any(
                len(x.encode('utf-16-le')) // 2 > 255 for x in path.parts):
            raise ValueError('OneDrive path limit: ' + str(path))
    return path


def write(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = safe_path(path.with_suffix(path.suffix + '.writing'))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    atomic_replace(tmp,path)


def save(path, /, **values):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = safe_path(path.with_suffix('.writing.npz'))
    np.savez_compressed(tmp, **values)
    atomic_replace(tmp,path)


def atomic_replace(tmp,path):
    """Allow brief external Windows readers to finish without losing a checkpoint."""
    with checkpoint_lock(path):
        started=time.monotonic()
        while True:
            try:
                tmp.replace(path)
                return
            except PermissionError as error:
                # Our readers share the named mutex. Antivirus/indexer handles
                # do not; Windows can deny replacement until those handles close.
                if (os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33)
                        or not tmp.is_file() or time.monotonic()-started>=5.):
                    raise
                time.sleep(.05)


@contextmanager
def checkpoint_lock(path):
    """Coordinate readers and atomic replacement across Windows processes."""
    if os.name!='nt':
        yield;return
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateMutexW;create.argtypes=[wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR]
    create.restype=wintypes.HANDLE
    wait=kernel.WaitForSingleObject;wait.argtypes=[wintypes.HANDLE,wintypes.DWORD];wait.restype=wintypes.DWORD
    release=kernel.ReleaseMutex;release.argtypes=[wintypes.HANDLE]
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE]
    name='Local\\RinkSpaceCheckpoint_'+hashlib.sha256(os.path.normcase(str(Path(path).resolve())).encode()).hexdigest()
    handle=create(None,False,name)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        status=wait(handle,0xffffffff)
        if status not in (0,0x80):raise ctypes.WinError(ctypes.get_last_error())
        try:yield
        finally:release(handle)
    finally:close(handle)


@contextmanager
def shared_reader(path):
    with checkpoint_lock(path),Path(path).open('rb') as stream:yield stream


def arrays(path):
    with shared_reader(path) as stream, np.load(stream, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepared_packet_path(name):
    """Use a local packet or an explicit immutable packet reference in setup."""
    local = PREPARED / (name + '.npz')
    if local.is_file():
        return local
    reference = Path(read(PREPARED / 'setup.json')['shots'][name]['observations'])
    if not reference.is_absolute():
        reference = PREPARED / reference
    if not reference.is_file():
        raise FileNotFoundError(reference)
    return reference


def programme_state_sha(state,names):
    """Hash exactly the shared scene and this programme's render dependencies."""
    keys=['scene_values','layout_revision']
    for name in sorted(names):
        keys.extend(name+suffix for suffix in ('_pts','_times','_controls','_rigs','_active_folds'))
    digest=hashlib.sha256()
    for key in keys:
        value=np.ascontiguousarray(state.get(key,np.array([0],np.int8)) if key.endswith('_active_folds') else state[key])
        digest.update(key.encode());digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def sha(path):
    with shared_reader(path) as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numeric_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def registration(programme):
    return read(REGISTRATIONS[programme])


def original_native(programme, *, core=None):
    rec = registration(programme)
    link = rec['geometry_binding']
    path = ROOT / link['structure']['path']
    model = SceneParameterModel(read(path), path, optimize_scene=True)
    values = arrays(ROOT / link['snapshot']['path'])['scene_values']
    assert numeric_sha(values) == link['value_sha256']
    return NativeSceneGPU(path, core or NativeCore(OLD_LIBRARY),
                          parameter_state=ParameterState(model.initial_state.layout, values))


def shot_inputs():
    result = {}
    for program, directory in PROGRAMMES.items():
        rec = registration(program)
        scope = read(EVENT / 'programmes' / directory / 'rebuild_scope.json')
        camera = Path(rec.get('camera', rec.get('camera_parameters'))['path'])
        z = arrays(camera)
        source = rec['source'].get('path', rec['source'].get('registered_path'))
        for s in scope['shots']:
            sid = s['id']
            if program == 'nayeon' and sid == 's01':
                continue
            if 'shot_ids' in z:
                i = z['shot_ids'].tolist().index(sid)
                camera_index = int(z['camera_indices'][i])
                a, b = z['control_offsets'][i:i+2]
                u, v = z['source_offsets'][i:i+2]
                pts, times = z['source_pts'][u:v], z['local_exposure_seconds'][u:v]
                controls, rigs = z['controls'][a:b], z['rig_by_source_pts'][u:v]
                first = float(np.atleast_1d(z['first_control_time'])[i] if np.size(z['first_control_time']) > 1 else z['first_control_time'])
                dt = float(np.atleast_1d(z['control_dt'])[i] if np.size(z['control_dt']) > 1 else z['control_dt'])
            else:
                camera_index = s['camera_index']
                pts, controls = z[sid+'_pts'], z[sid+'_controls']
                rigs = np.broadcast_to(z[sid+'_rig'], (len(pts), len(z[sid+'_rig']))).copy()
                times = z.get(sid+'_exposure_times', (pts-pts[0])/12800.)
                first, dt = -.06, .02
            span = s.get('processing', s)
            np.testing.assert_array_equal(pts, np.arange(span['first_pts'], span['stop_pts_exclusive'], 256))
            assert dt == .02
            result[program+'_'+sid] = dict(program=program, shot=sid, camera_index=camera_index,
                source=source, source_sha256=rec['source']['sha256'], registration=str(REGISTRATIONS[program]),
                camera_path=str(camera), pts=pts, times=times, controls=controls, rigs=rigs,
                first=first, dt=dt)
    return result


def observation_path(program, sid):
    if program in ('jin', 'okada'):
        return DATA / 'runs/ankara_joint_1113/prepared_guarded' / f'{program}_{sid}.npz'
    base = DATA / 'events/ankara_2026/programmes' / PROGRAMMES[program]
    folder = base / ('fit_1115/final_qa_1117/corrected_observations' if program == 'chen' else
                     'fit_1116/review_all_1116/whole_quality_1117/reviewed_observations')
    manifest = folder / 'shot_packets.json'
    if manifest.exists():
        return Path(read(manifest)[sid])
    return folder / (sid+'.npz')


def observations(name, shot):
    path = observation_path(shot['program'], shot['shot'])
    a = arrays(path)
    np.testing.assert_array_equal(a['pts'], shot['pts'][a['idx']])
    if 'source_sha256' in a:
        assert a['source_sha256'].item() == shot['source_sha256']
    eligible = a.get('eligible', np.ones(len(a['pts']), bool))
    result = {k: a[k][eligible] for k in ('pts','idx','uv','path','station','fold','family')}
    result['material'] = a.get('material', np.repeat('inherited_qualified_source', len(eligible)))[eligible]
    result['origin_row'] = np.flatnonzero(eligible)
    return result, str(path)


def subset(a, mask):
    return {k: v[mask] for k, v in a.items()}


def source_corrections(name, a):
    """Apply explicit source-only decisions without changing original pixels.

    Centre-button identity has not been established for the remaining readings.
    Their inactive role is an admission decision, not an error-based rejection.
    """
    a = {k:v.copy() for k,v in a.items()}
    a['fold'][np.char.endswith(a['path'], ':center')] = 5
    path = OUT/'centre_source/source_corrections.json'
    if path.exists():
        for item in read(path)['decisions']:
            if item['shot'] != name:
                continue
            use=(a['origin_row']==item['origin_row'])&(a['pts']==item['pts'])
            if not use.any():
                continue
            if item['decision']=='rebind_inner':
                a['path'][use]=item['path']
                a['fold'][use]=item['original_fold']
            else:
                a['fold'][use]=5
    return a


def prepare():
    """Bind current state and a spatially/temporally thinned geometry dataset.

    Raw packets remain untouched. Existing folds remain development/check roles;
    they are not renamed independent final evidence after previous repairs.
    """
    shots = shot_inputs()
    meta, baseline = {}, {}
    coverage = {}
    for name, s in shots.items():
        a, path = observations(name, s)
        # Dense adjacent exposures are correlated. Use a 200 ms geometry grid;
        # old sparse Jin/Okada readings keep their full original time coverage.
        keep = np.ones(len(a['idx']), bool) if s['program'] in ('jin','okada') else (
            (a['idx'] % 10 == 0) | (a['idx'] <= 2) | (a['idx'] >= len(s['pts'])-3))
        take = np.flatnonzero(keep)
        # At most one original measurement per 8-pixel image cell and named
        # path per exposure, retaining the source UV rather than averaging it.
        key = np.rec.fromarrays((a['pts'][take], a['path'][take],
              np.floor(a['uv'][take,0]/8).astype(int), np.floor(a['uv'][take,1]/8).astype(int)))
        _, unique = np.unique(key, return_index=True)
        selected = take[np.sort(unique)]
        b = source_corrections(name, subset(a, selected))
        save(OUT / 'prepared' / (name+'.npz'), **b)
        for k in ('pts','times','controls','rigs'):
            baseline[name+'_'+k] = s[k]
        meta[name] = {k: v for k,v in s.items() if k not in ('pts','times','controls','rigs')}
        meta[name]['observations'] = path
        coverage[name] = dict(native_frames=len(s['pts']), qualified_rows=len(a['pts']),
            sampled_rows=len(b['pts']), fit_rows=int((b['fold']==0).sum()),
            qualified_frames=len(np.unique(a['pts'])), sampled_frames=len(np.unique(b['pts'])),
            families={str(k):int(v) for k,v in zip(*np.unique(b['family'],return_counts=True))})
        print(name, coverage[name]['fit_rows'], '/', coverage[name]['qualified_rows'], flush=True)
    save(OUT/'prepared/baseline.npz', **baseline)
    write(OUT/'prepared/setup.json', dict(schema='rinkspace.four_programme_prepared.v1',shots=meta,
          source_clock='native integer PTS; 1/12800',fold_policy='inherited development/check roles; no new independence claim',
          sampling='200 ms dense-source geometry grid; one source sample per 8px cell/path/exposure'))
    write(OUT/'prepared/coverage.json', coverage)


if __name__ == '__main__':
    prepare()
