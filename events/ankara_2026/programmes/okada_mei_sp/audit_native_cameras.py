"""Read-only camera audit comparing explicit frozen native programme states."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.camera_diagnostics import inspect_shot_motion, inspect_lens_forward
from export_native_six_review import _manifest_for_library, _verify_native_library_binding


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()



def source_exposure_plan(manifest, observation_pts):
    """Full registered shot clocks; measurements only establish membership."""
    observed=np.asarray(observation_pts)
    if observed.ndim!=1 or observed.dtype.kind not in 'iu':
        raise ValueError('Observation PTS must be one-dimensional native integers')
    plan=[];row_stop=0
    for shot in manifest['shots']:
        # Historical prepared manifests omit this field; 256 is this
        # programme's registered 50 fps, TB 1/12800 exposure step.
        step=shot.get('native_pts_step',manifest.get('native_pts_step',
            manifest.get('source_clock',{}).get('native_pts_step',256)))
        first,last=shot['first_pts'],shot['last_pts']
        if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in (first,last,step))
                or step<=0 or last<first or (last-first)%step):
            raise ValueError('Invalid registered native exposure range: '+shot['id'])
        pts=np.arange(first,last+step,step,dtype=np.int64)
        begin,stop=shot['row_begin'],shot['row_stop']
        if begin!=row_stop or stop<begin or stop>len(observed):
            raise ValueError('Invalid shot observation row partition: '+shot['id'])
        current=np.unique(observed[begin:stop])
        if not np.isin(current,pts).all():
            raise ValueError('Observation PTS outside registered shot exposures: '+shot['id'])
        coverage=dict(native_pts_step=int(step),source_exposures=len(pts),
            frames_with_observations=len(current),frames_without_observations=len(pts)-len(current),
            observation_rows=stop-begin,observations_interpolated=False)
        plan.append((pts,coverage));row_stop=stop
    if row_stop!=len(observed):raise ValueError('Observations exist outside registered shot rows')
    return plan


def main(args):
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError('Preserve previous audit: '+str(args.output))
    manifest_path = args.run_root/'joint_manifest.json'
    prepared_manifest = read(manifest_path)
    native_library = getattr(args,'native_library',None)
    manifest = _manifest_for_library(prepared_manifest,native_library)
    if digest(manifest['native_library']) != manifest['native_library_sha256']:
        raise ValueError('Native runtime changed')
    protected = {str(manifest_path):digest(manifest_path),
                 manifest['native_library']:manifest['native_library_sha256']}
    states = []
    for name,path in (('baseline',args.baseline),('candidate',args.candidate)):
        protected[str(path)] = digest(path)
        with np.load(path,allow_pickle=False) as loaded:
            state = {key:loaded[key] for key in loaded.files}
        _verify_native_library_binding(manifest,state)
        states.append((name,path,state))
    core = NativeCore(manifest['native_library'])
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'], manifest['first_control_time'])
    # Sparse measurements never remove a registered source exposure.
    with np.load(args.run_root/'point_observations.npz', allow_pickle=False) as points:
        exposures = source_exposure_plan(manifest, points['pts'])
    metric_path = Path(__file__).parents[2]/'model_constraints/ankara_2026_metric_scale_assumed_v1.json'
    scale = read(metric_path)['converted_span_xy_m'][0]
    result = dict(schema='rinkspace.native_camera_audit.v1',
        run_root=str(args.run_root), manifest_sha256=protected[str(manifest_path)],
        prepared_native_library=prepared_manifest['native_library'],
        prepared_native_library_sha256=prepared_manifest['native_library_sha256'],
        selected_native_library=manifest['native_library'],
        native_library_override_explicit=native_library is not None,
        native_library=manifest['native_library'], native_library_sha256=manifest['native_library_sha256'],
        metric_source=str(metric_path), metres_per_unit=scale, metric_status='conditional assumed scale',
        state_mutated=False, optimizer_started=False, cross_cut_derivatives_computed=False,
        measured_motion_limits_available=False,
        source_exposure_basis='manifest shot first_pts..last_pts at native_pts_step',
        source_exposures=sum(c['source_exposures'] for _,c in exposures),
        frames_with_observations=sum(c['frames_with_observations'] for _,c in exposures),
        frames_without_observations=sum(c['frames_without_observations'] for _,c in exposures),
        observations_interpolated=False,states={})
    stored = {}
    started = time.monotonic()
    for name,path,state in states:
        if state['controls'].shape != (timeline.count, 3):
            raise ValueError('State/shot motion layout mismatch')
        binding = json.loads(str(state['joint_numeric_dependencies_json']))
        if binding['control_dt'] != manifest['control_dt'] or binding['first_control_time'] != manifest['first_control_time']:
            raise ValueError('State motion clock differs')
        for recorded, current in zip(binding['shots'], manifest['shots'], strict=True):
            for key in ('id', 'camera_index', 'control_begin', 'control_stop', 'reference_pts',
                        'time_base_num', 'time_base_den', 'first_control_time', 'first_pts', 'last_pts'):
                if recorded[key] != current[key]:
                    raise ValueError('State shot binding differs: '+key)
        shots = {}
        for shot, (pts, coverage) in zip(manifest['shots'], exposures, strict=True):
            seconds = pts*shot['time_base_num']/shot['time_base_den']
            times = ((pts-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den']
                     + timeline.offset(shot))
            rig = state['rig'][shot['camera_index']]
            support = [timeline.first+(shot['control_begin']+1)*timeline.dt,
                       timeline.first+(shot['control_stop']-2)*timeline.dt]
            halo = [times[0]-abs(rig[37])/2, times[-1]+abs(rig[37])/2]
            if halo[0] < support[0]-1e-10 or halo[1] > support[1]+1e-10:
                raise ValueError('Readout crosses this independent shot support: '+shot['id'])
            motion, arrays, camera = inspect_shot_motion(core, rig, state['controls'],
                timeline.first, timeline.dt, times, seconds, scale)
            optics, lens_arrays = inspect_lens_forward(core, rig, state['controls'],
                timeline.first, timeline.dt, times, seconds, camera, 1920, 1080)
            shots[shot['id']] = dict(camera_index=shot['camera_index'], source_coverage=coverage, motion=motion, optics=optics)
            for key, value in dict(pts=pts, **arrays, **lens_arrays).items():
                stored[name+'_'+shot['id']+'_'+key] = value
            print(json.dumps(dict(state=name, shot=shot['id'], lens=optics,
                peak_pan_speed=motion['channels']['pan']['speed']['continuous_peak'],
                peak_pan_acceleration=motion['channels']['pan']['acceleration']['continuous_peak'],
                elapsed_seconds=time.monotonic()-started)), flush=True)
        result['states'][name] = dict(path=str(path), sha256=protected[str(path)],shots=shots,
            native_library_provenance_sha256=str(state['native_library_provenance_sha256']))
    result['elapsed_seconds'] = time.monotonic()-started
    for path,expected in protected.items():
        if digest(path)!=expected:raise ValueError('Audit input changed: '+path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.npz'), **stored)
    print(str(args.output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'baseline', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--native-library',type=Path,
        help='Explicit actual DLL; its SHA256 must match both frozen states')
    main(parser.parse_args())
