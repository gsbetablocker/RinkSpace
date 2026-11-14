"""Frozen native multi-shot dispatch and an explicit static crop camera.

Both expose the existing body projection interface; no camera is optimized.
"""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .fullbody_state import camera_binding
from .rebuild_prepare import digest, load
from .source_visible_model import camera_centres


def registered_camera_binding(record_path, record, packet, shot_ids, *, native_library=None):
    """Bind the exact per-shot spline and exposure clock of a registered review."""
    from rinkspace.domain.parameters import ParameterState, SceneParameterModel
    from rinkspace.metric_scale import estimate_uniform_scale
    from .rebuild_body_fit import FixedNativeProjector

    camera = record['camera']
    geometry = record['geometry']
    for entry in (camera, geometry, geometry['structure'], geometry['native_library']):
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Registered camera or geometry changed: '+entry['path'])
    z = load(camera['path'])
    geometry_sha = geometry['value_sha256']
    if str(z['scene_values_sha256']) != geometry_sha:
        raise ValueError('Registered camera and geometry disagree')
    g = load(geometry['path'])
    if (sha256(g['scene_values'].tobytes()).hexdigest() != geometry_sha
            or str(g['layout_revision']) != geometry['layout_revision']
            or str(g['scene_file_sha256']) != geometry['structure']['sha256']):
        raise ValueError('Registered scene parameters do not match their structure')
    np.testing.assert_array_equal(packet['pts'], z['source_pts'])
    np.testing.assert_array_equal(np.unique(z['shot_ids']), np.unique(shot_ids))
    if len(z['shot_ids']) != len(set(z['shot_ids'])):
        raise ValueError('Duplicate registered shot')
    for offsets, limit in ((z['source_offsets'], len(shot_ids)), (z['control_offsets'], len(z['controls']))):
        if (offsets.shape != (len(z['shot_ids'])+1,) or offsets[0] != 0
                or offsets[-1] != limit or np.any(np.diff(offsets) <= 0)):
            raise ValueError('Invalid registered shot ranges')
    # The board dimensions here are in native scene units.
    path = Path(geometry['structure']['path'])
    scene = json.loads(path.read_text(encoding='utf-8-sig'))
    model = SceneParameterModel(scene, path, optimize_scene=True)
    if model.initial_state.layout['layout_revision'] != geometry['layout_revision']:
        raise ValueError('Registered scene layout differs from the compiled layout')
    board = ParameterState(model.initial_state.layout, g['scene_values']).apply_scene(scene)['board']
    scale = estimate_uniform_scale(2*np.asarray(board[:2]))
    solver_scale = float(packet['metres_per_scene_unit'])
    recorded = geometry['native_library']['path']
    library = native_library or recorded
    cameras, infos, poses = {}, {}, {}
    loaded_digests = {}
    for i, shot in enumerate(z['shot_ids'].tolist()):
        a, b = z['source_offsets'][i:i+2]
        c, d = z['control_offsets'][i:i+2]
        if not np.all(shot_ids[a:b] == shot):
            raise ValueError('Registered shot differs from the prepared body clock: '+shot)
        rigs = z['rig_by_source_pts'][a:b]
        if not np.all(rigs == rigs[:1]):
            raise ValueError('Expected a fixed native rig within registered shot '+shot)
        ts = z['local_exposure_seconds'][a:b]
        first, dt = float(z['first_control_time'][i]), float(z['control_dt'][i])
        if (np.any(np.diff(ts) <= 0) or not np.isfinite(ts).all()
                or not np.isfinite([first, dt]).all() or dt <= 0):
            raise ValueError('Invalid registered exposure timeline for '+shot)
        p = dict(packet, pts=z['source_pts'][a:b], rig=rigs[0], controls=z['controls'][c:d],
                 motion_times=ts, first_control_time=np.array(first), control_dt=np.array(dt))
        projector = FixedNativeProjector(p, library)
        pose = projector.core.evaluate_motion_camera(p['rig'], p['controls'], first, dt, ts)
        if pose['status'].any():
            raise ValueError('Registered camera has a source-midpoint domain error in '+shot)
        loaded = str(projector.core.library_path)
        if loaded not in loaded_digests:
            loaded_digests[loaded] = digest(loaded)
        lib_sha = loaded_digests[loaded]
        if native_library is None and lib_sha != geometry['native_library']['sha256']:
            raise ValueError('Loaded library differs from the registered camera library')
        cameras[shot] = projector
        poses[shot] = pose
        infos[shot] = dict(record=str(record_path), state_path=camera['path'],
            state_sha256=camera['sha256'], native_library=loaded, native_library_sha256=lib_sha,
            shot=dict(id=shot, camera_index=int(z['camera_indices'][i]), first_pts=int(p['pts'][0]),
                      last_pts=int(p['pts'][-1]), control_dt=dt, first_control_time=first),
            metres_per_scene_unit=scale, solver_metres_per_scene_unit=solver_scale,
            length_export_ratio=scale/solver_scale, camera_changed_during_fit=False,
            scene_values_sha256=geometry_sha, scene_binding=geometry,
            source_time_base=[1,12800], old_reference_reprojected_on_current_camera=True)
    return cameras, infos, poses


class ProgrammeCamera:
    image_height=1080
    def __init__(self,project,packet,shot_ids,*,record_path=None,native_library=None):
        self.shot_ids=np.asarray(shot_ids);self.cameras={};self.first={};infos={};poses={}
        record = {} if record_path is None else json.loads(Path(record_path).read_text(encoding='utf-8'))
        if record.get('schema') == 'rinkspace.programme_joint_scene_review.v1':
            self.cameras,infos,poses = registered_camera_binding(
                record_path, record, packet, self.shot_ids, native_library=native_library)
            self.first = {shot: int(np.flatnonzero(self.shot_ids==shot)[0]) for shot in self.cameras}
        else:
            for shot in dict.fromkeys(self.shot_ids.tolist()):
                ix=np.flatnonzero(self.shot_ids==shot);part=dict(packet,pts=packet['pts'][ix])
                camera,info,state,pose=camera_binding(project,SimpleNamespace(p=part),record_path=record_path,native_library=native_library,shot_id=shot)
                self.cameras[shot]=camera;self.first[shot]=int(ix[0]);infos[shot]=info;poses[shot]=pose
        # Only sparse S05 source factors consume this historical scalar readout.
        # Full body, exporter and dispatcher use per-exposure rig/readout below.
        representative=self.cameras.get('s05',next(iter(self.cameras.values())))
        self.p=dict(packet,rig=representative.p['rig'],controls=representative.p['controls'],first_control_time=representative.p['first_control_time'],control_dt=representative.p['control_dt'])
        self.p['motion_times']=np.empty(len(self.shot_ids))
        for shot,camera in self.cameras.items():self.p['motion_times'][self.shot_ids==shot]=camera.p['motion_times']
        self.scale=representative.scale;self.core=representative.core
        self.info=dict(next(iter(infos.values())))
        self.info.update(shot='programme',shots=infos,no_per_shot_alignment=True,source_time_base=[1,12800])
        self.midpoint={key:np.empty((len(self.shot_ids),*np.asarray(next(iter(poses.values()))[key]).shape[1:]))for key in ('rotation','centre','values','status','fx','fy','cx','cy','k1','k2','p1','p2')}
        for shot,pose in poses.items():
            mask=self.shot_ids==shot
            for key in self.midpoint:self.midpoint[key][mask]=pose[key]
    def _groups(self,frames):
        frames=np.asarray(frames,dtype=int)
        for shot in dict.fromkeys(self.shot_ids[frames].tolist()):
            ix=np.flatnonzero(self.shot_ids[frames]==shot);yield ix,self.cameras[shot],frames[ix]-self.first[shot]
    def readout_times(self,frames):
        frames=np.asarray(frames);tau=np.empty(len(frames))
        for ix,camera,local in self._groups(frames):tau[ix]=camera.p['rig'][37]
        return tau
    def camera_centres(self,frames,rows=None):
        out=np.empty((len(frames),3))
        for ix,camera,local in self._groups(frames):out[ix]=camera_centres(camera,local,None if rows is None else np.asarray(rows)[ix])
        return out
    def project(self,points,frames):
        points=np.asarray(points);out=np.empty((*points.shape[:-1],2))
        for ix,camera,local in self._groups(frames):out[ix]=camera.project(points[ix],local)
        return out
    def with_jacobian(self,points,frames):
        points=np.asarray(points);uv=np.empty((*points.shape[:-1],2));J=np.empty((*points.shape[:-1],2,3))
        for ix,camera,local in self._groups(frames):
            # FullBodyProblem configures this dispatcher's execution policy.
            # Every real shot projector must consume it, not only the S05
            # representative exposed as self.core.
            camera.core.motion_projection_threads=self.core.motion_projection_threads
            if hasattr(self,'body_execution'):camera.body_execution=self.body_execution
            uv[ix],J[ix]=camera.with_jacobian(points[ix],local)
        return uv,J

class StaticCropCamera:
    """Cam7 rectified crop pixels; world extrinsics are fixed official values."""
    def __init__(self,packet,K_input,R_w2c,C_world_m,*,image_height=1024):
        self.p=dict(packet,rig=np.zeros(43));self.K=np.asarray(K_input,float);self.R=np.asarray(R_w2c,float);self.C=np.asarray(C_world_m,float)
        if self.R.ndim==3:
            if not np.allclose(self.R,self.R[:1],rtol=0,atol=1e-12):raise ValueError('Static FS camera rotation changed')
            self.R=self.R[0]
        if self.C.shape!=(3,)or self.K.shape!=(len(packet['pts']),3,3):raise ValueError('Explicit crop camera arrays required')
        self.image_height=int(image_height);self.scale=1.
    def readout_times(self,frames):return np.zeros(len(frames))
    def camera_centres(self,frames,rows=None):return np.broadcast_to(self.C,(len(frames),3)).copy()
    def with_jacobian(self,points,frames):
        points=np.asarray(points);K=self.K[np.asarray(frames,int)];X=(points-self.C)@self.R.T;q=np.einsum('nij,nkj->nki',K,X)
        if np.any(q[...,2]<=1e-6):raise ValueError('FS point behind frozen camera')
        uv=q[...,:2]/q[...,2,None]
        D=(K[:,None,:2,:]-uv[...,None]*K[:,None,2:3,:])/q[...,2,None,None]
        return uv,D@self.R
    def project(self,points,frames):return self.with_jacobian(points,frames)[0]
