"""Source-clock result catalog and read-only display geometry."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.interpolate import BSpline
from rinkspace.storage import AssetStore
from rinkspace.metric_scale import estimate_uniform_scale
from .open_boot_topology import shell_triangles as open_shell_triangles


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _positive_scalar(value,label):
    value=float(np.asarray(value))
    if not np.isfinite(value) or value<=0:raise ValueError('Invalid '+label)
    return value


def _array(value,shape,label):
    value=np.asarray(value)
    if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid '+label)
    return value


def _rotations(value,shape,label):
    value=_array(value,shape,label)
    if not np.allclose(np.swapaxes(value,-1,-2)@value,np.eye(3),rtol=0,atol=1e-6) or not np.allclose(np.linalg.det(value),1.,rtol=0,atol=1e-6):
        raise ValueError('Invalid proper rotation: '+label)
    return value


def _faces(value,vertices,label):
    value=np.asarray(value)
    if value.dtype.kind not in 'iu' or value.ndim not in (2,3) or value.shape[-1]!=3 or (value.ndim==3 and value.shape[0]!=2) or np.any(value<0) or np.any(value>=vertices):
        raise ValueError('Invalid '+label)
    return value.astype(np.int32,copy=False)


def footwear_state(saved,n,ratio):
    """Validate/cache saved footwear, without evaluating or deforming it.

    All world points and distance metrics share the candidate's length unit.
    Fixed topology, rotations, angles and deformation coefficients do not scale.
    No shoe surface or diagnostics are invented for historical results.
    """
    out={}
    for key in ('footwear_binding_applied_mask','body_refinement_window_mask'):
        if key in saved:
            mask=np.asarray(saved[key])
            if mask.shape!=(n,) or mask.dtype.kind!='b':raise ValueError('Invalid '+key)
            out[key]=mask
    for source,target in [('boot_surface_world_mid','boot_surface'),('footbed_world_mid','footbed'),('cavity_world_mid','cavity'),('anchors_world_mid','anchors')]:
        if source not in saved:continue
        values=np.asarray(saved[source])
        if values.ndim!=4 or values.shape[:2]!=(n,2) or values.shape[-1]!=3 or not np.isfinite(values).all():raise ValueError('Invalid '+source)
        out[target]=values*ratio
    for target,source in [('boot_faces','boot_surface_faces'),('cavity_faces','cavity_faces')]:
        points='boot_surface' if target=='boot_faces' else 'cavity'
        if points in out:
            out[target]=_faces(saved[source],out[points].shape[2],source)
    if 'boot_surface' in out:
        if out['boot_surface'].shape[2]!=14:raise ValueError('Current boot surface schema requires the original 14 vertex ordering')
        np.testing.assert_array_equal(saved['boot_surface_world_mid'],saved['gear_world_mid'][:,:,:14])
        if 'boot_surface_region_ids' in saved:
            ids=np.asarray(saved['boot_surface_region_ids']);faces=out['boot_faces']
            if ids.shape!=faces.shape[:-1] or ids.dtype.kind not in 'iu':raise ValueError('Invalid boot face region IDs')
            out['boot_region_ids']=ids
        if 'collar_boundary' in saved:
            ids=np.asarray(saved['collar_boundary'])
            if ids.ndim not in (1,2) or (ids.ndim==2 and ids.shape[0]!=2) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=14):raise ValueError('Invalid collar boundary')
            out['collar_boundary']=ids
    if 'axes_world_mid' in saved:
        values=np.asarray(saved['axes_world_mid'])
        if values.ndim!=5 or values.shape[:2]!=(n,2) or values.shape[-2:]!=(2,3) or not np.isfinite(values).all():raise ValueError('Invalid axes_world_mid')
        out['axes']=values*ratio
    for points,source in [('anchors','anchor_labels'),('axes','axis_labels')]:
        if points in out:
            labels=np.asarray(saved[source])
            if labels.shape!=(out[points].shape[2],) or labels.dtype.kind not in 'US':raise ValueError('Invalid '+source)
            out[source]=labels
    if 'effective_ankle_relative_rotation_mid' in saved:
        out['effective_ankle_relative_rotation']=_rotations(saved['effective_ankle_relative_rotation_mid'],(n,2,3,3),'effective shank-foot relative rotation')
    if 'upper_deformation_mid' in saved:
        values=np.asarray(saved['upper_deformation_mid'])
        if values.ndim!=3 or values.shape[:2]!=(n,2) or not np.isfinite(values).all():raise ValueError('Invalid upper deformation coefficients')
        out['upper_deformation']=values
    for key in ('cavity_penetration_mid','heel_drift_mid','forefoot_error_mid','blade_tilt_rad_mid'):
        if key in saved:out[key.removesuffix('_mid')]=_array(saved[key],(n,2),key)*(1. if key=='blade_tilt_rad_mid' else ratio)
    return out


class ResultCatalog:
    """A comparison shares one source clock and one frozen display scene.

    source_run/scene_root/camera_snapshot accept Data-relative or absolute paths;
    metric_binding accepts project-relative or absolute paths by default; set
    metric_binding_base to "data" for Data-relative paths. A variant's result
    is source-run-relative or absolute, with its own metres_per_scene_unit (the
    saved scalar wins unless an equal explicit value is declared in the config).
    Omitting the new scene/scale options preserves the historical binding.
    """
    def __init__(self,project=None,data_root=None,config=None):
        self.project=Path(project or Path(__file__).resolve().parents[3]).resolve()
        self.assets=AssetStore(self.project,data_root=data_root)
        self.data_root=Path(self.assets.data)
        self.config_path=Path(config or self.project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_s01.json')
        self.config=json.loads(self.config_path.read_text(encoding='utf8'))
        metric_base=self.config.get('metric_binding_base','project')
        if metric_base not in ('project','data'):raise ValueError('Invalid metric_binding_base: '+str(metric_base))
        self.root=self.data_root/self.config['source_run']
        self.packet=load_npz(self.root/'input.npz')
        self.pts=self.packet['pts'].astype(np.int64)
        if len(self.pts)<2 or np.any(np.diff(self.pts)<=0):raise ValueError('Source exposures must be ordered and unique')
        self.time_base=(1,12800);self.seconds=self.pts/12800.
        self.variants=self.config['variants']
        # The source pose packet and a comparison scene can use different units.
        # scene_root is relative to Data; metric_binding uses its declared base.
        self.scene_root=self.data_root/self.config.get('scene_root',self.config['source_run'])
        self.camera_path=self.data_root/self.config['camera_snapshot'] if 'camera_snapshot' in self.config else self.scene_root/'camera_snapshot.npz'
        self.camera=load_npz(self.camera_path)
        camera_scale=estimate_uniform_scale(2*self.camera['board'][:2])
        metric_root=self.data_root if metric_base=='data' else self.project
        self.metric_path=metric_root/self.config['metric_binding'] if self.config.get('metric_binding') else None
        self.metric=json.loads(self.metric_path.read_text(encoding='utf8')) if self.metric_path else {}
        self.scale=_positive_scalar(self.config.get('display_metres_per_scene_unit',self.metric.get('metres_per_scene_unit',camera_scale)),'display scale')
        np.testing.assert_allclose(self.scale,camera_scale,rtol=1e-12)
        if self.metric:np.testing.assert_allclose(self.scale,self.metric['metres_per_scene_unit'],rtol=1e-12)
        self.legacy_scale=_positive_scalar(self.packet['metres_per_scene_unit'],'source packet scale')
        self.unit_ratio=self.scale/self.legacy_scale
        self.model_path=self.assets.resolve(self.project/self.config['model_asset'],must_exist=True)
        self.cache_root=self.data_root/'cache/results3d'/self.config['shot']
        self.session_root=self.data_root/'workbench/results3d';self.session_root.mkdir(parents=True,exist_ok=True)
        self._common_fingerprint=None

    def variant(self,name):return next(q for q in self.variants if q['id']==name)

    def result_path(self,variant):return self.root/variant['result']

    def frame_for_seconds(self,seconds):
        return int(np.argmin(abs(self.seconds-float(seconds))))

    def common_fingerprint(self):
        if self._common_fingerprint is None:
            paths=[self.root/'input.npz',self.root/'compact_smplx.npz',self.model_path,Path(__file__),Path(__file__).with_name('prepare.py'),self.project/'calibration/program_tracking/body_tracking/rebuild_body_geometry.py',self.project/'calibration/program_tracking/body_tracking/footwear_geometry.py']
            if hasattr(self,'scene_root'):paths.extend([self.scene_root/'review_scene.npz',self.scene_root/'review_scene.json',self.camera_path])
            if getattr(self,'metric_path',None):paths.append(self.metric_path)
            self._common_fingerprint=dict(files={str(p):digest(p) for p in paths},display_metres_per_scene_unit=self.scale,time_semantics='common source mid-exposure',schema='rinkspace.viewer-mesh-cache.v2')
        return self._common_fingerprint

    def source_identity(self):
        """Read the existing warning table and bind its exact bytes to the cache.

        This preserves warning values; it does not qualify or infer source data.
        Historical tables without an explicit PTS field retain that format.
        """
        from io import BytesIO
        path=self.project/'events/ankara_2026/programmes/okada_mei_sp/body_s01_joint_candidate_20261029.npz'
        raw=path.read_bytes()
        with np.load(BytesIO(raw),allow_pickle=False) as z:
            if 'pts' in z:np.testing.assert_array_equal(z['pts'],self.pts)
            flags=z['source_identity_suspect'].copy()
        if flags.shape!=self.pts.shape:
            raise ValueError('Source identity warnings must match the displayed exposures')
        return flags,hashlib.sha256(raw).hexdigest()

    def cache_path(self,variant):
        v=self.variant(variant) if isinstance(variant,str) else variant
        _,warning_identity=self.source_identity()
        identity=dict(common=self.common_fingerprint(),result=digest(self.result_path(v)),kind=v['kind'],source_identity_sha256=warning_identity)
        if 'metres_per_scene_unit' in v:identity['candidate_metres_per_scene_unit']=v['metres_per_scene_unit']
        if 'shell_topology' in v:identity['shell_topology']=v['shell_topology']
        if 'base' in v:identity['base']=digest(self.root/v['base'])
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
        return self.cache_root/(v['id']+'_'+key),identity

    def parameters(self,variant):
        v=self.variant(variant) if isinstance(variant,str) else variant
        z=load_npz(self.result_path(v));n=len(self.pts)
        pts=z['pts'] if 'pts' in z else z['source_pts']
        np.testing.assert_array_equal(pts,self.pts)
        if 'source_pts' in z:np.testing.assert_array_equal(z['source_pts'],self.pts)
        p=_array(z.get('P_world_mid',z.get('P_world_m')),(n,3),'root positions')
        q=_rotations(z.get('Q_world_mid',z.get('Q_world')),(n,3,3),'world root Q')
        candidate_scale=_positive_scalar(v.get('metres_per_scene_unit',z.get('metres_per_scene_unit',self.legacy_scale)),'candidate scale')
        if 'metres_per_scene_unit' in v and 'metres_per_scene_unit' in z:
            np.testing.assert_allclose(candidate_scale,z['metres_per_scene_unit'],rtol=1e-12)
        common=dict(P=p,Q=q,saved=z,variant=v,metres_per_scene_unit=candidate_scale,unit_ratio=self.scale/candidate_scale,length_ratio=1.)
        if v['kind']=='fullbody':
            local=_rotations(z['local_R_mid'],(n,21,3,3),'effective local rotations')
            solver_scale=_positive_scalar(z['solver_metres_per_scene_unit'],'solver scale')
            length_ratio=_positive_scalar(z['body_length_ratio'],'body length ratio')
            np.testing.assert_allclose(length_ratio,candidate_scale/solver_scale,rtol=1e-12)
            # The compact model comes from the old source packet. Its lengths
            # must be reexpressed once, before applying the candidate root Q/P.
            np.testing.assert_allclose(solver_scale,self.legacy_scale,rtol=1e-12)
            for key,shape in [('native22_world_mid',(n,22,3)),('COCO_world_mid',(n,17,3)),('boot_world_mid',(n,2,14,3)),('blade_world_mid',(n,2,34,3)),('gear_world_mid',(n,2,48,3)),('G_world_mid',(n,3))]:_array(z[key],shape,key)
            mask=np.asarray(z['window_mask'])
            if mask.shape!=(n,) or mask.dtype.kind!='b':raise ValueError('Invalid optimization window mask')
            lower=np.asarray(z['source_lower_body_inferred'])
            if lower.shape!=(n,) or lower.dtype.kind!='b':raise ValueError('Invalid lower-body source flags')
            np.testing.assert_array_equal(z['gear_world_mid'][:,:,:14],z['boot_world_mid'])
            np.testing.assert_array_equal(z['gear_world_mid'][:,:,14:],z['blade_world_mid'])
            common.update(delta=None,leg=None,geometry=None,local_R=local,length_ratio=length_ratio)
            common['footwear']=footwear_state(z,n,common['unit_ratio'])
            return common
        if v['kind']=='joint':
            base=load_npz(self.root/v['base'])
            np.testing.assert_array_equal(base['pts'],self.pts,err_msg='Base pose/source clock mismatch')
            delta=base['delta_local_rad']
            basis=BSpline.design_matrix(self.packet['fit_time_seconds'],z['correction_knots'],3,extrapolate=True)
            leg=basis@z['leg_controls'];geometry=z['shared_geometry']
        else:delta=z['delta_local_rad'];leg=np.zeros((len(self.pts),14));geometry=None
        common.update(delta=delta,leg=leg,geometry=geometry,local_R=None)
        return common

    def scene_geometry(self):
        root=getattr(self,'scene_root',self.root)
        packet=load_npz(root/'review_scene.npz');info=json.loads((root/'review_scene.json').read_text(encoding='utf8'));xyz=packet['xyz']*self.scale
        bottom=next(c for c in info['cuts'] if c['id']=='board/continuous/bottom')
        top=next(c for c in info['cuts'] if c['id']=='board/continuous/top')
        return dict(xyz=xyz,cuts=info['cuts'],bottom=xyz[bottom['begin']:bottom['end']],top=xyz[top['begin']:top['end']])


class CachedSequence:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'metadata.json').read_text(encoding='utf8'))
        self.vertices=np.load(self.path/'vertices.npy',mmap_mode='r')
        self.faces=np.load(self.path/'faces.npy')
        self.state=load_npz(self.path/'state.npz')
        if len(self.vertices)!=len(self.state['pts']):raise ValueError('Mesh/source clock mismatch')


def shell_triangles(points,topology='open_collar'):
    """Display faces only; explicit legacy hull remains available for comparison.

    Call once in a rigid reference frame. Dynamic upper geometry must provide
    its own fixed faces, never infer a new hull or ear clipping per exposure.
    """
    if topology=='open_collar':
        return open_shell_triangles(points)
    if topology!='closed_hull':raise ValueError('Unknown boot topology: '+str(topology))
    from scipy.spatial import ConvexHull
    p=np.asarray(points,float);h=ConvexHull(p);faces=h.simplices.copy()
    for i,face in enumerate(faces):
        normal=np.cross(p[face[1]]-p[face[0]],p[face[2]]-p[face[0]])
        if normal@h.equations[i,:3]<0:faces[i,[1,2]]=faces[i,[2,1]]
    return faces.astype(np.int32)


def ribbon_triangles(stations):
    return np.asarray([f for i in range(stations-1) for f in [(i,i+1,stations+i),(i+1,stations+i+1,stations+i)]],np.int32)


def simplify_curve(points,tolerance=.0006):
    """Display-only 3D RDP, retaining endpoints and all bends above 0.6 mm."""
    p=np.asarray(points,float)
    if len(p)<=2:return p
    vector=p[-1]-p[0];den=float(vector@vector)
    alpha=np.clip((p-p[0])@vector/max(den,1e-30),0,1)
    distance=np.linalg.norm(p-(p[0]+alpha[:,None]*vector),axis=1);k=int(np.argmax(distance))
    if distance[k]<=tolerance:return p[[0,-1]]
    return np.r_[simplify_curve(p[:k+1],tolerance)[:-1],simplify_curve(p[k:],tolerance)]
