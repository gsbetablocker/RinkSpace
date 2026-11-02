"""Source-clock result catalog and read-only display geometry."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.interpolate import BSpline
from rinkspace.storage import AssetStore
from rinkspace.metric_scale import estimate_uniform_scale


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class ResultCatalog:
    def __init__(self,project=None,data_root=None,config=None):
        self.project=Path(project or Path(__file__).resolve().parents[3]).resolve()
        self.assets=AssetStore(self.project,data_root=data_root)
        self.data_root=Path(self.assets.data)
        self.config_path=Path(config or self.project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_s01.json')
        self.config=json.loads(self.config_path.read_text(encoding='utf8'))
        self.root=self.data_root/self.config['source_run']
        self.packet=load_npz(self.root/'input.npz')
        self.pts=self.packet['pts'].astype(np.int64)
        if len(self.pts)<2 or np.any(np.diff(self.pts)<=0):raise ValueError('Source exposures must be ordered and unique')
        self.time_base=(1,12800);self.seconds=self.pts/12800.
        self.variants=self.config['variants']
        self.metric=json.loads((self.project/self.config['metric_binding']).read_text(encoding='utf8'))
        self.camera=load_npz(self.root/'camera_snapshot.npz')
        self.scale=estimate_uniform_scale(2*self.camera['board'][:2])
        np.testing.assert_allclose(self.scale,self.metric['metres_per_scene_unit'],rtol=1e-12)
        self.legacy_scale=float(self.packet['metres_per_scene_unit'])
        self.unit_ratio=self.scale/self.legacy_scale
        self.model_path=self.assets.resolve(self.project/self.config['model_asset'],must_exist=True)
        self.cache_root=self.data_root/'cache/results3d'/self.config['shot']
        self.session_root=self.data_root/'workbench/results3d';self.session_root.mkdir(parents=True,exist_ok=True)
        self._common_fingerprint=None

    def variant(self,name):return next(q for q in self.variants if q['id']==name)

    def frame_for_seconds(self,seconds):
        return int(np.argmin(abs(self.seconds-float(seconds))))

    def common_fingerprint(self):
        if self._common_fingerprint is None:
            paths=[self.root/'input.npz',self.root/'compact_smplx.npz',self.model_path,Path(__file__),Path(__file__).with_name('prepare.py'),self.project/'calibration/program_tracking/body_tracking/rebuild_body_geometry.py',self.project/'calibration/program_tracking/body_tracking/footwear_geometry.py']
            self._common_fingerprint=dict(files={str(p):digest(p) for p in paths},display_metres_per_scene_unit=self.scale,time_semantics='common source mid-exposure',schema='rinkspace.viewer-mesh-cache.v1')
        return self._common_fingerprint

    def cache_path(self,variant):
        v=self.variant(variant) if isinstance(variant,str) else variant
        identity=dict(common=self.common_fingerprint(),result=digest(self.root/v['result']),kind=v['kind'])
        if 'base' in v:identity['base']=digest(self.root/v['base'])
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
        return self.cache_root/(v['id']+'_'+key),identity

    def parameters(self,variant):
        v=self.variant(variant) if isinstance(variant,str) else variant
        z=load_npz(self.root/v['result']);np.testing.assert_array_equal(z['pts'],self.pts)
        p=z.get('P_world_mid',z.get('P_world_m'));q=z['Q_world']
        if v['kind']=='joint':
            base=load_npz(self.root/v['base']);delta=base['delta_local_rad']
            basis=BSpline.design_matrix(self.packet['fit_time_seconds'],z['correction_knots'],3,extrapolate=True)
            leg=basis@z['leg_controls'];geometry=z['shared_geometry']
        else:delta=z['delta_local_rad'];leg=np.zeros((len(self.pts),14));geometry=None
        return dict(P=p,Q=q,delta=delta,leg=leg,geometry=geometry,saved=z,variant=v)

    def scene_geometry(self):
        packet=load_npz(self.root/'review_scene.npz');info=json.loads((self.root/'review_scene.json').read_text(encoding='utf8'));xyz=packet['xyz']*self.scale
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


def shell_triangles(points):
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
