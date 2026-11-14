"""Bind the current Jin and Okada cameras to one real Ankara geometry."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
os.environ.setdefault('CUDA_PATH','D:/Program Files/CUDA')

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from rinkspace.storage import configured_data_root

DATA=configured_data_root(ROOT)
OUT=DATA/'runs/ankara_joint_1113'
H=DATA/'runs/jin_sp_20261103/audit/q38_source_1109/q71_source_continuity_1110'
import cv2
import numpy as np

from inspect_geometry import check, context, load_arrays, numeric_hash, pin, read, write
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU

STEP=256
TB=Fraction(1,12800)
W,HIMG=1920,1080
TOP='board/continuous/top'
BOTTOM='board/continuous/bottom'
JIN_FOCUS={
 's03':(39.50,40.94,41.66),'s04':(42.62,44.2,45.04,46.68,47.18,47.38),
 's06':(52.82,53.98,55.84,56.50,57.30),'s09':(72.56,74.94,77.76),
 's10':(78.28,78.5,83.52),'s12':(104.72,),'s13':(112.58,112.98,113.4),
 's14':(120.38,120.74,121.64,121.8,122.02,122.28,122.9,123.18,123.6,123.8),
 's15':(124.48,),'s16':(128.12,),'s18':(137.02,137.82),
 's19':(143.56,143.58),'s20':(144.94,),'s21':(147.44,147.66,148.12),
 's22':(149.24,153.92,155.04),'s24':(172.32,174.16,176.72,190.90,194.82)}
OKADA_FOCUS={'s05':(79.92,),'s06':(105.26,),'s08':(142.04,142.26,142.44)}
KEY_FEATURES={'A1','A2','A3','A4','B1','B2','C','goal_line_left','goal_line_right',
              'goal_crease_left','goal_crease_right','F1','F2','R1','R2',
              *(f'lane{i}' for i in range(1,5)),
              *(f'Ad{i}' for i in (1,2,4,5,6,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,26)),
              'board.top','board.bottom'}

def enable_historical_source_readers():
    """Expose the archived Q-series adapters for explicit source reconstruction."""
    folders = (
        H/'q871_C_rail_A4_1112', H/'q870_s13_lane3_1111', H/'q869_rims_and_motion_1111',
        H/'q868_material_optics_1111', H/'q867_tail_source_review_1111', H/'s14_ad_rail_rims_1111',
        H/'s14_parallel_paint_1111', H/'s14_lane_camera_diagnosis_1111', H/'q866_full_review_1111',
        H/'s14_original_ice_and_cap_1111', H/'programme_source_review_1111', H/'user_review_1111',
        H, H.parent, H.parents[2],
    )
    sys.path.extend(str(path) for path in folders if str(path) not in sys.path)
    if 'measure_source' not in sys.modules:
        # Legacy source scans import this reader by name after adding their own
        # directories. Load its repository implementation before those scans.
        from importlib.util import module_from_spec, spec_from_file_location

        helper = Path(__file__).resolve().parent/'source_helpers/measure_source.py'
        spec = spec_from_file_location('measure_source', helper)
        reader = module_from_spec(spec)
        spec.loader.exec_module(reader)
        sys.modules[spec.name] = reader


def lines(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_reader():
    enable_historical_source_readers()
    from current_871 import scan

    return scan


def family(key,native):
    p=native.paths[key]
    label=p['label'].removeprefix('ice.').split(':')[0]
    if p['layer']=='advertisement':
        return key.split('::',1)[0].rsplit('/',1)[-1]
    if key in (TOP,BOTTOM):
        return 'board.top' if key==TOP else 'board.bottom'
    if label.startswith('goal_trapezoid'):
        return 'trapezoid_left' if 'left' in label else 'trapezoid_right'
    return label

def unique_source(a, b):
    return int(round(a/b))

@dataclass
class Shot:
    name: str
    program: str
    shot: dict
    timeline: ShotMotionTimeline
    controls: np.ndarray
    rigs: np.ndarray
    source: str
    evidence: dict | None
    @property
    def id(self):
        return self.shot['id']
    @property
    def first(self):
        return float(self.shot['first_control_time'])
    @property
    def dt(self):
        return self.timeline.dt
    @property
    def pts(self):
        return np.arange(self.shot['first_pts'],self.shot['stop_pts_exclusive'],STEP)
    @property
    def times(self):
        return self.timeline.local_exposure_time(self.shot,self.pts)
    def at(self,pts):
        idx=(pts-self.pts[0])//STEP
        if np.any(idx<0) or np.any(idx>=len(self.pts)) or np.any(self.pts[idx]!=pts):
            raise ValueError((self.name,pts))
        return idx

def inputs():
    enable_historical_source_readers()
    from current_871 import Current

    a=context()
    ov=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/review_shared_q874_1113.json')
    seed=load_arrays(check(ov['geometry']['parameter_state'])['path'],('scene_values',))['scene_values']
    if numeric_hash(seed)!=ov['geometry']['scene_values_sha256']:
        raise ValueError('Shared rink state is not the saved review.')
    state=ParameterState(a['ss'].layout,seed)
    native=NativeSceneGPU(a['s14_scene']['path'],a['jc'],parameter_state=state)
    if len(native.paths)!=922 or native.unlocalized:
        raise ValueError('The shared full rink inventory changed.')
    j=Current(check(a['ji']['working_snapshot'])['path'])
    j_source=pin(j.source,j.m['source']['sha256'])
    o_source=check(ov['source']['video'])
    ok_t=ShotMotionTimeline(a['om']['shots'],a['om']['control_dt'],a['om']['first_control_time'])
    shots={}
    with np.load(check(a['ji']['numeric_bundle'])['path'],allow_pickle=False) as z:
        lookup={k:i for i,k in enumerate(z['shot_ids'].tolist())}
        for name,rec in a['ji']['shots'].items():
            s=j.by_id[name]
            ctrl,rig=j.control(name)
            if numeric_hash(ctrl)!=rec['reference_ctrl_sha'] or numeric_hash(rig)!=rec['reference_rig_sha']:
                raise ValueError('Jin saved camera source changed: '+name)
            if rec['numerical_override']:
                i=lookup[name]
                b,e=z['control_offsets'][i:i+2]
                u,v=z['pts_offsets'][i:i+2]
                np.testing.assert_array_equal(z['source_pts'][u:v],j.pts(name))
                ctrl,rig=z['controls'][b:e],z['rig_by_source_pts'][u:v]
            key='jin_'+name
            shots[key]=Shot(key,'jin',s,j.timeline,ctrl,rig,j_source['path'],check(rec['evidence']))
    for s in a['om']['shots']:
        name=s['id']
        c=a['orig']['controls'][s['control_begin']:s['control_stop']]
        rig=a['orig']['rig'][s['camera_index']]
        count=(s['stop_pts_exclusive']-s['first_pts'])//STEP
        key='okada_'+name
        shots[key]=Shot(key,'okada',s,ok_t,c,np.broadcast_to(rig,(count,len(rig))),o_source['path'],None)
    return a, native, shots, ov

def native_paths(native,names,count=129):
    path_samples={}
    for key in names:
        spec=native.paths[key]
        kind=spec['kind']
        board=native.scene['board']
        n=1025 if kind==7 else max(count,min(257,len(spec.get('vertices',[]))*2)) if kind==8 else count
        grid=np.linspace(0.,native.core.board_perimeter(board),n) if kind==7 else np.linspace(*(spec.get('support') or [0.,1.]),n)
        if kind==8:
            s,z=np.asarray(spec['vertices']).T
            points=native.core.board_points(board,np.interp(grid,spec['knots'],s),np.interp(grid,spec['knots'],z))
        else:
            points=native.core.sample_path(board,kind,spec['parameters'],spec.get('vertices',[]),grid)
        path_samples[key]=(np.asarray(points),grid)
    return path_samples

def project_groups(native, rig, controls, first, dt, groups, world):
    # Each record is one exposure/curve. Native polyline station interpolation
    # handles independent rolling-row exposure roots on the original clock.
    unique=list(dict.fromkeys(g[1] for g in groups))
    knots={key:np.linspace(0.,1.,len(world[key])) for key in unique}
    offsets=np.r_[0,np.cumsum([len(world[g[1]]) for g in groups])]
    ridx=np.asarray([g[0] for g in groups],int)
    r=rig[ridx]
    rigs,camera_id=np.unique(r,axis=0,return_inverse=True)
    mapping={key:i for i,key in enumerate(unique)}
    sizes=np.diff(offsets)
    codes=np.repeat([mapping[g[1]] for g in groups],sizes).astype(np.int32)
    with ResidentResidualProblem(native.core,board=native.scene['board'],layout=native.scene['shared_layout'],
          options=native.scene['shared_geometry_options'],rig=rigs if len(rigs)>1 else rigs[0],
          controls=controls,first_control_time=first,control_dt=dt,image_height=HIMG,codes=codes,
          stations=np.concatenate([knots[g[1]] for g in groups]),times=np.repeat([g[2] for g in groups],sizes),
          observed=np.zeros((len(codes),2)),camera_ids=np.repeat(camera_id,sizes)) as res:
        res.configure_geometry([dict(kind=4,vertices=world[k],parameters=[]) for k in unique])
        res.configure_projection_time_support(half_seconds=max(abs(rigs[:,37]))*.64)
        v=res.evaluate('cuda')
    uv=v['values'][:,3:5]
    st=v['status']
    return [(uv[a:b],st[a:b]) for a,b in zip(offsets[:-1],offsets[1:],strict=True)]

def nearest(uv,status,points):
    d=np.diff(uv,axis=0)
    l=np.sum(d*d,axis=1)
    ok=(status[:-1]==0)&(status[1:]==0)&(l>1e-9)&np.isfinite(l)
    if not ok.any():return None
    ix=np.flatnonzero(ok)
    a=uv[ix]
    v=d[ix]
    q=points[:,None,:]-a
    f=np.clip(np.einsum('nki,ki->nk',q,v)/l[ix],0.,1.)
    e=q-f[:,:,None]*v
    m=np.sum(e*e,axis=2)
    j=np.argmin(m,axis=1)
    ii=np.arange(len(points))
    seg=ix[j]; frac=f[ii,j]
    perp=np.column_stack((-d[seg,1],d[seg,0]))/np.sqrt(l[seg,None])
    edge=((seg==0)&(frac<=1e-7))|((seg==len(l)-1)&(frac>=1-1e-7))
    return seg,frac,perp,e[ii,j],edge

def group_points(g,cap=9):
    p=np.asarray(g['uv'],float)
    if len(p)>cap:
        u,s,_=np.linalg.svd(p-p.mean(0),full_matrices=False)
        order=np.argsort(u[:,0],kind='stable')
        p=p[order[np.rint(np.linspace(0,len(p)-1,cap)).astype(int)]]
    return p

def stats(errors):
    e=np.asarray(errors,float);v=e[np.isfinite(e)]
    return {'rows':len(e),'missing':len(e)-len(v),**({'rms':float(np.sqrt(np.mean(v*v))),
            'median':float(np.median(v)),'p90':float(np.quantile(v,.90)),
            'p95':float(np.quantile(v,.95))} if len(v) else {})}

def sparse_pts(shot,step=6,focus=True):
    a=shot.pts
    keep=set(a[::step].tolist())|{int(a[-1])}
    table=JIN_FOCUS if shot.program=='jin' else OKADA_FOCUS
    if focus:
        for sec in table.get(shot.id,()):
            p=round(sec*12800)
            if a[0]<=p<=a[-1]:
                keep.update(int(q) for q in p+STEP*np.arange(-8,9,4) if a[0]<=q<=a[-1])
    return np.asarray(sorted(keep),np.int64)
