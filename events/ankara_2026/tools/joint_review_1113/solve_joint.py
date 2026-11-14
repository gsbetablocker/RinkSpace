"""GPU refit of two Ankara programmes on one shared geometry."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from time import monotonic
from datetime import datetime, timezone
import json
import os
import sys

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='2'
os.environ.setdefault('CUDA_PATH','D:/Program Files/CUDA')
PROJECT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT/'src'))

import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, kron
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import ParameterState
from rinkspace.domain.motion_chart import UniformMotionChart, cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.storage import configured_data_root

ROOT=configured_data_root(PROJECT)/'runs/ankara_joint_1113'
INPUT=ROOT/'prepared_joint'
MOTION_UNIT=np.array([.0015,.0015,.004])
# Per physical camera, with S14 and S22 retaining their independent saved
# wide-angle calibration branches. All old per-exposure deviations are kept.
WIDE_FIELDS={26:(4.,12.),27:(4.,12.),29:(6.,15.),30:(6.,15.),39:(.032,.075)}
OPTICAL_FIELDS={26:(4.,10.),27:(4.,10.),29:(6.,12.),30:(6.,12.)}
ICE_KEYS={
    'hockey_carrier_position/3':(.0012,.0033),'hockey_carrier_slope/3':(.0015,.0040),
    'hockey_carrier_position/4':(.0012,.0033),'hockey_carrier_slope/4':(.0015,.0033),
    'faceoff_x/3':(.0018,.004),'faceoff_y/3':(.0018,.004),'hockey_circle_radius/4':(.0018,.004),
    'lane_width_at_F/0':(.0014,.0034),'lane_width_at_F/1':(.0014,.0034),
    'lane_width_at_R/0':(.0014,.0034),'lane_width_at_R/1':(.0014,.0034),
}
AD_NAMES=(4,9,11,13,14,17,18,19,24,26)


def finite_json(value):
    if isinstance(value,dict):return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(finite_json(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def atomic_arrays(path,**data):
    temp=path.with_name(path.stem+'.writing.npz')
    np.savez_compressed(temp,**data)
    temp.replace(path)


class Journal:
    def __init__(self,path):
        self.stream=path.open('x',encoding='utf8',buffering=1)
    def __call__(self,event,**details):
        rec=dict(time=datetime.now(timezone.utc).isoformat(),event=event,**details)
        self.stream.write(json.dumps(finite_json(rec),separators=(',',':'),allow_nan=False)+'\n')
        if event in {'start','prepared','finish','failure'}:
            print(event,details.get('variant'),details.get('rows'),details.get('reason'),flush=True)
        if event=='trial':
            s=details['summary']
            print(details['variant'],int(s['trial_count']),s['reason'],round(s['old_cost'],3),
                  round(s['trial_cost'],3),round(details['timings']['wall_ms']),flush=True)


def load():
    m=json.loads((INPUT/'setup.json').read_text(encoding='utf-8-sig'))
    core=NativeCore(m['core'])
    with np.load(INPUT/'baseline.npz',allow_pickle=False) as z:
        b={k:z[k] for k in z.files}
    series={}
    for k in m['shots']:
        with np.load(INPUT/(k+'.npz'),allow_pickle=False) as z:
            a={n:z[n] for n in z.files}
        np.testing.assert_array_equal(a['pts'],b[k+'_pts'][a['idx']])
        series[k]=a
    return m,core,b,series


def physical_group(name,meta):
    if name in ('jin_s14','jin_s22'):return name
    return meta['program']+'_camera_'+str(meta['camera_index'])


def optics_spec(shots):
    return {g:(WIDE_FIELDS if g in ('jin_s14','jin_s22') else OPTICAL_FIELDS)
            for g in sorted({physical_group(name,v) for name,v in shots.items()})}


def scene_spec(state):
    coords=[]
    k=next(k for k in state.blocks if k.endswith('/ice_tolerance'))
    block=state.blocks[k]
    for name,(scale,half) in ICE_KEYS.items():
        coords.append((k,block['values_meaning'].index(name),scale,half,name))
    for n in AD_NAMES:
        k=f'installation/Ad{n}/shared_placement'
        coords.extend(((k,0,.00050,.0017,f'Ad{n}_along_board'),(k,1,.00036,.0011,f'Ad{n}_height')))
    # Share the existing print chart between Ad4 and Ad17. No ad/single-shot override.
    k=next(k for k in state.blocks if k.startswith('print_shape/print_template/Ad4/'))
    coords.extend(((k,0,.0015,.004,'Ad4_Ad17_print_u'),(k,1,.0018,.005,'Ad4_Ad17_print_v')))
    return coords


def build_motion(b,series,shots):
    blocks=[];control=[];ranges={};cursor=0
    for name,meta in shots.items():
        c=b[name+'_controls'];t=b[name+'_times']
        n=len(c);first=meta['first'];dt=meta['dt']
        if not np.isclose(dt,.02):raise ValueError('Joint knot spacing differs')
        train=np.unique(series[name]['idx'][series[name]['fold']==0])
        stride=14 if name in ('jin_s13','jin_s14','jin_s22') else 19 if meta['program']=='jin' else 29
        chart=UniformMotionChart.create(n,first,dt,stride,t[0])
        base=chart.refinement
        ix,w=cubic_samples(n,first,dt,t[train])
        hits=np.einsum('ni,nik->nk',w,base[ix])
        active=np.flatnonzero(np.count_nonzero(abs(hits)>.06,axis=0)>=3)
        if not len(active):
            block=csr_matrix((3*n,0))
        else:
            d=base[:,active]
            grids=(t[[0,-1]]-first)/dt
            if np.max(abs(grids-np.rint(grids)))<1.e-8:
                d=continue_exposure_halos(np.repeat(d[:,None],3,axis=1),first,dt,t[[0,-1]])[:,0,:]
            block=kron(csr_matrix(d),csr_matrix(np.diag(MOTION_UNIT)),format='csr')
        blocks.append(block)
        control.append(c)
        ranges[name]=(cursor,cursor+n)
        cursor+=n
    return block_diag(blocks,format='csr'),np.vstack(control),ranges


def get_bound_rigs(b,series,shots):
    rigs=[];tags=[];by_shot={}
    for name,meta in shots.items():
        a=series[name];i=a['idx'][a['fold']==0]
        table,index=np.unique(b[name+'_rig'][i],axis=0,return_inverse=True)
        by_shot[name]=index+len(rigs)
        rigs.extend(table)
        tags.extend([physical_group(name,meta)]*len(table))
    return np.asarray(rigs),tags,by_shot


def build_measurements(b,series,shots,ranges,by_shot):
    chunks=defaultdict(list)
    for name,meta in shots.items():
        a=series[name];use=a['fold']==0
        for k in ('uv','path','pts','family','station','weight'):
            chunks[k].append(a[k][use])
        # Translation preserves each knot interval and each original row clock.
        shift=ranges[name][0]*.02-meta['first']
        chunks['times'].append(b[name+'_times'][a['idx'][use]]+shift)
        chunks['camera_ids'].append(by_shot[name])
        chunks['programme'].append(np.repeat(meta['program'],np.count_nonzero(use)))
    return {k:np.concatenate(v) for k,v in chunks.items()}


def scene_columns(state,q,start,coords):
    d=np.zeros((len(state.values),q))
    lo=np.full(len(state.values),-np.inf)
    hi=np.full(len(state.values),np.inf)
    for j,(k,i,scale,half,name) in enumerate(coords,start):
        block=state.blocks[k]
        if i not in block['independent_local_indices']:raise ValueError('Frozen scene coordinate: '+name)
        i=block['offset']+i
        d[i,j]=scale
        lo[i]=state.values[i]-half
        hi[i]=state.values[i]+half
    return d,lo,hi


def rig_columns(rigs,tags,q,start,spec):
    d=np.zeros((len(rigs)*31,q))
    lo=np.full(rigs.shape,-np.inf)
    hi=np.full(rigs.shape,np.inf)
    coords={}
    j=start
    for tag,fields in spec.items():
        for key,(scale,half) in fields.items():
            coords[tag,key]=j
            j+=1
    for i,tag in enumerate(tags):
        for k,(scale,half) in spec[tag].items():
            chart=k-11 if k<38 else k-12
            d[31*i+chart,coords[tag,k]]=scale
            lo[i,k]=rigs[i,k]-half
            hi[i,k]=rigs[i,k]+half
    return d,lo.ravel(),hi.ravel()


def row_loss(errors,family,programme,weight,huber):
    # Use identical native 2D whitening and rank-normalized group Huber.
    u=errors*weight
    rho=np.where(u<=huber,.5*u*u,huber*(u-.5*huber))
    layer=np.where(np.char.startswith(family,'Ad'),'ad',np.where(np.char.startswith(family,'board.'),'board','ice'))
    return {p:{k:float(rho[(programme==p)&(layer==k)].sum()) for k in ('board','ice','ad')}
            for p in ('jin','okada')}


def save(checkpoint,accepted,b,rigs,tags,meta,ranges,parameters):
    r=accepted['rig']
    r=np.atleast_2d(r)
    dd=r-rigs
    deltas={}
    for tag in sorted(set(tags)):
        use=np.asarray(tags)==tag
        d=np.median(dd[use],axis=0)
        if np.max(abs(dd[use]-d))>2.e-9:raise ValueError('Optics direction changed a frame-specific seed')
        deltas[tag]=d
    results={'scene_values':accepted.get('scene_values',parameters.values)}
    for k,v in meta['shots'].items():
        i,j=ranges[k]
        results[k+'_controls']=accepted['controls'][i:j]
        results[k+'_rig']=b[k+'_rig']+deltas[physical_group(k,v)]
    results['optical_groups']=np.asarray(list(deltas))
    results['optical_deltas']=np.asarray(list(deltas.values()))
    atomic_arrays(checkpoint,**results)


def solve_variant(variant,setup,core,b,series,out,log,stop_path):
    begin=monotonic()
    shots=setup['shots']
    params=ParameterState(setup['scene_layout'],np.ascontiguousarray(b['scene_values']))
    native=NativeSceneGPU(setup['scene_path'],core,parameter_state=params)
    md,control,ranges=build_motion(b,series,shots)
    rigs,tags,by_shot=get_bound_rigs(b,series,shots)
    data=build_measurements(b,series,shots,ranges,by_shot)
    fields=optics_spec(shots) if variant in ('camera','joint') else {}
    geo=scene_spec(params) if variant=='joint' else []
    additional=sum(len(a) for a in fields.values())+len(geo)
    q=md.shape[1]+additional
    if not q:raise ValueError('No retained directions')
    motion=hstack((md,csr_matrix((md.shape[0],additional))),format='csr')
    dirs,lo,hi=scene_columns(params,q,q-len(geo),geo) if geo else (None,None,None)
    out.mkdir(exist_ok=True)
    checkpoint=out/(variant+'.npz')
    mcam=dict(first_control_time=0.,control_dt=.02,image_height=1080,reference_pts=0,
              time_base_num=1,time_base_den=12800)
    n=len(data['uv'])
    log('binding',variant=variant,rows=n,retained_columns=q,rig_rows=len(rigs),
        motion_columns=md.shape[1],scene_columns=[a[-1] for a in geo],
        optical_groups=fields,checkpoint=str(checkpoint),data=str(INPUT))
    with native.resident(camera=mcam,controls=control,path_ids=data['path'],stations=data['station'],
          pts=data['pts'],observed=data['uv'],scene_directions=dirs,scene_lower=lo,scene_upper=hi,
          motion_times=data['times'],rigs=rigs,camera_ids=data['camera_ids']) as r:
        half=float(np.max(abs(rigs[:,37]))*.64+1.e-7)
        if fields:
            od,olo,ohi=rig_columns(rigs,tags,q,md.shape[1],fields)
            r.configure_camera_parameters(directions=od,max_readout=2*half,lower=olo,upper=ohi)
            del od,olo,ohi
        else:
            r.configure_projection_time_support(half_seconds=half)
        bounds=r._path_supports[r._codes].copy()
        stations=np.clip(data['station'],bounds[:,0],bounds[:,1])
        np.testing.assert_allclose(stations,data['station'],atol=1.e-12,rtol=0)
        ids=np.arange(n,dtype=np.int32)
        # One qualified original UV per statistical group, fixed family mass
        # per exposure and Huber at four image pixels before family scaling.
        weight=data['weight']*np.where(data['programme']=='jin',1.12,1.)
        white=np.zeros((n,4));white[:,0]=white[:,3]=weight
        plan=r.configure_linearization(directions=motion,local_ids=ids,
             local_scales=np.full(n,.0015),group_offsets=np.arange(n+1,dtype=np.int32),
             members=ids,ranks=np.full(n,2,np.int32),weight_offsets=np.arange(n+1,dtype=np.int32)*4,
             weights=white.ravel(),huber_delta=2.7)
        native_setup=r.configure_solver(lower=bounds[:,0],upper=bounds[:,1],device_budget_bytes=9*1024**3,
                          host_budget_bytes=12*1024**3,initial_lambda_relative=.0005,damping_metric='jacobi_image')
        error=r.evaluate_errors()
        bad=int(np.count_nonzero(~np.isfinite(error)))
        if bad:raise ValueError(f'Prepared problem has {bad} invalid initial projections')
        parts=row_loss(error,data['family'],data['programme'],weight,2.7)
        log('prepared',variant=variant,rows=n,setup_seconds=monotonic()-begin,plan=plan,
            gpu=native_setup,domain_failures=bad,objective_by_programme_layer=parts,checkpoint=str(checkpoint))
        accepted_at=None
        prev=None
        count=0
        def update(rec):
            nonlocal parts,prev,count,accepted_at
            s=rec['summary']
            if s['accepted']:
                count+=1
                accepted=r.export_state()
                save(checkpoint,accepted,b,rigs,tags,setup,ranges,params)
                accepted_at=count
                parts=row_loss(r.evaluate_errors(),data['family'],data['programme'],weight,2.7)
            log('trial',variant=variant,rows=n,retained_columns=q,objective_by_programme_layer=parts,
                checkpoint=str(checkpoint),**rec)
            if stop_path.exists():return False
            if s['accepted']:
                delta=abs(s['actual_decrease'])/max(abs(s['old_cost']),1.)
                # Stop when improvements are already far below held-out
                # differences, then compare on disjoint source intervals.
                if count>=5 and delta<7.e-5 and (prev or 1.)<2.e-4:return False
                prev=delta
            return True
        result=r.solve(relative_cost_tolerance=7.e-6,step_tolerance=1.e-5,progress=update,
                       numerical_progress=lambda rec:log('factorization_retry',variant=variant,**rec))
        if accepted_at is None:
            save(checkpoint,r.export_state(),b,rigs,tags,setup,ranges,params)
        log('finish',variant=variant,rows=n,reason=result.get('stop_reason'),
            checkpoint=str(checkpoint),seconds=monotonic()-begin,objective_by_programme_layer=parts,solver=result)
        return result


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--variants',nargs='+',choices=('motion','camera','joint'),default=['camera','joint'])
    ap.add_argument('--out',default='fit')
    a=ap.parse_args()
    target=ROOT/a.out
    target.mkdir(exist_ok=True)
    log=Journal(target/'solve.jsonl')
    log('start',variants=a.variants,device='native CUDA and cuDSS',prepared=str(INPUT))
    try:
        setup,core,b,series=load()
        for variant in a.variants:
            solve_variant(variant,setup,core,b,series,target,log,target/'stop')
            if (target/'stop').exists():break
    except BaseException as e:
        log('failure',reason=str(e))
        raise

if __name__=='__main__':main()
