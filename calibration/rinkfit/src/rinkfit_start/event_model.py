"""Compile preserved event seeds to the final shared physical parameter graph.

This is an initializer, never an acceptance test. All 182017 original rows are
retained. The legacy p superellipse and separated print paths are explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class SeedBoundary:
    """Independent dense initialization chart; C++ uses analytic quadrature."""
    def __init__(self,width,rx,ry,p):
        self.width,self.rx,self.ry,self.p=width,rx,ry,p
        self.long=1-2*rx;self.short=width-2*ry
        a=np.linspace(0,np.pi/2,65537); c=np.cos(a);s=np.sin(a)
        den=c**p+s**p;d=den**(-1/p)
        v=(c**(p-1)*s-s**(p-1)*c)/den
        speed=np.hypot(rx*d*(-s+c*v),ry*d*(c+s*v))
        ar=np.r_[0,cumulative_trapezoid(speed,a)]
        self.arc=float(ar[-1]);self.a_to_s=PchipInterpolator(a,ar);self.s_to_a=PchipInterpolator(ar,a)
        self.lengths=np.array([self.long,self.arc,self.short,self.arc,self.long,self.arc,self.short,self.arc])
        self.starts=np.r_[0,np.cumsum(self.lengths)];self.perimeter=float(self.starts[-1])
    def inverse(self,world):
        w=np.asarray(world);x,y=w[:,0],w[:,1];hx=.5;hy=self.width/2;rx=self.rx;ry=self.ry
        candidates=[]
        for seg in range(8):
            if seg==0: ss=np.clip(x+hx-rx,0,self.long)
            elif seg==2:ss=np.clip(y+hy-ry,0,self.short)
            elif seg==4:ss=np.clip(hx-rx-x,0,self.long)
            elif seg==6:ss=np.clip(hy-ry-y,0,self.short)
            else:
                k=(seg-1)//2;cx=[hx-rx,hx-rx,-hx+rx,-hx+rx][k];cy=[-hy+ry,hy-ry,hy-ry,-hy+ry][k]
                xx=(x-cx)*([1,1,-1,-1][k]);yy=(y-cy)*([-1,1,1,-1][k])
                a=np.clip(np.arctan2(np.maximum(yy,0)/ry,np.maximum(xx,0)/rx),0,np.pi/2)
                ar=self.a_to_s(a);ss=self.arc-ar if k in (0,2) else ar
            stations=ss+self.starts[seg];p=self.point(stations)
            candidates.append((np.sum((p[:,:2]-w[:,:2])**2,axis=1),stations))
        dist=np.stack([v[0] for v in candidates]);ix=dist.argmin(axis=0)
        station=np.stack([v[1] for v in candidates])[ix,np.arange(len(w))]
        return station,np.sqrt(dist[ix,np.arange(len(w))])
    def point(self,station,height=None):
        s=np.asarray(station)%self.perimeter;seg=np.minimum(np.searchsorted(self.starts,s,side='right')-1,7)
        local=s-self.starts[seg];out=np.zeros((s.size,3));hx=.5;hy=self.width/2;rx=self.rx;ry=self.ry
        for i in range(8):
            mask=seg==i;v=local[mask]
            if not np.any(mask):continue
            if i==0:out[mask,:2]=np.c_[-hx+rx+v,np.full(len(v),-hy)]
            elif i==2:out[mask,:2]=np.c_[np.full(len(v),hx),-hy+ry+v]
            elif i==4:out[mask,:2]=np.c_[hx-rx-v,np.full(len(v),hy)]
            elif i==6:out[mask,:2]=np.c_[np.full(len(v),-hx),hy-ry-v]
            else:
                k=(i-1)//2;a=self.s_to_a(np.clip(self.arc-v if k in (0,2) else v,0,self.arc));c=np.cos(a);sn=np.sin(a);d=(c**self.p+sn**self.p)**(-1/self.p)
                out[mask,0]=[hx-rx,hx-rx,-hx+rx,-hx+rx][k]+[1,1,-1,-1][k]*rx*c*d
                out[mask,1]=[-hy+ry,hy-ry,hy-ry,-hy+ry][k]+[-1,1,1,-1][k]*ry*sn*d
        if height is not None:out[:,2]=height
        return out


def compile_model(packet_dir, *, radius_max=.30, optical_candidate='legacy_lowdim_restricted_domain',
                  initialize_rows=True):
    root=Path(packet_dir);seed=json.loads((root/'seeds.json').read_text(encoding='utf-8'))
    boot=json.loads((root/'bootstrap_state.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    tables=json.loads((root/'string_tables.json').read_text(encoding='utf-8'))
    arrays=np.load(root/'world_seeds.npz',allow_pickle=False);cam=np.load(root/'camera_seeds.npz',allow_pickle=False)
    scene=boot['scene_seed'];head=boot['shared_head'];opt=boot['optics'];controls=boot['camera_controls'];basis=controls['basis'];n=basis['n_controls']
    boundary=SeedBoundary(scene['width'],scene['radius_x'],scene['radius_y'],scene['exponent'])
    blocks=[]
    def block(name,values,lower,upper,scales,constant=False):
        values=np.asarray(values,dtype=float).tolist();m=len(values)
        lower=np.broadcast_to(lower,(m,)).astype(float).tolist();upper=np.broadcast_to(upper,(m,)).astype(float).tolist();scales=np.broadcast_to(scales,(m,)).astype(float).tolist()
        if any(a<lo or a>hi for a,lo,hi in zip(values,lower,upper)):raise ValueError('seed outside declared physical domain: '+name)
        ix=len(blocks);blocks.append(dict(name=name,values=values,lower=lower,upper=upper,scales=scales,constant=constant));return ix
    g=[scene['width'],scene['radius_x'],scene['radius_y'],scene['exponent'],scene['board_height'],0,0,0,0,0,0,*head['P'],*head['d'],head['a'],head['pupil_linear'],head['pupil_quadratic'],*opt['principal_point'],opt['eta'],*opt['principal_linear'],opt['k1'][0],opt['k1'][1],opt['k2'][0],opt['k2'][1],opt['p1'],opt['p2'],boot['rolling_shutter']['signed_readout_seconds']]
    if len(g)!=32:raise AssertionError('global32 layout')
    if optical_candidate=='undistorted':g[25:31]=[0.]*6
    lower=[.1,.005,.005,1.05,.001]+[-1.]*3+[-.5,-.5,-1.]+[-2.,-2.,.001]+[-.2]*3+[-.2,-.5,-.5]+[-1920.,-1080.,-.15,-2000.,-2000.,-100.,-200.,-200.,-400.,-1.,-1.,-.02]
    upper=[.95,.49,.45,4.,.1]+[1.]*3+[.5,.5,1.]+[2.,2.,.5]+[.2]*3+[.2,.5,.5]+[3840.,2160.,.15,2000.,2000.,100.,200.,200.,400.,1.,1.,.02]
    scales=[.5,.1,.1,2.,.02]+[.1]*6+[.5,.5,.05]+[.01]*6+[1000.,1000.,.01,100.,100.,1.,1.,10.,10.,.01,.01,.02]
    block('global32',g,lower,upper,scales)
    dep=int(controls['angle_dependent_index']);free=np.delete(np.arange(n),dep)
    values=np.zeros((n,3));values[free,0]=controls['alpha_free_controls'];values[free,1]=controls['beta_free_controls'];values[:,2]=controls['log_f_controls']
    first=basis['first_control_time'];dt=basis['dt'];cell=int(np.floor(-first/dt));u=(-first/dt)-cell
    refinds=np.arange(cell-1,cell+3);refw=np.array([(1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3])/6
    # Dependent angle entries are unused storage; explicit elimination is in
    # the forward model, not an approximate penalty on a coefficient.
    values[dep,:2]=0
    for i,v in enumerate(values):block(f'control/{i:04d}',v,[-20.,-20.,np.log(100.)],[20.,20.,np.log(1e6)],[.1,.1,1.])
    blocks[1+dep]['fixed_indices']=[0,1]
    variants=seed['variants'];charts={};chart_errors={}
    for key,v in variants.items():
        if v['descriptor']['kind'].startswith('ad_'):
            w=arrays[v['world_array']];s,err=boundary.inverse(w);charts[key]=np.c_[s,w[:,2]];chart_errors[key]=float(np.max(err))
    ads={}
    for ad in sorted({int(v['descriptor']['advertisement_id']) for v in variants.values() if 'advertisement_id' in v['descriptor']}):
        members=[key for key,v in variants.items() if v['descriptor'].get('advertisement_id')==ad]
        corners=[key for key in members if variants[key]['descriptor']['kind']=='ad_corner']
        ch=np.concatenate([charts[key] for key in corners or members]);s=ch[:,0]
        if np.ptp(s)>boundary.perimeter/2:
            for key in members:charts[key][:,0]=np.where(charts[key][:,0]<boundary.perimeter/2,charts[key][:,0]+boundary.perimeter,charts[key][:,0])
            ch=np.concatenate([charts[key] for key in corners or members])
        if np.ptp(ch[:,0])<1e-8:
            ch=np.concatenate([charts[key] for key in members])
        s0,hi=float(ch[:,0].min()),float(ch[:,0].max());bot,top=float(ch[:,1].min()),float(ch[:,1].max())
        if hi-s0<1e-8 or top-bot<1e-8:raise ValueError('degenerate ad installation '+str(ad))
        ads[ad]=(block(f'ad/{ad}',[s0,hi-s0,bot,top],[-10.,1e-6,0.,0.],[10.,3.,.1,.1],[1.,.1,.02,.02]),np.array([s0,hi-s0,bot,top]))
    pathdefs={};circle_groups={}
    for key,v in variants.items():
        desc=v['descriptor'];geom=v['geometry'];kind=desc['kind'];paths=[];base=dict(variant_id=key,provenance=json.dumps(v,ensure_ascii=False),physical_point=v['observation_semantic']=='physical_point',closed=False)
        if kind=='board_edge':
            part_segments={'lower_long_side':0,'lower_right':1,'right_end':2,'upper_right':3,'upper_long_side':4,'upper_left':5,'left_end':6,'lower_left':7}
            parts=desc.get('association_policy',{}).get('candidate_parts')
            if not parts:raise ValueError('board edge has no sourced finite association union '+key)
            paths=[dict(base,id=key+'/'+part,kind=3,boundary_segment=part_segments[part],fixed_height_fraction=1. if desc['role']=='top' else 0.) for part in parts]
        elif kind.startswith('ice_'):
            if geom['kind']=='circle':
                feature=desc['curve_id'].split(':')[0]
                if feature not in circle_groups:
                    rings=[(k,w) for k,w in variants.items() if w['geometry']['kind']=='circle' and w['descriptor'].get('curve_id','').split(':')[0]==feature]
                    rings.sort(key=lambda kv:float(kv[1]['geometry']['radius']));radii=[float(w['geometry']['radius']) for _,w in rings]
                    q=block('ice/circle/'+feature,[*geom['center_xyz'][:2],*radii],[-.5,-.5]+[1e-5]*len(radii),[.5,.5]+[.4]*len(radii),[.5,.5]+[.05]*len(radii));circle_groups[feature]=(q,{k:2+i for i,(k,w) in enumerate(rings)})
                q,rad=circle_groups[feature];paths=[dict(base,id=key+'/circle',kind=1,geometry_block=q,radius_index=rad[key],start_angle=geom['angle_interval'][0],end_angle=geom['angle_interval'][1],closed=True)]
            elif geom['kind']=='segment':
                q=block('ice/line/'+key,[*geom['a_xyz'][:2],*geom['b_xyz'][:2]],[-.5,-.5,-.5,-.5],[.5,.5,.5,.5],[.5]*4)
                paths=[dict(base,id=key+'/line',kind=0,geometry_block=q)]
            else:raise ValueError('unsupported source ice primitive '+key)
        elif kind.startswith('ad_'):
            ad=int(desc['advertisement_id']);adblock,install=ads[ad];ch=charts[key];uv=np.c_[(ch[:,0]-install[0])/install[1],(ch[:,1]-install[2])/(install[3]-install[2])]
            if kind in ('ad_edge','ad_corner'):
                # Boundary coordinates are exact installation incidences.
                role=desc['role']
                if role in ('top_edge','bottom_edge'):
                    vv=1. if role=='top_edge' else 0.;points=np.array([[0.,vv],[1.,vv]])
                elif role in ('left_vertical','right_vertical'):
                    xx=float(np.mean(uv[:,0]));xx=float(round(xx));points=np.array([[xx,0.],[xx,1.]])
                elif kind=='ad_corner':points=np.rint(uv[:1])
                else:
                    # Older descriptors use left_edge/right_edge.
                    xx=float(round(np.mean(uv[:,0])));points=np.array([[xx,0.],[xx,1.]])
                paths=[dict(base,id=key+'/installation',kind=2,advertisement_block=adblock,points=points.tolist())]
            else:
                q=block('print/'+key,[1.,0.,0.,0.,1.,0.],[-3.,-2.,-1.,-2.,-3.,-1.],[3.,2.,1.,2.,3.,1.],[1.]*6)
                off=geom.get('path_offsets',[0,len(uv)])
                # Closure is source topology, not proximity of sampled ends.
                # Old exporters could omit the final repeated vertex while
                # retaining a genuine closed contour in source metadata.
                closed=geom.get('path_closed',[False]*(len(off)-1))
                if len(closed)!=len(off)-1:raise ValueError('Explicit path topology size mismatch '+key)
                topology_provenance=geom.get('topology_provenance',dict(method='explicit_path_closed' if 'path_closed'in geom else 'topology_not_exported_open_support_only',endpoint_distance_used=False))
                ids=desc.get('curve_topology',{}).get('path_ids',[f'path{j}' for j in range(len(off)-1)])
                for j in range(len(off)-1):
                    points=uv[off[j]:off[j+1]]
                    if len(points)<2:raise ValueError('invalid finite print path '+key)
                    isclosed=bool(closed[j]);
                    if isclosed and np.linalg.norm(points[0]-points[-1])>1e-10:points=np.r_[points,points[:1]]
                    paths.append(dict(base,id=key+'/'+ids[j],kind=2,geometry_block=q,advertisement_block=adblock,points=points.tolist(),closed=isclosed,topology_provenance=topology_provenance))
        else:raise ValueError('unsupported source material '+kind)
        pathdefs[key]=paths
    materials=[dict(label=label,paths=[p for key in next(m for m in seed['materials'] if m['label']==label)['variants'] for p in pathdefs[key]]) for label in tables['material_labels']]
    nseed=np.asarray(head['n']);axis=np.eye(3)[np.argmin(np.abs(nseed))];u=np.cross(nseed,axis);u/=np.linalg.norm(u);vv=np.cross(nseed,u)
    camera=dict(R0_seed=np.asarray(head['R0']).reshape(-1).tolist(),n_seed=nseed.tolist(),b_seed=head['b'],n_tangent1=u.tolist(),n_tangent2=vv.tolist(),f_ref=opt['f_ref'],radius_max=radius_max,dt=dt,first_control_time=first,n_controls=n,control_block_begin=1,dependent_control=dep,reference_indices=refinds.tolist(),reference_weights=refw.tolist(),tau_bound=.02)
    identity=manifest['immutable_source_hashes']['observations.npz']
    model=dict(schema='rinkfit.event_model.v1.1',model_id='shared_superellipse_polar_fixed_head_'+optical_candidate,source_identity=identity,source_hashes=manifest['immutable_source_hashes'],blocks=blocks,camera=camera,materials=materials,
               provenance=dict(seed_sha256=_sha(root/'seeds.json'),bootstrap_sha256=_sha(root/'bootstrap_state.json'),shape_seed_role='conditional initializer; shared geometry blocks are jointly adjustable',old_p_preserved=scene['exponent'],chart_boundary_reconstruction_max_L=max(chart_errors.values()),finite_path_union_preserved=True,static_print_basis='per-contour affine6 in shared ad chart, fixed finite polygon topology; explicit low-complexity candidate',optical_candidate=optical_candidate,distortion_radius_role='frozen candidate domain; not Sony hardware specification',initial_observation_row_time_role='only station initializer, final solver uses explicit RS equalities and independent true roots'),
               global_names=['width','rx','ry','p','board_height','R0_rx','R0_ry','R0_rz','n_tangent1','n_tangent2','b_phase','P_x','P_y','P_z','d_x','d_y','d_z','axis_spacing','pupil_linear','pupil_quadratic','cx','cy','eta','cx_z','cy_z','k1','k1_z','k2','k2_z','p1','p2','tau'])
    _write_json(root/'model.json',model)
    if initialize_rows:_initialize_stations(root,model,boundary,cam,manifest)
    _write_json(root/'model_manifest.json',dict(schema='rinkfit.event_model_manifest.v1',model_file='model.json',sha256=_sha(root/'model.json'),materials=len(materials),paths=sum(len(m['paths']) for m in materials),parameter_blocks=len(blocks),parameters=sum(len(b['values']) for b in blocks),model_id=model['model_id'],source_identity=identity,initialization_only=True))
    return model


def _initialize_stations(root,model,boundary,cam,manifest,output_dir=None):
    output=Path(output_dir) if output_dir is not None else root
    n=manifest['observations'];uv=np.fromfile(root/'uv.f64','<f8').reshape(n,2);frames=np.fromfile(root/'frame.i32','<i4');mats=np.fromfile(root/'material.u32','<u4')
    variant_ids=sorted(json.loads((root/'seeds.json').read_text(encoding='utf-8'))['variants'])
    sem=root/'semantic_packet'/'variant_index.u32';
    if not sem.exists():sem=root/'semantic_packet'/'variant.u32'
    if not sem.exists():sem=root.parent/'semantic_packet'/'variant_index.u32'
    variants=np.fromfile(sem,'<u4') if sem.exists() else None
    if variants is not None and len(variants)!=n:raise ValueError('semantic variant row count')
    station=np.zeros(n);branch=np.zeros(n,dtype='<u4');distance=np.full(n,np.inf)
    def world_samples(path,ss):
        q=np.asarray(model['blocks'][path['geometry_block']]['values']) if path.get('geometry_block',-1)>=0 else None
        if path['kind']==0:return np.c_[(1-ss)*q[0]+ss*q[2],(1-ss)*q[1]+ss*q[3],np.zeros(len(ss))]
        if path['kind']==1:
            a=path['start_angle']+ss*(path['end_angle']-path['start_angle']);r=q[path.get('radius_index',2)];return np.c_[q[0]+r*np.cos(a),q[1]+r*np.sin(a),np.zeros(len(ss))]
        if path['kind']==3:
            segment=path.get('boundary_segment',-1);s=ss*boundary.perimeter if segment<0 else boundary.starts[segment]+ss*boundary.lengths[segment]
            return boundary.point(s,np.full(len(ss),model['blocks'][0]['values'][4]*path['fixed_height_fraction']))
        points=np.asarray(path['points']);a=ss*(len(points)-1);i=np.minimum(np.floor(a).astype(int),max(0,len(points)-2));u=a-i
        uv=points[i]*(1-u[:,None])+points[np.minimum(i+1,len(points)-1)]*u[:,None]
        ad=np.asarray(model['blocks'][path['advertisement_block']]['values']);return boundary.point(ad[0]+ad[1]*uv[:,0],ad[2]+(ad[3]-ad[2])*uv[:,1])
    for mi,mat in enumerate(model['materials']):
        rowsmat=np.flatnonzero(mats==mi)
        for pi,path in enumerate(mat['paths']):
            rows=rowsmat
            if variants is not None:rows=rows[variants[rows]==variant_ids.index(path['variant_id'])]
            if not len(rows):continue
            count=1 if path.get('physical_point') else (max(33,min(1025,len(path.get('points',[]))*2)) if path['kind']==2 else (513 if path['kind']==3 else 257))
            ss=np.linspace(0,1,count);w=world_samples(path,ss)
            for frame in np.unique(frames[rows]):
                rr=rows[frames[rows]==frame];i=int(frame)-1;R=cam['R_world_to_camera'][i];C=cam['center_world'][i];xc=(w-C)@R.T;xy=xc[:,:2]/xc[:,2:3];r2=(xy*xy).sum(axis=1);k1,k2,p1,p2=cam['brown_k1_k2_p1_p2'][i];x,y=xy.T;d=1+k1*r2+k2*r2*r2
                pred=np.c_[x*d+2*p1*x*y+p2*(r2+2*x*x),y*d+p1*(r2+2*y*y)+2*p2*x*y]*cam['focal_px'][i]+cam['principal_xy'][i]
                if count==1:
                    dd=np.sum((uv[rr]-pred[0])**2,axis=1);ssbest=np.zeros(len(rr))
                else:
                    A=pred[:-1];D=np.diff(pred,axis=0);den=(D*D).sum(axis=1);delta=uv[rr,None,:]-A[None,:,:];f=np.clip(np.sum(delta*D[None,:,:],axis=2)/np.maximum(den,1e-30),0,1);d2=np.sum((delta-f[:,:,None]*D[None,:,:])**2,axis=2);invalid=(xc[:-1,2]<=0)|(xc[1:,2]<=0)|(~np.isfinite(D).all(axis=1));d2[:,invalid]=np.inf;j=d2.argmin(axis=1);dd=d2[np.arange(len(rr)),j];ssbest=(j+f[np.arange(len(rr)),j])/(count-1)
                good=dd<distance[rr];ids=rr[good];distance[ids]=dd[good];station[ids]=ssbest[good];branch[ids]=pi
    if not np.isfinite(distance).all():raise ValueError('some raw rows have no legal initializer path: '+str(np.count_nonzero(~np.isfinite(distance))))
    station.astype('<f8').tofile(output/'station.f64');branch.tofile(output/'path.u32')
    _write_json(output/'station_initialization.json',dict(schema='rinkfit.station_initializer.v1',rows=n,all_rows_preserved=True,method='all native frames; same finite path segment projection under declared candidate camera; initializer only',physical_root_acceptance=False,semantics_variant_binding_used=variants is not None,approximate_native_distance_px=dict(max=float(np.sqrt(distance).max()),median=float(np.median(np.sqrt(distance)))),station_sha256=_sha(output/'station.f64'),path_sha256=_sha(output/'path.u32')))


def compile_independent_candidates(packet_dir, registration_candidates=None,initialize_rows=True):
    """Actual independent installation and all-time registered head recovery.

    These are complete physical states with no per-frame pose bridge. Their
    native errors determine usefulness; names do not assert distinct basins.
    """
    import copy
    from scipy.optimize import least_squares,lsq_linear
    from scipy.spatial.transform import Rotation
    from scipy.sparse import lil_matrix
    from .bootstrap import _native_controls
    from .camera import UniformCubicBasis
    root=Path(packet_dir);base=json.loads((root/'model.json').read_text(encoding='utf-8'));manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    seed=json.loads((root/'seeds.json').read_text(encoding='utf-8'));world=np.load(root/'world_seeds.npz',allow_pickle=False)
    gf=np.asarray(base['blocks'][0]['values']);boundary=SeedBoundary(*gf[:4]);n=base['camera']['n_controls'];times=np.arange(1248)*.02
    cb=base['camera'];basis=UniformCubicBasis(cb['first_control_time'],cb['dt'],n)
    def rot(axis,a):return Rotation.from_rotvec(np.asarray(axis)*a).as_matrix()
    def save(name,m,ab,logf,ledger):
        out=root/'model_candidates'/name;out.mkdir(parents=True,exist_ok=True);cam=m['camera'];g=np.asarray(m['blocks'][0]['values']);full=_native_controls(basis,times,np.c_[ab,logf]);dep=cam['dependent_control'];full[dep,:2]=0
        for j in range(n):m['blocks'][cam['control_block_begin']+j]['values']=full[j].tolist()
        m['model_id']=name;m['provenance'].update(ledger);m['provenance']['candidate_is_fitted_or_accepted']=False
        # Complete native camera table is a derived initializer convenience;
        # it is never a parameter stream and never enters final likelihood.
        R0=np.asarray(cam['R0_seed']).reshape(3,3);nn=np.asarray(cam['n_seed']);bb=np.asarray(cam['b_seed']);R=np.array([rot(bb,b)@R0@rot(nn,a)for a,b in ab]);z=logf-np.log(cam['f_ref']);ell=g[18]*z+g[19]*z*z;D=np.tile(g[14:17],(1248,1));D[:,2]+=ell
        h=np.cross(nn,np.einsum('nji,j->ni',R,bb));h/=np.linalg.norm(h,axis=1)[:,None];C=g[11:14]+np.einsum('nji,nj->ni',R,D)+g[17]*h
        derived=dict(R_world_to_camera=R,center_world=C,focal_px=np.c_[np.exp(logf+g[22]/2),np.exp(logf-g[22]/2)],principal_xy=np.c_[g[20]+g[23]*z,g[21]+g[24]*z],brown_k1_k2_p1_p2=np.c_[g[25]+g[26]*z,g[27]+g[28]*z,np.full(1248,g[29]),np.full(1248,g[30])])
        _write_json(out/'model.json',m);_write_json(out/'candidate_ledger.json',dict(ledger,model_sha256=_sha(out/'model.json'),all_frames=1248,all_obligations=182017,temporary_pose_parameters_absent=True,source_identity=m['source_identity']))
        if initialize_rows:_initialize_stations(root,m,boundary,derived,manifest,out)
        return str(out/'model.json')
    outputs=[]
    # Installation geometry and reference focal are chosen independently of
    # old camera poses. Only the shared sourced static material seed is reused.
    mats=np.fromfile(root/'material.u32','<u4');frames=np.fromfile(root/'frame.i32','<i4');uv=np.fromfile(root/'uv.f64','<f8').reshape(-1,2)
    centroids=[]
    for mat in base['materials']:
        keys=next(m for m in seed['materials'] if m['label']==mat['label'])['variants'];centroids.append(np.mean(np.concatenate([world[seed['variants'][k]['world_array']]for k in keys]),axis=0))
    centroids=np.asarray(centroids);targets=np.zeros((1248,3))
    for i in range(1248):targets[i]=np.mean(centroids[mats[frames==i+1]],axis=0)
    for name,height,focal in [('installation_low_f2500',.03,2500.),('installation_high_f6000',.06,6000.)]:
        m=copy.deepcopy(base);g=np.asarray(m['blocks'][0]['values']);P=np.array([0.,-g[0]/2-.005,height]);direction=targets-P;alpha=np.unwrap(np.arctan2(direction[:,0],direction[:,1]));beta=np.arctan2(-direction[:,2],np.hypot(direction[:,0],direction[:,1]));off=np.array([alpha[0],beta[0]]);ab=np.c_[alpha,beta]-off
        Rbase=np.array([[1.,0,0],[0,0,-1],[0,1,0]]);R0=rot([1,0,0],off[1])@Rbase@rot([0,0,1],off[0]);m['camera'].update(R0_seed=R0.reshape(-1).tolist(),n_seed=[0.,0.,1.],b_seed=[1.,0.,0.],n_tangent1=[1.,0.,0.],n_tangent2=[0.,1.,0.],f_ref=focal,radius_max=2.)
        g[5:11]=0;g[11:14]=P;g[14:20]=0;g[20:22]=[959.5,539.5];g[22:31]=0;g[31]=0;m['blocks'][0]['values']=g.tolist()
        outputs.append(save(name,m,ab,np.full(1248,np.log(focal)),dict(initializer_family='independent_near_board_installation_reference_focal_search',camera_seed_used=False,installation_working_hypotheses_L=dict(pivot=P.tolist(),distance_outside_near_long_board=.005),aim_initialization='all 1248 frame label-support centroids; no point-ID correspondence or added measurement',metric_scale_verified=False)))
    if registration_candidates is not None:
        rows=[r for r in json.loads(Path(registration_candidates).read_text(encoding='utf-8')) if r['status']=='initializer_ready'];k=len(rows)
        if k<3:raise ValueError('registered fixed-head seed has fewer than 3 supported frames')
        Rtarget=np.asarray([r['R']for r in rows]);Rseed=Rtarget[0];ns=np.array([0.,0.,1.]);bs=np.array([1.,0.,0.]);u=np.array([1.,0,0]);v=np.array([0.,1.,0]);q=Rseed@ns;bs-=q*(bs@q);bs/=np.linalg.norm(bs)
        def headchart(x):
            R0=Rotation.from_rotvec(x[:3]).as_matrix()@Rseed;nn=ns+x[3]*u+x[4]*v;nn/=np.linalg.norm(nn);q=R0@nn;bb=bs-q*(bs@q);bb/=np.linalg.norm(bb);bb=np.cos(x[5])*bb+np.sin(x[5])*np.cross(q,bb);return R0,nn,bb
        a0=np.zeros((k,2));w0=Rseed.T@bs
        for i,ri in enumerate(Rtarget):
            wi=ri.T@bs;a0[i,0]=-np.arctan2(ns@np.cross(w0,wi),w0@wi);a0[i,1]=Rotation.from_matrix(ri@(Rseed@rot(ns,a0[i,0])).T).as_rotvec()@bs
        a0=np.unwrap(a0,axis=0);x0=np.r_[np.zeros(6),a0[1:].reshape(-1)]
        def residual(x):
            R0,nn,bb=headchart(x);aa=np.r_[np.zeros((1,2)),x[6:].reshape(-1,2)];pred=np.array([rot(bb,b)@R0@rot(nn,a)for a,b in aa]);return Rotation.from_matrix(pred@np.swapaxes(Rtarget,1,2)).as_rotvec().reshape(-1)
        sparsity=lil_matrix((3*k,len(x0)),dtype=int);sparsity[:,:6]=1
        for i in range(1,k):sparsity[3*i:3*i+3,6+2*(i-1):6+2*i]=1
        fit=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        R0,nn,bb=headchart(fit.x);angles=np.r_[np.zeros((1,2)),fit.x[6:].reshape(-1,2)];tt=np.array([(r['frame']-1)*.02 for r in rows]);ab=np.c_[np.interp(times,tt,angles[:,0]),np.interp(times,tt,angles[:,1])]
        K=np.asarray([r['K']for r in rows]);logf_src=np.mean(np.log(np.stack([K[:,0,0],K[:,1,1]],axis=1)),axis=1);logf=np.interp(times,tt,logf_src);fref=float(np.exp(np.mean(logf)));z=logf_src-np.log(fref)
        R=np.array([rot(bb,b)@R0@rot(nn,a)for a,b in angles]);h=np.cross(nn,np.einsum('nji,j->ni',R,bb));h/=np.linalg.norm(h,axis=1)[:,None];axis=np.swapaxes(R,1,2)[:,:,2]
        A=np.concatenate([np.tile(np.eye(3),(k,1,1)),np.swapaxes(R,1,2),h[:,:,None],axis[:,:,None]*z[:,None,None],axis[:,:,None]*z[:,None,None]**2],axis=2).reshape(3*k,9);C=np.asarray([r['C']for r in rows]);lo=np.array([-2.,-2.,.001,-.2,-.2,-.2,-.2,-.5,-.5]);hi=np.array([2.,2.,.5,.2,.2,.2,.2,.5,.5]);centerfit=lsq_linear(A,C.reshape(-1),bounds=(lo,hi),tol=1e-10,max_iter=200)
        m=copy.deepcopy(base);g=np.asarray(m['blocks'][0]['values']);g[5:11]=0;g[11:20]=centerfit.x;g[20:22]=np.median(K[:,:2,2],axis=0);g[22:31]=0;g[31]=0;m['blocks'][0]['values']=g.tolist();axis0=np.eye(3)[np.argmin(np.abs(nn))];tu=np.cross(nn,axis0);tu/=np.linalg.norm(tu);tv=np.cross(nn,tu)
        m['camera'].update(R0_seed=R0.reshape(-1).tolist(),n_seed=nn.tolist(),b_seed=bb.tolist(),n_tangent1=tu.tolist(),n_tangent2=tv.tolist(),f_ref=fref,radius_max=2.)
        outputs.append(save('registered_alltime_exact_head',m,ab,logf,dict(initializer_family='original_curve_registration_unknown_Q_conditional_upgrade_exact_head_recovery',camera_seed_used=False,registration_sha256=_sha(registration_candidates),supported_pose_candidates=k,pose_candidate_frame_range=[int(rows[0]['frame']),int(rows[-1]['frame'])],rotation_recovery=dict(success=bool(fit.success),rms_rad=float(np.sqrt(np.mean(fit.fun**2))),max_rad=float(np.max(np.abs(fit.fun))),nfev=int(fit.nfev)),center_recovery=dict(success=bool(centerfit.success),rms_L=float(np.sqrt(np.mean((A@centerfit.x-C.reshape(-1))**2))),rank=int(np.linalg.matrix_rank(A)),bound_active=centerfit.active_mask.tolist()),uncovered_time_initialization='nearest supported angle/focal boundary extrapolation; explicit weak initializer',K_upgrade_not_accepted=True)))
    allpaths=sorted(str(p)for p in (root/'model_candidates').glob('*/model.json'))
    _write_json(root/'model_candidates'/'manifest.json',dict(schema='rinkfit.independent_candidates.v1',complete_physical_models=allpaths,distinct_basins_certified=False,all_sources_retained=True))
    return outputs


def initialize_existing_candidate(packet_dir,model_path):
    from scipy.spatial.transform import Rotation
    root=Path(packet_dir);path=Path(model_path);model=json.loads(path.read_text(encoding='utf-8'));manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    camera=model['camera'];g=np.asarray(model['blocks'][0]['values']);n=camera['n_controls'];full=np.asarray([model['blocks'][camera['control_block_begin']+i]['values']for i in range(n)]).copy();dep=camera['dependent_control'];refi=camera['reference_indices'];refw=camera['reference_weights'];full[dep,:2]=0
    pivot=refw[refi.index(dep)]
    for i,w in zip(refi,refw):
        if i!=dep:full[dep,:2]-=w*full[i,:2]/pivot
    times=np.arange(1248)*.02;x=(times-camera['first_control_time'])/camera['dt'];i=np.floor(x).astype(int);u=x-i;w=np.c_[(1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3]/6;tr=np.sum(w[:,:,None]*full[i[:,None]+np.arange(-1,3)],axis=1)
    R0=Rotation.from_rotvec(g[5:8]).as_matrix()@np.asarray(camera['R0_seed']).reshape(3,3);nn=np.asarray(camera['n_seed'])+g[8]*np.asarray(camera['n_tangent1'])+g[9]*np.asarray(camera['n_tangent2']);nn/=np.linalg.norm(nn);q=R0@nn;bb=np.asarray(camera['b_seed']);bb=bb-q*(bb@q);bb/=np.linalg.norm(bb);bb=np.cos(g[10])*bb+np.sin(g[10])*np.cross(q,bb)
    R=Rotation.from_rotvec(tr[:,1,None]*bb).as_matrix()@R0@Rotation.from_rotvec(tr[:,0,None]*nn).as_matrix();z=tr[:,2]-np.log(camera['f_ref']);ell=g[18]*z+g[19]*z*z;D=np.tile(g[14:17],(1248,1));D[:,2]+=ell;h=np.cross(nn,np.einsum('nji,j->ni',R,bb));h/=np.linalg.norm(h,axis=1)[:,None];C=g[11:14]+np.einsum('nji,nj->ni',R,D)+g[17]*h
    derived=dict(R_world_to_camera=R,center_world=C,focal_px=np.c_[np.exp(tr[:,2]+g[22]/2),np.exp(tr[:,2]-g[22]/2)],principal_xy=np.c_[g[20]+g[23]*z,g[21]+g[24]*z],brown_k1_k2_p1_p2=np.c_[g[25]+g[26]*z,g[27]+g[28]*z,np.full(1248,g[29]),np.full(1248,g[30])])
    _initialize_stations(root,model,SeedBoundary(*g[:4]),derived,manifest,path.parent)
    return str(path.parent/'station_initialization.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet-dir',required=True);p.add_argument('--radius-max',type=float,default=.30);p.add_argument('--optical-candidate',choices=['legacy_lowdim_restricted_domain','undistorted'],default='legacy_lowdim_restricted_domain');p.add_argument('--no-stations',action='store_true');p.add_argument('--independent-candidates',action='store_true');p.add_argument('--registration-candidates');p.add_argument('--initialize-existing-model');a=p.parse_args()
    if a.initialize_existing_model:
        print(initialize_existing_candidate(a.packet_dir,a.initialize_existing_model));return
    if a.independent_candidates:
        print(json.dumps(compile_independent_candidates(a.packet_dir,a.registration_candidates,initialize_rows=not a.no_stations),ensure_ascii=False));return
    m=compile_model(a.packet_dir,radius_max=a.radius_max,optical_candidate=a.optical_candidate,initialize_rows=not a.no_stations);print(json.dumps(dict(model_id=m['model_id'],blocks=len(m['blocks']),materials=len(m['materials'])),ensure_ascii=False))
if __name__=='__main__':main()
