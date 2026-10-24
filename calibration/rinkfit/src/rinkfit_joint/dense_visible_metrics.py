"""Streamed same-source visual diagnostics for complete current joint models.

This module never changes source eligibility, W, observations, or solver state.
Its nearest finite curve distance is a sampled visual diagnostic. Formal Query
and weighted joint station elimination remain separate, explicitly pending.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .effective_input_audit_v1 import CheckedBytes, resolve_packet
from .joint_review_camera import CurrentJointCameras
from .joint_replay import fast_native_project, render_visibility
from rinkfit_start.native_scene import NativeScene

PROCESSED=1
FINITE=2
RS_CERTIFIED=4
VISIBLE=8
FIXED=16
QUERY_PENDING=32
SAMPLING_UNRESOLVED=64
DOMAIN_FAILURE=128
PHYSICAL=1
TRAINING=2
FIT=4
INSPECTION=8
VALID_BITS=PROCESSED|FINITE|RS_CERTIFIED|VISIBLE
ARRAY_FILENAMES={'distance':'distance.f32','nearest_uv':'nearest_uv.f32','station':'station.f64','path':'path.i32','status':'status.u8','domain':'domain.u8','operator':'operator.u8'}
ARRAYS={'distance':('<f4',()),'nearest_uv':('<f4',(2,)),
        'station':('<f8',()),'path':('<i4',()),'status':('|u1',()),
        'domain':('|u1',()),'operator':('|u1',())}

def file_ref(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return {'file':str(path.resolve()),'sha256':h.hexdigest(),'byte_count':path.stat().st_size}

def _json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')

class ReviewInputs:
    """Typed arrays with one immutable resolved packet; no all-source ID strings."""
    def __init__(self, packet_file, selection_file=None, *, store=None,
                 resolved_packet=None, ancestry=None, current_input=None):
        self.store=store or CheckedBytes()
        self.packet_ref=file_ref(packet_file)
        self.current_input_ref=None;self.current_reader=None;self.current_manifest=None
        if current_input is not None:
            from .joint_review_packet import CurrentReviewPacket,packet_field
            reference=file_ref(current_input) if not isinstance(current_input,dict) else current_input
            self.current_reader=CurrentReviewPacket(packet_file,current_input=reference)
            self.current_input_ref=reference;self.current_manifest=self.current_reader.current
            self.packet=dict(self.current_reader.packet)
            for key in ('source_row_ids','source_dataset_ids','queries','allowed_paths','source_station_domains','dense_source_binding','native_ad_source_bindings','native_ad_source_binding','material_binding','source_material_binding','current_print_path_overrides','current_model_subset_v1','stroke_pairs'):
                value=packet_field(packet_file,key)
                if value is not None:self.packet[key]=value
            self.ancestry=[]
            if selection_file is not None:raise ValueError('Current review membership must come from its current manifest')
        elif resolved_packet is None:
            self.packet,self.ancestry=resolve_packet(self.packet_ref,self.store)
        else:
            self.packet=resolved_packet;self.ancestry=ancestry or []
        self.n=int(self.packet.get('row_count',len(self.packet.get('source_row_ids',[]))))
        if self.n<=0:raise ValueError('Empty review source domain')
        self.arrays={}
        base=Path(packet_file).parent
        for k in ['uv','native_pts','shot_index','material_index','kind','path','station','row','alpha','admission','fit_role']:
            self.arrays[k]=self.current_reader.load(k) if self.current_reader else self.store.array(self.packet['arrays'][k],base)[0]
            if len(self.arrays[k])!=self.n:raise ValueError('Typed row count: '+k)
        for k in ['stroke_pair_index','stroke_source_side']:
            if k in self.packet['arrays']:self.arrays[k]=self.current_reader.load(k) if self.current_reader else self.store.array(self.packet['arrays'][k],base)[0]
        self.domain=np.asarray(self.arrays['admission']!=0,dtype=np.uint8)*PHYSICAL
        self.selection_ref=None;self.selection=None
        if selection_file:
            self.selection_ref=file_ref(selection_file);s=_json(selection_file);self.selection=s
            if s['source_packet']['sha256']!=self.packet_ref['sha256'] or int(s['source_row_count'])!=self.n:
                raise ValueError('Review selection is not current-packet bound')
            training=self.store.array(s['training_eligibility']['mask'],Path(selection_file).parent)[0]!=0
            fit=self.store.array(s['fit_rows'],Path(selection_file).parent)[0].astype(np.int64,copy=False)
            inspection=self.store.array(s['inspection_rows'],Path(selection_file).parent)[0].astype(np.int64,copy=False)
            if len(training)!=self.n or np.any(training & (self.domain==0)):raise ValueError('Review training role partition')
            fm=np.zeros(self.n,bool);im=np.zeros(self.n,bool)
            for rr,mask in [(fit,fm),(inspection,im)]:
                if len(rr) and (rr.min()<0 or rr.max()>=self.n):raise ValueError('Review role row outside raw domain')
                mask[rr]=True
                if np.count_nonzero(mask)!=len(rr):raise ValueError('Duplicate review role row')
            if np.any(fm&~training) or np.any(im!=((self.domain!=0)&~fm)):raise ValueError('Review fit/inspection exact complement')
            self.domain[training]|=TRAINING;self.domain[fit]|=FIT;self.domain[inspection]|=INSPECTION
        if self.current_reader:
            membership=self.current_reader.current_source_membership_v1
            physical=np.asarray(membership['physical'])!=0;training=np.asarray(membership['training'])!=0
            fit=np.asarray(membership['fit_rows']);inspection=np.asarray(membership['inspection_rows'])
            if physical.shape!=(self.n,) or training.shape!=(self.n,) or np.any(training&~physical):raise ValueError('Current review physical/training domain')
            fm=np.zeros(self.n,bool);im=np.zeros(self.n,bool)
            for ids,mask in ((fit,fm),(inspection,im)):
                if len(ids) and (ids[0]<0 or ids[-1]>=self.n or np.any(ids[1:]<=ids[:-1])):raise ValueError('Current review membership order/range')
                mask[ids]=True
            if np.any(fm&~training) or np.any(im!=(physical&~fm)):raise ValueError('Current review fit/inspection partition')
            if not np.array_equal(physical,(self.arrays['admission']==1)&(self.arrays['fit_role']==1)):raise ValueError('Current review physical values differ from source columns')
            self.domain=physical.astype(np.uint8)*PHYSICAL;self.domain[training]|=TRAINING;self.domain[fm]|=FIT;self.domain[im]|=INSPECTION
        self.source_material_index=self.current_reader.load('source_material_index') if self.current_reader and 'source_material_index' in self.packet['arrays'] else self.arrays['material_index']
        self.source_material_binding=self.packet.get('source_material_binding',self.packet.get('material_binding',{}))
        self.retired=np.asarray(self.arrays['material_index']==np.iinfo(np.uint32).max)
        if self.current_reader and 'model_retired' in self.packet['arrays']:
            if not np.array_equal(self.retired,self.current_reader.load('model_retired')!=0):raise ValueError('Current review retired-material column differs')
        if np.any(self.retired&(self.domain!=0)):raise ValueError('Retired source rows retain current qualification')
        self.source_verdicts={}
        if self.current_manifest:
            ref=self.current_manifest['qualification']['source_identity_verdicts'];verdict=self.store.json(ref,Path(self.current_input_ref['file']).parent)
            self.source_verdicts={int(r['raw_row']):r for r in verdict['rows']}
        self.query=np.zeros(self.n,bool);self.queries={}
        for q in self.packet.get('queries',[]):
            row=int(q['row_index'])
            if not 0<=row<self.n or row in self.queries:raise ValueError('Review Query duplicate/range')
            self.query[row]=True;self.queries[row]=q
        self.allowed=self.packet.get('allowed_paths',{})
        self.row_domains={int(q['row_index']):q for q in self.packet.get('source_station_domains',{}).get('rows',[])}
        self.shot_ids=list(self.packet.get('shot_ids',[]))
        self.dense_paths=None
        reg=self.packet.get('dense_source_binding')
        if reg and 'allowed_path_mask' in reg:
            self.dense_paths=(int(reg['row_start']),int(reg['count']),self.store.array(reg['allowed_path_mask'],base)[0])
        self.native_path_pools=[]
        for reg in self.packet.get('native_ad_source_bindings',[self.packet.get('native_ad_source_binding',{})]):
            if 'allowed_paths_by_material' in reg:self.native_path_pools.append((int(reg['row_start']),int(reg['count']),{int(k):list(map(int,v)) for k,v in reg['allowed_paths_by_material'].items()}))
        self.reviewed_paths={}
        for rr in self.packet.get('resolved_native_print_identity_revision_chain',[]):
            revision=self.store.json(rr,base);review=self.store.json(revision['review'],Path(rr['file']).parent)
            if review.get('source_model_projection_used') is not False or review.get('source_pixels_changed') is not False:raise ValueError('Visual path authority used model projection')
            for e in review['rows']:self.reviewed_paths[int(e['raw_row'])]=(int(e['new_material_index']),int(e.get('new_path',e.get('old_path',0))))
        for rr in self.packet.get('resolved_native_source_measurement_revision_chain',[]):
            revision=self.store.json(rr,base)
            for e in revision.get('rows',[]):
                self.reviewed_paths[int(e['raw_row'])]=(int(e['new_material_index']),int(e['new_path']))
        self.reviewed_paths.update({int(k):tuple(map(int,v)) for k,v in self.packet.get('current_print_path_overrides',{}).items()})
        changed=[(i,mi,pi)for i,(mi,pi)in self.reviewed_paths.items()if int(self.arrays['material_index'][i])!=mi or int(self.arrays['path'][i])!=pi]
        if changed:
            self.arrays['material_index']=np.array(self.arrays['material_index'],copy=True);self.arrays['path']=np.array(self.arrays['path'],copy=True)
            for i,mi,pi in changed:self.arrays['material_index'][i]=mi;self.arrays['path'][i]=pi
        self._identity_arrays={}
        self.source_registry_expanded=False
    def finite_paths(self,row,model):
        row=int(row);mi=int(self.arrays['material_index'][row]);seed=int(self.arrays['path'][row]);kind=int(self.arrays['kind'][row])
        if self.retired[row]:return []
        if row in self.reviewed_paths:
            target,pi=self.reviewed_paths[row]
            if target!=mi or pi!=seed:raise ValueError('Reviewed path authority does not match current typed identity')
            return [pi]
        result=[seed]
        for start,count,pool in self.native_path_pools:
            if start<=row<start+count:
                if mi not in pool:raise ValueError('Missing compact native material authority')
                result=pool[mi];break
        if str(row) in self.allowed:result=list(map(int,self.allowed[str(row)]))
        elif int(self.arrays['shot_index'][row])==0 and kind!=1:
            current=model['materials'][mi]['paths'][seed]
            if current['kind']==3 and len(model['materials'][mi]['paths'])>1:
                try:policy=json.loads(current.get('provenance','{}')).get('descriptor',{}).get('association_policy',{})
                except (TypeError,ValueError):policy={}
                if policy.get('schema')=='rinkspace.continuous-board-association.v1' and policy.get('original_part_is_context_not_a_hard_correspondence') and policy.get('edge_role_is_preserved'):
                    parts=policy['candidate_parts'];result=[j for j,p in enumerate(model['materials'][mi]['paths']) if p['id'].rsplit('/',1)[-1] in parts]
                    if len(result)!=len(parts):raise ValueError('Incomplete source board candidate parts')
        if self.dense_paths:
            start,count,mask=self.dense_paths
            if start<=row<start+count:result=[k for k in range(8) if int(mask[row-start])&(1<<k)]
        refinements=model.get('printed_material_path_refinements',{})
        if kind!=1 and str(mi) in refinements:
            if row in self.row_domains:raise ValueError('Refined entity conflicts with original explicit station domain')
            result=list(map(int,refinements[str(mi)]['children']))
        if not result or len(result)!=len(set(result)) or any(p<0 or p>=len(model['materials'][mi]['paths']) for p in result):raise ValueError('Invalid finite source path set')
        return result
    def source_id(self,row):
        row=int(row);prefix=self.packet.get('source_row_ids',[])
        if row<len(prefix):return prefix[row]
        if self.current_reader:
            value=self.current_reader.source_id(row);self.current_reader._ids.pop(row,None);return value
        regs=[self.packet.get('dense_source_binding',{})]+list(self.packet.get('native_ad_source_bindings',[]))+[self.packet.get('native_ad_source_binding',{})]
        for reg in regs:
            if 'source_id_offsets' not in reg:continue
            start=int(reg.get('row_start',0));count=int(reg['count'])
            if start<=row<start+count:
                key=(reg['source_id_offsets']['sha256'],reg['source_id_utf8']['sha256'])
                if key not in self._identity_arrays:self._identity_arrays[key]=(self.store.array(reg['source_id_offsets'],Path(self.packet_ref['file']).parent)[0],self.store.array(reg['source_id_utf8'],Path(self.packet_ref['file']).parent)[0])
                offsets,data=self._identity_arrays[key];i=row-start;return memoryview(data[int(offsets[i]):int(offsets[i+1])]).tobytes().decode('utf-8')
        raise ValueError('Missing immutable source identity: '+str(row))
    def auxiliary(self,manifest=None):
        out={k:self.arrays[k] for k in ['path','station','row','alpha']}
        if manifest is None:return out
        mf=_json(manifest);root=Path(manifest).parent
        if mf.get('schema')=='rinkfit.current_complete_primal_transport.v1':
            if not self.current_input_ref or mf['current_input_v1']['sha256']!=self.current_input_ref['sha256'] or int(mf['raw_rows'])!=self.n:raise ValueError('Current complete review auxiliary binding differs')
            for k,dtype in [('path','<u4'),('station','<f8'),('row','<f8'),('alpha','<f8')]:
                ref=mf['raw_auxiliary'][k]
                if ref['dtype']!=dtype or ref['shape']!=[self.n]:raise ValueError('Current complete review auxiliary shape: '+k)
                out[k]=self.store.array(ref,root)[0]
            return out
        old_n=int(mf['raw_rows']);prefix=False
        if old_n!=self.n or mf['packet_sha256']!=self.packet_ref['sha256']:
            append=(self.selection or {}).get('source_append_selection_v1',{})
            proofref=append.get('append_source_proof')
            if not proofref or old_n!=int(append.get('parent_raw_rows',-1)) or self.n!=int(append.get('current_raw_rows',-1)) or mf['packet_sha256']!=append.get('original_state_packet',{}).get('sha256'):
                raise ValueError('Complete review auxiliaries are not current packet bound')
            proof=self.store.json(proofref,Path(manifest).parent)
            if not proof.get('passed') or proof['source_packet']['sha256']!=self.packet_ref['sha256'] or proof['original_state_packet']['sha256']!=mf['packet_sha256']:
                raise ValueError('Unverified source-append review auxiliary prefix')
            prefix=True
        for k,dtype,ext in [('path','<u4','u32'),('station','<f8','f64'),('row','<f8','f64'),('alpha','<f8','f64')]:
            filename=k+'.'+ext;r=mf['files'][filename];p=root/filename
            if file_ref(p)['sha256']!=r['sha256'] or p.stat().st_size!=old_n*np.dtype(dtype).itemsize:
                raise ValueError('Incomplete review auxiliary: '+filename)
            old=np.memmap(p,mode='r',dtype=dtype,shape=(old_n,));out[k]=PrefixTail(old,self.arrays[k]) if prefix else old
        return out

class PrefixTail:
    """A verified old complete prefix plus authentic appended packet seeds."""
    def __init__(self,prefix,current):self.prefix=prefix;self.current=current
    def __getitem__(self,key):
        if np.isscalar(key):return self.prefix[key] if key<len(self.prefix) else self.current[key]
        ids=np.arange(len(self.current))[key] if isinstance(key,slice) else np.asarray(key)
        value=np.asarray(self.current[ids]).copy();old=ids<len(self.prefix);value[old]=self.prefix[ids[old]];return value

class ReviewScene(NativeScene):
    """Current blocks and primitives; no historical graph or per-shot scene copy."""
    def __init__(self,model):
        self.joint=model
        blocks=list(model['blocks'])
        blocks[0]=blocks[int(model['scene_block'])]
        super().__init__({'blocks':blocks,'materials':model['materials']})
        self.original_values=[np.asarray(b['values'],float) for b in model['blocks']]
    def geometry(self,p):
        q=self.original_values[int(p['geometry_block'])]
        ids=p.get('geometry_indices',[])
        return q[ids].copy() if ids else q[int(p.get('geometry_offset',0)):].copy()
    def world(self,material,path,station,width=0.,sign=0):
        p=self.path(material,path);s=np.asarray(station,float);kind=p['kind']
        if not np.isfinite(s).all() or np.any((s<0)|(s>1)):raise ValueError('Finite material station')
        if kind in (0,1,4,5):
            q=self.geometry(p);shift=float(width)*int(sign)
            if shift:
                if kind==0:
                    dx,dy=q[2]-q[0],q[3]-q[1];length=np.hypot(dx,dy)
                    if length<=0:raise ValueError('Degenerate paint stroke')
                    q[[0,2]]-=shift*dy/length;q[[1,3]]+=shift*dx/length
                elif kind in (1,4):q[int(p.get('radius_index',2)) if kind==1 else 2]+=shift
                else:q[0]+=shift*np.sin(q[3]);q[1]-=shift*np.cos(q[3])
            if kind==0:return np.stack((q[0]*(1-s)+q[2]*s,q[1]*(1-s)+q[3]*s,np.zeros_like(s)),axis=-1)
            if kind==5:
                f=1-2*s;return np.stack((q[0]+q[2]*np.cos(q[3])*f,q[1]+q[2]*np.sin(q[3])*f,np.zeros_like(s)),axis=-1)
            angle=q[3]+np.pi*s if kind==4 else p.get('start_angle',0)+s*(p.get('end_angle',2*np.pi)-p.get('start_angle',0))
            radius=q[2] if kind==4 else q[int(p.get('radius_index',2))]
            return np.stack((q[0]+radius*np.cos(angle),q[1]+radius*np.sin(angle),np.zeros_like(s)),axis=-1)
        if width and sign:raise ValueError('Paint width cannot be applied to a non-ice primitive')
        if kind==3 and int(p.get('geometry_block',-1))>=0:
            fraction=float(self.geometry(p)[0])
            if not 0<=fraction<=1:raise ValueError('Board shared-height domain')
            segment=int(p.get('boundary_segment',-1));lengths=np.array([self.long,self.arc,self.short,self.arc,self.long,self.arc,self.short,self.arc])
            if segment==-2:
                coordinate=s*8;part=np.minimum(7,np.floor(coordinate).astype(int));position=np.r_[0.,np.cumsum(lengths)][part]+(coordinate-part)*lengths[part]
            elif segment>=0:position=float(lengths[:segment].sum())+s*lengths[segment]
            else:position=s*self.perimeter
            return self.board(position,self.height*fraction)
        if kind==2 and int(p.get('geometry_block',-1))>=0:
            # NativeScene's legacy world consumer expects the actual affine slice.
            q=self.geometry(p);points=np.asarray(p['points'],float)
            if len(points)==1:uv=np.broadcast_to(points[0],s.shape+(2,)).copy()
            else:
                x=s*(len(points)-1);i=np.minimum(np.floor(x).astype(int),len(points)-2);u=x-i
                uv=points[i]*(1-u[...,None])+points[i+1]*u[...,None]
            uv=uv@np.array([[q[0],q[3]],[q[1],q[4]]])+q[[2,5]]
            ad=self.original_values[int(p['advertisement_block'])][int(p.get('advertisement_offset',0)):][:4]
            return self.board(ad[0]+ad[1]*uv[...,0],ad[2]+(ad[3]-ad[2])*uv[...,1])
        return super().world(material,path,s)
    def knots(self,material,path):
        p=self.path(material,path);kind=p['kind']
        if kind==2:
            n=len(p['points']);return np.linspace(0,1,max(1,n)) # N-1, never an invented closing edge
        if kind==6:
            n=len(p.get('vertex_blocks',[])) or len(self.geometry(p))//2
            edges=n if p.get('closed',False) and n>1 else n-1
            return np.linspace(0,1,max(1,edges+1))
        if p.get('physical_point',False):return np.array([0.])
        if kind in (1,4,7):return np.linspace(0,1,33)
        if kind==3:return np.linspace(0,1,33)
        return np.array([0.,1.])

def _merge_intervals(intervals):
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([float(lo),float(hi)])
    return merged

def station_intervals(inputs,scene,row,path,permissions=None):
    mi=int(inputs.arrays['material_index'][row]);model=scene.joint
    def original(pi):
        p=scene.path(mi,pi);result=[list(map(float,x)) for x in p.get('allowed_station_intervals',[[0.,1.]])]
        q=inputs.row_domains.get(int(row)) if int(row) not in inputs.reviewed_paths else None
        if q:
            if int(q['path_index'])!=pi:return []
            result=[[max(a,c),min(b,d)] for a,b in result for c,d in q['allowed_station_intervals'] if max(a,c)<=min(b,d)]
        return result
    parts=model.get('continuous_board_station_v1',{}).get('original_segments',{})
    current=int(inputs.arrays['path'][row]);key=str(mi)+'/'+str(current)
    if int(inputs.arrays['kind'][row])!=1 and key in parts:
        # Exact native numerical chart: merge only source-permitted original
        # parts, then use (segment+s)/8. Aliases share this same world curve.
        allowed=permissions if permissions is not None else inputs.finite_paths(row,model)
        intervals=[];ref=scene.path(mi,current)
        for pi in allowed:
            partkey=str(mi)+'/'+str(pi);p=scene.path(mi,pi)
            if partkey not in parts or p.get('geometry_block',-1)!=ref.get('geometry_block',-1) or p.get('fixed_height_fraction',0)!=ref.get('fixed_height_fraction',0):raise ValueError('Continuous board source permission changed edge role')
            segment=int(parts[partkey])
            intervals.extend([(segment+a)/8.,(segment+b)/8.] for a,b in original(pi))
        return _merge_intervals(intervals)
    return original(path)

def _project(scene,camera,local,mi,pi,s,width,sign,visibility):
    world=scene.world(mi,pi,np.asarray(s),width,sign)
    uv,valid,eq=fast_native_project(camera,world,local)
    finite=np.isfinite(uv).all(axis=1)
    visible=np.ones(len(s),bool)
    if visibility:
        p=scene.path(mi,pi);label=scene.materials[mi]['label']
        top=p['kind']==3 and (label=='board.top' or label.startswith('board.top.')) and abs(float(p.get('fixed_height_fraction',-1))-1)<1e-12
        visible=render_visibility(camera,scene,world,[dict(begin=0,end=len(s),label=label,board_material_surface=p['kind'] in (2,3,6,7),board_top_boundary=top)],local,uv)
    return uv,finite,valid,visible

def projected_curve(scene,camera,local,mi,pi,width=0.,sign=0,*,tolerance=.05,visibility=True,intervals=None):
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive visual chord tolerance required')
    knots=scene.knots(mi,pi)
    intervals=_merge_intervals([[0.,1.]] if intervals is None else intervals)
    if not intervals:return np.empty(0),np.empty((0,2)),np.empty(0,bool),False
    s=np.unique(np.concatenate([np.r_[lo,knots[(knots>lo)&(knots<hi)],hi] for lo,hi in intervals]))
    uv,finite,rs,vis=_project(scene,camera,local,mi,pi,s,width,sign,visibility)
    if len(s)==1:return s,uv,finite&rs&vis,False
    # Refinement terminates on a numerical tolerance, never a point-count quota.
    unresolved=False
    for depth in range(18):
        mid=(s[:-1]+s[1:])*.5
        mu,mf,mr,mv=_project(scene,camera,local,mi,pi,mid,width,sign,visibility)
        finite_end=finite[:-1]&finite[1:]&mf
        error=np.linalg.norm(mu-(uv[:-1]+uv[1:])*.5,axis=1)
        changed=(rs[:-1]!=rs[1:])|(vis[:-1]!=vis[1:])|(rs[:-1]!=mr)|(vis[:-1]!=mv)
        endpoint0=rs[:-1]&vis[:-1];endpoint1=rs[1:]&vis[1:];middle=mr&mv
        same_interval=np.zeros(len(mid),bool)
        for lo,hi in intervals:same_interval|=(s[:-1]>=lo)&(s[1:]<=hi)
        # Never chase huge off-domain perspective coordinates. Uncertified or
        # hidden spans cannot become visual-valid through extra subdivision.
        supported=endpoint0|endpoint1|middle
        refine=finite_end&same_interval&supported&(((endpoint0&endpoint1&middle)&(error>tolerance))|changed)&((s[1:]-s[:-1])>2e-14)
        if not np.any(refine):break
        if depth==17:unresolved=True;break
        insert=np.flatnonzero(refine)
        ns=np.r_[s,mid[insert]];order=np.argsort(ns)
        s=ns[order];uv=np.vstack((uv,mu[insert]))[order]
        finite=np.r_[finite,mf[insert]][order];rs=np.r_[rs,mr[insert]][order];vis=np.r_[vis,mv[insert]][order]
    return s,uv,finite&rs&vis,unresolved

def nearest_segments(points,stations,uv,valid,intervals):
    """Exact distance to the supplied projected chord set within source intervals.

    KD radii use a triangle-inequality lower bound, so distant long segments
    cannot be dropped merely because their endpoint is not a nearest vertex.
    """
    points=np.asarray(points,float);count=len(points)
    dist=np.full(count,np.inf);nearest=np.full((count,2),np.nan);where=np.full(count,np.nan)
    if len(stations)==1:
        legal=valid[0]
        if legal:
            for j,ivs in enumerate(intervals):
                if any(a<=stations[0]<=b for a,b in ivs):dist[j]=np.linalg.norm(points[j]-uv[0]);nearest[j]=uv[0];where[j]=stations[0]
        return dist,nearest,where
    use=valid[:-1]&valid[1:]&np.isfinite(uv[:-1]).all(axis=1)&np.isfinite(uv[1:]).all(axis=1)
    ids=np.flatnonzero(use)
    if not len(ids):return dist,nearest,where
    starts=uv[ids];ends=uv[ids+1];mid=(starts+ends)*.5;half=np.linalg.norm(ends-starts,axis=1)*.5
    tree=cKDTree(mid);upper=np.linalg.norm(points-starts[tree.query(points,k=1)[1]],axis=1)
    candidates=tree.query_ball_point(points,upper+float(half.max())+1e-10)
    for j,ixs in enumerate(candidates):
        if intervals[j]!=[[0.,1.]] and intervals[j]!=[(0.,1.)]:ixs=range(len(ids))
        for interval_lo,interval_hi in intervals[j]:
            ii=np.asarray(ixs,int);actual=ids[ii];lo=np.maximum(stations[actual],interval_lo);hi=np.minimum(stations[actual+1],interval_hi)
            good=lo<=hi
            if not np.any(good):continue
            ii=ii[good];actual=actual[good];lo=lo[good];hi=hi[good]
            span=stations[actual+1]-stations[actual];a=(lo-stations[actual])/span;b=(hi-stations[actual])/span
            v=ends[ii]-starts[ii];aa=starts[ii]+v*a[:,None];vv=v*(b-a)[:,None];den=np.sum(vv*vv,axis=1)
            t=np.divide(np.sum((points[j]-aa)*vv,axis=1),den,out=np.zeros(len(ii)),where=den>0);t=np.clip(t,0,1);q=aa+vv*t[:,None]
            d=np.sum((q-points[j])**2,axis=1);k=int(np.argmin(d))
            if d[k]<dist[j]**2:dist[j]=np.sqrt(d[k]);nearest[j]=q[k];where[j]=lo[k]+(hi[k]-lo[k])*t[k]
    return dist,nearest,where

def _width_groups(model,inputs):
    blocks=np.full(inputs.n,-1,np.int32);signs=np.zeros(inputs.n,np.int8)
    bindings={b['pair_id']:b for b in model.get('stroke_pair_model_bindings',[])}
    scopes={x['width_scope_id']:int(x['half_width_block']) for x in model.get('measurement_widths',[]) if x.get('scope_is_source_confirmed_section_not_uniform_whole_stroke')}
    for pair in inputs.packet.get('stroke_pairs',[]):
        rows=list(map(int,pair['rows']));b=bindings.get(pair['id'])
        if b is None:
            if any(inputs.domain[r]&PHYSICAL for r in rows):raise ValueError('Current admitted width pair lacks model binding')
            continue
        if b['width_scope_id']!=pair['width_scope_id'] or scopes.get(b['width_scope_id'])!=int(b['half_width_block']):raise ValueError('Width scope changed')
        if not np.isfinite(model['blocks'][int(b['half_width_block'])]['values'][0]) or model['blocks'][int(b['half_width_block'])]['values'][0]<=0:raise ValueError('Invalid current positive halfwidth')
        ss=list(map(int,b['source_world_signs']))
        if ss not in pair['allowed_world_side_assignments']:raise ValueError('Width world-side assignment not source-allowed')
        if len(rows)!=2 or len(ss)!=2 or sorted(ss)!=[-1,1]:raise ValueError('Width-side binding')
        for r,sg in zip(rows,ss):blocks[r]=int(b['half_width_block']);signs[r]=sg
    return blocks,signs

def _open_outputs(output,n):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);out={}
    for key,(dtype,tail) in ARRAYS.items():
        a=np.memmap(output/ARRAY_FILENAMES[key],mode='w+',dtype=dtype,shape=(n,)+tail)
        a[:]=np.nan if a.dtype.kind=='f' else (-1 if a.dtype.kind=='i' else 0);out[key]=a
    return out

def valid_status(status):return ((status&VALID_BITS)==VALID_BITS)&((status&(DOMAIN_FAILURE|SAMPLING_UNRESOLVED))==0)

def evaluate(model_file,packet_file,selection_file,output,*,nuisance_manifest=None,
             inputs=None,cameras=None,raw_rows=None,chord_tolerance_px=.05,
             visibility=True,progress=None,current_input=None):
    """Write all-source row-addressed metrics; optional raw_rows is declared scope."""
    started=time.perf_counter();operator_refs=_operator_refs();inputs=inputs or ReviewInputs(packet_file,selection_file,current_input=current_input)
    model=_json(model_file);scene=ReviewScene(model);cameras=cameras or CurrentJointCameras(model);cameras.validate()
    aux=inputs.auxiliary(nuisance_manifest);out=_open_outputs(output,inputs.n);out['domain'][:]=inputs.domain
    selected=np.arange(inputs.n,dtype=np.int64) if raw_rows is None else np.unique(np.asarray(raw_rows,np.int64))
    if len(selected) and (selected[0]<0 or selected[-1]>=inputs.n):raise ValueError('Review row outside immutable domain')
    row_state=np.memmap(Path(output)/'row_state.u8',mode='w+',dtype='u1',shape=(inputs.n,));row_state[:]=0
    wb,ws=_width_groups(model,inputs);a=inputs.arrays
    source_labels={int(v['index']):v['label']for v in inputs.source_material_binding.get('materials',[])}
    order=selected[np.lexsort((ws[selected],wb[selected],inputs.source_material_index[selected],a['material_index'][selected],a['native_pts'][selected],a['shot_index'][selected]))]
    if len(order):
        keys=np.c_[a['shot_index'][order],a['native_pts'][order],a['material_index'][order],inputs.source_material_index[order],wb[order],ws[order]]
        cuts=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))+1,len(order)]
    else:cuts=[0]
    stats=Path(output)/'entity_metrics.jsonl';done=0
    with stats.open('w',encoding='utf-8') as stream:
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            rows=order[lo:hi];sh=int(a['shot_index'][rows[0]]);pts=int(a['native_pts'][rows[0]]);mi=int(a['material_index'][rows[0]])
            sid=inputs.shot_ids[sh]
            if sid not in model['shots']:raise ValueError('Current review shot is absent from complete model')
            source_mi=int(inputs.source_material_index[rows[0]])
            if mi==int(np.iinfo(np.uint32).max):
                if not np.all(inputs.retired[rows]) or np.any(inputs.domain[rows]):raise ValueError('Retired material has active review membership')
                out['status'][rows]=PROCESSED|DOMAIN_FAILURE;out['operator'][rows]=np.where(inputs.query[rows],3,np.where(a['kind'][rows]==1,1,2));row_state[rows]=2
                record=dict(shot_id=sid,native_pts=pts,material_index=mi,label=None,source_material_index=source_mi,source_label=source_labels.get(source_mi),raw_rows=len(rows),valid=0,invalid=len(rows),model_retired=True,operator_failure='retired_from_current_model; no projection or entity reassignment',source_qualification_counts={'physical':0,'training':0,'fit':0,'inspection':0})
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');done+=len(rows)
                if progress:progress({'processed':done,'selected':len(order),'shot_id':sid,'native_pts':pts,'elapsed_seconds':time.perf_counter()-started})
                continue
            if not 0<=mi<len(model['materials']):raise ValueError('Non-retired raw row has no current model material')
            camera=cameras.native_camera(sid);local=float(cameras.local_seconds(sid,pts))
            row_state[rows]=1
            width=float(model['blocks'][int(wb[rows[0]])]['values'][0]) if wb[rows[0]]>=0 else 0.;sign=int(ws[rows[0]])
            out['status'][rows]=PROCESSED|np.where(inputs.query[rows],QUERY_PENDING,0).astype(np.uint8)
            out['operator'][rows]=np.where(inputs.query[rows],3,np.where(a['kind'][rows]==1,1,2))
            best=np.full(len(rows),np.inf);buv=np.full((len(rows),2),np.nan);bs=np.full(len(rows),np.nan);bp=np.full(len(rows),-1,np.int32);unresolved=np.zeros(len(rows),bool)
            fixed=a['kind'][rows]==1
            try:
                permissions=[inputs.finite_paths(int(r),model) for r in rows]
                for pi in range(len(model['materials'][mi]['paths'])):
                    chart=model.get('continuous_board_station_v1',{}).get('original_segments',{})
                    continuous=str(mi)+'/'+str(pi) in chart and model['materials'][mi]['paths'][pi].get('boundary_segment')==-2
                    if continuous:
                        representatives=[int(aux['path'][r]) if int(aux['path'][r]) in pp else pp[0] for r,pp in zip(rows,permissions)]
                        admissible=np.array([pi==chosen for chosen in representatives])
                    else:admissible=np.array([pi in pp for pp in permissions])
                    f=admissible&fixed&(np.asarray(aux['path'][rows])==pi)
                    if f.any():
                        rr=rows[f];s=np.asarray(aux['station'][rr]);ivs=[station_intervals(inputs,scene,int(r),pi,permissions[j]) for j,r in enumerate(rows) if f[j]]
                        legal=np.array([any(x-1e-12<=t<=y+1e-12 for x,y in iv) for t,iv in zip(s,ivs)])
                        uv,finite,rs,vis=_project(scene,camera,local,mi,pi,s,width,sign,visibility);ok=legal&finite&rs&vis
                        inds=np.flatnonzero(f)[ok];best[inds]=np.linalg.norm(uv[ok]-a['uv'][rr[ok]],axis=1);buv[inds]=uv[ok];bs[inds]=s[ok];bp[inds]=pi
                        out['status'][rr]|=FIXED
                    p=admissible&~fixed
                    if not p.any():continue
                    rr=rows[p];ivs=[station_intervals(inputs,scene,int(r),pi,permissions[j]) for j,r in enumerate(rows) if p[j]]
                    ss,uu,vv,pending=projected_curve(scene,camera,local,mi,pi,width,sign,tolerance=chord_tolerance_px,visibility=visibility,intervals=_merge_intervals([v for iv in ivs for v in iv]))
                    dd,qq,tt=nearest_segments(a['uv'][rr],ss,uu,vv,ivs)
                    inds=np.flatnonzero(p);better=dd<best[inds];ids=inds[better]
                    best[ids]=dd[better];buv[ids]=qq[better];bs[ids]=tt[better];bp[ids]=pi;unresolved[ids]=pending
                ok=np.isfinite(best);out['distance'][rows[ok]]=best[ok];out['nearest_uv'][rows[ok]]=buv[ok];out['station'][rows[ok]]=bs[ok];out['path'][rows[ok]]=bp[ok]
                out['status'][rows[ok]]|=FINITE|RS_CERTIFIED|VISIBLE;out['status'][rows[~ok]]|=DOMAIN_FAILURE;out['status'][rows[unresolved]]|=SAMPLING_UNRESOLVED
                why=None
            except (ValueError,IndexError,KeyError,FloatingPointError) as exc:
                out['status'][rows]|=DOMAIN_FAILURE;why=type(exc).__name__+': '+str(exc);ok=np.zeros(len(rows),bool)
            record_ok=ok&~unresolved;row_state[rows[~record_ok]]=3;data=best[record_ok];record={'shot_id':sid,'native_pts':pts,'material_index':mi,'label':model['materials'][mi]['label'],'width_block':int(wb[rows[0]]),'width_sign':sign,'raw_rows':len(rows),'valid':int(record_ok.sum()),'invalid':int((~record_ok).sum()),'sum_squared_distance_px':float(data@data),'max_distance_px':float(data.max()) if len(data) else None,'p50_px':float(np.median(data)) if len(data) else None,'p95_px':float(np.quantile(data,.95)) if len(data) else None,'formal_query_pending':int(inputs.query[rows].sum()),'sampling_unresolved_rows':int(unresolved.sum()),'operator_failure':why}
            record.update(model_retired=False,source_material_index=source_mi,source_label=source_labels.get(source_mi),source_qualification_counts={name:int(np.count_nonzero(inputs.domain[rows]&bit))for name,bit in [('physical',PHYSICAL),('training',TRAINING),('fit',FIT),('inspection',INSPECTION)]},source_decisions=dict(collections.Counter(inputs.source_verdicts[int(i)]['decision']for i in rows if int(i)in inputs.source_verdicts)))
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');done+=len(rows)
            if progress:progress({'processed':done,'selected':len(order),'shot_id':sid,'native_pts':pts,'elapsed_seconds':time.perf_counter()-started})
    for value in out.values():value.flush()
    row_state.flush()
    good=valid_status(out['status']);processed=(out['status']&PROCESSED)!=0
    report={'schema':'rinkfit.dense_visible_metrics.v1','source_packet':inputs.packet_ref,'selection':inputs.selection_ref,'model':file_ref(model_file),'model_id':model.get('model_id'),'model_source_clock_sha256':hashlib.sha256(json.dumps(model.get('source_clock'),sort_keys=True,separators=(',',':')).encode()).hexdigest(),'operator_source_files':operator_refs,'nuisance_manifest':file_ref(nuisance_manifest) if nuisance_manifest else None,'nuisance_binding_policy':'complete_SHA_bound_auxiliary_or_authenticated_append_prefix' if nuisance_manifest else 'packet_seed_only_not_current_solver_nuisance','raw_rows':inputs.n,'processed_rows':int(processed.sum()),'complete_raw_domain_processed':bool(np.all(processed)),'valid_rows':int(good.sum()),'visual_valid_rows':int(good.sum()),'valid_rows_are_visual_only':True,'formal_valid_rows':0,'formal_pending_rows':int(processed.sum()),'invalid_rows':int((processed&~good).sum()),'formal_query_pending':int(np.count_nonzero(out['status']&QUERY_PENDING)),'sampling_unresolved_rows':int(np.count_nonzero(out['status']&SAMPLING_UNRESOLVED)),'sampling':{'operator':'source_allowed_finite_curve_projected_chord_distance','visibility':bool(visibility),'midpoint_chord_tolerance_px':chord_tolerance_px,'maximum_subdivision_depth':18,'uniform_global_error_certificate':False,'unresolved_depth_reported':True,'kind2_invented_closing_edges':False},'operator_codes':{'1':'fixed_source_station_RS_projection','2':'source_finite_curve_visual_profile','3':'Query_bearing_visual_profile_formal_pending'},'source_lineage':{'packet_ancestry':inputs.ancestry,'all_source_strings_expanded':False,'source_UV_modified':False,'source_eligibility_modified':False,'weights_modified':False},'formal_Query_or_full_W_station_optimum_certified':False,'physical_fit_or_image_quality_accepted':False,'elapsed_seconds':time.perf_counter()-started,'entity_metrics':file_ref(stats),'arrays':{}}
    report['geometry_operator_contract_sha256']=_geometry_contract_sha()
    report['current_input_v1']=inputs.current_input_ref
    report['model_retired_rows']=int(np.count_nonzero(processed&inputs.retired))
    report['source_decision_counts']=dict(collections.Counter(v['decision']for i,v in inputs.source_verdicts.items()if processed[i]))
    report['qualified_visual_valid_rows']=int(np.count_nonzero(good&((inputs.domain&PHYSICAL)!=0)))
    report['row_state']=dict(file_ref(row_state.filename),dtype='|u1',shape=[inputs.n],codes={'0':'not_processed','1':'geometry_attempted','2':'retired_model_no_target','3':'geometry_domain_failure_or_sampling_unresolved'})
    for key,(dtype,tail) in ARRAYS.items():report['arrays'][key]=dict(file_ref(out[key].filename),dtype=dtype,shape=[inputs.n]+list(tail))
    for r in operator_refs.values():_checked_file(r)
    _write(Path(output)/'report.json',report);return report

def _operator_refs():
    root=Path(__file__).resolve().parents[1]
    names=['rinkfit_joint/dense_visible_metrics.py','rinkfit_joint/joint_review_camera.py',
           'rinkfit_joint/joint_replay.py','rinkfit_joint/joint_model.py',
           'rinkfit_joint/effective_input_audit_v1.py','rinkfit_joint/dense_fixed_anchor_guard.py',
           'rinkfit_joint/camera_graph.py','rinkfit_joint/joint_review_packet.py','rinkfit_start/native_scene.py',
           'rinkfit_start/native_camera.py']
    return {n:file_ref(root/n) for n in names}

def _checked_file(binding):
    actual=file_ref(binding['file'])
    if actual['sha256']!=binding['sha256'] or ('byte_count' in binding and actual['byte_count']!=binding['byte_count']):
        raise ValueError('Review artifact changed: '+binding['file'])
    return Path(binding['file'])

def _metric_arrays(report):
    out={}
    for k,b in report['arrays'].items():
        path=_checked_file(b);dtype=np.dtype(b['dtype']);shape=tuple(b['shape'])
        if shape!=(report['raw_rows'],)+ARRAYS[k][1] or dtype!=np.dtype(ARRAYS[k][0]) or path.stat().st_size!=math.prod(shape)*dtype.itemsize:
            raise ValueError('Metric array ABI/shape: '+k)
        out[k]=np.memmap(path,mode='r',dtype=dtype,shape=shape)
    if set(out)!=set(ARRAYS):raise ValueError('Incomplete metric array set')
    return out

def compare(baseline_report,candidate_report,output,*,worsened_threshold_px=0.,
            inputs=None,require_complete=True,baseline_inputs=None,candidate_inputs=None):
    """Complete same-source row comparison with actual SHA/ABI verification."""
    if baseline_inputs is not None or candidate_inputs is not None:
        if inputs is not None or baseline_inputs is None or candidate_inputs is None:raise ValueError('Supply both explicit comparison inputs, or the legacy single inputs argument')
        return _compare_source_transition(baseline_report,candidate_report,output,baseline_inputs,candidate_inputs,worsened_threshold_px,require_complete)
    if not math.isfinite(worsened_threshold_px) or worsened_threshold_px<0:raise ValueError('Invalid comparison threshold')
    before=_json(baseline_report);after=_json(candidate_report)
    if before['schema']!='rinkfit.dense_visible_metrics.v1' or after['schema']!=before['schema']:raise ValueError('Metric schema')
    if before['source_packet']['sha256']!=after['source_packet']['sha256'] or before['raw_rows']!=after['raw_rows']:
        raise ValueError('Visual comparison source identity/domain changed')
    for b in [before,after]:
        _checked_file(b['source_packet']);_checked_file(b['model'])
        if b.get('selection'):_checked_file(b['selection'])
        if b.get('nuisance_manifest'):_checked_file(b['nuisance_manifest'])
        for r in b['operator_source_files'].values():_checked_file(r)
        if require_complete and (not b['complete_raw_domain_processed'] or b['processed_rows']!=b['raw_rows']):
            raise ValueError('Incomplete raw domain cannot be used for candidate promotion')
    if (before.get('selection') or {}).get('sha256')!=(after.get('selection') or {}).get('sha256'):raise ValueError('Review selection differs')
    if before['sampling']!=after['sampling'] or before['operator_codes']!=after['operator_codes']:
        raise ValueError('Visual comparison operator changed')
    if {k:v['sha256'] for k,v in before['operator_source_files'].items()}!={k:v['sha256'] for k,v in after['operator_source_files'].items()}:
        raise ValueError('Numerical source operators differ')
    if before['model_source_clock_sha256']!=after['model_source_clock_sha256']:raise ValueError('Model source clock changed')
    aa=_metric_arrays(before);bb=_metric_arrays(after)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names=['newly_invalid','recovered','still_invalid','worsened','improved','common_valid']
    streams={k:(output/(k+'.u64')).open('wb') for k in names}
    counts=collections.Counter();roles={k:collections.Counter() for k in ['physical','training','fit','inspection']}
    sse_before=0.;sse_after=0.;maxreg=0.
    try:
        for lo in range(0,before['raw_rows'],65536):
            hi=min(before['raw_rows'],lo+65536);sa=aa['status'][lo:hi];sb=bb['status'][lo:hi]
            pa=(sa&PROCESSED)!=0;pb=(sb&PROCESSED)!=0
            if require_complete and not np.all(pa):raise ValueError('Incomplete processed raw status cannot be promoted')
            if not np.array_equal(pa,pb) or not np.array_equal(aa['domain'][lo:hi],bb['domain'][lo:hi]) or not np.array_equal(aa['operator'][lo:hi],bb['operator'][lo:hi]):raise ValueError('Comparison row scope/qualification/operator differs')
            va=valid_status(sa);vb=valid_status(sb);common=pa&va&vb;d0=aa['distance'][lo:hi].astype(float);d1=bb['distance'][lo:hi].astype(float)
            for mask,buf in [(va,aa),(vb,bb)]:
                if np.any(mask & (~np.isfinite(buf['nearest_uv'][lo:hi]).all(axis=1)|~np.isfinite(buf['station'][lo:hi])|(buf['station'][lo:hi]<0)|(buf['station'][lo:hi]>1)|(buf['path'][lo:hi]<0))):raise ValueError('Visual-valid coordinates/station/path invalid')
            if np.any(va&~np.isfinite(d0)) or np.any(vb&~np.isfinite(d1)) or np.any(d0[va]<0) or np.any(d1[vb]<0):
                raise ValueError('A visual-valid distance is nonfinite/negative')
            masks={'newly_invalid':pa&va&~vb,'recovered':pa&~va&vb,'still_invalid':pa&~va&~vb,'worsened':common&(d1>d0+worsened_threshold_px),'improved':common&(d1<d0-worsened_threshold_px),'common_valid':common}
            for k,m in masks.items():
                ids=np.flatnonzero(m).astype('<u8')+lo;streams[k].write(ids.tobytes());counts[k]+=len(ids)
                for role,bit in [('physical',PHYSICAL),('training',TRAINING),('fit',FIT),('inspection',INSPECTION)]:roles[role][k]+=int(np.count_nonzero(m&(aa['domain'][lo:hi]&bit!=0)))
            counts['processed']+=int(pa.sum());sse_before+=float(d0[common]@d0[common]);sse_after+=float(d1[common]@d1[common])
            if common.any():maxreg=max(maxreg,float(np.max(d1[common]-d0[common])))
    finally:
        for f in streams.values():f.close()
    report={'schema':'rinkfit.dense_visible_same_source_comparison.v1','baseline':file_ref(baseline_report),'candidate':file_ref(candidate_report),'baseline_model':before['model'],'candidate_model':after['model'],'source_packet':before['source_packet'],'selection':before['selection'],'operator_source_files':before['operator_source_files'],'sampling':before['sampling'],'raw_rows':before['raw_rows'],'counts':dict(counts),'counts_by_domain':{k:dict(v) for k,v in roles.items()},'same_valid_sum_squared_before_px':sse_before,'same_valid_sum_squared_after_px':sse_after,'same_valid_sum_squared_delta_px':sse_after-sse_before,'maximum_distance_increase_px':maxreg,'worsened_threshold_px':worsened_threshold_px,'failure_sets_complete_for_declared_scope':True,'full_raw_domain_compared':before['complete_raw_domain_processed'] and after['complete_raw_domain_processed'],'source_noise_W_objective_compared':False,'formal_pending_is_not_formal_valid':True,'quality_accepted':False,'sets':{k:file_ref(f.name) for k,f in streams.items()}}
    if inputs is not None:
        if inputs.packet_ref['sha256']!=before['source_packet']['sha256'] or inputs.n!=before['raw_rows']:raise ValueError('Comparison source input differs')
        report['entity_comparison'],report['automatic_candidates']=_entity_comparison(inputs,aa,bb,output,worsened_threshold_px)
    _write(output/'comparison.json',report)
    if inputs is not None:
        frames=[]
        for sid,r in sorted(report['automatic_candidates'].items()):
            raw=int(r['raw_row']);frames.append(dict(r,shot_id=sid,source_uv=inputs.arrays['uv'][raw].tolist(),source_row_id=inputs.source_id(raw)))
        automatic={'schema':'rinkfit.dense_visible_automatic_frames.v1','comparison':file_ref(output/'comparison.json'),'source_packet':before['source_packet'],'selection':before['selection'],'baseline_model':before['model'],'candidate_model':after['model'],'operator_source_files':before['operator_source_files'],'complete_raw_domain_compared':report['full_raw_domain_compared'],'frames':frames,'selection_uses_complete_numeric_comparison':True,'quality_accepted':False}
        _write(output/'automatic_frames.json',automatic)
    return report


def _original_source_catalog(inputs):
    """Validate compact ID payloads once, without hydrating suffix strings."""
    prefix=inputs.packet.get('source_row_ids',[]);datasets=inputs.packet.get('source_dataset_ids',[])
    registries=[inputs.packet.get('dense_source_binding',{})]+list(inputs.packet.get('native_ad_source_bindings',[]))+[inputs.packet.get('native_ad_source_binding',{})]
    result=[];seen=set()
    for reg in registries:
        if 'source_id_offsets'not in reg:continue
        key=(int(reg['row_start']),int(reg['count']),reg['source_id_offsets']['sha256'],reg['source_id_utf8']['sha256'],reg.get('source_dataset_id'))
        if key in seen:continue
        seen.add(key);result.append(key)
        if inputs.current_reader:
            offsets=inputs.current_reader._column(reg['source_id_offsets'],inputs.current_reader.root);data=inputs.current_reader._column(reg['source_id_utf8'],inputs.current_reader.root)
        else:
            offsets=inputs.store.array(reg['source_id_offsets'],Path(inputs.packet_ref['file']).parent)[0];data=inputs.store.array(reg['source_id_utf8'],Path(inputs.packet_ref['file']).parent)[0]
        if len(offsets)!=key[1]+1 or int(offsets[0])!=0 or int(offsets[-1])!=len(data)or np.any(offsets[1:]<=offsets[:-1]):raise ValueError('Compact source ID offsets are incomplete')
    return prefix,datasets,sorted(result)

def _dataset_id(inputs,row):
    prefix=inputs.packet.get('source_dataset_ids',[])
    if row<len(prefix):return prefix[row]
    regs=[inputs.packet.get('dense_source_binding',{})]+list(inputs.packet.get('native_ad_source_bindings',[]))+[inputs.packet.get('native_ad_source_binding',{})]
    for reg in regs:
        if int(reg.get('row_start',-1))<=row<int(reg.get('row_start',-1))+int(reg.get('count',0)):return reg.get('source_dataset_id')
    return None

def _source_mismatch(output,rows,reason):
    _write(Path(output)/'source_identity_mismatch.json',dict(schema='rinkfit.visible_source_identity_mismatch.v1',raw_rows=list(map(int,rows)),reason=reason,same_source_comparison_refused=True,geometry_improvement_claimed=False))
    raise ValueError('Same-source comparison refused: '+reason+'; raw rows '+str(list(map(int,rows[:16]))))

def _check_comparison_source(a,b,output):
    if a.n!=b.n:raise ValueError('Raw source denominator changed; no positional same-source comparison')
    if a.source_material_binding!=b.source_material_binding:
        left={int(v['index']):v for v in a.source_material_binding.get('materials',[])};right={int(v['index']):v for v in b.source_material_binding.get('materials',[])}
        changed=[i for i in set(left)|set(right)if left.get(i)!=right.get(i)]
        rows=np.flatnonzero(np.isin(a.source_material_index,changed))[:32]
        _source_mismatch(output,rows,'Original source material/path catalog changed: '+str(changed[:16]))
    ca,cb=_original_source_catalog(a),_original_source_catalog(b);catalog_equal=ca==cb
    for lo in range(0,a.n,65536):
        hi=min(a.n,lo+65536);bad=np.zeros(hi-lo,bool)
        for key in ('uv','native_pts'):
            x=np.asarray(a.arrays[key][lo:hi]);y=np.asarray(b.arrays[key][lo:hi])
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Original source array ABI differs: '+key)
            if x.dtype.kind=='f':same=(x.view('u1').reshape(len(x),-1)==y.view('u1').reshape(len(y),-1)).all(axis=1)
            else:same=x==y
            bad|=~same
        sa=np.asarray(a.arrays['shot_index'][lo:hi]);sb=np.asarray(b.arrays['shot_index'][lo:hi])
        if np.any(sa>=len(a.shot_ids))or np.any(sb>=len(b.shot_ids)):raise ValueError('Source shot index outside declared clock')
        mapping=np.array([b.shot_ids.index(s)if s in b.shot_ids else -1 for s in a.shot_ids]);bad|=mapping[sa]!=sb
        bad|=np.asarray(a.source_material_index[lo:hi])!=np.asarray(b.source_material_index[lo:hi])
        if bad.any():_source_mismatch(output,np.flatnonzero(bad)+lo,'Original UV/PTS/shot/source_material_index changed')
        if not catalog_equal:
            for row in range(lo,hi):
                if a.source_id(row)!=b.source_id(row)or _dataset_id(a,row)!=_dataset_id(b,row):_source_mismatch(output,np.array([row]),'Original raw source ID/dataset changed')
                # Fixed-frame readers cache requested IDs; a full streamed comparison
                # must not turn that bounded cache into four million Python strings.
                for inputs in (a,b):
                    if inputs.current_reader and row>=len(inputs.packet.get('source_row_ids',[])):inputs.current_reader._ids.pop(row,None)
    return dict(raw_index_stable=True,raw_ID_UV_PTS_shot_and_original_material_verified=True,original_material_binding_sha256=hashlib.sha256(json.dumps(a.source_material_binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),baseline_original_material_column=a.packet['arrays'].get('source_material_index',a.packet['arrays']['material_index']),candidate_original_material_column=b.packet['arrays'].get('source_material_index',b.packet['arrays']['material_index']),compact_identity_payload_catalog_equal=catalog_equal,all_source_ID_strings_materialized=False)

def _material_contract(model,mi):
    material=model['materials'][mi]
    def path_contract(path):
        keys=('id','variant_id','kind','physical_point','closed','radius_index','start_angle','end_angle','boundary_segment','fixed_height_fraction','points','geometry_offset','geometry_indices','advertisement_offset','allowed_station_intervals')
        result={key:path[key]for key in keys if key in path}
        for key in ('geometry_block','advertisement_block','course_envelope_block'):
            bi=int(path.get(key,-1));result[key]=model['blocks'][bi]['name']if bi>=0 else None
        if 'vertex_blocks'in path:result['vertex_blocks']=[model['blocks'][int(i)]['name']for i in path['vertex_blocks']]
        provenance=path.get('provenance',{})
        if isinstance(provenance,str):
            try:provenance=json.loads(provenance)
            except (ValueError,TypeError):provenance={}
        if isinstance(provenance,dict):result['source_association_policy']=provenance.get('descriptor',{}).get('association_policy')
        return result
    return json.dumps(dict(label=material['label'],paths=[path_contract(p)for p in material['paths']]),sort_keys=True,separators=(',',':'))

def _definition_changes(a,b,ma,mb):
    """Sparse source definitions plus vectorized compact path masks, no raw dicts."""
    changed=np.zeros(a.n,bool)
    for field in ('queries','row_domains','allowed','reviewed_paths'):
        da=getattr(a,field);db=getattr(b,field)
        for key in set(da)|set(db):
            if da.get(key)!=db.get(key):changed[int(key)]=True
    def masks(inputs):
        return inputs.dense_paths
    x,y=masks(a),masks(b)
    if x is not None or y is not None:
        if x is None or y is None or x[:2]!=y[:2]:
            for item in (x,y):
                if item:changed[item[0]:item[0]+item[1]]=True
        else:changed[x[0]:x[0]+x[1]]|=np.asarray(x[2])!=np.asarray(y[2])
    # Material numbers may compact; the legal path IDs are the authority.
    def pools(inputs,model):
        out={}
        for start,count,pool in inputs.native_path_pools:
            entries={model['materials'][mi]['label']:sorted(model['materials'][mi]['paths'][pi]['id']for pi in paths)for mi,paths in pool.items()if 0<=mi<len(model['materials'])}
            out[(start,count)]=entries
        return out
    pa,pb=pools(a,ma),pools(b,mb)
    for interval in set(pa)|set(pb):
        if pa.get(interval)==pb.get(interval):continue
        lo,count=interval;hi=lo+count
        left,right=pa.get(interval,{}),pb.get(interval,{})
        labels={k for k in set(left)|set(right)if left.get(k)!=right.get(k)}
        for inputs,model in ((a,ma),(b,mb)):
            ids=[i for i,m in enumerate(model['materials'])if m['label']in labels]
            changed[lo:hi]|=np.isin(inputs.arrays['material_index'][lo:hi],ids)
    return changed

def _geometry_contract_sha():
    import inspect
    from .joint_model import as_event_view,native_camera_from_joint
    functions=(ReviewScene,station_intervals,_merge_intervals,_project,projected_curve,nearest_segments,CurrentJointCameras,fast_native_project,render_visibility,as_event_view,native_camera_from_joint)
    payload='\n'.join(inspect.getsource(f)for f in functions)
    for module_file in (Path(__file__).resolve().parents[1]/'rinkfit_start/native_scene.py',Path(__file__).resolve().parents[1]/'rinkfit_start/native_camera.py'):payload+='\n'+file_ref(module_file)['sha256']
    return hashlib.sha256(payload.encode('utf8')).hexdigest()

def _compare_source_transition(baseline_report,candidate_report,output,a,b,threshold,complete):
    if not math.isfinite(threshold)or threshold<0:raise ValueError('Invalid comparison threshold')
    before,after=_json(baseline_report),_json(candidate_report);output=Path(output)
    if output.exists()and any(output.iterdir()):raise ValueError('Refuse to replace an existing source-transition comparison')
    output.mkdir(parents=True,exist_ok=True)
    for report,inputs in ((before,a),(after,b)):
        if report['schema']!='rinkfit.dense_visible_metrics.v1' or report['raw_rows']!=inputs.n or report['source_packet']['sha256']!=inputs.packet_ref['sha256']:raise ValueError('Explicit comparison inputs do not bind their metric report')
        for key in ('source_packet','model'):_checked_file(report[key])
        if report.get('current_input_v1') and (report['current_input_v1']['sha256']!=(inputs.current_input_ref or {}).get('sha256')):raise ValueError('Explicit comparison current membership binding differs')
        if report.get('selection'):_checked_file(report['selection'])
        if report.get('nuisance_manifest'):_checked_file(report['nuisance_manifest'])
        if complete and(not report['complete_raw_domain_processed']or report['processed_rows']!=inputs.n):raise ValueError('Incomplete declared source metric scope')
    source_check=_check_comparison_source(a,b,output)
    if before['model_source_clock_sha256']!=after['model_source_clock_sha256']:raise ValueError('Same-source comparison clock definition changed')
    ma,mb=_json(before['model']['file']),_json(after['model']['file']);aa,bb=_metric_arrays(before),_metric_arrays(after)
    ca=[_material_contract(ma,i)for i in range(len(ma['materials']))];cb=[_material_contract(mb,i)for i in range(len(mb['materials']))];codes={v:i for i,v in enumerate(sorted(set(ca+cb)))}
    ac=np.array([codes[v]for v in ca]+[-1],np.int64);bc=np.array([codes[v]for v in cb]+[-1],np.int64)
    source_definition_changed=_definition_changes(a,b,ma,mb)
    same_math=before.get('geometry_operator_contract_sha256')==after.get('geometry_operator_contract_sha256') and before.get('geometry_operator_contract_sha256')is not None
    if not same_math:same_math={k:v['sha256']for k,v in before['operator_source_files'].items()}=={k:v['sha256']for k,v in after['operator_source_files'].items()}
    same_math=same_math and before['sampling']==after['sampling']and before['operator_codes']==after['operator_codes']
    names=('newly_invalid','recovered','still_invalid','worsened','improved','common_valid','common_geometry_scope','retired','target_changed','source_definition_changed','qualification_changed','operator_changed','scope_changed','baseline_unknown_or_withheld','candidate_unknown_or_withheld','baseline_revoked','candidate_revoked')
    streams={k:(output/(k+'.u64')).open('wb')for k in names};counts=collections.Counter();roles={side:{name:collections.Counter()for name in ('physical','training','fit','inspection')}for side in ('baseline','candidate')}
    geometry_scope=np.memmap(output/'common_geometry_scope.u8',mode='w+',dtype='u1',shape=(a.n,));geometry_scope[:]=0
    sse0=sse1=0.;maxreg=0.;changes=output/'source_transition_rows.jsonl';revoked_a={i for i,v in a.source_verdicts.items()if v['decision']in ('rejected','revoked')};revoked_b={i for i,v in b.source_verdicts.items()if v['decision']in ('rejected','revoked')}
    try:
        with changes.open('w',encoding='utf8')as log:
            for lo in range(0,a.n,65536):
                hi=min(a.n,lo+65536);sa,sb=aa['status'][lo:hi],bb['status'][lo:hi];pa=(sa&PROCESSED)!=0;pb=(sb&PROCESSED)!=0;scope=pa&pb
                if complete and not np.all(scope):raise ValueError('Incomplete processed comparison scope')
                mia,mib=np.asarray(a.arrays['material_index'][lo:hi]),np.asarray(b.arrays['material_index'][lo:hi]);ra=a.retired[lo:hi];rb=b.retired[lo:hi]
                if np.any((mia>=len(ca))&~ra)or np.any((mib>=len(cb))&~rb):raise ValueError('Non-retired comparison target is outside model')
                da,db=aa['domain'][lo:hi],bb['domain'][lo:hi]
                if not np.array_equal(da,a.domain[lo:hi])or not np.array_equal(db,b.domain[lo:hi]):raise ValueError('Metric membership differs from its explicit current input')
                target=ac[np.where(ra,len(ca),mia)]!=bc[np.where(rb,len(cb),mib)];definition=source_definition_changed[lo:hi];qualification=da!=db;operator=(aa['operator'][lo:hi]!=bb['operator'][lo:hi])|np.logical_not(same_math)
                comparable=scope&~(ra|rb|target|definition|qualification|operator)&((da&PHYSICAL)!=0)&((db&PHYSICAL)!=0)
                geometry_scope[lo:hi]=comparable
                va,vb=valid_status(sa),valid_status(sb);d0=aa['distance'][lo:hi].astype(float);d1=bb['distance'][lo:hi].astype(float)
                for valid,buf,distance in ((va,aa,d0),(vb,bb,d1)):
                    invalid=~np.isfinite(buf['nearest_uv'][lo:hi]).all(axis=1)|~np.isfinite(buf['station'][lo:hi])|(buf['station'][lo:hi]<0)|(buf['station'][lo:hi]>1)|(buf['path'][lo:hi]<0)|~np.isfinite(distance)|(distance<0)
                    if np.any(valid&invalid):raise ValueError('Visual-valid metric contains invalid numerical values')
                common=comparable&va&vb;ix=np.arange(lo,hi);masks=dict(newly_invalid=comparable&va&~vb,recovered=comparable&~va&vb,still_invalid=comparable&~va&~vb,worsened=common&(d1>d0+threshold),improved=common&(d1<d0-threshold),common_valid=common,common_geometry_scope=comparable,retired=scope&(ra|rb),target_changed=scope&target,source_definition_changed=scope&definition,qualification_changed=scope&qualification,operator_changed=scope&operator,scope_changed=pa^pb,baseline_unknown_or_withheld=scope&((da&PHYSICAL)==0),candidate_unknown_or_withheld=scope&((db&PHYSICAL)==0),baseline_revoked=np.isin(ix,list(revoked_a)),candidate_revoked=np.isin(ix,list(revoked_b)))
                for name,mask in masks.items():
                    rows=(np.flatnonzero(mask)+lo).astype('<u8');streams[name].write(rows.tobytes());counts[name]+=len(rows)
                    for side,domain in (('baseline',da),('candidate',db)):
                        for role,bit in (('physical',PHYSICAL),('training',TRAINING),('fit',FIT),('inspection',INSPECTION)):roles[side][role][name]+=int(np.count_nonzero(mask&((domain&bit)!=0)))
                for offset in np.flatnonzero(scope&(ra|rb|target|definition|qualification)):
                    row=lo+int(offset)
                    def side(inputs,model,mi,retired,domain):return dict(material_index=int(mi),target_label=None if retired else model['materials'][int(mi)]['label'],retired=bool(retired),domain_bits=int(domain),source_decision=inputs.source_verdicts.get(row,{}).get('decision'),visual_status=int((aa if inputs is a else bb)['status'][row]))
                    record=dict(raw_row=row,source_row_id=a.source_id(row),source_uv=a.arrays['uv'][row].tolist(),native_pts=int(a.arrays['native_pts'][row]),shot_id=a.shot_ids[int(a.arrays['shot_index'][row])],source_material_index=int(a.source_material_index[row]),baseline=side(a,ma,mia[offset],ra[offset],da[offset]),candidate=side(b,mb,mib[offset],rb[offset],db[offset]),retired=bool(ra[offset]or rb[offset]),target_changed=bool(target[offset]),source_definition_changed=bool(definition[offset]),qualification_changed=bool(qualification[offset]),operator_changed=bool(operator[offset]),counted_as_geometry_improvement=False)
                    log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                counts['raw_rows']+=hi-lo;counts['both_processed']+=int(scope.sum());sse0+=float(d0[common]@d0[common]);sse1+=float(d1[common]@d1[common])
                if common.any():maxreg=max(maxreg,float(np.max(d1[common]-d0[common])))
    finally:
        for stream in streams.values():stream.close()
    report=dict(schema='rinkfit.dense_visible_source_transition_comparison.v1',baseline=file_ref(baseline_report),candidate=file_ref(candidate_report),baseline_model=before['model'],candidate_model=after['model'],baseline_source_packet=a.packet_ref,candidate_source_packet=b.packet_ref,baseline_current_input_v1=a.current_input_ref,candidate_current_input_v1=b.current_input_ref,source_identity=source_check,raw_rows=a.n,raw_denominator_preserved=True,full_raw_domain_compared=bool(before['complete_raw_domain_processed']and after['complete_raw_domain_processed']),counts=dict(counts),counts_by_domain={side:{role:dict(value)for role,value in groups.items()}for side,groups in roles.items()},same_geometry_operator=same_math,geometry_comparison_requires_same_source_definition_qualification_operator=True,definition_or_qualification_change_counted_as_improvement=False,change_set_counts_may_overlap=True,same_valid_sum_squared_before_px=sse0,same_valid_sum_squared_after_px=sse1,same_valid_sum_squared_delta_px=sse1-sse0,maximum_distance_increase_px=maxreg,worsened_threshold_px=threshold,source_noise_W_objective_compared=False,formal_pending_is_not_formal_valid=True,quality_accepted=False,transition_rows=file_ref(changes),sets={k:dict(file_ref(v.name),dtype='<u8',shape=[int(counts[k])])for k,v in streams.items()})
    geometry_scope.flush()
    report['common_geometry_scope_mask']=dict(file_ref(geometry_scope.filename),dtype='|u1',shape=[a.n])
    report['geometry_comparison_requires_both_physical_source_qualifications']=True
    report['geometry_failure_sets_complete_for_common_scope']=True;report['transition_sets_complete_for_declared_scope']=True
    report['baseline_operator_source_files']=before['operator_source_files'];report['candidate_operator_source_files']=after['operator_source_files']
    report['baseline_sampling']=before['sampling'];report['candidate_sampling']=after['sampling']
    report['baseline_entity_metrics']=before.get('entity_metrics');report['candidate_entity_metrics']=after.get('entity_metrics')
    report['entity_comparison'],report['automatic_candidates']=_entity_comparison(a,aa,bb,output,threshold,comparable_rows=geometry_scope)
    _write(output/'comparison.json',report);return report


def _entity_comparison(inputs,aa,bb,output,threshold,comparable_rows=None):
    a=inputs.arrays;n=inputs.n;allrows=np.arange(n,dtype=np.int64)
    order=allrows[np.lexsort((a['material_index'],a['native_pts'],a['shot_index']))]
    keys=np.c_[a['shot_index'][order],a['native_pts'][order],a['material_index'][order]]
    cuts=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))+1,n]
    labels={int(m['index']):m['label'] for m in inputs.packet.get('material_binding',{}).get('materials',[])}
    shot_ids=inputs.packet.get('shot_ids') or ['s01','s02','s03','s04','s05','s06'];automatic={}
    path=Path(output)/'entity_comparison.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            rows=order[lo:hi];processed=(aa['status'][rows]&PROCESSED)!=0
            if comparable_rows is not None:processed&=np.asarray(comparable_rows[rows])!=0
            if not processed.any():continue
            rows=rows[processed];va=valid_status(aa['status'][rows]);vb=valid_status(bb['status'][rows]);common=va&vb
            d0=aa['distance'][rows].astype(float);d1=bb['distance'][rows].astype(float);domain=aa['domain'][rows]
            sid=shot_ids[int(a['shot_index'][rows[0]])];pts=int(a['native_pts'][rows[0]]);mi=int(a['material_index'][rows[0]])
            record={'shot_id':sid,'native_pts':pts,'material_index':mi,'label':labels.get(mi,str(mi)),'rows':len(rows),'before_visual_valid':int(va.sum()),'after_visual_valid':int(vb.sum()),'newly_invalid':int(np.count_nonzero(va&~vb)),'recovered':int(np.count_nonzero(~va&vb)),'common_valid':int(common.sum()),'worsened':int(np.count_nonzero(common&(d1>d0+threshold))),'same_valid_sse_delta_px':float(d1[common]@d1[common]-d0[common]@d0[common]),'after_p95_px':float(np.quantile(d1[vb],.95)) if vb.any() else None,'maximum_increase_px':float(np.max(d1[common]-d0[common])) if common.any() else None,'counts_by_domain':{}}
            for role,bit in [('physical',PHYSICAL),('training',TRAINING),('fit',FIT),('inspection',INSPECTION)]:
                q=(domain&bit)!=0;record['counts_by_domain'][role]={'rows':int(q.sum()),'newly_invalid':int(np.count_nonzero(q&va&~vb)),'recovered':int(np.count_nonzero(q&~va&vb)),'after_invalid':int(np.count_nonzero(q&~vb)),'worsened':int(np.count_nonzero(q&common&(d1>d0+threshold)))}
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            for j,r in enumerate(rows):
                # Keep training/fit source evidence ahead of unqualified source
                # diagnostics; every domain still has a complete comparison log.
                role=2 if domain[j]&FIT else (1 if domain[j]&PHYSICAL else 0)
                newly_invalid=bool(va[j] and not vb[j]);increase=float(d1[j]-d0[j]) if common[j] else -math.inf
                absolute=float(d1[j]) if vb[j] else -math.inf
                score=(role,2 if newly_invalid else (1 if not vb[j] else 0),max(increase,0.),absolute)
                old=automatic.get(sid)
                if old is None or tuple(old['_score'])<score:
                    reason='newly_invalid' if newly_invalid else ('persistent_invalid' if not vb[j] else ('maximum_distance_increase' if increase>threshold else 'candidate_maximum_distance'))
                    automatic[sid]={'native_pts':pts,'raw_row':int(r),'material_index':mi,'reason':reason,'domain_bits':int(domain[j]),'before_distance_px':float(d0[j]) if va[j] else None,'candidate_distance_px':float(d1[j]) if vb[j] else None,'_score':list(score)}
    for value in automatic.values():value.pop('_score')
    return file_ref(path),automatic

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--model',required=True);e.add_argument('--packet',required=True);e.add_argument('--selection');e.add_argument('--current-input');e.add_argument('--nuisance');e.add_argument('--out',required=True);e.add_argument('--chord-tolerance-px',type=float,default=.05)
    c=sub.add_parser('compare');c.add_argument('--baseline-report',required=True);c.add_argument('--candidate-report',required=True);c.add_argument('--packet');c.add_argument('--selection');c.add_argument('--baseline-packet');c.add_argument('--baseline-selection');c.add_argument('--baseline-current-input');c.add_argument('--candidate-packet');c.add_argument('--candidate-selection');c.add_argument('--candidate-current-input');c.add_argument('--out',required=True);c.add_argument('--worsened-threshold-px',type=float,default=0.)
    args=p.parse_args(argv)
    if args.command=='evaluate':
        if not args.current_input and not args.selection:p.error('evaluate requires --selection or --current-input')
        last=[0.]
        def progress(v):
            now=time.perf_counter()
            if now-last[0]>=5 or v['processed']==v['selected']:
                print(json.dumps({'schema':'rinkfit.visible_metric_progress.v1',**v}),flush=True);last[0]=now
        result=evaluate(args.model,args.packet,args.selection,args.out,nuisance_manifest=args.nuisance,chord_tolerance_px=args.chord_tolerance_px,progress=progress,current_input=args.current_input)
    else:
        if args.baseline_packet or args.candidate_packet:
            if not args.baseline_packet or not args.candidate_packet:p.error('Both comparison packet sides are required')
            for side in ('baseline','candidate'):
                if not getattr(args,side+'_current_input')and not getattr(args,side+'_selection'):p.error('Each comparison side requires its selection or current manifest')
            baseline=ReviewInputs(args.baseline_packet,args.baseline_selection,current_input=args.baseline_current_input)
            candidate=ReviewInputs(args.candidate_packet,args.candidate_selection,current_input=args.candidate_current_input)
            result=compare(args.baseline_report,args.candidate_report,args.out,baseline_inputs=baseline,candidate_inputs=candidate,worsened_threshold_px=args.worsened_threshold_px)
        else:
            if not args.packet or not args.selection:p.error('Legacy comparison requires --packet and --selection')
            inputs=ReviewInputs(args.packet,args.selection)
            result=compare(args.baseline_report,args.candidate_report,args.out,inputs=inputs,worsened_threshold_px=args.worsened_threshold_px)
    print(json.dumps({'schema':result['schema'],'output':str(Path(args.out).resolve()),'quality_accepted':False}),flush=True)
if __name__=='__main__':main()
