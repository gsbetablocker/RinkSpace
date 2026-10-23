"""One shared scene, rig/optics blocks and shot-local native cubic controls."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
from .camera_graph import CameraGraph, SHOT_TO_RIG, REFERENCE_SHOTS, GraphError
SCHEMA='rinkfit.joint_event_model.v1'
def _slice(b,lo,hi,name):
    q={k:copy.deepcopy(b[k][lo:hi]) for k in ('values','lower','upper','scales') if k in b}
    q.update(name=name,constant=b.get('constant',False),fixed_indices=[x-lo for x in b.get('fixed_indices',[]) if lo<=x<hi]);return q
def _tangents(n):
    n=np.asarray(n,float);u=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,u);u/=np.linalg.norm(u);return u.tolist(),np.cross(n,u).tolist()
def build_joint_model(graph,s01_model,*,graph_ref=None,s01_ref=None,require_ready=False):
    cg=CameraGraph(graph);cg.validate(require_ready=require_ready)
    if s01_model['schema']!='rinkfit.event_model.v1.1':raise GraphError('explicit latest s01 adapter required')
    old=s01_model['blocks'];cam=s01_model['camera'];blocks=[]
    def add(b):
        i=len(blocks);blocks.append(b);return i
    scene=add(_slice(old[0],0,5,'scene/rink5'))
    rigs={};shots={};mapping={}
    for rid,ref in REFERENCE_SHOTS.items():
        g=graph['rigs'][rid]
        if any(g.get(k) is None for k in ('shared_head','optics','rolling_shutter')):continue
        if rid=='main':
            hb=_slice(old[0],5,20,'rig/'+rid+'/head15');ob=_slice(old[0],20,32,'rig/'+rid+'/optics12');meta={k:copy.deepcopy(cam[k]) for k in ('R0_seed','n_seed','b_seed','n_tangent1','n_tangent2','f_ref','radius_max','tau_bound')}
        else:
            h=g['shared_head'];o=g['optics'];gv=copy.deepcopy(old[0]);vals=[0.]*6+list(h['P'])+list(h['d'])+[h.get('a',0.),h.get('pupil_linear',0.),h.get('pupil_quadratic',0.)]
            gv['values'][5:20]=vals;hb=_slice(gv,5,20,'rig/'+rid+'/head15')
            op=list(o['principal_point'])+[o['eta']]+list(o['principal_linear'])+list(o['k1'])+list(o['k2'])+[o['p1'],o['p2'],g['rolling_shutter']['signed_readout_seconds']]
            gv['values'][20:32]=op;ob=_slice(gv,20,32,'rig/'+rid+'/optics12');u,v=_tangents(h['n'])
            meta=dict(R0_seed=np.asarray(h['R0']).reshape(-1).tolist(),n_seed=h['n'],b_seed=h['b'],n_tangent1=u,n_tangent2=v,f_ref=o['f_ref'],radius_max=o['radius_max'],tau_bound=g.get('tau_bound',.02))
        if abs(ob['values'][11])>meta['tau_bound']:raise GraphError('readout outside immutable allowed domain')
        rigs[rid]=dict(head_block=add(hb),optics_block=add(ob),camera=meta,reference_shot=ref)
    # Non-camera shared entities are registered once, retaining bounds and fixed chart.
    control_ids=set(range(cam['control_block_begin'],cam['control_block_begin']+cam['n_controls']))
    for i,b in enumerate(old):
        if i and i not in control_ids:mapping[i]=add(copy.deepcopy(b))
    materials=copy.deepcopy(s01_model['materials'])
    for material in materials:
        for path in material['paths']:
            for key in ('geometry_block','advertisement_block'):
                if path.get(key,-1)>=0:path[key]=mapping[path[key]]
    for sid,rid in SHOT_TO_RIG.items():
        s=graph['shots'][sid];cc=s.get('camera_controls')
        if rid not in rigs or cc is None or s.get('reference_angles') is None:continue
        a=np.asarray(s['reference_angles'],float);ref=sid==REFERENCE_SHOTS[rid]
        angles=add(dict(name='shot/'+sid+'/reference_angles',values=a.tolist(),lower=[-20.]*2,upper=[20.]*2,scales=[.1]*2,constant=ref,fixed_indices=[]))
        basis=cc['basis'];ct=np.asarray(cc['controls'],float);n=int(basis['n_controls'])
        if ct.shape!=(n,3):raise GraphError('control array mismatch')
        dt=float(basis['dt']);first=float(basis['first_control_time']);duration=(s['scope']['last_pts']-s['scope']['first_pts'])/12800.;half=rigs[rid]['camera']['tau_bound']/2
        if first+dt> -half+1e-12 or first+(n-2)*dt<duration+half-1e-12:raise GraphError(sid+': full allowed RS halo missing')
        if sid=='s01':
            inds=cam['reference_indices'];weights=cam['reference_weights'];dep=cam['dependent_control']
        else:
            x=-first/dt;cell=int(np.floor(x));u=x-cell;inds=list(range(cell-1,cell+3));weights=[(1-u)**3/6,(3*u**3-6*u*u+4)/6,(-3*u**3+3*u*u+3*u+1)/6,u**3/6];dep=inds[int(np.argmax(np.abs(weights)))]
            residual=np.dot(np.asarray(weights),ct[inds,:2])
            if np.max(np.abs(residual))>1e-8:raise GraphError(sid+': local reference controls must have zero yaw/pitch at t=0; move offset into reference_angles')
        cblocks=[]
        for j in range(n):
            if sid=='s01':b=copy.deepcopy(old[cam['control_block_begin']+j])
            else:b=dict(values=ct[j].tolist(),lower=[-20.,-20.,np.log(100.)],upper=[20.,20.,np.log(1e6)],scales=[.1,.1,1.],constant=False)
            b['name']='shot/'+sid+'/control/'+str(j);b['fixed_indices']=[0,1] if j==dep else [];cblocks.append(add(b))
        shots[sid]=dict(rig_id=rid,scope=copy.deepcopy(s['scope']),time_origin_pts=s['time_origin_pts'],reference_angles_block=angles,control_blocks=cblocks,camera=dict(dt=dt,first_control_time=first,n_controls=n,dependent_control=dep,reference_indices=inds,reference_weights=weights))
    linear_manifolds=[]
    declared=copy.deepcopy(s01_model.get('provenance',{}).get('time_subspace',{}))
    if declared:
        if declared.get('version')!='rinkfit.exact_adaptive_cubic_knot_removal.v1':raise GraphError('time subfamily needs an explicit joint linear chart adapter')
        from scipy.linalg import null_space
        removed=np.asarray(declared['basis']['removed_fine_indices'],int);first=int(removed.min()-2);last=int(removed.max()+2);indices=np.arange(first,last+1)
        if first<=cam['dependent_control']<=last:raise GraphError('local knot chart overlaps reference gauge; explicit combined manifold needed')
        C=np.zeros((len(removed),len(indices)))
        for k,r in enumerate(removed):C[k,r-first+np.arange(-2,3)]=[1.,-4.,6.,-4.,1.]
        N=null_space(C);B=np.kron(N,np.eye(3));control_ids=shots['s01']['control_blocks'];values=np.asarray([blocks[control_ids[i]]['values'] for i in indices]);membership=float(np.max(np.abs(C@values)))
        if membership>2e-10:raise GraphError('latest s01 is outside declared local time subfamily')
        group={k:np.asarray([blocks[control_ids[i]][k] for i in indices]).reshape(-1).tolist() for k in ('values','lower','upper','scales')};group.update(name='shot/s01/control_group/local_m4',constant=False,fixed_indices=[]);group_id=add(group)
        coordinates=[dict(block=k,offset=0) for k in control_ids]
        for i in indices:coordinates[i]=dict(block=group_id,offset=3*int(i-first));blocks[control_ids[i]]['constant']=True
        shots['s01']['control_coordinates']=coordinates
        linear_manifolds.append(dict(block=group_id,ambient_size=int(B.shape[0]),tangent_size=int(B.shape[1]),basis=B.tolist(),constraint_matrix=np.kron(C,np.eye(3)).tolist(),membership_error=membership,storage_control_indices=indices.tolist(),source_time_subspace=declared,not_motion_regularization=True))
    names=[b['name'] for b in blocks]
    if len(names)!=len(set(names)):raise GraphError('physical parameter block registered twice')
    for b in blocks:
        v=np.asarray(b['values']);lo=np.asarray(b['lower']);hi=np.asarray(b['upper']);sc=np.asarray(b['scales'])
        if not (np.isfinite(v).all() and np.all(v>=lo) and np.all(v<=hi) and np.all(sc>0)):raise GraphError('invalid joint parameter domain '+b['name'])
    return dict(schema=SCHEMA,source_clock=copy.deepcopy(graph['source_clock']),coordinate_gauge=copy.deepcopy(graph['coordinate_gauge']),scene_block=scene,blocks=blocks,materials=materials,rigs=rigs,shots=shots,s01_block_map={str(k):v for k,v in mapping.items()},s01_model_sha256=hashlib.sha256(json.dumps(s01_model).encode()).hexdigest(),source_s01_model_ref=s01_ref,source_graph_ref=graph_ref,linear_control_manifolds=linear_manifolds,s01_time_subspace_provenance=declared,physical_parameter_ownership='scene5+3_head15+3_optics12+shot_controls+six_reference_angles',source_identity=s01_model.get('source_identity'),print_only_chart_policy='inherited legal witness held during each local solve; all six print coefficients remain variable; explicit original-box inner approximation requiring full common-witness decoder for unrestricted trial domain',certificates=dict(joint_fit_complete=False,formal_engineering_budget_passed=False))

def append_confirmed_unobserved_connection(model,scene_file,*,feature_id='house_connection_F1_R1',fact_reference=None):
    """Restore a confirmed carrier, never promote unverified pixels or its endpoint."""
    result=copy.deepcopy(model);source=Path(scene_file);scene=json.loads(source.read_text(encoding='utf8'));matches=[f for f in scene['ice_features'] if f.get('id')==feature_id]
    if len(matches)!=1:raise GraphError('confirmed connection source missing or ambiguous')
    feature=matches[0];label='ice.'+feature_id+':segment0'
    if any(m['label']==label for m in result['materials']):return result,dict(already_present=True,label=label)
    bounds=np.asarray(scene['parameters']['bounds'],float);length=float(bounds[2]-bounds[0]);origin=(bounds[:2]+bounds[2:])/2
    if not length>0:raise GraphError('conditional source scene length invalid')
    segment=np.asarray(feature['segments_xyz'][0],float)
    if segment.shape!=(2,3) or np.max(np.abs(segment[:,2]))>1e-10:raise GraphError('connection source must be one finite ice segment')
    value=((segment[:,:2]-origin)/length).reshape(4).tolist()
    exemplar=next(p for m in result['materials'] if m['label']=='ice.house_connection_F2_R2:segment0' for p in m['paths']);block=copy.deepcopy(result['blocks'][exemplar['geometry_block']]);block.update(name='ice/line/unobserved_'+feature_id,values=value,constant=True,fixed_indices=[])
    bi=len(result['blocks']);mi=len(result['materials']);result['blocks'].append(block)
    proof=dict(source_scene=dict(path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),feature_id=feature_id,conditional_full_length=length,origin=origin.tolist(),source_units_are_not_verified_metres=True,user_identity_fact=fact_reference,source_feature=feature,current_native_observation_count=0,geometry_is_initializer_not_fitted=True,terminal_extent_not_video_measured=True,fixed_only_while_unobserved_not_an_optimization_prior=True,pixel_observations_created=0)
    path=dict(id='confirmed_unobserved/'+feature_id,variant_id='confirmed_identity_unobserved_carrier',kind=0,geometry_block=bi,advertisement_block=-1,physical_point=False,closed=False,provenance=json.dumps(proof,ensure_ascii=False))
    result['materials'].append(dict(label=label,paths=[path],unobserved_confirmed_carrier=True));result['confirmed_unobserved_connection_restoration']=proof
    return result,dict(schema='rinkfit.confirmed_unobserved_connection_restoration.v1',label=label,material_index=mi,block_index=bi,source=proof,old_material_indices_and_values_unchanged=True)

def restore_structural_parameter_flags(model,reference_model,*,reference_ref=None):
    """Restore original physical freedoms by name without replacing any numeric value."""
    result=copy.deepcopy(model);original={b['name']:b for b in reference_model['blocks']};restored=[]
    for block in result['blocks']:
        before=original.get(block['name'])
        if before is None:continue
        if len(block['values'])!=len(before['values']):raise GraphError('flag restoration block shape changed: '+block['name'])
        block['constant']=bool(before.get('constant',False));block['fixed_indices']=copy.deepcopy(before.get('fixed_indices',[]));restored.append(block['name'])
    result['temporary_initialization_flags_restored']=dict(reference=reference_ref,restored_blocks=restored,numeric_values_replaced=False,new_material_block_flags_preserved=True)
    return result

def as_event_view(model,shot_id):
    """Read-only physical view for independent native replay; no parameter copies optimize."""
    shot=model['shots'][shot_id];rig=model['rigs'][shot['rig_id']];b=copy.deepcopy(model['blocks']);g=[]
    for key in ('scene_block',):g+=b[model[key]]['values']
    g+=b[rig['head_block']]['values']+b[rig['optics_block']]['values']
    for i,c in enumerate(shot.get('control_coordinates',[])):
        storage=shot['control_blocks'][i];b[storage]['values']=b[c['block']]['values'][c['offset']:c['offset']+3]
    original=b[model['scene_block']];full={k:[] for k in ('lower','upper','scales')}
    for k in full:full[k]=original[k]+b[rig['head_block']][k]+b[rig['optics_block']][k]
    b[0]=dict(name='global32',values=g,constant=False,**full)
    cam=copy.deepcopy(rig['camera']);cam.update(shot['camera']);ids=shot['control_blocks']
    if ids!=list(range(ids[0],ids[0]+len(ids))):raise GraphError('event replay view needs contiguous serialized local controls')
    cam['control_block_begin']=ids[0]
    materials=copy.deepcopy(model['materials'])
    for material in materials:
        for path in material['paths']:
            if path['kind']==3 and path.get('geometry_block',-1)>=0:
                inds=path.get('geometry_indices',[]);idx=inds[0] if inds else int(path.get('geometry_offset',0))
                path.update(fixed_height_fraction=b[path['geometry_block']]['values'][idx],geometry_block=-1)
            if path['kind']==4:
                q=b[path['geometry_block']]['values'];path.update(kind=1,start_angle=q[3],end_angle=q[3]+np.pi,radius_index=2)
            elif path['kind']==5:
                q=b[path['geometry_block']]['values'];v=np.asarray(q[:2]);d=q[2]*np.asarray([np.cos(q[3]),np.sin(q[3])]);path.update(kind=0,geometry_block=len(b));b.append(dict(name='readonly/'+path['id'],values=np.r_[v+d,v-d].tolist(),lower=[-2.]*4,upper=[2.]*4,scales=[.5]*4,constant=True))
            if path.get('advertisement_block',-1)>=0 and (path.get('advertisement_offset',0) or len(b[path['advertisement_block']]['values'])!=4):
                off=int(path.get('advertisement_offset',0));original=b[path['advertisement_block']]
                q={k:original[k][off:off+4] for k in ('values','lower','upper','scales') if k in original};q.update(name='readonly/ad/'+path['id'],constant=True)
                if len(q['values'])!=4:raise GraphError('invalid shared advertisement instance offset')
                path['advertisement_block']=len(b);path['advertisement_offset']=0;b.append(q)
            off=int(path.get('geometry_offset',0));inds=path.get('geometry_indices',[])
            if path.get('geometry_block',-1)>=0 and (off or inds):
                original=b[path['geometry_block']];inds=inds or list(range(off,off+4));q={k:[original[k][i] for i in inds] for k in ('values','lower','upper','scales')};q.update(name='readonly/'+path['id'],constant=True);path['geometry_block']=len(b);b.append(q)
    return dict(schema='rinkfit.event_model.v1.1',model_id='joint_readonly_view/'+shot_id,source_identity=model.get('source_identity','joint'),blocks=b,camera=cam,materials=materials)
def graph_from_joint(model):
    from rinkfit_start.native_scene import camera_from_event
    source=model.get('source_graph_ref')
    if not source:raise GraphError('source graph reference missing')
    g=json.loads(Path(source).read_text(encoding='utf8'))
    for sid,shot in model['shots'].items():
        native=camera_from_event(as_event_view(model,sid)).source;rid=shot['rig_id']
        g['rigs'][rid].update(shared_head=native['shared_head'],optics=native['optics'],rolling_shutter=native['rolling_shutter'],status='joint_candidate')
        g['shots'][sid].update(camera_controls=native['camera_controls'],reference_angles=copy.deepcopy(model['blocks'][shot['reference_angles_block']]['values']),status='joint_candidate')
    g['certificates']=dict(joint_fit_complete=False,all_source_replay_complete=False);g['cache_generation']=int(g.get('cache_generation',0))+1
    return g

def validate_physical_domain(model,tolerance=1e-9):
    from rinkfit_start.native_scene import NativeScene
    if not model['shots']:raise GraphError('no initialized shot')
    sid=next(iter(model['shots']));result=NativeScene(as_event_view(model,sid)).check_physics(tolerance=tolerance)
    # Fixed legacy witness q boxes form an explicit local inner approximation.
    # The inherited five print-only ad origins are not new physical freedoms.
    result['print_chart_trial_domain']='inherited legal witness with original local q boxes; inner approximation of original common-witness domain'
    return result

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['build']);ap.add_argument('--graph',required=True);ap.add_argument('--s01-model',required=True);ap.add_argument('--output',required=True);ap.add_argument('--require-ready',action='store_true');a=ap.parse_args(argv)
    gp=Path(a.graph);sp=Path(a.s01_model);g=json.loads(gp.read_text(encoding='utf8'));s=json.loads(sp.read_text(encoding='utf8'));m=build_joint_model(g,s,graph_ref=str(gp.resolve()),s01_ref=str(sp.resolve()),require_ready=a.require_ready);m['s01_model_sha256']=hashlib.sha256(sp.read_bytes()).hexdigest();op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(blocks=len(m['blocks']),shots=list(m['shots']),rigs=list(m['rigs']),materials=len(m['materials']))))
if __name__=='__main__':main()


def append_lane_relations(model, relation_file):
    """Restore declared Lane carriers without adding observations or unseen edges."""
    q=copy.deepcopy(model); refpath=Path(relation_file).resolve()
    ref={'file':str(refpath),'sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),'field':'curling_structure.confirmed_layout_ties'}
    byname={b['name']:i for i,b in enumerate(q['blocks'])}
    name='ice/structure/Lane_shared_halfwidth_rightX'
    if name in byname:
        return q,{'idempotent':True,'carrier_block':byname[name]}
    entries=[]; widths=[]; right=[]
    for m in q['materials']:
        import re
        mt=re.fullmatch(r'ice[.]lane([1-4]):edge([02])',m['label'])
        if not mt: continue
        row=int(mt.group(1)); sign=1 if mt.group(2)=='0' else -1
        for path in m['paths']:
            if path['kind']!=0: raise ValueError('Lane long edge must be finite segment')
            bi=path['geometry_block']; ids=path.get('geometry_indices') or list(range(path.get('geometry_offset',0),path.get('geometry_offset',0)+4))
            if len(ids)!=4: raise ValueError('Lane segment coordinate count')
            b=q['blocks'][bi]; v=np.asarray(b['values'])[ids]; right_vertex=0 if v[0]>v[2] else 2
            cy=q['blocks'][byname[f'ice/circle/F{row}']]['values'][1]
            widths.append(sign*(np.mean(v[[1,3]])-cy));right.append(v[right_vertex]);entries.append((m['label'],row,sign,bi,ids,right_vertex))
    if not entries: raise ValueError('no observed finite Lane edges to bind')
    carrier=len(q['blocks']);q['blocks'].append({'name':name,'values':[float(np.mean(widths)),float(np.mean(right))], 'lower':[1e-6,-.5],'upper':[.25,.5],'scales':[.1,.5],'constant':False,'fixed_indices':[]})
    equalities=q.setdefault('physical_equalities',[]);existing={x['id'] for x in equalities}
    def add(rid,terms):
        if rid in existing: raise ValueError('duplicate Lane relation')
        equalities.append({'id':rid,'expression_kind':'linear','terms':[{'block':q['blocks'][b]['name'],'index':i,'coefficient':c} for b,i,c in terms], 'rhs':0.,'source_refs':[ref],'evaluation_scale':10000.})
    for label,row,sign,bi,ids,right_vertex in entries:
        house=byname[f'ice/circle/F{row}']
        for endpoint in (1,3): add(label+f'/endpoint{endpoint//2}_parallel_row_midline',[(bi,ids[endpoint],1),(house,1,-1),(carrier,0,-sign)])
        add(label+'/right_collinear',[(bi,ids[right_vertex],1),(carrier,1,-1)])
    report={'schema':'rinkfit.restored_lane_hard_relations.v1','carrier_block':carrier,'active_existing_long_edges':len(entries),'equalities_added':3*len(entries),'left_endpoints_independent':True,'equal_length_imposed':False,'new_observations_created':0,'unobserved_fourth_edge_created':False,'missing_existing_geometric_materials':['lane1:edge2','lane4:edge0',*[f'lane{i}:edge1' for i in range(1,5)]], 'finite_left_ends_are_support_not_measured_terminal_points':True,'source_ref':ref}
    q['Lane_relation_restoration']=report
    return q,report


def project_linear_physical_equalities(model):
    """Numerical seed projection only; all equalities remain in the actual PHR."""
    q=copy.deepcopy(model); names={b['name']:i for i,b in enumerate(q['blocks'])};defs=q.get('physical_equalities',[])
    variables=[]
    for d in defs:
        if d.get('expression_kind','linear')!='linear': raise ValueError('nonlinear seed relation requires its physical decoder')
        for t in d['terms']:
            b=names[t['block']] if isinstance(t['block'],str) else int(t['block']); i=int(t['index'])
            if not q['blocks'][b].get('constant',False) and i not in q['blocks'][b].get('fixed_indices',[]) and (b,i) not in variables: variables.append((b,i))
    lookup={v:i for i,v in enumerate(variables)};A=np.zeros((len(defs),len(variables)));rhs=[]
    for row,d in enumerate(defs):
        target=float(d['rhs'])
        for t in d['terms']:
            bi=names[t['block']] if isinstance(t['block'],str) else int(t['block']); i=int(t['index']); c=float(t['coefficient'])
            if (bi,i) in lookup:A[row,lookup[bi,i]]+=c
            else:target-=c*q['blocks'][bi]['values'][i]
        rhs.append(target)
    x=np.array([q['blocks'][b]['values'][i] for b,i in variables]); scales=np.array([q['blocks'][b]['scales'][i] for b,i in variables]);rhs=np.asarray(rhs)
    delta=np.linalg.lstsq(A*scales[None,:],rhs-A@x,rcond=None)[0]; result=x+scales*delta
    lower=np.array([q['blocks'][b]['lower'][i] for b,i in variables]);upper=np.array([q['blocks'][b]['upper'][i] for b,i in variables])
    if np.any(result<lower-1e-12) or np.any(result>upper+1e-12):raise ValueError('hard relation seed projection left original parameter domains; requires bounded exact QP, no clipping')
    residue=A@result-rhs
    if len(residue) and np.max(np.abs(residue))>1e-10:raise ValueError('incompatible fixed or redundant hard physical equalities')
    changes=[]
    for (b,i),old,new in zip(variables,x,result):
        q['blocks'][b]['values'][i]=float(new); changes.append({'block':q['blocks'][b]['name'],'index':i,'before':float(old),'after':float(new)})
    report={'schema':'rinkfit.hard_relation_numerical_seed_projection.v1','projection_role':'feasible numerical initialization, neither source evidence nor video fit','constraint_count':len(defs),'rank':int(np.linalg.matrix_rank(A)),'max_before_native_L':float(np.max(np.abs(A@x-rhs))) if len(defs) else 0.,'max_after_native_L':float(np.max(np.abs(residue))) if len(defs) else 0.,'weighted_distance_norm':float(np.linalg.norm(delta)),'changes':changes}
    return q,report


def apply_native_motion_patch(model,patch,*,source_model_sha256):
    """Apply SHA-bound navigation motion and exact physical-head reference rebase."""
    from scipy.spatial.transform import Rotation
    q=copy.deepcopy(model)
    if patch['source_model']['sha256']!=source_model_sha256:raise ValueError('motion patch belongs to another exact physical parent')
    graph_before=graph_from_joint(q);head_reports=[]
    for rid,d in patch.get('shared_head_reference_gauge_patch',{}).items():
        head=graph_before['rigs'][rid]['shared_head'];rig=q['rigs'][rid];block=q['blocks'][rig['head_block']];R0=np.asarray(d['R0_after'],float)
        if np.max(abs(np.asarray(head['R0'])-np.asarray(d['R0_before'])))>1e-12:raise ValueError('physical R0 patch parent mismatch')
        seed=np.asarray(rig['camera']['R0_seed'],float).reshape(3,3);delta=Rotation.from_matrix(R0@seed.T).as_rotvec();block['values'][:3]=delta.tolist()
        n=np.asarray(head['n']);b=np.asarray(head['b']);v=R0@n;bs=np.asarray(rig['camera']['b_seed'],float);e=bs-v*np.dot(bs,v);e/=np.linalg.norm(e);f=np.cross(v,e);block['values'][5]=float(np.arctan2(np.dot(b,f),np.dot(b,e)))
        head_reports.append({'rig':rid,'R0_delta_rotvec_after':delta.tolist(),'b_coordinate_rebased_to_preserve_physical_axis':block['values'][5]})
    for sid,d in patch['patch'].items():
        shot=q['shots'][sid];controls=np.asarray(d['camera_controls']['controls'],float);basis=d['camera_controls']['basis'];meta=shot['camera']
        if shot.get('control_coordinates'):raise ValueError('motion patch of reduced inherited time subspace needs its exact native transform')
        if controls.shape!=(len(shot['control_blocks']),3) or abs(basis['dt']-meta['dt'])>1e-12 or abs(basis['first_control_time']-meta['first_control_time'])>1e-12:raise ValueError('motion patch changed allocated cubic/halo')
        for b,value in zip(shot['control_blocks'],controls):q['blocks'][b]['values']=value.tolist()
        q['blocks'][shot['reference_angles_block']]['values']=list(d['reference_angles'])
    graph_after=graph_from_joint(q);cg=CameraGraph(graph_after);cg.validate(require_ready=True);maximum=0.
    for rid in patch.get('shared_head_reference_gauge_patch',{}):
        before=graph_before['rigs'][rid]['shared_head'];after=graph_after['rigs'][rid]['shared_head']
        for k in ('n','b','P','d','a','pupil_linear','pupil_quadratic'):
            maximum=max(maximum,float(np.max(abs(np.asarray(before[k])-np.asarray(after[k])))))
        if np.max(abs(np.asarray(after['R0'])-np.asarray(patch['shared_head_reference_gauge_patch'][rid]['R0_after'])))>1e-12:raise ValueError('physical R0 did not equal numerical gauge proposal')
    if maximum>1e-12:raise ValueError('navigation reference rebase changed physical axes/position/lever/pupil')
    report={'schema':'rinkfit.applied_native_motion_coordinate_seed.v1','source_model_sha256':source_model_sha256,'patched_shots':sorted(patch['patch']),'head_reference_rebases':head_reports,'maximum_other_head_physical_parameter_change':maximum,'source_UV_W_E_changed':False,'physical_calibration_complete':False,'new_user_geometry_equalities_added':False}
    q['optical_coordinate_initialization_patch']=report
    return q,report
