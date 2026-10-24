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
    result=dict(schema=SCHEMA,source_clock=copy.deepcopy(graph['source_clock']),coordinate_gauge=copy.deepcopy(graph['coordinate_gauge']),scene_block=scene,blocks=blocks,materials=materials,rigs=rigs,shots=shots,s01_block_map={str(k):v for k,v in mapping.items()},s01_model_sha256=hashlib.sha256(json.dumps(s01_model).encode()).hexdigest(),source_s01_model_ref=s01_ref,source_graph_ref=graph_ref,linear_control_manifolds=linear_manifolds,s01_time_subspace_provenance=declared,physical_parameter_ownership='scene5+3_head15+3_optics12+shot_controls+six_reference_angles',source_identity=s01_model.get('source_identity'),print_only_chart_policy='inherited legal witness held during each local solve; all six print coefficients remain variable; explicit original-box inner approximation requiring full common-witness decoder for unrestricted trial domain',certificates=dict(joint_fit_complete=False,formal_engineering_budget_passed=False))
    return revise_c0_midline_without_a(result)[0]

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
                original=b[path['geometry_block']];stop=len(original['values']) if path['kind'] in (6,7) else off+4;inds=inds or list(range(off,stop));q={k:[original[k][i] for i in inds] for k in ('values','lower','upper','scales')};q.update(name='readonly/'+path['id'],constant=True);path['geometry_block']=len(b);path['geometry_offset']=0;path.pop('geometry_indices',None);b.append(q)
    return dict(schema='rinkfit.event_model.v1.1',model_id='joint_readonly_view/'+shot_id,source_identity=model.get('source_identity','joint'),blocks=b,camera=cam,materials=materials)
def native_camera_from_joint(model, shot_id):
    """Exact read-only current camera, including the shot's reference angles.

    No historical graph initializer is required to consume a complete model.
    """
    from rinkfit_start.native_scene import camera_from_event
    from rinkfit_start.native_camera import NativeCamera, rotation
    source=copy.deepcopy(camera_from_event(as_event_view(model, shot_id)).source)
    offsets=np.asarray(model['blocks'][model['shots'][shot_id]['reference_angles_block']]['values'],float)
    if offsets.shape!=(2,) or not np.isfinite(offsets).all():raise GraphError('Current shot reference angles are incomplete')
    head=source['shared_head'];alpha,beta=offsets
    head['R0']=(rotation(np.asarray(head['b']),beta)@np.asarray(head['R0'])@rotation(np.asarray(head['n']),alpha)).tolist()
    return NativeCamera(source)

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
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['build','restore-lanes']);ap.add_argument('--graph');ap.add_argument('--s01-model');ap.add_argument('--model');ap.add_argument('--relations');ap.add_argument('--output',required=True);ap.add_argument('--require-ready',action='store_true');a=ap.parse_args(argv)
    if a.command=='restore-lanes':
        if not a.model or not a.relations:ap.error('restore-lanes requires --model and --relations')
        print(json.dumps(prepare_lane_relation_revision(a.model,a.relations,a.output),ensure_ascii=False));return
    if not a.graph or not a.s01_model:ap.error('build requires --graph and --s01-model')
    gp=Path(a.graph);sp=Path(a.s01_model);g=json.loads(gp.read_text(encoding='utf8'));s=json.loads(sp.read_text(encoding='utf8'));m=build_joint_model(g,s,graph_ref=str(gp.resolve()),s01_ref=str(sp.resolve()),require_ready=a.require_ready);m['s01_model_sha256']=hashlib.sha256(sp.read_bytes()).hexdigest();op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(blocks=len(m['blocks']),shots=list(m['shots']),rigs=list(m['rigs']),materials=len(m['materials']))))


def append_lane_relations(model, relation_file):
    """Restore the declared Lane family using existing free coordinates.

    The confirmed Lane2 upper edge supplies two coordinates of the family,
    not measured constants. Eliminating an auxiliary halfwidth/right-X block
    gives the same geometric family without new physical coordinates.
    Existing transverse right edges share the same corners; numerical source
    coverage caps never become measured physical endpoints.
    Left support ends stay independent and no unobserved edge is fabricated.
    """
    import re
    refpath=Path(relation_file).resolve()
    contract=json.loads(refpath.read_text(encoding='utf8'))
    if contract.get('schema')!='rinkspace.user-geometry-relations.current.v1':
        raise ValueError('Lane relations require the current human geometry definition')
    ties=contract['curling_structure']['confirmed_layout_ties']
    for key in ('equal_Lane_width','collinear_Lane_painted_right_edges','left_open_and_no_equal_length'):
        if not isinstance(ties.get(key),str) or not ties[key].strip():
            raise ValueError('Missing confirmed Lane relation: '+key)
    ref={'file':str(refpath),'sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),
         'field':'curling_structure.confirmed_layout_ties',
         'runtime_file_access_required':False,
         'current_fact_text':{key:ties[key]for key in ('equal_Lane_width','collinear_Lane_painted_right_edges','left_open_and_no_equal_length')}}
    q=copy.deepcopy(model);byname={b['name']:i for i,b in enumerate(q['blocks'])}
    if 'ice/structure/Lane_shared_halfwidth_rightX' in byname:
        raise ValueError('Existing auxiliary Lane layout needs an explicit model transition; do not delete its state')
    entries=[];coverage_caps=[];transverse=[]
    for material in q['materials']:
        mt=re.fullmatch(r'ice[.]lane([1-4]):edge([02])',material['label'])
        if not mt:continue
        row=int(mt.group(1));sign=1 if mt.group(2)=='0' else -1
        for path in material['paths']:
            if path['kind']!=0 or path.get('physical_point',False):
                raise ValueError('Lane long edge must be a finite non-point segment')
            bi=path['geometry_block']
            provenance=path.get('provenance',{})
            if isinstance(provenance,str):
                try:provenance=json.loads(provenance)
                except (ValueError,TypeError):provenance={}
            if provenance.get('numerical_coverage_cap',False):
                coverage_caps.append({'material':material['label'],'path_id':path.get('id'),'geometry_block':bi,'geometry_indices':path.get('geometry_indices'),'physical_endpoint':'unknown','role':'numerical_source_coverage_not_physical_endpoint'})
                continue
            ids=path.get('geometry_indices') or list(range(path.get('geometry_offset',0),path.get('geometry_offset',0)+4))
            if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Lane segment coordinate count')
            v=np.asarray(q['blocks'][bi]['values'],float)[ids]
            if not np.isfinite(v).all() or v[0]==v[2]:raise ValueError('Degenerate Lane carrier chart')
            house=byname[f'ice/circle/F{row}']
            if len(q['blocks'][house]['values'])<2:raise ValueError('Lane house-center shape')
            right_vertex=0 if v[0]>v[2] else 2
            entries.append((material['label'],row,sign,bi,ids,right_vertex,house))
    for material in q['materials']:
        mt=re.fullmatch(r'ice[.]lane([1-4]):edge1',material['label'])
        if not mt:continue
        row=int(mt.group(1));house=byname[f'ice/circle/F{row}']
        for path in material['paths']:
            if path['kind']!=0 or path.get('physical_point',False):
                raise ValueError('Observed Lane right edge must be a finite segment')
            bi=path['geometry_block']
            ids=path.get('geometry_indices') or list(range(path.get('geometry_offset',0),path.get('geometry_offset',0)+4))
            if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Lane right-edge coordinate count')
            v=np.asarray(q['blocks'][bi]['values'],float)[ids]
            if not np.isfinite(v).all() or v[1]==v[3]:raise ValueError('Degenerate ordered Lane right edge')
            transverse.append((material['label'],row,bi,ids,1 if v[1]>v[3] else -1,house))
    if len({e[0]for e in entries})!=len(entries):
        raise ValueError('Several base carriers for one finite Lane edge need explicit source topology')
    anchors=[e for e in entries if e[0]=='ice.lane2:edge0']
    if len(anchors)!=1:raise ValueError('The confirmed Lane2 upper edge must have one finite carrier')
    anchor=anchors[0];_,_,anchor_sign,ab,ai,ar,ah=anchor
    expected_equalities=[];expected_inequalities=[]
    def relation(rid,terms,rhs=0.,scale=10000.):
        merged={}
        for bi,index,coefficient in terms:
            merged[bi,index]=merged.get((bi,index),0.)+coefficient
        terms=[{'block':q['blocks'][bi]['name'],'index':index,'coefficient':coefficient}
               for (bi,index),coefficient in sorted(merged.items()) if coefficient!=0]
        if not terms:
            if rhs!=0:raise ValueError('Inconsistent eliminated Lane relation')
            return None
        return {'id':rid,'expression_kind':'linear','terms':terms,'rhs':rhs,
                'source_refs':[ref],'evaluation_scale':scale}
    for label,row,sign,bi,ids,right_vertex,house in entries:
        for endpoint in (1,3):
            d=relation(label+f'/endpoint{endpoint//2}_parallel_row_midline',
                [(bi,ids[endpoint],1),(house,1,-1),
                 (ab,ai[1],-sign*anchor_sign),(ah,1,sign*anchor_sign)])
            if d is not None:expected_equalities.append(d)
        d=relation(label+'/right_collinear',[(bi,ids[right_vertex],1),(ab,ai[ar],-1)])
        if d is not None:expected_equalities.append(d)
    for label,row,bi,ids,first_sign,house in transverse:
        for endpoint in (0,2):
            expected_equalities.append(relation(label+f'/endpoint{endpoint//2}_right_collinear',
                [(bi,ids[endpoint],1),(ab,ai[ar],-1)]))
            sign=first_sign if endpoint==0 else -first_sign
            expected_equalities.append(relation(label+f'/endpoint{endpoint//2}_row_side',
                [(bi,ids[endpoint+1],1),(house,1,-1),
                 (ab,ai[1],-sign*anchor_sign),(ah,1,sign*anchor_sign)]))
    # Same nondegenerate positive-coordinate convention as original native
    # radius/height constraints. The old arbitrary .25 auxiliary box is not
    # a human-confirmed physical maximum and is deliberately not introduced.
    expected_inequalities.append(relation('ice/Lane/common_positive_halfwidth',
        [(ab,ai[1],-anchor_sign),(ah,1,anchor_sign)],rhs=-1e-6,scale=1.))
    for cap in coverage_caps:
        bi=cap['geometry_block']
        base_x={ids[k]for label,row,sign,b,ids,right,house in entries if b==bi for k in (0,2)}
        cap_indices=cap['geometry_indices']
        if not cap_indices or len(cap_indices)!=4:
            raise ValueError('Numerical Lane coverage needs its explicit existing coordinate slice')
        added_x=set(cap_indices[::2])-base_x
        if len(added_x)!=1:raise ValueError('Numerical Lane coverage cap point is not uniquely defined')
        cx=next(iter(added_x));cap['numeric_cap_x_index']=cx
        expected_inequalities.append(relation(cap['material']+'/numerical_coverage_within_common_right_edge',
            [(bi,cx,1),(ab,ai[ar],-1)],rhs=0.,scale=1.))
    for key,wanted in [('physical_equalities',expected_equalities),
                       ('physical_inequalities',expected_inequalities)]:
        definitions=q.setdefault(key,[]);existing={d['id']:d for d in definitions}
        if len(existing)!=len(definitions):raise ValueError('Duplicate existing physical relation ID')
        for d in wanted:
            if d['id'] in existing:
                if existing[d['id']]!=d:raise ValueError('Existing Lane relation differs: '+d['id'])
            else:definitions.append(d)
    report={'schema':'rinkfit.restored_lane_hard_relations.v2',
        'representation':'eliminated_existing_free_coordinates',
        'active_existing_long_edges':len(entries),'active_existing_right_edges':len(transverse),
        'numerical_coverage_caps_not_used_as_physical_endpoints':coverage_caps,
        'finite_coverage_caps_constrained_by_common_right_edge':bool(coverage_caps),
        'numeric_cap_is_not_measured_paint_endpoint':True,
        'specific_cap_reading_physical_identity_certified':False,
        'equalities':len(expected_equalities),
        'inequalities':len(expected_inequalities),'new_physical_parameter_blocks':0,
        'parameter_registry_shape_unchanged':True,'original_block_values_unchanged':q['blocks']==model['blocks'],
        'common_halfwidth_expression':{'signed_difference':[
            {'block':q['blocks'][ab]['name'],'index':ai[1],'coefficient':anchor_sign},
            {'block':q['blocks'][ah]['name'],'index':1,'coefficient':-anchor_sign}]},
        'common_right_x_expression':{'block':q['blocks'][ab]['name'],'index':ai[ar]},
        'anchor_is_free_coordinate_not_measured_truth':True,
        'left_endpoints_independent':True,'equal_length_imposed':False,
        'finite_left_ends_are_support_not_measured_terminal_points':True,
        'new_observations_created':0,'unobserved_fourth_edge_created':False,
        'arbitrary_auxiliary_halfwidth_upper_bound_imposed':False,
        'missing_existing_geometric_materials':sorted({f'lane{i}:edge{k}'for i in range(1,5)for k in range(3)}-{m['label'].removeprefix('ice.')for m in q['materials']}),
        'equality_ids':[d['id']for d in expected_equalities],
        'inequality_ids':[d['id']for d in expected_inequalities],
        'complete_state_explicit_constraint_transition_required':True,
        'source_ref':ref}
    q['Lane_relation_restoration']=report
    return q,report


def prepare_lane_relation_revision(model_file, relation_file, output):
    """Write an explicit model proposal; never edit its parent model or CP."""
    parent=Path(model_file).resolve();target=Path(output).resolve()
    if target.exists():raise FileExistsError('A new explicit model revision is required')
    if len(str(target).encode('utf-16-le'))//2>=260 or len(target.name.encode('utf-16-le'))//2>255:
        raise ValueError('Output path exceeds the project Windows path limit')
    source=parent.read_bytes();model=json.loads(source)
    revised,report=append_lane_relations(model,relation_file)
    if revised['blocks']!=model['blocks']:raise ValueError('Lane relation revision moved original primal values')
    payload=(json.dumps(revised,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    report=dict(report,parent_model={'file':str(parent),'sha256':hashlib.sha256(source).hexdigest()},
        revised_model={'file':str(target),'sha256':hashlib.sha256(payload).hexdigest()},
        raw_UV_PTS_ID_Query_W_qualification_auxiliary_unchanged=True,
        parent_checkpoint_not_modified=True,new_checkpoint_or_old_token_claimed=False,
        production_input_binding_changed=False)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    target.with_suffix('.revision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


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



def validate_current_ice_definition(model):
    """Check the issued model itself; display never hides retired geometry."""
    policy=model.get('current_ice_definition_v1')
    if policy is None:return
    if policy.get('schema')!='rinkfit.current_ice_definition.v1':raise GraphError('Unknown current ice definition')
    c0=[]
    for material in model['materials']:
        entity=material['label'].split(':',1)[0]
        if entity in {'ice.A1','ice.A2','ice.A3','ice.A4'}:raise GraphError('Retired A circle is still in the current model')
        if entity=='ice.C0':c0.append(material)
    if c0:
        if len(c0)!=1 or c0[0]['label']!='ice.C0:middle' or len(c0[0]['paths'])!=1:raise GraphError('C0 must be one circular paint midline')
        path=c0[0]['paths'][0];block=model['blocks'][path['geometry_block']]
        if path['kind']!=1 or path.get('radius_index',2)!=2 or len(block['values'])!=3:raise GraphError('C0 has an extra boundary radius or a non-circular model')
    for binding in model.get('stroke_pair_model_bindings',[]):
        if model['materials'][binding['material_index']]['label'].startswith('ice.C0:'):raise GraphError('C0 boundary-pair residual survived its midline-only transition')


def revise_c0_midline_without_a(model):
    """Delete retired circle geometry and return explicit ordered state mappings.

    Raw measurements and checkpoints are not edited here. The caller must issue
    the corresponding qualification/material/pair transition and import the old
    complete primal with these exact parameter subsets. No coordinate is fitted,
    averaged, projected, clamped, or replaced by an observation.
    """
    import re
    if model.get('schema')!=SCHEMA:raise GraphError('C0/A revision requires a complete joint model')
    oldblocks=model['blocks'];oldmaterials=model['materials']
    names=[b['name']for b in oldblocks]
    if len(names)!=len(set(names)):raise GraphError('Duplicate parameter names')
    retired=lambda label:bool(re.match(r'^ice[.]A[1-4](?::|$)',label))
    c0=[i for i,m in enumerate(oldmaterials)if m['label'].split(':',1)[0]=='ice.C0']
    keepc0=[i for i in c0 if oldmaterials[i]['label'] in ('ice.C0:outer','ice.C0:middle')]
    if c0 and len(keepc0)!=1:raise GraphError('C0 requires one existing centreline carrier')
    c0mi=keepc0[0]if keepc0 else None
    remove_materials={i for i,m in enumerate(oldmaterials)if retired(m['label'])or(i in c0 and i!=c0mi)}
    remove_blocks={i for i,b in enumerate(oldblocks)if re.fullmatch(r'ice/circle/A[1-4]',b['name'])}
    removed_bindings=[b for b in model.get('stroke_pair_model_bindings',[])if b['material_index']in remove_materials or b['material_index']==c0mi]
    removed_pairs={b['pair_id']for b in removed_bindings}
    retained_bindings=[b for b in model.get('stroke_pair_model_bindings',[])if b['pair_id']not in removed_pairs]
    width_remove={b['half_width_block']for b in removed_bindings}
    if width_remove&{b['half_width_block']for b in retained_bindings}:raise GraphError('Retired circle width is shared with a retained material')
    for bi in width_remove:
        if not oldblocks[bi]['name'].startswith('measurement/paint_half_width/'):raise GraphError('Circle width is not a dedicated scalar nuisance')
    remove_blocks|=width_remove
    retained_width_scopes={b['width_scope_id']for b in retained_bindings}
    removed_width_scopes={b['width_scope_id']for b in removed_bindings}-retained_width_scopes
    subsets=[];c0bi=None
    if c0mi is not None:
        material=oldmaterials[c0mi]
        if len(material['paths'])!=1:raise GraphError('C0 has multiple existing centreline carriers')
        path=material['paths'][0];c0bi=path['geometry_block'];size=len(oldblocks[c0bi]['values'])
        if path['kind']!=1 or path.get('radius_index',2)!=2 or size not in (3,4):raise GraphError('Unsupported C0 radius layout')
        if size==4:subsets=[dict(name=oldblocks[c0bi]['name'],old_size=4,retained_coordinates=[0,1,2])]
    block_map=[];nextindex=0
    for i in range(len(oldblocks)):
        block_map.append(-1 if i in remove_blocks else nextindex)
        if i not in remove_blocks:nextindex+=1
    material_map=[];nextindex=0
    for i in range(len(oldmaterials)):
        material_map.append(-1 if i in remove_materials else nextindex)
        if i not in remove_materials:nextindex+=1
    path_map=[[-1]*len(m['paths'])if i in remove_materials else list(range(len(m['paths'])))for i,m in enumerate(oldmaterials)]
    q=copy.deepcopy(model)
    q['blocks']=[copy.deepcopy(b)for i,b in enumerate(oldblocks)if i not in remove_blocks]
    if c0bi is not None:
        b=q['blocks'][block_map[c0bi]]
        for key in ('values','lower','upper','scales'):
            if key in b:b[key]=b[key][:3]
        b['fixed_indices']=[x for x in b.get('fixed_indices',[])if x<3]
    q['materials']=[copy.deepcopy(m)for i,m in enumerate(oldmaterials)if i not in remove_materials]
    if c0mi is not None:
        m=q['materials'][material_map[c0mi]];m['label']='ice.C0:middle'
        m['paths'][0]['radius_index']=2
    q['stroke_pair_model_bindings']=[copy.deepcopy(b)for b in retained_bindings]
    q['measurement_widths']=[copy.deepcopy(w)for w in model.get('measurement_widths',[])if w['width_scope_id']not in removed_width_scopes]
    block_keys={'scene_block','geometry_block','advertisement_block','course_envelope_block','half_width_block','head_block','optics_block','reference_angles_block','shared_geometry_block','shared_advertisement_block','shared_height_block','block'}
    block_lists={'vertex_blocks','control_blocks','restored_original_control_blocks','unused_group_blocks'}
    moved_metadata=[]
    def bmap(v,path):
        if not 0<=v<len(block_map)or block_map[v]<0:raise GraphError('Retained model references a deleted parameter: '+path)
        return block_map[v]
    def remap(x,path=''):
        if isinstance(x,list):
            for i,v in enumerate(x):remap(v,path+'/'+str(i))
        elif isinstance(x,dict):
            for key,v in list(x.items()):
                child=path+'/'+key
                # These are historical source text, never executable model indices.
                if key in ('provenance','source_refs','source_evidence','source'):continue
                if key in block_keys and isinstance(v,int)and not isinstance(v,bool)and v>=0:
                    if key=='shared_height_block'and v in remove_blocks and path.startswith('/shared_board_material_interfaces/'):
                        moved_metadata.append(dict(path=child,original_block=v,note='Historical descriptive index; no current material path references this block'))
                        del x[key];x['legacy_shared_height_block_provenance']='old block '+str(v)+'; retired circle-width parameter, not a live board-height binding'
                    else:x[key]=bmap(v,child)
                elif key in block_lists and isinstance(v,list):x[key]=[bmap(i,child)for i in v]
                elif key=='material_index'and isinstance(v,int)and not isinstance(v,bool):
                    if not 0<=v<len(material_map)or material_map[v]<0:raise GraphError('Retained model references a retired material: '+child)
                    x[key]=material_map[v]
                else:remap(v,child)
    for key,value in q.items():
        if key!='blocks':remap(value,'/'+key)
    # Indexed metadata keys are not ordinary integer values.
    if 'printed_material_path_refinements'in q:
        q['printed_material_path_refinements']={str(material_map[int(k)]):v for k,v in q['printed_material_path_refinements'].items()if material_map[int(k)]>=0}
    if 'continuous_board_station_v1'in q:
        old=q['continuous_board_station_v1'].get('original_segments',{});new={}
        for key,value in old.items():
            mi,pi=map(int,key.split('/'))
            if material_map[mi]>=0:new[str(material_map[mi])+'/'+str(path_map[mi][pi])]=value
        q['continuous_board_station_v1']['original_segments']=new
    if 's01_block_map'in q:q['s01_block_map']={k:block_map[v]for k,v in q['s01_block_map'].items()if block_map[v]>=0}
    removed_names={oldblocks[i]['name']for i in remove_blocks}
    removed_explicit=[]
    for key in ('physical_equalities','physical_inequalities'):
        kept=[]
        for constraint in q.get(key,[]):
            terms=constraint.get('terms',[])
            dead=[t for t in terms if t.get('block')in removed_names or(t.get('block')==(oldblocks[c0bi]['name']if c0bi is not None else None)and int(t.get('index',-1))>=3)]
            if dead:
                if len(dead)!=len(terms):raise GraphError('Retired circle has an undeclared relation to a retained parameter: '+constraint['id'])
                removed_explicit.append(constraint['id'])
            else:kept.append(constraint)
        if key in q:q[key]=kept
    q['current_ice_definition_v1']=dict(schema='rinkfit.current_ice_definition.v1',C0='one_circular_painted_width_midline',C0_material_label='ice.C0:middle',A1_A4='removed_from_source_model_and_display',boundary_circle_parameters_removed=True,source_UV_modified=False)
    validate_current_ice_definition(q)
    residuals=[]
    def circle_ids(name,n):
        return [name+'/positive_radius_'+str(j)for j in range(2,n)]+[name+'/nested_radius_'+str(j)for j in range(2,n-1)]+[name+'/conservative_outer_square_'+str(sx)+'_'+str(sy)for sx in (-1,1)for sy in (-1,1)]
    for bi in sorted(remove_blocks):
        if oldblocks[bi]['name'].startswith('ice/circle/'):residuals+=circle_ids(oldblocks[bi]['name'],len(oldblocks[bi]['values']))
    if subsets:residuals += [oldblocks[c0bi]['name']+'/positive_radius_3',oldblocks[c0bi]['name']+'/nested_radius_2']
    report=dict(schema='rinkfit.C0_midline_A_retirement_model_transition.v1',old_block_count=len(oldblocks),new_block_count=len(q['blocks']),old_material_count=len(oldmaterials),new_material_count=len(q['materials']),old_block_to_new=block_map,old_material_to_new=material_map,old_path_to_new=path_map,component_subsets=subsets,removed_model_parameter_names=[oldblocks[i]['name']for i in sorted(remove_blocks)],removed_model_parameter_indices=sorted(remove_blocks),removed_stroke_pair_ids=sorted(removed_pairs),removed_measurement_width_scope_ids=sorted(removed_width_scopes),removed_materials=[dict(index=i,label=oldmaterials[i]['label'])for i in sorted(remove_materials)],removed_physical_constraint_ids=residuals+removed_explicit,C0_initialization='retain_existing_center_and_radius_coordinate_2_exactly; numerical_initialization_not_source_truth',C0_conservative_containment_rebinds_to_radius_coordinate_2=bool(subsets),retained_parameter_values_unchanged=True,raw_UV_PTS_ID_auxiliary_and_checkpoint_not_modified=True,retired_descriptive_metadata=moved_metadata)
    return q,report

if __name__=='__main__':main()
