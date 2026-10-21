"""SHA-bound full first-shot recovery inside the unchanged nine-shot scope.

No source readout, qualification, statistics, or old optimization dual is
modified by these coordinate/state import operations.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space,qr
from .joint_model import as_event_view
from .io_utils import file_sha256
from .camera_graph import SHOT_TO_RIG
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.native_camera import rotation


def load_navigation_context(manifest,source_model_sha256):
    """Find exact local history by SHA, never execute referenced old paths."""
    refs=[]
    def walk(value):
        if isinstance(value,dict):
            path=value.get('path',value.get('file'))
            if path and value.get('sha256')==source_model_sha256:refs.append(path)
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(manifest)
    for name in refs:
        path=Path(name)
        if not path.is_file() or file_sha256(path)!=source_model_sha256:continue
        model=json.loads(path.read_text(encoding='utf8'))
        if model.get('schema')!='rinkfit.joint_event_model.v1':continue
        contexts={}
        for rid in model['rigs']:
            sid=next(s for s,d in model['shots'].items() if d['rig_id']==rid)
            native=camera_from_event(as_event_view(model,sid)).source
            # as_event_view exposes the shared rig before the shot reference
            # offset. Preserve that chart exactly; do not subtract it again.
            h=copy.deepcopy(native['shared_head'])
            rig=dict(shared_head=h,optics=native['optics'],rolling_shutter=native['rolling_shutter'])
            contexts[rid]=dict(schema='rinkspace.navigation_seed_context.v1',source_model_sha256=source_model_sha256,
                source_model_path=str(path.resolve()),source_video_sha256=model['source_clock']['source_video_sha256'],
                time_base=model['source_clock']['time_base'],world_gauge=copy.deepcopy(model['coordinate_gauge']),
                head_chart_fingerprint=hashlib.sha256(json.dumps(rig,sort_keys=True).encode()).hexdigest(),
                rig_id=rid,rig=rig,chart_stage='optical_seed_pre_head_rebase',navigation_only=True,source_pixel_accuracy_claimed=False)
        return dict(status='resolved',contexts=contexts)
    return dict(status='unresolved',source_model_sha256=source_model_sha256,checked_paths=refs)


def rebase_same_physical_rig(context,shot_candidates,reference_shot):
    """Exact angular gauge redistribution for one complete physical rig."""
    if reference_shot not in shot_candidates:raise ValueError('reference shot candidate absent')
    rid=context['rig_id'];expected={s for s,r in SHOT_TO_RIG.items() if r==rid}
    if set(shot_candidates)!=expected:raise ValueError('all and only physical rig members must rebase together')
    for candidate in shot_candidates.values():
        if candidate.get('head_chart_fingerprint')!=context['head_chart_fingerprint']:raise ValueError('EXACT_REBASE_NOT_APPLICABLE: historical physical contexts differ')
    result=copy.deepcopy(shot_candidates);rig=copy.deepcopy(context['rig']);h=rig['shared_head'];old=np.asarray(h['R0'])
    delta=np.asarray(result[reference_shot]['reference_angles']);new=rotation(h['b'],delta[1])@old@rotation(h['n'],delta[0]);error=0.
    for candidate in result.values():
        before=np.asarray(candidate['reference_angles']);after=before-delta
        for q in np.asarray(candidate['camera_controls']['controls']):
            R1=rotation(h['b'],q[1]+before[1])@old@rotation(h['n'],q[0]+before[0])
            R2=rotation(h['b'],q[1]+after[1])@new@rotation(h['n'],q[0]+after[0])
            error=max(error,float(np.max(np.abs(R1-R2))))
        candidate['reference_angles']=after.tolist()
    if error>2e-12:raise ValueError('exact shared-head gauge equivalence failed')
    h['R0']=new.tolist()
    return dict(rig=rig,shots=result,maximum_rotation_arithmetic_error=error,physical_parameter_estimate=False)



def verify_baseline_complete_state(manifest_path):
    """Verify original full CP/auxiliary/raw source and all grouped-W files."""
    path=Path(manifest_path);manifest=json.loads(path.read_text(encoding='utf8'));checked={}
    def check(ref,base):
        name=ref.get('path',ref.get('file'));p=Path(name);p=p if p.is_absolute() else base/p
        digest=file_sha256(p)
        if digest!=ref['sha256']:raise ValueError('complete baseline dependency SHA mismatch: '+str(p))
        checked[str(p.resolve())]=digest;return p
    for ref in [manifest['model'],manifest['packet'],manifest['semantic'],manifest['predictions'],*manifest['row_state_files'].values(),*manifest['raw_source_arrays_verified_exact'].values()]:check(ref,path.parent)
    semantic_path=check(manifest['semantic'],path.parent);semantic=json.loads(semantic_path.read_text(encoding='utf8'))
    for ref in [*semantic['arrays'].values(),*semantic['assets'].values()]:check(ref,semantic_path.parent)
    statpath=check(semantic['active_statistics'],semantic_path.parent);stats=json.loads(statpath.read_text(encoding='utf8'))
    for replacement in stats.get('replacements',[]):
        check(replacement['W'],statpath.parent)
        if 'proof' in replacement:check(replacement['proof'],statpath.parent)
    return dict(schema='rinkfit.complete_baseline_dependency_verification.v1',model_sha256=manifest['model']['sha256'],
        original_source_identity=manifest['source_identity'],active_source_rows=manifest['active_rows'],all_frames=manifest['all_frames'],
        original_W_groups=semantic['statistics']['original_groups'],active_W_rank=stats['formal_residual_rows'],
        CP_source_nuisance_and_all_W_files_verified=True,verified_file_count=len(checked),verified_files=checked,
        old_duals_or_W_not_imported_into_new_engineering_objective=True)

def restore_s01_into_joint_container(current,baseline,*,baseline_sha256,baseline_ref,complete_state_ref=None):
    """Restore all coupled baseline state while preserving current row routing.

    Matching material indices stay fixed.  Shared scene, all original material
    path definitions, installation and print blocks, main head/optics and the
    original S01 control function are imported together.  Other-shot motion is
    retained as an unaccepted numerical estimate under the restored shared rig.
    """
    if set(current['shots'])!=set(SHOT_TO_RIG) or set(current['rigs'])!={'main','secondary','end'}:raise ValueError('full three-rig nine-shot scope required')
    q=copy.deepcopy(current);names={b['name']:i for i,b in enumerate(q['blocks'])};old=baseline['blocks'];cam=baseline['camera'];mapping={};index_maps={}
    current_materials={m['label']:m for m in current['materials']}
    for material in baseline['materials']:
        counterpart=current_materials.get(material['label'])
        if counterpart is None:continue
        for before,after in zip(material['paths'],counterpart['paths']):
            if before.get('kind')==after.get('kind')==1 and before.get('geometry_block',-1)>=0:
                index_maps.setdefault(before['geometry_block'],{0:0,1:1})[before.get('radius_index',2)]=after.get('radius_index',2)
    for i,b in enumerate(old):
        if i==0 or cam['control_block_begin']<=i<cam['control_block_begin']+cam['n_controls']:continue
        if b['name'] not in names:raise ValueError('complete baseline block mapping absent: '+b['name'])
        j=names[b['name']];mapping[i]=j
        if len(q['blocks'][j]['values'])!=len(b['values']):
            im=index_maps.get(i)
            if im is None or set(im)!=set(range(len(b['values']))):raise ValueError('changed material parameter chart lacks exact index mapping: '+b['name'])
            for key in ('values','lower','upper','scales'):
                for src,dst in im.items():q['blocks'][j][key][dst]=copy.deepcopy(b[key][src])
        else:
            q['blocks'][j]=copy.deepcopy(b)
    # An added print block is absent from the older baseline.  Restoring only
    # its installation would otherwise combine a decoded affine with the old
    # chart and collapse/move the surviving word (for example Ad8 Ankara).
    from .print_chart_domain import rebase_surviving_print_affines
    surviving_print_rebases=rebase_surviving_print_affines(current,q,mapping.values())
    global32=old[0];rig=q['rigs']['main']
    for block,lo,hi in ((q['scene_block'],0,5),(rig['head_block'],5,20),(rig['optics_block'],20,32)):
        target=q['blocks'][block]
        for key in ('values','lower','upper','scales'):target[key]=copy.deepcopy(global32[key][lo:hi])
        target['fixed_indices']=[i-lo for i in global32.get('fixed_indices',[]) if lo<=i<hi]
        target['constant']=bool(global32.get('constant',False))
    rig['camera']={k:copy.deepcopy(cam[k]) for k in ('R0_seed','n_seed','b_seed','n_tangent1','n_tangent2','f_ref','radius_max','tau_bound')}
    labels={m['label']:i for i,m in enumerate(q['materials'])};restored=[]
    for material in baseline['materials']:
        label=material['label']
        if label not in labels:raise ValueError('baseline material absent from current immutable routing: '+label)
        value=copy.deepcopy(material)
        for path in value['paths']:
            if path.get('kind')==1 and path.get('geometry_block') in index_maps:path['radius_index']=index_maps[path['geometry_block']][path.get('radius_index',2)]
            for key in ('geometry_block','advertisement_block'):
                if path.get(key,-1)>=0:path[key]=mapping[path[key]]
        q['materials'][labels[label]]=value;restored.append(label)
    s=q['shots']['s01'];s['camera']={k:copy.deepcopy(cam[k]) for k in ('dt','first_control_time','n_controls','dependent_control','reference_indices','reference_weights')}
    q['blocks'][s['reference_angles_block']]['values']=[0.,0.]
    controls=[old[cam['control_block_begin']+i]['values'] for i in range(cam['n_controls'])]
    if len(s['control_blocks'])!=len(controls):raise ValueError('baseline fine control allocation changed')
    for i,(bi,values) in enumerate(zip(s['control_blocks'],controls)):
        source=copy.deepcopy(old[cam['control_block_begin']+i]);source['name']=q['blocks'][bi]['name'];q['blocks'][bi]=source
        if s.get('control_coordinates'):
            c=s['control_coordinates'][i];q['blocks'][c['block']]['values'][c['offset']:c['offset']+3]=copy.deepcopy(values)
            if c['block']!=bi:q['blocks'][bi]['constant']=True
    q['s01_model_sha256']=baseline_sha256;q['source_s01_model_ref']=str(baseline_ref)
    q['s01_time_subspace_provenance']=copy.deepcopy(baseline.get('provenance',{}).get('time_subspace',{}))
    q['baseline_recovery_import']=dict(schema='rinkfit.full_s01_baseline_import.v1',baseline_sha256=baseline_sha256,
        baseline_ref=str(baseline_ref),complete_state_ref=complete_state_ref,full_coupled_model_import=True,
        restored_materials=restored,restored_shared_material_blocks=len(mapping),raw_UV_changed=False,
        previous_auxiliaries_or_duals_reused=False,requires_native_auxiliary_rebuild=True,
        other_shots_status='unaccepted numerical motion estimate under restored shared scene and main physical rig',
        exact_s01_time_function_preserved=True,surviving_print_chart_rebases=surviving_print_rebases)
    q['certificates']=dict(joint_fit_complete=False,formal_engineering_budget_passed=False,whole_shot_visual_accuracy_accepted=False)
    return q


def install_s01_supported_corrections(model,supported_native_pts,*,maximum_group_controls=32):
    """Small affine correction groups; original fine baseline is never refit.

    Two polynomial bump modes per group avoid frame-scale correction freedom.
    The inherited knot-removal group uses the same exact C constraints.  The
    reference/RS end controls stay at the baseline.  Ceres uses x+B delta with
    orthonormal B, so delta=0 is exactly the imported function.
    """
    q=copy.deepcopy(model);shot=q['shots']['s01'];meta=shot['camera'];n=meta['n_controls'];pts=np.asarray(supported_native_pts)
    if pts.dtype.kind not in 'iu' or not len(pts):raise ValueError('real qualified native source support required')
    if np.any((pts-shot['time_origin_pts'])%256):raise ValueError('source support off native clock')
    times=(pts-shot['time_origin_pts'])/12800.;coords=shot.get('control_coordinates') or [dict(block=b,offset=0) for b in shot['control_blocks']]
    values=np.asarray([q['blocks'][c['block']]['values'][c['offset']:c['offset']+3] for c in coords])
    inherited=q.get('linear_control_manifolds',[]);special=[]
    for item in inherited:
        if item['block'] in {c['block'] for c in coords}:
            indices=item['storage_control_indices'];special.append((min(indices),max(indices)+1,np.asarray(item['constraint_matrix'],float)))
    old_manifolds=[d for d in inherited if d['block'] not in {c['block'] for c in coords}]
    for bi in shot['control_blocks']:q['blocks'][bi]['constant']=True
    for c in coords:q['blocks'][c['block']]['constant']=True
    # Keep all controls touching the reference gauge and unsupported end halo.
    intervals=[];cursor=max(5,int(np.ceil((times.min()-meta['first_control_time'])/meta['dt']))-1);end=min(n-3,int(np.floor((times.max()-meta['first_control_time'])/meta['dt']))+2)
    for lo,hi,C in sorted(special):
        if cursor<lo:intervals.append((cursor,lo,None))
        intervals.append((lo,hi,C));cursor=hi
    if cursor<end:intervals.append((cursor,end,None))
    groups=[]
    for lo,hi,C in intervals:
        cuts=[(lo,hi,C)] if C is not None else [(a,min(a+maximum_group_controls,hi),None) for a in range(lo,hi,maximum_group_controls)]
        for a,b,C in cuts:
            size=b-a
            if size<4:continue
            x=np.linspace(0.,1.,size);raw=np.column_stack((x*(1-x),x*(1-x)*(2*x-1)))
            B=np.kron(raw,np.eye(3))
            if C is not None:
                # These polynomial modes are in the old knot-removal subfamily;
                # reject a changed chart rather than projecting baseline values.
                if np.max(np.abs(C@B))>1e-10:raise ValueError('correction violates inherited exact knot subspace')
            Q,R=qr(B,mode='economic');rank=int(np.linalg.matrix_rank(R));Q=Q[:,:rank]
            if rank==0:continue
            # Enforce deterministic signs for a stable serialized coordinate map.
            for col in range(rank):
                j=int(np.argmax(np.abs(Q[:,col])))
                if Q[j,col]<0:Q[:,col]*=-1
            bi=len(q['blocks']);indices=list(range(a,b));block={k:[] for k in ('values','lower','upper','scales')}
            for i in indices:
                source=q['blocks'][shot['control_blocks'][i]]
                block['values']+=values[i].tolist()
                for k in ('lower','upper','scales'):block[k]+=copy.deepcopy(source[k])
                coords[i]=dict(block=bi,offset=3*(i-a))
            block.update(name='shot/s01/control_group/recovery_'+str(a)+'_'+str(b-1),constant=False,fixed_indices=[]);q['blocks'].append(block)
            constraints=[] if C is None else C.tolist()
            group=dict(block=bi,ambient_size=3*size,tangent_size=rank,basis=Q.tolist(),constraint_matrix=constraints,
                membership_error=0. if C is None else float(np.max(np.abs(C@values[a:b].ravel()))),storage_control_indices=indices,
                affine_baseline_values=values[a:b].ravel().tolist(),not_motion_regularization=True,
                correction_role='source-supported low-frequency polynomial bump corrections to unchanged fine baseline',
                orthonormal_error=float(np.max(np.abs(Q.T@Q-np.eye(rank)))))
            groups.append(group)
    shot['control_coordinates']=coords;q['linear_control_manifolds']=old_manifolds+groups
    q['s01_affine_recovery_corrections']=dict(schema='rinkfit.source_supported_baseline_corrections.v1',baseline_at_zero_exact=True,
        source_support_first_pts=int(pts.min()),source_support_last_pts=int(pts.max()),groups=len(groups),
        tangent_scalars=sum(g['tangent_size'] for g in groups),maximum_group_controls=maximum_group_controls,
        baseline_function_resampled=False,unsupported_halo_and_reference_controls_fixed=True,original_exact_subspace_preserved=True,
        no_added_pixel_observations=True,no_temporal_smoothing_of_output=True)
    return q


def verify_zero_optimization_import(baseline,candidate,*,source_arrays=None):
    """All 1248 native exposures plus top/bottom readout and same-row Ad2."""
    original=camera_from_event(baseline);imported=camera_from_event(as_event_view(candidate,'s01'))
    shot=candidate['shots']['s01'];pts=np.arange(shot['scope']['first_pts'],shot['scope']['last_pts']+1,256,dtype=np.int64);t=(pts-shot['time_origin_pts'])/12800.
    halo=np.r_[t-original.tau/2,t,t+original.tau/2];a=original.spline(halo);b=imported.spline(halo)
    components=np.max(np.abs(a-b),axis=0)
    if np.max(components)>2e-12:raise ValueError('baseline control function changed')
    def pose(camera,abf):
        alpha,beta,logf=abf.T;R=rotation(camera.b,beta)@camera.R0@rotation(camera.n,alpha);z=logf-np.log(float(camera.optics['f_ref']))
        wb=np.einsum('...ji,j->...i',R,camera.b);h=np.cross(camera.n,wb);h/=np.linalg.norm(h,axis=1)[:,None];D=np.broadcast_to(camera.d,(len(abf),3)).copy()
        D[:,2]+=camera.head.get('pupil_linear',0)*z+camera.head.get('pupil_quadratic',0)*z*z
        C=camera.P+np.einsum('...ji,...j->...i',R,D)+camera.a*h
        return R,C,np.exp(logf)
    R0,C0,f0=pose(original,a);R1,C1,f1=pose(imported,b)
    pose_differences=dict(R_max_abs=float(np.max(np.abs(R0-R1))),C_max_abs_L1=float(np.max(np.abs(C0-C1))),focal_max_abs_native_px=float(np.max(np.abs(f0-f1))),signed_readout_difference_seconds=float(imported.tau-original.tau))
    if max(pose_differences.values())>2e-12:raise ValueError('baseline physical camera function changed')
    scene0=NativeScene(baseline);scene1=NativeScene(as_event_view(candidate,'s01'));labels={m['label']:i for i,m in enumerate(candidate['materials'])};world_delta=0.;witnesses=0
    for mi,mat in enumerate(baseline['materials']):
        mj=labels[mat['label']]
        for pi,path in enumerate(mat['paths']):
            stations=np.array([0.]) if path.get('physical_point') else np.linspace(0,1,5)
            for station in stations:
                X0=scene0.world(mi,pi,float(station));X1=scene1.world(mj,pi,float(station));world_delta=max(world_delta,float(np.max(np.abs(X0-X1))));witnesses+=1
    if world_delta>2e-12:raise ValueError('baseline shared material/installation geometry changed')
    report=dict(schema='rinkfit.full_s01_zero_optimization_equivalence.v1',native_exposures=len(pts),PTS_first=int(pts[0]),PTS_last=int(pts[-1]),
        RS_top_center_bottom_evaluations=len(halo),maximum_spline_component_change=components.tolist(),maximum_native_pose_change=pose_differences,
        maximum_material_world_change=world_delta,material_witnesses=witnesses,zero_optimization=True,source_UV_changed=False,
        exact_import_passed=True,fit_accuracy_accepted=False)
    if source_arrays:
        uv=np.fromfile(source_arrays['native_uv']['path'],dtype='<f8').reshape(-1,2);sp=np.fromfile(source_arrays['native_pts']['path'],dtype='<i8');mi=np.fromfile(source_arrays['material_index']['path'],dtype='<u4')
        ids=[i for i,m in enumerate(baseline['materials']) if m['label'].startswith('Ad2.') and m['paths'][0].get('physical_point')]
        ix=np.flatnonzero(np.isin(mi,ids));X=np.asarray([scene0.world(int(mi[i]),0,0.) for i in ix]);tt=(sp[ix]-shot['time_origin_pts'])/12800.
        p0=original.project_rs(X,tt);p1=imported.project_rs(X,tt);error=np.linalg.norm(p0['uv']-uv[ix],axis=1)
        delta=float(np.nanmax(np.abs(p1['uv']-p0['uv'])));same=bool(np.array_equal(p0['valid'],p1['valid']))
        if delta>2e-8 or not same:raise ValueError('same-source independent RS import changed')
        report['same_source_Ad2']=dict(n=len(ix),valid=int(p0['valid'].sum()),candidate_valid=int(p1['valid'].sum()),
            RMS_px=float(np.sqrt(np.mean(error*error))),median_px=float(np.median(error)),P95_px=float(np.quantile(error,.95)),maximum_px=float(error.max()),
            maximum_import_projection_change_px=delta,validity_identical=same)
    return report



def apply_source_supported_motion_flags(packet_path,model):
    """Freeze every correction coordinate without a qualified native readout.

    Support is the actual source exposure plus its complete allowed RS row
    interval. It is a temporal eligibility check, never a source qualification
    issuer or a rank/accuracy certificate. Unobserved baseline values stay put.
    """
    from .joint_metrics import PacketArrays
    from .board_source_qualification_guard import assert_board_material_source_qualification
    from rinkfit_start.camera import UniformCubicBasis
    from collections import defaultdict
    arrays=PacketArrays(packet_path);packet=arrays.packet
    assert_board_material_source_qualification(packet,packet_path)
    pts=arrays.load('native_pts');si=arrays.load('shot_index');active=arrays.load('admission').astype(bool)&(arrays.load('fit_role')==1)
    q=copy.deepcopy(model);manifolds={d['block']:d for d in q.get('linear_control_manifolds',[])};by_shot={};unsupported_by_block=defaultdict(set)
    for sid,shot in q['shots'].items():
        index=packet['shot_ids'].index(sid);selected=np.unique(pts[active&(si==index)]);m=shot['camera'];n=m['n_controls'];basis=UniformCubicBasis(m['first_control_time'],m['dt'],n)
        tau=abs(float(q['blocks'][q['rigs'][shot['rig_id']]['optics_block']]['values'][11]));times=(selected-shot['time_origin_pts'])/12800.;supported=np.zeros(n,bool)
        for time in times:
            lo=time-tau/2;hi=time+tau/2
            if lo<basis.domain[0]-1e-12 or hi>basis.domain[1]+1e-12:raise ValueError('qualified source RS time outside allocated native cubic')
            # Include knots and both interior halves so exactly zero endpoint
            # weights cannot create false positive or false negative support.
            knots=m['first_control_time']+m['dt']*np.arange(n);knots=knots[(knots>lo)&(knots<hi)]
            nodes=np.unique(np.r_[lo,knots,hi]);probes=np.unique(np.r_[nodes,(nodes[:-1]+nodes[1:])/2])
            for t in probes:
                inds,w=basis.local(float(t));supported[inds[np.abs(w)>0]]=True
        coords=shot.get('control_coordinates') or [dict(block=b,offset=0) for b in shot['control_blocks']]
        for i,coordinate in enumerate(coords):
            if not supported[i]:
                unsupported_by_block[coordinate['block']].update(range(coordinate['offset'],coordinate['offset']+3))
        by_shot[sid]=dict(qualified_source_exposures=len(selected),supported_controls=int(supported.sum()),unsupported_control_indices=np.flatnonzero(~supported).tolist(),RS_signed_readout_seconds=float(q['blocks'][q['rigs'][shot['rig_id']]['optics_block']]['values'][11]),support_is_not_an_observability_or_accuracy_certificate=True)
    for bi,indices in unsupported_by_block.items():
        block=q['blocks'][bi]
        if bi in manifolds:
            group=manifolds[bi];B=np.asarray(group['basis'],float);rows=np.asarray(sorted(indices),int);N=null_space(B[rows,:]);replacement=B@N
            if replacement.shape[1]==0:block['constant']=True
            else:
                Q,_=qr(replacement,mode='economic');group['basis']=Q.tolist();group['tangent_size']=int(Q.shape[1]);group['orthonormal_error']=float(np.max(np.abs(Q.T@Q-np.eye(Q.shape[1]))))
                if np.max(np.abs(Q[rows,:]))>1e-10:raise ValueError('unsupported control direction survived tangent restriction')
            group['qualified_source_unsupported_fixed_indices']=rows.tolist()
        elif len(indices)==len(block['values']):block['constant']=True
        else:block['fixed_indices']=sorted(set(block.get('fixed_indices',[]))|indices)
    q['source_supported_motion_flags']=dict(schema='rinkfit.qualified_native_motion_time_support.v1',packet_sha256=file_sha256(packet_path),by_shot=by_shot,
        unsupported_numeric_values_unchanged=True,navigation_or_model_pixels_grant_qualification=False,physical_rank_certified=False)
    return q

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--current',required=True);p.add_argument('--baseline',required=True);p.add_argument('--complete-state',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    op=Path(a.output)
    if op.exists():raise ValueError('new immutable candidate directory required')
    current=json.loads(Path(a.current).read_text(encoding='utf8'));base=json.loads(Path(a.baseline).read_text(encoding='utf8'));manifest=json.loads(Path(a.complete_state).read_text(encoding='utf8'))
    dependencies=verify_baseline_complete_state(a.complete_state)
    sha=file_sha256(a.baseline)
    if sha!=manifest['model']['sha256']:raise ValueError('baseline complete state belongs to another physical model')
    candidate=restore_s01_into_joint_container(current,base,baseline_sha256=sha,baseline_ref=a.baseline,complete_state_ref=a.complete_state)
    native_pts=np.fromfile(manifest['raw_source_arrays_verified_exact']['native_pts']['path'],dtype='<i8')
    candidate=install_s01_supported_corrections(candidate,native_pts)
    report=verify_zero_optimization_import(base,candidate,source_arrays=manifest['raw_source_arrays_verified_exact'])
    op.mkdir(parents=True);(op/'model.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf8')
    report.update(candidate_model_sha256=file_sha256(op/'model.json'),baseline_model_sha256=sha,complete_baseline_state_SHA_verified=True,complete_baseline_dependency_verification=dependencies,
        original_full_state_retained=str(Path(a.complete_state).resolve()),new_scope_auxiliaries_status='must rebuild from qualified generation; no v98 auxiliary state imported')
    (op/'import_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
