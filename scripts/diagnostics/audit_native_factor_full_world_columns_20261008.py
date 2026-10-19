"""Actual uncached full-world/camera FD audit for three native source factors."""
from pathlib import Path
from types import SimpleNamespace,MethodType
from copy import copy
import argparse,hashlib,json,os,sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor,_curve_error_on_association
from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
from structured_programme_source_candidate.native_board_material_point_source import NativeBoardMaterialPointFactor,SourceWithAdditionalNativeBoardMaterialPoints
from structured_programme_source_candidate.course_patterns_independent_placements import SourceWithIndependentPatternPlacements
from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v)for v in value]
    return value


def branches(factor,q,kind):
    cameras=factor._cameras(q);result=[]
    for row in factor.rows:
        if kind=='material' and row['curve']['kind']=='shared_board_pattern_material_points':continue
        world=(factor.world(q,row['advertisement_number']) if kind=='circle' else
               factor._world(q,row) if kind=='material' else factor._world(q,row['curve']))
        raw,association=_curve_error_on_association(cameras[row['actual_pts']],world,row['uv'])
        indices,t=association
        clamp=np.where(t<=0.,-1,np.where(t>=1.,1,0))
        result.extend(indices.tolist());result.extend(clamp.tolist())
        support=row.get('_source_optical_support')
        if support is not None:
            signed=np.einsum('ni,ni->n',raw.reshape(-1,2),support.normals)
            result.extend(np.where(signed<=support.bounds[:,0],-1,np.where(signed>=support.bounds[:,1],1,0)).tolist())
    return np.asarray(result,int)


def check(name,q,factor,value,jacobian,signature,world_indices,camera_columns,out):
    baseline=np.asarray(value(q),float);matrix=jacobian(q).tocsr();nominal=signature(q)
    if np.max(abs(baseline-np.asarray(factor._audit_block_r)))>1e-12:
        raise ValueError('Uncached actual residual differs from the nominal returned factor')
    def local_step(direction,h):
        trials=[]
        while True:
            plus,minus=q+h*direction,q-h*direction
            if not np.any(plus!=q)or not np.any(minus!=q):
                return None,dict(trials=trials,representable_same_branch=False)
            sp,sm=signature(plus),signature(minus)
            changed=np.flatnonzero((sp!=nominal)|(sm!=nominal))
            trials.append(dict(h=h,changed_branch_fields=int(len(changed))))
            if not len(changed):return h,dict(trials=trials,representable_same_branch=True)
            h*=.1
    rng=np.random.default_rng(9815360)
    all_columns=np.unique(np.r_[world_indices,camera_columns,getattr(factor,'pattern_columns',[]),
        getattr(factor,'circle_indices',[]),getattr(factor,'_audit_placement_indices',[])]).astype(np.int64)
    direction=np.zeros(len(q));direction[all_columns]=rng.normal(size=len(all_columns))
    direction/=max(float(abs(matrix@direction).max()),1.)
    h,branch=local_step(direction,1e-4)
    if h is None:raise ValueError((name,'No representable same-branch mixed FD',branch))
    actual=(value(q+h*direction)-value(q-h*direction))/(2*h);predicted=matrix@direction
    error=float(abs(actual-predicted).max());allowed=2e-3+2e-5*float(abs(actual).max())
    mixed=dict(full_world_and_actual_camera_columns=all_columns.tolist(),h=h,branch=branch,
        error=error,allowed=allowed,passed=error<=allowed)
    checks=[]
    for logical,col in enumerate(world_indices):
        direction=np.zeros(len(q));direction[int(col)]=1.
        h,branch=local_step(direction,1e-6)
        if h is None:
            record=dict(logical_world_column=logical,evaluation_column=int(col),passed=False,branch=branch)
        else:
            actual=(value(q+h*direction)-value(q-h*direction))/(2*h)
            predicted=matrix[:,int(col)].toarray().ravel()
            error=float(abs(actual-predicted).max());allowed=2e-3+2e-5*float(abs(actual).max())
            record=dict(logical_world_column=logical,evaluation_column=int(col),h=h,branch=branch,
                returned_J_max=float(abs(predicted).max()),uncached_actual_FD_max=float(abs(actual).max()),
                error=error,allowed=allowed,zero_J_but_nonzero_FD=bool(not np.any(predicted)and np.max(abs(actual))>2e-3),
                passed=error<=allowed)
        checks.append(record)
        (out/(name+'_all_world_FD_progress.json')).write_text(json.dumps(serial(dict(name=name,mixed=mixed,
            columns=checks,complete=len(checks)==len(world_indices))),indent=2)+'\n',encoding='utf8')
        if logical%10==0:print(json.dumps(dict(name=name,world_columns_checked=len(checks),last_error=record.get('error'),passed=record['passed'])),flush=True)
        if not record['passed']:raise AssertionError((name,record))
    if not mixed['passed']:raise AssertionError((name,mixed))
    return dict(name=name,row_count=len(baseline),J_shape=list(matrix.shape),all_world_columns=len(checks),mixed=mixed,
        column_max_error=max(r['error']for r in checks),zero_J_nonzero_FD_columns=[r for r in checks if r.get('zero_J_but_nonzero_FD')],
        uncached_value_used=True,source_UV_sigma_roles_or_solver_not_modified=True)


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--state',required=True,type=Path);cli.add_argument('--camera-identity',required=True,type=Path)
    cli.add_argument('--auxiliary',required=True,type=Path);cli.add_argument('--source-manifest',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path);cli.add_argument('--ice-only',action='store_true')
    cli.add_argument('--reference-state',type=Path)
    a=cli.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names','aux__installation_xyz_slots',
        'aux__current_shared_mount_parameter_indices','aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
        'aux__current_complete_parameter_layout_json','aux__shared_pattern_start')
    with np.load(a.state,allow_pickle=False)as archive:small={k:archive[k].copy()for k in keys}
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    identity=read(a.camera_identity);cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    if sha(identity['camera_path'])!=identity['camera_sha256']:raise ValueError('Immutable camera archive changed')
    state,camera_error=state_camera_adapter(small,cameras,provider);q=state.evaluation_q(small['q'])
    manifest=read(a.source_manifest);reports=[]
    packet=manifest['additional_real_native_ice_curves']['packet_path'];ice=NativeIceCurveFactor(state,packet,len(q))
    ib=ice.block(q,True);ice._audit_block_r=ib.r
    reports.append(check('ice_complete',q,ice,lambda v:ice._residual(v,ice.rows),lambda v:ib.J,
        lambda v:branches(ice,v,'ice'),state.geometry_parameter_indices,ice.camera_columns,a.output))
    crease=copy(ice);crease.rows=[r for r in ice.rows if r['curve']['kind']=='current_goal_crease']
    crease.held=[];crease.scales=np.concatenate([np.repeat(r['sigma'],2)for r in crease.rows]);crease.quadratic=np.ones(len(crease.scales),bool)
    crease.row_count=len(crease.scales);from collections import OrderedDict
    crease._cache=OrderedDict();cb=crease.block(q,True);crease._audit_block_r=cb.r
    reports.append(check('ice_crease_only',q,crease,lambda v:crease._residual(v,crease.rows),lambda v:cb.J,
        lambda v:branches(crease,v,'ice'),state.geometry_parameter_indices,crease.camera_columns,a.output))
    if a.ice_only:
        structural=None
        if getattr(provider.current,'derive_hockey_center_from_course',False):
            view=CurrentIceMaterialView(provider.current)
            original_features=json.dumps(serial(view.features(world)),sort_keys=True)
            queries=[dict(kind='hockey_circle',feature_id=f)for f in('C0','A1','A2','A3','A4')]
            direct=[ice._world(q,c).copy()for c in queries]
            invariance=[]
            for logical in(0,1):
                for delta in(-1e-3,1e-3):
                    trial=q.copy();trial[state.geometry_parameter_indices[logical]]+=delta
                    assert json.dumps(serial(view.features(trial[state.geometry_parameter_indices])),sort_keys=True)==original_features
                    direct_change=max(float(abs(ice._world(trial,c)-w).max())for c,w in zip(queries,direct))
                    actual_r_change=float(abs(ice._residual(trial,ice.rows)-ib.r).max())
                    assert direct_change==0. and actual_r_change==0.
                    invariance.append(dict(raw_world_column=logical,delta=delta,actual_ice_source_r_change=actual_r_change,
                        current_ice_all_feature_geometry_bitexact=True,direct_C0_A1_A4_curve_change_max=direct_change))
            structural=dict(derive_hockey_center_from_course=True,all_current_ice_feature_ids=list(view.features(world)),
                historical_raw_center_slots=world[:2].tolist(),effective_hockey_center=provider.current.effective_world(world)[:2].tolist(),
                raw_shadow_slot_invariance=invariance,actual_source_not_just_J_mask_verified=True)
        reference=None
        if a.reference_state:
            old_provider,_,old_proof=current_complete_source_geometry(a.reference_state,a.auxiliary)
            with np.load(a.reference_state,allow_pickle=False)as old:old_small={k:old[k].copy()for k in keys}
            old_state,_=state_camera_adapter(old_small,cameras,old_provider);old_q=old_state.evaluation_q(old_small['q'])
            assert np.array_equal(old_q,q)
            old_ice=NativeIceCurveFactor(old_state,packet,len(q));old_r=old_ice._residual(old_q,old_ice.rows)
            reference=dict(state_path=str(a.reference_state),state_sha256=sha(a.reference_state),world_proof=old_proof,
                canonical_q_bitexact=True,same_observation_UV_sigma_roles=True,
                old_ice_r_squared_sum=float(old_r@old_r),current_ice_r_squared_sum=float(ib.r@ib.r),
                current_minus_reference_r_max=float(abs(ib.r-old_r).max()),source_r_invariance_claimed=False)
        report=dict(schema='rinkspace.actual-native-full-world-and-camera-column-FD-audit.v1',state=str(a.state),
            state_sha256=sha(a.state),q_float64_sha256=hashlib.sha256(small['q'].tobytes()).hexdigest(),q_shape=list(q.shape),
            logical_world_count=provider.parameter_count,world_proof=proof,physical_camera_archive_reproduce_max=camera_error,
            factors=reports,current_structural_center_actual_value_check=structural,same_q_reference_r_change=reference,
            full_motion_graph_constructed=False,source_UV_sigma_roles_or_profiles_not_modified=True)
        target=a.output/'native_ice_circle_material_all_world_FD_check_v1.json'
        target.write_text(json.dumps(serial(report),indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(report=str(target),factors=[dict(name=r['name'],mixed_error=r['mixed']['error'],
            all_world_max_error=r['column_max_error'],zero_J_nonzero_FD=r['zero_J_nonzero_FD_columns'])for r in reports])),flush=True)
        return
    circle=NativeBoardCircleFactor(state,manifest['additional_real_native_board_circles']['packet_path'],len(q)-10)
    cq=q[:circle.parameter_count].copy();cirb=circle.block(cq,True);circle._audit_block_r=cirb.r
    reports.append(check('circle',cq,circle,lambda v:circle._residual(v,circle.rows),lambda v:cirb.J,
        lambda v:branches(circle,v,'circle'),state.geometry_parameter_indices,circle.camera_columns,a.output))
    from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor
    from structured_programme_source_candidate.current_full_source import CurrentPatterns
    scene=read(a.camera_identity.parent/'scene_UNACCEPTED.json')
    base=SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,facet_authority=scene['shared_board_patterns']['facet_authority'])
    original=SharedBoardPatternSourceFactor(base)
    owner=SimpleNamespace(start=int(small['aux__shared_pattern_start']),state=state,geometry=base,factor=original,external_factor=original,
        _original_course_pattern_geometry=base,_original_course_pattern_factor=original,_original_course_external_pattern_factor=original,
        course_world_geometry=provider,fit=[],held=[],bind_current_geometry=None)
    owner.bind_current_geometry=MethodType(CurrentPatterns.bind_current_geometry,owner);state.patterns=owner
    material=NativeBoardMaterialPointFactor(state,manifest['additional_real_native_board_material_points']['packet_path'],len(q)-4)
    empty=SimpleNamespace(count=len(q)-4,quadratic=np.empty(0,bool),scales=np.empty(0),contract={},updater=None,
        blocks=lambda value,jacobian:iter(()),guard=lambda value:provider.validate_parameters(value[state.geometry_parameter_indices]))
    source=SourceWithIndependentPatternPlacements(SourceWithAdditionalNativeBoardMaterialPoints(empty,material),state,
        Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/model_constraints/user_advertisement_pairs_20261005.json'))
    def material_value(value):
        with source.pattern_placement_context(value):return material._residual(value[:material.parameter_count],material.rows)
    def material_signature(value):
        with source.pattern_placement_context(value):return branches(material,value[:material.parameter_count],'material')
    mb=next(source.blocks(q,True));material._audit_block_r=mb.r;material._audit_placement_indices=source.placement_indices
    reports.append(check('material_points',q,material,material_value,lambda value:mb.J,material_signature,
        state.geometry_parameter_indices,material.camera_columns,a.output))
    report=dict(schema='rinkspace.actual-native-full-world-and-camera-column-FD-audit.v1',state=str(a.state),q_shape=list(q.shape),
        logical_world_count=provider.parameter_count,world_proof=proof,physical_camera_archive_reproduce_max=camera_error,
        factors=reports,full_motion_graph_constructed=False,source_UV_sigma_roles_or_profiles_not_modified=True)
    target=a.output/'native_ice_circle_material_all_world_FD_check_v1.json';target.write_text(json.dumps(serial(report),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(target),factors=[dict(name=r['name'],mixed_error=r['mixed']['error'],
        all_world_max_error=r['column_max_error'],zero_J_nonzero_FD=r['zero_J_nonzero_FD_columns'])for r in reports])),flush=True)


if __name__=='__main__':main()
