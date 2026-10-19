"""Small actual-scene audit of the corrected structured reference family."""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import sys,os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
B=_runtime_data_root();sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
from simple_wide_perspective_reset_candidate.audit_actual_world49_feasible_source import make_engine
from simple_wide_perspective_reset_candidate.structured_geometry_reference import *
from complete_ice_template_candidate.complete_ice_numeric_geometry import _boundary_hit
from complete_ice_template_candidate.structured_shared_ice_template import StructuredSharedIceTemplateGeometry
from complete_ice_template_candidate.full_free_boundary_geometry import StructuredSharedIceTemplateFullBoundaryGeometry
from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry_v2 import SharedUnmeasuredAdvertisementGeometryV2
from simple_wide_perspective_reset_candidate.qualified_structured_geometry import QualifiedStructuredFullBoundaryGeometry
from simple_wide_perspective_reset_candidate.differentiable_unmeasured_print_geometry import DifferentiableUnmeasuredPrintGeometryV2
import json,time,hashlib,pickle


def make_structured_geometry(*,align_qualified_goal=True):
    old,q,event,source=make_engine()
    seed=B/'constrained_physical_fit_candidate/certified_firstcap_goal_initializer_UNACCEPTED.npz'
    with np.load(seed,allow_pickle=False)as z:q=z['q'].copy()
    parentq=q[old.world_slice];spans=q[old.span_slice]
    g48=StructuredSharedIceTemplateGeometry(old.g,parentq,engineering_config=dict(goal_carrier_span_m=float(spans[0]),board_carrier_span_m=float(spans[1])))
    g50=QualifiedStructuredFullBoundaryGeometry(StructuredSharedIceTemplateFullBoundaryGeometry(g48))
    g=DifferentiableUnmeasuredPrintGeometryV2(g50,g50.initial);current=g.initial.copy()
    if align_qualified_goal:
        # Carry the actual certified first-cap goal carrier into the common
        # hockey frame. This is a FIT-based initialization, not a nominal
        # dimension or a historical-centre prior. Other source costs remain.
        original=old.g.source_scene(parentq)
        fs={f['id']:f for f in original['ice_features']}
        frame=fs['goal_line_right']['shared_physical_goal_frame'];unit=g.scene_unit_m
        anchor=np.asarray(frame['centre_xyz'],float)*unit
        a=-np.asarray(frame['inward_xyz'],float);v=np.asarray(frame['across_xyz'],float)
        centre=np.r_[current[:2],0.];centre+=v*((anchor-centre)@v)
        current[:2]=centre[:2];current[2]=np.arctan2(a[1],a[0])
        poly=np.asarray(g.source_scene(current)['ice_boundary']['points_xyz'])
        hit=_boundary_hit(centre/unit,a,poly)*unit
        current[8]=float((hit-anchor)@a);current[10:12]=spans
        g.validate_parameters(current)
        g.qualified_initialization_ledger=dict(original_goal_frame=str(seed),
            common_hockey_axis_initialized_from_certified_goal_carrier=True,
            centre_transverse_initialized_on_that_axis_without_origin_prior=True,
            right_goal_distance_from_current_actual_course_intersection=True,
            no_NEW_UV_or_nominal_dimension_or_HELD_used=True,not_source_accuracy_or_admission_certificate=True)
    return g,current,old,q,event,seed


def rigid_scene(scene,q,angle,shift):
    scene=deepcopy(scene);out=q.copy();c,s=np.cos(angle),np.sin(angle)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shift=np.r_[shift,0.];unit=.31353729994266055
    def xyz(value):return (np.asarray(value,float)@R.T+shift/unit).tolist()
    for feature in scene['ice_features']:
        for key in ('center_xyz','segments_xyz','polygon_xyz'):
            if key in feature:feature[key]=xyz(feature[key])
        if 'shared_physical_goal_frame'in feature:
            f=feature['shared_physical_goal_frame'];f['centre_xyz']=xyz(f['centre_xyz'])
            for k in ('inward_xyz','across_xyz'):f[k]=(R@np.asarray(f[k],float)).tolist()
    scene['ice_boundary']['points_xyz']=xyz(scene['ice_boundary']['points_xyz'])
    out[:2]=R[:2,:2]@q[:2]+shift[:2];out[2]+=angle
    out[12:14]=R[:2,:2]@q[12:14]+shift[:2];out[14]+=angle
    return scene,out


def main():
    start=time.perf_counter();g,q,old,qold,event,seed=make_structured_geometry()
    rows,nominal,scales,contract=build_structured_reference_contract(g)
    scene=g.source_scene(q);value=structured_reference_values(g,q,scene)
    actual=structured_reference_scene_values(g,q,scene);J=structured_reference_jacobian(g,q)
    err=float(abs(actual-value).max());assert err<1e-8,err
    fd=[];h=1e-6
    for k in range(g.parameter_count):
        qp=q.copy();qm=q.copy();qp[k]+=h;qm[k]-=h
        a=(structured_reference_values(g,qp)-structured_reference_values(g,qm))/(2*h)
        fd.append(float(abs(a-J[:,k]).max()))
    assert max(fd)<3e-9,max(fd)
    actual_fd=[]
    for k in sorted(set(ACTIVE_ROWS.values())|{0,1,2,14,44,45,46,47,48,49}):
        qp=q.copy();qm=q.copy();qp[k]+=h;qm[k]-=h
        a=(structured_reference_scene_values(g,qp,g.source_scene(qp))-structured_reference_scene_values(g,qm,g.source_scene(qm)))/(2*h)
        e=float(abs(a-J[:,k]).max());actual_fd.append(dict(column=k,name=g.parameter_names[k],max_absolute=e))
    assert max(x['max_absolute']for x in actual_fd)<3e-8,actual_fd
    transformed,qt=rigid_scene(scene,q,.173,[2.1,-1.7])
    gauge=structured_reference_scene_values(g,qt,transformed)
    gauge_error=float(abs(gauge-actual).max());assert gauge_error<1e-8,gauge_error
    assert np.all(J[:,:3]==0),'Historical hockey origin/yaw anchor reintroduced'
    assert np.array_equal(structured_reference_values(pickle.loads(pickle.dumps(g)),q),value)
    raw=value-nominal;r=raw/scales
    ledger=[dict(row,actual_value=float(value[i]),scene_reconstructed_value=float(actual[i]),raw_error=float(raw[i]),normalized_error=float(r[i]),cost=float(.5*r[i]**2))for i,row in enumerate(rows)]
    report=dict(schema=SCHEMA,status='PASS_SMALL_ACTUAL_SCENE_ONLY',parameter_count=g.parameter_count,
        original45_2144_and26_UV_roles_sigma_unchanged=True,reference_objective_changed=True,
        removed_historical_world_XY_anchor=True,source_initialization=str(seed),source_initialization_not_admitted=True,
        underlying_source_event=event['event'],new_structural_geometry_is_distinct_from_original_freeze=True,
        named_vs_actual_current_scene_max_absolute=err,all62_named_partial_FD_max_absolute=max(fd),
        actual_current_scene_FD=actual_fd,current_axis_rigid_world_gauge_max_absolute=gauge_error,
        rigid_gauge_is_diagnostic_not_candidate_geometry=True,reference_rows=ledger,reference_total_cost=float(.5*r@r),
        contract=contract,seconds=time.perf_counter()-start)
    out=B/'simple_wide_perspective_reset_candidate/actual_structured_current_axis_reference_audit.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    fixture=out.with_suffix('.npz');np.savez_compressed(fixture,q=q,values=value,scene_values=actual,J=J,nominal=nominal,scales=scales)
    print(json.dumps(dict(status=report['status'],report=str(out),scene_error=err,FD=max(fd),actual_scene_FD=max(x['max_absolute']for x in actual_fd),gauge=gauge_error,seconds=report['seconds'])),flush=True)

if __name__=='__main__':main()
