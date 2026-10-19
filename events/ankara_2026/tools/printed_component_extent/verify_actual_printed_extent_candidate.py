"""Data-only real-state equivalence and printed extent derivative audit."""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, os, sys, time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor as Original
from structured_programme_source_candidate.source_block_target_extension_replay import _native_append_plan

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('structured_programme_source_candidate._data_printed_extent_candidate',HERE/'native_board_edge_source.py')
candidate_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate_module)
Candidate=candidate_module.NativeBoardEdgeFactor

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def plain(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))
def load_state(directory,auxiliary,manifest):
    identity=read(directory/'state_identity.json');source=directory/'complete_state_UNACCEPTED.npz'
    assert sha(source)==identity['state_npz_sha256']
    assert sha(identity['camera_path'])==identity['camera_sha256']
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
        'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
        'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
        'aux__current_complete_parameter_layout_json')
    with np.load(source,allow_pickle=False)as z:small={k:z[k].copy()for k in keys}
    assert hashlib.sha256(small['q'].tobytes()).hexdigest()==identity['q_sha256']
    provider,_,_=current_complete_source_geometry(source,auxiliary)
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
    state,error=state_camera_adapter(small,cameras,provider)
    scope=manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope is not None:state.current_user_physical_course_scope_source=scope['path']
    return state,state.evaluation_q(small['q']),identity,error

def old_equivalence(state,q,parent_path):
    started=time.monotonic();old=Original(state,parent_path,len(q));new=Candidate(state,parent_path,len(q))
    assert old.contract==new.contract
    ob,nb=old.block(q,True),new.block(q,True)
    assert np.array_equal(ob.r,nb.r);assert np.array_equal(ob.scales,nb.scales);assert np.array_equal(ob.quadratic,nb.quadratic)
    assert np.array_equal(ob.J.data,nb.J.data);assert np.array_equal(ob.J.indices,nb.J.indices);assert np.array_equal(ob.J.indptr,nb.J.indptr)
    assert old.review(q)==new.review(q)
    return dict(actual_old_fit_groups=len(old.rows),actual_old_held_groups=len(old.held),scalar_rows=old.row_count,
        actual_full_old_fit_r_J_scales_quadratic_and_held_review_bit_exact=True,
        old_contract_all_keys_and_values_exact=True,elapsed_sec=time.monotonic()-started),new

def new_checks(state,q,parent,packet_path,output):
    packet=read(packet_path);old_packet=read(parent.packet_path)
    assert packet['rows'][:len(old_packet['rows'])]==old_packet['rows']
    new=Candidate(state,packet_path,len(q))
    plan=_native_append_plan({'additional_real_native_board_edges':parent.contract},
        {'additional_real_native_board_edges':new.contract},('additional_real_native_board_edges',))
    # Check only appended real source rows with the same actual public block.
    selected=copy.copy(new);selected.rows=new.rows[len(parent.rows):]
    from collections import OrderedDict
    selected._cache=OrderedDict();selected.scales=new.scales[len(parent.scales):]
    selected.quadratic=new.quadratic[len(parent.quadratic):];selected.row_count=len(selected.scales)
    selected.blocks={r['actual_pts']:new.blocks[r['actual_pts']]for r in selected.rows}
    block=selected.block(q,True);columns=np.flatnonzero(np.asarray(abs(block.J).sum(axis=0)).ravel())
    fd=[]
    try:
        for col in selected.active:
            h=20.*selected.camera_steps.get(int(col),1e-6)
            direction=np.zeros_like(q);direction[col]=1.
            h,_=selected.local_support_directional_step(q,direction,h)
            plus=q+h*direction;minus=q-h*direction
            estimate=(selected._residual(plus,selected.rows)-selected._residual(minus,selected.rows))/(2*h)
            actual=block.J[:,col].toarray().ravel()
            fd.append(dict(column=int(col),step=h,maximum_error=float(abs(estimate-actual).max()),
                derivative_maximum=float(abs(actual).max())))
    finally:selected._cameras(q)
    maximum=max(r['maximum_error']for r in fd)
    assert maximum<1e-3,maximum
    # The actual source support may already contain this complete state.
    # Audit raw geometry separately and activate the same immutable support
    # by an ephemeral camera perturbation; never tighten the measurement.
    raw_factor=copy.copy(selected);raw_factor.rows=[{k:v for k,v in r.items()if k!='normal_interval'}for r in selected.rows]
    raw_factor._cache=OrderedDict();raw_block=raw_factor.block(q,True)
    raw_fd=[]
    try:
        for col in raw_factor.active:
            h=20.*raw_factor.camera_steps.get(int(col),1e-6);direction=np.zeros_like(q);direction[col]=1.
            estimate=(raw_factor._residual(q+h*direction,raw_factor.rows)-raw_factor._residual(q-h*direction,raw_factor.rows))/(2*h)
            actual=raw_block.J[:,col].toarray().ravel()
            raw_fd.append(dict(column=int(col),step=h,maximum_error=float(abs(estimate-actual).max()),derivative_maximum=float(abs(actual).max())))
    finally:raw_factor._cameras(q)
    raw_maximum=max(r['maximum_error']for r in raw_fd);assert raw_maximum<1e-3,raw_maximum
    first_x=raw_block.J[0].toarray().ravel();camera_columns=selected.camera_columns
    col=int(camera_columns[np.argmax(abs(first_x[camera_columns]))]);assert abs(first_x[col])>0
    moved=q.copy();moved[col]-=30./(first_x[col]*selected.rows[0]['sigma'][0])
    active_factor=copy.copy(selected);active_factor._cache=OrderedDict();active_block=active_factor.block(moved,True)
    assert abs(active_block.r[0])>0
    assert active_block.J.nnz>0
    active_fd=[]
    try:
        for c in active_factor.active:
            direction=np.zeros_like(q);direction[c]=1.
            h,_=active_factor.local_support_directional_step(moved,direction,20.*active_factor.camera_steps.get(int(c),1e-6))
            estimate=(active_factor._residual(moved+h*direction,active_factor.rows)-active_factor._residual(moved-h*direction,active_factor.rows))/(2*h)
            actual=active_block.J[:,c].toarray().ravel()
            active_fd.append(dict(column=int(c),step=h,maximum_error=float(abs(estimate-actual).max()),derivative_maximum=float(abs(actual).max())))
    finally:active_factor._cameras(q)
    active_maximum=max(r['maximum_error']for r in active_fd);assert active_maximum<1e-3,active_maximum
    examples=[]
    for row in new.rows[len(parent.rows):]+new.held[len(parent.held):]:
        camera=new._cameras(q)[row['actual_pts']];world=new._world(q,row['curve'])
        from z190_source_camera_diagnostic import absolute_value_jac15
        raw=new._residual(q,[row],apply_optical_support=False)*np.repeat(row['sigma'],2)
        supported=new._residual(q,[row])*np.repeat(row['sigma'],2)
        examples.append(dict(id=row['id'],pts=row['pts'],actual_pts=row['actual_pts'],split=row['split'],
            sigma=row['sigma'].tolist(),uv=row['uv'].tolist(),normal_interval_px=row['normal_interval'].tolist(),
            raw_signed_xy_px=raw.tolist(),outside_support_xy_px=supported.tolist(),world_vertices=len(world)))
    reject=[]
    def bad(name,mutate):
        badpacket=copy.deepcopy(packet);mutate(badpacket['rows'][-1]);path=output/(name+'.json')
        path.write_text(json.dumps(badpacket,ensure_ascii=False,indent=2),encoding='utf8')
        try:Candidate(state,path,len(q))
        except (ValueError,KeyError)as e:reject.append(dict(case=name,rejected=True,error=str(e)))
        else:raise AssertionError('Bad source accepted: '+name)
    bad('bad_template_id',lambda r:r['curve'].update(template_id='fabricated_template'))
    bad('bad_component_id',lambda r:r['curve'].update(component_id='fabricated_component'))
    bad('bad_instance_number',lambda r:r['curve'].update(advertisement_number=9))
    bad('bad_parent_sha',lambda r:r['curve']['frozen_print_source_packet'].update(sha256='0'*64))
    bad('free_world_coordinates',lambda r:r.update(world=[[0.,0.,0.]]))
    bad('free_material_coordinates',lambda r:r['curve'].update(material_coordinates_normalized=[[0.,0.]]))
    bad('point_correspondence',lambda r:r.update(point_correspondence=True))
    bad('non_horizontal_normal',lambda r:r.update(normal=[[0.,1.]]))
    bad('two_source_points',lambda r:r.update(uv=r['uv']+r['uv'],normal=r['normal']+r['normal'],normal_interval_px=r['normal_interval_px']+r['normal_interval_px']))
    bad('bad_actual_pts',lambda r:r.update(actual_pts=int(r['actual_pts'])+256))
    bad('nonfinite_source_support',lambda r:r.update(normal_interval_px=[[float('nan'),0.]]))
    return dict(actual_public_append_plan=plain(plan),appended_examples=examples,
        appended_scalar_rows=selected.row_count,parameter_count_before_and_after=len(q),
        actual_active_FD_columns=len(fd),nonzero_J_columns=columns.tolist(),
        public_J_vs_independent_same_support_FD_maximum=maximum,coordinate_FD=fd,negative_cases=reject,
        raw_geometry_public_J_vs_independent_FD_maximum=raw_maximum,raw_geometry_coordinate_FD=raw_fd,
        raw_geometry_nonzero_J_columns=np.flatnonzero(np.asarray(abs(raw_block.J).sum(axis=0)).ravel()).tolist(),
        original_support_activation=dict(temporary_camera_column=col,temporary_parameter_delta=float(moved[col]-q[col]),
            temporary_raw_signed_x_px=float(raw_factor._residual(moved,raw_factor.rows)[0]*selected.rows[0]['sigma'][0]),
            temporary_supported_residual=active_block.r.tolist(),public_J_nnz=int(active_block.J.nnz),
            public_J_vs_same_support_FD_maximum=active_maximum,coordinate_FD=active_fd,
            original_source_UV_sigma_and_support_unchanged=True,temporary_parameter_state_not_accepted_or_saved_as_solution=True),
        old_packet_full_typed_prefix_unchanged=True,source_UV_and_uncertainty_not_replaced_by_model=True,
        geometry_DOF_added=0,accuracy_accepted=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state-directory',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--auxiliary',type=Path,required=True)
    ap.add_argument('--new-packet',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--old-equivalence-evidence',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);manifest=read(a.manifest)
    state,q,identity,camera_error=load_state(a.state_directory,a.auxiliary,manifest)
    result=dict(schema='rinkspace.Data-only-actual-public-printed-component-extent-factor-regression.v1',
        q_sha256=identity['q_sha256'],actual_camera_archive_reproduce_max=camera_error,
        candidate_sha256=sha(HERE/'native_board_edge_source.py'),state_directory=str(a.state_directory))
    if a.old_equivalence_evidence is None:
        result['old_equivalence'],parent=old_equivalence(state,q,manifest['additional_real_native_board_edges']['packet_path'])
    else:
        previous=read(a.old_equivalence_evidence)
        assert previous['q_sha256']==identity['q_sha256']
        assert previous['candidate_sha256']==result['candidate_sha256']
        assert previous['old_equivalence']['actual_full_old_fit_r_J_scales_quadratic_and_held_review_bit_exact']is True
        result['old_equivalence']=previous['old_equivalence']
        result['authenticated_old_equivalence_evidence']=dict(path=str(a.old_equivalence_evidence),sha256=sha(a.old_equivalence_evidence))
        parent=Candidate(state,manifest['additional_real_native_board_edges']['packet_path'],len(q))
    (a.output/'old_equivalence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    if a.new_packet is not None:result['new_source']=new_checks(state,q,parent,a.new_packet,a.output)
    result['passed']=True;result['live_source_changed']=False
    (a.output/'actual_public_factor_regression.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(q_sha256=identity['q_sha256'],passed=True,old=result['old_equivalence'],
        new_FD_max=result.get('new_source',{}).get('public_J_vs_independent_same_support_FD_maximum'))))

if __name__=='__main__':main()
