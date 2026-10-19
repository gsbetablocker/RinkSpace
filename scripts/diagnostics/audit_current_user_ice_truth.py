"""Same-q actual public factors for the latest registered user ice correction."""
from pathlib import Path
import os,sys,json,copy,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
RUNS=DATA/'calibration/runs/solver_acceleration_20261003'
HELP=RUNS/'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
HERE=DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
STATE=RUNS/'okada_full9_current_review71_clarabel_20261009_152520_411/accepted_states/27fa214561703310c832f3884656cf16c9461870a44ed3acf808bd866a4eeb47'
sys.path[:0]=[str(HELP),str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
from audit_user_left_goal_line_and_ad_edges_candidate import load_state,read,sha,ref,exact,bind_human,wide_factor,values
from existing_LOW_circle_material_owner_light import existing_LOW_source_owners
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth,direct_manual_centre_factors
import numpy as np

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def main():
    began=time.perf_counter()
    config=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_speed_candidate_20261010.json'
    previous=read(config)['source_options'];new=dict(previous)
    receiptpath=HERE/'current_user_ice_corrections/truth_append/latest_user_ice_truth_append_receipt.json';receipt=read(receiptpath)
    mapping=[('--additional-board-source','board'),('--additional-ice-source','ice'),('--direct-human-truth-authority','human_authority'),
        ('--direct-human-full-stroke-guard','full_guard'),('--direct-human-stroke-cap-support','stroke_caps'),
        ('--practical-human-location-envelope','location'),('--user-confirmed-source-truth-authority','confirmed_authority')]
    for key,name in mapping:
        entry=receipt['paths'][name];assert sha(entry['path'])==entry['sha256'];new[key]=entry['path']
    state,q,identity,camera_error=load_state(STATE,Path(previous['--world-init']),read(STATE.parent.parent/'input_manifest.json'))
    assert camera_error==0.;identity['_accepted_directory']=str(STATE);before=q.tobytes()
    board=NativeBoardEdgeFactor(state,new['--additional-board-source'],len(q))
    ices=[NativeIceCurveFactor(state,p['--additional-ice-source'],len(q)) for p in (previous,new)]
    humans=[bind_human(state,board,ice,p,len(q)) for ice,p in zip(ices,(previous,new))]
    wide=wide_factor(state,identity,new,len(q),new['--practical-human-location-envelope'])
    old71=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review71_20261009.json')['source_options']
    circle,material,placement=existing_LOW_source_owners(state,q,STATE,old71,read(STATE/'scene_UNACCEPTED.json'))
    confirmed=UserConfirmedSourceTruth(state,new['--user-confirmed-source-truth-authority'],board,ices[1],parameter_count=len(q),
        circle_factor=circle,material_factor=material,placement_source=placement)
    a,b=[values(h,wide,confirmed,q) for h in humans]
    for name in ('primary','full'):assert exact(a[name],b[name][:50]),name
    for family in ('wide','confirmed'):assert exact(a['buffers'][family],b['buffers'][family])
    protectedpath=HERE/'current82_caps_values_bound_to_exact27fa_v1.json';protected=read(protectedpath)
    families={}
    for family,count in [('native',50),('wide',2),('confirmed',30)]:
        saved=protected['families'][family];actual=b['buffers'][family]
        ids=[r['id'] for r in b['primary' if family=='native' else 'wide' if family=='wide' else 'confirmed_details']]
        assert ids[:count]==saved['group_ids']
        assert np.max(actual[:count]-np.asarray(saved['limits']))<=1e-8
        limits=copy.deepcopy(saved['limits']);uncertainty=copy.deepcopy(saved['uncertainties'])
        if family=='native':
            for v in actual[count:]:limits.append(float(v) if v>1e-8 else 0.);uncertainty.append(6.)
        families[family]=dict(group_ids=ids,limits=limits,uncertainties=uncertainty,old_count=count,
            old_ids_limits_uncertainties_exact=True,violations_px=actual.tolist())
    source=ices[1];start=len(ices[0].rows)
    selected=copy.copy(source);selected.rows=source.rows[start:];selected.held=[];selected.blocks={r['actual_pts']:source.blocks[r['actual_pts']] for r in selected.rows}
    from collections import OrderedDict
    selected._cache=OrderedDict();selected.scales=source.scales[len(ices[0].scales):];selected.quadratic=source.quadratic[len(ices[0].quadratic):];selected.row_count=len(selected.scales)
    block=selected.block(q,True)
    assert q.tobytes()==before
    result=dict(schema='rinkspace.same-q-current-registered-user-source-truth-append-audit.v1',passed=True,
        complete_q_sha256=identity['q_sha256'],complete_state_reference=ref(STATE/'complete_state_UNACCEPTED.npz'),
        candidate_receipt=ref(receiptpath),protected_same_q_caps=ref(protectedpath),
        parent_manifest_reference=ref(STATE.parent.parent/'input_manifest.json'),
        actual_camera_archive_reproduce_max=camera_error,parent_group_count=82,new_group_count=85,
        original82_all_query_values_dense_guards_supports_bounds_caps_and_uncertainties_exact=True,
        new_source_options={k:ref(new[k]) for k,_ in mapping},families=families,
        new_actual_public_J=dict(rows=len(block.r),columns=block.J.shape[1],nnz=block.J.nnz),
        new_groups=[dict(id=r['id'],primary_max_px=float(np.max(abs(p['query']['residual']))),
            complete_guard_max_px=float(np.max(abs(f['query']['residual'])))) for r,p,f in zip(read(new['--direct-human-truth-authority'])['records'][50:],b['primary'][50:],b['full'][50:])],
        no_source_J_H_or_profile_cache_reused=True,whole_programme_accuracy_accepted=False,seconds=time.perf_counter()-began)
    report=HERE/'same_q27fa_latest_user3_truth85_audit.json';save(report,result)
    valuespath=HERE/'current85_caps_values_bound_to_exact27fa_v1.json'
    save(valuespath,dict(schema='rinkspace.exact-old82-plus-current3-cap-values.v1',audit=ref(report),
        complete_q_sha256=identity['q_sha256'],families=families,this_is_not_a_restoration_packet_until_bound_to_actual_new_startup_manifest=True))
    print(json.dumps(dict(report=str(report),values=str(valuespath),groups=result['new_groups'],seconds=result['seconds'])))

if __name__=='__main__':main()
