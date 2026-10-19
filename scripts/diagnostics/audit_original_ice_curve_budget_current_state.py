"""Actual original ICE R/one public world-FD column; no graph/solver/full J."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import argparse,copy,json,hashlib,os,pickle,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2];DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
SYS=DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
from calibration.solver_runtime import install
install()
import audit_current_user_ice_truth as a
from calibration.solver_runtime.original_ice_curve_integration import SCHEMA,wrap_original_ice_curve_integration
from calibration.solver_runtime.original_board_curve_integration import file_reference
from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
from calibration.solver_runtime.original_source_point_qualification import OriginalSourcePointQualification
from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction
from structured_programme_source_candidate.original_review_source import load_original_absolute_evaluators
from z190_streaming_source_normal import SourceRowBlock,accumulate_source_normal
from calibration.solver_runtime.source_loss import weighted_source_cost,source_irls_weights
import numpy as np
from scipy.sparse import csr_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    opt=p.parse_args();began=time.perf_counter();options=a.read(opt.config)['source_options']
    queuepath=SYS/'original_absolute_ice_FIT_review_queue_20261010.json';queue=a.read(queuepath)
    indices=[r['original_packet_row_index'] for r in queue['rows']];assert len(indices)==79
    correction=a.read(options['--original-ice-pixel-correction-source']);parent=correction['original_absolute_packet']
    policy=SYS/'original_ICE_all79_physical_curve_per_exposure_integration_POLICY_20261011.json'
    a.save(policy,dict(schema=SCHEMA,curve_budget=1.,enable_point_compression=False,support_basis='isolated_authenticated_source_anchors',
        source_video_sha256=correction['source_video_sha256'],original_packet=parent,original_packet_row_indices=indices,
        original_source_inventory=file_reference(queuepath),source_membership_is_not_new_pixel_identity_verification=True,
        UV_sigma_WORLD_human_support_and_roles_not_modified=True))
    entries=pickle.loads(Path(parent['path']).read_bytes())['programme_absolute_rows']
    manifest=a.read(opt.state.parent.parent/'input_manifest.json')
    state,q,identity,error=a.load_state(opt.state,Path(options['--world-init']),manifest);assert error==0.
    before=q.tobytes();state.q=q
    blocks=[];names=[]
    for e in entries:
        if e['collection_role']=='FIT' and e['block']not in names:names.append(e['block'])
    for name in names:
        source=[e for e in entries if e['collection_role']=='FIT' and e['block']==name]
        groups=[None]*(max(e['group_order']for e in source)+1)
        for e in source:
            gi=e['group_order'];ri=e['row_order'];actual=int(e['group_actual_pts'])
            if groups[gi]is None:groups[gi]=(actual,[])
            assert ri==len(groups[gi][1]);groups[gi][1].append(copy.deepcopy(e['source_row']))
            state.original.aliases[int(e['source_row']['pts'])]=actual;state.original.aliases[actual]=actual
        count=sum(2*len(r['uv'])for _,rr in groups for r in rr)
        blocks.append(SimpleNamespace(name=name,groups=groups,quadratic=np.zeros(count,bool),bundle=SimpleNamespace(motion_rows=0,source_residual_rows=count)))
    state.blocks=blocks
    register=set(indices)|{r['original_packet_row_index']for r in correction['records']}
    for i in register:state.geometry.register_row(entries[i]['source_row'],actual_pts=int(entries[i]['group_actual_pts']))
    install_original_ice_pixel_correction(state,options['--original-ice-pixel-correction-source'])
    qualification=OriginalSourcePointQualification(options['--original-source-point-qualification-source']);qualification.qualify_state(state)
    exclusions={r['record']['original_packet_row_index']:r['record']['excluded_point_indices']for r in qualification.records}
    byname={b.name:b for b in blocks}
    for i,excluded in exclusions.items():
        e=entries[i];b=byname[e['block']];rr=b.groups[e['group_order']][1];r=rr[e['row_order']]
        rr[e['row_order']]=slice_observed_points(r,np.setdiff1d(np.arange(len(r['uv'])),excluded))
    for b in blocks:
        count=sum(2*len(r['uv'])for _,rr in b.groups for r in rr);b.quadratic=np.zeros(count,bool);b.bundle.source_residual_rows=count
    actors={b.name:b for b in blocks};locations={};residuals={};matrices={};scales={};ledgers={}
    for b in blocks:
        cursor=0;locations[b.name]={}
        for gi,(_,rr)in enumerate(b.groups):
            for ri,r in enumerate(rr):locations[b.name][(gi,ri)]=np.arange(cursor,cursor+2*len(r['uv']));cursor+=2*len(r['uv'])
        residuals[b.name]=np.zeros(cursor);scales[b.name]=np.concatenate([np.repeat(np.broadcast_to(r['sigma'],(len(r['uv']),)),2)for _,rr in b.groups for r in rr])
        matrices[b.name]=np.zeros(cursor);ledgers[b.name]=dict(family='programme:'+b.name,source_sha256=correction['source_video_sha256'])
    evaluators,_=load_original_absolute_evaluators()
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={int(p):v.copy()for p,v in zip(z['pts'],z['camera25'])}
    world=q[state.geometry_parameter_indices];h=1e-6;plus=world.copy();minus=world.copy();plus[14]+=h;minus[14]-=h
    checks=[];compute=time.perf_counter()
    for i in indices:
        e=entries[i];b=actors[e['block']];row=b.groups[e['group_order']][1][e['row_order']];slots=locations[b.name][(e['group_order'],e['row_order'])]
        def value(w):return np.asarray(evaluators[b.name](cameras[e['group_actual_pts']], [state.geometry.rebind(row,w)],False),float)
        r=value(world);j=(value(plus)-value(minus))/(2*h)
        assert len(r)==len(slots);residuals[b.name][slots]=r;matrices[b.name][slots]=j
        checks.append(dict(original_packet_row_index=i,id=row['id'],actual_pts=int(e['group_actual_pts']),original_points=len(e['source_row']['uv']),
            active_points=len(row['uv']),actual_original_normalized_residual=r.tolist(),actual_world14_public_directional_FD=j.tolist()))
    r_seconds=time.perf_counter()-compute
    source_blocks=[]
    for b in blocks:
        j=matrices[b.name];matrix=csr_matrix(j[:,None]);source_blocks.append(SourceRowBlock(residuals[b.name],matrix,b.quadratic,scales[b.name],ledgers[b.name]))
    class Source(SimpleNamespace):
        def blocks(self,q,jacobian=True):yield from source_blocks
    base=Source(count=1,scales=np.concatenate(list(scales.values())),quadratic=np.concatenate([b.quadratic for b in blocks]),contract={})
    owner=wrap_original_ice_curve_integration(base,state,policy);revised=list(owner.blocks(np.zeros(1)))
    rowmap={r['original_packet_row_index']:r for r in owner.original_ice_curve_integration['source_rows']}
    oldcost=newcost=0.;oldinfo=newinfo=0.;changed=0;points=0
    for check in checks:
        i=check['original_packet_row_index'];e=entries[i];name=e['block'];slots=locations[name][(e['group_order'],e['row_order'])]
        index=names.index(name);raw=source_blocks[index];weighted=revised[index]
        assert raw.r is weighted.r and raw.J is weighted.J and raw.scales is weighted.scales and raw.quadratic is weighted.quadratic
        w=weighted.loss_weights.integration_weights[slots];c=weighted.loss_weights.working_normalization_multiplier
        c=np.ones(len(slots)) if c is None else c[slots];sigma=weighted.scales[slots];r=raw.r[slots]
        oldinfo+=float(np.sum(1/sigma[::2]**2));newinfo+=float(np.sum(w[::2]*c[::2]**2/sigma[::2]**2))
        from calibration.solver_runtime.source_loss import SourceLossWeights
        payload=SourceLossWeights(w,c);oc=weighted_source_cost(r,raw.quadratic[slots],None);nc=weighted_source_cost(r,raw.quadratic[slots],payload)
        oldcost+=oc;newcost+=nc;changed+=int(np.count_nonzero((abs(r)>2.)!=(abs(r*c)>2.)))
        check.update(rowmap[i],integration_weights=w[::2].tolist(),working_normalization_multiplier=c[::2].tolist(),old_cost=oc,new_cost=nc,
                     effective_new_inverse_working_sigma_squared_sum=float(np.sum(w[::2]*c[::2]**2/sigma[::2]**2)))
        points+=check['active_points']
    assert points==668 and sum(c['original_points']for c in checks)==684 and q.tobytes()==before
    oldn=[accumulate_source_normal(lambda b=b:iter((b,)),1,expected_row_count=len(b.r),frozen_profile_token='same-q-actual-world14')for b in source_blocks]
    newn=[accumulate_source_normal(lambda b=b:iter((b,)),1,expected_row_count=len(b.r),frozen_profile_token='same-q-actual-world14')for b in revised]
    # No extra source operator calls in apply/stream/normal algebra.
    started=time.perf_counter()
    for _ in range(100):
        for b in source_blocks:owner.apply_block(b)
    apply_seconds=(time.perf_counter()-started)/100
    groups=owner.original_ice_curve_integration['groups'];duplicates=sum(g['stored_measured_copies']-g['unique_source_operator_UV_anchors']for g in groups)
    result=dict(schema='rinkspace.actual-original-ICE-physical-curve-budget-fixed-q-one-column-audit.v1',passed=True,
        complete_state_reference=file_reference(opt.state/'complete_state_UNACCEPTED.npz'),complete_q_sha256=identity['q_sha256'],
        source_config_reference=file_reference(opt.config),policy=file_reference(policy),contract=owner.original_ice_curve_integration,
        original_ICE_rows=79,original_ICE_UV=684,active_ICE_UV=points,active_ICE_scalars=2*points,
        original_active_R_J_scales_quadratic_and_human_support_unchanged=True,actual_original_operator_R_and_one_world14_FD_column=True,
        full_programme_J_graph_or_solver_not_created=True,original_packet_bytes_untouched=True,
        exact_same_operator_UV_duplicates=duplicates,additional_point_pruning=0,
        no_new_spacing_or_window_rule=True,old_sum_inverse_stored_sigma_squared=oldinfo,new_sum_inverse_working_sigma_squared=newinfo,
        actual_original_ICE_old_cost=oldcost,actual_original_ICE_new_cost=newcost,working_Huber_classification_changed_scalars=changed,
        actual_world14_old_gradient=sum(float(n.gradient[0])for n in oldn),actual_world14_new_gradient=sum(float(n.gradient[0])for n in newn),
        actual_world14_old_normal=sum(float(n.normal[0,0])for n in oldn),actual_world14_new_normal=sum(float(n.normal[0,0])for n in newn),
        actual_original_R_and_oneFD_seconds=r_seconds,policy_preparation_seconds=owner.original_ice_curve_integration_preparation_seconds,
        all_families_apply_average_seconds=apply_seconds,total_seconds=time.perf_counter()-began,rows=checks,accuracy_accepted=False)
    report=SYS/'actual_original_ICE79_fixed_qfe074_curve_budget_R_oneFD_and_loss_20261011.json';a.save(report,result)
    print(json.dumps({k:v for k,v in result.items()if k in ('active_ICE_UV','active_ICE_scalars','exact_same_operator_UV_duplicates','old_sum_inverse_stored_sigma_squared','new_sum_inverse_working_sigma_squared','actual_original_ICE_old_cost','actual_original_ICE_new_cost','policy_preparation_seconds','all_families_apply_average_seconds','total_seconds')}|dict(report=str(report)),ensure_ascii=True))

if __name__=='__main__':main()
