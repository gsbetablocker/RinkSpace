"""Read-only diagnosis and publication of a stopped, non-accepted candidate."""
from pathlib import Path
import json
import numpy as np
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .contact_feasibility import physical_feasibility
from .skate_physics import FEASIBILITY_TOLERANCE,SUPPORT_BAND


def source_support_shift_intervals(packet,evidence):
    """Instantaneous metric root-Z intervals; never modify the candidate.

    The body/attachment/shape/orientation stay fixed. A common vertical shift
    adds equally to every skate feature height and its semantic clearance.
    Empty intervals diagnose a fixed-pose conflict, not global infeasibility.
    """
    pts=np.asarray(packet['pts']);clear=np.asarray(packet['skate_clearance_m_mid'])
    support=np.asarray(packet['skate_support_gap_m_mid']);rows=[]
    for source in evidence:
        ids=np.flatnonzero(pts==source['pts'])
        row=dict(pts=int(source['pts']),source_seconds=source['pts']/12800.)
        if len(ids)!=1:
            rows.append(dict(row,status='unknown',reason='source exposure absent or duplicated'));continue
        i=int(ids[0]);height=float(support[i])
        if not np.isfinite(height) or not np.isfinite(clear[i]).all():
            rows.append(dict(row,status='unknown',reason='nonfinite physical height'));continue
        nonpenetration_lower=-float(clear[i].min())
        lower=max(nonpenetration_lower,SUPPORT_BAND[0]-height);upper=SUPPORT_BAND[1]-height
        rows.append(dict(row,status='fixed_pose_vertical_diagnostic',support_height_m=height,
            per_foot_minimum_semantic_clearance_m=clear[i].tolist(),
            nonpenetration_shift_lower_bound_m=nonpenetration_lower,
            all_constraints_shift_interval_m=[lower,upper],
            interval_nonempty=bool(lower<=upper),
            empty_interval_gap_m=max(0.,lower-upper)))
    return dict(schema='rinkspace.source-support-vertical-diagnostic.v1',applied=False,
        units='conditional scene metres',support_band_m=list(SUPPORT_BAND),bands_calibrated=False,
        scope='reviewed source exposure midpoints; same frozen pose, static attachment and shared template',
        limitations='No optimization. Neither an image-quality test nor a neighboring-time or global feasibility proof.',rows=rows)


def inspect_checkpoint(problem,saved,path,out):
    """No minimization, parameter updates or tolerance calibration occurs here."""
    out=Path(out);x=saved['x'].copy()
    snapshot=out/'candidate.npz';prior_report={}
    # Publication must retain the actual budget/quality stop from the solve.
    # Only reuse a report already attached to these exact checkpoint bytes.
    if snapshot.exists() and (out/'candidate.json').exists() and snapshot.read_bytes()==Path(path).read_bytes():
        prior_report=json.loads((out/'candidate.json').read_text(encoding='utf8'))
    if 'physics_times' in saved:problem.set_physics_times(saved['physics_times'])
    if 'multiplier' in saved:
        if len(saved['multiplier'])!=len(problem.al.multiplier):raise ValueError('Saved physical row layout differs')
        problem.al.multiplier=saved['multiplier'].copy()
        problem.al.penalty=float(saved['penalty'])
    value=problem.evaluate(x,False)
    dense=np.unique(np.r_[np.arange(*problem.body.state.bounds_time,.0025),problem.physics_times,
                          problem.body.state.bounds_time])
    check=problem.contact(x,times=dense)
    gate=physical_feasibility(check['margin_m'],tolerance_m=FEASIBILITY_TOLERANCE,row_labels=check['labels'])
    core=np.array([35.2e9-1<=row[0]<=36.8e9+1 for row in check['labels']])
    core_gate=physical_feasibility(check['margin_m'][core],tolerance_m=FEASIBILITY_TOLERANCE,
        row_labels=[row for row,used in zip(check['labels'],core) if used])
    t=np.array([35.20-problem.origin]);contact=problem.contact(x,times=t)
    features=contact['features'];heights=features['height'][0]
    eligible=[i for i,n in enumerate(features['ids']) if n in
              problem.binding.packet()['template']['support_feature_ids']['glide']]
    supports=heights[:,eligible].min(1)
    clear=contact['clearance'][0]
    height=float(supports.min())
    lower=max(float(-clear.min()),SUPPORT_BAND[0]-height)
    upper=SUPPORT_BAND[1]-height
    state=problem.body.state;basis=state.basis(t)[0]
    # A single explicit counterfactual diagnoses coupling; it is not a fit
    # and never replaces the accepted control vector. Root translation leaves
    # the static body/boot attachment unchanged.
    trial=x.copy();body=trial[17:].reshape(state.m,state.d)
    column=int(np.flatnonzero(state.active==2)[0])
    body[:,column]+=upper/problem.binding.ratio*basis/float(basis@basis)
    trial_contact=problem.contact(trial,times=dense)
    trial_gate=physical_feasibility(trial_contact['margin_m'][core],tolerance_m=FEASIBILITY_TOLERANCE,
        row_labels=[row for row,used in zip(trial_contact['labels'],core) if used])
    trial_value=problem.evaluate(trial,False)
    counterfactual=dict(applied_to_candidate=False,optimization_performed=False,
        construction='minimum-norm root-Z coefficients for the one 35.20s support upper bound; other coordinates unchanged',
        desired_vertical_shift_m=upper,core_physical_gate_before=core_gate,core_physical_gate_after=trial_gate,
        raw_image_cost_before=value.get('image_cost'),raw_image_cost_after=trial_value.get('image_cost'),
        raw_pose_cost_before=value.get('pose_cost'),raw_pose_cost_after=trial_value.get('pose_cost'))
    problem.binding.x=x[:17].copy()
    content=Path(path).read_bytes()
    if snapshot.exists() and snapshot.read_bytes()!=content:
        archive=out/'history'/digest(snapshot)[:12];archive.mkdir(parents=True,exist_ok=True)
        (archive/'candidate.npz').write_bytes(snapshot.read_bytes())
    snapshot.write_bytes(content)
    diagnosis=dict(source_checkpoint=str(path),sha256=digest(path),no_optimization=True,
        at_source_seconds=35.20,actual_body_interval=list(state.seconds),
        outer_basis_sum=float(basis.sum()),nonzero_basis_indices=np.flatnonzero(basis).tolist(),
        state_columns=state.active.tolist(),
        minimum_clearance_m_per_foot=clear.tolist(),support_height_m_per_foot=supports.tolist(),
        feature_height_m={name:heights[:,i].tolist() for i,name in enumerate(features['ids'])},
        root_vertical_only_legal_shift_interval_m=[lower,upper],
        root_vertical_only_interval_nonempty=bool(lower<=upper),
        interpretation='Instantaneous vertical diagnostic only; not a trajectory or image-quality feasibility proof',
        support_band_m=list(SUPPORT_BAND),feasibility_tolerance_m=FEASIBILITY_TOLERANCE,
        source_support_optimization_weights=[dict(label=list(label),weight=float(weight)) for label,weight in
            zip(value.get('raw_ice',value['contact'])['labels'],value.get('raw_ice',value['contact'])['weights']) if label[0]==35200000000 and 'source_support' in label[1]],
        root_vertical_counterfactual=counterfactual,
        threshold_status='first-round debugging values, not calibrated acceptance criteria')
    write_json(out/'start_3520_diagnosis.json',diagnosis)
    report=dict(prior_report,event='candidate_saved',stop=prior_report.get('stop','caller stopped for diagnosis and human visual review'),
        candidate_checkpoint=str(snapshot),source_checkpoint_sha256=digest(path),
        source_seconds=list(state.seconds),parts=value['parts'],**value['gate'],dense_gate=gate,core_dense_gate=core_gate,
        raw_data_parts=value.get('raw_data_parts',{}),
        temporal_check_step_s=.0025,finite_sampling_is_not_continuous_proof=True,
        dense_gate_scope='skate nonpenetration and explicitly supported source exposure; foot/collar tested at every source midpoint and neutral pose',
        numerical_convergence=False,user_accepted=False,acceptance_calibrated=False,
        support_parameters_status='debug only; 3 mm upper and 1 mm slack remain uncalibrated',
        initial_penalty=float(problem.al.penalty),automatic_penalty_increase_stopped=True)
    write_json(out/'candidate.json',report)
    return report
