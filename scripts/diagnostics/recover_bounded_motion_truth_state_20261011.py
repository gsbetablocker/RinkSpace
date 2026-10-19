"""Recover the saved accepted state's missing original continuation packet.

Uses the exact completed SOC dynamic guard values and immutable parent limits.
No source evaluation, image decoding, J, optimization or model change occurs.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install
install()
from run_current_named66_full_programme import (
    truth_restoration_state_packet,load_truth_restoration_state,
    truth_restoration_source_target,qtoken,
)

RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q8_bounded_motion_nuisance_fix_20261011')
ACCEPTED='8dcd3b9104dd1281bfcb6c09e2f7edb62af4e2712351ec2e7acfba1c4ce94817'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stage=RUN/'accepted_states'/ACCEPTED
    identity=read(stage/'state_identity.json');manifest=read(RUN/'input_manifest.json')
    statepath=stage/'complete_state_UNACCEPTED.npz'
    if sha(statepath)!=identity['state_npz_sha256']or sha(RUN/'input_manifest.json')!=identity['input_manifest_sha256']:
        raise ValueError('Exact restored state/manifest bytes required')
    with np.load(statepath,allow_pickle=False)as arrays:
        q=arrays['q'].copy();initial=arrays['initial_q'].copy()
    if qtoken(q)!=ACCEPTED or identity['q_sha256']!=ACCEPTED:
        raise ValueError('Accepted complete state identity differs')
    resume=read(RUN/'actual_truth_restoration_state_resume.json')
    parentpath=Path(resume['path']);parent=read(parentpath)
    if sha(parentpath)!=resume['sha256']or qtoken(initial)!=resume['complete_q_sha256']:
        raise ValueError('Exact parent cap authority and initial complete state required')
    if parent['source_target']!=truth_restoration_source_target(manifest):
        raise ValueError('Original complete source target changed')
    iterations=read(RUN/'solver/iterations.json')
    if len(iterations)!=1 or iterations[0]['accepted'] is not True:
        raise ValueError('One completed actual accepted cycle required')
    iteration=iterations[0];report=iteration['global_schur_linear_report']
    trials=[row for row in iteration['global_schur_trial_records']if row.get('accepted')]
    if len(trials)!=1:raise ValueError('Unique accepted original trial required')
    trial=trials[0];recovery=trial['trial_feasibility_recovery']['complete_state_recovery_identity']
    if (recovery['proposed_q_sha256']!=ACCEPTED or recovery['recovered_q_sha256']!=ACCEPTED
            or recovery['changed_complete_state_parameter_indices']!=[]):
        raise ValueError('Saved SOC values must describe the unretracted accepted state')
    selected=np.asarray(report['global_parameter_indices'],int)
    displacement=np.asarray(trial['shared_step'],float)
    if not np.array_equal(initial[selected]+displacement,q[selected]):
        raise ValueError('Accepted shared displacement differs from saved measured truth state')
    candidates=[row for row in report['independent_trial_fraction_source_steps']
                if row['accepted_numerical_direction']is True and row['alpha']==trial['line_search_fraction']]
    if len(candidates)!=1:raise ValueError('Unique matching independent SOC direction required')
    candidate=candidates[0]
    measurements=candidate['QP_and_SOC']['sequential_measured_shared_source_curvature_tightening']['actual_original_nonlinear_feasibility_progress']['measurements']
    measurement=measurements[-1]
    if (measurement['direction_sha256']!=candidate['direction_sha256']
            or measurement['current_complete_KKT_direction_retained']is not True
            or measurement.get('current_candidate_discarded',False)):
        raise ValueError('Final original measured guard direction was not retained')
    actual=measurement['actual_original_face_feasibility']
    if actual['complete_original_source_groups']!=86 or set(actual['families'])!=set(parent['families']):
        raise ValueError('Every original protected truth family required')
    progress=read(RUN/'actual_joint_source_truth_restoration_progress.json')
    if progress['complete_q_sha256']!=ACCEPTED:raise ValueError('Same-q review progress required')
    factors={};limits={};uncertainties={};ratcheted=0
    for family,record in parent['families'].items():
        v=np.asarray(actual['families'][family]['actual_group_violations'],float)
        old=np.asarray(record['limits'],float);unc=np.asarray(record['uncertainties'],float)
        if v.shape!=old.shape or not np.isfinite(v).all()or np.any(v<0.)or np.any(v>old+1e-8):
            raise ValueError('Accepted original source cap failed: '+family)
        saved=next(row for row in progress['groups']if row['family']==family)
        if float(v.max())!=saved['maximum_violation_px']or float(v@v)!=saved['squared_violation']:
            raise ValueError('Final SOC truth values do not exactly reproduce accepted review progress')
        restored=np.where(v<=1e-8,0.,v+unc)
        limits[family]=np.minimum(old,restored);uncertainties[family]=unc.copy()
        ratcheted+=int(np.count_nonzero(limits[family]<old))
        # These identity-only factor views are authenticated by the exact
        # parent loader/resume receipt. No new source factor or guard is claimed.
        key='record'if family=='wide'else'row'
        factors[family]=SimpleNamespace(contract={'authority':record['authority']},
            records=[{key:{'id':name}}for name in record['group_ids']])
    packet=truth_restoration_state_packet(q,stage,RUN/'input_manifest.json',manifest,factors,limits,uncertainties)
    target=stage/'truth_restoration_state.json'
    encoded=json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if target.exists():
        if read(target)!=packet:raise ValueError('Preserve an existing different truth restoration packet')
    else:target.write_text(encoded,encoding='utf8')
    loaded=load_truth_restoration_state(target,q,manifest,factors)
    if loaded!=packet:raise ValueError('Original production continuation loader changed restored packet')
    print(json.dumps(dict(path=str(target),sha256=sha(target),complete_q_sha256=ACCEPTED,
        groups=sum(len(f['group_ids'])for f in packet['families'].values()),ratcheted_groups=ratcheted,
        same_q_shared_displacement_bitexact=True,saved_dynamic_guard_values_match_accepted_review_bitexact=True,
        production_packet_and_continuation_loader_verified=True,source_evaluations=0,new_source_images=0)))


if __name__=='__main__':main()
