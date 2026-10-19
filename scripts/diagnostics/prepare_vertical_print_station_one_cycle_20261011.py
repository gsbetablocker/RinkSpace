"""Bind one authorized complete-cycle validation to the stopped latest state."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from z190_joint_source_solver import _frozen_profile_token
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_samea454_actual947_minimal_pivot_trial_resume_memory_sha')
FINAL='2766454eeb8282632e6bf1226d0305ffcda70b74fbe7d916b2614b8672f8b308'
OLD=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_samea454_actual947_roles_SOC_sufficient_minimal_pivot_trial_20261011.json'
DEST=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
OUTPUT=RUN.parent/'okada_review86_q276_vertical_print_station_one_cycle_20261011'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    checkpoint=RUN/'solver/latest_accepted_state.npz';accepted=RUN/'accepted_states'/FINAL
    caps=accepted/'truth_restoration_state.json';packet=read(caps)
    with np.load(checkpoint,allow_pickle=False)as z:
        q=z['q'].copy();profiles={k[9:]:z[k].copy()for k in z.files if k.startswith('profile__')}
        damping=float(z['damping']);fields=len(z.files);scales_sha=array_sha(z['scales']);mask_sha=array_sha(z['quadratic'])
    with np.load(accepted/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:np.testing.assert_array_equal(q,z['q'])
    assert array_sha(q)==FINAL==packet['complete_q_sha256'] and len(q)==34340
    token=_frozen_profile_token(SimpleNamespace(iter_checkpoint_arrays=lambda:profiles.items()))
    identities={}
    for family,entry in packet['families'].items():
        identities[family]=dict(group_ids=entry['group_ids'],authority=entry['authority'],
            limits_float64_sha256=array_sha(np.asarray(entry['limits'],float)),
            uncertainties_float64_sha256=array_sha(np.asarray(entry['uncertainties'],float)),
            zero_limits=int(np.count_nonzero(np.asarray(entry['limits'])==0)))
    assert sum(len(x['group_ids'])for x in identities.values())==86
    cfg=copy.deepcopy(read(OLD));options=cfg['source_options']
    options['--camera-seed']=str(checkpoint);options['--truth-restoration-state']=str(caps)
    options['--initial-damping']=damping
    options.pop('--truth-restoration-source-role-proof',None) # source target is already the same 947-role packet
    options.pop('--save-initial-linearization-cache',None) # live fresh normal is reused in memory, without another bulk J cache
    cfg.pop('saved_full_direction_retraction_comparison',None)
    continued=cfg['continuation'];continued.update(complete_accepted_state=str(checkpoint),
        checkpoint_file_sha256=sha(checkpoint),truth_restoration_state=str(caps),truth_restoration_state_sha256=sha(caps),
        state=dict(q_sha256=FINAL,parameters=34340,fields=fields,frozen_profile_token=token,damping=damping),
        previous_input_configuration=ref(OLD),original_manual_group_cap_and_uncertainty_identity=identities,
        published_same_q_complete_state_reference=ref(accepted/'complete_state_UNACCEPTED.npz'),
        latest_profile_array_references=[dict(name=k,shape=list(v.shape),dtype=v.dtype.str,sha256=array_sha(v))for k,v in profiles.items()],
        source_scales_sha256=scales_sha,source_quadratic_mask_sha256=mask_sha)
    audit=Path(str(RUN)+'.bootstrap')/'native29_saved_direction_causal_audit.json';a=read(audit)
    for item in a['repaired_station_model_sources']:assert sha(item['path'])==item['sha256']
    cfg['one_complete_cycle_validation']=dict(user_authorization='Current user steering delegated by root on 2026-10-11: one complete validation of the diagnosed vertical-print station correction.',
        output=str(OUTPUT),launcher='scripts/diagnostics/start_solver_speed_candidate.ps1 -OneCompleteCycle',
        accepted_steps=1,unbounded_continuation=False,restart_on_failure=False,
        diagnosis_evidence=ref(audit),implementation=a['repaired_station_model_sources'],
        physical_and_all86_truth_and_complete_source_cost_gates_unchanged=True,
        source_image_extraction_and_cache_writes=0,new_complete_source_J_cache_written=False,
        acceptance_alone_not_a_whole_nine_shot_accuracy_claim=True)
    assert not OUTPUT.exists()and not Path(str(OUTPUT)+'.bootstrap').exists()
    DEST.write_text(json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(config=ref(DEST),output=str(OUTPUT),checkpoint=ref(checkpoint),truth_caps=ref(caps),
        q_sha256=FINAL,frozen_profile_token=token,groups=86,source_rows=4502662,normal_cache_written=False),ensure_ascii=False))
if __name__=='__main__':main()
