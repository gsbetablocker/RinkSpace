"""Compare actual saved clocks with original/current preparation only; no FK."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT=Path('D:/OneDrive/codex_projs/RinkSpace')
BASE=Path('D:/CodexData/RinkSpace_Data/runs/body_s01_20261029')
OUT=BASE/'blade_support_profile_1105'
sys.path.insert(0,str(PROJECT))
from calibration.program_tracking.body_tracking.spin_grasp_factors import SpinGraspFactors

backup=OUT/'implementation_before_phase_clip/spin_grasp_factors.py'
spec=importlib.util.spec_from_file_location('calibration.program_tracking.body_tracking._spin_grasp_before_phase_clip',backup)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(cls,pts,seconds,relations):
    times=(pts-pts[0])/12800.
    reference=SimpleNamespace(p={'pts':pts},times=times,grasp_geometry={},at=lambda t:{'P':np.zeros((len(t),3))})
    state=SimpleNamespace(seconds=seconds,m=1,basis=lambda t:np.ones((len(t),1)))
    problem=SimpleNamespace(reference=reference,body=SimpleNamespace(state=state,tau=.01),binding=None)
    factor=cls(problem,relations)
    cfg=relations['factor_parameters']['continuous_support']
    support_pts=factor.pts[factor.support_mask]
    coefficient=np.sqrt(cfg['weight']*factor.support_dt*factor.support_phase_weight[factor.support_mask])/cfg['scale_m']
    return dict(all_selected_pts=factor.pts,support_pts=support_pts,support_dt=factor.support_dt,
        grip_pts=factor.pts[factor.held],grip_dt=factor.grip_weights,support_coefficient_per_m=coefficient,
        counts=dict(selected=len(factor.pts),support=len(support_pts),grip=int(factor.held.sum())),
        maximum_support_dt=float(factor.support_dt.max()),
        maximum_support_time_gap_s=float(np.diff(factor.support_times).max()),
        total_support_quadrature_seconds=float(factor.support_dt.sum()))


def main():
    roots=dict(local_281=BASE/'spin_local_280_1105/fit1/continuous_stage',entry=BASE/'blade_orientation_1105/entry_81_2_82_0',support=BASE/'blade_orientation_1105/support_87_6_90_0')
    cases={}
    for name,root in roots.items():
        with np.load(root/'solve/candidate.npz',allow_pickle=False) as z:seconds=z['seconds'].copy()
        with np.load(root/'common/exmr_hand_nofoot/fitted/result.npz',allow_pickle=False) as z:pts=z['pts'].copy()
        relations=js(root/'grasp_relations.json')
        cases[name]=dict(source_result=str(root/'common/exmr_hand_nofoot/fitted/result.npz'),seconds=seconds,
            before=prepare(old.SpinGraspFactors,pts,seconds,relations),after=prepare(SpinGraspFactors,pts,seconds,relations))
        chosen=cases[name]['after']['all_selected_pts']/12800.
        assert np.all((chosen>=seconds[0])&(chosen<=seconds[1]))
    root=roots['local_281']
    with np.load(root/'common/exmr_hand_nofoot/fitted/result.npz',allow_pickle=False) as z:pts=z['pts'].copy()
    relations=js(root/'grasp_relations.json')
    for name,seconds in [('full_76_95',(76.,95.)),('planned_spin_context',(86.4,92.86))]:
        cases[name]=dict(seconds=seconds,before=prepare(old.SpinGraspFactors,pts,seconds,relations),after=prepare(SpinGraspFactors,pts,seconds,relations))
    a=cases['full_76_95'];equal_keys=['all_selected_pts','support_pts','support_dt','grip_pts','grip_dt','support_coefficient_per_m']
    full_equality={k:np.array_equal(a['before'][k],a['after'][k]) for k in equal_keys}
    assert all(full_equality.values())
    result=dict(schema='rinkspace.phase-context-clip-comparison.v1',
        scope='actual archived native PTS/current candidate seconds/relations passed through original and current constructors with clock/basis stubs; no body/FK, inference, objective perturbation or optimization',
        source_before=str(backup),source_before_sha256=sha(backup),
        source_after=str(PROJECT/'calibration/program_tracking/body_tracking/spin_grasp_factors.py'),source_after_sha256=sha(PROJECT/'calibration/program_tracking/body_tracking/spin_grasp_factors.py'),
        test_file=str(PROJECT/'tests/test_spin_grasp_sampling.py'),
        regression=dict(command='python -B -m pytest tests/test_spin_grasp_sampling.py tests/test_boot_axis_factors.py -q -p no:cacheprovider',
            python='D:/CodexData/RinkSpace_Data/runtime_tools/repository_validation/cpu-python311-20261011/Scripts/python.exe',passed=10),
        unchanged_full_context_exact_equality=full_equality,
        changes=['Only native phase-boundary rows inside the body source context are admitted.',
            'Empty grip groups have empty quadrature/residual and a correctly shaped zero-row Jacobian.',
            'Nonempty grip/support formulas, support intervals, fixed observations, shoe geometry and objective coefficients are unchanged.'],
        cases=cases)
    def default(v):return v.tolist() if isinstance(v,np.ndarray) else v.item()
    path=OUT/'phase_clip_before_after.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=default),encoding='utf-8')
    print(json.dumps(dict(path=str(path),full_context_exact=full_equality,cases={name:dict(before=v['before']['counts'],after=v['after']['counts'],support_dt_max_before=v['before']['maximum_support_dt'],support_dt_max_after=v['after']['maximum_support_dt']) for name,v in cases.items()}),ensure_ascii=False))


if __name__=='__main__':main()
