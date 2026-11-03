"""Check saved S01 footwear kinematics without fitting or neural inference.

Preserved from the run-local check_invariants.py. The report destination is
explicit so that historical invariants and candidate files remain unchanged.
These are engineering invariants, not anatomical or 3D accuracy acceptance.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from rinkspace.path_policy import guard_project_write
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.prepare import FullSurface
from .footwear_binding import FootwearBinding, IntegratedBodyState
from .fullbody_state import camera_binding, WindowState
from .rebuild_prepare import load, digest


def maximum(a):
    return float(np.max(np.abs(a)))


def rigid(R, t, points):
    return np.einsum('ij,...j->...i', R, points)+t


def run(root, output, *, report_path, project=None, data_root=None, final=False):
    """Read existing exports and write only the explicitly selected report."""
    PROJECT = Path(project or Path(__file__).resolve().parents[3]).resolve()
    ROOT, OUT = Path(root).resolve(), Path(output).resolve()
    DATA = Path(data_root).resolve() if data_root is not None else ROOT.parents[1]
    report_path = guard_project_write(report_path)
    if report_path.exists():
        raise FileExistsError(report_path)

    def check(name, value, tolerance=1e-10, **extras):
        results[name] = dict(maximum_absolute=float(value), tolerance=float(tolerance),
            passed=bool(np.isfinite(value) and value <= tolerance), **extras)

    reference=IntegratedBodyState(ROOT,ROOT/'fullbody_1102')
    refinement_paths=[]
    if final:
        joint=load(OUT/'joint_result.npz')
        for i in (1,2):
            path=OUT/f'w{i}'/'result.npz'
            z=load(path);window=WindowState(reference,z['seconds'],'full')
            np.testing.assert_array_equal(window.active,z['active'])
            np.testing.assert_allclose(window.knots,z['knots'],rtol=0,atol=0)
            np.testing.assert_array_equal(z['x'],joint['x'][joint['offsets'][i-1]:joint['offsets'][i]])
            reference.refinements.append((window,z['x'].copy()))
            refinement_paths.append(path)
    camera,info,_,_=camera_binding(PROJECT,reference)
    catalog=ResultCatalog(project=PROJECT,data_root=DATA,config=PROJECT/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_fullbody_1102.json')
    surface=FullSurface(catalog)
    binding=FootwearBinding(reference,surface,info)
    pts=np.array([325120,332800,336128,346624,421888,455680,461056,489216,521728],dtype=np.int64)
    indices=np.searchsorted(reference.p['pts'],pts)
    np.testing.assert_array_equal(reference.p['pts'][indices],pts)
    times=reference.times[indices]
    state_input=reference.at(times)
    base=reference.world(state_input)
    coco,joints,_,frames=reference.body.forward_rotations(state_input['local_R'],with_frames=True)
    local_recovered=np.einsum('nji,nkj->nki',base['Q'],base['native']-base['P'][:,None])
    results={}
    check('world_to_body_joint_recovery',maximum(local_recovered-joints))
    zero=binding.evaluate(base['P'],base['Q'],joints,frames,x=np.zeros(14),deformation=False)
    check('zero_registration_disabled_upper_replays_legacy_gear',maximum(zero['gear']-base['gear']))
    check('zero_registration_legacy_boot_rotation',maximum(zero['boot_R']-base['boot_R']))
    registration=OUT/('joint_result.npz' if final else 'registration.npz')
    if registration.is_file():
        binding.x=load(registration)['x'][:14].copy()
    on=binding.evaluate(base['P'],base['Q'],joints,frames,deformation=True)
    off=binding.evaluate(base['P'],base['Q'],joints,frames,deformation=False)
    upper=np.array([4,5,11,12])
    fixed=np.setdiff1d(np.arange(48),upper)
    change=np.linalg.norm(on['gear']-off['gear'],axis=-1)
    check('upper_modes_leave_all_other_gear_vertices_fixed',maximum(change[:,:,fixed]),fixed_vertex_ids=fixed.tolist())
    changed=np.flatnonzero(np.any(change>1e-12,axis=(0,1)))
    results['actual_upper_activity']=dict(changed_vertex_ids=changed.tolist(),allowed_vertex_ids=upper.tolist(),maximum_displacement_solver_m=float(change.max()),maximum_mode_displacement_solver_m=np.max(on['upper_deformation'],axis=(0,1)).tolist(),passed=bool(len(changed)>0 and set(changed).issubset(set(upper))))
    check('upper_modes_leave_blade_world_vertices_fixed',maximum(on['gear'][:,:,14:]-off['gear'][:,:,14:]))
    for field in ('boot_R','boot_O','blade_R','blade_O'):
        check('upper_modes_leave_'+field+'_fixed',maximum(on[field]-off[field]))
    R=Rotation.from_rotvec([.31,-.22,.17]).as_matrix()
    t=np.array([.37,-.24,.19])
    newP=rigid(R,t,base['P']);newQ=R@base['Q']
    moved=binding.evaluate(newP,newQ,joints,frames,deformation=True)
    for field in ('gear','shank_world'):
        check('root_rigid_equivariance_'+field,maximum(moved[field]-rigid(R,t,on[field])))
    for field in ('boot_R','blade_R'):
        check('root_rigid_equivariance_'+field,maximum(moved[field]-R@on[field]))
    for field in ('boot_O','blade_O'):
        check('root_rigid_equivariance_'+field,maximum(moved[field]-rigid(R,t,on[field])))
    check('root_rigid_preserves_effective_ankle_relative_rotation',maximum(moved['effective_ankle_relative_rotation']-on['effective_ankle_relative_rotation']))
    check('root_rigid_preserves_upper_mode_values',maximum(moved['upper_deformation']-on['upper_deformation']))
    shankR=np.stack([base['Q']@frames[:,a] for a in (4,5)],axis=1)
    newShankR=np.stack([newQ@frames[:,a] for a in (4,5)],axis=1)
    relative=np.swapaxes(on['boot_R'],-1,-2)@shankR
    newRelative=np.swapaxes(moved['boot_R'],-1,-2)@newShankR
    check('root_rigid_preserves_shank_to_lower_boot_rotation',maximum(relative-newRelative))
    installed=on['blade_O'][:,:,None]+np.einsum('nfij,kj->nfki',on['blade_R'],binding.blade_L)
    check('blade_local_template_reconstructs_world_blade',maximum(installed-on['gear'][:,:,14:]))
    R_BL_measured=np.swapaxes(on['boot_R'],-1,-2)@on['blade_R']
    t_BL_measured=np.einsum('nfji,nfj->nfi',on['boot_R'],on['blade_O']-on['boot_O'])
    check('blade_rigid_installation_rotation',maximum(R_BL_measured-binding.R_BL))
    check('blade_rigid_installation_translation',maximum(t_BL_measured-binding.t_BL))
    full=surface.sample_vertices(joints,frames,double_precision=True)
    subset=surface.sample_vertices(joints,frames,binding.sample_ids,double_precision=True)
    check('selected_full_surface_vertices_equal_same_fk_full_expansion',maximum(subset-full[:,binding.sample_ids]))
    neutral_full=surface.sample_vertices(binding.neutral_native,binding.neutral_frames,double_precision=True)
    check('stored_neutral_samples_equal_same_fk_full_expansion',maximum(binding.neutral_samples-neutral_full[:,binding.sample_ids]))
    check('neutral_sample_id_mapping_left',maximum(subset[:,binding.foot_indices[0]]-full[:,binding.foot_ids[0]]))
    check('neutral_sample_id_mapping_right',maximum(subset[:,binding.foot_indices[1]]-full[:,binding.foot_ids[1]]))
    saved_path=OUT/'fitted/result.npz' if final else ROOT/'fullbody_1102/full/result.npz'
    saved=load(saved_path)
    ratio=info['length_export_ratio']
    check('evaluated_state_P_matches_saved_fullbody',maximum(base['P']*ratio-saved['P_world_mid'][indices]))
    check('evaluated_state_Q_matches_saved_fullbody',maximum(base['Q']-saved['Q_world_mid'][indices]))
    check('evaluated_state_local_R_matches_saved_fullbody',maximum(state_input['local_R']-saved['local_R_mid'][indices]))
    check('evaluated_state_native_matches_saved_fullbody',maximum(base['native']*ratio-saved['native22_world_mid'][indices]))
    saved_coco,saved_joints,_,saved_frames=surface.body.forward_rotations(saved['local_R_mid'][indices],with_frames=True)
    saved_subset=surface.sample_vertices(saved_joints,saved_frames,binding.sample_ids,double_precision=True)
    live_world=base['P'][:,None]*ratio+np.einsum('nij,nvj->nvi',base['Q'],subset*ratio)
    saved_world=saved['P_world_mid'][indices,None]+np.einsum('nij,nvj->nvi',saved['Q_world_mid'][indices],saved_subset*float(saved['body_length_ratio']))
    check('saved_state_foot_subset_world_matches_live_same_fk',maximum(live_world-saved_world))
    if final:
        check('final_export_gear_scaled_once',maximum(saved['gear_world_mid'][indices]-on['gear']*ratio))
        check('final_export_upper_modes_same_state',maximum(saved['upper_deformation_mid'][indices]-on['upper_deformation']/binding.maximum_modes))
        final_blade_pose=np.concatenate((on['blade_R'],(on['blade_O']*ratio)[...,None]),axis=3)
        check('final_export_blade_pose_scaled_once',maximum(saved['blade_pose_world_mid'][indices]-final_blade_pose))
    cache_evidence=[]
    for meta_path in sorted((DATA/'cache/results3d/s01').glob('*/metadata.json')):
        meta=json.loads(meta_path.read_text(encoding='utf8'))
        if meta.get('identity',{}).get('result')!=digest(saved_path):
            continue
        vertices_path=meta_path.parent/'vertices.npy'
        cached=np.load(vertices_path,mmap_mode='r')
        # The historical display cache intentionally stores float32 world vertices.
        expected=base['P'][:,None]*ratio+np.einsum('nij,nvj->nvi',base['Q'],subset*ratio)
        error=maximum(np.asarray(cached[indices])[:,binding.sample_ids]-expected)
        check('saved_full_surface_foot_subset_matches_same_fk',error,tolerance=8e-6,cache_path=str(vertices_path),storage_dtype=str(cached.dtype))
        cache_evidence.append(str(meta_path))
        break
    if not cache_evidence:
        results['saved_full_surface_cache_comparison']=dict(status='not_run_no_exact_result_hash_cache',passed=True,reason='Live same-FK full/selected expansion and saved fullbody local_R/P/Q comparisons performed instead')
    record=binding.record()
    if final:
        exported_binding=json.loads(str(saved['footwear_binding_json']))
        check('final_saved_T_FB_same_parameters_and_unit',maximum(np.asarray(exported_binding['T_FB'])-np.asarray(record['T_FB'])))
        check('final_saved_T_BL_same_parameters_and_unit',maximum(np.asarray(exported_binding['T_lower_boot_to_blade'])-np.asarray(record['T_lower_boot_to_blade'])))
    T_FB=np.asarray(record['T_FB']);T_BL=np.asarray(record['T_lower_boot_to_blade'])
    RF,tF,_=binding.parameters()
    check('export_T_FB_rotation_preserved',maximum(T_FB[:,:3,:3]-RF))
    check('export_T_FB_translation_scaled_once',maximum(T_FB[:,:3,3]-tF*ratio))
    check('export_T_BL_rotation_preserved',maximum(T_BL[:3,:3]-binding.R_BL))
    check('export_T_BL_translation_scaled_once',maximum(T_BL[:3,3]-binding.t_BL*ratio))
    report=dict(schema='rinkspace.footwear-independent-kinematic-invariants.v1',passed=all(r['passed'] for r in results.values()),no_new_fit=True,no_neural_inference=True,registration_used=registration.is_file(),source_pts=pts.tolist(),source_seconds=(pts/12800).tolist(),source_time_base=[1,12800],state='final joint_result plus two saved body refinements' if final else 'saved 1102 fullbody baseline; optional shared registration, no running joint snapshot',length_unit='legacy solver conditional metres unless field explicitly says exported',length_export_ratio=ratio,foot_sample_count=int(len(binding.sample_ids)),foot_sample_counts=[int(len(a)) for a in binding.foot_ids],source_hashes={str(p):digest(p) for p in [Path(__file__),PROJECT/'calibration/program_tracking/body_tracking/footwear_binding.py',PROJECT/'src/rinkspace/viewer3d/prepare.py',ROOT/'input.npz',ROOT/'compact_smplx.npz',saved_path,registration,*refinement_paths] if p.is_file()},camera=dict(state_sha256=info['state_sha256'],native_library_sha256=info['native_library_sha256']),checks=results,limitations=['Engineering invariants only, not anatomical registration, material, contact or 3D accuracy validation','Final saved joint_result and both saved refinement controls are consumed without refitting' if final else 'No joint_result snapshot is consumed while fitting continues'])
    historical = OUT/'check_invariants.py'
    if historical.is_file():
        report['historical_script'] = dict(path=str(historical), sha256=digest(historical))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project')
    parser.add_argument('--data-root')
    parser.add_argument('--root', required=True, help='Existing body_s01 run directory')
    parser.add_argument('--output', required=True, help='Existing boot integration directory')
    parser.add_argument('--report', required=True, help='New report file; existing files are refused')
    parser.add_argument('--final', action='store_true', help='Consume saved joint_result and both body refinements')
    args = parser.parse_args(argv)
    report = run(args.root, args.output, project=args.project, data_root=args.data_root,
                 report_path=args.report, final=args.final)
    print(json.dumps(dict(report=str(Path(args.report).resolve()), passed=report['passed'],
                         checks=len(report['checks'])), ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
