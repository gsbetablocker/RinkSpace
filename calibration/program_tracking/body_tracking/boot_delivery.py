"""Input identity and compact programme delivery for the footwear supplement."""
from pathlib import Path
import json
import numpy as np
from .rebuild_prepare import load, digest


def check_inputs(reference, binding, camera_info, output):
    """Refuse reuse after changes to shape, observations or the parent motion."""
    output = Path(output)
    semantics = json.loads((output/'observation_semantics.json').read_text(encoding='utf8'))
    paths = [reference.baseline/'full/result.npz', reference.baseline/'ledger.json',
             reference.baseline/'observations.npz']
    paths += [reference.baseline/'full'/f'w{i}'/'result.npz' for i in range(1, 6)]
    equipment = semantics['sources']['equipment']
    contract = dict(schema='rinkspace.footwear-inputs.v1',
        reference_hashes=reference.hashes,
        parent_motion={str(p): digest(p) for p in paths},
        camera=camera_info,
        body_model_hash=binding.record()['body_model_hash'],
        body_shape_hash=binding.record()['body_shape_hash'],
        equipment_sources=equipment,
        geometry=binding.reference.geometry.tolist(),
        engineering_priors=dict(lining=binding.lining, heel_bed=binding.heel_bed,
            ball_bed=binding.ball_bed, shank_radius=binding.shank_radius,
            cuff_clearance=binding.cuff_clearance,
            maximum_modes=binding.maximum_modes.tolist(),
            mode_regularizer=binding.mode_regularizer.tolist(), ankle_sigma_px=24.),
        windows=[[35.2, 36.8], [40., 41.5]],
        state_definition='shared 14-parameter binding; right-composed full-body refinements; contact-driven upper v1')
    path = output/'inputs.json'
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf8'))
        if saved != contract:
            changed = [k for k in set(saved) | set(contract) if saved.get(k) != contract.get(k)]
            raise ValueError('Footwear input changed; saved results retained. Use a separate output after reviewing: '+', '.join(sorted(changed)))
    else:
        path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf8')
    return contract


def deliver(project, output):
    project, output = Path(project), Path(output)
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    parameter_path = programme/'boot_integration_1102.npz'
    binding_path = programme/'boot_integration_1102.json'
    for path in (parameter_path, binding_path):
        if len(str(path.resolve()).encode('utf-16-le'))//2 >= 260 or len(path.name.encode('utf-16-le'))//2 > 255:
            raise ValueError('OneDrive path exceeds project limits')
    read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
    result_path = output/'fitted/result.npz'
    result_hash = digest(result_path)
    review = read(output/'boot_report.json')
    if review['sources']['fitted']['sha256'] != result_hash:
        raise ValueError('Footwear report refers to a different fitted result; regenerate the report first')
    binding = read(output/'binding.json')
    joint = load(output/'joint_result.npz')
    arrays = dict(shared_binding=joint['x'][:14], binding_json=np.array(json.dumps(binding)),
                  time_base=np.array([1, 12800]),
                  parent_parameters_sha256=np.array(digest(programme/'body_fullbody_1102.npz')))
    for i in range(1, 3):
        for key, value in load(output/f'w{i}'/'result.npz').items():
            arrays[f'w{i}_{key}'] = value
    np.savez_compressed(parameter_path, **arrays)
    reloaded = load(parameter_path)
    for key, value in arrays.items():
        np.testing.assert_array_equal(reloaded[key], value)
    evidence = {}
    for name in ('inputs.json', 'registration.json', 'joint_result.json', 'migration_check.json',
                 'derivative_checks.json', 'precision_check.json', 'invariants.json', 'invariants_final.json',
                 'observation_semantics.json', 'boot_report.json', 'results.txt',
                 'player_check/checks.json', 'player_check/cache_contract.json',
                 'section_check/checks.json', 'section_check/normal_audit.json',
                 'clean_player_check/checks.json', 'clean_section_check/checks.json',
                 'source_shoes.json', 'source_shoes.png', 'visual_review.json', 'verification.json'):
        path = output/name
        if path.is_file():
            if name.endswith('/checks.json'):
                check = read(path)
                if not check.get('passed') or check.get('source_sha256',{}).get(str(result_path.resolve())) != result_hash:
                    raise ValueError('Viewer check does not match the delivered footwear result: '+name)
            evidence[name] = dict(path=str(path), sha256=digest(path))
    source_files = ['boot_integration.py', 'footwear_binding.py', 'boot_delivery.py',
                    'boot_review.py', 'footwear_observations.py', 'footwear_pixel_factors.py',
                    'fullbody_state.py', 'fullbody_fit.py']
    source_hashes = {str(Path(__file__).parent/name): digest(Path(__file__).parent/name)
                     for name in source_files if (Path(__file__).parent/name).is_file()}
    record = dict(schema='rinkspace.programme-footwear-supplement.v1',
        event='ankara_2026', programme='okada_mei_sp', shot='s01',
        available_source_seconds=[22.12, 44.96], source_time_base=[1, 12800],
        shared_binding_frames=1143, body_refinement_frames=157,
        body_refinement_seconds=[[35.2, 36.8], [40., 41.5]],
        parent_motion=dict(path=str(programme/'body_fullbody_1102.json'),
                           sha256=digest(programme/'body_fullbody_1102.json')),
        compact_parameters=dict(path=str(parameter_path), sha256=digest(parameter_path)),
        result=dict(path=str(result_path), sha256=digest(result_path)),
        footwear_binding=binding, input_contract=read(output/'inputs.json'),
        solver=read(output/'joint_result.json'),
        observation_ledger=dict(path=str(output/'ledger.json'), sha256=digest(output/'ledger.json')),
        candidates=[dict(id='boot_legacy', meaning='original fitted motion and closed convex shell'),
                    dict(id='boot_open', meaning='identical old vertices and motion with an open collar'),
                    dict(id='boot_fitted', meaning='shared foot-boot registration, cavity and contact-driven upper with two refined body windows')],
        viewer_config=str(programme/'viewer3d_boot_1102.json'),
        viewer_entry=str(project/'START_BOOT_INSPECTION.cmd'),
        replay_entry=str(project/'scripts/run_boot_integration.py'),
        replay_source_hashes=source_hashes, evidence=evidence,
        remaining_scope=['Existing complete articulated body state is S01 only.',
                        'Toe-pick geometry and an identified toe-pick contact observation are absent.',
                        'The cavity, footbed and upper regularizers are geometric fitting priors.'])
    binding_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    return dict(binding=str(binding_path), parameters=str(parameter_path), result=str(result_path))
