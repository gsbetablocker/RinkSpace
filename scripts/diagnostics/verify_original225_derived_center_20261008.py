"""Light original225 proof of the confirmed course-derived hockey centre.

Source pickle inputs are immutable, hash-certified numerical dictionaries.
Only NumPy's numeric buffer and dtype constructors are allowed; no general
external pickle callable or object is accepted. No graph or solver is built.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, io, json, os, pickle, sys, time
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
sys.dont_write_bytecode = True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate import original_review_source as review
from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView, install_source_ice_material_view
from z190_physical_camera import PhysicalCameraModel


class NumericDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('numpy', 'dtype'):
            return np.dtype
        if (module, name) in (('numpy._core.numeric', '_frombuffer'), ('numpy.core.numeric', '_frombuffer')):
            return np._core.numeric._frombuffer
        raise pickle.UnpicklingError('Non-numerical external callable forbidden: '+module+'.'+name)


def numeric_data_loads(raw):
    return NumericDataUnpickler(io.BytesIO(raw)).load()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def numeric_geometry(row):
    return {name: np.asarray(row[name], dtype=float) for name in
            ('world', 'world_xyz', 'world_xyz_scene_units') if name in row}


def feature_geometry(feature):
    result = {name: np.asarray(value, dtype=float) for name, value in feature.items()
              if name.endswith('_xyz') or name == 'radius'}
    assert result, 'No actual numeric geometry tested for '+feature['id']
    return result


def maximum_difference(a, b):
    assert a.keys() == b.keys()
    return max((float(abs(a[k]-b[k]).max()) if a[k].size else 0. for k in a), default=0.)


def main():
    parser = argparse.ArgumentParser()
    for name in ('state', 'old-state', 'auxiliary', 'camera', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(); start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    # A local module-only facade leaves the shared standard pickle module intact.
    review.pickle = SimpleNamespace(loads=numeric_data_loads)
    rows = review.load_original_review_rows()
    calls, authorities = review.load_original_absolute_evaluators()
    signatures = [review.observation_signature(item['source_row']) for item in rows]
    provider, world, proof = current_complete_source_geometry(args.state, args.auxiliary)
    old_provider, old_world, old_proof = current_complete_source_geometry(args.old_state, args.auxiliary)
    install_source_ice_material_view(provider); install_source_ice_material_view(old_provider)
    assert provider.current.derive_hockey_center_from_course
    assert not getattr(old_provider.current, 'derive_hockey_center_from_course', False)
    assert provider.current.strict_half_circle_crease and old_provider.current.strict_half_circle_crease
    with np.load(args.state, allow_pickle=False) as z: full_q = z['q'].copy()
    with np.load(args.old_state, allow_pickle=False) as z: old_q = z['q'].copy()
    assert np.array_equal(full_q, old_q) and np.array_equal(world, old_world)
    for item in rows:
        for owner in (provider, old_provider):
            owner.register_row(item['source_row'], actual_pts=int(item['group_actual_pts']),
                               allow_previously_qualified_held_ad2_representation=True)
    model, pts, aliases, views = PhysicalCameraModel.load_npz(args.camera)
    cameras = {}; records = []; world_max = 0.; residual_max = 0.; changed_rows = 0
    original_fields = [numeric_geometry(provider.rebind(item['source_row'], world)) for item in rows]
    perturbations = []
    for logical in (0, 1):
        for sign in (-1., 1.):
            changed = world.copy(); changed[logical] += sign*1e-5
            maximum = 0.
            for item, nominal in zip(rows, original_fields):
                current = numeric_geometry(provider.rebind(item['source_row'], changed))
                delta = maximum_difference(nominal, current); maximum = max(maximum, delta)
                assert all(np.array_equal(nominal[k], current[k]) for k in nominal)
            perturbations.append(dict(logical_world_index=logical, raw_initializer_delta=sign*1e-5,
                                      original225_world_max_difference=maximum, bitexact=True))
    print('original225 source WORLD bitexact under all four raw centre perturbations', flush=True)
    for item, signature, current_values in zip(rows, signatures, original_fields):
        row = item['source_row']; actual = int(item['group_actual_pts']); representative = aliases[actual]
        if representative not in cameras: cameras[representative] = model.camera25(views[representative])
        camera = cameras[representative]
        before = old_provider.rebind(row, old_world); after = provider.rebind(row, world)
        assert review.observation_signature(before) == review.observation_signature(after) == signature
        dw = maximum_difference(numeric_geometry(before), current_values); world_max = max(world_max, dw)
        rb = np.asarray(calls[item['block']](camera, [before], False), dtype=float)
        ra = np.asarray(calls[item['block']](camera, [after], False), dtype=float)
        assert rb.shape == ra.shape == (2*len(row['uv']),) and np.isfinite(rb).all() and np.isfinite(ra).all()
        delta = float(abs(ra-rb).max()); residual_max = max(residual_max, delta); changed_rows += bool(delta)
        pixel_delta = float(abs((ra-rb)*np.repeat(np.asarray(row['sigma']), 2)).max())
        records.append(dict(block=item['block'], group_order=int(item['group_order']), row_order=int(item['row_order']),
                            id=row['id'], actual_pts=actual, source_row_pts=int(row['pts']),
                            split_present='split' in row, split=row.get('split'), collection_role=item['collection_role'],
                            UV_count=len(row['uv']), observation_signature=signature,
                            strict_before_to_derived_world_max_delta=dw, normalized_r_max_delta=delta,
                            original_operator_component_px_max_delta=pixel_delta,
                            before_normalized_r_rms=float(np.sqrt(np.mean(rb*rb))),
                            after_normalized_r_rms=float(np.sqrt(np.mean(ra*ra))),
                            same_original_operator_and_sigma=True))
    view = CurrentIceMaterialView(provider.current); features = view.features(world)
    scene = {f['id']: f for f in provider.scene(world)['ice_features']}
    feature_checks = []; all_feature_perturbations = []
    for logical in (0, 1):
        for sign in (-1., 1.):
            changed = world.copy(); changed[logical] += sign*1e-5
            shifted_view = view.features(changed)
            shifted_scene = {f['id']: f for f in provider.scene(changed)['ice_features']}
            maximum = 0.
            for name, feature in features.items():
                nominal = feature_geometry(feature)
                assert maximum_difference(nominal, feature_geometry(scene[name])) == 0.
                for alternate in (shifted_view[name], shifted_scene[name]):
                    values = feature_geometry(alternate)
                    assert all(np.array_equal(nominal[k], values[k]) for k in nominal)
                    maximum = max(maximum, maximum_difference(nominal, values))
            all_feature_perturbations.append(dict(logical_world_index=logical, raw_initializer_delta=sign*1e-5,
                                                  scene_and_material_view_max_difference=maximum, bitexact=True))
    for name in features:
        if name in ('C0', 'C', 'A1', 'A2', 'A3', 'A4') or 'goal' in name or 'crease' in name:
            feature_checks.append(dict(id=name, producer_render_equal=True,
                                       raw_center_slots_do_not_move_geometry=True))
    assert all(review.observation_signature(item['source_row']) == signature
               for item, signature in zip(rows, signatures))
    effective = provider.effective_world(world)
    chain = provider.current.effective_world_chain(world)
    assert np.array_equal(chain[:, :2], np.zeros_like(chain[:, :2]))
    report = dict(schema='rinkspace.original225-confirmed-course-derived-hockey-center-proof.v1',
                  state_path=str(args.state), state_npz_sha256=sha(args.state), old_strict_state_path=str(args.old_state),
                  old_strict_state_npz_sha256=sha(args.old_state), complete_q_sha256=hashlib.sha256(full_q.tobytes()).hexdigest(),
                  canonical_q_all34338_bitexact_before_after=True, source_packet_sha256=review.PACKET_SHA256,
                  evaluator_fixture_sha256=review.EVALUATOR_FIXTURE_SHA256, external_pickle_numeric_only_allowlist=True,
                  original_source_rows=225, source_UV_context_count=3495, all_original_UV_sigma_roles_and_splits_preserved=True,
                  camera_path=str(args.camera), camera_sha256=sha(args.camera), actual_original_evaluators_by_block=authorities,
                  raw_initializer_center=world[:2].tolist(), effective_current_course_center=effective[:2].tolist(),
                  raw_center_course_relative_offset_conditional_m=float(np.linalg.norm(world[:2]-effective[:2])),
                  effective_world_chain_raw_center_columns_zero=True, original225_center_slot_perturbations=perturbations,
                  scene_material_feature_perturbations=all_feature_perturbations, named_hockey_feature_checks=feature_checks,
                  current_feature_count=len(features), strict_to_derived_changed_source_rows=changed_rows,
                  strict_to_derived_world_max_delta=world_max, strict_to_derived_normalized_r_max_delta=residual_max,
                  necessary_structural_source_change_not_bitexact_residual_claim=True, rows=records,
                  no_sources_promoted_or_added=True, full_source_graph_loaded=False, full_solver_started=False,
                  accuracy_accepted=False, elapsed_seconds=time.perf_counter()-start)
    dest=args.output/'original225_derived_center_actual_source_check_v1.json'
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps(dict(report=str(dest), sha256=sha(dest), changed_rows=changed_rows,
                          max_normalized_r_delta=residual_max, source_world_perturbations_bitexact=True,
                          feature_perturbations_bitexact=True, feature_count=len(features), elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__': main()
