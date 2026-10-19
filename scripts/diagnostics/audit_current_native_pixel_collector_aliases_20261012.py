"""Replay only admitted native pixel data from one saved startup, without its graph."""
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.solver_runtime import install
from calibration.solver_runtime import truth_source_location_revision as common
from calibration.program_tracking.current_source_point_collection import collect_current_active_source_points


def main():
    run, output = map(Path, sys.argv[1:]); manifest_path = run / 'input_manifest.json'
    manifest = common._read(manifest_path)
    camera = run / 'complete_physical_camera_model_UNACCEPTED_5126c97aa85b9576.npz'
    with np.load(camera, allow_pickle=False) as z:
        aliases = dict(zip(map(int, z['pts']), map(int, z['exposure_representative_pts'])))
    state = NS(original=NS(aliases=aliases), bypts=dict.fromkeys(set(aliases.values())),
        blocks=[], patterns=None, external=None, cross_fit=[])
    install()
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from calibration.solver_runtime.source_point_qualification import load_source_point_qualification
    from calibration.solver_runtime.truth_source_scanline_revision import original_qualified_packet
    source = NS(); source_refs = {}
    families = [('native_ice_factor', 'additional_real_native_ice_curves'),
        ('native_board_factor', 'additional_real_native_board_edges'),
        ('native_board_material_point_factor', 'additional_real_native_board_material_points'),
        ('circle_factor', 'additional_real_native_board_circles'),
        ('native_board_track_factor', 'additional_native_static_board_tracks'),
        ('persistent_ice_factor', 'additional_real_native_persistent_ice_long_baseline')]
    for name, key in families:
        contract = manifest[key]; reference = dict(path=contract['packet_path'], sha256=contract['packet_sha256'])
        path = common._checked(reference); packet = common._read(path); source_refs[name] = reference
        if name == 'persistent_ice_factor':
            rows, held = [], []
            for original in packet['new_rows']:
                row = deepcopy(original)
                for prefix in ('anchor', 'target'):
                    endpoint = row[prefix]; native = int(endpoint['source_native_pts'])
                    if 'source_actual_pts' in endpoint and int(endpoint['source_actual_pts']) != aliases[native]:
                        raise ValueError('Current persistent source clock differs')
                    row[prefix+'_endpoint'] = dict(actual_pts=aliases[native])
                    row[prefix+'_uv'] = np.asarray(endpoint['uv'], float)
                if not (np.array_equal(row['anchor_uv'], row['xy0']) and np.array_equal(row['target_uv'], row['xy1'])):
                    raise ValueError('Persistent packet UV and admitted endpoints differ')
                (rows if row['split'] == 'fit' else held).append(row)
            factor = NS(rows=rows, held=held, contract=contract)
        elif name == 'native_board_track_factor':
            rows, held, anchors = [], [], []
            for original in packet['rows']:
                row = deepcopy(original)
                for endpoint in ('source', 'target'):row[endpoint+'_actual_pts'] = aliases[int(row[endpoint+'_pts'])]
                (rows if row['split'] == 'fit' else held).append(row)
            for original in packet.get('source_anchor_observations', []):
                row = deepcopy(original); row['_anchor'] = True
                if aliases[int(row['native_pts'])] != row['actual_pts']:raise ValueError('Current track anchor clock differs')
                (anchors if row['split'] == 'fit' else held).append(row)
            lookup = {r['id']:r for r in rows+anchors}
            evaluation = [lookup[k] for k in packet.get('fit_row_order', list(lookup))]
            factor = NS(rows=rows, held=held, contract=contract, anchor_rows=anchors, fit_evaluation_rows=evaluation)
        else:
            rows, held = [], []
            for original in packet['rows']:
                row = deepcopy(original); native = int(row['pts']); actual = aliases[native]
                if 'actual_pts' in row and row['actual_pts'] != actual:raise ValueError('Current native source clock differs')
                row['actual_pts'] = actual; row['uv'] = np.asarray(row['uv'], float)
                row['sigma'] = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(row['uv']),)).copy()
                (rows if row['split'] == 'fit' else held).append(row)
            if name == 'native_ice_factor':
                factor = NativeIceCurveFactor.__new__(NativeIceCurveFactor)
                factor.rows, factor.held, factor.contract = rows, held, contract
                original = original_qualified_packet(path, packet)
                factor.source_point_qualification = load_source_point_qualification(path, original,
                    source_family='additional_native_ice_curves')
                factor._active_fit_view_cache = None
            else:factor = NS(rows=rows, held=held, contract=contract)
        setattr(source, name, factor)
    authority = common._read(common._checked(manifest['highest_direct_human_hard_truth']['native']['authority']))
    ids = {r['id'] for r in authority['records']}; source.factors = []
    for name in ('native_ice_factor', 'native_board_factor'):
        factor = copy(getattr(source, name)); factor.rows = [r for r in factor.rows if r['id'] in ids]
        if not factor.rows:continue
        factor.contract = dict(factor.contract, direct_manual_centre_secondary_objective=True)
        source.factors.append(factor)
    class PlacementAlias:
        def __init__(self, base):self.base = base; self.material_factor = base.native_board_material_point_factor
        def __getattr__(self, name):return getattr(self.base, name)
    wide = NS(contract=dict(source_pts=56218880), uvs=[], ad_packet=dict(rows=[]), trace_uv=[])
    groups, receipts, _ = collect_current_active_source_points(state, PlacementAlias(source), wide)
    expected = sum(r['pixel_input_count'] for r in receipts)
    if sum(len(uv) for _, uv in groups) != expected:raise ValueError('Collected native receipt counts differ')
    alias = next(r for r in receipts if r['family'] == 'material_factor')
    if alias.get('alias_of') != 'native_board_material_point_factor' or alias['pixel_input_count']:
        raise ValueError('Actual placement alias was not recognized without recollecting')
    report = dict(schema='rinkspace.current-native-pixel-source-collector-alias-replay.v1',
        manifest_reference=common.reference(manifest_path), camera_reference=common.reference(camera),
        source_references=source_refs, all_six_current_native_pixel_families_collected=True,
        actual_native_ice_qualified_public_view_used=True, object_identity_alias_deduplicated=True,
        direct_manual_source_subsets_collected=True, family_receipts=receipts,
        programme_motion_absolute_patterns_external_wide_not_replayed_here=True,
        whole_programme_collection_claimed=False, graph_residual_J_or_physical_guard_evaluated=False)
    print(json.dumps(common._write(output, report), ensure_ascii=False))


if __name__ == '__main__':main()
