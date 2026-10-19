"""Replay six original RGB scans without writing source images or fitting."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import truth_source_location_revision as common
from calibration.solver_runtime import truth_source_scanline_revision as binding

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
H = DATA / 'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
PARENT = H / 'source_pixel_review_additional/F2_Ad24_26_sparse_temporal_actual_RGB_candidate/FINAL_native_ice108_original101_plus7_actual_F2_filled_outer65_FIT_QUALIFIED_20261011.json'
PARENT_SHA = 'cca5550513c541ececbabb4021f4951847fd6a3ef4836c8dc0e7bafd0cc05610'
PRODUCER = DATA / 'calibration/runs/primary823_cut6_user_priority_review_20261005/read_cut6_native_ice_source_only.py'
PRODUCER_SHA = '0d76f0f102de532d57e7e82eb4a5cbca6f894a547e462eb897613ce7a7e62aa0'
EXTRACTION = PRODUCER.with_name('cut6_source_only_gray_paint_profiles_PENDING_VIEW.json')
EXTRACTION_SHA = '87a2e68e704937e5af592db8083f3c26acd6b116cc9a086ef6d9600d31659650'
ROW_INDICES = (25, 28)
PACKET = ROOT / 'events/ankara_2026/observations/source_packets/ice108_saved_F3_scanline_locations_20261011/packet.json'
RECEIPT = H / 'source_pixel_review_additional/ice_scanline_source_semantics/ice108_saved_F3_scanline_locations_revision_20261011.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def estimator():
    """Execute only the frozen pure function definition, never script main."""
    require(common._sha(PRODUCER) == PRODUCER_SHA, 'Frozen original producer changed')
    text = PRODUCER.read_text(encoding='utf8')
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'edge']
    require(len(functions) == 1, 'One original pure edge function required')
    namespace = dict(np=np, gaussian_filter1d=gaussian_filter1d)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(PRODUCER), 'exec'), namespace)
    return namespace['edge'], hashlib.sha256(ast.get_source_segment(text, functions[0]).encode('utf8')).hexdigest()


def qualify():
    require(common._sha(PARENT) == PARENT_SHA, 'Frozen ICE108 source packet changed')
    require(common._sha(EXTRACTION) == EXTRACTION_SHA, 'Frozen original RGB profile receipt changed')
    parent = common._read(PARENT)
    extraction = common._read(EXTRACTION)
    edge, ast_hash = estimator()
    frames = {f['pts']: f for f in extraction['frames']}
    records = []
    cv2.setNumThreads(1)
    for index in ROW_INDICES:
        row = parent['rows'][index]
        require(row['id'] == f"cut6_source_{row['pts']}_F3_outer_{row['split']}", 'Original F3 row identity changed')
        require(row['curve'] == dict(kind='house_ring', side='F', row=3, ring='outer'), 'Original physical curve changed')
        require(not any(k in row for k in binding.MODE_FIELDS), 'Existing operator must not be reclassified')
        path = Path(row['source_raw_frame'])
        require(common._sha(path) == row['source_file_sha256'], 'Original PNG hash changed')
        raw = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        require(raw is not None and raw.shape == (1080, 1920, 3), 'Native RGB shape changed')
        require(hashlib.sha256(raw.tobytes()).hexdigest() == row['source_native_bgr_pixels_sha256'], 'Native RGB pixel bytes changed')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(float)
        samples = frames[row['pts']]['source_samples']['F3_outer']
        require(common._same(samples, row['source_search_support']), 'Original saved profiles differ from source row')
        strength = []
        for uv, sigma, sample in zip(row['uv'], row['sigma_px'], samples):
            y, lo, hi = sample['manual_source_bracket']
            replay = edge(gray, y, lo, hi)
            require(common._same(replay, sample), 'Original pure RGB estimator did not replay bit exactly')
            require(replay['uv'] == uv and replay['sigma_px'] == sigma, 'Original UV/sigma did not replay bit exactly')
            require(replay['source_scan_axis'] == 'x' and uv[1] == y, 'Only fixed-y x-location measured')
            strength.append(replay['source_signal_strength_gray'])
        records.append(dict(original_row_index=index, id=row['id'], split=row['split'],
            source_row_reference=common.reference(PARENT), source_row_canonical_json_sha256=common._row_hash(row),
            measured_source_row=deepcopy(row), original_estimator='positive_gray_gradient_mass_q50_fixed_y',
            fixed_axis='y', physical_branch='max_x', replayed_original_UV_sigma_profiles_bitexact=True,
            source_signal_strength_gray=strength,
            actual_RGB_review=dict(reviewer='solver_continuation', date='2026-10-11',
                basis='existing_native_RGB_opened_without_model',
                finding='The three fixed-y source windows cross the visible F3 outer gray-to-white transition on its right branch. The other house intersection lies outside the image; the reviewed windows have no skater or separate paint-edge occlusion. Broad motion blur is preserved in the original horizontal engineering sigma.'),
            latent_edge_accuracy_certified=False, original_working_sigma_not_new_covariance=True))
    return dict(schema=binding.RECEIPT_SCHEMA, original_packet_reference=common.reference(PARENT),
        original_producer_reference=common.reference(PRODUCER), pure_estimator_AST_sha256=ast_hash,
        original_extraction_reference=common.reference(EXTRACTION),
        qualification_script_reference=common.reference(__file__), rows=records,
        source_only_RGB_replayed=True, model_or_camera_used_for_measurement=False,
        source_images_written=0, source_pixels_decoded_only_in_RAM=True,
        fit_and_holdout_roles_preserved=True, scanline_sigma_is_original_measured_axis_working_weight=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true', help='Write only the small receipt and source packet')
    args = parser.parse_args()
    receipt = qualify()
    if args.write:
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        PACKET.parent.mkdir(parents=True, exist_ok=True)
        receipt_ref = common._write(RECEIPT, receipt)
        packet = common._read(PARENT)
        for index in ROW_INDICES:
            packet['rows'][index].update(curve_observation_kind='axis_scanline_branch',
                scanline_fixed_axis='y', scanline_branch='max_x', scanline_location_measurement=receipt_ref)
        packet[binding.REVISION_KEY] = dict(parent_packet_reference=common.reference(PARENT),
            receipt_reference=receipt_ref, changed_row_indices=list(ROW_INDICES),
            changed_row_ids=[packet['rows'][i]['id'] for i in ROW_INDICES],
            all_original_source_rows_except_four_scanline_fields_unchanged=True)
        packet_ref = common._write(PACKET, packet)
        binding.validate_scanline_revision(PARENT, PACKET, ROW_INDICES)
        print(json.dumps(dict(packet_reference=packet_ref, receipt_reference=receipt_ref,
            revised_rows=list(ROW_INDICES), replayed_original_samples=sum(len(r['measured_source_row']['uv']) for r in receipt['rows']),
            source_images_written=0)))
    else:
        print(json.dumps(dict(revised_rows=list(ROW_INDICES), source_only_bitexact_replay=True,
            source_images_written=0, samples=sum(len(r['measured_source_row']['uv']) for r in receipt['rows']))))


if __name__ == '__main__':
    main()
