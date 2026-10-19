"""Read current original advertisement corners from two local RGB interfaces.

Old UV locate finite searches only. Each corner is the intersection of two
locally measured print edges in the same original exposure; no camera is read.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
from PIL import Image
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
ORIGINAL = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl')
SOURCE = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def main():
    original = pickle.loads(ORIGINAL.read_bytes())['programme_absolute_rows']
    results = []
    for ordinal in (10401, 11491):
        item = original[ordinal]
        row = item['source_row']
        pts = int(item['group_actual_pts'])
        path = BASE / 'original_ice_FIT_RGB_review' / f'{pts}_independently_decoded_RAW_NATIVE.png'
        rgb = np.asarray(Image.open(path).convert('RGB')).copy()
        binding = dict(source_sha256=SOURCE, source_pts=pts, time_base='1/15360',
            shot_id=item['block'], native_hw=[1080, 1920],
            decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        corners = np.asarray(row['uv'], float)
        records = []
        for i, corner in enumerate(corners):
            measured_edges = []
            for neighbor in ((i - 1) % 4, (i + 1) % 4):
                tangent = corners[neighbor] - corner
                tangent /= np.linalg.norm(tangent)
                # Three separate native transects inside the corner. Orient
                # outward so the physical print-to-white transition rises.
                normal = np.array([-tangent[1], tangent[0]])
                if normal @ (np.mean(corners, axis=0) - corner) > 0:
                    normal *= -1
                points = corner + np.array([16., 24., 32.])[:, None] * tangent
                half = 35. if abs(normal[0]) > .8 and ordinal == 11491 else 12.
                profile = read_source_material_transition_profiles(rgb, binding=binding,
                    search_points_uv=points, normals=np.tile(normal, (3, 1)),
                    search_intervals_px=np.tile([-half, half], (3, 1)),
                    transition='dark_to_light', tangent_half_width_px=1,
                    keep_profiles=True)
                valid = [r for r in profile['readings'] if not any(r['source_flags'].values())]
                line = None
                if len(valid) == 3:
                    uv = np.asarray([r['actual_source_uv'] for r in valid], float)
                    center = uv.mean(axis=0)
                    _, _, vectors = np.linalg.svd(uv - center)
                    n = vectors[-1]
                    line = dict(normal=n.tolist(), constant=float(n @ center),
                        measured_uv=uv.tolist(), max_deviation_px=float(np.max(abs((uv-center) @ n))))
                measured_edges.append(dict(neighbor_corner=neighbor, profile=profile, line=line))
            corrected = None
            if all(e['line'] is not None for e in measured_edges):
                a = np.asarray([e['line']['normal'] for e in measured_edges])
                b = np.asarray([e['line']['constant'] for e in measured_edges])
                if abs(np.linalg.det(a)) > .2:
                    corrected = np.linalg.solve(a, b).tolist()
            records.append(dict(original_point_index=i, original_uv=corner.tolist(),
                candidate_uv=corrected, source_measured_edges=measured_edges,
                method='intersection_of_two_same_exposure_locally_measured_print_interfaces',
                model_used_for_navigation_or_peak_selection=False))
        results.append(dict(original_packet_row_index=ordinal, id=row['id'], source_binding=binding,
            source_rgb=dict(path=str(path), sha256=sha(path), pts=pts), records=records))
    output = BASE / 'same_manual_B2_contact_RGB_review_20261012' / 'original_ad_corners_two_interface_RGB_candidates_20261012.json'
    output.write_text(json.dumps(plain(dict(schema='rinkspace.original-ad-corner-source-interface-review.v1',
        original_absolute_packet=dict(path=str(ORIGINAL), sha256=sha(ORIGINAL)),
        rows=results, installed_in_FIT=False, source_visual_acceptance_pending=True)), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(row=r['original_packet_row_index'], points=[dict(index=x['original_point_index'],
        old=x['original_uv'], measured=x['candidate_uv'], flags=[sorted({k for y in e['profile']['readings'] for k, bad in y['source_flags'].items() if bad}) for e in x['source_measured_edges']]) for x in r['records']]) for r in results]))


if __name__ == '__main__':
    main()
