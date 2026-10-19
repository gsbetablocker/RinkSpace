"""Repair a source search switching from the upper rail to Ad7's blue edge."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from .extract_first_shot_board_source import DATA, read, write, profile_read, segment_records, native_overlay, source_segments


def repair():
    manifest = {int(r['index']): r for r in read(DATA / 'frame_manifest.json')['frames']}
    repaired = []
    for index in range(201, 1249):
        path = DATA / f'board_candidates/{index:06d}.json'
        packet = read(path)
        bad_features = [f for f in packet['features'] if f['target_id'].startswith('board.top.')
                        and any(s.get('source_curve_fit_max_residual_px', 0) > 6 for s in f['segments'])]
        if not bad_features:
            continue
        raw = cv2.imread(manifest[index]['image_path'])
        mask = cv2.dilate((cv2.imread(manifest[index]['mask_path'], 0) > 0).astype('uint8'), np.ones((5, 5), np.uint8)) > 0
        mask[:120, :570] = True
        mask[945:, 1660:] = True
        changes = []
        for feature in bad_features:
            evidence = np.vstack([s.get('source_native_readout_uv', s['uv']) for s in feature['segments']])
            center = np.mean(evidence[:, 0])
            scale = max(np.ptp(evidence[:, 0]), 1.)
            t = (evidence[:, 0] - center) / scale
            design = np.column_stack((np.ones(len(t)), t, t*t))
            initial = np.linalg.lstsq(design, evidence[:, 1], rcond=None)[0]
            result = least_squares(lambda c: design @ c - evidence[:, 1], initial, loss='soft_l1', f_scale=1.)
            accepted = np.abs(design @ result.x - evidence[:, 1]) < 4.
            if accepted.sum() < 20:
                raise ValueError(f'{index}: insufficient definite same-source rail to reject competing edge')
            coefficients = np.linalg.lstsq(design[accepted], evidence[accepted, 1], rcond=None)[0]
            x = np.linspace(evidence[:, 0].min(), evidence[:, 0].max(), max(2, int(np.ceil(np.ptp(evidence[:, 0])/12))+1))
            t = (x - center) / scale
            navigation = np.column_stack((x, np.column_stack((np.ones(len(t)), t, t*t)) @ coefficients))
            # The robust source curve only navigates the new current-image
            # profiles. It is never emitted as a pixel measurement itself.
            readouts, _ = profile_read(raw, mask, navigation, [1, -.5, -.5], 10,
                                      maximum_blur=18., minimum_amplitude=10., navigation_tolerance=5.,
                                      gradient_direction=[0, 1])
            segments = segment_records(readouts, spacing=200, smooth_curve=True, target_id=feature['target_id'])
            if not segments or any(s.get('source_curve_fit_max_residual_px', 0) > 4 for s in segments):
                raise ValueError(f'{index}: reread upper rail is still incoherent')
            changes.append(dict(target_id=feature['target_id'],
                                rejected_competing_edge_native_uv=evidence[~accepted].tolist(),
                                previous_segments=feature['segments'],
                                reason='blue_Ad7_top_is_not_blue_board_upper_rail',
                                current_source_reread_count=sum(len(s.get('source_native_readout_uv', s['uv'])) for s in segments)))
            feature['segments'] = segments
        packet.setdefault('source_upper_rail_switch_repair', []).extend(changes)
        packet['point_count'] = sum(len(s['uv']) for f in packet['features'] for s in f['segments'])
        if packet['point_count'] > 300:
            from .single_frame_source_review import compact_review_points
            glyphs = [f for f in packet['features'] if '.print.' in f['id']]
            board_count = sum(len(s['uv']) for f in packet['features'] if '.print.' not in f['id'] for s in f['segments'])
            compact_review_points(glyphs, 300-board_count)
            packet['point_count'] = sum(len(s['uv']) for f in packet['features'] for s in f['segments'])
        packet['diagnostics']['board_point_count'] = sum(len(s['uv']) for f in packet['features'] if '.print.' not in f['id'] for s in f['segments'])
        packet['diagnostics']['printed_point_count'] = packet['point_count'] - packet['diagnostics']['board_point_count']
        write(path, packet)
        cv2.imwrite(str(DATA / f'board_overlays/{index:06d}.png'), native_overlay(raw, list(source_segments(packet))), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        repaired.append(dict(index=index, point_count=packet['point_count'], changes=changes))
    # Include already repaired packets when rerunning after an interrupted
    # extraction; their original readouts remain in each packet's provenance.
    repaired = [dict(index=index, point_count=packet['point_count'], changes=packet['source_upper_rail_switch_repair'])
                for index in range(201, 1249)
                if (packet := read(DATA / f'board_candidates/{index:06d}.json')).get('source_upper_rail_switch_repair')]
    write(DATA / 'board_upper_rail_switch_repair.json', dict(frames=repaired, source_only=True, contact_changed=False, print_pixel_coordinates_changed=False))
    print('Upper rail reread frames:', [(r['index'], r['point_count']) for r in repaired])


if __name__ == '__main__':
    repair()
