"""Native-PTS, source-only navigation between selected shot-two exposures.

Intermediate exposures supply mask-isolated LK correspondence only. They never
produce field observations, camera fits, feature bindings or acceptance claims.
Ice and board support are processed separately, preserving failed links.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.ndimage import median_filter
from calibration.program_tracking.prepare_first_shot_review import decoded
from rinkspace.local_registration import _features, _fit, RegistrationConfig


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


class SourceDomains:
    """Conservative navigation domains, never geometric field observations."""
    def __init__(self, reference):
        manifest = read(reference / 'frame_manifest.json')
        self.provenance = dict(reference_root=str(reference),
            manifest_sha256=hashlib.sha256((reference / 'frame_manifest.json').read_bytes()).hexdigest(), source_boundary_and_exclusion_inputs=[])
        self.rows = sorted(manifest['frames'], key=lambda r: r['native_pts'])
        self.pts = np.asarray([r['native_pts'] for r in self.rows])
        self.boundaries = []
        self.boxes = []
        for row in self.rows:
            packet_path = reference / f"board_candidates/{row['index']:06d}.json"
            packet = read(packet_path)
            self.provenance['source_boundary_and_exclusion_inputs'].append(dict(index=row['index'], native_pts=row['native_pts'],
                boundary_packet_path=str(packet_path), boundary_packet_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                person_mask_path=row['mask_path'], person_mask_sha256=hashlib.sha256(Path(row['mask_path']).read_bytes()).hexdigest()))
            edges = {}
            for role in ('top', 'contact'):
                points = [uv for f in packet['features'] if f['target_id'].startswith('board.' + role + '.')
                          for s in f['segments'] for uv in s.get('source_native_readout_uv', s['uv'])]
                uv = np.asarray(points, float)
                if len(uv) < 2:
                    raise ValueError('Missing source boundary navigation: ' + str(row['index']))
                order = np.argsort(uv[:, 0]); uv = uv[order]
                edges[role] = np.interp(np.arange(1920), uv[:, 0], uv[:, 1])
            self.boundaries.append(edges)
            person = cv2.imread(row['mask_path'], 0)
            ys, xs = np.nonzero(person)
            if not len(xs):
                raise ValueError('Empty current-source person exclusion')
            self.boxes.append(np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1], float))

    def build(self, pts, raw):
        right = int(np.clip(np.searchsorted(self.pts, pts), 0, len(self.pts)-1))
        left = max(0, right-1) if pts <= self.pts[right] else right
        fraction = 0. if left == right else float((pts-self.pts[left])/(self.pts[right]-self.pts[left]))
        top = self.boundaries[left]['top']*(1-fraction) + self.boundaries[right]['top']*fraction
        bottom = self.boundaries[left]['contact']*(1-fraction) + self.boundaries[right]['contact']*fraction
        # Bounding envelope is exclusion only: it cannot fabricate material or
        # source readouts. Wide padding covers intermediate arms and pale skates.
        a, b = self.boxes[left], self.boxes[right]
        box = np.r_[np.minimum(a[:2], b[:2])-[100, 100], np.maximum(a[2:], b[2:])+[100, 100]]
        box = np.clip(box, [0, 0, 0, 0], [1920, 1080, 1920, 1080]).astype(int)
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        blue = (hsv[:, :, 0] >= 93)&(hsv[:, :, 0] <= 130)&(hsv[:, :, 1] >= 70)&(hsv[:, :, 2] >= 45)
        yellow = (hsv[:, :, 0] >= 15)&(hsv[:, :, 0] <= 42)&(hsv[:, :, 1] >= 65)&(hsv[:, :, 2] >= 70)
        yy = np.arange(1080)[:, None]
        rail = blue & (abs(yy-top[None]) <= 35)
        contact = yellow & (abs(yy-bottom[None]) <= 35)
        rail_good = rail.any(0); contact_good = contact.any(0)
        source_top = np.argmax(rail, axis=0).astype(float)
        source_bottom = 1079-np.argmax(contact[::-1], axis=0).astype(float)
        if rail_good.sum() >= 500:
            source_top = median_filter(np.interp(np.arange(1920), np.flatnonzero(rail_good), source_top[rail_good]), size=31)
            top = np.maximum(top, source_top)
        if contact_good.sum() >= 500:
            source_bottom = median_filter(np.interp(np.arange(1920), np.flatnonzero(contact_good), source_bottom[contact_good]), size=31)
            # Board uses the higher boundary; ice uses the lower, leaving an
            # explicit uncertain gap rather than sharing ambiguous pixels.
            board_bottom = np.minimum(bottom, source_bottom)
            ice_bottom = np.maximum(bottom, source_bottom)
        else:
            board_bottom = bottom-35; ice_bottom = bottom+35
        person = np.zeros((1080, 1920), np.uint8)
        person[box[1]:box[3], box[0]:box[2]] = 255
        masks = {'board': ((yy > top[None]+12)&(yy < board_bottom[None]-12)).astype(np.uint8)*255,
                 'ice': (yy > ice_bottom[None]+35).astype(np.uint8)*255}
        for mask in masks.values():
            mask[person > 0] = 0; mask[:140, :600] = 0; mask[930:, 1640:] = 0
            mask[:20] = 0; mask[-20:] = 0; mask[:, :20] = 0; mask[:, -20:] = 0
        receipt = dict(native_pts=pts, time_base='1/12800',
            exclusion_anchor_pts=[int(self.pts[left]), int(self.pts[right])], person_exclusion_box=box.tolist(),
            current_native_blue_rail_columns=int(rail_good.sum()), current_native_yellow_contact_columns=int(contact_good.sum()),
            interpolated_curves_used_only_to_locate_current_colour_and_exclude_uncertain_material=True,
            domains_are_navigation_candidates_not_surface_identity_acceptance=True)
        return masks, receipt


def pair(previous, current, previous_mask, current_mask, region):
    # At maxLevel=1, 13px LK plus blur sees at most 18px around a centre.
    # All pyramid input outside the physical-domain candidate is removed first.
    safe_a = cv2.erode(previous_mask, np.ones((39, 39), np.uint8))
    safe_b = cv2.erode(current_mask, np.ones((39, 39), np.uint8))
    ga = cv2.GaussianBlur(np.where(previous_mask > 0, previous, 0).astype(np.uint8), (5, 5), 1.)
    gb = cv2.GaussianBlur(np.where(current_mask > 0, current, 0).astype(np.uint8), (5, 5), 1.)
    p = _features(ga, safe_a, 1200)
    base = dict(region=region, usable=False, raw_pixel_tracks_are_navigation_only=True,
                field_observations_emitted=False, actual_native_frame_pair=True,
                camera_or_field_model_fitted=False, source_domain_and_patch_masks_isolated=True)
    if len(p) < 12:
        return dict(base, reason='insufficient_source_material_texture', detected=len(p)), None
    params = dict(winSize=(13, 13), maxLevel=1,
                  criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 40, .005), minEigThreshold=1e-6)
    q, sf, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p[:, None], None, **params)
    if q is None:
        return dict(base, reason='forward_lk_missing'), None
    z, sb, _ = cv2.calcOpticalFlowPyrLK(gb, ga, q, p[:, None].copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
    if z is None:
        return dict(base, reason='backward_lk_missing'), None
    q, z = q[:, 0], z[:, 0]
    ij = np.rint(np.nan_to_num(q, nan=-1000, posinf=-1000, neginf=-1000)).astype(int)
    inside = np.isfinite(q).all(1)&(ij[:, 0] >= 0)&(ij[:, 0] < 1920)&(ij[:, 1] >= 0)&(ij[:, 1] < 1080)
    fb = np.linalg.norm(z-p, axis=1)
    good = inside&(sf[:, 0] > 0)&(sb[:, 0] > 0)&(fb < 1.)
    ii = np.flatnonzero(good); good[ii] &= safe_b[ij[ii, 1], ij[ii, 0]] > 0
    correlation = np.zeros(len(p), float)
    for j in np.flatnonzero(good):
        a = cv2.getRectSubPix(previous, (9, 9), tuple(map(float, p[j]))).astype(float)
        b = cv2.getRectSubPix(current, (9, 9), tuple(map(float, q[j]))).astype(float)
        if min(a.std(), b.std()) < 1.:
            good[j] = False; continue
        a -= a.mean(); b -= b.mean()
        correlation[j] = float((a*b).sum()/max(np.linalg.norm(a)*np.linalg.norm(b), 1.e-9))
    good &= correlation > .85
    a, b = p[good], q[good]
    config = RegistrationConfig(min_tracks=12, min_inliers=10, min_support_fraction=.001,
        min_support_width_fraction=.10, min_support_height_fraction=.035, ransac_threshold_px=1.7)
    fit = _fit(a, b, previous.shape, config)
    raw_fit_reason = fit.get('reason')
    consensus = np.asarray(fit.get('inlier_flags', [False]*len(a)), bool)
    # Pale ice compression texture can dominate the raw corner population.
    # Keep that failure, then qualify the actual local static consensus on its
    # own spatial extent and residual. It remains navigation, never a surface
    # label or field observation; rejecting white texture does not create one.
    if fit['status'] != 'candidate' and fit.get('reason') == 'too_few_ice_plane_inliers' and len(a) >= 12:
        h, flags = cv2.findHomography(a, b, cv2.RANSAC, 1.7, maxIters=3000, confidence=.999)
        if h is not None and flags is not None:
            consensus = flags.ravel() > 0
            if consensus.sum() >= 12:
                qualified = _fit(a[consensus], b[consensus], previous.shape, config)
                if qualified['status'] == 'candidate':
                    final = consensus.copy(); final[consensus] = np.asarray(qualified['inlier_flags'], bool)
                    fit = qualified; consensus = final
    # _fit maps its first argument to its second: here previous -> current.
    result = dict(base, detected=len(p), accepted_lk_tracks=len(a),
        lk_forward_backward_median_px=float(np.median(fb[good])) if good.any() else None,
        lk_forward_backward_p95_px=float(np.percentile(fb[good], 95)) if good.any() else None,
        patch_correlation_median=float(np.median(correlation[good])) if good.any() else None,
        original_all_texture_consensus_reason=raw_fit_reason,
        coherent_static_track_count=int(consensus.sum()), coherent_static_track_fraction=float(consensus.mean()) if len(consensus) else 0.,
        **{k:v for k, v in fit.items() if k not in ('H_reference_from_current',)})
    tracks = dict(previous_uv=a, current_uv=b, forward_backward_px=fb[good], patch_correlation=correlation[good],
                  inliers=consensus, detected_previous_uv=p, detected_current_uv=q, detected_reverse_uv=z,
                  forward_status=sf[:, 0], reverse_status=sb[:, 0], accepted_lk_filter=good,
                  all_forward_backward_px=fb, all_patch_correlation=correlation)
    if fit['status'] == 'candidate':
        result.update(usable=True, H_current_from_previous=fit['H_reference_from_current'],
            support_previous_uv=fit['support_polygon'], geometry_level='source_local_projective_navigation',
            homography_is_not_camera_solution=True, individual_texture_material_identity_not_claimed=True)
    return result, tracks


def run(data, reference, anchors=None):
    started = time.perf_counter(); cv2.setNumThreads(2)
    manifest = read(data / 'frame_manifest.json'); source = manifest['source']; scope = manifest['shot_scope']
    out = data / 'temporal_diagnostics'; out.mkdir(exist_ok=True)
    domains = SourceDomains(reference); records = []; exclusions = []; previous = None
    selected = {int(r['native_pts']):int(r['index']) for r in manifest['frames']}
    track_arrays = {}; previous_pts = None; selected_states = {}
    for pts, raw in decoded(Path(source['path']), scope['first_pts'], scope['stop_pts_exclusive']):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY); masks, mask_record = domains.build(pts, raw)
        mask_record.update(decoded_BGR_sha256=sha_array(raw), domain_mask_sha256={k:sha_array(v) for k,v in masks.items()})
        exclusions.append(mask_record)
        if pts in selected:
            selected_states[pts] = dict(index=selected[pts], native_pts=pts, mask_provenance=mask_record)
        if previous is not None:
            old_gray, old_masks = previous
            for region in ('ice', 'board'):
                record, tracks = pair(old_gray, gray, old_masks[region], masks[region], region)
                record.update(previous_pts=previous_pts, current_pts=pts, time_base='1/12800', elapsed_source_seconds=(pts-previous_pts)/12800)
                key = f'{region}_{previous_pts}_{pts}'
                if tracks is not None:
                    for field, values in tracks.items(): track_arrays[key+'_'+field] = values
                    record['track_array_prefix'] = key
                records.append(record)
        previous = (gray, masks); previous_pts = pts
        if len(exclusions) % 100 == 0:
            print(json.dumps(dict(stage='native_material_navigation', native_pts=pts, exposures=len(exclusions),
                usable_ice=sum(r['usable'] for r in records if r['region']=='ice'),
                usable_board=sum(r['usable'] for r in records if r['region']=='board'))), flush=True)
    tracks_path = out / 'native_material_tracks.npz'; np.savez_compressed(tracks_path, **track_arrays)
    summary = dict(schema='rinkspace.native-material-navigation.v1', source=source, shot_scope=scope,
        domain_input_provenance=domains.provenance, module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_native_exposures_processed=len(exclusions), selected_pts=sorted(selected), selected_states=list(selected_states.values()),
        pair_count=len(records), usable_counts={k:sum(r['usable'] for r in records if r['region']==k) for k in ('ice', 'board')},
        all_intermediate_source_frames_navigation_only=True, intermediate_observation_packets_created=False,
        source_pixels_unchanged=True, model_or_camera_modified=False, source_sampling_modified=False,
        track_arrays_path=str(tracks_path), records=records, exclusion_states=exclusions,
        elapsed_seconds=time.perf_counter()-started)
    write(out / 'native_material_navigation.json', summary)
    if anchors:
        transport_anchors(out, summary, read(anchors))
    print(json.dumps({k:summary[k] for k in ('official_native_exposures_processed', 'pair_count', 'usable_counts', 'elapsed_seconds')}), flush=True)
    return summary


def transport_anchors(out, summary, anchor_file):
    lookup = {(r['region'], r['previous_pts'], r['current_pts']):r for r in summary['records']}
    guides = []
    for anchor in anchor_file['anchors']:
        region = anchor.get('region', 'ice'); start = int(anchor['source_pts_anchor'])
        for target in summary['selected_pts']:
            points = np.asarray(anchor['uv'], float); errors = []; failures = []; support = []
            step = 256 if target >= start else -256; current = start
            while current != target:
                next_pts = current+step; pair_key = (region, min(current, next_pts), max(current, next_pts)); record = lookup.get(pair_key)
                if not record or not record['usable']:
                    failures.append(dict(previous_pts=current, target_pts=next_pts, reason=(record or {}).get('reason', 'native_pair_missing'))); break
                matrix = np.asarray(record['H_current_from_previous'], float)
                if step < 0: matrix = np.linalg.inv(matrix)
                points = cv2.perspectiveTransform(points[None], matrix)[0]
                errors.append(float(record.get('lk_forward_backward_p95_px') or 0.))
                support.append(record['accepted_lk_tracks']); current = next_pts
            guides.append(dict(target_id=anchor['target_id'], region=region, source_pts_anchor=start, target_pts=target,
                target_programme_index=(target-256256)//256+1, uv=points.tolist() if not failures else [],
                usable=not failures, failure_links=failures, native_adjacent_links=len(errors),
                max_pair_forward_backward_p95_px=max(errors) if errors else 0.,
                accumulated_fb_rss_px=float(np.linalg.norm(errors)), minimum_pair_tracks=min(support) if support else None,
                navigation_only=True, coordinates_are_not_current_frame_material_readouts=True,
                source_anchor_authority=anchor.get('authority'), current_target_material_reread_required=True,
                accumulated_fb_is_diagnostic_not_accuracy_or_identity_proof=True))
    write(out / 'transported_source_guides.json', dict(schema='rinkspace.source-guide-native-navigation.v1',
        anchors=anchor_file['anchors'], guides=guides, field_observations_created=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--boundary-reference', type=Path, required=True)
    parser.add_argument('--anchors', type=Path)
    args = parser.parse_args(); run(args.data, args.boundary_reference, args.anchors)
