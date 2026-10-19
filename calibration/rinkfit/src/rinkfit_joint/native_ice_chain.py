"""Adjacent-native-frame ice material navigation and actual RGB remeasurement.

Optical flow only transports source search regions. Every surviving track is
remeasured by the existing current-RGB reader; its output, never flow or a
smoothed coordinate, becomes the current source mean. Only the previous gray
image/mask and compact current tracks remain resident.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import numpy as np
import cv2
from .io_utils import read_json, read_jsonl, file_sha256, object_sha256
from rinkspace.vision_ice_temporal_identity import (_flow_transform,
    _curve_local_navigation, _measured_material_descriptors)

VERSION = 'adjacent_native_ice_material_chain_v1'
_METADATA = ('target_id','joint_material_id','joint_material_index','physical_surface',
             'measurement_operator_resolved','parent_legacy_row','category')


def _reference(value):
    p = Path(value.get('file', value.get('path'))).resolve()
    if file_sha256(p) != value['sha256']:
        raise ValueError('Changed source-qualified temporal anchor: ' + str(p))
    return p


def _source_support(row, result=None):
    q = result if result is not None else row['source_operator_result']
    uv = np.asarray(q['current_uv'],float)
    n = np.asarray(q.get('normal',row.get('normal')),float)
    if uv.shape != (2,) or n.shape != (2,) or not np.isfinite(uv).all() or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
        raise ValueError('Temporal anchor lacks a current finite source mean/normal')
    n = n/np.linalg.norm(n)
    op = row['measurement_operator_resolved']
    if op == 'paint_center':
        sides = np.asarray(q.get('flank_uv'),float)
        if sides.shape != (2,2) or not np.isfinite(sides).all():
            raise ValueError('Paint navigation requires both actual source flanks')
    elif op == 'material_interface':
        # These are actual adjacent material context pixels, not paint flanks.
        sides = uv + np.array([-6.,6.])[:,None]*n
    else:
        raise ValueError('Unsupported temporal ice source operator')
    return uv,n,sides


def _descriptor(rgb,uv,sides,normal):
    a,strength = _measured_material_descriptors(rgb,uv[None],sides[None],normal[None])
    return a[0],float(strength[0])


class NativeIceChain:
    def __init__(self, *, source_sha256, shot_id, role_guard, time_base='1/12800', direction=1):
        if str(Fraction(time_base)) != '1/12800' or direction not in (-1,1):
            raise ValueError('Native ice chain requires the official clock and an explicit adjacent-frame direction')
        self.source_sha256 = source_sha256; self.shot_id = shot_id
        self.role_guard = role_guard; self.time_base = '1/12800'; self.direction = direction
        self.previous_gray = self.previous_mask = self.previous_pts = None
        self.tracks = {}; self.qualification_ref = None

    def clear(self):
        self.previous_gray = self.previous_mask = self.previous_pts = None
        self.tracks = {}

    def seed(self, rgb, mask, native_pts, source_rows, *, source_qualification_ref):
        self.role_guard.assert_training_pixels([int(native_pts)],shot_id=self.shot_id)
        source_rows = list(source_rows)
        bundle_path = _reference(source_qualification_ref); bundle = read_json(bundle_path)
        requested = {r['source_row_id']:r for r in source_rows}
        if len(requested) != len(source_rows) or not requested:
            raise ValueError('Temporal seeds require unique nonempty actual source rows')
        bound = {}
        for chunk_ref in bundle.get('closed_chunks', []):
            chunk = read_json(_reference(chunk_ref))
            for row in read_jsonl(_reference(chunk['geometry_rows_ref'])):
                rid = row['source_row_id']
                if rid in requested:
                    bound[rid] = row
        if set(bound) != set(requested) or any(object_sha256(bound[k]) != object_sha256(v) for k,v in requested.items()):
            raise ValueError('Temporal source rows do not exactly belong to the qualified anchor bundle')
        rgb = np.asarray(rgb); mask = np.asarray(mask,bool)
        if rgb.shape != (1080,1920,3) or mask.shape != (1080,1920):
            raise ValueError('Wrong native chain image or mask domain')
        digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        tracks = {}
        for row in source_rows:
            b = row['native_source_binding']
            if (row.get('source_geometry_eligible') is not True
                or row.get('point_correspondence') is not False
                or row['shot_id'] != self.shot_id or int(row['native_pts']) != int(native_pts)
                or b.get('source_video_sha256') != self.source_sha256
                or b.get('native_pts') != int(native_pts) or b.get('time_base') != self.time_base
                or b.get('decoded_native_RGB_sha256') != digest):
                raise ValueError('Wrong qualified native temporal source anchor')
            uv,n,sides = _source_support(row); xy = np.rint(uv).astype(int)
            if not (0 <= xy[0] < 1920 and 0 <= xy[1] < 1080) or mask[xy[1],xy[0]]:
                raise ValueError('Temporal anchor is excluded in this actual native frame')
            d,strength = _descriptor(rgb,uv,sides,n)
            if strength <= 0:
                raise ValueError('Temporal anchor has no complete actual material context')
            track = {name:row.get(name) for name in _METADATA}
            track.update(track_id=row['source_row_id'],anchor_source_row_id=row['source_row_id'],
                anchor_native_pts=int(native_pts),current_uv=uv,normal=n,candidate_sides=sides,
                material_descriptor=d,source_operator_result=row['source_operator_result'],
                actual_pts=int(native_pts),navigation_pts=int(native_pts))
            tracks[track['track_id']] = track
        self.previous_gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.previous_mask = mask.copy(); self.previous_pts = int(native_pts)
        self.tracks = tracks; self.qualification_ref = dict(source_qualification_ref)
        return dict(version=VERSION,status='qualified_native_source_seeded',native_pts=int(native_pts),
            tracks=len(tracks),source_qualification_ref=self.qualification_ref,
            source_RGB_sha256=digest,flow_coordinates_are_observations=False)

    def advance(self, rgb, mask, native_pts, reader):
        pts = int(native_pts)
        try:
            self.role_guard.assert_training_pixels([pts],shot_id=self.shot_id)
        except ValueError as error:
            self.clear()
            return dict(version=VERSION,status='protected_or_cross_cut_source_chain_broken',native_pts=pts,
                reason=str(error),current_reads=[],dropped=[],native_RGB_consumed=False)
        if self.previous_pts is None or not self.tracks:
            return dict(version=VERSION,status='source_qualified_reentry_required',native_pts=pts,
                current_reads=[],dropped=[],native_RGB_consumed=False)
        if pts != self.previous_pts + self.direction*256:
            old = self.previous_pts; self.clear()
            raise ValueError('Native ice identity chain cannot skip exposures: ' + str(old) + ' -> ' + str(pts))
        rgb = np.asarray(rgb); mask = np.asarray(mask,bool)
        if rgb.shape != (1080,1920,3) or mask.shape != (1080,1920):
            raise ValueError('Wrong adjacent native source image/mask domain')
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        previous_pts = self.previous_pts
        groups = {}
        for track in self.tracks.values():
            # Physical instance and operator are immutable throughout a chain.
            key = (track['target_id'],track['measurement_operator_resolved'])
            groups.setdefault(key,[]).append(track)
        current_reads = []; dropped = []; navigation = []; retained = {}
        for (target,op), members in groups.items():
            uv = np.asarray([t['current_uv']for t in members]); n = np.asarray([t['normal']for t in members])
            sides = np.asarray([t['candidate_sides']for t in members]); tangent = np.c_[-n[:,1],n[:,0]]
            # Centre, both material sides, and both local tangent supports take
            # part in actual-image forward/backward LK navigation.
            support = np.concatenate([uv,uv-4*tangent,uv+4*tangent])
            side_support = np.concatenate([sides,sides-4*tangent[:,None],sides+4*tangent[:,None]])
            transform, global_nav = _flow_transform(self.previous_gray,gray,self.previous_mask,mask,
                {'material':dict(actual_uv=support,candidate_sides=side_support)})
            if transform is not None:
                def apply(points):
                    h = np.c_[points,np.ones(len(points))]@transform.T
                    return h[:,:2]/h[:,2,None]
                proposal = apply(uv); tang2 = apply(uv+4*tangent)-apply(uv-4*tangent)
                length = np.linalg.norm(tang2,axis=1)
                current_n = np.c_[-tang2[:,1],tang2[:,0]]/np.maximum(length[:,None],1e-9)
                current_n[(current_n*n).sum(1)<0] *= -1
                nav = dict(global_nav,source_RGB_tracking_confirmed=True,actual_tangent_supports_included=True)
            else:
                proposal, nav = _curve_local_navigation(self.previous_gray,gray,self.previous_mask,mask,
                    dict(actual_uv=uv,navigation_uv=uv,candidate_sides=sides,
                        actual_pts=previous_pts,navigation_pts=previous_pts))
                current_n = n.copy()
                nav = dict(nav,global_navigation=global_nav,actual_tangent_supports_included=True)
            aperture_reacquisition = not nav.get('source_RGB_tracking_confirmed')
            if aperture_reacquisition:
                # Flat paint interfaces often have no tangential LK texture.
                # Retain the last actual material ROI for an adjacent-frame
                # normal-only reread; uniqueness, fixed layer/operator and
                # current material context below must independently pass.
                proposal = uv.copy(); current_n = n.copy()
                nav = dict(status='adjacent_current_material_profile_reacquisition_required',
                    source_RGB_tracking_confirmed=False, LK_verified=False,
                    previous_LK_result=nav, normal_only=True, full_2D_homography_estimated=False,
                    used_as_fitting_observations=False, used_as_UV_output=False)
            nav_record = dict(target_id=target,operator=op,previous_native_pts=previous_pts,native_pts=pts,
                **nav,interface_context_pixels_are_paint_flanks=op=='paint_center')
            navigation.append(nav_record)
            group_read_count = 0
            for i,track in enumerate(members):
                seed = proposal[i]; normal = current_n[i]
                if not np.isfinite(seed).all() or not np.isfinite(normal).all():
                    dropped.append(dict(track_id=track['track_id'],target_id=target,reason='nonfinite_source_navigation')); continue
                # The callback must execute the existing RGB reader in this
                # current frame. The transported position is never emitted UV.
                q = reader(track,seed.copy(),normal.copy())
                if not isinstance(q,dict) or q.get('current_uv') is None:
                    dropped.append(dict(track_id=track['track_id'],target_id=target,reason=(q or {}).get('status','actual_current_reader_failed'))); continue
                previous_receipt = track['source_operator_result']['source_material_receipt']
                receipt = q.get('source_material_receipt',{})
                immutable = ('operator_version','material_signal') if op=='paint_center' else ('operator_version','layer_role','expected_outward_luminance_polarity')
                if any(receipt.get(k)!=previous_receipt.get(k)for k in immutable):
                    dropped.append(dict(track_id=track['track_id'],target_id=target,reason='current_read_switched_material_layer_or_operator')); continue
                if receipt.get('configuration') != previous_receipt.get('configuration'):
                    dropped.append(dict(track_id=track['track_id'],target_id=target,reason='current_read_changed_frozen_reader_configuration')); continue
                current_row = dict(track,source_operator_result=q)
                now,normal_now,sides_now = _source_support(current_row)
                xy = np.rint(now).astype(int)
                if not(0<=xy[0]<1920 and 0<=xy[1]<1080)or mask[xy[1],xy[0]]:
                    dropped.append(dict(track_id=track['track_id'],target_id=target,reason='current_source_mean_excluded')); continue
                descriptor,strength = _descriptor(rgb,now,sides_now,normal_now)
                correlation = float(descriptor@track['material_descriptor'])
                if strength<=0 or correlation<.9:
                    dropped.append(dict(track_id=track['track_id'],target_id=target,reason='current_material_context_changed_or_competing',correlation=correlation)); continue
                temporal = dict(version=VERSION,anchor_source_row_id=track['anchor_source_row_id'],
                    anchor_native_pts=track['anchor_native_pts'],previous_native_pts=previous_pts,native_pts=pts,
                    direction=self.direction,source_video_sha256=self.source_sha256,time_base=self.time_base,
                    shot_id=self.shot_id,target_id=target,measurement_operator=op,
                    source_RGB_sha256=digest,source_qualification_ref=self.qualification_ref,
                    navigation_only=True,current_UV_from_actual_RGB_reader=True,
                    flow_or_smoothed_coordinates_installed_as_UV=False,
                    current_material_descriptor_correlation=correlation,
                    current_readout_sha256=object_sha256(receipt),
                    actual_native_step=256,previous_source_state_sha256=object_sha256(dict(
                        pts=previous_pts,target_id=target,current_uv=track['current_uv'].tolist(),
                        source_readout_sha256=object_sha256(previous_receipt))))
                value = {name:track.get(name)for name in _METADATA}
                value.update(track_id=track['track_id'],anchor_source_row_id=track['anchor_source_row_id'],
                    anchor_native_pts=track['anchor_native_pts'],result=q,temporal_identity_receipt=temporal)
                current_reads.append(value)
                group_read_count += 1
                temporal['normal_motion_from_actual_current_source_mean_px'] = float((now-track['current_uv'])@track['normal'])
                temporal['navigation_mode'] = 'adjacent_actual_material_profile_normal_reacquisition' if aperture_reacquisition else nav['status']
                temporal['LK_verified'] = not aperture_reacquisition
                state = dict(track,current_uv=now,normal=normal_now,candidate_sides=sides_now,
                    material_descriptor=descriptor,source_operator_result=q,actual_pts=pts,navigation_pts=pts)
                retained[state['track_id']] = state
            if aperture_reacquisition:
                nav_record.update(status='adjacent_actual_material_profile_normal_reacquisition' if group_read_count else 'actual_current_material_profile_reacquisition_failed',
                    source_RGB_tracking_confirmed=bool(group_read_count),actual_current_source_readouts=group_read_count,
                    actual_reader_and_material_context_verified=bool(group_read_count))
        self.previous_gray = gray; self.previous_mask = mask.copy(); self.previous_pts = pts
        self.tracks = retained
        return dict(version=VERSION,status='actual_current_native_material_reread'if retained else 'source_qualified_reentry_required',
            previous_native_pts=previous_pts,native_pts=pts,source_RGB_sha256=digest,
            current_reads=current_reads,dropped=dropped,navigation=navigation,
            retained_tracks=len(retained),flow_coordinates_are_observations=False,native_RGB_consumed=True)
