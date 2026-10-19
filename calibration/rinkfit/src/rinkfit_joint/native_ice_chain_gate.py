"""Verify a complete adjacent-native material witness before source sealing.

This is a continuity verifier, not a source qualification certificate. The
production material gate must separately replay and review the selected current
native pixels. Flow never supplies observed UV or a statistical precision.
"""
from pathlib import Path
import hashlib
import numpy as np
from .io_utils import read_json, read_jsonl, file_sha256, object_sha256
from .role_guard import FrameRoleGuard
from .native_ice_profile_motion import normal_material_profile_correspondence


def validate_native_ice_chain(acquisition, selected_rows):
    a = read_json(acquisition) if not isinstance(acquisition,dict) else acquisition
    checked = {}
    def verify(r):
        p = Path(r.get('file',r.get('path'))).resolve(); key = (str(p),r['sha256'])
        if key not in checked:
            if file_sha256(p) != r['sha256']:
                raise ValueError('Changed native material chain evidence: '+str(p))
            checked[key] = p
        return p
    needed = ('native_temporal_chain','native_temporal_actual_readouts','native_temporal_chain_code','source_identity_anchors','source_roles','independent_actor_ledgers')
    if any(k not in a for k in needed):
        raise ValueError('A long source identity continuation requires the actual complete native chain artifacts')
    code = verify(a['native_temporal_chain_code'])
    if a['native_temporal_chain_code']['sha256'] != file_sha256(Path(__file__).with_name('native_ice_chain.py')):
        raise ValueError('Native source chain implementation changed after execution')
    source = a['source_video']
    if (source['id'] != 'O6j8L8TmfKo' or source['time_base'] != '1/12800'
        or source['native_fps'] != '50/1' or source['native_pts_step'] != 256
        or file_sha256(source['path']) != source['sha256']):
        raise ValueError('Native chain is not bound to the official source bytes and clock')
    guard = FrameRoleGuard(verify(a['source_roles']))
    anchor_refs = a['source_identity_anchors']
    if isinstance(anchor_refs,dict):anchor_refs = [anchor_refs]
    anchors = {}; anchor_bundle_sha = set()
    for br in anchor_refs:
        b = read_json(verify(br)); anchor_bundle_sha.add(br['sha256'])
        if not b.get('source_ledger_preserved'):
            raise ValueError('Temporal seed is not an immutable source-qualified bundle')
        for cr in b.get('closed_chunks',[]):
            chunk = read_json(verify(cr))
            for r in read_jsonl(verify(chunk['geometry_rows_ref'])):
                if r.get('source_geometry_eligible') is True:
                    rid = r['source_row_id']
                    if rid in anchors and object_sha256(r) != object_sha256(anchors[rid]):
                        raise ValueError('Ambiguous qualified source anchor ID')
                    anchors[rid] = r
    frame_index = {}
    for frame in read_jsonl(verify(a['native_temporal_chain'])):
        sid = frame.get('shot_id'); pts = frame.get('native_pts')
        if sid is None or pts is None:raise ValueError('Native chain step lacks its source clock/shot')
        pts = int(pts); key = (sid,pts)
        old = frame_index.get(key,{})
        digest = frame.get('source_RGB_sha256')
        if old.get('source_RGB_sha256') and digest and old['source_RGB_sha256'] != digest:
            raise ValueError('One native source step contains conflicting RGB bindings')
        if digest:
            guard.assert_training_pixels([pts],shot_id=sid)
            old['source_RGB_sha256'] = digest
        old['status'] = frame.get('status',old.get('status'))
        old['native_RGB_consumed'] = frame.get('native_RGB_consumed',old.get('native_RGB_consumed',False))
        old['readout_hashes'] = frame.get('current_readout_hashes',old.get('readout_hashes'))
        frame_index[key] = old
    rgb_bindings = {}
    for lr in a['independent_actor_ledgers']:
        for ld in read_jsonl(verify(lr)):
            key = (ld['shot_id'],int(ld['native_pts']))
            if key not in frame_index or not frame_index[key].get('source_RGB_sha256'):continue
            digests = set()
            if ld.get('decoded_native_RGB_sha256'):digests.add(ld['decoded_native_RGB_sha256'])
            for actor in ld.get('current_person_boxes',[]):
                b = actor['source_binding']
                if (b['source_pts'] != key[1] or b['shot_id'] != key[0]
                    or b['source_sha256'] != source['sha256']):
                    raise ValueError('Intermediate actor/frame binding belongs to another native source')
                digests.add(b['decoded_native_rgb_sha256'])
            if digests and digests != {frame_index[key]['source_RGB_sha256']}:
                raise ValueError('Actual intermediate RGB differs from independent native actor source ledger')
            rgb_bindings[key] = digests
    states = {}; selected = {}; readouts = 0; used_frames = set(); directions = set(); invalid_branches = set(); material_continuity_failures = []
    for item in read_jsonl(verify(a['native_temporal_actual_readouts'])):
        t = item['temporal_identity_receipt']; q = item['result']; target = item['target_id']
        sid = item['shot_id']; pts = int(item['native_pts']); key = (sid,pts)
        guard.assert_training_pixels([pts],shot_id=sid)
        anchor_id = t['anchor_source_row_id']; anchor = anchors.get(anchor_id)
        if anchor is None:raise ValueError('Native identity chain has no exact source-qualified starting row')
        br = t['source_qualification_ref']
        verify(br)
        if br['sha256'] not in anchor_bundle_sha:raise ValueError('Current native chain uses a different qualified source anchor')
        op = anchor['measurement_operator_resolved']; direction = int(t['direction'])
        if direction not in (-1,1):raise ValueError('Invalid adjacent source chain direction')
        directions.add(direction)
        if (t['source_video_sha256'] != source['sha256'] or t['time_base'] != '1/12800'
            or t['shot_id'] != sid or t['native_pts'] != pts
            or item['target_id'] != anchor['target_id'] or t['target_id'] != target
            or t['measurement_operator'] != op or item['measurement_operator_resolved'] != op
            or item.get('joint_material_index') != anchor.get('joint_material_index')
            or item.get('joint_material_id') != anchor.get('joint_material_id')
            or t['flow_or_smoothed_coordinates_installed_as_UV'] is not False
            or t['current_UV_from_actual_RGB_reader'] is not True
            or t['actual_native_step'] != 256
            or t['current_material_descriptor_correlation'] < .9):
            raise ValueError('Native chain changed source identity, material, operator or actual readout meaning')
        digest = t['source_RGB_sha256']
        if key not in frame_index or frame_index[key].get('source_RGB_sha256') != digest or item['source_RGB_sha256'] != digest:
            raise ValueError('Native readout is absent from the actual frame chain')
        if not rgb_bindings.get(key):
            # Rare frames without an actor still need an independent exact
            # native-byte check; decode only that one exposure, never all RGB.
            import av
            container = av.open(source['path']); stream = container.streams.video[0]
            container.seek(pts,stream=stream); found = None
            for f in container.decode(stream):
                if f.pts == pts:found = f.to_ndarray(format='rgb24'); break
                if f.pts > pts:break
            container.close()
            if found is None or hashlib.sha256(found.tobytes()).hexdigest() != digest:
                raise ValueError('Intermediate native RGB has no independent exact-source binding')
            rgb_bindings[key] = {digest}
        track_id = item['track_id']; state_key = (sid,direction,track_id); previous = states.get(state_key)
        if previous is None:
            previous = dict(pts=int(anchor['native_pts']),uv=anchor['current_uv'],receipt=anchor['source_operator_result']['source_material_receipt'],result=anchor['source_operator_result'])
            if int(t['anchor_native_pts']) != previous['pts']:
                raise ValueError('Temporal current read does not start at the exact qualified source exposure')
        if int(t['previous_native_pts']) != previous['pts'] or pts != previous['pts']+direction*256:
            raise ValueError('Native identity witness skips an exposure or silently rejoins a lost track')
        guard.assert_training_pixels([previous['pts']],shot_id=sid)
        expected = object_sha256(dict(pts=previous['pts'],target_id=target,current_uv=previous['uv'],source_readout_sha256=object_sha256(previous['receipt'])))
        if t['previous_source_state_sha256'] != expected:
            raise ValueError('Native chain predecessor state differs from the actual previous source read')
        receipt = q['source_material_receipt']; mean = np.asarray(q['current_uv'],float)
        if mean.shape != (2,) or not np.isfinite(mean).all() or t['current_readout_sha256'] != object_sha256(receipt):
            raise ValueError('Current native source mean/readout hash is not exact')
        immutable = ('operator_version','material_signal') if op=='paint_center' else ('operator_version','layer_role','expected_outward_luminance_polarity')
        if any(receipt.get(k) != previous['receipt'].get(k)for k in immutable) or receipt.get('configuration') != previous['receipt'].get('configuration'):
            raise ValueError('Native chain switched target layer, material channel, or frozen source operator')
        announced = frame_index[key].get('readout_hashes')
        if announced is not None and not any(r['track_id']==track_id and r['target_id']==target and r['readout_sha256']==t['current_readout_sha256']for r in announced):
            raise ValueError('Native frame ledger does not bind this actual source readout')
        profile_motion = None
        if t.get('LK_verified') is False:
            profile_motion = normal_material_profile_correspondence(previous['result'],q,previous['uv'],previous['result']['normal'])
            if not profile_motion['unique_source_material_profile_match']:
                invalid_branches.add(state_key)
                material_continuity_failures.append(dict(shot_id=sid,native_pts=pts,track_id=track_id,target_id=target,reason='actual_complete_signed_material_context_has_no_unique_same_layer_continuation',profile_motion=profile_motion))
        elif t.get('LK_verified') is not True:
            raise ValueError('Native source witness does not distinguish LK from normal-only material reacquisition')
        states[state_key] = dict(pts=pts,uv=mean.tolist(),receipt=receipt,result=q)
        selected[(sid,pts,track_id)] = dict(target_id=target,uv=mean.tolist(),temporal=t,
            source_readout_sha256=object_sha256(receipt),continuity_verified=state_key not in invalid_branches)
        readouts += 1; used_frames.add(key)
    verified_ids = []
    for row in selected_rows:
        t = row.get('native_temporal_identity')
        if not t:raise ValueError('Selected long continuation has no current native identity witness')
        key = (row['shot_id'],int(row['native_pts']),t['anchor_source_row_id'])
        found = selected.get(key)
        if (found is None or not found['continuity_verified'] or found['target_id'] != row['target_id']
            or found['uv'] != row['current_uv'] or object_sha256(found['temporal']) != object_sha256(t)
            or found['source_readout_sha256'] != object_sha256(row['source_operator_result']['source_material_receipt'])):
            raise ValueError('Selected source mean does not exactly equal its actual native-chain readout')
        verified_ids.append(row['source_row_id'])
    return dict(schema='rinkfit.native_ice_continuity_verified.v1',
        source_video_sha256=source['sha256'],time_base='1/12800',native_pts_step=256,
        native_actual_readout_steps=len(used_frames),native_actual_readouts=readouts,
        selected_rows=len(verified_ids),verified_source_row_ids=verified_ids,directions=sorted(directions),
        actual_material_continuity_failures=material_continuity_failures,
        verified_current_readouts=[dict(shot_id=k[0],native_pts=k[1],track_id=k[2],target_id=value['target_id'],source_readout_sha256=value['source_readout_sha256'])for k,value in selected.items()if value['continuity_verified']],
        normal_profile_correspondence_verifier=dict(file=str(Path(__file__).with_name('native_ice_profile_motion.py').resolve()),sha256=file_sha256(Path(__file__).with_name('native_ice_profile_motion.py'))),
        source_qualified_anchor_bundles=a['source_identity_anchors'],
        native_temporal_chain=a['native_temporal_chain'],native_actual_readouts_ref=a['native_temporal_actual_readouts'],
        actual_current_selected_pixel_review_and_frozen_operator_replay_still_required=True,
        source_qualification_granted_by_this_continuity_verifier=False,
        optical_flow_or_smoothed_coordinates_used_as_observations=False)
