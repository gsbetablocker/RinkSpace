"""Propagate reviewed shoe points using adjacent original-source pixels only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from source_review import SourceVideo, crop_bounds, tracks


def endpoint_pairs(rows):
    """Keep the two observed ends of one boot/blade together during flow."""
    groups={}
    for i,row in enumerate(rows):
        key=row.get('association_group',f"physical:{row['foot']}")
        groups.setdefault(key,[]).append(i)
    pairs=[]
    for indices in groups.values():
        for features in (('heel','toe'),('runner_rear','runner_front')):
            selected=[i for i in indices if rows[i]['feature'] in features]
            if len(selected)==2 and {rows[i]['feature'] for i in selected}==set(features):
                pairs.append(selected)
    return pairs


def qualify_coupled_flow(document):
    """Reject an existing endpoint track after its paired end loses visibility.

    A good image round trip alone can follow a rink line through a turn. This
    conservative qualification preserves every original pixel and weight; it
    does not fabricate the missing end or relabel a physical foot.
    """
    anchors={};flows={}
    for row in document['samples']:
        if row.get('role')!='fit':continue
        if row.get('source_kind')=='adjacent_source_flow':
            flows.setdefault(row['source_anchor_id'],{})[row['pts']]=row
        elif row.get('source_kind')=='assistant_native_source_review':
            anchors.setdefault((row['pts'],row['foot']),[]).append(row)
    rejected=[]
    for (stamp,foot),rows in anchors.items():
        ids=[r.get('id',f'{stamp}:{foot}:{i}') for i,r in enumerate(rows)]
        for pair in endpoint_pairs(rows):
            a,b=[flows.get(ids[i],{}) for i in pair];shared=set(a)&set(b)
            for track in (a,b):
                for pts,row in track.items():
                    if pts not in shared:
                        row['role']='rejected'
                        row['source_flow_rejection']='paired endpoint lost in original-source tracking; remaining image track is not sufficient material identity evidence'
                        rejected.append(row['id'])
    document['coupled_flow_qualification']=dict(rejected_ids=sorted(set(rejected)),
        rejected_count=len(set(rejected)),pixels_and_weights_unchanged=True,
        independent_accuracy_claim=False)
    return document['coupled_flow_qualification']


def transport_flow_region(row,anchor):
    """Move a source uncertainty region with its tracked material point.

    Leaving the anchor rectangle stationary silently changes a moving image
    observation into an attraction to an earlier frame. The original region
    and every observed/tracked UV stay available for provenance.
    """
    if 'pixel_box' not in row:return False
    origin=np.asarray(row.get('source_region_anchor_box',row['pixel_box']),float)
    if origin.shape!=(2,2) or np.any(origin[0]>origin[1]):
        raise ValueError('Invalid original source flow region')
    shift=np.asarray(row['uv'],float)-np.asarray(anchor['uv'],float)
    row.setdefault('source_region_anchor_box',origin.tolist())
    row['pixel_box']=(origin+shift).tolist()
    row['source_region_motion']='original anchor region translated by verified native-image flow; not a newly detected boundary'
    return True


def transport_existing_flow_regions(document):
    """Repair derived boxes without rerunning or relabelling source tracks."""
    anchors={r['id']:r for r in document['samples']
             if r.get('id') and r.get('source_kind')!='adjacent_source_flow'}
    changed=[];active=[]
    for row in document['samples']:
        if row.get('source_kind')!='adjacent_source_flow' or 'pixel_box' not in row:continue
        anchor=anchors.get(row['source_anchor_id'])
        if anchor is None:raise ValueError('Source region flow is missing its original anchor')
        previous=np.asarray(row['pixel_box'])
        transport_flow_region(row,anchor)
        if not np.array_equal(previous,row['pixel_box']):
            row.setdefault('source_region_previous_pixel_box',previous.tolist())
            changed.append(row['id'])
            if row['role']=='fit':active.append(row['id'])
    record=dict(changed_ids=changed,active_changed_ids=active,
        direct_source_pixels_unchanged=True,tracked_uv_unchanged=True,
        weights_and_roles_unchanged=True,missing_boundaries_not_invented=True)
    document['source_flow_region_transport']=record
    return record


def verify_tracks(source,registration,regions,review_batches=None):
    """Verify claimed cross-frame boot links with both directions of source motion.

    A failed image track makes the association unknown; it does not create an
    alternative anatomical label. The original reviewed coordinates survive.
    """
    doc=json.loads(source.read_text(encoding='utf8'));ledger=dict(np.load(regions))
    pts=ledger['source_pts'];groups={}
    batches=['source_occlusion_1116'] if review_batches is None else review_batches
    for row in doc['samples']:
        if row.get('source_review_batch') in batches and row.get('source_kind')=='assistant_native_source_review' and row.get('role')=='fit':
            groups.setdefault(row.get('association_track',row['id']),[]).append(row)
    media=json.loads(registration.read_text(encoding='utf8'))['primary_source'];reader=SourceVideo(media['path'],[1,12800])
    records=[];changed={}
    try:
        for key,rows in groups.items():
            stamps=sorted({r['pts'] for r in rows})
            if len(stamps)<2:continue
            accepted=True;checks=[]
            for a,b in zip(stamps,stamps[1:]):
                first,last=np.searchsorted(pts,[a,b])
                features=set(r['feature'] for r in rows if r['pts']==a)&set(r['feature'] for r in rows if r['pts']==b)
                features=[v for v in ('cuff_center','heel','toe','runner_rear','runner_front') if v in features]
                if not features or ledger['shot_ids'][first]!=ledger['shot_ids'][last]:
                    accepted=False;checks.append(dict(first=a,last=b,reason='No shared visible feature or source shot'));continue
                start=np.array([next(r['uv'] for r in rows if r['pts']==a and r['feature']==v) for v in features],np.float32)
                target=np.array([next(r['uv'] for r in rows if r['pts']==b and r['feature']==v) for v in features],np.float32)
                grays={int(i):cv2.cvtColor(reader.read(int(pts[i])),cv2.COLOR_RGB2GRAY) for i in range(first,last+1)}
                errors=[];alive_all=[]
                for direction in (1,-1):
                    xy=start.copy() if direction==1 else target.copy();dest=target if direction==1 else start
                    order=range(first,last) if direction==1 else range(last,first,-1)
                    alive=np.ones(len(xy),bool)
                    for i in order:
                        bounds=crop_bounds(np.r_[xy,start,target],grays[i].shape,64)
                        moved,fb,photo=tracks(grays[i],grays[i+direction],xy,bounds)
                        alive&=np.isfinite(moved).all(1)&(fb<1.5)&(photo<25.)
                        xy=np.where(np.isfinite(moved),moved,xy)
                    errors.append(np.linalg.norm(xy-dest,axis=1));alive_all.append(alive)
                good=alive_all[0]&alive_all[1]&(errors[0]<14.)&(errors[1]<14.)
                passed=bool(good.any());accepted&=passed
                checks.append(dict(first=a,last=b,features=features,forward_error=errors[0].tolist(),
                    backward_error=errors[1].tolist(),passed=passed))
            if not accepted:
                for row in rows:
                    row['association_track']=f'{key}:source_motion_unresolved:{row["pts"]}'
                    row['association_note']='Cross-frame source flow did not establish this physical boot link; keep the pixel observation but infer identity separately.'
                    changed[row['id']]=row['association_track']
            records.append(dict(track=key,source_link_retained=accepted,checks=checks))
    finally:reader.close()
    for row in doc['samples']:
        if row.get('source_anchor_id') in changed:row['association_track']=changed[row['source_anchor_id']]
    key='source_occlusion_track_verification' if review_batches is None else 'source_direction_track_verification'
    doc[key]=dict(method='Adjacent native LK in both directions across the full anchor interval; no model projection',review_batches=batches,tracks=records,
        retained=sum(r['source_link_retained'] for r in records),unresolved=sum(not r['source_link_retained'] for r in records))
    source.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in doc[key].items() if k!='tracks'}),flush=True)


def propagate(source, registration, regions, native_neighbours=None):
    document = json.loads(source.read_text(encoding='utf-8'))
    # Re-running replaces only this procedure's machine tracks.
    rows = [r for r in document['samples']
            if r.get('source_kind') != 'adjacent_source_flow']
    anchors = [r for r in rows if r.get('source_kind') == 'assistant_native_source_review'
               and r.get('role') == 'fit']
    ledger = dict(np.load(regions))
    pts = ledger['source_pts']
    if native_neighbours is not None and native_neighbours < 1:
        raise ValueError('A positive native source neighbourhood is required')
    radius = 16 if native_neighbours is None else native_neighbours
    media = json.loads(registration.read_text(encoding='utf-8'))['primary_source']
    reader = SourceVideo(media['path'], np.array(document['time_base']))
    generated, checks = [], []
    try:
        for stamp, foot in sorted({(r['pts'], r['foot']) for r in anchors}):
            group = [r for r in anchors if r['pts'] == stamp and r['foot'] == foot]
            centre = int(np.searchsorted(pts, stamp))
            np.testing.assert_equal(pts[centre], stamp)
            # Short material tracks avoid extending a visible silhouette end
            # through a change of view or an occlusion. They are correlated
            # cues, not additional independent source measurements.
            nearby = np.arange(max(0, centre-radius), min(len(pts), centre+radius+1))
            nearby = nearby[ledger['shot_ids'][nearby] == ledger['shot_ids'][centre]]
            grays = {i: cv2.cvtColor(reader.read(int(pts[i])), cv2.COLOR_RGB2GRAY)
                     for i in nearby}
            initial = np.asarray([r['uv'] for r in group], np.float32)
            coupled=endpoint_pairs(group)
            per_point = [[] for _ in group]
            for direction in (-1, 1):
                previous, xy = centre, initial.copy()
                alive = np.ones(len(group), bool)
                cumulative = np.zeros(len(group))
                ordered = nearby[nearby < centre][::-1] if direction < 0 else nearby[nearby > centre]
                for i in ordered:
                    elapsed = abs(int(pts[i])-int(stamp))*document['time_base'][0]/document['time_base'][1]
                    limit_key = 'flow_before_seconds' if direction < 0 else 'flow_after_seconds'
                    if native_neighbours is None:
                        alive &= np.array([elapsed <= r.get(limit_key, .32)+1.e-9 for r in group])
                    if not alive.any():
                        break
                    bounds = crop_bounds(xy, grays[previous].shape, 96)
                    prediction, fb, photo = tracks(grays[previous], grays[i], xy, bounds)
                    alive &= np.isfinite(prediction).all(1) & (fb < 1.) & (photo < 22.)
                    for j,row in enumerate(group):
                        contours=[ledger['region'][i,f] for f in row.get('foot_candidates',[foot])
                                  if ledger['visible'][i,f] and ledger['role'][i,f]==1]
                        contours=[q for q in contours if np.isfinite(q).all()]
                        if contours:
                            alive[j] &= np.isfinite(prediction[j]).all() and any(
                                cv2.pointPolygonTest(q.astype(np.float32),tuple(map(float,prediction[j])),True)>=-12.
                                for q in contours)
                    cumulative += np.where(alive, fb, 0.)
                    if native_neighbours is not None:
                        # A short track must also return to its ORIGINAL anchor.
                        # Adjacent checks alone can accumulate a drift into ice.
                        anchor_bounds=crop_bounds(np.r_[prediction[np.isfinite(prediction).all(1)],initial],grays[i].shape,96)
                        safe_prediction=np.where(np.isfinite(prediction),prediction,xy)
                        returned,anchor_fb,anchor_photo=tracks(grays[i],grays[centre],safe_prediction,anchor_bounds)
                        anchor_error=np.linalg.norm(returned-initial,axis=1)
                        alive &= (anchor_error<1.5)&(anchor_fb<1.)&(anchor_photo<22.)
                    for pair in coupled:
                        alive[pair]=alive[pair].all()
                    # Native neighbours retain every source exposure, but the
                    # whole correlated track still receives one anchor weight.
                    if native_neighbours is not None or abs(int(i)-centre) % 4 == 0:
                        for j in np.flatnonzero(alive):
                            row = dict(group[j], pts=int(pts[i]), uv=prediction[j].astype(float).tolist(),
                                       source_kind='adjacent_source_flow',
                                       source_flow_fb_px=float(fb[j]), source_flow_photo=float(photo[j]),
                                       normalization_px=float(group[j]['normalization_px']+cumulative[j]))
                            identity=group[j].get('id',f'{stamp}:{foot}:{j}')
                            transport_flow_region(row,group[j])
                            row.update(id=f'{identity}:flow:{int(pts[i])}',source_anchor_pts=int(stamp),
                                       source_anchor_id=identity)
                            if 'association_group' in row:
                                row['association_track']=group[j].get('association_track',group[j]['association_group'])
                                row['association_group']=f'{row["association_group"]}:flow:{int(pts[i])}'
                            if native_neighbours is not None:
                                row['source_flow_anchor_return_px']=float(anchor_error[j])
                            per_point[j].append(row)
                    if not alive.any():
                        break
                    xy = np.where(np.isfinite(prediction), prediction, xy)
                    previous = int(i)
            for original, propagated in zip(group, per_point, strict=True):
                for row in propagated:
                    row['weight'] = original['weight']/len(propagated)
                generated.extend(propagated)
            checks.append(dict(anchor_pts=stamp, foot=foot, retained=[len(v) for v in per_point],
                               features=[r['feature'] for r in group]))
            print(json.dumps(checks[-1]), flush=True)
    finally:
        reader.close()
    document['samples'] = rows+generated
    document['flow_policy'] = dict(method='adjacent original-pixel LK, forward/backward check',
        scope=('within 0.32 s of each reviewed source point, same shot; per-anchor visibility limits may be shorter'
               if native_neighbours is None else f'Every original exposure within {native_neighbours} source frames of each anchor, same shot; explicit replacement of previous zero-length flow limits'),
        native_neighbour_frames=native_neighbours,
        paired_material_endpoints_stop_together=True,
        observed_source_pixels_changed=False, used_model_projection=False,
        total_weight='one additional anchor weight per correlated propagated track',
        checks=checks)
    source.write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(reviewed_points=len(anchors), propagated_points=len(generated),
                          source_bytes=source.stat().st_size)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registration', 'regions'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--native-neighbours',type=int)
    p.add_argument('--verify-tracks',action='store_true')
    p.add_argument('--review-batches',nargs='+')
    a = p.parse_args()
    if a.verify_tracks:verify_tracks(a.source,a.registration,a.regions,a.review_batches)
    else:propagate(a.source, a.registration, a.regions,a.native_neighbours)
