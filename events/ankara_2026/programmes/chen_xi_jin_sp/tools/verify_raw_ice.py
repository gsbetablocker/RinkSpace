"""Recheck source material colour and visibility independently of fit error."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256

import cv2
import numpy as np

from common import (HEIGHT, OUTPUT, WIDTH, frozen_scene, observation_packet, save, scope,
                    source_frames, visibility_packet, write)
from ice_evidence import MIDPOINT, assert_ice_evidence, ice_roles
from physical_reads import CHANNELS
from precision_observations import exclude
from review import restore_source_graphics
from stability_observations import read_arrays


def read_pixels(raw, xy):
    points = xy.reshape(-1, xy.shape[-2], 2).astype(np.float32)
    return cv2.remap(raw, points[:, :, 0], points[:, :, 1], cv2.INTER_LINEAR).reshape((*xy.shape[:-1], 3))


def red_spot_claims(raw, pairs):
    """Neutral lane paint and a chromatic hockey dot cannot be the same track."""
    a, b = np.moveaxis(pairs, 1, 0)
    d = b - a
    w = np.linalg.norm(d, axis=1)
    tangent = np.c_[-d[:,1], d[:,0]] / w[:, None]
    ext = np.clip(3.7 / w, .23, .55)
    parameter = np.c_[-1.55 * ext, -ext, -0.57 * ext,
                     np.full((len(w), 3), [.32, .5, .68]),
                     1. + .57 * ext, 1. + ext, 1. + 1.55 * ext]
    xy = a[:,None] + parameter[:,:,None] * d[:,None]
    short = xy[:,None,:,:] + np.array([-3., 0., 3.])[None,:,None,None] * tangent[:,None,None,:]
    rgb = read_pixels(raw, short)
    delta = np.asarray(rgb, np.float32) @ np.asarray(CHANNELS['red'], np.float32)
    levels = np.median(delta, axis=1)
    center = np.median(levels[:,3:6], axis=1)
    both = np.minimum(center - np.median(levels[:,:3], axis=1),center - np.median(levels[:,6:], axis=1))
    return (center > 4.4) & (both > 2.8)


def visible_inside(mask, points):
    x, y = np.moveaxis(np.rint(points).astype(np.int32), -1, 0)
    valid = (x >= 2)&(x < WIDTH-2)&(y >= 2)&(y < HEIGHT-2)
    blocked = ~valid
    blocked[valid] |= mask[y[valid], x[valid]] != 0
    return ~blocked


def check_shot(record, shot, source, out, roles, visibility):
    sid = shot['id']
    packet = observation_packet(source,sid)
    arrays = read_arrays(packet)
    good = arrays['eligible'].copy()
    assert_ice_evidence(arrays,good,roles)
    digests = dict(zip(arrays['source_frame_pts'],arrays['source_frame_BGR_sha256'],strict=True))
    with np.load(visibility_packet(visibility,sid)) as v:
        pts, packed, valid = v['pts'], v['packed'], v['valid']
        assert v['source_sha256'].item() == record['source']['sha256']
    np.testing.assert_array_equal(pts[arrays['idx']],arrays['pts'])
    lane = np.char.startswith(arrays['family'],'lane')&(arrays['ice_measurement_role']==MIDPOINT)
    material_is_width = arrays['ice_measurement_role']==MIDPOINT
    rows = np.flatnonzero(good)
    order = rows[np.argsort(arrays['pts'][rows],kind='stable')]
    blocks = np.split(order,np.flatnonzero(np.diff(arrays['pts'][order]))+1)
    by_pts = {int(arrays['pts'][x[0]]):x for x in blocks if len(x)}
    occluded = np.zeros(len(good),bool)
    wrong_tone = np.zeros(len(good),bool)
    for p, fram in source_frames(record['source']['registered_path'],np.asarray(sorted(by_pts),np.int64)):
        raw = fram.to_ndarray(format='bgr24')
        if sha256(raw.tobytes()).hexdigest()!=digests[p]:
            raise ValueError(f'Decoded source frame differs at {p}')
        i = int(np.searchsorted(pts,p))
        j = by_pts[p]
        person = np.unpackbits(packed[i],axis=1,count=WIDTH) if valid[i] else None
        if person is not None:
            person = cv2.dilate(person,np.ones((3,3),np.uint8))
        m = restore_source_graphics(raw,p,person)
        occluded[j] = ~visible_inside(m,arrays['uv'][j])
        width = j[material_is_width[j] & ~occluded[j]]
        pair = arrays['source_bracket'][width]
        all_flanks = pair[:,:1] + np.array([0.,.24,.5,.76,1.])[None,:,None] * (pair[:,1:2]-pair[:,:1])
        occluded[width] = ~visible_inside(m,all_flanks).all(axis=1)
        k = j[lane[j]&~occluded[j]]
        if len(k):
            wrong_tone[k] = red_spot_claims(raw, arrays['source_bracket'][k])
    exclude(arrays,occluded,'recorded_athlete_or_broadcast_overlay_covers_source_measurement')
    exclude(arrays,wrong_tone,'source_chromatic_red_paint_is_not_the_claimed_lane_stroke')
    # The red arc is clearly visible late in S13. Reserve source station
    # bands without omitting the whole fast pan from absolute fitting.
    repartition=np.zeros(len(good),bool)
    if sid == 's13':
        t=arrays['pts']/12800.
        repartition=arrays['eligible']&(arrays['family']=='A4')&(t>=125.72)&(t<=126.56)
        phase=(np.floor(28*arrays['station'][repartition]).astype(np.int32)%4)
        arrays['fold'][repartition]=np.where(phase==2,3,0)
    changed=occluded|wrong_tone|repartition
    if changed.any():
        packet=out/(sid+'.npz')
        save(packet,**arrays)
    report=dict(masked_source_rows=int(occluded.sum()),wrong_colour_lane_rows=int(wrong_tone.sum()),
                reviewed_arc_rows=int(repartition.sum()),retained=int(arrays['eligible'].sum()),
                chromatic_false_lanes=dict(Counter(arrays['family'][wrong_tone])),
                source_clock_verified=len(by_pts),packet=str(packet))
    print(sid,report['masked_source_rows'],report['wrong_colour_lane_rows'],report['reviewed_arc_rows'],flush=True)
    return report


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--observations',required=True)
    a.add_argument('--out',required=True)
    a.add_argument('--visibility',default='visibility_fast_checked')
    args=a.parse_args()
    cv2.setNumThreads(2)
    r=scope()
    roles=ice_roles(frozen_scene(r,verify=True))
    source,out=OUTPUT/args.observations,OUTPUT/args.out
    out.mkdir(exist_ok=False)
    reports={shot['id']:check_shot(r,shot,source,out,roles,OUTPUT/args.visibility) for shot in r['shots']}
    write(out/'shot_packets.json',{k:v['packet'] for k,v in reports.items()})
    write(out/'qualification.json',reports)


if __name__=='__main__':
    main()
