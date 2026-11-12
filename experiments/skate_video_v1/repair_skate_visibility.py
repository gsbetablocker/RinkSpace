"""Check complete skate visibility through overlaps and reflection changes."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

from evidence import segments
from follow_whole_skates import (
    PRIOR_UNCERTAIN, PRIOR_WHOLE, SOURCE_PROMPT, VIDEO_RECOVERY, VIDEO_WHOLE,
    WholeShape, boundary_support, extra_part, review_frame, write_navigation,
)
from inputs import PROGRAMME, SIZE, WEIGHTS, Inputs, atomic_json, atomic_npz, distribute, emit, geometry, read
from model import SkateModel
from recover import advance
from relink_whole_skates import RECOVER, REJECT, REPLACE, Shot, diagonal, distance, report
from render_review import render
from track import Navigation, VideoMasks, iou, mask_overlap, measure
from whole_skate_review import filled


RELINK, EXTENT, OCCLUDED = 1, 2, 3


def region_area(data):
    area = np.zeros(data["status"].shape, np.float32)
    for j, f in zip(*np.where(data["status"] > 0)):
        area[j, f] = cv2.contourArea(data["region"][j, f])
    return area


def exclusive_changes(inputs, data):
    """Follow the observation even when its left/right array slot changes."""
    present = data["status"] > 0
    alone = present.sum(axis=1) == 1
    result = []
    for j in np.flatnonzero(alone):
        f = int(present[j].argmax())
        for k in range(j+1, min(len(present), j+4)):
            if inputs.shots[j] != inputs.shots[k] or present[k].all():
                break
            if not alone[k]:
                continue
            g = int(present[k].argmax())
            if f != g:
                p, q = data["region"][j, f], data["region"][k, g]
                size = max(12., .5*(diagonal(p)+diagonal(q)))
                if np.linalg.norm(p.mean(0)-q.mean(0)) < .85*size:
                    result.append((int(j), k, f, g))
            break
    return result


def close_occlusions(inputs, nav, data, area):
    """Find single detections bracketed by two separate observed skates."""
    present = data["status"] > 0
    separation = np.linalg.norm(nav.ankle[:, 0]-nav.ankle[:, 1], axis=1)/np.maximum(nav.size.mean(axis=1), 12.)
    events = []
    for a, b in inputs.ranges():
        for f in range(2):
            for x, y in segments(present[a:b, f] & ~present[a:b, 1-f]):
                x, y = x+a, y+a
                if x == a or y == b-1 or not (present[x-1].all() and present[y+1].all()):
                    continue
                if np.median(separation[x:y+1]) > .72:
                    continue
                normal = .5*(area[x-1, f]+area[y+1, f])
                if (area[x:y+1, f] > 1.12*normal).any():
                    events.append((int(x), int(y)))
    return events


def merge_windows(inputs, spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a-merged[-1][1] <= 5 and inputs.shots[a] == inputs.shots[merged[-1][0]]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def paired_anchor(inputs, nav, data, j):
    if not (data["status"][j] == 1).all() or (data["extent"][j] == PRIOR_UNCERTAIN).any():
        return False
    p, q = data["region"][j]
    if mask_overlap(p, q)[1] > .28:
        return False
    for f in range(2):
        poly = data["region"][j, f]
        if not .59*nav.size[j, f] < diagonal(poly) < 1.70*nav.size[j, f]:
            return False
        if nav.direct[j, f, 1] and distance(poly, nav.ankle[j, f]) > .30*nav.size[j, f]:
            return False
    return True


class Clip:
    def __init__(self, inputs, nav, before, a, b):
        self.a, self.b = a, b
        positions = np.concatenate((before["region"][a:b+1].reshape(-1,2), nav.ankle[a:b+1].reshape(-1,2)))
        positions = positions[np.isfinite(positions).all(axis=1)]
        low, high = positions.min(0), positions.max(0)
        width = int(np.ceil(max(high-low)+max(70., .70*nav.size[a:b+1].max())))
        if width <= 420:
            side = max(165, width)
            centres = np.broadcast_to(.5*(low+high), (b-a+1,2))
        else:
            side = int(np.clip(3.25*np.median(nav.size[a:b+1])+70., 240, 490))
            centres = nav.ankle[a:b+1].mean(axis=1).copy()
            centres += nav.centre[a:b+1].mean(axis=1)*.33-centres*.33
        start = np.clip(np.rint(centres-side/2).astype(int),0,np.asarray(SIZE)-side)
        self.bounds = np.hstack((start, start+side))
        self.chips = [np.ascontiguousarray(frame[y:y+side, x:x+side])
                      for (pts, frame), (x,y) in zip(inputs.frames(a,b+1), start, strict=True)]
        self.nav = nav

    def at(self, j):
        return self.chips[j-self.a], self.bounds[j-self.a]

    def warp(self, p, start, stop):
        direction = 1 if stop > start else -1
        for j in range(start, stop, direction):
            p = advance(self.chips, self.bounds, j-self.a, j+direction-self.a, p)
            if p is None:
                break
        return p

    def track(self, model, data, anchor, stop):
        model.reset()
        direction = 1 if stop > anchor else -1
        path = {}
        for n, j in enumerate(range(anchor, stop+direction, direction)):
            chip, bb = self.at(j)
            masks = {f:filled(chip.shape[:2], data["region"][anchor, f]-bb[:2])
                     for f in range(2)} if n == 0 else {}
            found = model.step(chip, n, {}, seed_masks=masks)
            pair = {}
            for f, output in found.items():
                item = measure(*output, bb, max(18., self.nav.size[j, f]))
                if item is not None:
                    pair[f] = data["region"][anchor,f] if n == 0 else item[0]
            path[j] = pair
        return path


def write_outline(data, inputs, j, f, outline, seed, support, kind, *, partial=False, hidden=False):
    present = bool(data["status"][j, f])
    data["visibility_revision"][j, f] = kind
    data["region"][j, f] = np.nan if outline is None else outline
    data["status"][j, f] = int(outline is not None)
    data["extent"][j, f] = (PRIOR_UNCERTAIN if outline is None or partial else VIDEO_WHOLE if present else VIDEO_RECOVERY)
    data["tracking_anchor_pts"][j, f] = inputs.pts[seed] if outline is not None else -1
    data["mask_agreement"][j, f] = support if outline is not None else np.nan
    data["revision"][j, f] = REJECT if outline is None else REPLACE if present else RECOVER
    data["partial_occlusion"][j, f] = partial
    data["skate_occluded"][j, f] = hidden


def agreed_pair(clip, shape, j, left, right, anchors):
    accepted = {}
    chip, bounds = clip.at(j)
    for f in range(2):
        p, q = left[j].get(f), right[j].get(f)
        if p is None or q is None:
            continue
        support = iou(p, q)
        if support < .61:
            continue
        size = clip.nav.size[j, f]
        scored = [(boundary_support(chip, bounds, v), v, k) for v,k in zip((p,q),anchors)]
        boundary, outline, seed = max(scored, key=lambda z:z[0])
        length = diagonal(outline)
        partial = length < .61*size
        if not .34*size < length < 1.85*size or boundary < .38:
            continue
        if not partial and shape.error(outline,f) > .235:
            continue
        if clip.nav.direct[j,f,1] and distance(outline,clip.nav.ankle[j,f]) > (.55 if partial else .44)*size:
            continue
        motion = []
        for bank in (left,right):
            for k in (j-1,j+1):
                if k not in bank or f not in bank[k]:
                    continue
                prediction = clip.warp(bank[k][f],k,j)
                if prediction is not None:
                    motion.append(iou(outline,prediction))
        if max(motion,default=0.) < .49 and support < .81:
            continue
        accepted[f] = (outline, seed, support, partial)
    return accepted


def repair_overlap(inputs, nav, model, shape, before, after, span, flips):
    x, y = span
    possible_left = [j for j in range(x-1, max(-1, x-22), -1)
                     if inputs.shots[j] == inputs.shots[x] and paired_anchor(inputs, nav, before, j)]
    possible_right = [j for j in range(y+1, min(len(inputs.pts), y+22))
                      if inputs.shots[j] == inputs.shots[y] and paired_anchor(inputs, nav, before, j)]
    if not possible_left or not possible_right:
        return False
    a,b=possible_left[0],possible_right[0]
    clip = Clip(inputs, nav, before, a, b)
    left, right = clip.track(model,before,a,b), clip.track(model,before,b,a)
    # Some episodes contain a couple of separate observations between two
    # occlusions. Those already resolved observations stay at their old pixels.
    flip_members = {j for k,l,_,_ in flips if x <= k <= y for j in (k,l)}
    for j in range(x,y+1):
        if (before["extent"][j] == SOURCE_PROMPT).any() or after["visibility_revision"][j].any():
            continue
        present = before["status"][j] > 0
        raw = before["region"][j]
        if present.all() and mask_overlap(raw[0],raw[1])[1] <= .48:
            continue
        q = agreed_pair(clip, shape, j, left, right, (a,b))
        if len(q)==2 and mask_overlap(q[0][0],q[1][0])[1] <= .32:
            chip, bb = clip.at(j)
            combined = filled(chip.shape[:2], q[0][0]-bb[:2]) | filled(chip.shape[:2], q[1][0]-bb[:2])
            old = np.zeros_like(combined)
            for f in np.flatnonzero(present):
                old |= filled(chip.shape[:2],raw[f]-bb[:2])
            if old.any() and np.count_nonzero(old&combined)/np.count_nonzero(old) < .73:
                continue
            changes = {}
            for f in range(2):
                outline, seed, support, partial = q[f]
                if present[f] and iou(raw[f],outline) > .81:
                    # Preserve a correct outline unless it encloses visible
                    # pixels of the other independently followed skate.
                    if mask_overlap(raw[f],q[1-f][0])[1] < .41:
                        continue
                if present[f] and before["extent"][j,f] == PRIOR_WHOLE and support < .71:
                    continue
                changes[f] = (outline, seed, support, partial)
            final = [changes[f][0] if f in changes else raw[f] for f in range(2)]
            if all(np.isfinite(p).all() for p in final) and mask_overlap(*final)[1] > .40:
                continue
            for f,(outline,seed,support,partial) in changes.items():
                write_outline(after,inputs,j,f,outline,seed,support,RELINK,partial=partial)
        elif len(q)==1 and present.sum()==1 and j in flip_members:
            f, (outline, seed, support, partial) = next(iter(q.items()))
            old = int(present.argmax())
            if f == old or support < .76 or iou(raw[old],outline) < .61:
                continue
            other = [bank[j].get(old) for bank in (left,right)]
            if any(v is not None and mask_overlap(v,outline)[1] < .42 for v in other):
                continue
            # The same observed outline crossed array slots. Preserve its
            # pixels when the two independent tracks identify its owner.
            write_outline(after,inputs,j,f,raw[old],seed,support,RELINK,partial=partial)
            write_outline(after,inputs,j,old,None,seed,0.,OCCLUDED)
    return True


def complete_outline(shape, chip, bb, j, f, raw, proposal, other, nav, *, reviewed=False):
    if other is not None and mask_overlap(proposal,other)[1] > (.32 if reviewed else .22):
        return None
    if boundary_support(chip,bb,proposal,raw) < (.27 if reviewed else .42):
        return None
    shank = nav.joints[j,f] if nav.direct[j,f].all() else None
    merged = extra_part(raw,proposal,chip,bb,shank,nav.size[j,f])
    if merged is None or shape.error(merged,f) > .239:
        return None
    if other is not None and mask_overlap(merged,other)[1] > (.32 if reviewed else .23):
        return None
    return merged


def reviewed_crossings(inputs, nav, model, shape, before, after):
    record = read(PROGRAMME/"whole_skate_overlap_review_1112.json")
    if record["source_video_sha256"] != inputs.media_sha or tuple(record["time_base"]) != tuple(inputs.tb):
        raise ValueError("Reviewed overlap is bound to a different video")
    protected = np.zeros(len(inputs.pts),bool)
    for case in record["visibility_review"]:
        a,b,c,d=(inputs.index[case[key]] for key in ("before_pts","after_pts","next_whole_pts","later_whole_pts"))
        f=case["rear_foot"]
        g=1-f
        clip = Clip(inputs,nav,before,a,d)
        entering,leaving = clip.track(model,before,a,b),clip.track(model,before,b,a)
        after_forward,from_next,from_later = (clip.track(model,before,x,y) for x,y in ((b,d),(c,a),(d,b)))
        hidden = set(case["rear_hidden_pts"])
        for pts in case["crossing_pts"]:
            j=inputs.index[pts]
            near, far = entering[j], leaving[j]
            fore_support = iou(near[g],far[g]) if g in near and g in far else 0.
            third = from_next[j].get(g)
            extra_support = iou(third,far[g]) if third is not None and g in far else 0.
            if not (fore_support >= .67 or (fore_support >= .60 and extra_support >= .75)):
                raise ValueError(f"Failed to follow foreground skate at {pts} (agreement {fore_support:.3f})")
            foreground = near[g] if j-a <= b-j else far[g]
            write_outline(after,inputs,j,g,foreground,a if j-a<=b-j else b,iou(near[g],far[g]),RELINK)
            if pts in hidden:
                write_outline(after,inputs,j,f,None,b,0.,OCCLUDED,hidden=True)
                continue
            if f not in near or f not in far or mask_overlap(near[f],foreground)[1] > .36:
                raise ValueError(f"Failed to separate the two visible skates at {pts}")
            write_outline(after,inputs,j,f,near[f],a,iou(near[f],far[f]),RELINK,partial=True)
        for pts in case["missing_extent_pts"]:
            j=inputs.index[pts]
            chip, bb=clip.at(j)
            if j < b:
                paths = ((leaving,b),(entering,a))
            else:
                paths = ((after_forward,b),(from_next,c),(from_later,d))
            choices = [(bank[j][f],seed) for bank,seed in paths if j in bank and f in bank[j]]
            if len(choices)<2:
                continue
            old, peer = before["region"][j,f], before["region"][j,g]
            ranked = []
            for p,seed in choices:
                support = max(iou(p,q) for q,src in choices if seed != src)
                if support < .68:
                    continue
                q=complete_outline(shape,chip,bb,j,f,old,p,peer,nav,reviewed=True)
                if q is not None:
                    ranked.append((boundary_support(chip,bb,q,old)+support,q,seed,support))
            if ranked:
                _,q,seed,support=max(ranked,key=lambda item:item[0])
                write_outline(after,inputs,j,f,q,seed,support,EXTENT)
        protected[a:d+1] = True
        details = [(int(inputs.pts[j]),int(f),int(after["visibility_revision"][j,f]))
                   for j,f in zip(*np.where(after["visibility_revision"]>0)) if a<=j<=d]
        emit("reviewed_crossing", source_pts=int(inputs.pts[a]), corrections=details)
    return protected


def flicker_neighbors(inputs, data, area, protected):
    """Find short extent losses while both ankles keep their established tracks."""
    status, extent, region = data["status"], data["extent"], data["region"]
    centre = region.mean(axis=2)
    scale = diagonal(region)
    result = {}
    good = (status == 1) & (extent > 0) & ~data["partial_occlusion"]
    for j,f in zip(*np.where((status == 1) & (extent != SOURCE_PROMPT) & ~data["partial_occlusion"])):
        if j < 4 or j+4 >= len(status) or protected[j] or data["visibility_revision"][j,f]:
            continue
        sides = []
        for ix in (np.arange(j-4,j),np.arange(j+1,j+5)):
            valid=(inputs.shots[ix]==inputs.shots[j]) & good[ix,f] & (area[j,f]<.91*area[ix,f])
            valid &= (scale[ix,f]>.74*scale[j,f]) & (scale[ix,f]<1.35*scale[j,f])
            valid &= np.linalg.norm(centre[ix,f]-centre[j,f],axis=-1)<.57*scale[j,f]
            ix=ix[valid]
            if not len(ix):
                break
            order=np.argsort(abs(ix-j))
            sides.append(ix[order][:2].tolist())
        if len(sides)==2:
            result[(int(j),int(f))]=sides
    return result


def restore_flicker(inputs, nav, shape, source, data, anchors):
    count=0
    for a,b in inputs.ranges():
        target=[(j,f,v) for (j,f),v in anchors.items() if a<=j<b]
        if not target:
            continue
        start=time.perf_counter()
        packet={key:source[key][a:b] for key in ("region","status","extent","tracking_anchor_pts","mask_agreement")}
        shot=Shot(inputs,nav,shape,int(a),int(b),packet)
        for j,f,(first,last) in target:
            k=j-a
            chip,bb=shot.chips[f][k],shot.bounds[k,f]
            p=source["region"][j,f]
            other=data["region"][j,1-f] if data["status"][j,1-f] else None
            if other is not None and mask_overlap(p,other)[1] > .26:
                continue
            banks=[]
            for options in (first,last):
                paths=[]
                for anchor in options:
                    q=source["region"][anchor,f]
                    step=1 if j>anchor else -1
                    for at in range(anchor,j,step):
                        q=shot.warp(f,at-a,at+step-a,q)
                        if q is None:
                            break
                    if q is not None:
                        paths.append((q,anchor))
                banks.append(paths)
            choices=[]
            for left,sa in banks[0]:
                for right,sb in banks[1]:
                    agreement=iou(left,right)
                    if agreement < .77:
                        continue
                    joint=filled(chip.shape[:2],left-bb[:2]) & filled(chip.shape[:2],right-bb[:2])
                    old=filled(chip.shape[:2],p-bb[:2])
                    if np.count_nonzero(joint&old) < .90*np.count_nonzero(old):
                        continue
                    contours,_=cv2.findContours(joint,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    consensus=distribute(max(contours,key=cv2.contourArea)[:,0]+bb[:2],len(p),closed=True)
                    q=complete_outline(shape,chip,bb,j,f,p,consensus,other,nav)
                    if q is not None:
                        choices.append((agreement,q,sa if j-sa<=sb-j else sb))
            if choices:
                support,q,seed=max(choices,key=lambda item:item[0])
                write_outline(data,inputs,j,f,q,seed,support,EXTENT)
                count+=1
        emit("whole_boundary_review", shot=str(inputs.shots[a]), candidates=len(target),
             cumulative_repaired=count, seconds=round(time.perf_counter()-start,2))
    return count


def matching_pixels(clip, p, q, a, b):
    """Return independent overlap scores when native round trips survive."""
    p1, q0 = clip.warp(p,a,b), clip.warp(q,b,a)
    return [iou(x,y) for x,y in ((p1,q),(q0,p)) if x is not None]


def same_observation(scores):
    return (len(scores) == 2 and max(scores) >= .66 and min(scores) >= .49
            or len(scores) == 1 and scores[0] >= .82)


def ownership_witness(clip, inputs, nav, data, edge, step, f):
    """Connect one end of a visible run to a separated two-foot observation."""
    p = data["region"][edge,f]
    for j in range(edge+step, edge+5*step, step):
        if not clip.a <= j <= clip.b or inputs.shots[j] != inputs.shots[edge]:
            break
        if not paired_anchor(inputs,nav,data,j):
            continue
        q = data["region"][j]
        scores = [max(matching_pixels(clip,p,r,edge,j),default=0.) for r in q]
        owner = int(np.argmax(scores))
        if scores[owner] >= .59 and scores[owner]-scores[1-owner] >= .23:
            return owner,j,scores[owner]
    return None


def identity_review(inputs, nav, before, after):
    source = {k:v.copy() for k,v in after.items()}
    visible = source["status"] > 0
    alone = visible.sum(axis=1) == 1
    owners = visible.argmax(axis=1)
    unknown = after.setdefault("identity_unresolved",np.zeros_like(visible))
    episodes=[]
    handled=set()
    edges=exclusive_changes(inputs,source)
    for j,k,f,g in edges:
        if j in handled or k in handled:
            continue
        if (source["extent"][j] == SOURCE_PROMPT).any() or (source["extent"][k] == SOURCE_PROMPT).any():
            continue
        a,b=j,k
        while a>max(0,j-7) and inputs.shots[a-1]==inputs.shots[j] and alone[a-1]:
            a-=1
        while b<min(len(visible)-1,k+7) and inputs.shots[b+1]==inputs.shots[k] and alone[b+1]:
            b+=1
        lo,hi=a,b
        while lo>max(0,a-4) and inputs.shots[lo-1]==inputs.shots[j]:
            lo-=1
        while hi<min(len(visible)-1,b+4) and inputs.shots[hi+1]==inputs.shots[k]:
            hi+=1
        clip=Clip(inputs,nav,source,lo,hi)
        p,q=source["region"][j,f],source["region"][k,g]
        cross=matching_pixels(clip,p,q,j,k)
        if not same_observation(cross):
            continue
        # Carry one visible observation through the overlap. Its physical
        # label is only established by separated views at the ends.
        start,stop=j,k
        for step in (-1,1):
            t=start if step<0 else stop
            limit=a if step<0 else b
            while t!=limit:
                m=t+step
                if not same_observation(matching_pixels(clip,source["region"][t,owners[t]],
                                                        source["region"][m,owners[m]],t,m)):
                    break
                t=m
            if step<0:
                start=t
            else:
                stop=t
        path=[t for t in range(start,stop+1) if alone[t]]
        if any((source["extent"][t] == SOURCE_PROMPT).any() or t in handled for t in path):
            continue
        first=ownership_witness(clip,inputs,nav,source,start,-1,owners[start])
        last=ownership_witness(clip,inputs,nav,source,stop,1,owners[stop])
        proof = [first,last] if first is not None and last is not None and first[0] == last[0] else []
        resolved = bool(proof)
        if resolved:
            owner, anchor, agreement = min(proof,key=lambda v:abs(v[1]-.5*(start+stop)))
            for t in path:
                old=int(owners[t])
                if old==owner:
                    continue
                p=source["region"][t,old]
                if nav.direct[t,:,1].all() and distance(p,nav.ankle[t,owner]) > .50*nav.size[t,owner]:
                    resolved=False
                    break
        for t in path:
            old=int(owners[t])
            if resolved and old!=owner:
                write_outline(after,inputs,t,owner,source["region"][t,old],anchor,agreement,RELINK,
                              partial=bool(source["partial_occlusion"][t,old]))
                after["extent"][t,owner]=source["extent"][t,old]
                write_outline(after,inputs,t,old,None,anchor,0.,OCCLUDED)
            if not resolved:
                unknown[t,old]=True
            handled.add(t)
        episodes.append(dict(first_pts=int(inputs.pts[start]),last_pts=int(inputs.pts[stop]),
                             original_switch_pts=[int(inputs.pts[j]),int(inputs.pts[k])],
                             native_overlap=cross, owner=int(owner) if resolved else None,
                             observed_frames=len(path)))
    return episodes


def publish(inputs, before, result, args, window_count, run, old_edges, checked):
    np.testing.assert_array_equal(result["status"]>0,np.isfinite(result["region"]).all(axis=(2,3)))
    changed=result["visibility_revision"]>0
    np.testing.assert_array_equal(result["region"][~changed],before["region"][~changed])
    np.testing.assert_array_equal(result["status"][~changed],before["status"][~changed])
    direct=before["extent"]==SOURCE_PROMPT
    np.testing.assert_array_equal(result["region"][direct],before["region"][direct])
    atomic_npz(args.output/"whole_follow_evidence.npz",**result)
    write_navigation(inputs,args.navigation,args.output,result)
    path=args.output/"navigation.json"
    nav=read(path)
    previous=read(args.history/"navigation.json")
    existing={row["id"] for row in nav["events"]}
    nav["events"].extend(row for row in previous["events"] if row["id"].startswith(("reviewed-","jump-motion-"))
                         and row["id"] not in existing)
    nav["events"].append(dict(id="early_skate_crossing", label="29.600s 双脚交叠和亮刃连续性复核",
                             start_pts=378880,end_pts=381696,worst_pts=379392,exposure_count=12))
    remaining=exclusive_changes(inputs,result)
    for start,stop,f,g in remaining:
        nav["events"].append(dict(id=f"exclusive-skate-{start}-{stop}",label=f"{inputs.times[start]:.3f}s 单圈换脚待复核",
                                  start_pts=int(inputs.pts[start]),end_pts=int(inputs.pts[stop]),
                                  worst_pts=int(inputs.pts[stop]),exposure_count=stop-start+1))
    nav["events"].sort(key=lambda row:(row["start_pts"],row["end_pts"]))
    atomic_json(path,nav)
    stats=report(inputs,before,result)
    stats.update(
        pair_windows=window_count, paired_video_windows=run, native_boundary_candidates=checked,
        exclusive_switch_candidates_before=len(old_edges),exclusive_switch_candidates_after=len(remaining),
        visibility_updates=dict(zip(("unchanged","relinked","extent","withdrawn"),map(int,np.bincount(
            result["visibility_revision"].ravel(),minlength=4)))),
        older_complete_labels_revised=int(((before["extent"]==PRIOR_WHOLE)&changed).sum()),
        newly_hidden=int(result["skate_occluded"].sum()),
        revised_source_foot_pts=[[int(inputs.pts[j]),int(f),int(result["visibility_revision"][j,f])]
                                 for j,f in zip(*np.where(changed))],
        exclusive_candidates_remaining=[dict(start_pts=int(inputs.pts[a]),end_pts=int(inputs.pts[b]),
                                              source_foot=f,next_foot=g) for a,b,f,g in remaining],
    )
    atomic_json(args.output/"whole_visibility_checks.json",stats)
    emit("visibility_review_complete", **{k:v for k,v in stats.items() if k not in
         ("revised_source_foot_pts","exclusive_candidates_remaining")})


def publish_identities(inputs, nav, before, args, path):
    with np.load(path,allow_pickle=False) as f:
        result={k:f[k] for k in f.files}
    stats_path=args.output/"whole_visibility_checks.json"
    original=read(stats_path)
    episodes=identity_review(inputs,nav,before,result)
    publish(inputs,before,result,args,original["pair_windows"],original["paired_video_windows"],
            exclusive_changes(inputs,before),original["native_boundary_candidates"])
    stats=read(stats_path)
    stats["native_identity_episodes"]=episodes
    stats["unresolved_visible_identity"]=int(result["identity_unresolved"].sum())
    atomic_json(stats_path,stats)
    nav_path=args.output/"navigation.json"
    document=read(nav_path)
    for episode in episodes:
        start,stop=episode["first_pts"],episode["last_pts"]
        a,b=inputs.index[start],inputs.index[stop]
        label="双脚交叠身份未定" if episode["owner"] is None else "双脚交叠身份已接续"
        document["events"].append(dict(id=f"overlap-identity-{start}",label=f"{inputs.times[a]:.3f}s {label}",
                                       start_pts=start,end_pts=stop,worst_pts=start,exposure_count=b-a+1))
    document["events"].sort(key=lambda row:(row["start_pts"],row["end_pts"]))
    atomic_json(nav_path,document)
    emit("visible_identity_review", measured=len(episodes), resolved=sum(x["owner"] is not None for x in episodes),
         pending=sum(x["owner"] is None for x in episodes), frames=stats["unresolved_visible_identity"])


def main(args):
    cv2.setNumThreads(1)
    inputs=Inputs()
    path=args.output/"whole_follow_evidence.npz"
    if args.video:
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"],inputs.pts)
            np.testing.assert_array_equal(f["source_video_sha256"],inputs.media_sha)
            data={k:f[k] for k in ("region","status","extent","partial_occlusion","skate_occluded","identity_unresolved")
                  if k in f}
        frames=(review_frame(inputs,source,data,j) for j,(_,source) in enumerate(inputs.frames(0,len(inputs.pts))))
        render(inputs,args.output,"whole_skate_follow_review.mp4",frames,(2400,1080),refresh=args.refresh)
        return
    with np.load(args.history/"whole_follow_evidence.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["source_pts"],inputs.pts)
        np.testing.assert_array_equal(f["source_video_sha256"],inputs.media_sha)
        before={key:f[key] for key in f.files}
    nav=Navigation(inputs)
    if args.identities:
        publish_identities(inputs,nav,before,args,path)
        return
    result={key:value.copy() for key,value in before.items()}
    result["visibility_revision"]=np.zeros_like(before["status"])
    result["skate_occluded"]=np.zeros(before["status"].shape,bool)
    model=VideoMasks(WEIGHTS,image_size=768)
    shape=WholeShape(SkateModel(geometry(inputs)[0]))
    protected=reviewed_crossings(inputs,nav,model,shape,before,result)
    if args.focus:
        atomic_npz(args.output/"reviewed_crossing_only.npz",**result)
        return
    area=region_area(before)
    old_edges=exclusive_changes(inputs,before)
    spans=merge_windows(inputs,[(x,y) for x,y,_,_ in old_edges]+close_occlusions(inputs,nav,before,area))
    ran=0
    for x,y in spans:
        if protected[x:y+1].any():
            continue
        before_count=int(np.count_nonzero(result["visibility_revision"][x:y+1]))
        attempted=repair_overlap(inputs,nav,model,shape,before,result,(x,y),old_edges)
        if attempted:
            ran+=1
            after_count=int(np.count_nonzero(result["visibility_revision"][x:y+1]))
            emit("paired_visibility_review", shot=str(inputs.shots[x]), first_pts=int(inputs.pts[x]),
                 last_pts=int(inputs.pts[y]), repaired=after_count-before_count)
    frozen={k:v.copy() for k,v in result.items()}
    candidates=flicker_neighbors(inputs,frozen,region_area(frozen),protected)
    restore_flicker(inputs,nav,shape,frozen,result,candidates)
    publish(inputs,before,result,args,len(spans),ran,old_edges,len(candidates))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--navigation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--focus", action="store_true")
    parser.add_argument("--identities", action="store_true")
    args=parser.parse_args()
    if not args.video and (args.history is None or args.navigation is None):
        parser.error("--history and --navigation are required")
    main(args)
