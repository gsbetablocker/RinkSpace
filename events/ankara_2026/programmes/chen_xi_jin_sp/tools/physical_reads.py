"""Read signed rail, painted stripe, and Ad2 panel boundaries at native PTS."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
from time import monotonic

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import cv2
import numpy as np
from scipy.signal import find_peaks

from common import HEIGHT, JOINT, OUTPUT, WIDTH, frozen_scene, save, scope, source_frames, visibility_packet, write
from measure import BOTTOM, DT, FIRST, TOP, broadcast_overlay, fold_map, group_foreground, load_camera, motion_clock

AD2 = {
    "left_vertical": ("purple", "rise", 0),
    "right_vertical": ("purple", "fall", 0),
    "top_edge": ("purple", "rise", 1),
    "bottom_edge": ("purple", "fall", 1),
}
CHANNELS = {
    "red": [-.48, -.52, 1.],
    "blue": [1., -.42, -.58],
    "yellow": [-1., .39, .61],
    "purple": [1., -1.58, .58],
    "dark": [-.114, -.587, -.299],
}
RECORD = None
READER = None


def samples(poly, stations, *, axis=None, step=37.):
    start = poly[:-1].copy()
    delta = np.diff(poly, axis=0)
    length = np.linalg.norm(delta, axis=1)
    valid = np.isfinite(delta).all(1) & (length > 1.e-5)
    if not valid.any():
        return None
    starts, ends = start[valid], start[valid] + delta[valid]
    # Faster inspection of long native segments while protecting against an
    # offscreen endpoint creating hundreds of useless profile positions.
    on_screen = (
        (np.minimum(starts, ends) <= [WIDTH - 5., HEIGHT - 5.]).all(1)
        & (np.maximum(starts, ends) >= [5., 5.]).all(1)
    )
    indices = np.flatnonzero(valid)[on_screen]
    if not len(indices):
        return None
    pieces = length[indices]
    arclen = np.r_[0., np.cumsum(pieces)]
    level = np.arange(step * .38, arclen[-1], step)
    if len(level) < 4:
        return None
    segment = np.minimum(np.searchsorted(arclen, level, side="right") - 1, len(indices) - 1)
    local = (level - arclen[segment]) / pieces[segment]
    idx = indices[segment]
    points = start[idx] + local[:, None] * delta[idx]
    st = stations[idx] + local * (stations[idx + 1] - stations[idx])
    tangents = delta[idx] / length[idx, None]
    normal = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    if axis is not None:
        normal *= np.where(normal[:, axis] >= 0., 1., -1.)[:, None]
    # Keep a uniform sign along a curve and allow samples close to the image
    # edge. Individual colour readings outside the frame remain unavailable.
    good = ((points >= [4., 4.]) & (points <= [WIDTH - 5., HEIGHT - 5.])).all(1)
    if len(points) > 192:
        good[192:] = False
    return points[good], normal[good], tangents[good], st[good]


def source_profiles(raw, mask, points, normal, tangent, radius, tone):
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    line = points[:, None, :] + normal[:, None, :] * grid[None, :, None]
    # Average a short real slice along the material line. Do not draw or
    # resynthesize the transition used for the observation.
    locations = line[:, :, None, :] + tangent[:, None, None, :] * np.array([-3., 0., 3.])[None, None, :, None]
    r = locations.reshape(len(points), -1, 2).astype(np.float32)
    map_x, map_y = np.split(r, 2, axis=2)
    map_x, map_y = map_x[:, :, 0], map_y[:, :, 0]
    value = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR)
    hidden = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    color = (value.astype(np.float32) @ np.asarray(CHANNELS[tone], np.float32)).reshape(len(points), -1, 3)
    rgb = np.median(value.reshape(len(points),-1,3,3),axis=2)
    profile = np.median(color, axis=2)
    bad = hidden.reshape(len(points), -1, 3).any(axis=2)
    smooth = cv2.GaussianBlur(profile, (9, 1), 1.55, borderType=cv2.BORDER_REPLICATE)
    return smooth, bad, rgb


def subpixel(v, k):
    a, b, c = v[k - 1:k + 2]
    curve = a + c - 2. * b
    return float(np.clip(.5 * (a - c) / curve, -.8, .8)) if curve < -1.e-5 else 0.


def stripe_candidates(profile, hidden, radius, *, expected_min, max_width, min_contrast):
    """Return candidates each supported by both material edges and neutral flanks."""
    gradient = np.gradient(profile)
    peaks = find_peaks(gradient, height=.23, distance=3)[0]
    troughs = find_peaks(-gradient, height=.23, distance=3)[0]
    choices = []
    for l in peaks:
        contains = troughs[(troughs >= l + expected_min) & (troughs <= l + max_width)]
        for r in contains:
            width = r - l
            flank = min(12, max(4, int(width * .25)))
            middle = (.5 * (l + r)) - radius
            if abs(middle) > 48 or l < flank or r + flank + 1 >= len(profile):
                continue
            if hidden[l - flank:r + flank + 1].any():
                continue
            band = profile[l + width // 4:r - width // 4 + 1]
            before = float(np.median(profile[l - flank:l - 1]))
            after = float(np.median(profile[r + 2:r + flank + 1]))
            interior = float(np.median(band))
            contrast = interior - max(before, after)
            weakest = min(gradient[l], -gradient[r])
            strongest = max(gradient[l], -gradient[r])
            if contrast < min_contrast or weakest < max(.27, strongest * .18):
                continue
            if abs(before - after) > max(4., 1.1 * contrast):
                continue
            # Use the gradient flanks in original pixels, including both
            # flanks of very broad strokes. A peak interior does not define
            # the line's location.
            left = l + subpixel(gradient, l) - radius
            right = r + subpixel(-gradient, r) - radius
            quality = contrast * np.sqrt(weakest) / (1. + abs(middle) / 45.)
            choices.append((.5 * (left + right), quality, left, right, contrast))
    return sorted(choices, key=lambda z: -z[1])[:3]


def boundary_candidates(profile, hidden, colour, radius, *, sign, tone):
    gradient = sign * np.gradient(profile)
    peaks = find_peaks(gradient, height=.75, distance=4)[0]
    choices = []
    for p in peaks:
        displacement = p - radius
        if abs(displacement) > radius - 11 or hidden[p - 8:p + 9].any():
            continue
        exterior = float(np.median(profile[p-8:p-3] if sign > 0. else profile[p+4:p+9]))
        internal = slice(p+4,p+9) if sign > 0. else slice(p-8,p-3)
        inside = float(np.median(profile[internal]))
        if tone == "yellow":
            minimum, prominence = 18., 7.
        elif tone == "purple":
            minimum, prominence = 17., 10.
        else:
            minimum, prominence = 13., 7.
            blue,green,red=np.median(colour[internal],axis=0)
            # The dark arena cap is distinct from the light blue or purple
            # advertisement panel that sometimes lies in the search strip.
            if red>112. or blue-red<19. or blue-green<10.:
                continue
        delta = inside - exterior
        if inside < minimum or delta < prominence or exterior > .84 * inside:
            continue
        quality = min(delta, 100.) * np.sqrt(gradient[p]) / (1. + abs(displacement) / 32.)
        choices.append((displacement + subpixel(gradient, p), quality, np.nan, np.nan, delta))
    return sorted(choices, key=lambda z: -z[1])[:3]


def coherent(choices, points, *, stripe, minimum_span=None):
    """Follow a coherent material edge without selecting per-column false edges."""
    if sum(bool(c) for c in choices) < 4:
        return []
    candidates = [(i, z) for i, c in enumerate(choices) for z in c]
    offsets = np.array([z[0] for _, z in candidates])
    score = np.array([z[1] for _, z in candidates])
    weights = np.minimum(np.sqrt(score), np.quantile(np.sqrt(score), .8))
    index = np.array([i for i, _ in candidates])
    centred = points - np.median(points, axis=0)
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ direction
    initial_span=(62. if stripe else 78.) if minimum_span is None else float(minimum_span)
    accepted_span=(72. if stripe else 96.) if minimum_span is None else float(minimum_span)
    if initial_span<=0:raise ValueError('minimum_span must be positive')
    if np.ptp(along) < initial_span:
        return []
    x = along / max(100., np.ptp(along))
    half = 6.5 if stripe else 4.5
    best = -1.
    model = None
    for m in (-15., -7., 0., 7., 15.):
        intercept = offsets - m * x[index]
        nearby = np.maximum(0., 1. - np.abs(intercept[:,None] - intercept[None,:]) / half)
        votes = nearby @ weights
        j = np.argmax(votes)
        if votes[j] > best:
            best = float(votes[j])
            model = (float(intercept[j]), m)
    path = []
    for _ in range(2):
        path = []
        for i, cs in enumerate(choices):
            if not cs:
                continue
            expected = model[0] + model[1] * x[i]
            z = min(cs, key=lambda q: abs(q[0] - expected) - .03 * np.log1p(q[1]))
            if abs(z[0] - expected) <= half:
                path.append((i,z))
        if len(path) < 4:
            return []
        use = np.array([i for i,_ in path])
        if np.ptp(along[use]) < accepted_span:
            return []
        if np.ptp(use) < 3 or len(path) < .42 * sum(bool(c) for c in choices):
            return []
        design = np.column_stack((np.ones(len(use)), x[use]))
        model = np.linalg.lstsq(design, [z[0] for _,z in path], rcond=None)[0]
    if stripe:
        width = np.array([z[3] - z[2] for _, z in path])
        typical = np.median(width)
        path = [(i,z) for (i,z),w in zip(path,width,strict=True) if .42 * typical < w < 2.35 * typical]
    return path if len(path) >= 4 else []


class PhysicalReader:
    def __init__(self, record):
        sys.path.insert(0, str(JOINT))
        from joint_inputs import family, native_paths
        self.native = frozen_scene(record, verify=True)
        paths = [TOP,BOTTOM]
        self.info = {TOP: ("blue","rise",1,"source_blue_rail_top"),
                     BOTTOM: ("yellow","fall",1,"source_yellow_rail_bottom")}
        for k in self.native.paths:
            f = family(k,self.native)
            if f in {"B1", "B2", "C"} and not k.startswith("print_instance/"):
                self.info[k] = ("red" if f == "C" else "blue", "stripe", None, "source_two_flanks_paint_centre")
                paths.append(k)
            if k.startswith(("print_instance/Ad2::", "print_instance/Ad15::")):
                role = k.split("::Ad2.",1)[-1]
                if role in AD2:
                    self.info[k] = (*AD2[role],"source_purple_panel_edge")
                    paths.append(k)
        self.paths = paths
        self.family = {k:family(k,self.native) for k in paths}
        self.data = native_paths(self.native, paths, count=281)
        ends = np.r_[0, np.cumsum([len(self.data[k][0]) for k in paths])]
        self.slices = {k:slice(a,b) for k,a,b in zip(paths,ends[:-1],ends[1:],strict=True)}
        self.xyz = np.vstack([self.data[k][0] for k in paths])
        self.course = self.data[TOP][0][:,:2]
        along = np.gradient(self.course,axis=0)
        self.inward = np.c_[along[:,1],-along[:,0]]
        self.inward *= np.where((self.inward*(-self.course)).sum(1)<0,-1.,1.)[:,None]
        self.lateral={}
        for k in paths:
            if self.info[k][1]=="stripe":
                line=self.data[k][0]
                a=line[-1]-line[0]
                b=np.array([-a[1],a[0],0.])
                self.lateral[k]=.004*b/np.linalg.norm(b)

    def stripe_footprint(self,k,station,normal,rig,controls,t):
        world,native_stations=self.data[k]
        centre=np.column_stack([np.interp(station,native_stations,world[:,i]) for i in range(3)])
        side=self.lateral[k]
        q=self.native.core.project_motion(rig,controls,FIRST,DT,HEIGHT,t,np.vstack([centre-side,centre+side]))
        a,b=q["uv"].reshape(2,len(centre),2)
        width=np.abs(np.sum((b-a)*normal,axis=1))
        width[(q["status"].reshape(2,len(centre))!=0).any(0)]=0.
        return width

    @staticmethod
    def board_foot(values):
        projected=values[np.isfinite(values).all(1)]
        use=(projected[:,0]>=-6)&(projected[:,0]<=WIDTH+5)&(projected[:,1]>=2)&(projected[:,1]<HEIGHT)
        projected=projected[use]
        if len(projected)<10:
            return np.zeros(WIDTH)
        x=np.clip(np.rint(projected[:,0]).astype(int),0,WIDTH-1)
        lowest=np.full(WIDTH,np.inf)
        np.minimum.at(lowest,x,projected[:,1])
        found=np.flatnonzero(np.isfinite(lowest))
        return np.interp(np.arange(WIDTH),found,lowest[found])

    def read(self, raw, mask, rig, controls, t, camera_centre):
        result = self.native.core.project_motion(rig, controls, FIRST, DT, HEIGHT, t, self.xyz)
        uv = result["uv"]
        uv[result["status"]!=0] = np.nan
        visible = np.sum((self.course - camera_centre[:2]) * self.inward, axis=1) <= 0.
        for k in (TOP,BOTTOM):
            uv[self.slices[k]][~visible] = np.nan
        foot=self.board_foot(uv[self.slices[BOTTOM]])
        floor_mask=None
        for k in self.paths:
            tone, kind, axis, label = self.info[k]
            stripe = kind == "stripe"
            profiles = samples(uv[self.slices[k]], self.data[k][1], axis=axis, step=34. if stripe else 47.)
            if profiles is None:
                continue
            points, normal, tangent, station = profiles
            if len(points)<4:
                continue
            radius = 134 if stripe else 49
            if stripe and floor_mask is None:
                floor_mask=mask|((np.arange(HEIGHT)[:,None]<foot[None,:]+14.)&(foot[None,:]>0.1))
            values, hidden, colours = source_profiles(raw, floor_mask if stripe else mask,
                                                     points,normal,tangent,radius,tone)
            if stripe:
                rough_width=self.stripe_footprint(k,station,normal,rig,controls,t)
                # The footprint is a broad search check, not fitted stroke
                # width. Take the actual midpoint only from the two raw flanks.
                hi=3.05 if tone=="blue" else 1.78
                lo=.38 if tone=="blue" else .49
                cap=145 if tone=="blue" else 230
                limit=2.6 if tone=="blue" else 3.4
                choices=[stripe_candidates(v,m,radius,expected_min=max(3 if tone=="blue" else 5,int(lo*w)),
                           max_width=min(cap,int(hi*w+3)),min_contrast=limit) if w>=2.5 else []
                         for v,m,w in zip(values,hidden,rough_width,strict=True)]
            else:
                sign = 1. if kind=="rise" else -1.
                choices = [boundary_candidates(v,m,c,radius,sign=sign,tone=tone)
                           for v,m,c in zip(values,hidden,colours,strict=True)]
            accepted = coherent(choices,points,stripe=stripe)
            if not accepted:
                continue
            j = np.array([i for i,_ in accepted])
            d = np.asarray([z for _,z in accepted])
            source_uv = points[j] + d[:,0,None] * normal[j]
            bracket = points[j,None,:] + d[:,2:4,None] * normal[j,None,:]
            if k==BOTTOM:
                order=np.argsort(source_uv[:,0])
                x=source_uv[order,0]
                shift=source_uv[order,1]-np.interp(x,np.arange(WIDTH),foot)
                good=(np.interp(x,np.arange(WIDTH),foot)>0.1)&(abs(shift)<38.)
                if good.sum()>=4:
                    correction=np.interp(np.arange(WIDTH),x[good],shift[good])
                    foot=np.maximum(0.,foot+correction)
            yield k, label, station[j], source_uv, normal[j], d[:,4], d[:,0], bracket


def initialize(record):
    global RECORD, READER
    cv2.setNumThreads(1)
    RECORD = record
    READER = PhysicalReader(record)


def process(task):
    shot, guide, out, stride, silhouettes = task
    sid = shot["id"]
    started = monotonic()
    pts, ctrl, rig = load_camera(shot, guide)
    time = motion_clock(shot,guide,pts)
    camera=READER.native.core.evaluate_motion_camera(rig,ctrl,FIRST,DT,time)
    assert np.all(camera["status"]==0)
    roles = fold_map(shot,len(pts))
    with np.load(OUTPUT/"foreground"/f"{sid}.npz",allow_pickle=False) as a:
        np.testing.assert_array_equal(a["pts"],pts)
        boxes,offsets = a["boxes"],a["offsets"]
    mask_source = visibility_packet(silhouettes,sid) if silhouettes else None
    if mask_source is not None and mask_source.is_file():
        with np.load(mask_source,allow_pickle=False) as a:
            np.testing.assert_array_equal(a["pts"],pts)
            assert a["source_sha256"].item()==RECORD["source"]["sha256"]
            packed, person_valid = a["packed"],a["valid"]
    else:
        packed,person_valid = None,None
    rows=defaultdict(list)
    hashes=[]
    stats=Counter()
    how=Counter()
    indices=np.unique(np.r_[np.arange(0,len(pts),stride),np.flatnonzero((roles==1)|(roles==2)),len(pts)-1])
    for p,frame in source_frames(RECORD["source"]["registered_path"],pts[indices]):
        i=int(np.searchsorted(pts,p))
        raw=frame.to_ndarray(format="bgr24")
        if packed is not None and person_valid[i]:
            person=np.unpackbits(packed[i],axis=1,count=WIDTH)
            mask=broadcast_overlay()|cv2.dilate(person,np.ones((13,13),np.uint8))
            how["instance_silhouette"]+=1
        else:
            mask=group_foreground(boxes,offsets,i).astype(np.uint8)
            how["source_detection_search_box"]+=1
        for k,label,station,pixels,normal,contrast,shift,bracket in READER.read(
                 raw,mask,rig,ctrl,time[i],camera["centre"][i]):
            n=len(pixels)
            stats[READER.family[k]]+=n
            item=dict(path=np.repeat(k,n),family=np.repeat(READER.family[k],n),
                  pts=np.repeat(p,n),idx=np.repeat(i,n),fold=np.repeat(roles[i],n),
                  station=station,uv=pixels,material=np.repeat(label,n),normal=normal,
                  contrast=contrast,guide_error=abs(shift),source_projection_signed_offset_px=shift,
                  source_bracket=bracket)
            for field,value in item.items():
                rows[field].append(value)
        hashes.append((p,hashlib.sha256(raw.tobytes()).hexdigest()))
        if len(hashes)%160==0:
            print(sid,"frames",len(hashes),"/",len(indices),"points",sum(stats.values()),
                  f"{monotonic()-started:.1f}s",flush=True)
    if not rows:
        raise ValueError("No readable physical profiles in "+sid)
    row={k:np.concatenate(v) for k,v in rows.items()}
    sustained=np.zeros(len(row["pts"]),bool)
    for k in np.unique(row["path"]):
        selected=row["path"]==k
        present=np.unique(row["idx"][selected])
        hit=np.r_[False,np.diff(present)<=max(4,2*stride)]|np.r_[np.diff(present)<=max(4,2*stride),False]
        sustained|=selected & np.isin(row["idx"],present[hit])
    row["eligible"]=sustained
    row["eligible_before_quality_review"]=np.ones(len(sustained),bool)
    row["excluded_reason"]=np.where(sustained,"","material_profile_has_no_adjacent_source_support")
    row["source_sha256"]=RECORD["source"]["sha256"]
    row["geometry_sha256"]=RECORD["geometry_binding"]["value_sha256"]
    row["source_frame_pts"]=np.array([p for p,_ in hashes],np.int64)
    row["source_frame_BGR_sha256"]=np.array([h for _,h in hashes])
    save(out/f"{sid}.npz",**row)
    info=dict(shot=sid,native_frames=len(hashes),eligible=int(sustained.sum()),source_points=dict(stats),
              by_exclusion_mask=dict(how),seconds=monotonic()-started)
    write(out/f"{sid}.json",info)
    print(info,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--silhouettes")
    parser.add_argument("--stride",type=int,default=2)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--shots",nargs="+")
    a=parser.parse_args()
    out=OUTPUT/a.out
    out.mkdir(exist_ok=True)
    record=scope()
    work=[(s,OUTPUT/a.guide,out,a.stride,OUTPUT/a.silhouettes if a.silhouettes else None)
          for s in record["shots"] if (not a.shots or s["id"] in a.shots) and not (out/f"{s['id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(record,)) as pool:
        list(pool.map(process,work))


if __name__=="__main__":
    main()
