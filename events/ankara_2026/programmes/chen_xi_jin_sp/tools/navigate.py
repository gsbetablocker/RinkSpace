"""Initialize fixed-installation PTZ paths from actual static source transfer."""
from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
from time import monotonic

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import EVENT, OUTPUT, TB, STEP, WIDTH, HEIGHT, frozen_scene, read, save, scope, source_frames, source_pts

W, H = 960, 540
CENTRE = np.array([(WIDTH-1)*.5, (HEIGHT-1)*.5])
ROI = np.ones((H, W), np.uint8)*255
ROI[:142, :268] = 0
ROI[-67:, -140:] = 0
ROI[[0, 1, -1, -2], :] = 0
ROI[:, [0, 1, -1, -2]] = 0
STILL = ROI.copy()
STILL[201:505, 417:546] = 0
DT = .02
MIN_FOCAL, MAX_FOCAL = 420., 44900.
CLAHE = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 8))


def gray(frame):
    return frame.reformat(width=W, height=H, format="gray").to_ndarray()


def full_uv(points):
    return points.astype(np.float64)*2 + .5


def intrinsics(cam):
    fx, fy, cx, cy, *dist = cam[16:24]
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]), np.r_[dist, 0.]


def world_directions(cam, matches):
    matrix, coeff = intrinsics(cam)
    norm = cv2.undistortPoints(np.ascontiguousarray(matches[:, None, :]), matrix, coeff)[:, 0]
    return np.c_[norm, np.ones(len(norm))] @ cam[4:13].reshape(3, 3)


class Pose:
    def __init__(self, core):
        self.core = core
        self.controls = np.empty((6, 3), dtype=np.float64)

    def camera(self, rig, theta):
        self.controls[:] = theta
        p = self.core.evaluate_motion_camera(rig, self.controls, -2., 1., 0.)
        if int(p["status"][0]):
            raise ValueError("Navigation camera outside projection domain")
        return p["values"][0]

    def fit(self, rig, guess, directions, observed, *, rough=False):
        guess = np.asarray(guess, float).copy()
        guess[2] = np.clip(guess[2], np.log(MIN_FOCAL)+1.e-8, np.log(MAX_FOCAL)-1.e-8)
        reference = self.camera(rig, guess)
        # Unit rays approximate same-installation image transfer. This result
        # locates source searches only; no ray becomes an observation.
        xyz = reference[13:16] + 80.*directions

        def residual(theta):
            self.controls[:] = theta
            result = self.core.project_motion(rig, self.controls, -2., 1., HEIGHT, 0., xyz)
            uv = result["uv"]
            error = uv - observed
            error[result["status"] != 0] = 3000.
            return error.ravel()

        bound = [.75, .55] if rough else [.095, .095]
        lower = [*(guess[:2]-bound), np.log(MIN_FOCAL)]
        upper = [*(guess[:2]+bound), np.log(MAX_FOCAL)]
        solved = least_squares(residual, guess, bounds=(lower, upper), x_scale=[.12, .12, .7],
                loss="soft_l1", f_scale=3.0 if rough else 1.5,
                max_nfev=25 if rough else 10, ftol=3.e-5, xtol=3.e-6, gtol=1.e-4)
        return solved.x, np.linalg.norm(solved.fun.reshape(-1, 2), axis=1)


def source_features(sift, image):
    k, d = sift.detectAndCompute(image, STILL)
    if d is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    return cv2.KeyPoint_convert(k), d


def reference_views(record, native, *, step=26):
    path = OUTPUT/"reference_views.npz"
    reference_bank = Path(os.environ.get("RINKSPACE_ANKARA_NAVIGATION_BANK", path))
    if reference_bank.exists():
        with np.load(reference_bank, allow_pickle=False) as z:
            assert z["geometry"].item() == record["geometry_binding"]["value_sha256"]
            return {key: z[key] for key in z.files}
    reference = EVENT/"programmes/jin_shuxian_sp"
    source = read(reference/"rebuild_scope.json")["source"]["registered_path"]
    data = np.load(reference/"stability_review_1113_camera.npz", allow_pickle=False)
    geometry = record["geometry_binding"]["value_sha256"]
    assert data["scene_values_sha256"].item() == geometry
    pts = data["source_pts"]
    camera, c_offset = data["rig_by_source_pts"], data["control_offsets"]
    rigs = []
    rows = {}
    starts = data["source_offsets"]
    for i, (a, b) in enumerate(zip(starts, starts[1:])):
        rig_id = int(data["camera_indices"][i])
        k = np.unique(np.r_[np.arange(a, b, step), (a+b)//2, b-1])
        c = data["controls"][c_offset[i]:c_offset[i+1]]
        for j in k:
            cam = native.core.evaluate_motion_camera(camera[j], c, data["first_control_time"][i],
                   data["control_dt"][i], data["local_exposure_seconds"][j])
            if not cam["status"][0]:
                rows[int(pts[j])] = (rig_id, cam["values"][0])
    for rig_id in range(3):
        use = [camera[a:b] for i, (a,b) in enumerate(zip(starts, starts[1:]))
               if data["camera_indices"][i] == rig_id]
        vals, counts = np.unique(np.vstack(use), axis=0, return_counts=True)
        rigs.append(vals[np.argmax(counts)])
    sift = cv2.SIFT_create(nfeatures=1250, contrastThreshold=.019)
    descriptors, pixels, offset, cam, rig_ids, clocks = [], [], [0], [], [], []
    for p, frame in source_frames(source, sorted(rows)):
        uv, ds = source_features(sift, gray(frame))
        if len(uv) < 16:
            continue
        rig_id, lens = rows[p]
        descriptors.append(ds.astype(np.uint8))
        pixels.append(uv)
        offset.append(offset[-1]+len(ds))
        cam.append(lens)
        rig_ids.append(rig_id)
        clocks.append(p)
    bank = dict(desc=np.vstack(descriptors), uv=np.vstack(pixels), offsets=np.array(offset),
                ref_cameras=np.asarray(cam), ref_camera_ids=np.array(rig_ids),
                ref_pts=np.array(clocks), rigs=np.array(rigs), geometry=geometry)
    save(path, **bank)
    print("reference views", len(clocks), "keypoints", len(bank["desc"]), flush=True)
    return bank


def matcher(ds):
    result = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    result.add([np.ascontiguousarray(ds, dtype=np.float32)])
    result.train()
    return result


class Bank:
    def __init__(self, data):
        self.data = data
        self.sift = cv2.SIFT_create(nfeatures=1450, contrastThreshold=.018)
        self.lookups = []
        for rig in range(3):
            refs = np.flatnonzero(data["ref_camera_ids"] == rig)
            ds = np.vstack([self.descriptors(i) for i in refs])
            spans = np.repeat(refs, np.diff(data["offsets"])[refs])
            self.lookups.append((spans, matcher(ds)))

    def descriptors(self, i):
        a,b = self.data["offsets"][i:i+2]
        return self.data["desc"][a:b]

    @lru_cache(maxsize=96)
    def reference_matcher(self, i):
        return matcher(self.descriptors(i))

    def query(self, im, rig_id, pose, hint=None):
        pixel, desc = source_features(self.sift, im)
        if len(pixel) < 16:
            return None
        index, search = self.lookups[rig_id]
        votes = np.zeros(len(self.data["ref_camera_ids"]))
        for nearest in search.knnMatch(desc, k=6):
            if not nearest:
                continue
            ref = set()
            for j, m in enumerate(nearest):
                i = index[m.trainIdx]
                if i not in ref:
                    votes[i] += (1. - j*.10)/(55. + m.distance)
                    ref.add(i)
        best = np.argsort(-votes)[:14]
        if hint is not None:
            poses = self.data["ref_cameras"][:, :3]
            near = np.linalg.norm((poses - hint)*[4., 2., .4], axis=1)
            near[self.data["ref_camera_ids"] != rig_id] = np.inf
            best = np.unique(np.r_[best, np.argsort(near)[:7]])
        possible = []
        rig = self.data["rigs"][rig_id]
        for i in best:
            if votes[i] <= 0 or self.data["ref_camera_ids"][i] != rig_id:
                continue
            pairs = self.reference_matcher(int(i)).knnMatch(desc, k=2)
            m = [(x.queryIdx,x.trainIdx) for x,y in pairs if x.distance < .76*y.distance]
            if len(m) < 15:
                continue
            query_i,ref_i = np.asarray(m).T
            a,b = self.data["offsets"][i:i+2]
            ref = self.data["uv"][a:b][ref_i]
            target = pixel[query_i]
            matrix, inlier = cv2.findHomography(ref, target, cv2.USAC_MAGSAC, 1.8,
                                                maxIters=1000, confidence=.995)
            if inlier is None or np.count_nonzero(inlier) < 13:
                continue
            ok = inlier.ravel().astype(bool)
            r, q = ref[ok], target[ok]
            area_r = cv2.contourArea(cv2.convexHull(r))
            area_q = cv2.contourArea(cv2.convexHull(q))
            if min(area_q, area_r) < 850 or max(np.ptp(q, axis=0)) < 95:
                continue
            score = len(q)*(min(80000., area_q)*min(80000., area_r))**.20
            possible.append((score, r, q, int(i), area_q/area_r))
        if not possible:
            return None
        tested = []
        for _, r, q, i, scale in sorted(possible, key=lambda x:-x[0])[:5]:
            ref_cam = self.data["ref_cameras"][i]
            guess = ref_cam[:3].copy()
            guess[2] += .5*np.log(scale)
            good = world_directions(ref_cam, full_uv(r))
            target = full_uv(q)
            try:
                theta, errors = pose.fit(rig, guess, good, target, rough=True)
            except ValueError:
                continue
            valid = errors <= 5.5
            if valid.sum() < 12 or valid.mean() < .50:
                continue
            theta, errors = pose.fit(rig, theta, good[valid], target[valid], rough=True)
            med, p90 = np.quantile(errors, [.5, .90])
            if p90 > 7.5:
                continue
            spread = np.ptp(q[valid], axis=0)
            score = valid.sum()*min(1., (.25+.75*spread[1]/160.))/(1.+med/3.)
            tested.append((score, theta, med, p90, int(valid.sum()), int(self.data["ref_pts"][i])))
        return max(tested, default=None, key=lambda x:x[0])


def flow_step(a,b):
    xy = cv2.goodFeaturesToTrack(a, 480, .003, 10., mask=STILL, blockSize=7)
    if xy is None or len(xy) < 12:
        return np.eye(3), 0, float("inf")
    kwargs = dict(winSize=(23, 23), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 24, .02))
    q, status, error = cv2.calcOpticalFlowPyrLK(a,b,xy,None,**kwargs)
    if q is None:
        return np.eye(3), 0, float("inf")
    rev, back, _ = cv2.calcOpticalFlowPyrLK(b,a,q,None,**kwargs)
    if rev is None:
        return np.eye(3), 0, float("inf")
    valid = (status[:, 0] > 0) & (back[:, 0] > 0) & (error[:, 0] < 26.)
    valid &= (np.linalg.norm(rev[:,0]-xy[:,0],axis=1)<.62)
    a0,b0 = full_uv(xy[valid,0]),full_uv(q[valid,0])
    if len(a0) < 12:
        return np.eye(3), 0, float("inf")
    h, ok = cv2.findHomography(a0, b0, cv2.USAC_MAGSAC, 2.0, maxIters=750, confidence=.995)
    if ok is None or np.count_nonzero(ok)<12:
        return np.eye(3), 0, float("inf")
    ok = ok.ravel() > 0
    a0,b0 = a0[ok],b0[ok]
    if np.ptp(a0[:,0])<260. or np.ptp(a0[:,1])<45.:
        return np.eye(3), 0, float("inf")
    err = np.linalg.norm(cv2.perspectiveTransform(a0[None], h)[0]-b0,axis=1)
    return h, len(err), float(np.quantile(err,.90))


def choose_grid():
    xx,yy = np.meshgrid(np.linspace(180.,1740.,6), np.linspace(145.,939.,4))
    return np.c_[xx.ravel(), yy.ravel()]


def transfer(pose, rig, theta, h, x):
    target = cv2.perspectiveTransform(x[None], h)[0]
    # The centre of a virtual optical transfer grid has no source authority.
    shift = np.median(target-x, axis=0)
    r = np.linalg.norm(x-CENTRE,axis=1)
    s = np.linalg.norm(target-CENTRE-shift,axis=1)
    guess = theta.copy()
    guess[2] += np.clip(np.median(np.log(np.maximum(s,1.)/r)), -.12, .12)
    res, err = pose.fit(rig, guess, world_directions(pose.camera(rig, theta), x),target)
    return res if np.median(err) < 35. else theta


def sequential(pts, anchors, matrices, support, pose, rig, *, reference_index=None,
               max_pan_mismatch=None):
    n = len(pts)
    reports = []
    grid = choose_grid()
    weights = np.asarray([a[0] for a in anchors.values()])
    start, end = min(anchors), max(anchors)
    if len(anchors) > 1:
        minimum = max(3., np.quantile(weights,.20)*.32)
        anchors = {i:a for i,a in anchors.items() if a[0]>=minimum or i==reference_index}
    origin = max(anchors, key=lambda i:anchors[i][0]) if reference_index is None else reference_index
    guided = np.empty((n,3), dtype=np.float64)
    guided[origin] = anchors[origin][1]
    for direction in (1,-1):
        indices = range(origin+direction,n if direction==1 else -1,direction)
        for i in indices:
            j = i-direction
            k = min(i,j)
            h = matrices[k] if direction==1 else np.linalg.inv(matrices[k])
            if support[k]:
                guided[i] = transfer(pose, rig, guided[j], h, grid)
            elif 0 <= j-direction < n:
                guided[i] = guided[j] + np.clip(guided[j]-guided[j-direction], -.03,.03)
            else:
                guided[i] = guided[j]
    indices = np.array(sorted(anchors),dtype=np.int64)
    obs=np.array([anchors[i][1] for i in indices])
    delta=obs-guided[indices]
    delta[:,0] -= np.rint(delta[:,0]/(2*np.pi))*2*np.pi
    consistent = np.ones(len(indices), dtype=bool)
    if max_pan_mismatch is not None:
        consistent &= abs(delta[:,0]) <= max_pan_mismatch
    admissible = np.flatnonzero(consistent)
    if len(admissible)>=3:
        for left,j,right in zip(admissible,admissible[1:],admissible[2:]):
            if indices[j] == reference_index:
                continue
            scale = (indices[j]-indices[left])/(indices[right]-indices[left])
            pred = delta[left]*(1-scale)+delta[right]*scale
            e = abs(delta[j]-pred)
            if e[0]>.027 or e[1]>.022 or e[2]>.080:
                # An isolated atlas jump relative to bidirectional source
                # flow is navigation ambiguity, not a new cut or an observation.
                neigh = abs(delta[right]-delta[left])
                if np.all(neigh < [.065,.06,.19]) and indices[right]-indices[left]<100:
                    consistent[j] = False
    for j,i in enumerate(indices):
        score,theta,med,p90,count,ref = anchors[int(i)]
        reports.append(dict(pts=int(pts[i]),reference_pts=ref,median_px=float(med),p90_px=float(p90),
                            matches=count,score=float(score),retained=bool(consistent[j])))
        if max_pan_mismatch is not None:
            reports[-1]["anchor_pan_minus_source_flow_rad"] = float(delta[j,0])
    times=np.arange(n)
    corrected=np.column_stack([np.interp(times,indices[consistent],x[consistent]) for x in delta.T])
    out = guided+corrected
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite navigation result")
    return out,reports


def make_shot(shot, filename, bank, pose, save_to, *, stride=13):
    started=monotonic()
    sid=shot["id"]
    rig_id=int(shot["camera_index"])
    pts=source_pts(shot)
    source_n=len(pts)
    if sid=="s20":
        # This same shot continues until the next cut. Its clearer near-tail
        # texture may navigate back into the fit window; never enter as UV rows.
        pts=np.arange(pts[0],round(199.32*12800),STEP,dtype=np.int64)
    anchors={}
    prev=None
    steps,counts, p90s = [],[],[]
    cache=save_to/f"{sid}_flow.npz"
    get = set(np.unique(np.r_[np.arange(0,len(pts),stride), len(pts)-1]))
    get.update(np.searchsorted(pts, [shot["representative_pts"]]))
    for i,(p,frame) in enumerate(source_frames(filename,pts)):
        image=gray(frame)
        current=CLAHE.apply(image)
        if prev is not None:
            h, count, error=flow_step(prev,current)
            steps.append(h)
            counts.append(count)
            p90s.append(error)
        if i in get:
            hint=anchors[max(anchors)][1] if anchors else None
            candidate=bank.query(image,rig_id,pose,hint)
            if candidate is not None:
                anchors[i]=candidate
        prev=current
        if i and i%200==0:
            print(sid,"source flow",i+1,"of",len(pts),"anchors",len(anchors),
                  f"{monotonic()-started:.1f}s",flush=True)
    if not anchors:
        raise ValueError(f"{sid}: no same-camera source initializer")
    support=np.asarray(counts)>0
    location,reports=sequential(pts,anchors,np.asarray(steps),support,pose,bank.data["rigs"][rig_id])
    save(cache,pts=pts,adjacent_h=steps,counts=counts,error_p90=p90s)
    save(save_to/f"{sid}.npz",pts=pts[:source_n],pose=location[:source_n],rig=bank.data["rigs"][rig_id],
         flow_supported=np.r_[True,support][:source_n],anchor_pts=[x["pts"] for x in reports if x["retained"]])
    report=dict(shot=sid,source_frame_count=source_n,flow_supported=int(support.sum()),flow_total=len(support),
                anchors=reports,projections_are_navigation_only=True,seconds=monotonic()-started)
    write(save_to/f"{sid}.json",report)
    print(sid,"navigation",source_n,"flow",int(support.sum()),"anchors",len(reports),
          "focal",np.quantile(np.exp(location[:source_n,2]),[0,.5,1]),f"{report['seconds']:.1f}s",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots",nargs="+")
    args=ap.parse_args()
    cv2.setNumThreads(2)
    record=scope()
    native=frozen_scene(record,verify=True)
    data=reference_views(record,native)
    bank=Bank(data)
    pose=Pose(native.core)
    out=OUTPUT/"navigation"
    out.mkdir(exist_ok=True)
    for shot in record["shots"]:
        sid=shot["id"]
        if (args.shots is None or sid in args.shots) and not (out/f"{sid}.json").exists():
            make_shot(shot,record["source"]["registered_path"],bank,pose,out)


if __name__=="__main__":
    from common import write
    main()
