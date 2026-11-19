"""Audit the visibility of both LANE4 upper-longitudinal paint flanks."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import warnings

import av
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from lane_side_1118 import BASE, CAP, SIDE, WORK, X, Y, inverse, project
from review import camera_state, ji, native_for, source_frames
from shared import DATA, EVENT, OUT, PREPARED, arrays, prepared_packet_path, read, write
from rinkspace.calibration_companion.wide_angle_camera import nearest_valid_polyline_vectors


OUTPUT = WORK / 'lane_source'
SCOUT = WORK / 'lane_upper_scout'
REGIONS = {
    'left': (-.425, -.388, .2175),
    'left_mid': (-.287, -.245, .2165),
    'middle': (-.219, .165, .2150),
    'right': (.272, .308, .2125),
}
Y_MIN, Y_MAX = .200, .228
STRIP_ROWS = (Y >= Y_MIN) & (Y <= Y_MAX)
WIDE_DIR = OUT.parent / 'trapezoid_joint_1116'
MEDIA = DATA / 'events/ankara_2026/media/originals/oxXu1avhrvs.mp4'
WIDE_RECORD = DATA / 'events/ankara_2026/evidence/delivery_a/broadcast/hd_source_validation.json'
WIDE_CENTRE = np.array([959.5, 539.5])
WIDE_SCALE = 960.
TIME_BASE = Fraction(1, 15360)
EMPTY_TIMES = [3610, 8640, 14340]
WIDE_TIMES = [3610, 3624, 3645, 3660, 3671, 3685, 3710, 3760, 8640, 8660, 8700,
              11222, 11230, 11260, 14340, 14378, 14407, 14418, 14450, 14500]
STATIC_PATCHES = [(193,251), (452,209), (814,245), (1104,292), (1340,329), (1606,376), (1873,471)]


def wide_frames():
    """Decode native exposures and match the wide-view camera on seven board regions."""
    SCOUT.mkdir(parents=True, exist_ok=True)
    records = {}
    rows = []
    with av.open(str(MEDIA)) as video:
        stream = video.streams.video[0]
        if stream.time_base != TIME_BASE or (stream.width, stream.height) != (1920,1080):
            raise ValueError('Unexpected full-broadcast source domain')
        for sec in WIDE_TIMES:
            pt = int(sec / TIME_BASE)
            video.seek(pt, backward=True, stream=stream)
            frame = next(f for f in video.decode(stream) if f.pts >= pt)
            if frame.pts != pt:
                raise ValueError(('Missing original exposure', pt))
            raw = frame.to_ndarray(format='bgr24')
            if sec == WIDE_TIMES[0]:
                reference = raw
            matches = []
            for x,y in STATIC_PATCHES:
                search = raw[y-25:y+26,x-31:x+31]
                template = reference[y-12:y+13,x-18:x+18]
                _,score,_,loc = cv2.minMaxLoc(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))
                matches.append(dict(uv=[x,y], ncc=score, shift_px=[loc[0]-13,loc[1]-13]))
            confirmed = sum(m['ncc']>.90 and max(map(abs,m['shift_px']))<=1 for m in matches)
            if sec in EMPTY_TIMES:
                cv2.imwrite(str(SCOUT/f'wide_{pt}.png'),raw)
            if confirmed >= 4:
                panel = cv2.resize(raw[230:465,220:1660], (864,141))
                rows.append(label(panel,f'{pt}   {sec}s   {confirmed}/7 static board regions agree'))
            records[str(pt)] = dict(pts=pt,source_seconds=sec,agreeing_regions=confirmed,
                                    frame_bgr_sha256=sha256(raw.tobytes()).hexdigest(),matches=matches)
    cv2.imwrite(str(SCOUT/'wide_hunt.jpg'), np.vstack(rows))
    write(SCOUT/'wide_camera_match.json', dict(
        source=str(MEDIA), source_sha256=read(WIDE_RECORD)['source_sha256'],
        registration_reference_pts=int(EMPTY_TIMES[0]/TIME_BASE), time_base=str(TIME_BASE), views=records))


def wide_remap(param, xy, *, inverse_map=False):
    """The existing wide-view plane homography and two-term division distortion."""
    H = np.r_[param[:8], 1.].reshape(3, 3)
    k1, k2 = param[8:10]
    if inverse_map:
        uv = (xy - WIDE_CENTRE) / WIDE_SCALE
        r2 = np.sum(uv**2, axis=1)
        xy = uv / (1 + k1*r2 + k2*r2*r2)[:,None]
        H = np.linalg.inv(H)
    q = np.c_[xy, np.ones(len(xy))] @ H.T
    q = q[:,:2] / q[:,2:]
    if inverse_map:
        return q
    r = np.linalg.norm(q, axis=1)
    d = r.copy()
    for _ in range(20):
        r2 = d*d
        derivative = 1 - r*(2*k1*d + 4*k2*d**3)
        f = d - r*(1 + k1*r2 + k2*r2*r2)
        d = np.clip(d - f / np.where(abs(derivative)>.05, derivative, np.inf), 0, 2.5)
    uv = q * np.divide(d, r, out=np.ones_like(d), where=r>1.e-9)[:,None]
    return uv * WIDE_SCALE + WIDE_CENTRE


def wide_preview():
    native = native_for(BASE, 'jin')
    k = [p for p,d in native.paths.items() if 'lane4' in p or 'lane4' in d['label'].lower()]
    k.extend(p for p,d in native.paths.items() if d['label'] in ('ice.F4:outer','ice.R4:outer'))
    path = ji.native_paths(native, k, count=241)
    pts = 55449600
    raw = cv2.imread(str(SCOUT/f'wide_{pts}.png'))
    guide = np.asarray(read(WIDE_DIR/'wide_ground.json')['views']['221460480']['calibration'])
    img = raw.copy()
    for name, (xy, _) in path.items():
        uv = wide_remap(guide, xy[:,:2])
        cv2.polylines(img, [np.rint(uv*16).astype('i4')], False, (5,200,255), 1, cv2.LINE_AA, 4)
        print(name, uv[[0,len(uv)//2,-1]].round(1).tolist())
    x = np.linspace(-.465,.365,1041)
    y = np.linspace(.246,.125,241)
    xx,yy = np.meshgrid(x,y)
    uv = wide_remap(guide, np.c_[xx.ravel(),yy.ravel()]).reshape(len(y),len(x),2).astype('f4')
    chart = cv2.remap(raw, uv, None, cv2.INTER_LINEAR)
    lum = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY).astype('f4')
    bg = cv2.GaussianBlur(lum, (0,0), 4, sigmaY=9)
    enhanced = cv2.cvtColor(np.clip(149 + (lum-bg)*13, 0, 255).astype('u1'), cv2.COLOR_GRAY2BGR)
    for y0 in np.arange(.135,.241,.005):
        row=round((y[0]-y0)/(y[0]-y[-1])*(len(y)-1))
        cv2.putText(enhanced,f'{y0:.3f}',(0,row+4),cv2.FONT_HERSHEY_SIMPLEX,.31,(20,70,218),1,cv2.LINE_AA)
    cv2.imwrite(str(SCOUT/'wide_nav_55449600.jpg'),np.vstack([raw[210:495,:],img[210:495,:]]))
    cv2.imwrite(str(SCOUT/'wide_strip_55449600.jpg'),np.vstack([chart,enhanced]))


def wide_calibration():
    """Refresh the wide-plane map with current geometry and measured non-Lane contours."""
    native = native_for(BASE, 'jin')
    wide = read(WIDE_DIR/'wide_ground.json')['views']
    obs = {}
    for key, uv in wide['221460480']['source_material'].items():
        if key in native.paths and ('curl' in native.paths[key]['label'].lower()
                or ':outer' in native.paths[key]['label']
                or any(text in native.paths[key]['label'].lower()
                       for text in ('goal_line', 'c:segment','b2:segment','bottom'))):
            obs[key] = np.array(uv, dtype='f8')
    # The right crease was unavailable at sufficient accuracy in the prior global fit.
    # Use actual outer paint flanks, transverse midlines and the floor/board edge.
    paths = ji.native_paths(native, list(obs), count=257)
    world = {key: xy[:, :2] for key, (xy, _) in paths.items()}
    held = [key for key in obs if native.paths[key]['label'] in ('ice.F3:outer','ice.R2:outer')]
    train = [key for key in obs if key not in held]
    if len(train) < 6 or len(held) != 2:
        raise ValueError((train, held, [native.paths[key]['label'] for key in obs]))

    def errors(p, key):
        poly = wide_remap(p, world[key])
        return nearest_valid_polyline_vectors(obs[key], poly, np.isfinite(poly).all(axis=1))[0]

    def residual(p):
        return np.concatenate([errors(p, key).ravel() * np.sqrt(40/len(obs[key])) for key in train])

    seed = np.array(wide['221460480']['calibration'])
    model = least_squares(residual, seed, loss='soft_l1', f_scale=2., max_nfev=65,
                           x_scale='jac', bounds=(np.r_[[-np.inf]*8,-.95,-.35],np.r_[[np.inf]*8,.75,.4]))
    stats = {key: dict(label=native.paths[key]['label'], group='held' if key in held else 'train',
                       p50=float(np.percentile(err,50)),p90=float(np.percentile(err,90)),
                       count=len(err)) for key in obs
             for err in [np.linalg.norm(errors(model.x,key),axis=1)]}
    print('wide-calibration', model.x.tolist(), stats, flush=True)
    write(SCOUT/'wide_calibration.json', dict(params=model.x.tolist(), stats=stats,
          old_source_calibration_pts=221460480, source_time_base='1/15360'))
    return model.x


def cap_support():
    """Audit observed parts of the right cap past the current upper endpoint."""
    source = OUT/'lane_motion_1118'/'reads'
    folder = WORK/'lane_cap_limit'
    folder.mkdir(parents=True, exist_ok=True)
    state = camera_state(BASE)
    native = native_for(BASE, 'jin')
    s = read(PREPARED/'setup.json')['shots']
    frames = {'okada_s02': [55.22,55.44,55.46,55.64], 'chen_s07': [59.52,59.70]}
    endpoints = np.array(native.paths[CAP]['parameters']).reshape(2,3)
    records = []
    for name, secs in frames.items():
        shot = s[name]
        z = arrays(source/(name+'.npz'))
        fitted = arrays(prepared_packet_path(name))
        if str(z['source_sha256']) != shot['source_sha256']:
            raise ValueError(('Source identity disagreement', name))
        close = z['lane'] == 3
        for pt, raw in source_frames(shot['source'], np.rint(np.array(secs)*12800)):
            take = close & (z['pts'] == pt)
            idx = int(np.flatnonzero(state[name+'_pts'] == pt)[0])
            if not np.all(z['index'][take] == idx):
                raise ValueError(('Source index disagreement', name, pt))
            uv = z['uv'][take]
            br = z['bracket'][take]
            if not len(uv) or not np.allclose(uv, br.mean(axis=1), atol=1.e-8):
                raise ValueError(('Missing or invalid opposite paint edges', name, pt))
            xy = inverse(native,state,name,shot,idx,uv)
            order = np.argsort(-xy[:,1])
            rs = order[(abs(xy[order,0]-.315)<.0025) & (xy[order,1]>.185)]
            desc = project(native,state,name,shot,idx,np.linspace(*endpoints,111))['uv']
            top = desc[0]
            direction = top-desc[4]
            direction /= np.linalg.norm(direction)
            rel = (uv[rs]-top)@direction
            if len(rs) < 4:
                continue
            # Resolve the raw pixels separately from the registered coordinate
            # limit and the station where the already-fitted short line ends.
            bx,by = np.maximum(np.floor(np.minimum(uv[rs].min(0),top)-[88,46]), [0,0]).astype(int)
            ex,ey = np.minimum(np.ceil(np.maximum(uv[rs].max(0),top)+[89,57]), [1920,1080]).astype(int)
            photo = raw[by:ey,bx:ex].copy()
            lum = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY).astype('f4')
            contrast = cv2.cvtColor(np.clip(142+15*(lum-cv2.GaussianBlur(lum,(0,0),6.5)),
                                           0,255).astype('u1'),cv2.COLOR_GRAY2BGR)
            zoom = 3
            photo = cv2.resize(photo,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC)
            contrast = cv2.resize(contrast,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC)
            marked = contrast.copy()
            loc = np.array([bx,by])
            pts_to_mark = (((br[rs]-loc)*zoom).round()).astype(int)
            for j,(a,b) in enumerate(pts_to_mark):
                color = (80,234,89) if rel[j]>0 else (225,150,31)
                cv2.circle(marked,tuple(a),2,color,1,cv2.LINE_AA)
                cv2.circle(marked,tuple(b),2,color,1,cv2.LINE_AA)
                cv2.circle(marked,tuple((a+b)//2),1,(255,255,255),-1,cv2.LINE_AA)
            top_px = np.rint((top-loc)*zoom).astype(int)
            cv2.drawMarker(marked,tuple(top_px),(50,58,240),cv2.MARKER_TILTED_CROSS,18,1,cv2.LINE_AA)
            cards = [
                label(photo,f'{name} {pt/12800:.3f}s   source crop x{zoom}'),
                label(contrast,'Local contrast'),
                label(marked,'Source cap flanks; green beyond current endpoint (red cross)'),
            ]
            image = folder/(f'{name}_{pt}.jpg')
            cv2.imwrite(str(image),np.vstack(cards),[cv2.IMWRITE_JPEG_QUALITY,96])
            same = (fitted['pts'] == pt) & (fitted['path'] == CAP)
            other_uv = fitted['uv'][same]
            duplicate = (np.linalg.norm(uv[rs,None,:]-other_uv[None,:,:],axis=2) < .005).any(axis=1)
            records.append(dict(shot=name,pts=pt,idx=idx,source_seconds=pt/12800,
                                source=shot['source'],source_sha256=shot['source_sha256'],
                                path=CAP, material='source_paired_cap_flanks', image=str(image),
                                source_packet=str(source/(name+'.npz')),
                                exact_prepared_pixel_duplicate=duplicate.tolist(),
                                prepared_same_source_cap_folds=sorted(set(fitted['fold'][same].tolist())),
                                uv=uv[rs].tolist(),bracket=br[rs].tolist(),registered_xy=xy[rs].tolist(),
                                contrast=z['contrast'][take][rs].tolist(),current_endpoint_uv=top.tolist(),
                                extension_along_cap_px=rel.tolist()))
    write(folder/'limit.json',dict(path=CAP,scene=BASE,current_endpoint_xyz=endpoints[0].tolist(),
            time_base='1/12800', pixel_domain='1920x1080 native distorted source',
            interpretation='可靠可读漆线的最高站位给封口上端 y 的条件下限。可读性随油漆、反光与遮挡变化；最高可见点不等同于实际接角。本审计包含曾入拟合的镜头和部分原像素，不能充作冻结留出或长边 UV。',
            views=records))


def transverse_brackets(raw, raw_guide, direction):
    """Find paired source flanks within a narrow path through accepted source points."""
    normal = np.array([-direction[1],direction[0]])
    along = np.arange(-120., 48., 3.)
    across = np.arange(-14.,14.5,.5)
    seed = raw_guide + along[:,None]*direction
    points = seed[:,None,:] + across[None,:,None]*normal
    taps = points[:,:,None,:] + np.array([-3.,0.,3.])[None,None,:,None]*direction
    maps = taps.reshape(len(along),-1,2).astype('f4')
    gray = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype('f4')
    f = cv2.remap(gray,maps,None,cv2.INTER_LINEAR).reshape(len(along),len(across),3).mean(2)
    f = cv2.GaussianBlur(f,(5,3),sigmaX=1.,sigmaY=.8)
    readings = []
    for i, profile in enumerate(f):
        if (taps[i]<[12,12]).any() or (taps[i]>[1908,1068]).any():
            continue
        g = np.gradient(profile)
        rise = find_peaks(g,height=.045)[0]
        fall = find_peaks(-g,height=.045)[0]
        options=[]
        for lo in fall:
            for hi in rise:
                if lo < 7 or hi+8 >= len(profile) or not 4 <= hi-lo <= 25:
                    continue
                if abs(across[[lo,hi]].mean()) > 4:
                    continue
                inside = float(np.mean(profile[lo+(hi-lo)//3:hi-(hi-lo)//3+1]))
                d = min(np.mean(profile[lo-7:lo-2]), np.mean(profile[hi+3:hi+8]))-inside
                if d >= .75:
                    options.append((d/(1+abs(across[[lo,hi]].mean())*.16),lo,hi,float(d)))
        if options:
            _,lo,hi,d = max(options)
            readings.append((i,points[i,[lo,hi]],d))
    if not readings:
        return np.empty((0,2,2)), np.empty(0)
    i,b,d = zip(*readings)
    keep = np.split(np.arange(len(i)), np.flatnonzero(np.diff(i)>2)+1)
    run = max(keep,key=len)
    if len(run) < 8:
        return np.empty((0,2,2)), np.empty(0)
    return np.asarray(b)[run],np.asarray(d)[run]


def transverse_support():
    """Re-read the visible upper reach of the independently named F crossbar."""
    folder = WORK/'lane_cross_limit'
    folder.mkdir(parents=True, exist_ok=True)
    state = camera_state(BASE)
    native = native_for(BASE, 'jin')
    shots = read(PREPARED/'setup.json')['shots']
    key = 'ice/lane4:F_transverse'
    end = np.asarray(native.paths[key]['parameters']).reshape(2,3)[0]
    dates = {'okada_s07':[1738240,1738752,1739008,1739264], 'jin_s10':[1126656,1129984]}
    records=[]
    for name, checks in dates.items():
        shot = shots[name]
        packet = arrays(prepared_packet_path(name))
        for pt, raw in source_frames(shot['source'],checks):
            idx = np.flatnonzero(state[name+'_pts']==pt)[0]
            m = (packet['path']==key)&(packet['pts']==pt)
            known_uv = packet['uv'][m]
            prior_xy = inverse(native,state,name,shot,idx,known_uv)
            good = (prior_xy[:,1]>.178)&(abs(prior_xy[:,0]+.232)<.008)
            uv0 = known_uv[good][np.argsort(-prior_xy[good,1])]
            if len(uv0)<3:
                continue
            normal = np.linalg.svd(uv0 - uv0.mean(0),full_matrices=False)[2][0]
            normal *= np.sign((uv0[0]-uv0[-1])@normal)
            points,contrast = transverse_brackets(raw,uv0[0],normal)
            uv = points.mean(1)
            xy = inverse(native,state,name,shot,idx,uv) if len(uv) else np.empty((0,2))
            fits = np.isfinite(xy).all(axis=1)&(abs(xy[:,0]+.232)<.01)
            if not fits.any():
                continue
            anchor = project(native,state,name,shot,idx,end.reshape(1,3))['uv'][0]
            lo = np.maximum(np.floor(uv[fits].min(axis=0)-[46,35]),0).astype(int)
            hi = np.minimum(np.ceil(uv[fits].max(axis=0)+[48,35]),[1920,1080]).astype(int)
            raw_crop = raw[lo[1]:hi[1],lo[0]:hi[0]]
            lum = cv2.cvtColor(raw_crop,cv2.COLOR_BGR2GRAY).astype('f4')
            edge_view = cv2.cvtColor(np.clip(146+16*(lum-cv2.GaussianBlur(lum,(0,0),7)),
                                             0,255).astype('u1'),cv2.COLOR_GRAY2BGR)
            zoom = 3
            photo = cv2.resize(raw_crop,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC)
            edge_view = cv2.resize(edge_view,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC)
            overlay = edge_view.copy()
            for br,y in zip(points[fits],xy[fits,1],strict=True):
                color = (50,230,80) if y > end[1] else (250,170,37)
                for p in br:
                    cv2.circle(overlay,tuple(np.rint((p-lo)*zoom).astype(int)),2,color,1,cv2.LINE_AA)
            cv2.drawMarker(overlay,tuple(np.rint((anchor-lo)*zoom).astype(int)),(50,55,243),
                           cv2.MARKER_TILTED_CROSS,16,1,cv2.LINE_AA)
            image = folder/f'{name}_{pt}_paired.jpg'
            rows = [label(photo,f'{name} {pt/12800:.3f}s   source crop x{zoom}'),
                    label(edge_view,'Local contrast'),
                    label(overlay,'Source crossbar flanks beyond current model endpoint (red cross)')]
            cv2.imwrite(str(image),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,97])
            records.append(dict(shot=name,pts=pt,idx=int(idx),source_seconds=pt/12800,
                                source=shot['source'],source_sha256=shot['source_sha256'],
                                original_prepared_points=uv0.tolist(),source_prepared=str(prepared_packet_path(name)),
                                uv=uv[fits].tolist(),bracket=points[fits].tolist(),
                                registered_xy=xy[fits].tolist(),contrast=contrast[fits].tolist(),
                                current_endpoint_uv=anchor.tolist(),crop=str(image)))
    write(folder/'limit.json',dict(path=key,scene=BASE,current_endpoint_xyz=end.tolist(),
          time_base='1/12800',pixel_domain='1920x1080 native distorted source',views=records,
          interpretation='同源原片窄条搜索中的横段双沿，仅支撑原片可见范围和已知接边拓扑。旧源点帮助寻找物理横段，照明或压缩均可能造成端部渐隐，最后可读点不能当作实际接角。'))


def scout():
    """Rank original-frame rectifications for manual reading of paint identity."""
    candidates = defaultdict(list)
    for file in sorted((WORK / 'survey').glob('*.npz')):
        programme = file.stem.split('_')[0]
        z = arrays(file)
        for region, (lo, hi, centre) in REGIONS.items():
            widths = np.linspace(lo, hi, max(2, round((hi - lo) / .055) + 1))
            for left, right in zip(widths[:-1], widths[1:]):
                xs = (X >= left) & (X <= right)
                for idx, pt, frame in zip(z['indices'], z['pts'], z['survey'], strict=True):
                    data = frame[STRIP_ROWS][:, xs, 0].astype(np.float32) / 4
                    data[data == -32] = np.nan
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        profile = np.nanmedian(data, axis=1)
                    valid = np.sum(np.isfinite(data), axis=1) >= max(9, int(xs.sum() * .60))
                    if valid.sum() < 15:
                        continue
                    ys = Y[STRIP_ROWS]
                    peaks, props = find_peaks(np.nan_to_num(profile), prominence=.25)
                    if not len(peaks):
                        continue
                    best = None
                    for iy, prominence in zip(peaks, props['prominences'], strict=True):
                        if iy < 6 or iy + 7 >= len(ys) or not valid[iy-6:iy+7].all():
                            continue
                        delta = data[iy-1:iy+2].mean(axis=0) - np.maximum(
                            data[iy-6:iy-3].mean(axis=0), data[iy+4:iy+7].mean(axis=0))
                        support = np.mean(np.nan_to_num(delta) > .25)
                        strength = min(profile[iy], prominence) * support**.65
                        score = strength / (1 + ((ys[iy] - centre) / .005)**2)
                        if score < .20 or (best is not None and score <= best[0]):
                            continue
                        best = (score, float(ys[iy]))
                    if best is not None:
                        candidates[(programme, region)].append(
                            (best[0], file.stem, int(pt), int(idx), float(left), float(right), best[1], data))
    SCOUT.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for (programme, region), found in candidates.items():
        accepted = []
        for candidate in sorted(found, key=lambda c: -c[0]):
            score, shot, pt, idx, lo, hi, y, data = candidate
            if sum(r[1] == shot for r in accepted) >= 4 or any(
                r[1] == shot and abs(r[2] - pt) < 12800 and abs(r[4] - lo) < .01 for r in accepted
            ):
                continue
            accepted.append(candidate)
            if len(accepted) == 16:
                break
        panels = []
        for j, (score, shot, pt, idx, lo, hi, y, data) in enumerate(accepted):
            key = f'{programme}/{region}/{j}'
            metadata[key] = dict(shot=shot, pts=pt, idx=idx, x0=lo, x1=hi, y=y, score=float(score))
            lum = np.clip(193 - np.nan_to_num(data) * 37, 10, 235).astype('u1')
            img = cv2.cvtColor(lum, cv2.COLOR_GRAY2BGR)
            img[~np.isfinite(data)] = (80, 85, 105)
            img = cv2.resize(img, (378, 79), interpolation=cv2.INTER_NEAREST)
            panel = np.full((105, 400, 3), 37, dtype='u1')
            panel[22:101, 20:398] = img
            iy = round(22 + (Y_MAX-y) / (Y_MAX-Y_MIN) * 78)
            cv2.line(panel, (9, iy), (18, iy), (20, 210, 255), 1)
            label = f'{j:02} {shot} {pt / 12800:.2f}s {lo:+.3f}..{hi:+.3f} {y:.4f}'
            cv2.putText(panel, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, .41, (240,)*3, 1, cv2.LINE_AA)
            panels.append(panel)
        for _ in range((-len(panels)) % 2):
            panels.append(np.zeros_like(panels[0]))
        if panels:
            cv2.imwrite(str(SCOUT / f'{programme}_{region}.jpg'),
                        np.vstack([np.hstack(panels[i:i+2]) for i in range(0, len(panels), 2)]))
    write(SCOUT / 'candidates.json', metadata)


def extract_strip(native, state, name, shot, idx, raw, x0, x1, y0, y1):
    xs = np.linspace(x0, x1, 321)
    ys = np.linspace(y1, y0, 129)
    xgrid, ygrid = np.meshgrid(xs, ys)
    grid = np.stack((xgrid, ygrid, np.zeros_like(xgrid)), axis=-1).reshape(-1, 3)
    result = project(native, state, name, shot, idx, grid)
    uv = result['uv'].reshape(129, 321, 2).astype('f4')
    invalid = (result['status'].reshape(129, 321) != 0) | ~np.isfinite(uv).all(axis=-1)
    uv[invalid] = -1
    return cv2.remap(raw, uv, None, cv2.INTER_LINEAR), uv


def label(img, label):
    card = cv2.copyMakeBorder(img, 19, 0, 0, 0, cv2.BORDER_CONSTANT, value=(31, 31, 31))
    cv2.putText(card, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (236,)*3, 1, cv2.LINE_AA)
    return card


def higher_resolution(native, state, shots):
    ranges = {
        'left_hi': (-.439, -.389, .2175),
        'transverse_hi': (-.260, -.201, .216),
        'middle_hi': (-.020, .080, .214),
        'right_hi': (.265, .332, .212),
    }
    stations = [(a+b)*.5 for a,b,_ in ranges.values()]
    probes = np.array([[[x,y,0], [x+.015,y,0], [x,y+.006,0]]
                       for x, (_,_,y) in zip(stations, ranges.values(), strict=True)]).reshape(-1,3)
    candidates = defaultdict(list)
    for name, shot in shots.items():
        z = arrays(WORK/'survey'/(name+'.npz'))
        for idx, pt in zip(z['indices'], z['pts'], strict=True):
            p = project(native, state, name, shot, idx, probes)
            for region, uv, status in zip(ranges, p['uv'].reshape(-1,3,2), p['status'].reshape(-1,3), strict=True):
                if status.any() or not ((uv > [50,305]) & (uv < [1870,1010])).all():
                    continue
                tangent = uv[1]-uv[0]
                normal = np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
                scale = abs((uv[2]-uv[0]) @ normal) / .006
                candidates[shot['program'],region].append((scale, name, int(idx),int(pt)))
    result = {}
    for (programme, region), rows in candidates.items():
        accepted = []
        for row in sorted(rows, reverse=True):
            scale,name,idx,pt = row
            same_shot = [other for other in accepted if other[1] == name]
            if len(same_shot) >= 2 or any(abs(pt-other[3]) < 25600 for other in same_shot):
                continue
            x0,x1,y = ranges[region]
            result[f'{programme}/{region}/{len(accepted)}'] = dict(
                shot=name,idx=idx,pts=pt,x0=x0,x1=x1,y=y,score=float(scale))
            accepted.append(row)
            if len(accepted) == 7:
                break
    return result


def preview():
    """Render original pixels and horizontal-coherence views with a separate ruler."""
    state = camera_state(BASE)
    native = native_for(BASE, 'jin')
    shots = read(PREPARED / 'setup.json')['shots']
    cards = defaultdict(list)
    cases = defaultdict(dict)
    reads = higher_resolution(native, state, shots)
    write(SCOUT / 'high_resolution.json', reads)
    for key, item in reads.items():
        cases[item['shot']].setdefault(item['pts'], []).append((key, item))
    for name, lookup in cases.items():
        shot = shots[name]
        for pt, raw in source_frames(shot['source'], sorted(lookup)):
            for key, item in lookup[pt]:
                x0, x1 = item['x0'], item['x1']
                y0, y1 = .185, .234
                img, uv = extract_strip(native, state, name, shot, item['idx'], raw, x0, x1, y0, y1)
                lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('f4')
                residue = cv2.GaussianBlur(lum, (0,0), 10, sigmaY=1.3) - cv2.GaussianBlur(
                    lum, (0,0), 10, sigmaY=8)
                rb = img[:,:,2].astype('f4') - img[:,:,0].astype('f4')
                chroma = cv2.GaussianBlur(rb, (0,0), 10, sigmaY=1.3) - cv2.GaussianBlur(
                    rb, (0,0), 10, sigmaY=8)
                contrast = np.clip(154 + residue[:,:,None] * 26
                    + chroma[:,:,None] * [-14,0,14], 2, 252).astype('u1')
                black = (img == 0).all(axis=-1)
                contrast[black] = 0
                ruler = np.zeros((129, 44, 3), dtype='u1')
                for y in np.arange(.19,.231,.005):
                    row = round((y1-y)/(y1-y0)*128)
                    cv2.line(ruler, (37, row), (43, row), (99, 184, 218), 1)
                    cv2.putText(ruler, f'{y:.3f}', (1,row+3), cv2.FONT_HERSHEY_SIMPLEX, .29, (238,)*3, 1)
                good = np.isfinite(uv).all(-1) & (uv[:,:,0]>0) & (uv[:,:,0]<1920)
                good &= (uv[:,:,1]>0) & (uv[:,:,1]<1080)
                xx, yy = uv[good].T
                ul = np.maximum(np.floor([xx.min()-35, yy.min()-30]), [0,0]).astype(int)
                br = np.minimum(np.ceil([xx.max()+36, yy.max()+31]), [1920,1080]).astype(int)
                crop = raw[ul[1]:br[1],ul[0]:br[0]]
                scale = min(455/crop.shape[1], 129/crop.shape[0])
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                border = cv2.copyMakeBorder(crop, 0, 129-crop.shape[0], 0, 455-crop.shape[1],
                                            cv2.BORDER_CONSTANT, value=(15, 15, 15))
                panels = np.hstack([border, img, ruler, contrast])
                case = key.split('/')
                cards[tuple(case[:2])].append((int(case[-1]), label(panels,
                    f'{key}   {name}   {pt / 12800:.3f} s    x={x0:.3f}..{x1:.3f}     '
                    f'source crop  |  rectified source  |  contrast enhanced')))
    for (programme, region), rows in cards.items():
        rows.sort(key=lambda row: row[0])
        for page, start in enumerate(range(0, len(rows), 5)):
            cv2.imwrite(str(SCOUT / f'raw_{programme}_{region}_{page}.jpg'),
                        np.vstack([p for _, p in rows[start:start+5]]), [cv2.IMWRITE_JPEG_QUALITY, 96])


def summary():
    shots = read(PREPARED/'setup.json')['shots']
    scout_counts = Counter(key.split('/')[0] for key in read(SCOUT/'candidates.json'))
    highs = read(SCOUT/'high_resolution.json')
    counts = Counter(key.split('/')[0] for key in highs)
    matched = read(SCOUT/'wide_camera_match.json')
    calibration = read(SCOUT/'wide_calibration.json')
    limit_path = WORK/'lane_cap_limit'/'limit.json'
    caps = read(limit_path)
    cross_path = WORK/'lane_cross_limit'/'limit.json'
    crossbars = read(cross_path)
    sources = {shot['program']: dict(source=shot['source'], source_sha256=shot['source_sha256'],
                    time_base='1/12800', original_size=[1920, 1080]) for shot in shots.values()}
    examples = [
        ('jin_s08', 852224, '../inspection/jin_s08_852224_R.jpg', '旧长边点落入 R4 圈带。'),
        ('nayeon_s04', 869888, '../inspection/nayeon_s04_869888_R.jpg', '旧长边点落入 R4 圈带。'),
        ('jin_s11', 1182208, 'raw_jin_left_0.jpg', '左段主峰随蓝色广告倒影斜穿。'),
        ('okada_s01', 443136, '../inspection/okada_s01_443136_left.jpg', '左段无可读连续双沿，蓝色广告倒影较强。'),
        ('okada_s04', 937728, '../inspection/okada_s04_937728_mid.jpg', '中段有冰刀或计时牌遮挡，可辨处缺少稳定双沿。'),
        ('okada_s02', 704000, 'junction_okada_s02.jpg', '封口上方弱亮度带在粉色广告倒影下，无法排除反光。'),
        ('okada_s03', 730112, 'junction_okada_s03.jpg', '最强候选斜穿，鞋与围挡反光干扰。'),
        ('chen_s07', 764160, 'cap_chen_s07.jpg', '可见封口短线的部分两沿，上接长边的两沿无法独立确定。'),
        ('chen_s12', 1506816, 'junction_chen_s12.jpg', '右端强峰受画面边界、计时牌与斜向反光影响。'),
    ]
    report = dict(
        path=SIDE, material='traced_lane4_upper', baseline=BASE, admitted_rows=0,
        scope='四套单节目和同场全场高机位的源图可辨性审阅。没有通过实体身份及双沿准入的长边原始像素。',
        single_programmes={p: dict(**s, ranked_windows=scout_counts[p], high_resolution_windows=counts[p],
                                      admitted_rows=0)
                            for p,s in sources.items()},
        original_examples=[dict(shot=s,pts=pt,source_seconds=pt/12800,
                            image=str(SCOUT/file),disposition=reason) for s,pt,file,reason in examples],
        wide=dict(source=matched['source'],source_sha256=matched['source_sha256'],time_base=matched['time_base'],
                  empty_ice=[dict(**matched['views'][str(int(sec/TIME_BASE))],
                    image=str(SCOUT/f'wide_{int(sec/TIME_BASE)}.png')) for sec in EMPTY_TIMES],
                  calibration=calibration, agreement=str(SCOUT/'wide_camera_match.json'),
                  image_comparison=str(SCOUT/'wide_hunt.jpg'),
                  limitation='远侧 LANE4 上沿无稳定可追的双沿，广告反射的强峰会斜穿。空冰是额外曝光和视角，三个时刻共用一台机位，不能当三个独立机位。'),
        cap_support=dict(file=str(limit_path), distinct_source_exposures=len(caps['views']),
                         maximum_registered_y=max(view['registered_xy'][0][1] for view in caps['views']),
                         current_endpoint_y=caps['current_endpoint_xyz'][1],
                         interpretation=caps['interpretation']),
        transverse_support=dict(file=str(cross_path), distinct_source_exposures=len(crossbars['views']),
                                current_endpoint_y=crossbars['current_endpoint_xyz'][1],
                                maxima_per_exposure=[
                                    dict(shot=view['shot'],pts=view['pts'],
                                         max_registered_y=max(p[1] for p in view['registered_xy']))
                                    for view in crossbars['views']],
                                interpretation=crossbars['interpretation']),
        prior_review=[
            dict(path=str(EVENT/'model_constraints'/'lane_transverse_20261029.json'),
                 disposition='用户确认四条横线的实体、共线和平行关系及接边拓扑，未测定长边坐标。'),
            dict(path=str(EVENT/'programmes/okada_mei_sp/s04_lane_identity_review_20261027.json'),
                 disposition='多处历史长边取点落入 R4 圈带，已按原始像素保留撤回。'),
        ],
        holdout=dict(admitted_rows=0,
                     limitation='未建立 LANE4 上长边的独立像素留出。宽机位 R2/F3 留出只检验宽机位校准。'))
    write(OUTPUT/'summary.json', report)
    print('LANE4 audit', dict(counts), 'admitted', report['admitted_rows'], flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    actions = {'scout': scout, 'preview': preview, 'wide-frames': wide_frames,
               'wide-preview': wide_preview, 'wide-calibration': wide_calibration,
               'cap-support': cap_support, 'transverse-support': transverse_support, 'summary': summary}
    parser.add_argument('action', choices=actions)
    actions[parser.parse_args().action]()
