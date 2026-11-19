"""Check the two right-hand S22 emblems with original-image registration."""
import cv2
import numpy as np

from board_displacement_1118 import TARGET
from feedback_source import contour_samples, read_contour, selected_paths, signals
from lane_end_evidence import foreground, matte
from review import camera_state, ji, native_for, source_frames
from shared import OUT, PREPARED, arrays, read, save, sha, write


CANDIDATE = 'lane4_side_1118/shield_lens'
REF_NAME = 'nayeon_s04'
REF_PTS = 813568
TARGET_NAME = 'jin_s22'
STAMPS = (154.98, 155., 155.02, 155.04, 155.06, 155.08)
KEYS = {
    'Ad25': 'print_instance/Ad25::Ad12.shield_print_component001',
    'Ad26': 'print_instance/Ad26::Ad13.print.outer_circle_print_stroke',
}
LOCAL_TEMPLATE = {'Ad25': (1820, 331), 'Ad26': (1688, 320)}
PROTECTION_TIMES = (154.96, 155., 155.04, 155.06, 155.08)
PROTECTION_GUIDES = {
    'Ad1': ((1755,334), (1669,334), (1590,334), (1558,334), (1533,335)),
    'Ad2': ((1345,323), (1280,324), (1214,325), (1186,326), (1158,326)),
    'Ad4': ((617,315), (552,318), (494,320), (466,321), (436,323)),
}


def channels(frame, family):
    b, g, r = np.moveaxis(frame.astype('f4'), 2, 0)
    chroma = b-(r+g)/2. if family == 'Ad25' else r-(b+g)/2.
    return {'dark': -cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('f4'), 'ink_color': chroma}


def source_shift(reference, current, x, y, w, h):
    a = reference[y-h//2:y-h//2+h, x-w//2:x-w//2+w]
    l, t = x-70, y-36
    b = current[t:y+43, l:]
    surface = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (dx, dy) = cv2.minMaxLoc(surface)
    row = surface[dy].copy()
    row[max(0, dx-8):dx+9] = -1.
    return dict(shift_xy_px=[int(l+dx-(x-w//2)), int(t+dy-(y-h//2))],
        correlation=float(peak), horizontal_peak_gap=float(peak-row.max()),
        search_touches_side=dx in (0, surface.shape[1]-1))


def projection(native, base, setup, name, index, world):
    s = setup[name]
    result = native.core.project_motion(base[name+'_rigs'][index], base[name+'_controls'],
        s['first'], s['dt'], 1080, base[name+'_times'][index], world)
    result['uv'][result['status'] != 0] = np.nan
    return result['uv'], result['status']


def isolated_ink(src, native_poly, shift):
    polygon = np.rint(native_poly+shift).astype('i4')
    mask = np.zeros_like(src, 'u1')
    cv2.fillPoly(mask, [polygon], 1)
    mask = cv2.dilate(mask, np.ones((5, 5), 'u1'))
    x0, y0 = polygon.min(0)
    _, y1 = polygon.max(0)
    neutral = src[y0-2:y1+3, x0-18:x0-7]
    return np.maximum(0., src-float(np.median(neutral)))*mask


def isolated_registration(ref_shape, current_shape, ink, current, variant):
    a = np.floor(current_shape.min(0)).astype(int) - [44, 8]
    b = np.ceil(current_shape.max(0)).astype(int) + [45, 9]
    left = np.maximum([0, 0], a-[35, 12])
    right = np.minimum([1920, 1080], b+[43, 13])
    width, height = b-a
    if ((right-left) <= [width, height]).any():
        return None
    transform, _ = cv2.findHomography(ref_shape, current_shape, method=0)
    if transform is None:
        return None
    local = np.array([[1., 0., -a[0]], [0., 1., -a[1]], [0., 0., 1.]])@transform
    plane = cv2.warpPerspective(ink, local, (width,height), flags=cv2.INTER_LINEAR)
    match = current[left[1]:right[1], left[0]:right[0]]
    if variant == 'edge_bands':
        # The official mark lies on white board. Its name and the nearest
        # logo are outside the top and bottom registration bands.
        rows = np.arange(height)
        block = (rows >= 8+height*.34) & (rows <= 8+height*.54)
        plane[block] = 0
    best = None
    for blur in (21, 33, 45, 57, 69, 81):
        pattern = cv2.blur(plane, (blur, 1))
        res = cv2.matchTemplate(match, pattern, cv2.TM_CCOEFF_NORMED)
        _, strength, _, (dx, dy) = cv2.minMaxLoc(res)
        if best is None or strength > best[0]:
            best = strength, dx, dy, blur, res
    strength, dx, dy, blur, surface = best
    row = surface[dy].copy()
    row[max(0, dx-8):dx+9] = -1.
    return dict(shift_xy_px=(left+np.array([dx, dy])-a).tolist(), correlation=float(strength),
        horizontal_peak_gap=float(strength-row.max()), horizontal_blur_px=blur,
        search_touches_side=dx in (0, surface.shape[1]-1))


def side_intersection(bottom, side):
    if min(len(bottom), len(side)) < 5:
        return None
    right = np.median(side[:, 0])
    bottom = bottom[(bottom[:, 0] > right-235) & (bottom[:, 0] < right+7)]
    if len(bottom) < 5 or np.ptp(bottom[:, 0]) < 33 or np.ptp(side[:, 1]) < 14:
        return None
    hb = np.polyfit(bottom[:, 0], bottom[:, 1], 1)
    vr = np.polyfit(side[:, 1], side[:, 0], 1)
    eb = abs(bottom[:, 1]-np.polyval(hb,bottom[:, 0]))
    er = abs(side[:, 0]-np.polyval(vr,side[:, 1]))
    scatter = [float(np.quantile(eb,.9)), float(np.quantile(er,.9))]
    if max(scatter) > 2.9:
        return None
    x = (vr[0]*hb[1]+vr[1])/(1.-vr[0]*hb[0])
    y = np.polyval(hb,x)
    if min(abs(bottom[:, 0]-x)) > 120 or y-side[:, 1].max() > 20 or y < side[:, 1].min():
        return None
    return np.array([x, y]), scatter


def protection(native, state, setup):
    source_file = OUT/'lane4_side_1118/jin_source/jin_s22.npz'
    measured = arrays(source_file)
    name = TARGET_NAME
    shot = setup[name]
    assert measured['source_sha256'].item() == shot['source_sha256']
    pts = state[name+'_pts']
    records = []
    for family in ('Ad1', 'Ad2', 'Ad4'):
        keys = {suffix: next(k for k in native.paths
                            if k.startswith(f'print_instance/{family}::') and suffix in k)
                for suffix in ('.bottom_edge', '.right_vertical')}
        bottom, side = list(keys.values())
        table = ji.native_paths(native, keys.values(), count=1)
        b, st = table[bottom]
        edge = table[side][0]
        endpoints = np.array([0, len(b)-1])
        separation = np.linalg.norm(b[endpoints,None]-edge[None,[0,-1]], axis=2)
        col = int(np.unravel_index(np.argmin(separation), separation.shape)[0])
        if separation.min() > .00001:
            continue
        fixed = endpoints[col]
        for pt in sorted(set(measured['pts'][measured['family']==family])):
            if pt < round(154.90*12800):
                continue
            match = measured['pts'] == pt
            a = measured['uv'][match & (measured['path']==bottom)]
            z = measured['uv'][match & (measured['path']==side)]
            found = side_intersection(a,z)
            if found is None:
                continue
            uv, scatter = found
            i = np.searchsorted(pts,pt)
            predicted, _ = projection(native,state,setup,name,i,b[fixed:fixed+1])
            records.append(dict(pts=int(pt),idx=int(i),path=bottom,family=family,
                material='source_plate_lower_right',uv=uv,station=float(st[fixed]),station_fixed=True,
                source_line_scatter_p90_px=scatter,projected_uv=predicted[0]))
    return source_file, records


def chroma_midpoint(profile, start):
    smooth = cv2.GaussianBlur(profile[None, :].astype('f4'), (0,0), 1.8).ravel()
    n = len(smooth)
    before = np.median(smooth[n//10:n//4])
    after = np.median(smooth[-n//4:-n//10])
    level = (before+after)*.5
    if before-after < 10.:
        return np.nan
    hits = np.flatnonzero((smooth[:-1] >= level) & (smooth[1:] < level))
    if not len(hits):
        return np.nan
    i = hits[np.argmin(abs(hits-(n-1)/2))]
    return float(start+i+(smooth[i]-level)/(smooth[i]-smooth[i+1]))


def color_protection(native, state, setup):
    name = TARGET_NAME
    pts = state[name+'_pts']
    shot = setup[name]
    raw_pts = [round(t*12800) for t in PROTECTION_TIMES]
    measurements, profiles = [], []
    for i, (pt, raw) in enumerate(source_frames(shot['source'], raw_pts)):
        b, g, r = np.moveaxis(raw.astype('f4'), 2, 0)
        chroma = {'Ad1': b-r, 'Ad2': (b+r)*.5-g, 'Ad4': b-r}
        for family, locations in PROTECTION_GUIDES.items():
            x, y = locations[i]
            color = chroma[family]
            sid = [(v, chroma_midpoint(color[v-1:v+2,x-65:x+66].mean(0), x-65))
                   for v in np.arange(y-22, y-8, 3)]
            bot = [(u, chroma_midpoint(color[y-15:y+18,u-4:u+5].mean(1), y-15))
                   for u in np.arange(x-175,x-48,11)]
            sid, bot = (np.asarray(p)[np.isfinite(np.asarray(p)[:,1])] for p in (sid, bot))
            if len(sid) < 5 or len(bot) < 8:
                continue
            a, b0 = np.polyfit(sid[:,0], sid[:,1], 1)
            c, d = np.polyfit(bot[:,0], bot[:,1], 1)
            uv = np.array([(b0+a*d)/(1.-a*c),(d+b0*c)/(1.-a*c)])
            scatter = (np.quantile(abs(sid[:,1]-np.polyval((a,b0), sid[:,0])), .9),
                       np.quantile(abs(bot[:,1]-np.polyval((c,d), bot[:,0])), .9))
            if max(scatter) > 3.4 or (abs(uv-[x,y]) > [18,9]).any():
                continue
            key = next(k for k in native.paths if k.startswith(f'print_instance/{family}::') and '.bottom_edge' in k)
            # Each lower edge traverses from the source-right corner. Verify
            # its endpoint against the independently typed right vertical.
            side_key = next(k for k in native.paths if k.startswith(f'print_instance/{family}::') and '.right_vertical' in k)
            paths = ji.native_paths(native, [key, side_key], count=1)
            xyz = paths[key][0][:1]
            side = paths[side_key][0][[0,-1]]
            if np.linalg.norm(side-xyz, axis=1).min() > .00001:
                raise ValueError((family, 'Plate corner is inconsistent across its native paths'))
            frame = np.searchsorted(pts, pt)
            prediction, _ = projection(native, state, setup, name, frame, xyz)
            measurements.append(dict(pts=int(pt), idx=int(frame), path=key, family=family, station=0.,
                station_fixed=True, uv=uv, material='source_plate_chroma_lower_right', fold=0,
                scaffold_uv=prediction[0], source_line_scatter_p90_px=scatter))
            profiles.append(dict(pts=int(pt), family=family, vertical_chroma_midpoints=sid.tolist(),
                horizontal_chroma_midpoints=bot.tolist(), measured_lower_right=uv.tolist(),
                source_minus_current_px=(uv-prediction[0]).tolist(), projected_uv=prediction[0].tolist()))
    return measurements, profiles


def run():
    cv2.setNumThreads(1)
    setup = read(PREPARED/'setup.json')['shots']
    native = native_for(CANDIDATE, 'jin')
    state = camera_state(CANDIDATE)
    ref = setup[REF_NAME]
    pts = state[REF_NAME+'_pts']
    rid = np.searchsorted(pts, REF_PTS)
    _, rgb = next(source_frames(ref['source'], [REF_PTS]))
    mask = foreground(ref, matte(ref, pts), rid, REF_PTS)
    references = {}
    paths = selected_paths(native)
    tab = ji.native_paths(native, KEYS.values(), count=193)
    for family, key in KEYS.items():
        ref_poly, status = projection(native, state, setup, REF_NAME, rid, tab[key][0])
        parts = contour_samples({key: (ref_poly, status)}, tab, paths[family], family)
        hits = read_contour(signals(rgb), mask, parts, family)
        shift = np.median([h[6] for h in hits], axis=0)
        photometry = {k: isolated_ink(v,ref_poly,shift) for k,v in channels(rgb,family).items()}
        references[family] = (ref_poly+shift, photometry, len(hits), shift)
    pts = state[TARGET_NAME+'_pts']
    shot = setup[TARGET_NAME]
    examples = dict(source_frames(shot['source'], [round(t*12800) for t in STAMPS]))
    out = {}
    for family, key in KEYS.items():
        ref_shape, ink, count, shift = references[family]
        x, y = LOCAL_TEMPLATE[family]
        anchor = channels(examples[round(STAMPS[-1]*12800)], family)
        notes = []
        for pt, raw in examples.items():
            i = np.searchsorted(pts, pt)
            poly, _ = projection(native, state, setup, TARGET_NAME, i, tab[key][0])
            raw_signals = channels(raw, family)
            look = dict(pts=pt, seconds=pt/12800., projected_box_px=[poly.min(0).tolist(), poly.max(0).tolist()],
                        detector_visible=bool((poly[:, 0] < 1920).any()))
            if pt != max(examples):
                look['relative_to_last'] = {signal: [source_shift(anchor[signal], raw_signals[signal], x, y, w, h)
                    for w,h in ((52,36),(60,24),(44,44),(42,32),(46,26))] for signal in anchor}
            if (family == 'Ad26' and pt >= round(155.02*12800)) or pt == max(examples):
                look['isolated_registration'] = {
                    signal: {variant: isolated_registration(ref_shape, poly, ink[signal], raw_signals[signal], variant)
                        for variant in ('whole_emblem', 'edge_bands')} for signal in anchor}
            notes.append(look)
        out[family] = dict(reference_source_hits=count, reference_registration_offset_px=shift.tolist(),
                          exposures=notes)
    directory = TARGET/'s22_right_print'
    write(directory/'source_checks.json', dict(candidate=CANDIDATE,
        candidate_state_sha256=sha(OUT/CANDIDATE/'state.npz'), source=shot['source'],
        source_sha256=shot['source_sha256'], reference_source=ref['source'],
        reference_source_sha256=ref['source_sha256'], reference_pts=REF_PTS, ads=out))
    path, rows = protection(native, state, setup)
    if rows:
        save(directory/'plate_corners_review.npz',
            **{k: np.asarray([r[k] for r in rows]) for k in rows[0]},
            source_file=str(path), source_file_sha256=sha(path), source_sha256=shot['source_sha256'])
    direct, profiles = color_protection(native, state, setup)
    if direct:
        save(directory/'chroma_protection.npz',
            **{k: np.asarray([r[k] for r in direct]) for k in direct[0]},
            source_sha256=shot['source_sha256'], source=shot['source'])
        write(directory/'chroma_protection.json', dict(rows=profiles, source=shot['source'],
            source_sha256=shot['source_sha256'], registration='Lower-right intersection of two measured plate color midpoints',
            limitations='Color spread and extrapolation under horizontal motion blur affect the inferred corner position.'))


if __name__ == '__main__':
    run()
