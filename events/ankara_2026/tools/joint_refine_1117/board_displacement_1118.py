"""Track static board print and ice paint around flagged Ankara exposures."""
from argparse import ArgumentParser
from collections import Counter, defaultdict, deque

import cv2
import numpy as np

from flow_depths import bind
from lane_end_evidence import foreground, matte
from lane_side_1118 import BASE, WORK
from review import camera_state, native_for, source_frames
from shared import PREPARED, arrays, read, save, sha, write


TARGET = WORK / 'motion_evidence'
X = [0, 480, 960, 1440, 1760, 1919]
LAGS = (1, 2, 4, 8)
# Rail readings from the original frames delimit the print search regions.
# Interior guards keep the search pixels clear of both rail transitions.
RAILS = {
    'jin_s14': [
        (122.84, [393, 394, 394, 394, 398, 399], [455, 447, 432, 426, 425, 425]),
        (122.98, [390, 389, 390, 391, 394, 395], [446, 437, 423, 418, 419, 423]),
        (123.06, [378, 378, 378, 381, 385, 386], [439, 429, 416, 414, 418, 420]),
        (123.30, [366, 367, 368, 369, 384, 399], [414, 410, 404, 415, 435, 465]),
    ],
    'jin_s22': [
        (154.72, [213, 222, 227, 232, 237, 238], [264, 274, 280, 290, 299, 301]),
        (154.94, [286, 285, 285, 290, 295, 300], [319, 322, 329, 334, 342, 352]),
        (155.06, [288, 289, 291, 296, 302, 310], [333, 338, 340, 344, 353, 379]),
        (155.08, [289, 290, 291, 297, 304, 311], [334, 338, 341, 346, 364, 386]),
    ],
}
S22_ONSET = [
    (154.56, [203, 202, 205, 211, 219, 222], [245, 242, 255, 270, 280, 285]),
    (154.60, [207, 206, 210, 217, 223, 226], [246, 245, 258, 274, 283, 288]),
    (154.64, [212, 211, 215, 222, 229, 232], [249, 250, 263, 280, 292, 298]),
    (154.68, [216, 215, 220, 228, 235, 238], [253, 255, 268, 286, 294, 300]),
    (154.70, [219, 218, 221, 228, 236, 239], [254, 258, 271, 286, 295, 303]),
    RAILS['jin_s22'][0],
    (154.80, [238, 245, 250, 256, 262, 264], [281, 292, 302, 313, 322, 328]),
]
S14_EXTENDED = [
    (122.14, [412, 410, 409, 411, 415, 419], [478, 474, 463, 448, 441, 444]),
    (122.34, [424, 424, 424, 426, 430, 434], [489, 483, 471, 455, 453, 456]),
    (122.54, [414, 414, 415, 417, 418, 420], [482, 473, 462, 447, 441, 442]),
    (122.74, [401, 401, 400, 401, 401, 403], [463, 454, 440, 428, 430, 435]),
    *RAILS['jin_s14'],
    (123.42, [355, 356, 356, 368, 397, 411], [404, 397, 394, 429, 497, 541]),
    (123.58, [356, 358, 360, 377, 410, 425], [399, 398, 396, 444, 537, 577]),
    (123.74, [363, 366, 368, 392, 431, 449], [407, 398, 399, 476, 577, 631]),
    (123.88, [366, 368, 369, 403, 435, 449], [405, 402, 401, 477, 584, 645]),
    (123.98, [369, 371, 373, 413, 450, 464], [409, 405, 401, 502, 609, 669]),
]
S14_EARLY = [
    (121.20, [368,362,356,354,353,354], [468,454,444,431,429,429]),
    (121.50, [371,368,370,372,375,375], [466,457,440,427,425,427]),
    (121.90, [400,400,398,397,399,403], [479,468,449,434,429,433]),
    S14_EXTENDED[0],
]
S14_BRIDGE = [
    (120.04, [381,380,378,377,375,375], [513,510,503,488,478,474]),
    (120.32, [383,383,381,380,380,380], [511,505,493,481,472,468]),
    (120.58, [384,381,377,377,376,377], [507,497,479,455,449,444]),
    (120.82, [380,379,375,375,375,375], [494,480,467,447,441,437]),
    (121.06, [367,361,358,356,354,353], [481,470,452,433,428,424]),
    S14_EARLY[0],
]
PROGRAMME_PROBES = {
    'jin_s04': ([
        (42.04, [275,275,273,273,272,272], [546,543,538,532,528,527]),
        (42.20, [352,353,357,360,365,369], [493,491,488,491,498,501]),
    ], []),
    'jin_s09': ([
        (72.76, [319,319,321,323,326,327], [401,396,390,394,401,405]),
        (72.84, [319,319,323,324,329,331], [402,396,389,395,401,405]),
        (72.92, [319,320,321,325,331,333], [404,396,390,395,403,406]),
    ], [(33, 450, 1630, 930)]),
    'jin_s10': ([
        (84.02, [391,393,395,400,403,404], [526,532,540,546,550,552]),
        (84.18, [393,394,397,401,408,412], [522,529,535,544,550,554]),
    ], []),
    'jin_s16': ([
        (129.78, [428,429,432,439,447,450], [507,514,524,541,554,559]),
        (129.94, [432,433,441,447,456,459], [508,517,530,543,556,561]),
    ], [(20, 620, 950, 1020), (1110, 625, 1640, 1015)]),
    'jin_s19': ([
        (143.22, [179,179,177,175,172,170], [309,309,309,309,307,306]),
        (143.28, [179,179,177,175,172,170], [308,309,309,309,307,306]),
    ], [(990, 400, 1640, 960), (1620, 440, 1919, 905)]),
    'chen_s12': ([
        (113.12, [416, 416, 417, 421, 424, 428], [536, 537, 548, 558, 568, 575]),
        (113.24, [418, 419, 420, 425, 428, 432], [548, 549, 559, 570, 578, 585]),
        (113.32, [421, 422, 422, 426, 431, 436], [544, 548, 558, 574, 581, 587]),
        (113.44, [426, 427, 428, 430, 434, 439], [546, 551, 562, 578, 586, 593]),
    ], [(0, 750, 1230, 914)]),
    'nayeon_s03': ([
        (52.16, [402,403,404,405,404,403], [476,469,459,443,435,430]),
        (52.38, [400,401,402,401,403,406], [476,471,461,442,435,431]),
        (52.64, [397,399,400,401,404,406], [480,474,460,448,442,441]),
        (52.90, [397,397,399,400,402,404], [480,474,464,456,450,446]),
    ], [(60, 535, 1890, 866)]),
    'nayeon_s06': ([
        (121.96, [267, 270, 271, 271, 270, 270], [378, 373, 360, 345, 334, 329]),
        (122.06, [270, 272, 274, 274, 275, 275], [377, 373, 361, 344, 334, 332]),
        (122.12, [275, 276, 278, 279, 279, 280], [384, 378, 360, 345, 339, 335]),
        (122.24, [275, 277, 279, 281, 281, 282], [387, 384, 363, 350, 340, 335]),
    ], [(565, 680, 1260, 915), (1220, 760, 1919, 920)]),
    'okada_s07': ([
        (113.74, [207, 204, 203, 208, 215, 219], [266, 262, 261, 270, 277, 284]),
        (113.78, [212, 209, 207, 210, 215, 222], [274, 268, 267, 269, 279, 286]),
        (113.86, [222, 218, 217, 220, 227, 233], [277, 272, 270, 278, 288, 296]),
        (113.94, [199, 198, 201, 208, 215, 219], [255, 254, 253, 265, 275, 281]),
        (114.02, [184, 186, 188, 192, 195, 198], [242, 243, 245, 254, 264, 267]),
        (114.12, [171, 174, 176, 179, 182, 187], [221, 225, 225, 234, 242, 253]),
    ], [(40, 285, 1919, 620)]),
}
CHEN_S20_FULL = [
    (196.98, [0]*6, [559, 554, 545, 536, 529, 523]),
    (197.14, [0]*6, [496, 491, 481, 473, 462, 458]),
    (197.30, [0]*6, [472, 465, 458, 449, 439, 433]),
    (197.46, [0]*6, [459, 454, 445, 436, 426, 419]),
    (197.64, [0]*6, [432, 426, 416, 407, 397, 392]),
    (197.80, [0]*6, [423, 416, 410, 402, 394, 390]),
    (197.96, [0]*6, [407, 401, 394, 386, 378, 373]),
    (198.12, [0]*6, [402, 397, 389, 381, 373, 369]),
]
OKADA_S07_EXTENDED = [
    (113.46, [187, 187, 186, 189, 192, 195], [272, 263, 258, 257, 261, 266]),
    (113.56, [184, 184, 185, 188, 191, 193], [262, 255, 252, 254, 258, 263]),
    (113.66, [196, 193, 194, 198, 202, 206], [268, 264, 259, 262, 270, 275]),
    *PROGRAMME_PROBES['okada_s07'][0],
    (114.24, [166, 167, 170, 173, 176, 179], [224, 226, 230, 235, 241, 245]),
    (114.34, [176, 178, 180, 183, 185, 186], [235, 237, 239, 243, 245, 246]),
    (114.44, [193, 194, 196, 198, 199, 200], [255, 255, 256, 258, 259, 261]),
    (114.54, [217, 219, 221, 223, 225, 225], [282, 283, 284, 287, 289, 291]),
]
NAYEON_S03_EXTENDED = [
    *PROGRAMME_PROBES['nayeon_s03'][0],
    (53.04, [408,406,404,401,402,405], [483,474,462,452,451,450]),
    (53.16, [403,402,401,400,401,405], [480,473,462,456,454,452]),
    (53.28, [393,392,392,394,395,394], [473,469,458,451,450,451]),
    (53.44, [385,385,387,389,391,391], [461,460,453,448,448,448]),
]
NAYEON_S06_LATE = [
    (124.04, [312,313,313,316,319,322], [450,451,451,452,454,457]),
    (124.24, [312,311,312,316,319,320], [454,453,452,451,450,452]),
    (124.44, [302,302,303,306,310,314], [449,448,448,451,455,455]),
    (125.18, [284,284,284,287,290,292], [448,450,451,455,457,458]),
    (125.44, [281,282,284,287,289,292], [448,448,450,453,457,460]),
    (125.68, [289,291,295,296,297,300], [455,456,460,465,469,471]),
]
GUARD_PROBES = {
    'jin_s22': [
        (154.26, [216,215,216,221,228,232], [265,265,267,276,283,287]),
        (154.34, [209,208,210,213,221,226], [260,258,260,268,276,280]),
        (154.40, [209,208,210,215,223,229], [261,260,263,270,280,286]),
        (154.48, [210,211,214,218,227,234], [265,267,269,276,285,292]),
    ],
    'jin_s08': [
        (71.66, [337,338,338,337,340,344], [410,399,387,390,408,416]),
        (71.78, [341,343,342,343,347,351], [415,408,392,398,411,423]),
        (71.86, [347,349,348,351,355,360], [426,413,400,407,421,433]),
        (72.04, [350,352,354,357,359,365], [434,421,407,416,431,443]),
    ],
    'jin_s15': [
        (127.26, [247,246,245,244,242,241], [556,548,538,531,524,520]),
        (127.36, [247,247,245,243,241,240], [554,548,540,533,525,520]),
        (127.52, [250,250,247,247,245,243], [559,552,545,535,529,525]),
        (127.60, [264,262,259,258,256,255], [567,560,552,542,534,532]),
    ],
}
S08_EXTENDED = [
    (71.60, [345,344,341,338,338,339], [429,414,405,398,399,405]),
    *GUARD_PROBES['jin_s08'],
    (72.16, [379,378,375,378,382,384], [450,438,433,444,452,457]),
]
S15_EXTENDED = [
    (126.82, [264,264,263,260,258,255], [554,550,545,541,538,536]),
    (126.98, [249,248,246,244,241,239], [550,546,540,534,531,529]),
    (127.10, [250,248,245,242,238,236], [551,545,537,532,526,523]),
    (127.22, [248,247,245,241,239,236], [552,546,539,532,525,522]),
    *GUARD_PROBES['jin_s15'],
]


def masks(rail_readings, pt, frame, person, clearance, ice_support):
    at = np.array([round(row[0]*12800) for row in rail_readings])
    rails = np.array([row[1:] for row in rail_readings])
    y = np.arange(frame.shape[0])[:, None]
    x = np.arange(frame.shape[1])
    top, bottom = (np.interp(x, X, [np.interp(pt, at, col) for col in positions.T])
                   for positions in np.moveaxis(rails, 1, 0))
    board = (y >= top+7) & (y <= bottom-8)
    chroma = np.ptp(frame, axis=2)
    paint = cv2.dilate((chroma >= 7).astype('u1'), np.ones((9, 9), 'u1'))
    ice = (y >= bottom+24) & (y < 1030) & (paint != 0)
    if ice_support:
        confirmed = np.zeros(frame.shape[:2], dtype=bool)
        for x0, y0, x1, y1 in ice_support:
            confirmed[y0:y1, x0:x1] = True
        ice &= confirmed
    clear = cv2.dilate((person != 0).astype('u1'), np.ones((clearance, clearance), 'u1')) == 0
    clear[:280, :560] = False
    clear[919:, 1650:] = False
    clear[:, :14] = clear[:, -14:] = False
    return board & clear, ice & clear


def features(image, domains):
    points, families = [], []
    for surface, domain in enumerate(domains):
        for x in range(0, 1920, 160):
            for y in (range(0, 1080, 180) if surface else [0]):
                x1, y1 = x+160, min(y+(180 if surface else 1080), 1080)
                texture = cv2.goodFeaturesToTrack(image[y:y1, x:x1], 20 if surface else 32,
                    .004 if surface else .006, 6, mask=domain[y:y1, x:x1].astype('u1'))
                if texture is None:
                    continue
                points.extend(texture[:, 0]+[x, y])
                families.extend([surface]*len(texture))
    return np.asarray(points, dtype='f4').reshape(-1, 1, 2), np.asarray(families, dtype='i1')


def correlation(a, b, left, right):
    delta = np.arange(-5, 6, dtype='f4')
    ox, oy = np.meshgrid(delta, delta)
    def patches(image, centre):
        mx = centre[:, :1]+ox.ravel()[None, :]
        my = centre[:, 1:]+oy.ravel()[None, :]
        patch = cv2.remap(image, mx, my, cv2.INTER_LINEAR).astype('f4')
        return patch-patch.mean(1, keepdims=True)
    x, y = patches(a, left), patches(b, right)
    return (x*y).sum(1)/np.maximum(1., np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1))


def update(previous, image, masks, points, family, initial=None):
    if not len(points):
        return points.copy(), np.zeros(0, bool), np.zeros(0), np.zeros(0)
    opts = dict(winSize=(19, 19), maxLevel=3 if initial is None else 1,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    seed = 0 if initial is None else cv2.OPTFLOW_USE_INITIAL_FLOW
    pred, ok, _ = cv2.calcOpticalFlowPyrLK(previous, image, points, initial, flags=seed, **opts)
    safe = pred.copy()
    safe[~np.isfinite(safe)] = 0.
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(image, previous, safe, points.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, winSize=(19, 19), maxLevel=1,
        criteria=opts['criteria'])
    end = safe[:, 0]
    x, y = np.rint(end).astype(int).T
    dx = np.linalg.norm(back[:, 0]-points[:, 0], axis=1)
    gate = (ok[:, 0] != 0) & (reverse[:, 0] != 0) & (dx <= np.where(family == 0, .65, .9))
    gate &= (x >= 14) & (x < 1906) & (y >= 14) & (y < 1066)
    rows = np.flatnonzero(gate)
    gate[rows] &= np.asarray(masks)[family[rows], y[rows], x[rows]] != 0
    ncc = np.zeros(len(points))
    rows = np.flatnonzero(gate)
    if len(rows):
        ncc[rows] = correlation(previous, image, points[rows, 0], end[rows])
    gate &= ncc >= np.where(family == 0, .83, .84)
    return safe, gate, dx, ncc


def filter_bins(left, right, family, lag, quality):
    global_keep = np.ones(len(left), bool)
    disp = right-left
    # Per-surface local consensus. A single affine estimate across the image
    # would suppress real variation with depth and curved-board perspective.
    for k, (x, y) in enumerate(left):
        local = (family == family[k]) & (abs(left[:, 0]-x) <= 150) & (abs(left[:, 1]-y) <= 75)
        if local.sum() >= 5:
            delta = np.median(disp[local], axis=0)
            spread = np.median(np.linalg.norm(disp[local]-delta, axis=1))
            global_keep[k] = np.linalg.norm(disp[k]-delta) <= max(1.3+.23*lag, 3.5*spread)
    bins = defaultdict(list)
    for k in np.flatnonzero(global_keep):
        x, y = left[k]
        bins[family[k], int(x)//160, 0 if family[k] == 0 else int(y)//160].append(k)
    keep = np.zeros(len(left), bool)
    for rows in bins.values():
        # Keep an even support ceiling in each screen strip and depth band.
        keep[np.asarray(rows)[np.argsort(quality[rows], kind='stable')[:5]]] = True
    return keep


def measure(name, target=TARGET, rail_readings=None, clearance=25, ice_support=(), board_only=False,
            board_support=None, lags=LAGS):
    cv2.setNumThreads(1)
    rail_readings = RAILS[name] if rail_readings is None else rail_readings
    shot = read(PREPARED/'setup.json')['shots'][name]
    state = camera_state(BASE)
    native = native_for(BASE, shot['program'])
    pts = state[name+'_pts']
    lo, hi = (round(v[0]*12800) for v in (rail_readings[0], rail_readings[-1]))
    indices = np.flatnonzero((pts >= lo) & (pts <= hi))
    packed = matte(shot, pts)
    history = deque(maxlen=max(lags)+1)
    raw = defaultdict(list)
    absent = []
    for index, (pt, frame) in zip(indices, source_frames(shot['source'], pts[indices]), strict=True):
        person = foreground(shot, packed, index, pt)
        if person is None:
            absent.append(int(pt))
            history.clear()
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        domain = masks(rail_readings, pt, frame, person, clearance, ice_support)
        if board_only:
            domain[1][:] = False
        if board_support is not None:
            domain[0][:] &= board_support(pt, frame)
        for track in history:
            active = np.flatnonzero(track['active'])
            pred, ok, _, _ = update(last, gray, domain, track['current'][active], track['family'][active])
            track['current'][active] = pred
            track['active'][active] &= ok
            lag = index-track['index']
            if lag not in lags:
                continue
            valid = active[ok]
            left = track['left'][valid]
            family = track['family'][valid]
            right, checked, fb, ncc = update(track['gray'], gray, domain, left, family,
                track['current'][valid].copy())
            # The direct match must agree with the path through the original
            # intermediate exposures. All stored right pixels are direct LK.
            checked &= np.linalg.norm(right[:, 0]-track['current'][valid, 0], axis=1) <= 1.8
            left, right, family, fb, ncc = (a[checked] for a in (left[:, 0], right[:, 0], family, fb, ncc))
            quality = fb + 4.*(1.-ncc)
            keep = filter_bins(left, right, family, lag, quality)
            fields = dict(index=np.full(len(left), index), lag=np.full(len(left), lag),
                left_uv=left, uv=right, family=family, quality=quality, keep=keep, fb=fb,
                ncc=ncc, anchor_index=np.full(len(left), track['index']))
            for key, val in fields.items():
                raw[key].append(val)
        left, family = features(gray, domain)
        history.append(dict(index=index, gray=gray, left=left, family=family,
                            active=np.ones(len(left), bool), current=left.copy()))
        last = gray
    if not raw or not any(len(batch) for batch in raw['index']):
        raise ValueError('No verified original texture displacements for '+name)
    rows = {k: np.concatenate(v) for k, v in raw.items()}
    index = rows['index']
    left = rows['left_uv']
    family = rows['family']
    world, inverse = bind(native, shot, state[name+'_rigs'], state[name+'_controls'],
        state[name+'_times'], rows['anchor_index'], left, family)
    valid = np.isfinite(world).all(1)
    rows.update(world=world, inverse_error=inverse, valid=valid,
        held=np.isin(left[:, 0].astype(int)//160, [2, 8]),
        source_pair_pts=pts[np.c_[rows['anchor_index'], index]])
    manifest = {}
    for surface, label in enumerate(('board', 'ice')):
        use = family == surface
        if not use.any():
            continue
        data = {key: val[use] for key, val in rows.items()}
        path = (target if surface == 0 else target/'ice')/(name+'.npz')
        save(path, **data, pts=pts, source_sha256=shot['source_sha256'])
        certificate = path.parent/'quality'/path.name
        save(certificate, keep=data['keep'], pts=pts, source_sha256=shot['source_sha256'])
        keep = data['keep']
        manifest[label] = dict(packet=str(path), sha256=sha(path), certificate=str(certificate),
            certificate_sha256=sha(certificate), tracked=len(keep), certified=int(keep.sum()),
            valid_and_certified=int((keep & data['valid']).sum()),
            certified_lags=dict(Counter(map(int, data['lag'][keep]))),
            certified_right_exposures=len(np.unique(data['index'][keep])),
            x_strips=dict(Counter(map(int, data['left_uv'][keep, 0]//160))),
            held_valid=int((keep & data['valid'] & data['held']).sum()))
    save(target/'detail'/(name+'.npz'), transient=indices,
        source_pt_limits=np.array([lo, hi]), pts=pts)
    write(target/(name+'.json'), dict(source=shot['source'], source_sha256=shot['source_sha256'],
        source_time_base='1/12800', rail_readings=rail_readings, missing_foreground_pts=absent,
        foreground_dilation_kernel_px=clearance, ice_support_xyxy=ice_support,
        depth_state=BASE, depth_state_sha256=sha(WORK.parent/BASE/'state.npz'),
        raw_roi_samples=str(target/(name+'_board_ranges.jpg')),
        role='Development evidence from original grayscale patch displacement. Depth is conditional on the native state.',
        held_role='Spatial diagnostic strips. Related source patches can overlap other development data.',
        surfaces=manifest))
    print(name, manifest, flush=True)


def merge_surfaces(target, name):
    p, q = (directory/(name+'.npz') for directory in (target, target/'ice'))
    board, ice = arrays(p), arrays(q)
    for key in ('pts', 'source_sha256'):
        np.testing.assert_array_equal(board[key], ice[key])
    combined = {key: np.concatenate([board[key], ice[key]]) for key in board
                if key not in ('pts', 'source_sha256')}
    directory = target/'mixed'
    save(directory/(name+'.npz'), **combined, pts=board['pts'], source_sha256=board['source_sha256'])
    save(directory/'quality'/(name+'.npz'), keep=combined['keep'], pts=board['pts'],
        source_sha256=board['source_sha256'])
    write(directory/'source.json', dict(board=str(p), board_sha256=sha(p), ice=str(q), ice_sha256=sha(q),
        family_codes={'board': 0, 'ice': 1}, use='Native consumers must preserve surface family on inverse binding.'))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--shots', nargs='+', default=tuple(RAILS), choices=tuple(RAILS))
    parser.add_argument('--s22-onset', action='store_true', help='Measure source around the S22 refit boundary')
    parser.add_argument('--s14-extended', action='store_true', help='Measure S14 texture to the cut')
    parser.add_argument('--s14-early', action='store_true', help='Measure S14 texture before the late window')
    parser.add_argument('--s14-bridge', action='store_true', help='Measure the S14 early transition')
    parser.add_argument('--programme-probes', nargs='+', choices=tuple(PROGRAMME_PROBES))
    parser.add_argument('--chen-s20', action='store_true', help='Track board texture across Chen S20')
    parser.add_argument('--okada-s07-extended', action='store_true', help='Track Okada S07 around the fast pan')
    parser.add_argument('--nayeon-s03-extended', action='store_true', help='Track Nayeon S03 through the tail transition')
    parser.add_argument('--nayeon-s06-late', action='store_true', help='Track Nayeon S06 around the late painted line')
    parser.add_argument('--guard-probes', nargs='+', choices=tuple(GUARD_PROBES))
    parser.add_argument('--s08-extended', action='store_true', help='Track full S08 interval across the guard boundary')
    parser.add_argument('--s15-extended', action='store_true', help='Track static board print across the S15 transition')
    args = parser.parse_args()
    if args.s22_onset:
        measure('jin_s22', TARGET/'onset_probe', S22_ONSET)
    elif args.s14_extended:
        measure('jin_s14', TARGET/'s14_extended', S14_EXTENDED, clearance=45)
    elif args.s14_early:
        measure('jin_s14', TARGET/'s14_early', S14_EARLY, clearance=63,
                ice_support=[(35, 550, 1640, 1000)])
    elif args.s14_bridge:
        measure('jin_s14', TARGET/'s14_bridge', S14_BRIDGE, clearance=63,
                ice_support=[(25, 560, 1650, 1005)])
    elif args.programme_probes:
        for name in args.programme_probes:
            readings, ice = PROGRAMME_PROBES[name]
            measure(name, TARGET/'programme_probes', readings, 57, ice)
    elif args.chen_s20:
        measure('chen_s20', TARGET/'chen_s20_full', CHEN_S20_FULL, clearance=51, board_only=True)
    elif args.okada_s07_extended:
        measure('okada_s07', TARGET/'okada_s07_extended', OKADA_S07_EXTENDED, clearance=55, board_only=True)
    elif args.nayeon_s03_extended:
        target = TARGET/'nayeon_s03_extended'
        measure('nayeon_s03', target, NAYEON_S03_EXTENDED, clearance=55,
                ice_support=PROGRAMME_PROBES['nayeon_s03'][1])
        merge_surfaces(target, 'nayeon_s03')
    elif args.nayeon_s06_late:
        measure('nayeon_s06', TARGET/'nayeon_s06_late', NAYEON_S06_LATE, clearance=63,
                ice_support=[(30, 540, 1850, 972)])
    elif args.guard_probes:
        for name in args.guard_probes:
            measure(name, TARGET/'guard_boundaries', GUARD_PROBES[name], clearance=63,
                    board_only=name=='jin_s15')
    elif args.s08_extended:
        target = TARGET/'guard_s08_extended'
        measure('jin_s08', target, S08_EXTENDED, clearance=63)
        merge_surfaces(target, 'jin_s08')
    elif args.s15_extended:
        measure('jin_s15', TARGET/'guard_s15_extended', S15_EXTENDED, clearance=63, board_only=True)
    else:
        for name in args.shots:
            measure(name)
