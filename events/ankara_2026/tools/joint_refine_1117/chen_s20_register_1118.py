"""Measure the fixed Ad2 panel corner across the original closing close-up."""
from argparse import ArgumentParser
from shutil import copyfile

import cv2
import numpy as np

from board_displacement_1118 import CHEN_S20_FULL, TARGET, measure
from lane_end_evidence import foreground, matte
from review import source_frames
from shared import PREPARED, arrays, prepared_packet_path, read, save, sha, write


NAME = 'chen_s20'
ROOT = TARGET / 'chen_s20_registration'
PATH = 'print_instance/Ad2::Ad2.bottom_edge'
RIGHT = 'print_instance/Ad2::Ad2.right_vertical'
MATERIAL = 'source_ad2_panel_lower_right'
# Approximate support reads on the original image. Local color transitions
# supply the measured pixels across the full registered shot.
GUIDE = np.array([
    [196.98, 665., 477.], [197.14, 800., 406.], [197.30, 913., 378.],
    [197.46, 1021., 355.], [197.64, 1127., 352.], [197.80, 1225., 338.],
    [197.96, 1344., 324.], [198.12, 1460., 309.],
])


def transition(values, available, offset):
    signal = cv2.GaussianBlur(values[None, :].astype('f4'), (7,1), 1.3).ravel()
    slope = -np.gradient(signal)
    candidates = np.arange(8, len(signal)-8)
    accepted = []
    for i in candidates:
        before = np.median(signal[i-9:i-3]) if i >= 9 else signal[0]
        after = np.median(signal[i+3:i+9])
        if before-after < 12. or before < 17. or after > 11.:
            continue
        if not available[i-8:i+9].all():
            continue
        midpoint = (before+after)*.5
        if signal[i] >= midpoint > signal[i+1]:
            loc = offset+i+(signal[i]-midpoint)/(signal[i]-signal[i+1])
            accepted.append((before-after+slope[i]*2, loc))
    return float(max(accepted)[1]) if accepted else None


def fit_line(x, y):
    if len(x) < 8:
        return None
    design = np.column_stack((x, np.ones(len(x))))
    best = np.zeros(len(x), bool)
    for gap in (3, 6, 10):
        for i in range(len(x)-gap):
            j = i+gap
            if abs(x[j]-x[i]) < 18:
                continue
            slope = (y[j]-y[i])/(x[j]-x[i])
            if abs(slope) > .14:
                continue
            visible = abs(y-(slope*(x-x[i])+y[i])) < 2.
            if visible.sum() > best.sum():
                best = visible
    if best.sum() < 8:
        return None
    slope, intercept = np.linalg.lstsq(design[best], y[best], rcond=None)[0]
    return float(slope), float(intercept), best


def corner(raw, person, x, y):
    blue, green, red = np.moveaxis(raw.astype('f4'), 2, 0)
    ink = (red+blue)*.5-green
    blocked = cv2.dilate((person != 0).astype('u1'), np.ones((35,35),'u1')) != 0
    blocked[:279, :560] = True
    ix, iy = round(x), round(y)
    xx, yy = np.arange(ix-64, ix+65), np.arange(iy-52, iy+53)
    vertical = []
    for row in range(iy-172, iy-16, 7):
        v = transition(ink[row-2:row+3, xx].mean(0), ~blocked[row-2:row+3, xx].any(0), xx[0])
        if v is not None:
            vertical.append((float(row), v))
    horizontal = []
    for col in range(max(12, ix-260), ix-26, 9):
        v = transition(ink[yy, col-2:col+3].mean(1), ~blocked[yy, col-2:col+3].any(1), yy[0])
        if v is not None:
            horizontal.append((float(col), v))
    if len(vertical) < 8 or len(horizontal) < 8:
        return None
    v, h = np.asarray(vertical), np.asarray(horizontal)
    fitted_v, fitted_h = fit_line(*v.T), fit_line(*h.T)
    if fitted_v is None or fitted_h is None:
        return None
    vx, v0, use_v = fitted_v
    hy, h0, use_h = fitted_h
    px = (vx*h0+v0)/(1.-vx*hy)
    py = hy*px+h0
    if abs(px-x) > 55 or abs(py-y) > 45:
        return None
    return np.array([px, py]), v[use_v][:, ::-1], h[use_h]


def read_source(shot, pts):
    cv2.setNumThreads(1)
    foreground_packet = matte(shot, pts)
    points, indices, support, vertical_support, horizontal_support = [], [], [], [], []
    for idx, (pt, raw) in enumerate(source_frames(shot['source'], pts)):
        person = foreground(shot, foreground_packet, idx, pt)
        if person is None:
            continue
        seconds = pt/12800.
        x, y = [float(np.interp(seconds, GUIDE[:, 0], GUIDE[:, k])) for k in (1, 2)]
        found = corner(raw, person, x, y)
        if found is None:
            continue
        uv, vert, hor = found
        points.append(uv)
        indices.append(idx)
        support.append([len(vert), len(hor)])
        vertical_support.append(vert)
        horizontal_support.append(hor)
    indices = np.asarray(indices, dtype=int)
    return dict(uv=np.asarray(points), idx=indices, pts=pts[indices], support=np.asarray(support),
        vertical_uv=np.vstack(vertical_support), horizontal_uv=np.vstack(horizontal_support),
        vertical_offsets=np.r_[0, np.cumsum([len(a) for a in vertical_support])],
        horizontal_offsets=np.r_[0, np.cumsum([len(a) for a in horizontal_support])],
        source_sha256=shot['source_sha256'])


def qualified_print(pt, raw, glyphs=False):
    seconds = pt/12800.
    x, y = [float(np.interp(seconds, GUIDE[:, 0], GUIDE[:, k])) for k in (1, 2)]
    blue, green, red = np.moveaxis(raw.astype('f4'), 2, 0)
    violet = ((red+blue)*.5-green) > 14.5
    ink = cv2.dilate(violet.astype('u1'), np.ones((15,15), 'u1')) != 0
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    structure = cv2.cornerEigenValsAndVecs(gray, 11, 3)
    has_corner = (structure[:, :, 1] > .000013) & (structure[:, :, 1] > .10*structure[:, :, 0])
    has_corner[:, round(x)+18:] = False
    has_corner[round(y)+18:, :] = False
    if glyphs:
        white = (raw.min(axis=2) >= 143) & (np.ptp(raw, axis=2) <= 30)
        lettering = cv2.dilate(white.astype('u1'), np.ones((13,13), 'u1')) != 0
        # The interior white material in this window is the printed lettering.
        # The right white board and lower transition cannot enter its support.
        lettering[:, round(x)-40:] = False
        lettering[round(y)-24:, :] = False
        ink &= lettering
    return ink & has_corner


def prepare():
    setup = read(PREPARED/'setup.json')
    shot = setup['shots'][NAME]
    clock = arrays(PREPARED/'baseline.npz')[NAME+'_pts']
    source = read_source(shot, clock)
    if not len(source['pts']) or source['pts'][0] != clock[0] or source['pts'][-1] != clock[-1]:
        raise ValueError('The original print corner must support both registered shot endpoints')
    source_path = ROOT/'source_corners.npz'
    save(source_path, **source)
    path = prepared_packet_path(NAME)
    old = arrays(path)
    shadow = np.zeros(len(old['idx']), bool)
    for index, uv in zip(source['idx'], source['uv'], strict=True):
        dx, dy = (old['uv']-uv).T
        same_region = (((old['path'] == RIGHT) & (dy > -176) & (dy < 10))
                      | ((old['path'] == PATH) & (dx > -266) & (dx < 10)))
        shadow |= (old['idx'] == index) & same_region & (old['fold'] < 5) & ~old['station_fixed']
    changed = np.flatnonzero(shadow)
    previous_role = old['fold'][changed].copy()
    old['fold'][changed] = 9
    origin = np.arange(len(source['pts']))-173000000
    code = max(map(int, shot['source_packets']))+1
    extra = dict(pts=source['pts'], idx=source['idx'], uv=source['uv'], path=PATH,
        station=0., fold=0, family='Ad2', material=MATERIAL, source_fold=0, source_packet=code,
        source_row=np.arange(len(origin)), station_fixed=True, baseline_display=False,
        origin_row=origin, restored_admission_reason='Original purple print corner')
    for key, column in old.items():
        values = np.broadcast_to(extra[key], (len(origin),)+column.shape[1:])
        old[key] = np.concatenate((column, values.astype(column.dtype)))
    options = ROOT/'prepared'
    options.mkdir(parents=True, exist_ok=True)
    payload = options/(NAME+'.npz')
    save(payload, **old)
    save(ROOT/'changed_source_roles.npz', rows=changed, previous_fold=previous_role, current_fold=np.full(len(changed),9))
    shot['observations'] = str(payload)
    shot['source_packets'][str(code)] = dict(path=str(source_path), source_sha256=shot['source_sha256'],
        source_row='Original Ad2 lower-right corner reconstructed from both local color transitions')
    copyfile(PREPARED/'baseline.npz', options/'baseline.npz')
    write(options/'setup.json', setup)
    write(ROOT/'corner_binding.json', dict(source=shot['source'], source_sha256=shot['source_sha256'],
        measured=len(source['pts']), count=len(clock), missing_pts=np.setdiff1d(clock, source['pts']).tolist(),
        original_packet=str(path), original_packet_sha256=sha(path), packet=str(payload), packet_sha256=sha(payload),
        retired_correlated_edge_rows=len(changed), corner_path=PATH, corner_station=0.))
    print('Ad2 fixed corners', len(source['pts']), 'retired correlated rows', len(changed), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--track-print', action='store_true', help='Measure the colored print with qualified two-axis texture')
    parser.add_argument('--track-glyphs', action='store_true', help='Measure only the white lettering inside the panel')
    args = parser.parse_args()
    if args.track_glyphs:
        measure(NAME, ROOT/'glyphs', CHEN_S20_FULL, clearance=67, board_only=True,
                board_support=lambda pt, raw: qualified_print(pt, raw, glyphs=True),
                lags=(1, 2, 4, 8, 12, 15))
    elif args.track_print:
        measure(NAME, ROOT/'qualified', CHEN_S20_FULL, clearance=67, board_only=True,
                board_support=qualified_print, lags=(1, 2, 4, 8, 12, 15))
    else:
        prepare()
