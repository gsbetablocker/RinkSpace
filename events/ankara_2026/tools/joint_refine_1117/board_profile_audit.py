"""Check previously admitted dense rail pixels against both sides of the physical trim."""
from collections import Counter

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from feedback_source import sample_image
from lane_end_evidence import foreground, matte
from review import camera_state, source_frames
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write
from side_repair import positions


PARENT = 'lane4_side_1118/verified'
OUTPUT = OUT / 's14_boards_1118'
MATERIALS = ('review_rail_edge', 'blue_rail_pair', 'front_band_top')
OFFSETS = np.arange(-48., 49.)


def original_profiles(image, mask, xy, normal):
    tangent = np.column_stack((normal[:, 1], -normal[:, 0]))
    base = xy[:, None] + normal[:, None] * OFFSETS[None, :, None]
    samples, visibility = [], []
    for shift in (-9., -4., 0., 4., 9.):
        grid = base + shift * tangent[:, None]
        legal = (grid >= 1).all(2) & (grid < [1918, 1078]).all(2)
        samples.append(sample_image(image, grid).astype('f4'))
        visibility.append(legal & (sample_image(mask, grid, cv2.INTER_NEAREST) == 0))
    samples = np.stack(samples)
    blue, green, red = samples.transpose(3, 0, 1, 2)
    luminance = .114 * blue + .587 * green + .299 * red
    chroma = np.stack((blue - .5 * (red + green), .5 * (red + green) - blue))
    return luminance, chroma, np.stack(visibility)


def transitions(luminance, chroma, visible, top, lip):
    """Resolve a coloured band with a front/background and an ice/white-face side."""
    light = gaussian_filter1d(luminance, .9)
    colour = gaussian_filter1d(chroma, .9)
    signal = colour - (.25 * light if top else 0)
    strength = np.gradient(signal, axis=-1)
    # Five separate parallel cuts must agree on the location. Median across the
    # cuts alone would invent a continuous edge through an occlusion or a bolt.
    signed = np.full(luminance.shape[:2], np.nan)
    spread = np.full_like(signed, np.nan)
    for line in range(len(luminance)):
        g = strength[line]
        for row in range(len(g)):
            d, c, l, v = g[row], colour[line, row], light[line, row], visible[line, row]
            rising = np.flatnonzero((d[1:-1] >= d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > (1.35 if top else .50))) + 1
            falling = np.flatnonzero((d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < (-2. if top else -.60))) + 1
            best = None
            for a in rising:
                if a < 5:
                    continue
                for b in falling[(falling >= a + 2) & (falling <= a + 37)]:
                    if b + 5 >= len(c) or not v[a-4:b+5].all():
                        continue
                    target = a if top and not lip[row] else b
                    if abs(OFFSETS[target]) > 19:
                        continue
                    inside = np.median(c[a+1:b]); before = np.mean(c[a-4:a-1]); after = np.mean(c[b+2:b+5])
                    if top:
                        if min(inside - after, inside) < 5. or np.mean(l[b+2:b+5]) < 117:
                            continue
                        if np.mean(l[b+2:b+5]) - np.mean(l[a+1:b]) < 15:
                            continue
                        if inside - before < 3. and np.mean(l[a-4:a-1]) - np.mean(l[a+1:b]) < 15:
                            continue
                    elif min(inside - before, inside - after) < 3.5 or min(np.mean(l[a-4:a-1]), np.mean(l[b+2:b+5])) < 105:
                        continue
                    slope = d if target == a else -d
                    peak = max(slope[target], 1.e-5)
                    sub = np.clip((slope[target+1] - slope[target-1]) / max(.01, 2*(2*peak-slope[target-1]-slope[target+1])), -.5, .5)
                    # Sharpness expresses uncertainty only. Membership below
                    # additionally requires agreement along the physical edge.
                    sigma = float(np.clip((inside-after)/(2*max(.4,-d[b])), 1., 5.))
                    quality = min(40., inside - after) * min(5., -d[b]) / (1 + (OFFSETS[target]/5.)**2)
                    if best is None or quality > best[0]:
                        best = (quality, OFFSETS[target]+sub, sigma)
            if best is not None:
                _, signed[line, row], spread[line, row] = best
    return signed, spread


def cross_check(name):
    cv2.setNumThreads(1)
    packet = prepared_packet_path(name)
    q = arrays(packet)
    select = np.isin(q['family'], ('board.top', 'board.bottom')) & np.isin(q['material'], MATERIALS) & (q['fold'] < 9)
    rows = np.flatnonzero(select)
    if not len(rows):
        return
    a = {k:q[k][rows] for k in ('idx','path','uv','pts','station','family','fold','material')}
    guide, normal = positions(PARENT, name, a)
    normal *= np.where(normal[:, 1:] < 0., -1., 1.)
    indices = np.unique(a['idx'])
    pts = camera_state(PARENT)[name+'_pts']
    shot = read(PREPARED/'setup.json')['shots'][name]
    mattes = matte(shot, pts)
    original = np.full_like(a['uv'], np.nan, dtype=float)
    tolerance = np.full(len(rows), np.nan)
    cuts = np.zeros(len(rows), dtype='i1')
    reason = np.full(len(rows), 'unresolved_original_profile', dtype='U32')
    for (pt, raw), i in zip(source_frames(shot['source'], pts[indices]), indices, strict=True):
        if pt != pts[i]:
            raise ValueError((name, pt, pts[i]))
        mask = foreground(shot, mattes, i, pt)
        at = np.flatnonzero(a['idx'] == i)
        if mask is None:
            reason[at] = 'missing_foreground_identity'
            continue
        mask = cv2.dilate(mask, np.ones((7,7), 'u1'))
        mask[:259, :552] = 255
        mask[935:, 1660:] = 255
        for top in (False, True):
            ids = at[(a['family'][at] == ('board.top' if top else 'board.bottom')) & (normal[at,1] >= .24)]
            if not len(ids):
                continue
            points, no = a['uv'][ids], normal[ids]
            lum, col, vis = original_profiles(raw, mask, points, no)
            # The grazing west return shows the top surface in Nayeon's third
            # shot. Its front lip is visible where the white face begins.
            lip = (name == 'nayeon_s03') & (points[:,0] < 735) & (np.abs(no[:,0]) > .25)
            offsets, blur = transitions(lum, col[int(not top)], vis, top, lip)
            for j, row in enumerate(ids):
                ok = np.isfinite(offsets[:,j])
                if np.count_nonzero(ok)<4 or not ok[2]:
                    continue
                center = np.median(offsets[ok,j])
                agree = ok & (np.abs(offsets[:,j]-center) < 2.1)
                if agree.sum()<4 or not agree[2]:
                    continue
                delta = offsets[2,j]
                original[row] = points[j] + delta*no[j]
                cuts[row] = agree.sum()
                tolerance[row] = np.clip(.75 + .45*np.median(blur[agree,j]),1.6,2.7)
                reason[row] = 'same_material_edge' if abs(delta) <= tolerance[row] else 'different_source_edge'
    # A logo can have two parallel colour flanks inside a board. Check that
    # the measured transition continues across other source-image columns.
    # Each image cell has one vote irrespective of the legacy point density.
    gap = np.full(len(rows), np.nan)
    offset = np.sum((original-guide)*normal,axis=1)
    for i in indices:
        for family in ('board.top','board.bottom'):
            at = np.flatnonzero((a['idx']==i) & (a['family']==family) & np.isfinite(offset))
            if not len(at):
                continue
            x = original[at,0]; cells = (x//26).astype(int)
            unique = np.unique(cells)
            centres = np.array([np.median(x[cells==c]) for c in unique])
            shifts = np.array([np.median(offset[at][cells==c]) for c in unique])
            for j, row in enumerate(at):
                dx = abs(centres-x[j])
                nearby = (dx>22)&(dx<165)
                if np.count_nonzero(nearby)<3:
                    reason[row]='unresolved_source_continuity'
                    continue
                gap[row]=abs(offset[row]-np.median(shifts[nearby]))
                if gap[row]>max(3.,1.35*tolerance[row]):
                    reason[row]='unresolved_source_continuity'
    target = OUTPUT/'cross_rails'/name
    used = reason == 'same_material_edge'
    save(target.with_suffix('.npz'), row=rows, original_uv=a['uv'], measured_uv=original,
         pts=a['pts'], idx=a['idx'], family=a['family'], path=a['path'], material=a['material'],
         station=a['station'], old_fold=a['fold'], normal=normal, agreeing_profiles=cuts,
         tolerance=tolerance, cross_column_deviation=gap, reason=reason,
         source_sha256=shot['source_sha256'], parent_packet_sha256=sha(packet))
    write(target.with_suffix('.json'),dict(source=shot['source'],source_sha256=shot['source_sha256'],
        parent_packet=str(packet), parent_sha256=sha(packet), exposures=len(indices), rows=len(rows),
        reasons=dict(Counter(reason.tolist())), unchanged_fit=int((used&(a['fold']<5)).sum()),
        families={f:dict(Counter(reason[a['family']==f].tolist())) for f in np.unique(a['family'])}))
    print(name, len(indices), Counter(reason.tolist()), flush=True)
