"""Read the ice-side rail, kickplate and distinctive print edges on Nayeon's source frames."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, kron
from scipy.spatial import cKDTree

from feedback_source import contour_samples, read_contour, sample_image, selected_paths
from jin_board_1118 import paths
from lane_end_evidence import foreground, matte
from lane_side_1118 import BASE, WORK
from review import camera_state, ji, native_for, source_frames
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write


OUTPUT = WORK / 'nayeon_source'
WINDOWS = {'nayeon_s03': (49.48, 51.34), 'nayeon_s06': (124.26, 126.06)}
FEEDBACK = {'nayeon_s03': 639744, 'nayeon_s06': 1601536}
DETAIL = {'nayeon_s03': [49.70, 49.98, 50.20, 50.50, 50.80, 51.10],
          'nayeon_s06': [124.50, 124.80, 125.12, 125.40, 125.70]}
PROTECT = ('Ad8', 'Ad9', 'Ad10', 'Ad11', 'Ad12')


def peak_offset(samples, rows, peaks):
    centre = samples[rows, peaks]
    before = samples[rows, peaks-1]
    after = samples[rows, peaks+1]
    return np.clip((after-before) / np.maximum(.01, 2*(2*centre-before-after)), -.5, .5)


def cohorts(p, residual, valid, span, limit, minimum):
    """Require a continuing source edge, despite unrelated lines in the search strip."""
    out = np.zeros(len(p), dtype=bool)
    for i in np.flatnonzero(valid):
        other = valid & (np.linalg.norm(p-p[i], axis=1) <= span)
        agree = np.abs(residual[other]-residual[i]) <= limit
        out[i] = agree.sum() >= minimum and agree.mean() >= .54
    return out


def contours_of_coloured_panel(raw, mask, parts, mode):
    """Measure the actual coloured ink to white substrate transition on each side."""
    hits = []
    for key, points, normal, station, sign in parts:
        if len(points) < 3:
            continue
        # A positive probe points to the coloured interior.
        no = -normal if sign else normal
        if not sign:
            centre = np.vstack([p for _, p, _, _, _ in parts]).mean(axis=0)
            no = normal * np.sign(np.sum((centre-points)*normal, axis=1))[:, None]
        stride = max(1, len(points)//36)
        points, no, station = points[::stride], no[::stride], station[::stride]
        off = np.arange(-27., 28.)
        xy = points[:, None] + no[:, None]*off[None, :, None]
        b, g, r = sample_image(raw, xy).astype('f4').transpose(2, 0, 1)
        if mode == 'red':
            val = r - .5*(g+b)
        elif mode == 'green':
            val = g - .5*(b+r)
        else:
            val = b - r
        lum = (r+g+b)/3
        vis = (xy>2).all(2) & (xy<[1917,1077]).all(2)
        vis &= sample_image(mask, xy, cv2.INTER_NEAREST) == 0
        val = gaussian_filter1d(val, .95, axis=1)
        grad = np.gradient(val, axis=1)
        strength = np.zeros_like(grad)
        strength[:, 5:-5] = (val[:, 9:-1]+val[:, 8:-2]-val[:, 1:-9]-val[:, 2:-8])/2
        visible = vis.copy()
        for d in range(-5, 6):
            visible &= np.roll(vis, d, axis=1)
        possible = visible & (grad>1.1) & (strength>(6 if mode == 'green' else 13))
        # Neutral substrate and colour on the correct side also distinguish
        # the plate boundary from text and logos inside it.
        possible &= (np.roll(val,4,axis=1)<(14 if mode == 'green' else 22))
        possible &= np.roll(lum,4,axis=1)>137
        possible &= (grad > np.roll(grad,1,axis=1)) & (grad >= np.roll(grad,-1,axis=1))
        weight = np.where(possible, np.minimum(strength, 65)*(1+np.minimum(grad,20)/20), 0)
        smooth_votes = gaussian_filter1d(weight, 2.1, axis=1).sum(axis=0)
        mode_idx = int(np.argmax(smooth_votes / (1+.015*abs(off))))
        rows = np.arange(len(points))
        candidate = weight * (abs(np.arange(len(off))-mode_idx)<=4)
        peak = np.argmax(candidate,axis=1)
        valid = candidate[rows,peak] > 0
        if valid.sum() < max(3, len(points)*.23):
            continue
        rows = rows[valid]
        peak = peak[valid]
        delta = off[peak]+peak_offset(grad,rows,peak)
        for j, k, d in zip(rows, peak, delta):
            hits.append((key,station[j],points[j]+no[j]*d,points[j],no[j],strength[j,k]))
    return hits


def rectangle_parts(curves, table):
    parts = []
    for k, (uv, state) in curves.items():
        if len(uv) < 15 or not np.all(state == 0):
            continue
        n = np.gradient(uv, axis=0)
        length = np.linalg.norm(n, axis=1)
        if not np.isfinite(length).all() or np.sum(length) < 8:
            continue
        n = np.c_[-n[:,1],n[:,0]] / np.maximum(length,1.e-9)[:,None]
        jj = np.arange(10, len(uv)-10, 4)
        parts.append((k,uv[jj],n[jj],table[k][1][jj],0))
    return parts if len(parts) == 4 else []


def bar_transitions(raw, mask, curves, table):
    hits=[]
    for k, projected in curves.items():
        bars = contour_samples({k:projected},table,{k:1},'Ad5')
        if not bars or len(bars[0][1])<12:
            continue
        key,p,nn,st,sign = bars[0]
        if np.median(np.sum((p-p.mean(axis=0))*nn,axis=1))<0:
            nn=-nn
        side = 2*np.argmax(abs(nn),axis=1)+(np.sum(nn,axis=1)>0)
        sides = [(key,p[side==j],nn[side==j],st[side==j],sign) for j in range(4)
                 if np.count_nonzero(side==j)>=3]
        hits.extend(contours_of_coloured_panel(raw,mask,sides,'green' if 'green' in k else 'blue'))
    return hits


def rail_sections(raw, mask, parts, target):
    out = []
    checks = []
    for key, p, normal, station, _ in parts:
        top = key.endswith('top')
        no = normal * np.sign(normal[:,1,None])
        drawable = (p>[-1,8]).all(1)&(p<[1920,1069]).all(1)&(no[:,1]>.28)
        # This leaves support for the steep, foreshortened near board.
        p, no, station = p[drawable], no[drawable], station[drawable]
        if len(p) < 3:
            continue
        # The same native station step can span 2 or 30 pixels depending
        # on foreshortening. Select in the image domain before checking runs.
        cells = np.floor(p/(13 if top else 16)).astype(np.int32)
        _, take = np.unique(cells,axis=0,return_index=True)
        take.sort()
        p, no, station = p[take], no[take], station[take]
        off = np.arange(-81., 52.)
        grid = p[:,None] + no[:,None]*off[None,:,None]
        b,g,r = sample_image(raw,grid).astype('f4').transpose(2,0,1)
        vis = (grid>2).all(2) & (grid<[1917,1077]).all(2)
        vis &= sample_image(mask, grid, cv2.INTER_NEAREST)==0
        bg = gaussian_filter1d(b-r if top else .5*(r+g)-b, 1.1, axis=1)
        light = gaussian_filter1d((b+g+r)/3, 1.15, axis=1)
        dl = np.gradient(light,axis=1)
        dc = np.gradient(bg,axis=1)
        rows = np.arange(len(p))
        if top:
            grad = dl - 1.3*dc
            # A very visible junction on the ice side: coloured trim to the
            # white front face. Its stronger light step stays near the lip.
            inner = np.roll(bg,3,axis=1)
            innerlight = np.roll(light,3,axis=1)
            outer = np.roll(bg,-3,axis=1)
            outerlight = np.roll(light,-4,axis=1)
            diff = outerlight-innerlight
            admissible = (diff>29)&(inner>8)&(outerlight>147)&(outer<30)
            admissible &= (off>=-19)&(off<=24)
        else:
            grad = -dc+.11*dl
            inner = np.roll(bg,3,axis=1)
            outer = np.roll(bg,-4,axis=1)
            diff = inner-outer
            admissible = (diff>3.8)&(inner>6.)&(outer<18)
            admissible &= (off>=-18)&(off<=16)
        for d in range(-4,5):
            admissible &= np.roll(vis,d,axis=1)
        weight = np.where(admissible,grad/(1+.032*abs(off)),0)
        peak = np.argmax(weight,axis=1)
        offset = off[peak]+peak_offset(grad,rows,np.clip(peak,1,len(off)-2))
        valid = weight[rows,peak] > (5.3 if top else .7)
        valid = cohorts(p,offset,valid,115,4.2 if top else 4.5,4)
        for j in np.flatnonzero(valid):
            k = peak[j]
            uv = p[j]+no[j]*offset[j]
            if not top:
                out.append(('board.bottom','review_rail_edge',key,station[j],uv,p[j],no[j],diff[j,k]))
                continue
            # The upper edge of a blue vertical trim is visibly separate on
            # the far board. On the sloping near segment the dark cap extends
            # back from the lip. Keep that top surface separate entirely.
            steep_near = target and p[j,0]<730 and abs(no[j,0])>.25
            upper_strength = -dl[j]+.65*dc[j]
            sep = k-np.arange(len(off))
            peaks = (upper_strength>np.roll(upper_strength,1)) & (upper_strength>=np.roll(upper_strength,-1))
            narrow = peaks & (sep>=2)&(sep<=17) & (upper_strength>5.) & (dc[j]>1.0)
            narrow &= np.roll(vis[j],3)&np.roll(vis[j],-3)
            broad = peaks & (sep>=9)&(sep<=73) & (upper_strength>3.5)
            broad &= np.roll(vis[j],3)&np.roll(vis[j],-3)
            broad &= np.roll(bg[j],-4)>6
            rear = None
            if steep_near and broad.any():
                q = int(np.argmax(np.where(broad,upper_strength,0)))
                rear = p[j]+no[j]*(off[q]+peak_offset(upper_strength[None],np.array([0]),np.array([q]))[0])
                checks.append(('cap_rear_candidate',key,station[j],rear,p[j],no[j],float(upper_strength[q])))
            checks.append(('blue_trim_lower',key,station[j],uv,p[j],no[j],float(diff[j,k])))
            if steep_near:
                # This resolved blue-cap-to-face lip follows the diagonal white
                # front of the boards, rather than glass bases high on the cap.
                out.append(('board.top','front_rail_edge',key,station[j],uv,p[j],no[j],diff[j,k]))
            elif narrow.any():
                q = int(np.argmax(np.where(narrow,upper_strength,0)))
                upper = off[q]+peak_offset(upper_strength[None],np.array([0]),np.array([q]))[0]
                out.append(('board.top','review_rail_edge',key,station[j],p[j]+no[j]*upper,p[j],no[j],upper_strength[q]))
    return out, checks


def summary(q, pt):
    result = {}
    for f in np.unique(q['family'][q['pts']==pt]):
        sel = (q['pts']==pt)&(q['family']==f)
        for label, choose in [('all',sel), ('horizontal',sel&(abs(q['normal'][:,0])>.8)),
                              ('near',sel&(q['uv'][:,0]<300)), ('far',sel&(q['uv'][:,0]>=300))]:
            if not choose.any():
                continue
            direction = q['normal'][choose]
            delta = q['uv'][choose]-q['guide'][choose]
            signed = np.sum(delta*direction,axis=1)
            result[str(f)+'.'+label] = dict(rows=int(choose.sum()),
                median_signed_normal=round(float(np.median(signed)),2),
                p90_absolute_normal=round(float(np.quantile(abs(signed),.9)),2),
                median_source_minus_parent_xy=np.round(np.median(delta,axis=0),2).tolist(),
                median_source_minus_parent_x_horizontal=round(float(np.median(signed*direction[:,0])),2))
    return result


def measure(name, output):
    cv2.setNumThreads(1)
    state = camera_state(BASE)
    native = native_for(BASE,'nayeon')
    s = read(PREPARED/'setup.json')['shots'][name]
    pts = state[name+'_pts']; seconds = pts/12800.
    low, high = WINDOWS[name]
    dense = np.flatnonzero((seconds>=low)&(seconds<=high))
    control = np.array([round(t*12800) for t in DETAIL[name]])
    ind = np.unique(np.r_[np.arange(0,len(pts),23),len(pts)-1,dense,np.searchsorted(pts,control)])
    outline = paths(native)
    # Ad5's modeled plate outline is repeated by the two coloured bars. The
    # bars supply direct chromatic transitions with their own native paths.
    bars = {k:1 for k in native.paths if k.startswith('print_instance/Ad5::') and 'native3976' in k}
    prints = selected_paths(native)
    boards = prints['board']
    names = ('Ad6','Ad7')
    panels = {f:outline[f] for f in names}
    scalar = {f:{} for f in names}
    for k, spec in native.paths.items():
        f=spec['label']
        if f=='Ad6' and '.print.component00' in k and k.endswith('/path0'):
            scalar[f][k]=-1
        if f=='Ad7' and any(a in k for a in ('top_01_N_part00','top_05_N_part00',
                         'bottom_00_S_part00','bottom_01_K_part00','bottom_07_U_part00')):
            scalar[f][k]=-1
    protect = {f:prints[f] for f in PROTECT} if name=='nayeon_s03' else {}
    keys = sorted(set(bars)|{k for fam in (*panels.values(),*protect.values(),*scalar.values()) for k in fam})
    table = ji.native_paths(native,keys,count=129)
    table.update(ji.native_paths(native,boards,count=3073))
    world = {k:v[0] for k,v in table.items()}
    mattes = matte(s,pts)
    frames = source_frames(s['source'],pts[ind])
    data = defaultdict(list); checks = defaultdict(list); counts = []
    for lo in range(0,len(ind),10):
        indices = ind[lo:lo+10]
        jobs = [(i,k,state[name+'_times'][i]) for i in indices for k in table]
        projections = ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,world)
        by = {(i,k):v for (i,k,_),v in zip(jobs,projections)}
        for i in indices:
            pt, img = next(frames)
            assert pt==pts[i]
            person = foreground(s,mattes,i,pt)
            if person is None:
                counts.append((int(pt),0,'mask_missing'))
                continue
            mask = cv2.dilate(person,np.ones((9,9),np.uint8))
            # Broadcast graphics: S06 shows the technical score under the
            # upper-left score card. The nearby boards can be as high as y280.
            mask[:104,:552]=255
            if name=='nayeon_s06':
                mask[102:258,:395]=255
            mask[941:,1680:]=255
            current_count = len(data['pts'])
            def append(f,kind,hit,dest=data):
                key,st,pos,p,no,contrast = hit[:6]
                # Hold one disjoint temporal block per eight source blocks.
                # These are development checks and still share physical entities.
                fold = (6 if kind in ('letter_edge','protected_print_edge') or ((i//7)%8==5 and pt!=FEEDBACK[name])
                        else ((i//7)%5))
                for k,v in dict(pts=int(pt),idx=i,path=key,family=f,material=kind,uv=pos,guide=p,
                        station=float(st),normal=no,contrast=float(contrast),fold=fold,
                        group=f'{name}:{i//7}:{f}').items():
                    dest[k].append(v)
            curves = {k:by[i,k] for k in boards}
            parts = contour_samples(curves,table,boards,'board')
            rails, diagnostic = rail_sections(img,mask,parts,name=='nayeon_s03' and low<=seconds[i]<=high)
            for family,kind,*hit in rails:
                append(family,kind,hit)
            for kind,*hit in diagnostic:
                append('board.top',kind,hit,checks)
            for h in bar_transitions(img,mask,{k:by[i,k] for k in bars},table):
                append('Ad5','plate_transition',h)
            for f,keys_f in panels.items():
                pp = rectangle_parts({k:by[i,k] for k in keys_f},table)
                hits = contours_of_coloured_panel(img,mask,pp,'red' if f=='Ad6' else 'blue')
                for h in hits:
                    append(f,'plate_transition',h)
                if len(hits)<12:
                    continue
                margins = []
                for axis in range(2):
                    edge = [h for h in hits if abs(h[4][axis])>.81]
                    margins.append(float(np.median([h[2][axis]-h[3][axis] for h in edge])) if edge else 0.)
                sign = scalar[f]
                # Restrict white letter readings to the uniquely identified
                # coloured panel after its own material boundaries register.
                parts = contour_samples({k:by[i,k] for k in sign},table,sign,f)
                if not parts:
                    continue
                shifted = []
                for k,p,no,st,v in parts:
                    # Labelled glyph silhouettes have white ink inside. Use
                    # the projected geometry to establish outward orientation.
                    if np.median(np.sum((p-p.mean(axis=0))*no,axis=1))<0:
                        no=-no
                    shifted.append((k,p+margins,no,st,v))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype('f4')
                for k,st,pos,p,no,strength,shift in read_contour(gray,mask,shifted,f,(6,5)):
                    if abs(np.dot(pos-p,no))<=6 and np.linalg.norm(shift)<=5:
                        append(f,'letter_edge',(k,st,pos,p-margins,no,strength))
            if protect and 50.50<=seconds[i]<=51.10 and (i%3==0 or pt in control):
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype('f4')
                for f,keys_f in protect.items():
                    parts = contour_samples({k:by[i,k] for k in keys_f},table,keys_f,f)
                    if not parts:
                        continue
                    for h in read_contour(gray,mask,parts,f,(15,12)):
                        append(f,'protected_print_edge',h)
            counts.append((int(pt),len(data['pts'])-current_count,'decoded'))
    q = {k:np.asarray(v) for k,v in data.items()}
    ancillary = {k:np.asarray(v) for k,v in checks.items()}
    save(output/(name+'.npz'),**q,source_sha256=s['source_sha256'],time_base=np.array([1,12800]))
    save(output/('rail_diagnostic_'+name+'.npz'),**ancillary,source_sha256=s['source_sha256'],time_base=np.array([1,12800]))
    write(output/(name+'_measurements.json'),dict(shots_exposures=len(pts),sampled_exposures=len(ind),
        rows=len(q['pts']),source=s['source'],source_sha256=s['source_sha256'],time_base=[1,12800],
        exposure_rows=counts, exact={str(pt):summary(q,pt) for pt in control},
        material_rows={str(m):int((q['material']==m).sum()) for m in np.unique(q['material'])}))
    print(name,len(q['pts']),len(ind),flush=True)
    if name=='nayeon_s03':
        audit_old(s,q,output)
    evidence(s,name,q,ancillary,control,output)


def audit_old(shot, q, output):
    name = 'nayeon_s03'
    packet = prepared_packet_path(name)
    old = arrays(packet)
    eligible = (old['path']=='board/continuous/top')&(old['fold']<5)&~old['station_fixed']
    skipped = []
    reasons = []
    for pt in np.unique(q['pts']):
        if not WINDOWS[name][0]<=pt/12800.<=WINDOWS[name][1]:
            continue
        rows = np.flatnonzero(eligible&(old['pts']==pt))
        for family in ('near','far'):
            sel = (q['pts']==pt)&(q['family']=='board.top')
            sel &= (q['material']=='front_rail_edge') if family=='near' else (q['material']=='review_rail_edge')
            if sel.sum()<5 or not len(rows):
                continue
            tree = cKDTree(q['uv'][sel])
            d,j = tree.query(old['uv'][rows])
            uv=q['uv'][sel][j]; no=q['normal'][sel][j]
            diff = old['uv'][rows]-uv
            dn = np.sum(diff*no,axis=1)
            dt = abs(np.sum(diff*np.c_[no[:,1],-no[:,0]],axis=1))
            if family=='near':
                # Upper visual band ambiguity can span a few pixels. Only the
                # strongly separated behind-the-lip points are superseded.
                wrong = (old['uv'][rows,0]<680)&(dn<-12)&(dn>-78)&(dt<11)&(d<80)
                reason = 'behind_visible_ice_side_lip'
            else:
                # The dark material below the rail is usually print. A gap
                # of 9+ px passes the actual narrow rail and white substrate.
                wrong = (abs(no[:,0])<.18)&(dn>9)&(dn<46)&(dt<11)
                reason = 'below_verified_rail_on_front_face'
            skipped.extend(rows[wrong]); reasons.extend([reason]*wrong.sum())
    ix = np.asarray(skipped,dtype=np.int64)
    keep = np.argsort(ix); ix=ix[keep]
    assert len(np.unique(ix))==len(ix) and not old['station_fixed'][ix].any()
    save(output/'superseded_nayeon_s03.npz',rows=ix,reason=np.asarray(reasons)[keep],
         pts=old['pts'][ix],idx=old['idx'][ix],uv=old['uv'][ix],path=old['path'][ix],
         source_packet=old['source_packet'][ix],source_row=old['source_row'][ix],origin_row=old['origin_row'][ix],
         prepared_packet=str(packet),source_sha256=shot['source_sha256'])
    print('superseded',len(ix),flush=True)


def evidence(shot, name, q, check, control, output):
    for pt,img in source_frames(shot['source'],control):
        hit = q['pts']==pt
        if not hit.any():
            continue
        marks=img.copy()
        for material,xy in zip(check['material'][check['pts']==pt],check['uv'][check['pts']==pt]):
            color=(40,140,240) if material=='cap_rear_candidate' else (200,125,205)
            cv2.circle(marks,tuple(np.rint(2*xy).astype(int)),2,color,-1,cv2.LINE_AA,1)
        for f,m,xy,g in zip(q['family'][hit],q['material'][hit],q['uv'][hit],q['guide'][hit]):
            color=(40,220,250) if f=='board.top' else (250,200,20) if f=='board.bottom' else (60,235,95)
            cv2.circle(marks,tuple(np.rint(2*xy).astype(int)),3,color,-1,cv2.LINE_AA,1)
            if m in ('front_rail_edge','plate_transition'):
                cv2.line(marks,tuple(np.rint(2*g).astype(int)),tuple(np.rint(2*xy).astype(int)),(75,95,235),1,cv2.LINE_AA,1)
        stem=f'{name}_{pt}_measured'
        cv2.imwrite(str(output/(stem+'.jpg')),marks,[1,97])


def comparison(label, candidates):
    from side_repair import positions

    result = dict(parent=BASE, time_base=[1,12800],
        definition='source minus projection in the source-measured normal at the same native path station',
        scope='development measurements, including role 6 checks; shared physical entities', shots={})
    for name, candidate in zip(FEEDBACK,candidates,strict=True):
        file = OUTPUT/'verified_inputs'/(name+'.npz')
        q = arrays(file)
        before, after = (positions(model,name,q)[0] for model in (BASE,candidate))
        u = q['uv']; normal=q['normal']; pt=q['pts']; kind=q['material']; family=q['family']
        finite = np.isfinite(before).all(1)&np.isfinite(after).all(1)
        evidence = dict(candidate=candidate, measurements=str(file), measurements_sha256=sha(file),
            parent_state_sha256=sha(OUT/BASE/'state.npz'), candidate_state_sha256=sha(OUT/candidate/'state.npz'),
            source_sha256=q['source_sha256'].item(), groups={})
        result['shots'][name]=evidence
        low,high=WINDOWS[name]
        time = pt/12800.
        windows = {'feedback':pt==FEEDBACK[name], 'dense':(time>=low)&(time<=high),
                   'outside':(time<low)|(time>high)}
        if name=='nayeon_s03':
            windows.update(guard_50_800=pt==650240, left_guard=(time>=50.50)&(time<=51.10))
        masks = {'front_rail_near':(family=='board.top')&(kind=='front_rail_edge'),
                 'top_rail_far':(family=='board.top')&(kind=='review_rail_edge'),
                 'ice_junction_left':(family=='board.bottom')&(u[:,0]<310),
                 'ice_junction_middle':(family=='board.bottom')&(u[:,0]>=310)&(u[:,0]<1270),
                 'ice_junction_right':(family=='board.bottom')&(u[:,0]>=1270)}
        horizontal = abs(normal[:,0])>.80
        vertical = abs(normal[:,1])>.80
        for f in ('Ad5','Ad6','Ad7'):
            panel=(family==f)&(kind=='plate_transition')
            letter=(family==f)&(kind=='letter_edge')
            for axis,axis_sel in (('horizontal',horizontal),('vertical',vertical)):
                for subcase,sel in (('panel',panel),('letter',letter)):
                    masks[f+'.'+subcase+'.'+axis]=sel&axis_sel
            if f in ('Ad6','Ad7'):
                # The panel normal points from the white support into the
                # ink. Its x sign identifies the left and right print edges.
                for side,sx in [('left',1),('right',-1)]:
                    masks[f+'.panel.'+side]=panel&(sx*normal[:,0]>.80)
            else:
                for bar in ('blue','green'):
                    bar_sel=np.char.find(q['path'],f'{bar}_bar')>=0
                    masks[f+'.'+bar+'.horizontal']=panel&bar_sel&horizontal
                    masks[f+'.'+bar+'.vertical']=panel&bar_sel&vertical
        for f in PROTECT:
            masks[f+'.protected_print']=(family==f)&(kind=='protected_print_edge')
        displacements = [np.sum((u-p)*normal,axis=1) for p in (before,after)]
        for window,time_mask in windows.items():
            groups={}
            for key,spatial in masks.items():
                sel=time_mask&spatial
                for role,which in [('all',sel),('check',sel&(q['fold']==6))]:
                    if not which.any():
                        continue
                    ready=which&finite
                    catalog=dict(rows=int(which.sum()), paired_rows=int(ready.sum()),
                                 exposures=int(len(np.unique(pt[ready]))))
                    if ready.any():
                        frame_groups=np.split(np.flatnonzero(ready),np.flatnonzero(np.diff(pt[ready]))+1)
                        for version,d in zip(('parent','candidate'),displacements):
                            vals=d[ready]
                            # Signs are in screen coordinates, positive right
                            # on vertical print edges and down on horizontal ones.
                            axis=np.argmax(abs(normal[ready]),axis=1)
                            direction=normal[ready][np.arange(len(axis)),axis]
                            signed=vals*np.sign(direction)
                            record=dict(median_signed=round(float(np.median(signed)),3),
                                p50_abs=round(float(np.median(abs(vals))),3),
                                p90_abs=round(float(np.quantile(abs(vals),.9)),3),
                                exposure_median_abs_p90=round(float(np.quantile([
                                    np.median(abs(d[rows])) for rows in frame_groups],.9)),3))
                            if np.count_nonzero(ready&horizontal)==ready.sum():
                                record['median_horizontal_source_minus_projection']=round(float(np.median(
                                    vals/normal[ready,0])),3)
                            catalog[version]=record
                    groups[key+'.'+role]=catalog
            evidence['groups'][window]=groups
        print(name,candidate,'paired',int(finite.sum()),flush=True)
    write(WORK/(f'comparison_{label}.json'),result)


def cross_survey(programmes, focused=False):
    cv2.setNumThreads(1)
    destination=WORK/'cross_print_probe'
    if focused:
        destination/='low_end'
    state=camera_state(BASE)
    native=native_for(BASE,'nayeon')
    setup=read(PREPARED/'setup.json')['shots']
    target=('Ad5','Ad6','Ad7')
    outlines=paths(native)
    bars=[k for k in native.paths if k.startswith('print_instance/Ad5::') and 'native3976' in k]
    families={f:tuple(outlines[f]) for f in target}
    keys=[k for f in target for k in families[f]]+bars
    table=ji.native_paths(native,keys,count=129)
    world={k:v[0] for k,v in table.items()}
    reports={}
    for name,s in setup.items():
        if s['program'] not in programmes or (focused and s['camera_index']!=2):
            continue
        pts=state[name+'_pts']; frames=state[name+'_times']
        probes=np.arange(4,len(pts),7 if focused else max(23,int(len(pts)/43)))
        jobs=[(i,k,frames[i]) for i in probes for k in keys]
        pr=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,world)
        lookup={(i,k):v for (i,k,_),v in zip(jobs,pr,strict=True)}
        expected={}
        for i in probes:
            for f in target:
                curves=[lookup[i,k] for k in families[f]]
                if any(np.any(st!=0) for _,st in curves):
                    continue
                uv=np.vstack([u for u,_ in curves]); w,h=np.ptp(uv,axis=0)
                minimum_width=(34 if f=='Ad5' else 86) if focused else (42 if f=='Ad5' else 120)
                margin=np.r_[uv.min(axis=0)-[7,12], [1913,1050]-uv.max(axis=0)]
                if np.all(margin>0) and w>=minimum_width and h>=20:
                    expected.setdefault(i,{})[f]=[float(w),float(h),uv.min(0),uv.max(0)]
        indices=sorted(i for i,fs in expected.items() if len(fs)>=2)
        if not indices:
            continue
        mattes=matte(s,pts); records=[]; data=defaultdict(list); best=None
        for (pt,img),i in zip(source_frames(s['source'],pts[indices]),indices,strict=True):
            assert pt==pts[i]
            person=foreground(s,mattes,i,pt)
            if person is None:
                continue
            mask=cv2.dilate(person,np.ones((9,9),np.uint8));mask[:266,:552]=255;mask[940:,1670:]=255
            candidates={}
            for f,(w,h,lo,hi) in expected[i].items():
                if f=='Ad5':
                    hits=bar_transitions(img,mask,{k:lookup[i,k] for k in bars},table)
                else:
                    parts=rectangle_parts({k:lookup[i,k] for k in families[f]},table)
                    hits=contours_of_coloured_panel(img,mask,parts,'red' if f=='Ad6' else 'blue')
                ends=[np.array([j for j,h0 in enumerate(hits) if sign*h0[4][0]>.85],int)
                      for sign in (1,-1)]
                if min(map(len,ends))<5:
                    continue
                sides=[]; p90=[]; brightness=[]
                for idx in ends:
                    dx=np.array([hits[j][2][0]-hits[j][3][0] for j in idx]); centre=np.median(dx)
                    sides.append(float(centre)); p90.append(float(np.quantile(abs(dx-centre),.90)))
                    brightness.append(float(np.median([hits[j][5] for j in idx])))
                if max(p90)>3.4 or abs(sides[0]-sides[1])>min(11.,4.+.024*w) or min(brightness)<10:
                    continue
                candidates[f]=(w,h,lo,hi,hits,ends,sides,p90,brightness)
            if len(candidates)<2:
                continue
            row=dict(pts=int(pt),idx=int(i),seconds=pt/12800.,ads={},relative_ad6_station_px=None)
            for f,(w,h,lo,hi,hits,ends,sides,p90,brightness) in candidates.items():
                score=dict(left=sides[0],right=sides[1],centre=float(np.mean(sides)),
                           center_x_px=float((lo[0]+hi[0])/2),width_px=w,height_px=h,
                           max_p90_side_scatter=max(p90),min_contrast=min(brightness))
                row['ads'][f]=score
                for ix in ends:
                    for j in ix[np.linspace(0,len(ix)-1,min(7,len(ix))).round().astype(int)]:
                        key,st,pos,p,no,strength=hits[j]
                        if abs(pos[0]-p[0]-np.median([hits[k][2][0]-hits[k][3][0] for k in ix]))>3.4:
                            continue
                        for k,v in dict(pts=int(pt),idx=i,uv=pos,guide=p,station=float(st),path=key,family=f,
                            material='plate_transition',normal=no,contrast=strength,
                            fold=6 if ((i//42)%3==2 or s['program']=='chen') else 0,
                            group=f'{name}:{i//42}:{f}').items():
                            data[k].append(v)
            if len(candidates)==3:
                a,b,c=(row['ads'][f] for f in target)
                fraction=(b['center_x_px']-a['center_x_px'])/(c['center_x_px']-a['center_x_px'])
                row['relative_ad6_station_px']=b['centre']-((1-fraction)*a['centre']+fraction*c['centre'])
                row['relative_ad6_widths']=row['relative_ad6_station_px']/b['width_px']
            quality=1000*len(candidates)+min(v[0] for v in candidates.values())
            if best is None or quality>best[0]:
                left=np.maximum(0,np.rint(np.min([v[2] for v in candidates.values()],axis=0)-[15,15])).astype(int)
                right=np.minimum([1920,1080],np.rint(np.max([v[3] for v in candidates.values()],axis=0)+[15,15])).astype(int)
                pair=img.copy()
                for f,(_,_,_,_,hits,ends,*metrics) in candidates.items():
                    for key,st,pos,p,no,strength in hits:
                        if abs(no[0])<.85:
                            continue
                        cv2.line(pair,tuple(np.rint(2*p).astype(int)),tuple(np.rint(2*pos).astype(int)),(30,90,245),1,cv2.LINE_AA,1)
                        cv2.circle(pair,tuple(np.rint(2*pos).astype(int)),3,(70,239,95),-1,cv2.LINE_AA,1)
                crop=np.vstack([z[left[1]:right[1],left[0]:right[0]] for z in (img,pair)])
                best=(quality,pt,crop)
            records.append(row)
        if not records:
            continue
        q={k:np.asarray(v) for k,v in data.items()}
        save(destination/(name+'.npz'),**q,source_sha256=s['source_sha256'],time_base=np.array([1,12800]))
        path=destination/f'{name}_{best[1]}_source.jpg'; cv2.imwrite(str(path),best[2],[1,97])
        reports[name]=dict(programme=s['program'], camera_index=s['camera_index'], source=s['source'],
            source_sha256=s['source_sha256'], evidence_crop=str(path), candidate_exposures=len(indices), measurement_rows=len(q['pts']), reads=records)
        print(name,s['camera_index'],len(records),sum(d['relative_ad6_station_px'] is not None for d in records),flush=True)
    write(destination/'probe.json',dict(parent=BASE,shifts='source minus parent, native px',shots=reports))


def whole_print_material(shot, scene, n, measurements):
    from feedback_source import PARENT

    reference=OUT/'review_fixes_1118/print_reference/left_end_transfer.npz'
    source=arrays(reference)
    registrations=source['registrations']
    valid=(registrations[:,0]>=round(50.48*12800))&(registrations[:,0]<=round(51.12*12800))
    valid&=np.isin(registrations[:,1],[10,11])&(registrations[:,4]>=.76)&(registrations[:,5]>=.035)
    direct=camera_state(PARENT); n0=native_for(PARENT,'nayeon')
    groups=selected_paths(n0)
    table=ji.native_paths(n0,[k for f in ('Ad10','Ad11') for k in groups[f]],count=193)
    data=defaultdict(list); used_registration_rows=[]
    for registration_row in np.flatnonzero(valid):
        pt,ad,dx,dy,corr,gap,blur=registrations[registration_row]
        family='Ad'+str(int(ad)); i=int(np.searchsorted(direct[shot+'_pts'],int(pt)))
        keys=tuple(groups[family]); stations=np.concatenate([table[k][1] for k in keys]); world=np.concatenate([table[k][0] for k in keys])
        p=n0.core.project_motion(direct[shot+'_rigs'][i],direct[shot+'_controls'],scene['first'],scene['dt'],
                                 1080,direct[shot+'_times'][i],world)
        if np.any(p['status']!=0) or not np.isfinite(p['uv']).all():
            continue
        # The translation was recovered for the entire unique pattern. A
        # representative physical station supplies its local residual chart.
        nearest=int(np.argmin(np.linalg.norm(p['uv']-np.median(p['uv'],axis=0),axis=1)))
        path_ids=np.repeat(keys,[len(table[k][1]) for k in keys])
        path=str(path_ids[nearest])
        assert path in n.paths
        used_registration_rows.append(registration_row)
        for field,value in dict(pts=int(pt),idx=i,path=path,uv=p['uv'][nearest]+[dx,dy],guide=p['uv'][nearest],
                normal=[1.,0.],station=stations[nearest],family=family,fold=0,material='whole_print_transfer',
                contrast=corr,group=f'{shot}:{i//7}:{family}').items():
            data[field].append(value)
    return ({k:np.concatenate((v,np.asarray(data[k],dtype=v.dtype if v.dtype.kind!='U' else str)))
             if k in data else v for k,v in measurements.items()},reference,np.asarray(used_registration_rows))


def export_guard_inputs(output, shot, q, picked, weights, old, guard, reference, transfer_rows, source_file, source_count):
    admitted_root=WORK/'admitted'
    descriptor=read(admitted_root/'setup.json')['shots'][shot]
    baseline_file=Path(descriptor['observations'])
    baseline=arrays(baseline_file)
    packet_id=next(int(k) for k,val in descriptor['source_packets'].items() if Path(val['path'])==source_file)
    original_role=q['fold'][picked].astype('i1')
    has_direct_uv=picked<source_count
    code=np.where(has_direct_uv,0,1).astype('i1')
    original_role[~has_direct_uv]=-1
    transfer_input=np.full(len(picked),-1,dtype=np.int64)
    transfer_input[~has_direct_uv]=transfer_rows[picked[~has_direct_uv]-source_count]
    matches=np.flatnonzero(baseline['source_packet']==packet_id)
    by_source=np.full(source_count,-1,dtype=np.int64)
    by_source[baseline['source_row'][matches]]=matches
    admitted_rows=np.full(len(picked),-1,dtype=np.int64)
    admitted_rows[has_direct_uv]=by_source[picked[has_direct_uv]]
    base_role=np.full(len(picked),-1,dtype='i1')
    existing=admitted_rows>=0
    base_role[existing]=baseline['fold'][admitted_rows[existing]]
    np.testing.assert_array_equal(baseline['uv'][admitted_rows[existing]],q['uv'][picked[existing]])
    role=np.where((base_role>=0)&(base_role<5),base_role,0).astype('i1')
    fields={k:q[k][picked] for k in ('pts','idx','uv','station','path','material','family','normal','contrast','group')}
    fields.update(fold=role,source_fold=original_role,station_fixed=~has_direct_uv,
                  prior_admitted_row=admitted_rows,prior_admitted_fold=base_role,
                  guard_objective_weight=weights,source_uv_mixture=np.where(guard,.45,1.),
                  input_packet=code,source_row=np.where(has_direct_uv,picked,transfer_input))
    sha_src=q['source_sha256'].item()
    save(output/'used_source_uv.npz',**fields,source_sha256=sha_src)
    new=~existing
    save(output/'new_render_uv.npz',**{k:val[new] for k,val in fields.items()},source_sha256=sha_src)
    promote=existing&(base_role>=5)
    save(output/'admitted_role_updates.npz',rows=admitted_rows[promote],prior_fold=base_role[promote],
        fold=role[promote],pts=fields['pts'][promote],uv=fields['uv'][promote],path=fields['path'][promote],
        source_row=fields['source_row'][promote],source_sha256=sha_src)
    save(output/'protection_constraints.npz',actual_source_row=fields['source_row'][guard],
         used_source_uv_row=np.flatnonzero(guard),pts=fields['pts'][guard],path=fields['path'][guard],
         prior_projection=old[guard],mixture=np.full(np.count_nonzero(guard),.55),
         note='prior projection in compound local objective; exclude from observation/render packet')
    write(output/'source_roles.json',dict(shot=shot,source_sha256=sha_src,old_admitted=str(baseline_file),
        source_inputs={'0':dict(path=str(source_file),measure='original pixel sampled directly',source_row='NPZ row'),
                       '1':dict(path=str(reference),measure='same material 2D whole print registration, horizontal residual at representative native station',source_row='registrations array row')},
        fit_rows=len(picked),existing_fit_rows=int((existing&~promote).sum()),existing_promoted_rows=int(promote.sum()),
        new_render_rows=int(new.sum()),whole_print_rows=int((~has_direct_uv).sum()),
        profiles_or_motion_observations=0,prior_projection_constraints=int(guard.sum()),
        promotion_reason='source pixel was used in compact camera guard; no longer a check in the adopted result',
        baseline_role_map_packet=packet_id, exact_50_800_and_neighboring_frames_used=False,
        render='reuse preexisting actual fit rows, apply admitted_role_updates and append new_render_uv once; exclude protection_constraints'))


def guard_s03(balance=False, whole=False, export_only=False):
    """Keep the fitted rig fixed and adjust only a compact interval of motion knots."""
    from side_repair import positions

    out=WORK/'nayeon_guard'
    if whole:
        out/='whole_print'
    elif balance:
        out/='balanced'
    out.mkdir(parents=True,exist_ok=True)
    cand='lane4_side_1118/weighted/nayeon_s03'; shot='nayeon_s03'
    if whole:
        cand='lane4_side_1118/nayeon_guard/balanced_partial'
    state=camera_state(cand); native=native_for(cand,'nayeon')
    s=read(PREPARED/'setup.json')['shots'][shot]
    controls=state[shot+'_controls']; rigs=state[shot+'_rigs']; times=state[shot+'_times']
    pts=state[shot+'_pts']; sec=pts[0]/12800.
    t=sec+s['first']+s['dt']*np.arange(len(controls))
    centres=np.array([50.60,50.80,51.00]); radius=.255
    basis=np.maximum(0.,1-((t[:,None]-centres[None])/radius)**2)**3
    scale=np.array([.003,.003,.006])
    # Leave the exact feedback and its adjacent frames outside this local
    # objective, then assess the output on their original source reads.
    source=OUTPUT/'verified_inputs'/(shot+'.npz'); q=arrays(source); original_rows=len(q['pts'])
    reference=None
    if whole:
        q,reference,transfer_rows=whole_print_material(shot,s,native,q)
    ts=q['pts']/12800.; f=q['family']; kind=q['material']; no=q['normal']; u=q['uv']
    print_guard=kind=='protected_print_edge'; front=kind=='front_rail_edge'
    iceleft=(f=='board.bottom')&(u[:,0]<310)
    far_rails=((f=='board.top')|(f=='board.bottom'))&~front&~iceleft
    far_ads=(kind=='plate_transition')&(np.abs(no[:,0])>.8)
    near_print=print_guard&(np.abs(no[:,0])>.80)
    whole_print=kind=='whole_print_transfer'
    if whole:
        near_print &= ~np.isin(f,['Ad10','Ad11'])
    rows=(ts>=50.48)&(ts<=51.12)&(abs(ts-50.80)>.041)&(((q['idx']%2)==0)|whole_print)
    rows&=front|iceleft|far_rails|far_ads|near_print|whole_print
    rows&=(q['fold']<5)|near_print
    picks=[]
    weights=[]
    if whole:
        priorities=(1.45,4.15,7.8,4.0,2.5,6.25)
    elif balance:
        priorities=(2.20,3.40,6.50,2.25,1.65)
    else:
        priorities=(2.25,3.25,3.25,.45,1.20)
    sections=(near_print,front,iceleft,far_rails,far_ads)
    if whole:
        sections=(*sections,whole_print)
    regions=tuple(zip(sections,priorities,strict=True))
    for pt in np.unique(q['pts'][rows]):
        at=rows&(q['pts']==pt)
        for family in np.unique(f[at]):
            group=at&(f==family)
            for region,strength in regions:
                ix=np.flatnonzero(group&region)
                if not len(ix):
                    continue
                ix=ix[np.linspace(0,len(ix)-1,min(28,len(ix))).round().astype(int)]
                picks.extend(ix); weights.extend(np.full(len(ix),strength/np.sqrt(len(ix))))
    row_id=np.asarray(picks,int); weight=np.asarray(weights)
    assert len(np.unique(row_id))==len(row_id)
    z={k:q[k][row_id] for k in ('path','uv','normal','idx','station','pts','family','material')}
    old=positions(BASE,shot,z)[0]
    if whole:
        export_guard_inputs(out,shot,q,row_id,weight,old,near_print[row_id],reference,transfer_rows,source,original_rows)
    if export_only:
        return
    paths3d=ji.native_paths(native,np.unique(z['path']),count=513)
    for key in [k for k in paths3d if k.startswith('board/')]:
        paths3d.update(ji.native_paths(native,[key],count=4097))
    xyz=np.empty((len(row_id),3))
    for key,(path,st) in paths3d.items():
        sel=z['path']==key
        xyz[sel]=np.column_stack([np.interp(z['station'][sel],st,v) for v in path.T])
    frames={int(i):np.flatnonzero(z['idx']==i) for i in np.unique(z['idx'])}
    def predict(x):
        motion=controls+basis@(x.reshape(-1,3)*scale)
        uv=np.empty((len(row_id),2)); status=np.empty(len(row_id),np.int32)
        for i,ix in frames.items():
            p=native.core.project_motion(rigs[i],motion,s['first'],s['dt'],1080,times[i],xyz[ix])
            uv[ix]=p['uv']; status[ix]=p['status']
        return uv,status
    zeros=np.zeros(basis.shape[1]*3)
    log=out/'solve.jsonl'
    with log.open('w',encoding='utf-8') as stream:
        started=perf_counter()
        def record(**entry):
            entry['elapsed_seconds']=round(perf_counter()-started,3)
            stream.write(json.dumps(entry,ensure_ascii=False)+'\n');stream.flush()
        record(event='input', rows=len(row_id), exposures=len(frames), base=cand,
            source_path=str(source),source_sha256=sha(source),state_output=str(out/'state.npz'),
            parameters='9 coefficients for fixed-rig pan/tilt/log focal spline corrections',
            group_priorities=priorities,whole_print_packet=str(reference) if reference else None,
            whole_print_matches=int(np.count_nonzero(whole_print[row_id])),
            support_seconds=[float(t[np.flatnonzero(basis.any(axis=1))[v]]) for v in (0,-1)])
        p0,status=predict(zeros); assert (status==0).all() and np.isfinite(p0).all()
        target=z['uv'].copy(); guard=near_print[row_id]
        # Protect verified prior placement when the raw projected glyph is
        # foreshortened. The actual source still contributes on the other half.
        target[guard]=.45*target[guard]+.55*old[guard]
        e=np.sum((p0-target)*z['normal'],axis=1)
        jac=[]
        for col in range(len(zeros)):
            x=zeros.copy(); x[col]=.12
            a,st1=predict(x); b,st2=predict(-x)
            assert (st1==0).all() and (st2==0).all()
            jac.append(np.sum((a-b)*z['normal'],axis=1)/.24)
        jac=np.column_stack(jac)
        record(event='linearization',blocks=len(centres),domain_failures=0,rows=len(row_id),
               initial_weighted_rms=float(np.sqrt(np.mean((e*weight)**2))))
        def loss(x):
            return np.r_[(e+jac@x)*weight,x*1.5]
        optimized=least_squares(loss,zeros,loss='soft_l1',f_scale=1.35,bounds=(-1.1,1.1),max_nfev=34)
        trials=[]
        def penalty(v):
            return float(np.sum(1.35**2*(np.sqrt(1+(v/1.35)**2)-1)))
        for frac in (0.,.25,.50,.75,1.):
            x=frac*optimized.x; pred,status=predict(x)
            d=np.sum((pred-target)*z['normal'],axis=1)
            losses={str(fam):penalty((d*weight)[z['family']==fam]) for fam in np.unique(z['family'])}
            total=penalty(np.r_[d*weight,x*1.5]); available=bool(np.all(status==0))
            info=dict(event='trial',fraction=frac,cost=total,parts=losses,domain_failures=int((status!=0).sum()),
                      accepted=available and (not trials or total<min(r[0] for r in trials)))
            record(**info)
            if available:
                trials.append((total,frac,x))
        cost,frac,best=min(trials,key=lambda row:row[0])
        state[shot+'_controls']=controls+basis@(best.reshape(-1,3)*scale)
        assert np.array_equal(state[shot+'_rigs'],rigs)
        save(out/'state.npz',**state)
        record(event='export',model=str(out/'state.npz'),chosen_fraction=frac,before=trials[0][0],after=cost,
               model_sha256=sha(out/'state.npz'),function_evaluations=int(optimized.nfev))
    write(out/'inputs.json',dict(parent=cand,source_pack=str(source),source_sha256=q['source_sha256'].item(),
        pts_used=[int(pts[i]) for i in frames],support_seconds=[50.34,51.26],rig_fixed=True,geometry_fixed=True,
        whole_print_packet=str(reference) if reference else None,
        scope='S03 compact control changes with exact 50.800 and adjacent exposures excluded from the objective',
        coefficients=(best.reshape(-1,3)*scale).tolist()))


def ad1_audit():
    from side_repair import positions

    name='nayeon_s03'; out=WORK/'nayeon_guard'/'ad1'; out.mkdir(parents=True,exist_ok=True)
    models={'parent':BASE,'first':'lane4_side_1118/first_camera/nayeon_s03',
            'weighted':'lane4_side_1118/weighted/nayeon_s03','lens':'lane4_side_1118/nayeon_2_lens',
            'guard':'lane4_side_1118/nayeon_guard'}
    b=camera_state(models['weighted']); n=native_for(models['weighted'],'nayeon')
    s=read(PREPARED/'setup.json')['shots'][name]; pts=b[name+'_pts']; times=b[name+'_times']
    query=pts[(pts>=round(42.26*12800))&(pts<=round(42.50*12800))]
    keys={k:(f,polarity) for k, spec in n.paths.items() for f,match,polarity in
          [('Ad1_left','Ad1.left_vertical',-1),('Ad1_QR_left','Ad1.print.QR_white_frame',1),('Ad2_left','Ad2.left_vertical',-1)]
          if k.endswith(match)}
    table=ji.native_paths(n,keys,count=513); world={k:v[0] for k,v in table.items()}
    mats=matte(s,pts); data=defaultdict(list)
    offs=np.arange(-24.,25.)
    for pt,img in source_frames(s['source'],query):
        i=int(np.searchsorted(pts,pt)); person=foreground(s,mats,i,pt)
        if person is None:
            continue
        mask=cv2.dilate(person,np.ones((9,9),np.uint8))
        jobs=[(i,k,times[i]) for k in keys]
        uv=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        for (k,(label,sign)),(p,status) in zip(keys.items(),uv,strict=True):
            if not np.all(status==0):
                continue
            hi=p.max(0); lo=p.min(0); size=hi-lo
            pick=np.flatnonzero((p[:,1]>lo[1]+.17*size[1])&(p[:,1]<hi[1]-.16*size[1]))
            if label=='Ad1_QR_left':
                pick=pick[p[pick,0]<lo[0]+.24*size[0]]
            if len(pick)<12:
                continue
            pick=pick[np.linspace(0,len(pick)-1,26).round().astype(int)]
            guide=p[pick]; grid=guide[:,None]+np.c_[offs,np.zeros(len(offs))][None]
            val=sample_image(img,grid).astype('f4').mean(2)
            val=gaussian_filter1d(val,1.15,axis=1); grad=sign*np.gradient(val,axis=1)
            good=(grid>1).all(2)&(grid<[1918,1078]).all(2)&(sample_image(mask,grid,cv2.INTER_NEAREST)==0)
            ix=[]
            strength=[]
            deltas=[]
            for j,g in enumerate(grad):
                k0=int(np.argmax(np.where((abs(offs)<20)&good[j],g/(1+.023*abs(offs)),0)))
                if k0<5 or k0>=len(offs)-5 or not good[j,k0-5:k0+6].all():
                    continue
                contrast=sign*(val[j,k0+2:k0+6].mean()-val[j,k0-5:k0-1].mean())
                if contrast<16 or g[k0]<2.0:
                    continue
                d=offs[k0]+peak_offset(grad,np.array([j]),np.array([k0]))[0]
                ix.append(j); strength.append(contrast); deltas.append(d)
            if len(ix)<6:
                continue
            mode=np.median(deltas)
            for j,d,c in zip(ix,deltas,strength):
                if abs(d-mode)>3.4:
                    continue
                actual=guide[j]+[d,0]
                for a,value in dict(path=k,pts=int(pt),idx=i,family=label,uv=actual,guide=guide[j],
                        station=float(table[k][1][pick[j]]),normal=[1.,0.],contrast=float(c)).items():
                    data[a].append(value)
    q={k:np.asarray(v) for k,v in data.items()}
    save(out/'direct_ad1.npz',**q,source_sha256=s['source_sha256'])
    values={}
    for model,cand in models.items():
        p,_=positions(cand,name,q);dx=q['uv'][:,0]-p[:,0]; rows={}
        for pt in np.unique(q['pts']):
            rows[str(pt)]={}
            for key in np.unique(q['family'][q['pts']==pt]):
                d=dx[(q['pts']==pt)&(q['family']==key)]
                rows[str(pt)][str(key)]=dict(n=int(len(d)),median_horizontal=round(float(np.median(d)),3),
                    p90_horizontal_abs=round(float(np.quantile(abs(d),.90)),3))
        values[model]=rows
        print(model,rows.get('541952',{}),flush=True)
    write(out/'direct_ad1.json',dict(source=s['source'],source_sha256=s['source_sha256'],time_base=[1,12800],
        colors='left contour outer plate Ad1 and Ad2 and white QR frame Ad1; native source horizontal pixel positions',
        sign='source minus model x; positive means projection to the left of the material', models=values))


def native_guard(seed):
    from shared import numeric_sha
    from solve import Journal, stats

    shot='nayeon_s03'
    source=WORK/'nayeon_guard'/'whole_print'/'used_source_uv.npz'
    out=WORK/'nayeon_guard'/'native'/seed
    out.mkdir(parents=True,exist_ok=True)
    log=Journal(out/'solve.jsonl')
    parents={'weighted':'lane4_side_1118/weighted/nayeon_s03',
             'partial':'lane4_side_1118/nayeon_guard/balanced_partial',
             'whole':'lane4_side_1118/nayeon_guard/whole_print'}
    parent=parents[seed]
    q=arrays(source);s=read(PREPARED/'setup.json')['shots'][shot]
    b=camera_state(parent);n=native_for(parent,'nayeon')
    controls=b[shot+'_controls']; pts=b[shot+'_pts']; times=b[shot+'_times']
    rig=b[shot+'_rigs']; rigs, camera_ids=np.unique(rig[q['idx']],axis=0,return_inverse=True)
    assert np.array_equal(pts[q['idx']],q['pts']) and q['source_sha256'].item()==s['source_sha256']
    assert not np.any(abs(q['pts']/12800.-50.8)<=.041)
    knot=pts[0]/12800.+s['first']+s['dt']*np.arange(len(controls))
    centres=np.array([50.60,50.80,51.00]); radius=.255
    basis=np.maximum(0.,1-((knot[:,None]-centres[None])/radius)**2)**3
    motion=kron(csr_matrix(basis),csr_matrix(np.diag([.003,.003,.006])),format='csr')
    count=len(q['pts']); ids=np.arange(count,dtype=np.int32)
    fixed=q['station_fixed']; normals=q['normal']; tangents=np.c_[-normals[:,1],normals[:,0]]
    w=q['guard_objective_weight']
    # Whole-print registration is compared in x. Direct curve reads retain a
    # tangential term and use their usual independent native station variable.
    ranks=np.where(fixed,1,2).astype(np.int32)
    blocks=w[:,None,None]*np.stack([normals,.6*tangents],axis=1)
    whiten=blocks[np.column_stack((np.ones(count,bool),~fixed))].ravel()
    factors=dict(local_ids=np.where(~fixed,np.cumsum(~fixed)-1,-1).astype(np.int32),
        group_offsets=np.arange(count+1,dtype=np.int32),members=ids,ranks=ranks,
        weight_offsets=np.r_[0,np.cumsum(2*ranks,dtype=np.int32)].astype(np.int32),weights=whiten)
    log('start',rows=count,source=str(source), source_sha256=sha(source),
        parent=parent,adjacent_check_pts=[650240-512,650240,650240+512],
        motion_direction_count=motion.shape[1],active_rig_fields=[],active_field_fields=[],
        loss='native source-pixel factors only; mixed rank; Huber')
    mcam=dict(first_control_time=s['first'],control_dt=s['dt'],image_height=1080,
        reference_pts=0,time_base_num=1,time_base_den=12800)
    with n.resident(camera=mcam,controls=controls,rigs=rigs,camera_ids=camera_ids.astype(np.int32),
                    path_ids=q['path'],stations=q['station'],pts=q['pts'],observed=q['uv'],
                    motion_times=times[q['idx']]) as p:
        p.configure_projection_time_support(half_seconds=float(np.max(abs(rigs[:,37]))*.65+1e-6))
        supports=p._path_supports[p._codes[:count]]
        window=np.where(np.char.startswith(q['path'],'board/continuous/'),.035,.25)
        lower=np.maximum(supports[:,0],q['station']-window)[~fixed]
        upper=np.minimum(supports[:,1],q['station']+window)[~fixed]
        plan=p.configure_linearization(directions=motion,local_scales=np.full(np.count_nonzero(~fixed),.0015),
            huber_delta=1.35,**factors)
        gpu=p.configure_solver(lower=lower,upper=upper,device_budget_bytes=11*1024**3,
              host_budget_bytes=24*1024**3,initial_lambda_relative=.001,damping_metric='jacobi_image')
        initial=p.evaluate_errors()
        assert np.all(np.isfinite(initial))
        log('prepared',rows=count,plan=plan,gpu=gpu,before=stats(initial))
        def checkpoint():
            export=p.export_state()
            state=dict(b)
            state[shot+'_controls']=export['controls']
            state['scene_values_sha256']=numeric_sha(state['scene_values'])
            np.testing.assert_array_equal(state[shot+'_controls'][~basis.any(axis=1)],controls[~basis.any(axis=1)])
            save(out/'state.npz',**state)
        def progress(rec):
            if rec['summary']['accepted']:
                checkpoint()
            log('trial',**rec,checkpoint=str(out/'state.npz'))
            return not (out/'stop').exists()
        answer=p.solve(relative_cost_tolerance=1e-6,step_tolerance=1e-6,progress=progress,
                       numerical_progress=lambda rec:log('factorization_retry',**rec))
        checkpoint()
        after=p.evaluate_errors()
        save(out/'errors.npz',before=initial,after=after,pts=q['pts'],family=q['family'])
        save(out/'selected_rows.npz',used_uv_row=ids,pts=q['pts'],family=q['family'],
            fixed_station=fixed,prior_admitted_row=q['prior_admitted_row'],fold=q['fold'])
        write(out/'report.json',dict(parent=parent,fit_rows=count,source=str(source),before=stats(initial),after=stats(after),solver=answer))
        log('finish',rows=count,before=stats(initial),after=stats(after),solver=answer)


if __name__=='__main__':
    arg = ArgumentParser()
    arg.add_argument('shot', nargs='?', choices=tuple(FEEDBACK))
    arg.add_argument('--verified', action='store_true')
    arg.add_argument('--compare', nargs=3, metavar=('REPORT','S03_STATE','S06_STATE'))
    arg.add_argument('--cross', nargs='+', choices=('jin','okada','chen','nayeon'))
    arg.add_argument('--cross-low-end', action='store_true')
    arg.add_argument('--guard', action='store_true')
    arg.add_argument('--guard-balance', action='store_true')
    arg.add_argument('--guard-whole-print', action='store_true')
    arg.add_argument('--guard-export-only', action='store_true')
    arg.add_argument('--ad1-audit', action='store_true')
    arg.add_argument('--native-guard', choices=('weighted','partial','whole'))
    args = arg.parse_args()
    if args.compare:
        comparison(args.compare[0],args.compare[1:])
    elif args.cross:
        cross_survey(args.cross, args.cross_low_end)
    elif args.guard:
        guard_s03(balance=args.guard_balance,whole=args.guard_whole_print,export_only=args.guard_export_only)
    elif args.ad1_audit:
        ad1_audit()
    elif args.native_guard:
        native_guard(args.native_guard)
    else:
        if args.shot is None:
            arg.error('Select a shot or give --compare REPORT S03_STATE S06_STATE')
        measure(args.shot, OUTPUT/'verified_inputs' if args.verified else OUTPUT)
