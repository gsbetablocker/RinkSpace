"""Read paint and print boundaries in the registered source, with the projection used for search only."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from inspect_source import HERE, PROGRAM, PROJECT, SAMPLES, exact_frames, read

os.environ.setdefault('CUDA_PATH', 'D:/Program Files/CUDA')
sys.path.insert(0, str(PROJECT/'src'))
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import ParameterState, SceneParameterModel
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.gpu_geometry import NativeSceneGPU

from rinkspace.storage import AssetStore, configured_data_root

DATA = configured_data_root(PROJECT)
RUN = DATA/'runs/jin_sp_20261103'
W = read(AssetStore(PROJECT).resolve(PROGRAM/'working_fit_1103.json', must_exist=True))
PREP = Path(W['current_prepared'])
STATE = Path(W['latest_full_state']['path'])
MASKS = RUN/'source_bootstrap/person_masks/frames.jsonl'
ICE_COLOR = {'blue': np.array([-1., 0., 1.]), 'red': np.array([1., -.56, -.44]),
             'dark': -np.array([.25, .50, .25])}
AD_IDS = {1, 4, 6, 7, 8, 9, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23, 24}
LUMA = np.array([.25,.5,.25])
BLUE = r'^ice\.B[12]:'
RED = r'^ice\.(?:C:|C0:|A[1-4]:|goal_line_|goal_trapezoid_)'
LANE = r'^ice\.lane[1-4]:'
HOUSE = r'^ice\.[RF][1-4]:(inner|middle|outer)$'
PRINT = re.compile(r'^print_instance/Ad(\d+)::')

def mask_index():
    with MASKS.open(encoding='utf-8') as stream:
        for line in stream:
            rec = json.loads(line)
            yield rec['source_binding']['source_pts'], rec

def source_mask(record):
    path = record['mask_path']
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (1080, 1920):
        raise ValueError(f'Missing original person mask {path}')
    if record['diagnostic']['unresolved_support']:
        for box in record['current_source_person_boxes']:
            x, y, xx, yy = box['box_xyxy']
            dx, dy = .16*(xx-x), .16*(yy-y)
            a, b, c, d = np.clip([x-dx, y-dy, xx+dx, yy+dy], 0, [1920, 1080, 1920, 1080]).astype(int)
            mask[b:d, a:c] = 255
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
    mask[:245, :545] = 255
    mask[935:, 1620:1880] = 255
    return mask > 0

def world_sample(core, native, key, count):
    path = native.paths[key]
    board = native.scene['board']
    if path['kind'] == 7:
        s = np.linspace(0, core.board_perimeter(board), 512)
    else:
        s = np.linspace(*(path.get('support') or (0., 1.)), count)
    if path['kind'] == 8:
        a, b = np.asarray(path['vertices']).T
        q = path['knots']
        xyz = core.board_points(board, np.interp(s, q, a), np.interp(s, q, b))
    else:
        xyz = core.sample_path(board, path['kind'], path['parameters'], path['vertices'], s)
    return np.asarray(xyz)

def material_index(scene, native):
    materials = {}
    for path in scene['paths']:
        name = path['label']
        if re.match(BLUE, name):
            materials[path['id']] = ('blue', name)
        elif re.match(RED, name):
            materials[path['id']] = ('red', name)
        elif re.match(LANE, name):
            materials[path['id']] = ('dark', name)
        elif re.match(HOUSE, name):
            materials[path['id']] = ('house', name)
    for key in native.paths:
        match = PRINT.match(key)
        if not match or int(match[1]) not in AD_IDS:
            continue
        ad = int(match[1])
        path = key.split('::')[1]
        is_panel = any(s in path for s in ('.top_edge', '.bottom_edge', '.left_vertical', '.right_vertical'))
        if is_panel:
            materials[key] = ('panel', f'Ad{ad} panel')
        elif 'QR_white_frame' in path:
            materials[key] = ('white', f'Ad{ad} QR')
        elif ad in {9, 22} and 'red_field.boundary' in path:
            materials[key] = ('ink_red', f'Ad{ad} badge')
        elif ad in {11, 24} and 'red_shield' in path:
            materials[key] = ('ink_red', f'Ad{ad} shield')
        elif ad in {10, 23} and ('lower_arc' in path or 'upper_arc' in path or re.search(r'glyph_[1-8]_.+0000',path)):
            materials[key] = ('ink_red' if re.search(r'upper_arc|glyph_[5-8]',path) else 'ink_dark', f'Ad{ad} wordmark')
        elif ad in {6, 19} and re.search(r'component00[0-7]/path0', path):
            materials[key] = ('white', f'Ad{ad} glyph')
        elif ad in {1, 14} and re.search(r'\.print\.(ALL_|SKATING_|ONE_|APP_)',path) and 'inner' not in path:
            materials[key] = ('white', f'Ad{ad} glyph')
    return {key: value for key,value in materials.items() if key in native.paths}

def choose_queries(uv, status, radius, spacing, far_foot=None):
    ok = (status == 0) & np.isfinite(uv).all(axis=1)
    idx = np.flatnonzero(ok[1:-1] & ok[:-2] & ok[2:])+1
    idx = idx[(uv[idx] >= [radius+10, radius+7]).all(1)
              & (uv[idx] <= [1909-radius, 1072-radius]).all(1)]
    if far_foot is not None and len(idx):
        foot = far_foot[np.rint(uv[idx, 0]).astype(int)]
        idx = idx[uv[idx, 1] > foot+8.]
    if len(idx)<2:
        return np.empty((0, 2)), np.empty((0, 2))
    a,b = uv[idx]-uv[idx-1],uv[idx+1]-uv[idx]
    scale = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    good = (scale>1.e-5)&(np.sum(a*b,axis=1)>.83*scale)
    idx=idx[good]
    tangent = (a+b)[good]
    n = np.c_[-tangent[:,1],tangent[:,0]]
    n /= np.maximum(1.e-7,np.linalg.norm(n,axis=1)[:,None])
    chosen = []
    for k in range(len(idx)):
        if not chosen or (np.linalg.norm(uv[idx[k]]-uv[idx[chosen]],axis=1)>=spacing).all():
            chosen.append(k)
    return uv[idx[chosen]], n[chosen]

def pixel_profiles(rgb, mask, uv, normals, offsets, tangent):
    xy = uv[:,None,None,:]+normals[:,None,None,:]*offsets[None,None,:,None]
    tangents = np.c_[-normals[:,1],normals[:,0]]
    xy = xy+tangents[:,None,None,:]*tangent[None,:,None,None]
    shape = xy.shape[:-1]
    a,b = xy.reshape(-1,len(offsets),2).astype('f').transpose(2,0,1)
    values = cv2.remap(rgb,a,b,cv2.INTER_LINEAR).reshape(*shape,3)
    blocked = mask[np.clip(np.rint(xy[...,1]).astype(int),0,1079),np.clip(np.rint(xy[...,0]).astype(int),0,1919)]
    return np.median(values,axis=1).astype(float), np.any(blocked,axis=1)

def get_ice(rgb,mask,key,mode,name,uv,n,center,rad,min_span=None):
    if len(uv) < (4 if mode != 'house' else 3):
        return []
    t = np.arange(-rad-22,rad+23,dtype=float)
    v,blocked = pixel_profiles(rgb,mask,uv,n,t,np.array([-5,-2,0,2,5]))
    options = []
    for i, samples in enumerate(v):
        if mode == 'house':
            chroma = samples@ICE_COLOR['red']
            use_red = float(np.ptp(chroma))>3.0 and name[4]=='R'
            signal = chroma if use_red else samples@ICE_COLOR['dark']
            # The colored/dark band begins at its inner boundary and ends at the outer boundary.
            sign = 1. if (uv[i]-center)@n[i]>=0. else -1.
            if name.endswith((':outer', ':inner')):
                sign *= -1.
            strength = sign*np.gradient(gaussian_filter1d(signal,1.45))
            threshold = .57 if use_red else 1.05
            peaks,_ = find_peaks(strength,height=threshold,distance=5)
            widths = peak_widths(strength, peaks, rel_height=.50)[0]
            options.append([(t[j],float(strength[j]*2.),float(w),j-3,j+3)
                 for j,w in zip(peaks,widths) if abs(t[j])<=rad and 1.4<=w<=11
                 and np.ptp(signal[j-7:j+8])>(2.7 if use_red else 5.)
                 and not blocked[i,j-7:j+8].any()])
            continue
        signal = gaussian_filter1d(samples@ICE_COLOR[mode], 1.15)
        threshold = {'blue': 1.6, 'red': 2.0, 'dark': 2.9}[mode]
        if mode == 'blue' and signal.max() < 2.8:
            options.append([])
            continue
        if mode == 'red' and signal.max() < 3.5:
            options.append([])
            continue
        peaks, props = find_peaks(signal, prominence=threshold, wlen=71, distance=5)
        widths, _, lo, hi = peak_widths(signal, peaks, rel_height=.5, prominence_data=(
            props['prominences'], props['left_bases'], props['right_bases']))
        upper = 35 if mode in ('red','blue') else 23
        options.append([(float(np.interp((a+b)/2, np.arange(len(t)), t)),float(s),float(w),float(a),float(b))
            for j,w,a,b,s in zip(peaks,widths,lo,hi,props['prominences']) if abs(t[j])<=rad
            and (3.1 if mode != 'dark' else 2.4)<=w<=upper
            and not blocked[i,max(0,int(a)-5):int(b)+6].any()
            and (mode == 'dark' or signal[j] >= (3.1 if mode=='red' else 2.5))])
    # Require repeated, spatially separated evidence. A predicted curve alone cannot produce a point.
    tol = 6.4 if mode in ('blue','red') else 4.8
    offsets = [r[0] for o in options for r in o]
    if not offsets:
        return []
    def inliers(c):
        return [(i,min(p,key=lambda a:abs(a[0]-c)-.055*min(a[1],16.)))
                for i,o in enumerate(options) if (p:=[r for r in o if abs(r[0]-c)<=tol])]
    def score(c, rows):
        return len(rows)+sum(.085*min(r[1],12.) for _,r in rows)-.013*abs(c)
    ranked = sorted(((score(c,p),c,p) for c in offsets if (p:=inliers(c))),reverse=True,key=lambda a:a[0])
    quality, shift, best = ranked[0]
    if len(best)<(4 if mode=='dark' else 3):
        return []
    rival = next((x for x in ranked if abs(x[1]-shift)>2.1*tol), None)
    if rival is not None and rival[0] > quality*.87 and abs(shift)>5:
        return []
    i = np.array([k for k,_ in best])
    d = np.linalg.norm(uv[i,None,:]-uv[None,i,:],axis=-1).max()
    if d < (min_span if min_span is not None else (140. if mode=='dark' else 82. if mode=='house' else 115.)):
        return []
    rows = []
    for j,(offset,strength,width,a,b) in best:
        e = uv[j]+offset*n[j]
        side = uv[j]+np.interp([a,b],np.arange(len(t)),t)[:,None]*n[j]
        rows.append(dict(path_id=key, kind=1, uv=e.tolist(), source_normal=n[j].tolist(), measurement_operator=(
            'source_signed_house_boundary' if mode=='house' else 'source_two_material_edges_half_contrast_midpoint'),
            model_UV_synthesized=False, raw_physical_role=name, guide_uv=uv[j].tolist(),
            source_side_uv=side.tolist(), contrast=strength, width_px=width, normal_offset_px=offset))
    return rows

def get_print(rgb,mask,key,mode,name,uv,n,center,rad):
    if len(uv)<3:
        return []
    t=np.arange(-rad-11,rad+12,dtype=float)
    profiles,blocked=pixel_profiles(rgb,mask,uv,n,t,np.array([-3,-1,0,1,3]))
    rows=[]
    for i,samples in enumerate(profiles):
        # Color vector magnitude permits panel edges with brightness reversals.
        if mode=='panel':
            components=gaussian_filter1d(samples,1.05,axis=0)
            grad=np.linalg.norm(np.gradient(components,axis=0),axis=1)/np.sqrt(3.)
        else:
            w = {'white':LUMA,'ink_red':ICE_COLOR['red'],'ink_dark':-LUMA}[mode]
            scalar=gaussian_filter1d(samples@w,.95)
            # Known closed silhouettes can be searched in their actual ink-to-background direction.
            grad=np.abs(np.gradient(scalar))
        peaks,_=find_peaks(grad,height=2.3 if mode!='ink_red' else 1.7,distance=4)
        widths=peak_widths(grad,peaks,rel_height=.5)[0]
        choices=[]
        for j,width in zip(peaks,widths):
            if abs(t[j])>rad or not (1.55<=width<=8.0) or blocked[i,j-5:j+6].any():
                continue
            left,right=samples[j-6:j-3].mean(axis=0),samples[j+4:j+7].mean(axis=0)
            if mode=='panel':
                contrast=np.linalg.norm(left-right)/np.sqrt(3.)
            else:
                contrast=abs((left-right)@w)
                if mode=='white' and max(left@w,right@w)<150:
                    continue
            if contrast<({'white':23.,'panel':21.,'ink_red':11.,'ink_dark':17.}[mode]):
                continue
            choices.append((grad[j]-.20*abs(t[j]),j,width,contrast))
        if not choices:
            continue
        choices.sort(reverse=True)
        score,j,width,contrast=choices[0]
        if len(choices)>1 and choices[1][0]>.8*score and abs(choices[1][1]-j)>4:
            continue
        q=np.clip(.5*(grad[j-1]-grad[j+1])/(grad[j-1]-2*grad[j]+grad[j+1]),-.5,.5)
        offset=t[j]+q
        rows.append(dict(path_id=key,kind=1,uv=(uv[i]+offset*n[i]).tolist(),source_normal=n[i].tolist(),
            model_UV_synthesized=False,raw_physical_role=name,guide_uv=uv[i].tolist(),
            measurement_operator='source_RGB_print_edge_gradient',contrast=float(contrast),
            transition_fwhm_px=float(width),normal_offset_px=float(offset)))
    if len(rows)<3:
        return []
    # Long panel borders should agree across the side. Glyphs can move in both image axes.
    if mode=='panel' and np.ptp([r['normal_offset_px'] for r in rows])>12:
        med=np.median([r['normal_offset_px'] for r in rows])
        rows=[r for r in rows if abs(r['normal_offset_px']-med)<=5.5]
    return rows if len(rows)>=3 else []

def visible_foot(points):
    uv=points[(points[:,0]>=-12)&(points[:,0]<=1931)&(points[:,1]>=15)&(points[:,1]<=1079)]
    x=np.clip(np.rint(uv[:,0]).astype(int),0,1919)
    low=np.full(1920,np.inf)
    np.minimum.at(low,x,uv[:,1])
    xx=np.flatnonzero(np.isfinite(low))
    if len(xx)<12:
        return np.zeros(1920)
    return np.interp(np.arange(1920),xx,low[xx])

def scan(sid,stride):
    cv2.setNumThreads(1)
    m=read(PREP/'joint_manifest.json')
    s=next(s for s in m['shots'] if s['id']==sid)
    scene_path=Path(m['scene_path'])
    scene=read(scene_path)
    state=dict(np.load(STATE,allow_pickle=False))
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    core=NativeCore(m['native_library'])
    native=NativeSceneGPU(scene_path,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    material=material_index(scene,native)
    xyz=[]
    slices={}
    j=0
    for key in [*material,'board/continuous/bottom']:
        mode=material.get(key,('board',))[0]
        count=161 if mode in (*ICE_COLOR,'house') else 63
        v=world_sample(core,native,key,count)
        slices[key]=slice(j,j+len(v))
        j+=len(v)
        xyz.append(v)
    xyz=np.vstack(xyz)
    shot_pts=np.arange(s['first_pts'],s['last_pts']+1,256).astype(int)
    want=set(shot_pts[::stride].tolist())
    for p in SAMPLES.get(sid,()):
        want.update(q for q in shot_pts if abs(q-p)<=10*256)
    index={p:r for p,r in mask_index() if p in want and r['source_binding']['shot_id']==sid}
    if set(index)!=want:
        print(sid, 'skipping exposures without same-source occlusion masks', sorted(want-set(index))[:6],len(want-set(index)),flush=True)
        want=set(index)
    if not want:
        return
    tl=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    controls=state['controls'][s['control_begin']:s['control_stop']]
    rig=state['rig'][s['camera_index']]
    folder=HERE/'measurements'
    folder.mkdir(exist_ok=True)
    count=Counter()
    start=monotonic()
    with (folder/f'{sid}.jsonl').open('x',encoding='utf-8') as out:
        for n_frame,(p,bgr) in enumerate(exact_frames(m['source']['registered_path'],want,time_base=Fraction(1,12800)),1):
            rgb=np.ascontiguousarray(bgr[:,:,::-1])
            mask=source_mask(index[p])
            proj=core.project_motion(rig,controls,s['first_control_time'],tl.dt,1080,tl.local_exposure_time(s,p),xyz)
            uv,st=proj['uv'],proj['status']
            bottom=slices['board/continuous/bottom']
            foot=visible_foot(uv[bottom][st[bottom]==0])
            rows=[]
            for key,(mode,name) in material.items():
                sl=slices[key]
                is_ice=mode in (*ICE_COLOR,'house')
                rad=33 if mode in ('red','blue','house') else 22 if mode=='dark' else 24 if mode in ('panel','ink_red') else 13
                spacing=108 if mode=='dark' else 90 if is_ice else 22 if mode in ('panel','ink_red') else 11
                q,d=choose_queries(uv[sl],st[sl],rad+(22 if is_ice else 11),spacing,foot if is_ice else None)
                if not len(q):
                    continue
                center=np.mean(uv[sl][st[sl]==0],axis=0)
                new=(get_ice if is_ice else get_print)(rgb,mask,key,mode,name,q,d,center,rad)
                rows.extend(new)
            binding=dict(event_id='ankara_2026',programme_id='jin_shuxian_sp',source_id=m['source']['id'],
                         source_sha256=m['source']['sha256'],source_pts=p,time_base='1/12800',shot_id=sid,
                         native_hw=[1080,1920],image_domain='native_brown_pixels')
            rec=dict(shot_id=sid,pts=p,source_binding=binding,source_video_sha256=m['source']['sha256'],
                source_RGB_sha256=sha256(rgb.tobytes()).hexdigest(),
                source_BGR_sha256=sha256(bgr.tobytes()).hexdigest(),rows=rows)
            out.write(json.dumps(rec,separators=(',',':'))+'\n')
            count.update(r['raw_physical_role'] for r in rows)
            if p in SAMPLES.get(sid,()):
                mark=bgr.copy()
                for r in rows:
                    col=(5,230,255) if r['path_id'].startswith('print_instance/') else (20,80,250)
                    cv2.circle(mark,tuple(np.rint(r['uv']).astype(int)),5,col,-1,cv2.LINE_AA)
                    a=np.rint(r['guide_uv']).astype(int);b=np.rint(r['uv']).astype(int)
                    if np.linalg.norm(a-b)>4:
                        cv2.line(mark,tuple(a),tuple(b),(53,240,104),1,cv2.LINE_AA)
                cv2.imwrite(str(folder/f'{sid}_{p}.jpg'),cv2.resize(mark,(1152,648),interpolation=cv2.INTER_AREA))
            if n_frame%50==0 or n_frame==len(want):
                print(sid,n_frame,len(want),sum(count.values()),round(monotonic()-start,1),flush=True)
    (folder/f'{sid}_counts.json').write_text(json.dumps(count,indent=2),encoding='utf-8')

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('shots',nargs='+')
    cli.add_argument('--stride',type=int,default=1)
    args=cli.parse_args()
    for sid in args.shots:
        scan(sid,args.stride)
