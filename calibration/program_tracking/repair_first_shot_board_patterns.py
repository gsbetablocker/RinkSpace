"""Current-source outer circular print readouts without text-detail gates."""
from __future__ import annotations
from copy import deepcopy
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from .extract_first_shot_board_source import (DATA, MODEL, read, write, sha, feature_world,
    project_fast, segment_records, native_overlay, source_segments, print_inside_current_board_band)

REPAIR = DATA / 'repair_20261014'
TARGET = 'Ad11.print.outermost_visible_circular_print_stroke'


def current_mask_path(index, record):
    repaired=REPAIR/f'masks/{index:06d}.png'
    return repaired if repaired.is_file() else record['mask_path']


def sample_outer_ink(raw, mask, path, center, radius, *,minimum_prominence=7.,minimum_depth=9.):
    """Measure the outer white-to-ink interface of the first circular stroke.

    The stroke is a narrow dark trough rather than a step with a dark end
    plateau. Its outer half-depth interface is read from actual gray pixels.
    """
    path = np.asarray(path, float)
    normals = path-center
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], .01)
    offsets = np.arange(-radius, radius+1, dtype=float)
    locations = path[:, None, :] + normals[:, None, :]*offsets[None, :, None]
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype('float32')
    signal = cv2.remap(gray, locations[:, :, 0].astype('float32'), locations[:, :, 1].astype('float32'), cv2.INTER_LINEAR)
    signal = gaussian_filter1d(signal.astype(float), .7, axis=1)
    rows = []
    for i, values in enumerate(signal):
        if not (0 <= path[i, 0] <= 1919 and 0 <= path[i, 1] <= 1079):
            rows.append(None); continue
        peaks, info = find_peaks(-values, prominence=minimum_prominence, distance=3)
        limit = min(radius-3, 12+22*abs(normals[i, 0]))
        peaks = [int(k) for k in peaks if 3 <= k < len(offsets)-4 and abs(offsets[k]) < limit]
        if not peaks:
            rows.append(None); continue
        # Outside of the badge is plain board material. Pick the outermost
        # real dark trough; its inner text or inner rings are later inwards.
        peak = peaks[-1]
        end = min(len(values)-1, peak+12)
        baseline = float(np.max(values[peak+2:end+1]))
        depth = baseline-float(values[peak])
        if depth < minimum_depth:
            rows.append(None); continue
        part = values[peak:end+1]
        fractions=[]
        for fraction in (.25, .5, .75):
            level = values[peak]+fraction*depth
            crossings = np.flatnonzero((part[:-1] <= level) & (part[1:] > level))
            if not len(crossings): break
            k = int(crossings[0])+peak
            fractions.append(k+(level-values[k])/max(values[k+1]-values[k], 1.e-8))
        if len(fractions) != 3 or fractions[2]-fractions[0] > 14:
            rows.append(None); continue
        uv = path[i]+(fractions[1]-radius)*normals[i]
        x,y = np.rint(uv).astype(int)
        if not(3 <= x < 1917 and 3 <= y < 1077) or mask[y-3:y+4,x-3:x+4].any():
            rows.append(None); continue
        rows.append((uv, float(fractions[2]-fractions[0]), depth))
    return rows


def fit_source_ellipse(rows, initial_path):
    """A source-only smooth navigator; no fitted ellipse points are emitted."""
    good=np.array([r[0] for r in rows if r is not None])
    if len(good)<12: return None
    low=initial_path.min(0);high=initial_path.max(0);center=(low+high)/2;axes=(high-low)/2
    theta=np.arctan2((good[:,1]-center[1])/axes[1],(good[:,0]-center[0])/axes[0])
    initial=np.r_[center,axes,0.]
    def residual(v):
        angle=v[4];c,s=np.cos(angle),np.sin(angle)
        delta=good-v[:2]
        local=np.column_stack((delta[:,0]*c+delta[:,1]*s,-delta[:,0]*s+delta[:,1]*c))
        return (np.sqrt(np.sum((local/v[2:4])**2,1))-1)*np.mean(v[2:4])
    result=least_squares(lambda v:np.r_[residual(v), (v[2:4]-axes)*.5, v[4]*20],initial,loss='soft_l1',f_scale=1.,
                         bounds=(np.r_[center-40,axes*.85,-.1],np.r_[center+40,axes*1.15,.1]))
    if np.count_nonzero(np.abs(residual(result.x))<3)<12:return None
    count=max(64,int(np.ceil(2*np.pi*np.mean(result.x[2:4])/3)))
    angles=np.linspace(0,2*np.pi,count,endpoint=False)
    c,s=np.cos(result.x[4]),np.sin(result.x[4]);local=np.column_stack((np.cos(angles)*result.x[2],np.sin(angles)*result.x[3]))
    navigator=local@np.array([[c,s],[-s,c]])+result.x[:2]
    return navigator,result.x,dict(initial_current_source_peak_count=len(good),current_source_ellipse_normal_residual_median_px=float(np.median(np.abs(residual(result.x)))))


def circle_feature(raw, mask, initial_path, features, source_navigation_override=None):
    center=(initial_path.min(0)+initial_path.max(0))/2
    first=sample_outer_ink(raw,mask,initial_path,center,38)
    fit=fit_source_ellipse(first,initial_path)
    if source_navigation_override is None:
        if fit is None:return None,dict(reason='insufficient_visible_current_source_outer_stroke')
        navigator,v,diagnostic=fit
    else:
        v=source_navigation_override
        navigator=ellipse_navigator(v)
        diagnostic=dict(source_navigation_from_adjacent_source_shape=True,
                        initial_current_source_peak_count=sum(r is not None for r in first))
    rows=sample_outer_ink(raw,mask,navigator,v[:2],12)
    # Source measured pixels may differ slightly from a projected circle.
    # A smooth source navigator rejects unrelated text, with no fake fill.
    for i,row in enumerate(rows):
        if row is None:continue
        if np.linalg.norm(row[0]-navigator[i])>4 or not print_inside_current_board_band(features,np.asarray([row[0]])).all():rows[i]=None
    segments=circle_segments(rows,spacing=35.)
    if not segments:return None,dict(reason='no_source_verified_arc_after_occlusion_and_material_checks',**diagnostic)
    feature=dict(id=TARGET,target_id=TARGET,label='outermost_visible_circular_print_stroke',label_zh='Ad11 最外圆轮廓',
                 physical_surface='board',review_layer='ad2_print',physical_shape='circle_on_continuous_unwrapped_board',
                 segments=segments,allowed_for_fit=False,current_source_only_readout=True,model_or_template_generated_observation=False,
                 source_measurement_definition='outer_white_to_dark_half_depth_interface_of_outermost_print_stroke',
                 source_visibility_navigation='model_identity_and_same_source_smooth_circle_navigator_only')
    diagnostic.update(source_ellipse_navigation_parameters=v.tolist(),source_pixels_read=sum(r is not None for r in rows),
                      retained_points=sum(len(s['uv'])for s in segments),retained_arc_segments=len(segments))
    return feature,diagnostic


def ellipse_navigator(v):
    count=max(64,int(np.ceil(2*np.pi*np.mean(v[2:4])/3)))
    angles=np.linspace(0,2*np.pi,count,endpoint=False)
    c,s=np.cos(v[4]),np.sin(v[4]);local=np.column_stack((np.cos(angles)*v[2],np.sin(angles)*v[3]))
    return local@np.array([[c,s],[-s,c]])+v[:2]


def circle_segments(rows, spacing=35.):
    """Use the same spacing on circle arcs; glyph-corner rules do not apply."""
    full=all(r is not None for r in rows)
    ordered=rows
    if not full:
        cut=next(i for i,r in enumerate(rows)if r is None)
        ordered=rows[cut+1:]+rows[:cut+1]+[None]
    runs=[];run=[]
    for row in ordered:
        if row is None:
            if len(run)>=2:runs.append(run)
            run=[]
        else:run.append(row)
    if run:runs.append(run)
    segments=[]
    for run in runs:
        uv=np.asarray([r[0]for r in run]);distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(uv,axis=0),axis=1))]
        if distance[-1]<8:continue
        count=max(3 if distance[-1]>15 else 2,int(np.ceil(distance[-1]/spacing))+1)
        selected=sorted({int(np.argmin(abs(distance-d)))for d in np.linspace(0,distance[-1],count)})
        segments.append(dict(uv=uv[selected].tolist(),closed=full,vertex_count=len(selected),
                             measurement_origin='current_native_outer_circular_stroke_half_depth_interface',pixels_measured_in_this_frame=True,
                             source_native_readout_uv=uv.tolist(),source_native_transition_iqr_px=[r[1]for r in run],
                             transition_iqr_px=[run[j][1]for j in selected],localization_sigma_px=[max(1.,run[j][1]/1.349)for j in selected],
                             source_query_count=len(run),target_circle_arc_spacing_px=spacing))
    return segments


def repair(indices=None):
    destination=REPAIR/'board_candidates';destination.mkdir(parents=True,exist_ok=True)
    diagnostics=REPAIR/'board_diagnostics';diagnostics.mkdir(exist_ok=True)
    archive=np.load(DATA/'camera_seeds.npz');cameras={int(i):c for i,c in zip(archive['programme_indices'],archive['camera25'])}
    manifest={int(r['index']):r for r in read(DATA/'frame_manifest.json')['frames']}
    registry=read(MODEL/'feature_registry.json');scene=read(MODEL/'scene.json');templates=read(MODEL/'ad2_material_template.json')
    world=feature_world(scene,registry['targets'][TARGET],templates);reports=[]
    active=list(indices or range(790,918));anchors=[]
    for index in range(790,918):
        path,depth=project_fast(cameras[index],world)
        if not np.isfinite(path).all() or depth.min()<=0 or path[:,0].max()<-40 or path[:,0].min()>1959:continue
        record=manifest[index];raw=cv2.imread(record['image_path']);mask=cv2.dilate((cv2.imread(str(current_mask_path(index,record)),0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
        center=(path.min(0)+path.max(0))/2;axes=(path.max(0)-path.min(0))/2
        fit=fit_source_ellipse(sample_outer_ink(raw,mask,path,center,38),path)
        if fit is not None and fit[2]['initial_current_source_peak_count']>=100 and fit[2]['current_source_ellipse_normal_residual_median_px']<4:
            v=fit[1];anchors.append((index,np.r_[(v[:2]-center)/axes,v[2:4]/axes,v[4]]))
    anchor_indices=np.array([r[0]for r in anchors]);corrections=np.array([r[1]for r in anchors])
    if len(anchors)>=3:
        from scipy.ndimage import median_filter
        corrections=median_filter(corrections,size=(5,1),mode='nearest')
    for index in active:
        original=read(DATA/f'board_candidates/{index:06d}.json')
        current=destination/f'{index:06d}.json'
        packet=read(current)if current.is_file()else deepcopy(original);record=manifest[index]
        path,depth=project_fast(cameras[index],world)
        if not np.isfinite(path).all() or depth.min()<=0:continue
        if path[:,0].max() < -40 or path[:,0].min() > 1959:continue
        mask_path=current_mask_path(index,record)
        raw=cv2.imread(record['image_path']);mask=cv2.dilate((cv2.imread(str(mask_path),0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
        mask[:120,:570]=True;mask[945:,1660:]=True
        source_navigation=None
        if len(anchors)>=3:
            relative=np.array([np.interp(index,anchor_indices,corrections[:,j])for j in range(5)])
            center=(path.min(0)+path.max(0))/2;axes=(path.max(0)-path.min(0))/2
            source_navigation=np.r_[center+relative[:2]*axes,axes*relative[2:4],relative[4]]
        feature,diagnostic=circle_feature(raw,mask,path,packet['features'],source_navigation)
        diagnostic['navigation_source_anchor_indices']=anchor_indices.tolist()
        old=[f for f in original['features']if f['target_id']==TARGET]
        packet['features']=[f for f in packet['features']if f['target_id']!=TARGET]
        if feature:packet['features'].append(feature)
        packet['source'].update(mask_path=str(mask_path),mask_sha256=sha(mask_path))
        packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
        packet['diagnostics']['board_point_count']=sum(len(s['uv'])for f in packet['features']if'.print.'not in f['id']for s in f['segments'])
        packet['diagnostics']['printed_point_count']=packet['point_count']-packet['diagnostics']['board_point_count']
        packet['repair_revision']=dict(parent_candidate_sha256=sha(DATA/f'board_candidates/{index:06d}.json'),
                                      current_mask_sha256=sha(mask_path),reason='Ad11_outer_circle_missing_due_to_text_detail_profile_gates_and_false_person_occlusion',
                                      replaced_target_id=TARGET,other_current_candidate_features_preserved_exactly=True,diagnostic=diagnostic)
        write(destination/f'{index:06d}.json',packet)
        cv2.imwrite(str(diagnostics/f'Ad11_{index:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
        reports.append(dict(index=index,old_circle_points=sum(len(s['uv'])for f in old for s in f['segments']),**diagnostic))
    audit_path=REPAIR/'Ad11_source_repair_audit.json'
    previous=read(audit_path).get('rows',[])if audit_path.is_file()else[]
    replaced={r['index']for r in reports}
    combined=sorted([r for r in previous if r['index']not in replaced]+reports,key=lambda r:r['index'])
    write(audit_path,dict(schema='rinkspace.Ad11-current-source-circle-repair.v1',target_id=TARGET,
        model_geometry_changed=False,original_delivery_unchanged=True,rows=combined))
    print(json.dumps(reports,ensure_ascii=False),flush=True)


def render_source_pages():
    audit=read(REPAIR/'Ad11_source_repair_audit.json')
    destination=REPAIR/'Ad11_sequential_source_pages';destination.mkdir(exist_ok=True)
    archive=np.load(DATA/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    world=feature_world(read(MODEL/'scene.json'),read(MODEL/'feature_registry.json')['targets'][TARGET],read(MODEL/'ad2_material_template.json'))
    pages=[];canvas=np.zeros((960,1200,3),np.uint8)
    for position,row_record in enumerate(audit['rows']):
        index=row_record['index'];raw=cv2.imread(str(DATA/f'frames/{index:06d}.png'));path,depth=project_fast(cameras[index],world)
        center=(path.min(0)+path.max(0))/2;x0=int(center[0]-140);y0=int(center[1]-105)
        tile=np.zeros((240,300,3),np.uint8);sx0=max(0,x0);sx1=min(1920,x0+280);sy0=max(0,y0);sy1=min(1080,y0+210)
        tile[27+sy0-y0:27+sy1-y0,10+sx0-x0:10+sx1-x0]=raw[sy0:sy1,sx0:sx1]
        packet=read(REPAIR/f'board_candidates/{index:06d}.json')
        for feature in packet['features']:
            if feature['target_id']!=TARGET:continue
            for segment in feature['segments']:
                uv=np.asarray(segment['uv'])-[x0-10,y0-27]
                cv2.polylines(tile,[np.rint(uv).astype('int32')],bool(segment['closed']),(0,0,255),1,cv2.LINE_AA)
                for xy in uv:cv2.circle(tile,tuple(np.rint(xy).astype(int)),2,(0,0,255),-1,cv2.LINE_AA)
        label=f"{index:04d} Ad11 points={row_record.get('retained_points',0)}"
        cv2.putText(tile,label,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        row,column=divmod(position%16,4);canvas[row*240:(row+1)*240,column*300:(column+1)*300]=tile
        if position%16==15 or position==len(audit['rows'])-1:
            page=position//16+1;file=destination/f'page_{page:02d}.jpg';cv2.imwrite(str(file),canvas,[cv2.IMWRITE_JPEG_QUALITY,96])
            pages.append(dict(page=page,path=str(file),index_range=[audit['rows'][position-position%16]['index'],index]));canvas.fill(0)
    write(destination/'manifest.json',dict(pages=pages,source_native_crop_scale=1.,points_are_current_source_only=True))
    print('Ad11 sequential native crop pages',len(pages))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--indices');parser.add_argument('--render',action='store_true');args=parser.parse_args()
    if args.render:render_source_pages()
    else:repair([int(i)for i in args.indices.split(',')]if args.indices else None)
