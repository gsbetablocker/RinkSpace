"""Recover source-qualified ad observations as an immutable baseline increment."""
from pathlib import Path
import sys,collections,json,hashlib,gzip,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_replace_package_s01 as package
import ad_color_review_s01 as color
sys.path.insert(0,str(b.ROOT/'src'))
from rinkspace.vision_source_shapes import read_source_material_transition_profiles
import numpy as np
import cv2

BASE=b.OUT
OUT=BASE/'increments/001'

def source_candidates(data,assets):
    index=b.read(BASE/'source_evidence_index.json')
    hint={int(k):v for k,v in index['hints'].items()}
    banned={int(k):v for k,v in index['withdrawals'].items()}
    banned.update({int(k):v for k,v in index['retired'].items()})
    for r in b.read(BASE/'boundary_identity_review.json')['changes']:
        if r['decision']=='exclude':banned[r['row']]=r
    previous={}
    for p in (BASE/'verified_final').glob('*.json'):
        d=b.read(p)
        for r in d['confirmed']:
            previous[r['row']]=dict(path=r['path_id'],proof_file=str(p),proof_sha256=b.sha(p),method='existing_actual_source_entity_correspondence',registration=d['registrations'][r['registration_index']],record=r)
    result={}
    for row in np.flatnonzero(data['ad']>0):
        row=int(row)
        if row in banned:continue
        n=int(data['ad'][row]);label=str(data['labels'][row]);options=[]
        if row in hint:
            options.append(dict(path=hint[row],method='existing_later_pointwise_source_identity',proof_file=str(BASE/'source_evidence_index.json')))
        if row in previous:options.append(previous[row])
        ids=assets[n]['aliases'].get(label,[])
        if len(ids)==1 and data['old_active'][row]:
            options.append(dict(path=ids[0],method='source_qualified_exact_entity_identity',proof_file=str(assets[n]['path'])))
        if data['kind'][row]==1 and label.split('.')[-1] in ('left_top','right_top','left_bottom','right_bottom') and data['old_active'][row]:
            edge='top_edge' if label.endswith('_top') else 'bottom_edge';ids=assets[n]['aliases'].get(f'Ad{n}.{edge}',[])
            if len(ids)==1:options.append(dict(path=ids[0],method='existing_named_physical_panel_corner',proof_file=str(assets[n]['path'])))
        if options:result[row]=options
    return result,banned


def region_ok(path,im,uv,purple):
    x,y=np.rint(uv).astype(int)
    if not (0<=x<im.shape[1] and 0<=y<im.shape[0]):return False
    if path.startswith('Ad4.') and ('ANKARA' in path or 'shot4_' in path):
        magenta=bool(purple[y,x]);return magenta if 'ANKARA' in path else not magenta
    if path.startswith('Ad2.print.'):
        patch=im[max(0,y-12):y+13,max(0,x-12):x+13].astype(float);B,G,R=patch[:,:,0],patch[:,:,1],patch[:,:,2]
        # A mislabeled red Japanese panel cannot inherit an ISU-SKATING identity.
        red=np.count_nonzero((R-G>35)&(R-B>30));return red<.25*patch.shape[0]*patch.shape[1]
    return True


def recover():
    if (OUT/'increment_manifest.json').exists():raise ValueError('Frozen increment cannot be rewritten; prepare a new increment')
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    baseline=b.read(BASE/'incomplete_baseline_manifest.json')
    if b.sha(package.BUNDLE/'point_observations.npz')!=baseline['point_input_sha256']:raise ValueError('Baseline changed')
    data=b.load();assets=b.assets();candidates,banned=source_candidates(data,assets)
    old=np.load(package.ADS/'point_observations.npz');oldrows=set(map(int,old['source_rows']))
    candidates={r:v for r,v in candidates.items() if r not in oldrows}
    vis=b.read(BASE/'visibility.json')['frames'];accepted=[];pending=collections.Counter();checks=0;start=time.monotonic()
    core,model,native=package.current_native();perframe=[]
    with gzip.open(OUT/'recovery_decisions.jsonl.gz','wt',encoding='utf-8') as log:
        for fi,frame in enumerate(vis):
            pt=frame['pts'];rows=np.array([r for r in np.flatnonzero((data['ad']>0)&(data['pts']==pt)) if int(r) in candidates],int)
            if not len(rows):continue
            im=cv2.imread(frame['image']);mask=cv2.imread(frame['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Actual source changed')
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);smooth=cv2.GaussianBlur(gray.astype(float),(3,3),.65)
            dx=cv2.Scharr(smooth,cv2.CV_64F,1,0)/32;dy=cv2.Scharr(smooth,cv2.CV_64F,0,1)/32
            rgb=np.ascontiguousarray(im[:,:,::-1]);purple=color.purple_region(im)
            good=[];choices=[];normal=[]
            for row in rows:
                row=int(row);uv=data['uv'][row];x,y=np.rint(uv).astype(int)
                if not (4<=x<1916 and 4<=y<1076) or mask[y,x]:pending['occluded_or_source_border']+=1;continue
                options=[o for o in candidates[row] if region_ok(o['path'],im,uv,purple)]
                if not options:pending['physical_advertisement_region_conflict']+=1;continue
                choice=options[0]
                # Prefer a same-frame image repair when it proves a different advertisement.
                for option in options:
                    if option['method']=='existing_actual_source_entity_correspondence' and option['path'].split('.')[0]!=choice['path'].split('.')[0]:choice=option
                if choice['path'].startswith('Ad4.') and any('ANKARA' in o['path'] for o in options):
                    choice=next(o for o in options if 'ANKARA' in o['path'])
                g=np.array([dx[y,x],dy[y,x]])
                if np.linalg.norm(g)<1.:
                    pending['current_pixel_has_no_resolved_local_transition']+=1;continue
                good.append(row);choices.append(choice);normal.append(g/np.linalg.norm(g))
            if not good:continue
            good=np.array(good);normals=np.array(normal)
            binding=dict(source_sha256=frame['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id='s01',event_id='ankara_2026',rig_id='main',installation_id='main',pixel_domain='native_distorted',width=1920,height=1080)
            profiles=read_source_material_transition_profiles(rgb,binding=binding,search_points_uv=data['uv'][good],normals=normals,
                search_intervals_px=np.tile([-4.,4.],(len(good),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.7,engineering_support_margin_px=.25)
            # This is a current material measurement, not a correlation or a word-level sharpness score.
            records=profiles['readings'] if isinstance(profiles,dict) and 'readings' in profiles else profiles
            if isinstance(records,dict):raise ValueError('Inspect public profile schema: '+str(records.keys()))
            gained=0
            for row,choice,profile in zip(good,choices,records):
                row=int(row);checks+=1;flags=profile['source_flags'];q=np.asarray(profile['source_q10_q50_q90_normal_offsets_px'])
                reason=None
                if any(flags.values()):reason='current_source_transition_unresolved'
                elif not q[0]-1.<=0<=q[2]+1.:reason='original_pixel_outside_actual_transition_support'
                if reason:
                    pending[reason]+=1
                    log.write(json.dumps(dict(row=row,pts=pt,decision='needs_review',reason=reason,identity=choice['path'],flags=flags,offsets=q.tolist()),allow_nan=True)+'\n');continue
                station=.5;native_kind=1;corner=None
                if data['kind'][row]==1:
                    bound=package.corner_binding(row,dict(path_id=choice['path']),data,assets,core,model)
                    if bound is None:pending['named_corner_binding_unresolved']+=1;continue
                    target,station,corner=bound;native_kind=0
                else:target=choice['path']
                n=int(target.split('.')[0][2:]);pid=f'print_instance/Ad{n}::{target}'
                if pid not in native.paths:raise ValueError('Retired native target')
                r=dict(source_row=row,stable_id=str(data['ids'][row]),pts=pt,path_id=pid,kind=native_kind,station=station,decision='add_existing_source_observation',identity_evidence={k:v for k,v in choice.items() if k not in ['record','registration']},
                    actual_source_profile=dict(offsets=q.tolist(),source_flags=flags,source_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256']),
                    original_UV_unchanged=True,native_geometry_used_for_identity=False,corner=corner)
                accepted.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');gained+=1
            perframe.append(dict(pts=pt,candidates=len(rows),added=gained))
            if fi%50==0:print(json.dumps(dict(stage='source_recovery',frame=fi+1,total=1248,added=len(accepted),pending=dict(pending),seconds=round(time.monotonic()-start,1))),flush=True)
    rows=np.array([r['source_row'] for r in accepted],np.int64);path_ids=list(native.paths);lookup={p:i for i,p in enumerate(path_ids)}
    np.savez_compressed(OUT/'additions.npz',source_rows=rows,path_indices=np.array([lookup[r['path_id']] for r in accepted],np.int32),stations=np.array([r['station'] for r in accepted]),pts=data['pts'][rows],uv=data['uv'][rows],kind=np.array([r['kind'] for r in accepted],np.uint8),observation_ids=data['ids'][rows])
    b.write(OUT/'mapping.json',dict(path_ids=path_ids,scene_revision=native.scene_revision,baseline_id=baseline['baseline_id'],baseline_point_input_sha256=baseline['point_input_sha256']))
    b.write(OUT/'recovery.json',dict(added_rows=len(rows),by_ad=dict(collections.Counter(r['path_id'].split('::')[0] for r in accepted)),pending_reasons=dict(pending),source_profiles_checked=checks,per_frame=perframe,
        frame_global_blur_or_template_correlation_veto_used=False,baseline_rewritten=False,original_source_pixels_unchanged=True,native_model_unchanged=True))
    print(json.dumps({k:v for k,v in b.read(OUT/'recovery.json').items() if k!='per_frame'}),flush=True)

if __name__=='__main__':recover()
