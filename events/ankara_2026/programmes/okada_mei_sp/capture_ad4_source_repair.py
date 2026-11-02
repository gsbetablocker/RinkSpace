"""Reread visible Ad4 edges and its known ANKARA shape from original RGB.

Native projection is search navigation only. Actual observations are public RGB
transition q50 readings; source photographs, masks and rejected rows are kept.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, time
from collections import Counter
import cv2
import numpy as np
from programme_source_frames import iter_registration_source_frames
from capture_independent_materials import spaced_indices
from capture_independent_ad4_glyph_gap import read_public, save_profiles, safe, write
from build_independent_shot_packets import build
from export_native_six_review import _manifest_for_library, _verify_native_library_binding
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.print_assets import evaluate_print_path
from rinkspace.session.print_patterns import PrintPattern
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

RIGHT='print_instance/Ad4::Ad4.right_vertical/path0'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source_translation_seed(pattern,image,H):
    """Whole photographed word search; the resulting H is navigation only."""
    q=cv2.perspectiveTransform(pattern.points[None].astype(float),H)[0]
    lo=np.floor(q.min(0)-2).astype(int);hi=np.ceil(q.max(0)+3).astype(int)
    if np.any(lo<0) or np.any(hi>[1920,1080]) or np.any(hi-lo<8):return None
    shift=np.eye(3);shift[:2,2]=-lo;size=tuple(hi-lo)
    template=cv2.warpPerspective(pattern.ink,shift@H,size)
    a=np.maximum(lo-[40,20],[0,0]);b=np.minimum(hi+[40,20],[1920,1080])
    target=pattern.ink_signal(image[a[1]:b[1],a[0]:b[0]]).astype(np.float32)
    score=cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED)
    _,correlation,_,at=cv2.minMaxLoc(score)
    correction=np.eye(3);correction[:2,2]=a+np.asarray(at)-lo
    return correction@H,dict(correlation=float(correlation),translation_px=correction[:2,2].tolist(),used_as_source_measurement=False)

def right_readings(image, mask, old_uv, binding):
    """Locate a blue-to-exterior-white flank, then read its actual luminance."""
    imagef=image.astype(np.float32); queries=[]; proofs=[]
    # A lone old sample can be on the horizontal bottom rather than the side.
    # Search the blue material above it as well; these are queries, not UVs.
    navigation=old_uv
    if len(old_uv)==1:
        navigation=np.vstack([old_uv,old_uv-[0.,24.],old_uv-[0.,48.]])
    for uv in navigation:
        x,y=uv; yi=int(round(y)); lo=max(1,int(x)-75); hi=min(1918,int(x)+86)
        if yi<3 or yi>1076 or hi-lo<30:continue
        strip=imagef[yi-1:yi+2,lo:hi].mean(0)
        blue=strip[:,0]-strip[:,2]; light=strip.min(1)
        # A white letter is not the exterior: require an extended white region
        # to its right and actual blue material immediately to its left.
        candidates=[]
        for k in range(12,len(strip)-25):
            left=strip[k-10:k-4];right=strip[k+4:k+24]
            if (np.median(left[:,0]-left[:,2])<9 or np.median(right.max(1)-right.min(1))>18
                    or np.median(right.min(1))<145 or np.mean(right.min(1)>140)<.85):continue
            exterior=imagef[max(0,yi-7):min(1080,yi+8),lo+k+4:lo+k+24]
            if np.mean((exterior.max(2)-exterior.min(2)<18)&(exterior.min(2)>145))<.9:continue
            contrast=float(np.median(right.min(1))-np.median(left.min(1)))
            if contrast<15:continue
            rise=float(light[k+2:k+5].mean()-light[k-4:k-1].mean())
            if rise>3:candidates.append((rise,k,contrast))
        if not candidates:continue
        # In this known main-camera view Ad4's right exterior is the last
        # coherent blue-to-white transition, not a strong white letter stroke.
        _,k,contrast=max(candidates,key=lambda c:c[1])
        q=np.array([lo+k,y]);xx=int(round(q[0]))
        if mask[yi-3:yi+4,max(0,xx-13):min(1920,xx+14)].any():continue
        queries.append(q);proofs.append(dict(old_uv=uv.tolist(),source_search_uv=q.tolist(),blue_white_contrast=contrast))
    if not queries:return [],dict(reason='no_source_blue_to_exterior_white_flank',queries=[])
    queries=np.asarray(queries);normals=np.tile([1.,0.],(len(queries),1))
    public=read_source_material_transition_profiles(np.ascontiguousarray(image[:,:,::-1]),binding=binding,
        search_points_uv=queries,normals=normals,search_intervals_px=np.tile([-10.,10.],(len(queries),1)),
        transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,
        engineering_support_margin_px=.25,keep_profiles=True)
    rows=[]
    for item,proof in zip(public['readings'],proofs):
        proof.update(source_flags=item['source_flags'],source_uv=np.asarray(item['actual_source_uv']).tolist())
        if any(item['source_flags'].values()) or np.asarray(item['actual_source_uv']).shape!=(2,):continue
        uv=np.asarray(item['actual_source_uv']);x,y=np.rint(uv).astype(int)
        if mask[max(0,y-3):y+4,max(0,x-14):x+15].any():continue
        rows.append(dict(path_id=RIGHT,kind=1,uv=uv.tolist(),normal=[1.,0.],
            actual_normal_quantiles=np.asarray(item['source_q10_q50_q90_normal_offsets_px']).tolist(),
            measurement_operator='actual_RGB_blue_panel_to_exterior_white_luminance_q50',
            source_profile_flags=item['source_flags'],model_UV_synthesized=False,fit_eligible=True))
    # A straight upright panel side must have one coherent visible source run.
    if len(rows)>=2:
        q=np.asarray([r['uv'] for r in rows]);order=np.argsort(q[:,1]);q=q[order]
        design=np.c_[q[:,1],np.ones(len(q))];line=np.linalg.lstsq(design,q[:,0],rcond=None)[0]
        if np.max(abs(design@line-q[:,0]))>3.:rows=[]
        else:
            ordered=[rows[i] for i in order]
            rows=[ordered[i] for i in spaced_indices(q,spacing=48.)]
    else:rows=[]
    return rows,dict(queries=proofs,public_readings=public['readings'],accepted=len(rows))

def word_reference(manifest,state,core):
    scene_path=Path(manifest['scene_path']);scene=read(scene_path)
    ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4)
    asset_path=(scene_path.parent/ref['path']).resolve();asset=read(asset_path)
    template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
    paths=[p for p in template['paths'] if '.ANKARA.' in p['id'] and not p.get('source_support_parent')]
    origin=paths[0]['shape_provenance'];source=Path(origin['source_frame']['path'])
    if sha(source)!=origin['source_frame']['sha256']:raise ValueError('Known word source changed')
    image=cv2.imread(str(source));atlas=dict(ad='Ad4',image=str(source),source_image=str(source),
        image_sha256=sha(source),source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),paths=[])
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    parameter_state=ParameterState(model.initial_state.layout,state['scene_values'])
    board=parameter_state.apply_scene(scene)['board'];world=[];reference=[]
    instance=ref.get('instance_id') or ref['instance_ids'][0]
    for p in paths:
        q=np.asarray(p['shape_provenance']['source_native_boundary'],float)
        uv=np.asarray(p['material_uv'],float)
        if len(q)!=len(uv):raise ValueError('Source contour and native seed vertex binding differs')
        if not np.array_equal(q[0],q[-1]):q=np.vstack([q,q[0]])
        stations=np.linspace(0,1,len(q))
        data=evaluate_print_path(asset,instance,p['id'],board,core,stations,parameter_state=parameter_state,jacobian=False)
        take=np.arange(0,len(q),max(1,len(q)//60))
        world.extend(data['xyz'][take]);reference.extend(q[take])
        atlas['paths'].append(dict(path_id=p['id'],native_path_id=instance+'::'+p['id'],pixels=q.tolist(),
            source_closed=True,role=p['role'],parent_contour_key=p.get('parent_contour_key'),
            proof=dict(kind='current_native_shape_source',asset=dict(path=str(asset_path),sha256=sha(asset_path)))))
    return PrintPattern(atlas,image),np.asarray(reference),np.asarray(world),atlas


def right_reference(manifest,state,core):
    """The finite physical side supplies search locations, never source UV."""
    scene_path=Path(manifest['scene_path']);scene=read(scene_path)
    ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4)
    asset=read((scene_path.parent/ref['path']).resolve())
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    parameter_state=ParameterState(model.initial_state.layout,state['scene_values'])
    board=parameter_state.apply_scene(scene)['board']
    instance=ref.get('instance_id') or ref['instance_ids'][0]
    return evaluate_print_path(asset,instance,'Ad4.right_vertical/path0',board,core,
        np.linspace(.08,.92,33),parameter_state=parameter_state,jacobian=False)['xyz']

def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);cv2.setNumThreads(2);started=time.monotonic()
    m=_manifest_for_library(read(a.prepared/'joint_manifest.json'),a.native_library)
    mapping=read(a.prepared/'mapping.json');z=np.load(a.prepared/'point_observations.npz',allow_pickle=False)
    pts=z['pts'];uv=z['uv'];path=z['path_indices'];state=dict(np.load(a.state,allow_pickle=False))
    _verify_native_library_binding(m,state)
    if sha(a.prepared/'point_observations.npz')!=str(state['point_input_sha256']):raise ValueError('Input differs')
    core=NativeCore(m['native_library']);timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    shot=next(s for s in m['shots'] if s['id']==a.shot);rig=state['rig'][shot['camera_index']]
    right_index=mapping['path_ids'].index(RIGHT)
    selected=np.asarray(a.pts if a.pts else range(round(a.first*50)*256,round(a.last*50)*256+256,256),np.int64)
    pattern,refpoints,world,atlas=word_reference(m,state,core)
    right_world=right_reference(m,state,core) if a.right else None
    write(a.output/'word_reference.json',atlas)
    folder=a.output/'qualified'/a.shot;folder.mkdir(parents=True);proofs=a.output/'profiles';proofs.mkdir()
    protected={str(a.state):sha(a.state),m['native_library']:sha(m['native_library']),str(a.prepared/'point_observations.npz'):m['point_input_sha256'],
        str(Path(__file__).resolve()):sha(__file__),str(Path(PrintPattern.register.__code__.co_filename).resolve()):sha(PrintPattern.register.__code__.co_filename)}
    records=[];decisions=[];total=Counter();word_prev=None
    representative=set(a.pts if a.pts and len(a.pts)<=12 else [selected[0],selected[-1],round(184*50)*256,round(186.6*50)*256,round(127.8*50)*256,round(128.2*50)*256])
    with (folder/'source_rows.jsonl').open('x',encoding='utf-8') as output,(a.output/'source_audit.jsonl').open('x',encoding='utf-8') as audit:
      for frame in iter_registration_source_frames(a.shot,bootstrap_root=a.bootstrap,selected_pts=selected):
        pt=frame['pts'];image=frame['image'];mask=frame['source_exclusion_mask'];rows=[];wr=None;rr=None
        binding=dict(source_sha256=frame['source_video_sha256'],source_pts=pt,shot_id=a.shot,time_base='1/12800',decoded_native_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image[:,:,::-1]).tobytes()).hexdigest())
        oldix=np.flatnonzero((pts==pt)&(path==right_index))
        if a.right:
            t=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
            side=core.project_motion(rig,state['controls'],timeline.first,timeline.dt,1080,t,right_world)
            side_uv=side['uv']
            visible=(side['status']==0)&np.isfinite(side_uv).all(1)&(side_uv[:,0]>=12)&(side_uv[:,0]<1908)&(side_uv[:,1]>=8)&(side_uv[:,1]<1072)
            navigation=side_uv[visible]
            # Query all visible parts, including frames with no old source rows.
            # Dense queries only locate real RGB transitions; retained readings
            # still use the common 48 px contour spacing and foreground mask.
            if len(navigation):
                order=np.argsort(navigation[:,1]);navigation=navigation[order]
                keep=np.r_[True,np.diff(np.floor(navigation[:,1]/3))>0]
                navigation=navigation[keep]
            new,rr=right_readings(image,mask,navigation,binding);rows.extend(new)
            rr['finite_native_side_navigation_only']=True
            rr['old_rows_required']=False
            newq=np.asarray([r['uv'] for r in new])
            for i in oldix:
                x,y=np.rint(uv[i]).astype(int);patch=image[max(0,y-2):y+3,max(0,x-4):min(1920,x+5)].astype(float)
                inside_blue=(patch[:,:,0]-patch[:,:,2]>8).mean()>.9 and np.std(patch.min(2))<4
                if not len(new) and (inside_blue or mask[y,x]):
                    decisions.append(dict(parent_row=int(i),pts=pt,uv=uv[i].tolist(),source_BGR_sha256=frame['source_BGR_sha256'],
                        reason='Source pixel lies inside uniform blue material without a visible exterior boundary.' if inside_blue else 'Same-exposure foreground obscures the claimed panel boundary.',
                        source_review_disposition='confirmed_not_target_material_boundary' if inside_blue else 'source_material_support_occluded'))
        if a.ankara:
            t=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
            projected=core.project_motion(rig,state['controls'],timeline.first,timeline.dt,1080,t,world)
            good=(projected['status']==0)&np.isfinite(projected['uv']).all(1)
            H=None
            if good.sum()>=4:H,_=cv2.findHomography(refpoints[good],projected['uv'][good],0)
            if H is not None:
                seed=source_translation_seed(pattern,image,H)
                search_H=H if seed is None else seed[0]
                wr=pattern.register(image,mask.astype('uint8'),search_H)
                wr['photographic_translation_navigation']=None if seed is None else seed[1]
                if wr['status']=='source_pattern_registered':
                    unthinned,readout,captured=read_public(pattern,image,mask,wr,binding)
                    profile=proofs/f'{pt}_word.npz';save_profiles(profile,captured)
                    wr['source_readout']=readout;wr['profile_path']=str(profile)
                    for name in sorted({r['path_id'] for r in unthinned}):
                        group=[r for r in unthinned if r['path_id']==name]
                        for i in spaced_indices([r['uv'] for r in group],spacing=48.):
                            row=group[i];row.update(kind=1,fit_eligible=True,model_UV_synthesized=False,
                                measurement_operator='known_ANKARA_shape_original_RGB_ink_q50',source_profile_proof=dict(path=str(profile),sha256=sha(profile)))
                            rows.append(row)
        proof=proofs/f'{pt}_right.json'
        if rr is not None:write(proof,rr)
        record={k:frame[k] for k in ('pts','source_video_sha256','source_BGR_sha256','mask_path','mask_sha256')}
        record.update(shot_id=a.shot,rows=rows,fit_eligible=bool(rows),review_state='current_original_RGB_known_Ad4_material',native_projection_used_for_search_only=True)
        output.write(json.dumps(safe(record),ensure_ascii=False,allow_nan=False)+'\n')
        note=dict(pts=pt,source_BGR_sha256=frame['source_BGR_sha256'],right_profile=None if rr is None else str(proof),word_registration=wr,
            rows_by_path=dict(Counter(r['path_id'] for r in rows)),current_old_right_rows=oldix.tolist())
        audit.write(json.dumps(safe(note),ensure_ascii=False,allow_nan=False)+'\n')
        total.update(note['rows_by_path']);records.append(dict(pts=pt,rows=len(rows),paths=note['rows_by_path'],word_status=None if wr is None else wr['status']))
        if pt in representative:
            cv2.imwrite(str(a.output/f'{pt}_original.png'),image);im=image.copy()
            for r in rows:cv2.circle(im,tuple(np.rint(r['uv']).astype(int)),2,(0,0,255),-1)
            cv2.imwrite(str(a.output/f'{pt}_source_points.png'),im)
        if len(records)%25==0 or a.pts:print(json.dumps(dict(pts=pt,frames=len(records),rows=len(rows),total=sum(total.values()),word_status=records[-1]['word_status'])),flush=True)
    write(a.output/'source_withdrawal_candidates.json',dict(parent_input_sha256=m['point_input_sha256'],rows=decisions))
    shutil.copyfile(a.bootstrap/a.shot/'navigation/poses_reanchored.npz',folder/'poses.npz')
    write(a.output/'qualified/binding.json',dict(source_UV_synthesized=False,native_projection_navigation_only=True,fit_eligible=True,
        source_atlas=str(a.output/'word_reference.json'),source_atlas_sha256=sha(a.output/'word_reference.json')))
    if total:build(a.bootstrap,a.output/'qualified',a.output/'source_packets',shot_ids=[a.shot])
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
    write(a.output/'report.json',dict(frames=records,rows_by_path=dict(total),withdrawal_candidates=len(decisions),input_files=protected,
        optimizer_started=False,original_input_modified=False,source_UV_synthesized=False,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(complete=True,frames=len(records),rows=sum(total.values()),withdrawal_candidates=len(decisions),seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('prepared','state','native-library','bootstrap','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pts',type=int,nargs='+');p.add_argument('--first',type=float,default=170.);p.add_argument('--last',type=float,default=188.68)
    p.add_argument('--shot',choices=['s07','s09'],default='s09')
    p.add_argument('--right',action='store_true');p.add_argument('--ankara',action='store_true')
    main(p.parse_args())
