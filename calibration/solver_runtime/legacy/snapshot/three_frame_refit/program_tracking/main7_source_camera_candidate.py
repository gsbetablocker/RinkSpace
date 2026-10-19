"""Shot 7 conditional pinhole resection from unchanged real source evidence.

This small diagnostic fixes the shared scene and records the centred principal
point, square pixels and zero distortion/skew as candidate assumptions. It
fits C/R/f on real circles, board outlines and known ad corners, preserving
pre-existing holdout rows. A data fit is not an accuracy acceptance.
"""
from __future__ import annotations
import argparse,json,hashlib,time,sys
from pathlib import Path
PROGRAM=Path(__file__).resolve().parent
if str(PROGRAM) not in sys.path:sys.path.insert(0,str(PROGRAM))
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from scipy.optimize import least_squares
import av
from projection import project_fast
from ice_refine import _world_curves,_curve_vector_error
from render_result import SceneRenderer,decode_selected,PersonMasks,DEFAULT_PEOPLE,DEFAULT_MASKS
from ad_landmarks import AdGeometry

ROOT=PROGRAM.parents[1]
B=ROOT/'outputs/three_camera_joint_20260930/run11_joint_C_LM_diagnostic/B_field/programme_review'
TMP=Path('D:/CodexData/RinkSpace_program_tracking_tmp')
SOURCES=ROOT/'evidence/three_camera_production_20260930/r2_source_audit'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as p:return {k:p[k].copy() for k in p.files}
def pinhole(q):
    c=np.zeros(25);c[:3]=q[:3]*100.;c[3:6]=q[3:6];c[6:8]=q[6];c[8:10]=[960,540];return c
def q7(c):return np.r_[c[:3]/100.,c[3:6],np.mean(c[6:8])]

def source_rows(scene):
    world=_world_curves(scene)
    # A classified installation can pan from a short end to the upper long
    # side. "Far" describes source-image visibility, not a fixed world Y half.
    # Keep every real finite perimeter segment, including lower short ends.
    boundary=np.asarray(scene['ice_boundary']['points_xyz'],float)
    top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
    world['far_contact']=boundary.copy() if np.array_equal(boundary[0],boundary[-1]) else np.vstack([boundary,boundary[0]])
    world['far_top']=top.copy() if np.array_equal(top[0],top[-1]) else np.vstack([top,top[0]])
    rows=[];bindings=[]
    for file in (SOURCES/'measurements/source_curves.json',SOURCES/'measurements/additional_source_cap.json',
                 SOURCES/'holdout_measurements/r2/source_curves.json',SOURCES/'holdout_measurements/additional_source_cap.json'):
        data=json.loads(file.read_text(encoding='utf8'));bindings.append(dict(path=str(file),sha256=sha(file)))
        for frame in data['frames']:
            for item in frame['curves']:
                name=item['id'];uv=np.asarray(item['source_uv'],float)
                if any(r['pts']==frame['source_pts'] and r['id']==name and r['split']==frame['split'] for r in rows):continue
                rows.append(dict(pts=int(frame['source_pts']),id=name,split=frame['split'],uv=uv,
                    world=world['far_top'] if name=='board_top' else world[name],
                    sigma=np.asarray(item['point_sigma_px'],float),origin=str(file)))
    for stamp in (230571776,230779136):
        file=TMP/f'main_curves_temporal_reviewed/{stamp}.npz';p=arrays(file);bindings.append(dict(path=str(file),sha256=sha(file)))
        for name in np.unique(p['physical_id']):
            if name=='ice_board_contact':nameworld='far_contact'
            else:nameworld=str(name)
            if nameworld not in world:continue
            uv=p['observed_uv'][p['physical_id']==name].copy()
            rows.append(dict(pts=stamp,id=str(name),split='fit',uv=uv,world=world[nameworld],
                sigma=np.full(len(uv),3.),origin=str(file)))
    ads=json.loads((TMP/'program_ad_corners_native/corners_joint_v9.json').read_text(encoding='utf8'))
    adfile=TMP/'program_ad_corners_native/corners_joint_v9.json';bindings.append(dict(path=str(adfile),sha256=sha(adfile)))
    points={}
    for item in ads['observations']:
        pts=int(item['pts'])
        if not 230518016<=pts<=230898432:continue
        if item.get('status')!='measured':continue
        ad=AdGeometry(scene,int(item['user_number']))
        points.setdefault(pts,[]).append(dict(uv=item['uv'],world=ad.corner(ad.initial,item['role']),sigma=float(item['sigma_px']),
            id=f"ad{item['user_number']}:{item['role']}"))
    for pts,items in points.items():
        rows.append(dict(pts=pts,id='known_ad_corners',split='fit',uv=np.array([a['uv'] for a in items],float),
            world=np.array([a['world'] for a in items],float),sigma=np.array([a['sigma'] for a in items]),
            origin=str(adfile),point_correspondence=True,physical_ids=[a['id'] for a in items]))
    # This physical lower-board edge was directly identified by the user.
    # Preserve its full source-normal blur interval and exact finite scene side;
    # it is not an invented point correspondence along the board.
    from main_nearboard_reviewed_source import PACKET,CONFIRMATION,PACKET_SHA
    from secondary_nearboard_top_source import lower_long_side_top
    confirmation=json.loads(CONFIRMATION.read_text(encoding='utf8'))
    if sha(PACKET)!=PACKET_SHA or confirmation['candidate_packet_sha256']!=PACKET_SHA:
        raise ValueError('User-reviewed near-board source binding changed')
    near=arrays(PACKET);near_world,near_geometry=lower_long_side_top(scene)
    bindings.extend([dict(path=str(PACKET),sha256=PACKET_SHA),dict(path=str(CONFIRMATION),sha256=sha(CONFIRMATION))])
    for i,stamp in enumerate(near['pts']):
        lo,hi=map(int,near['offsets'][i:i+2]);uv=near['observed_uv'][lo:hi].astype(float)
        normal=near['outward_normal'][lo:hi].astype(float);normal/=np.linalg.norm(normal,axis=1)[:,None]
        rows.append(dict(pts=int(stamp),id='reviewed_lower_board_top',split='fit',uv=uv,world=near_world,
            sigma=np.full(len(uv),4.*np.sqrt(len(uv))),origin=str(PACKET),
            normal=normal,normal_interval=near['normal_interval_px'][lo:hi].astype(float),geometry=near_geometry))
    edgefile=TMP/'programme_native_evidence_v13/driver_inputs/source_edges/shot07.npz';edge=arrays(edgefile)
    bindings.append(dict(path=str(edgefile),sha256=sha(edgefile)))
    fit_stamps=sorted({r['pts'] for r in rows if r['split']=='fit' and r['id']!='known_ad_corners'})
    for stamp in fit_stamps:
        i=np.flatnonzero(edge['pts']==stamp)
        if len(i)!=1:continue
        lo,hi=map(int,edge['offsets'][int(i[0]):int(i[0])+2]);uv=edge['observed_uv'][lo:hi];kind=edge['edge_id'][lo:hi]
        for number,name in ((0,'far_top'),(1,'far_contact')):
            # Keep native order/UV. A fixed source-domain far-board ROI avoids
            # lower-left foreground/near-ramp edges; no camera residual gate.
            take=(kind==number)&(uv[:,1]<650)&(uv[:,0]>40)&(uv[:,0]<1880)
            selected=uv[take][::3].copy()
            if len(selected)>5:
                rows.append(dict(pts=stamp,id=name,split='fit',uv=selected,world=world[name],
                    sigma=np.full(len(selected),4.),origin=str(edgefile),source_roi='40<x<1880,y<650; native stride3'))
    return rows,bindings

def value_jac(camera,rows):
    values=[];jac=[]
    for row in rows:
        uv,world=row['uv'],row['world'];scale=np.repeat(row['sigma'],2)
        if row.get('point_correspondence'):
            projected,depth,j=project_fast(camera,world,with_jacobian=True)
            r=(projected-uv).ravel()
        else:
            r,j=_curve_vector_error(camera,world,uv,with_jacobian=True)
        if 'normal_interval' in row:
            from source_edge_intervals import SourceNormalIntervals
            support=SourceNormalIntervals(row['normal'],row['normal_interval'],np.ones(len(uv),bool))
            r,matrices=support.apply(r,[j]);j=matrices[0]
        # project_fast already differentiates C/100 (its C columns include
        # x100); do not apply that chain a second time here.
        values.append(r/scale);jac.append(np.c_[j[:,:6],j[:,6]+j[:,7]]/scale[:,None])
    return np.concatenate(values),np.vstack(jac)

def value_only(camera,rows):
    """Same source objective without allocating unused trial-step Jacobians."""
    values=[]
    for row in rows:
        if row.get('point_correspondence'):
            r=(project_fast(camera,row['world'])[0]-row['uv']).ravel()
        else:r=_curve_vector_error(camera,row['world'],row['uv'])
        if 'normal_interval' in row:
            from source_edge_intervals import SourceNormalIntervals
            support=SourceNormalIntervals(row['normal'],row['normal_interval'],np.ones(len(row['uv']),bool))
            r=support.apply(r)
        values.append(r/np.repeat(row['sigma'],2))
    return np.concatenate(values)

def errors(camera,rows):
    result=[]
    for row in rows:
        if row.get('point_correspondence'):r=(project_fast(camera,row['world'])[0]-row['uv']).ravel()
        else:r=_curve_vector_error(camera,row['world'],row['uv'])
        e=np.linalg.norm(r.reshape(-1,2),axis=1)
        item=dict(pts=row['pts'],id=row['id'],split=row['split'],points=len(e),
            median_px=float(np.median(e)),p95_px=float(np.percentile(e,95)),maximum_px=float(e.max()))
        if 'normal_interval' in row:
            from source_edge_intervals import SourceNormalIntervals
            support=SourceNormalIntervals(row['normal'],row['normal_interval'],np.ones(len(row['uv']),bool))
            supported=np.linalg.norm(support.apply(r).reshape(-1,2),axis=1)
            item['outside_source_support_median_px']=float(np.median(supported))
            item['outside_source_support_p95_px']=float(np.percentile(supported,95))
        result.append(item)
    return result

def optimize(initial,rows):
    cache=[None,None]
    def both(q):
        if cache[0] is None or not np.array_equal(q,cache[0]):cache[:]=[q.copy(),value_jac(pinhole(q),rows)]
        return cache[1]
    lower=np.array([-np.inf,-np.inf,.001,-np.inf,-np.inf,-np.inf,np.log(100.)])
    upper=np.array([np.inf,np.inf,1.,np.inf,np.inf,np.inf,np.log(50000.)])
    linear=least_squares(lambda q:both(q)[0],q7(initial),jac=lambda q:both(q)[1],bounds=(lower,upper),
        loss='linear',max_nfev=120,x_scale='jac',ftol=1e-9,xtol=1e-10,gtol=1e-8)
    robust=least_squares(lambda q:both(q)[0],linear.x,jac=lambda q:both(q)[1],bounds=(lower,upper),
        loss='huber',f_scale=2,max_nfev=80,x_scale='jac',ftol=1e-9,xtol=1e-10,gtol=1e-8)
    return pinhole(robust.x),dict(cost=float(robust.cost),linear_nfev=linear.nfev,robust_nfev=robust.nfev,
        optimality=float(robust.optimality),success=bool(robust.success))

def common_centre(candidates,rows,stamps):
    q0=np.r_[np.median(np.array([c[:3] for c in candidates.values()]),axis=0)/100.,
             np.concatenate([q7(candidates[p])[3:] for p in stamps])]
    source=[[r for r in rows if r['pts']==p and r['split']=='fit'] for p in stamps]
    def cameras(q):return {p:pinhole(np.r_[q[:3],q[3+4*i:7+4*i]]) for i,p in enumerate(stamps)}
    def both(q):
        values=[];blocks=[]
        for i,p in enumerate(stamps):
            v,j=value_jac(pinhole(np.r_[q[:3],q[3+4*i:7+4*i]]),source[i])
            block=np.zeros((len(v),len(q)));block[:,:3]=j[:,:3];block[:,3+4*i:7+4*i]=j[:,3:]
            values.append(v);blocks.append(block)
        return np.concatenate(values),np.vstack(blocks)
    cache=[None,None]
    def cached(q):
        if cache[0] is None or not np.array_equal(q,cache[0]):cache[:]=[q.copy(),both(q)]
        return cache[1]
    lower=np.full(len(q0),-np.inf);upper=np.full(len(q0),np.inf);lower[2]=.001;upper[2]=1.
    lower[6::4]=np.log(100.);upper[6::4]=np.log(50000.)
    result=least_squares(lambda q:cached(q)[0],q0,jac=lambda q:cached(q)[1],bounds=(lower,upper),
        loss='huber',f_scale=2,max_nfev=150,x_scale='jac',ftol=1e-9,xtol=1e-10,gtol=1e-8)
    return cameras(result.x),dict(C=(result.x[:3]*100).tolist(),cost=float(result.cost),nfev=result.nfev,
        optimality=float(result.optimality),success=bool(result.success),
        observations=[dict(pts=p,errors=errors(cameras(result.x)[p],source[i])) for i,p in enumerate(stamps)])

def fit_rf(initial,rows,previous=None,a=None,b=None):
    if previous is not None:
        from secondary_source_ray_candidate import camera_from_matches
        initial,_=camera_from_matches(previous,a,b)
    q0=q7(initial)
    source=list(rows)
    if previous is not None:
        rot=cv2.Rodrigues(previous[3:6])[0];ray=np.c_[(a-previous[8:10])/np.exp(previous[6]),np.ones(len(a))]@rot
        source.append(dict(uv=b,world=previous[:3]+ray,sigma=np.full(len(a),max(2.,np.sqrt(len(a))/3.)),point_correspondence=True))
    def both(q):
        c=pinhole(np.r_[q0[:3],q]);v,j=value_jac(c,source);return v,j[:,3:]
    cache=[None,None]
    def cached(q):
        if cache[0] is None or not np.array_equal(q,cache[0]):cache[:]=[q.copy(),both(q)]
        return cache[1]
    # An actual adjacent-frame ray resection supplies the starting view. Keep
    # absolute polishing inside a local numerical trust box, so an ambiguous
    # parallel wall cannot send an LM trial into the backward optical branch.
    # Active bounds are reported; they are not a statement about lens hardware.
    delta=np.array([.06,.06,.06,.10])
    result=least_squares(lambda q:cached(q)[0],q0[3:],jac=lambda q:cached(q)[1],loss='huber',f_scale=2.,
        bounds=(q0[3:]-delta,q0[3:]+delta),max_nfev=40,x_scale='jac',ftol=1e-8,xtol=1e-9,gtol=1e-7)
    c=pinhole(np.r_[q0[:3],result.x]);diagnostic=dict(cost=float(result.cost),nfev=result.nfev,optimality=float(result.optimality),
        local_polish_guard_active=result.active_mask.tolist(),local_polish_guard=[.06,.06,.06,.10])
    if previous is not None:
        e=np.linalg.norm(project_fast(c,source[-1]['world'])[0]-b,axis=1)
        diagnostic.update(motion_matches=len(e),motion_median_px=float(np.median(e)),motion_p95_px=float(np.percentile(e,95)))
    return c,diagnostic

def continuous(output,scene,packet,rows,common):
    from secondary_source_ray_candidate import tracks
    cv2.setNumThreads(1)
    groupfile=TMP/'source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment/exposure_groups.json'
    groups=[g for g in json.loads(groupfile.read_text(encoding='utf8'))['groups'] if g['shot_index']==7]
    canonical={int(p):int(g['full_pts'][0]) for g in groups for p in g['full_pts']};wanted=sorted(set(canonical.values()))
    pos={p:i for i,p in enumerate(wanted)}
    frames=np.lib.format.open_memmap(output/'source_gray.npy',mode='w+',dtype=np.uint8,shape=(len(wanted),540,960))
    src=json.loads((PROGRAM/'source_shots.json').read_text(encoding='utf8'))
    start,end=src['shots'][7]['start_pts'],src['shots'][7]['end_pts']
    with av.open(src['source_path']) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';container.seek(start-15360,stream=stream,backward=True)
        decoded=set()
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts<start:continue
            if frame.pts>=end:break
            if int(frame.pts) in pos:
                frames[pos[int(frame.pts)]]=cv2.resize(frame.to_ndarray(format='gray'),(960,540));decoded.add(int(frame.pts))
    if decoded!=set(wanted):raise ValueError('Shot7 canonical source decode lacks actual PTS')
    people={int(r['pts']):r['boxes'] for r in json.loads(DEFAULT_PEOPLE.read_text(encoding='utf8'))['frames']}
    edge=arrays(TMP/'programme_native_evidence_v13/driver_inputs/source_edges/shot07.npz')
    edgeidx={int(p):i for i,p in enumerate(edge['pts'])}
    far=np.asarray(scene['ice_boundary']['points_xyz'],float)
    top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
    if not np.array_equal(far[0],far[-1]):far=np.vstack([far,far[0]])
    if not np.array_equal(top[0],top[-1]):top=np.vstack([top,top[0]])
    observations={p:[] for p in wanted}
    for row in rows:
        if row['split']!='fit' or row['pts'] not in canonical:continue
        rep=canonical[row['pts']]
        if row['id'] in ('far_top','far_contact','ice_board_contact','board_top'):continue
        if not any(r['id']==row['id'] for r in observations[rep]):observations[rep].append(row)
    for p in wanted:
        i=edgeidx[p];lo,hi=map(int,edge['offsets'][i:i+2]);uv=edge['observed_uv'][lo:hi];k=edge['edge_id'][lo:hi]
        for n,name,w in ((0,'far_top',top),(1,'far_contact',far)):
            selected=uv[(k==n)&(uv[:,1]<650)&(uv[:,0]>40)&(uv[:,0]<1880)][::6]
            if len(selected)>=3:observations[p].append(dict(pts=p,id=name,split='fit',uv=selected,world=w,sigma=np.full(len(selected),4.)))
    anchor=canonical[230571776];camera_by={anchor:common[230571776].copy()};diags=[];pairs=[]
    lookup={int(p):i for i,p in enumerate(packet['pts'])};common_C=common[230571776][:3]
    def propagate(sequence):
        for pa,pb in zip(sequence[:-1],sequence[1:]):
            a,b=tracks(frames[pos[pa]],frames[pos[pb]],pa,pb,people)
            c,diagnostic=fit_rf(camera_by[pa],observations[pb],camera_by[pa],a,b)
            camera_by[pb]=c;diags.append(dict(source_pts=pa,target_pts=pb,absolute_ids=[r['id'] for r in observations[pb]],**diagnostic));pairs.append((pa,pb,a,b))
            if len(diags)%50==0:
                np.savez_compressed(output/'partial_source_candidate_cameras.npz',pts=np.array(sorted(camera_by)),camera25=np.array([camera_by[p] for p in sorted(camera_by)]))
                (output/'partial_source_diagnostics.json').write_text(json.dumps(diags,indent=2),encoding='utf8')
            if len(diags)%150==0:print(json.dumps(dict(stage='source_rf',canonical_done=len(diags)+1,canonical_total=len(wanted))),flush=True)
    loc=wanted.index(anchor);propagate(wanted[loc:]);propagate(wanted[:loc+1][::-1])
    pts=np.array(sorted(canonical),np.int64);cameras=np.array([camera_by[canonical[int(p)]] for p in pts])
    np.savez_compressed(output/'continuous_candidate_cameras.npz',pts=pts,camera25=cameras,
        exposure_representative_pts=np.array([canonical[int(p)] for p in pts]),
        source_sha256=src['source_sha256_from_timeline'],model_sha256=sha(B/'candidate_scene.json'),
        status=np.full(len(pts),'source_pinhole_candidate_absolute_unvalidated'))
    np.savez_compressed(output/'source_motion_matches.npz',pair_pts=np.array([[a,b] for a,b,_,_ in pairs]),
        offsets=np.r_[0,np.cumsum([len(a) for _,_,a,_ in pairs])],
        source_uv=np.vstack([a for _,_,a,_ in pairs]),target_uv=np.vstack([b for _,_,_,b in pairs]))
    report=dict(native_frames=len(pts),canonical_frames=len(wanted),source_motion_rows=diags,
        source_exposure_group_sha256=sha(groupfile),fixed_C=common_C.tolist(),same_exposure_camera_exact=True,
        absfit=[dict(pts=r['pts'],id=r['id'],errors=errors(camera_by[canonical[r['pts']]],[r])) for r in rows if r['pts'] in canonical],
        source_consistency_is_not_absolute_accuracy=True,accuracy_accepted=False)
    (output/'continuous_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE);renderer=SceneRenderer(scene)
    inspect=sorted(set([start,end-256,230547968,230565376,230593280,230571776,230676480,230699520,230708224,230722560,230779136]))
    for stamp,bgr in decode_selected(Path(src['source_path']),inspect):
        mask=masks.get(stamp,require_sam2=True)[0]
        old,_=renderer.draw(bgr,packet['camera25'][lookup[stamp]],mask,title='Saved B / existing failure retained')
        new,_=renderer.draw(bgr,camera_by[canonical[stamp]],mask,title='Shared C pinhole + real source motion / candidate unvalidated')
        cv2.imwrite(str(output/f'{stamp}_continuous_before_after.jpg'),np.vstack((cv2.resize(old,(1280,720)),cv2.resize(new,(1280,720)))))
    print(json.dumps(dict(stage='continuous_complete',native_frames=len(pts),canonical_frames=len(wanted))),flush=True)
    return report

def run(output,run_continuous=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    scene=json.loads((B/'candidate_scene.json').read_text(encoding='utf8'));packet=arrays(B/'candidate_poses.npz')
    rows,bindings=source_rows(scene);lookup={int(p):i for i,p in enumerate(packet['pts'])}
    selected=[230571776,230676480,230699520,230722560,230779136]
    candidates={};fits=[];started=time.perf_counter()
    for stamp in selected:
        observations=[r for r in rows if r['pts']==stamp and r['split']=='fit']
        camera,metrics=optimize(packet['camera25'][lookup[stamp]],observations)
        candidates[stamp]=camera
        fits.append(dict(pts=stamp,camera25=camera.tolist(),**metrics,old=errors(packet['camera25'][lookup[stamp]],observations),
            candidate=errors(camera,observations)))
        print(json.dumps(dict(stage='single_resection',pts=stamp,C=camera[:3].tolist(),f=float(np.exp(camera[6])),**metrics)),flush=True)
    np.savez_compressed(output/'absolute_key_cameras.npz',pts=np.array(selected,np.int64),camera25=np.array([candidates[p] for p in selected]))
    # The 109s centre-line/board view alone has weak optical-depth freedom;
    # estimate C from the four spatially informative R2/B2 views, then resect
    # that later frame only in R/f using the resulting common physical C.
    common,shared_report=common_centre(candidates,rows,selected[:4])
    shared_report['late_depth_weak_keyframe']=dict(pts=selected[-1],used_to_fit_C=False,
        reason='centre-line/board source has weak optical-depth freedom; handled by continuous source resection')
    np.savez_compressed(output/'common_C_key_cameras.npz',pts=np.array(selected[:4],np.int64),camera25=np.array([common[p] for p in selected[:4]]))
    (output/'common_C_key_report.json').write_text(json.dumps(shared_report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report=dict(schema='rinkspace.main7-pinhole-source-resection.v1',method='independent7dC/R/f candidate; no production change',
        scene_fixed=True,scene_path=str(B/'candidate_scene.json'),scene_sha256=sha(B/'candidate_scene.json'),
        original_camera_sha256=sha(B/'candidate_poses.npz'),source_bindings=bindings,
        lens_assumptions=['principal_point_960_540','fx_equals_fy','zero_skew','zero_Brown'],
        fit_stamps=selected,all_source_groups=[dict(pts=r['pts'],id=r['id'],split=r['split'],points=len(r['uv']),origin=r['origin']) for r in rows],
        fits=fits,accuracy_accepted=False,seconds=time.perf_counter()-started)
    (output/'absolute_key_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE);renderer=SceneRenderer(scene)
    source=json.loads((PROGRAM/'source_shots.json').read_text(encoding='utf8'))
    for stamp,bgr in decode_selected(Path(source['source_path']),selected):
        mask=masks.get(stamp,require_sam2=True)[0]
        old,_=renderer.draw(bgr,packet['camera25'][lookup[stamp]],mask,title='Saved B / existing failure retained')
        new,_=renderer.draw(bgr,candidates[stamp],mask,title='7d pinhole real-source resection / candidate, unvalidated')
        cv2.imwrite(str(output/f'{stamp}_before_after.jpg'),np.vstack((cv2.resize(old,(1280,720)),cv2.resize(new,(1280,720)))))
    print(json.dumps(dict(stage='complete_keys',seconds=report['seconds'],output=str(output))),flush=True)
    if run_continuous:continuous(output,scene,packet,rows,common)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--continuous',action='store_true');a=p.parse_args();run(a.output,a.continuous)
