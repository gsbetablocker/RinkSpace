"""Read-only development evidence for the three exported S01 body states.

No solving, inferred observations, camera loading, or video production occurs here.
All comparisons use one ledger and one source-PTS denominator. Static examples do
not constitute continuous review or independent 3D validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

WINDOWS = ((24.5, 27.0), (32.2, 33.8), (35.2, 36.8), (37.4, 39.0), (40.0, 41.5))
KEY_PTS = (320512, 332288, 336128, 421888, 461056, round(38.22 * 12800), 521728, 341504, 458240)
GROUPS = dict(all=list(range(17)), head_torso=[*range(7), 11, 12], upper_limb=list(range(7, 11)), lower_limb=list(range(13, 17)))
MODES = ('reference', 'legacy', 'full')
LOCAL_JOINTS = dict(left_shoulder=16, right_shoulder=17, left_elbow=18, right_elbow=19,
                    left_knee=4, right_knee=5, left_ankle=7, right_ankle=8)
COCO_EDGES = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def stats(value, denominator=None):
    """Missing predictions remain in denominator, never silently improve a score."""
    value = np.asarray(value, float).ravel()
    finite = value[np.isfinite(value)]
    denominator = len(value) if denominator is None else int(denominator)
    return dict(n=denominator, finite_n=len(finite), missing_n=denominator-len(finite),
                median=float(np.median(finite)) if len(finite) else None,
                p95=float(np.percentile(finite,95)) if len(finite) else None,
                max=float(np.max(finite)) if len(finite) else None)


def rotation_degrees(a, b):
    relative = np.asarray(a) @ np.swapaxes(np.asarray(b), -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1)-1)/2
    antisymmetric=np.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),axis=-1)
    sine=np.linalg.norm(antisymmetric,axis=-1)/2
    return np.rad2deg(np.arctan2(sine,np.clip(cosine,-1,1)))


def window_masks(pts):
    seconds = np.asarray(pts)/12800.0
    masks = {f'w{i+1}': (seconds >= a-1e-9) & (seconds <= b+1e-9) for i,(a,b) in enumerate(WINDOWS)}
    masks['all_windows'] = np.logical_or.reduce(list(masks.values()))
    return masks


def body_statistics(prediction, observed, use, masks, pts):
    error = np.linalg.norm(prediction-observed, axis=-1)
    result = {}
    for name, mask in masks.items():
        groups = {}
        for group, ids in GROUPS.items():
            selected = use[:,ids] & mask[:,None]
            values = error[:,ids][selected]
            group_stats = stats(values)
            group_stats['frames_with_points'] = int(np.any(selected,axis=1).sum())
            group_stats['possible_points'] = int(mask.sum())*len(ids)
            if np.isfinite(values).any():
                f, j = np.nonzero(selected)
                worst = int(np.nanargmax(values))
                group_stats['worst'] = dict(pts=int(pts[f[worst]]), joint=int(np.asarray(ids)[j[worst]]))
            groups[group] = group_stats
        result[name] = dict(source_frames=int(mask.sum()), groups=groups)
    return result


def footwear_statistics(root, packets, pts, masks):
    # Reuse admitted source rows and the exact finite-segment distance/sampling.
    from .fullbody_fit import footwear_rows
    from .footwear_pixel_factors import nearest
    from .footwear_fit import sample_polyline
    from scipy.spatial import ConvexHull
    gear, sources = footwear_rows(root, np.arange(len(pts)), pts)
    records = []
    for kind, rows in (('boot_contour', gear.visible_rows), ('blade_curve', gear.blade_rows), ('heel_toe', gear.landmark_rows)):
        for row_number, row in enumerate(rows):
            fi = row['frame']
            if not masks['all_windows'][fi]:
                continue
            common = dict(kind=kind, row=row_number, pts=int(pts[fi]), foot=int(row['foot']),
                          physical_identity_status=row.get('physical_identity_status','existing track hypothesis'),
                          source_role='development', sigma_px=float(row['sigma_px']), modes={})
            if kind == 'boot_contour':
                source = np.asarray(row['observed'],float)
            elif kind == 'blade_curve':
                source = np.asarray(row['uv'],float)
            else:
                source = np.asarray([row['heel'],row['toe']],float)
            common['original_source_uv'] = source.tolist()
            for mode, packet in packets.items():
                projected = packet['equipment_uv'][fi,row['foot']]
                if not np.isfinite(projected).all():
                    missing=np.full(len(source),np.nan)
                    common['modes'][mode]=dict(original_source_points_px=missing.tolist(),summary_px=stats(missing),missing_prediction=True)
                    continue
                if kind == 'boot_contour':
                    if row.get('reverse_boundary',True):
                        raise ValueError('Expected source-only admitted partial boot contour')
                    line = projected[ConvexHull(projected[:14]).vertices]
                    distance = nearest(source,line,closed=True)[0]
                    values = dict(original_source_points_px=distance.tolist(), summary_px=stats(distance))
                elif kind == 'blade_curve':
                    sampled = sample_polyline(source,24)
                    candidates = (projected[14:31],projected[31:48])
                    candidate_distances = [nearest(sampled,line)[0] for line in candidates]
                    best = int(np.argmin([np.sum(d*d) for d in candidate_distances]))
                    distance = nearest(source,candidates[best])[0]
                    values = dict(original_source_points_px=distance.tolist(), summary_px=stats(distance),
                                  sampled24_px=candidate_distances[best].tolist(),
                                  sampled24_summary_px=stats(candidate_distances[best]),
                                  selected_candidate=('blade_left','blade_right')[best], source_side_confirmed=False,
                                  selection='one candidate for the entire source curve, selected by sampled24 squared distance')
                else:
                    distance = np.array([nearest(source[k:k+1],projected[ids])[0][0] for k,ids in enumerate(([0,7],[1,8]))])
                    values = dict(original_source_points_px=distance.tolist(), summary_px=stats(distance), names=['heel','toe'])
                common['modes'][mode] = values
            records.append(common)
    grouped = {}
    for name, mask in masks.items():
        allowed = set(map(int,pts[mask]))
        grouped[name] = {}
        for kind in ('boot_contour','blade_curve','heel_toe'):
            rows = [r for r in records if r['pts'] in allowed and r['kind']==kind]
            grouped[name][kind] = dict(source_records=len(rows), modes={mode:stats([d for r in rows for d in r['modes'][mode]['original_source_points_px']]) for mode in MODES})
    return dict(sources=sources, definition='original source points to same-semantic finite model boundary; no model-generated observations; source-only distances',
                blade_side='unknown physical side; curve-level finite candidate choice retained separately for each mode',
                empty_groups='n=0 means no admitted original source observations, not a zero-error success',
                grouped=grouped, records=records)


def geometric_statistics(root, packets, pts, masks, output):
    reference = packets['reference']
    parents = load(Path(root)/'compact_smplx.npz')['parents'][:22].astype(int)
    seconds = pts/12800.0
    camera_path = output/'camera_pose.npz'
    camera = load(camera_path) if camera_path.is_file() else None
    centre = None
    if camera is not None:
        camera_pts = camera.get('pts',camera.get('source_pts'))
        if camera_pts is None:
            raise ValueError('camera_pose.npz requires exact pts/source_pts')
        np.testing.assert_array_equal(camera_pts,pts)
        if 'metres_per_scene_unit' in camera:
            np.testing.assert_equal(camera['metres_per_scene_unit'],reference['metres_per_scene_unit'])
        centre = np.asarray(camera['centre'])*float(reference['metres_per_scene_unit'])
    result = {}
    for mode, packet in packets.items():
        P = packet['P_world_mid'];delta = P-reference['P_world_mid']
        G = packet.get('G_world_mid',np.column_stack((P[:,:2],np.zeros(len(P)))))
        refG = reference.get('G_world_mid',np.column_stack((reference['P_world_mid'][:,:2],np.zeros(len(P)))))
        velocity = np.gradient(P,seconds,axis=0,edge_order=2)
        acceleration = np.gradient(velocity,seconds,axis=0,edge_order=2)
        delta_velocity=np.gradient(delta,seconds,axis=0,edge_order=2)
        delta_acceleration=np.gradient(delta_velocity,seconds,axis=0,edge_order=2)
        lengths = np.linalg.norm(packet['native22_world_mid'][:,1:]-packet['native22_world_mid'][:,parents[1:]],axis=-1)
        blade_height=np.min(packet['blade_world_mid'][...,2],axis=2)
        bone = [dict(native_joint=j,parent=int(parents[j]),mean_m=float(np.mean(lengths[:,j-1])),
                     temporal_range_m=float(np.ptp(lengths[:,j-1])),std_m=float(np.std(lengths[:,j-1]))) for j in range(1,22)]
        groups = {}
        for name, mask in masks.items():
            groups[name] = dict(P_delta_norm_m=stats(np.linalg.norm(delta[mask],axis=-1)),
                G_delta_norm_m=stats(np.linalg.norm((G-refG)[mask],axis=-1)),
                P_delta_velocity_m_s=stats(np.linalg.norm(delta_velocity[mask],axis=-1)),
                P_delta_acceleration_m_s2=stats(np.linalg.norm(delta_acceleration[mask],axis=-1)),
                P_height_m=stats(P[mask,2]),P_speed_m_s=stats(np.linalg.norm(velocity[mask],axis=-1)),
                P_acceleration_m_s2=stats(np.linalg.norm(acceleration[mask],axis=-1)),
                blade_sample_minimum_height_m=dict(summary=stats(blade_height[mask]),
                    lowest_sample_m=float(np.min(blade_height[mask])),
                    foot_frames_below_frozen_plane=int(np.count_nonzero(blade_height[mask]<0)),
                    interpretation='saved lower-edge sample Z relative to native ice plane; not observed contact'),
                root_Q_change_deg=stats(rotation_degrees(packet['Q_world'][mask],reference['Q_world'][mask])),
                local_change_deg={joint:stats(rotation_degrees(packet['local_R_mid'][mask,index-1],reference['local_R_mid'][mask,index-1])) for joint,index in LOCAL_JOINTS.items()})
            if centre is not None:
                ref_ray = reference['P_world_mid']-centre
                ray = P-centre
                ref_ray = ref_ray/np.linalg.norm(ref_ray,axis=-1,keepdims=True)
                unit = ray/np.linalg.norm(ray,axis=-1,keepdims=True)
                radial = np.sum(delta*ref_ray,axis=-1)
                lateral = delta-radial[:,None]*ref_ray
                angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(ref_ray,unit),axis=-1),np.sum(ref_ray*unit,axis=-1)))
                groups[name]['camera_ray'] = dict(signed_radial_delta_m=stats(radial[mask]),lateral_delta_m=stats(np.linalg.norm(lateral[mask],axis=-1)),direction_change_deg=stats(angle[mask]))
        boundaries = []
        for wi,(a,b) in enumerate(WINDOWS):
            for label,when in (('start',a),('end',b)):
                index = int(np.argmin(np.abs(seconds-when)))
                indices = np.arange(max(0,index-1),min(len(pts),index+2))
                row = dict(window=f'w{wi+1}',boundary=label,requested_seconds=when,nearby_pts=pts[indices].tolist(),
                    P_delta_m=delta[indices].tolist(),
                    delta_velocity_m_s=delta_velocity[indices].tolist(),delta_acceleration_m_s2=delta_acceleration[indices].tolist(),
                    local_change_deg={joint:rotation_degrees(packet['local_R_mid'][indices,index-1],reference['local_R_mid'][indices,index-1]).tolist() for joint,index in LOCAL_JOINTS.items()},root_Q_change_deg=rotation_degrees(packet['Q_world'][indices],reference['Q_world'][indices]).tolist(),
                    P_step_m=np.linalg.norm(np.diff(P[indices],axis=0),axis=-1).tolist(),
                    delta_step_m=np.linalg.norm(np.diff(delta[indices],axis=0),axis=-1).tolist())
                boundaries.append(row)
        outside = ~masks['all_windows']
        result[mode] = dict(groups=groups,bone_lengths=bone,max_bone_temporal_range_m=max(r['temporal_range_m'] for r in bone),
            P_delta_outside_windows_m=stats(np.linalg.norm(delta[outside],axis=-1)),
            Q_delta_outside_windows_deg=stats(rotation_degrees(packet['Q_world'][outside],reference['Q_world'][outside])),
            local_delta_outside_windows_deg=stats(rotation_degrees(packet['local_R_mid'][outside],reference['local_R_mid'][outside])),
            outside_state_exact_equal={key:bool(np.array_equal(packet[key][outside],reference[key][outside])) for key in ('P_world_mid','Q_world','local_R_mid','native22_world_mid','gear_world_mid')},
            boundary_samples=boundaries)
    return dict(modes=result,length_unit='conditional metres under the frozen current scene scale',
        derivative_definition='anatomical pelvis finite differences at native source PTS; not measured centre-of-mass or blade speed',
        boundary_definition='adjacent source samples around the attachment boundary, not an analytic C2 proof',
        camera_ray=dict(available=centre is not None,path=str(camera_path) if centre is not None else None,
                        sha256=sha256(camera_path) if centre is not None else None,
                        camera_centre_unit='scene units multiplied by current metres_per_scene_unit',
                        note='direction from camera centre to anatomical pelvis; not gaze or ground-contact ray'))


def find_source(root):
    review_path = root/'next_round_20261029/J1_s2/source_review.json'
    if review_path.is_file():
        source = read_json(review_path).get('source',{})
        candidate = source.get('registered_path')
        if candidate and Path(candidate).is_file():
            return dict(path=candidate,registered_sha256=source.get('sha256'),sha256_status='registered identity, not rehashed in this report',record=str(review_path))
    project = Path(__file__).resolve().parents[3]
    record = project/'events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json'
    source = read_json(record)['primary_source']
    candidate = source['path']
    # A historical selection record contains doubled Windows separators.
    if not Path(candidate).is_file():
        candidate = candidate.replace('\\\\','\\')
    if not Path(candidate).is_file():
        raise FileNotFoundError(candidate)
    return dict(path=candidate,registered_sha256=source.get('sha256'),sha256_status='registered identity, not rehashed in this report',record=str(record))


def exact_frames(source, targets):
    import av
    frames = {}
    with av.open(source) as container:
        stream = container.streams.video[0]
        if (stream.time_base.numerator,stream.time_base.denominator)!=(1,12800):
            raise ValueError('Unexpected source time_base')
        for pts in targets:
            container.seek(int(pts),stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts==pts:
                    frames[pts] = frame.to_ndarray(format='rgb24')
                    break
                if frame.pts is not None and frame.pts>pts:
                    break
            if pts not in frames:
                raise ValueError(f'Exact requested source PTS missing: {pts}')
    return frames


def make_figures(root, output, packets, obs, pts, parents, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frames = exact_frames(source['path'],KEY_PTS)
    inp = load(root/'input.npz')
    example_groups = (KEY_PTS[:3],KEY_PTS[3:5],KEY_PTS[5:7],KEY_PTS[7:])
    excluded={(int(c['pts']),int(c['joint'])) for c in read_json(output/'ledger.json')['changes'] if c['adoption'].startswith('excluded')}
    artifacts = []
    for group_index, examples in enumerate(example_groups,1):
        fig, axes = plt.subplots(len(examples),3,figsize=(15,5*len(examples)),squeeze=False)
        for row, point in enumerate(examples):
            fi = int(np.searchsorted(pts,point));image = frames[point]
            if pts[fi]!=point:
                raise ValueError('Requested example absent from all-frame export')
            box = inp['source_bbox'][fi]
            lo = np.maximum(np.floor(box[:2]-45),[0,0]).astype(int)
            hi = np.minimum(np.ceil(box[2:]+45),[image.shape[1],image.shape[0]]).astype(int)
            crop = image[lo[1]:hi[1],lo[0]:hi[0]]
            for col,mode in enumerate(MODES):
                ax = axes[row,col];ax.imshow(crop,extent=(lo[0],hi[0],hi[1],lo[1]))
                uv = packets[mode]['COCO_uv'][fi];selected = obs['use'][fi]
                for a,b in COCO_EDGES:
                    ax.plot(uv[[a,b],0],uv[[a,b],1],color='#00d9ff',lw=1.3)
                ax.scatter(*obs['uv'][fi,selected].T,s=14,facecolors='none',edgecolors='#ffb000',linewidths=1.1)
                changed = np.linalg.norm(obs['uv'][fi]-obs['raw_coco_uv'][fi],axis=-1)>1e-5
                if changed.any():
                    ax.scatter(*obs['raw_coco_uv'][fi,changed].T,s=35,marker='x',c='#ff4488',linewidths=1.3)
                omitted=[j for j in range(17) if (point,j) in excluded]
                if omitted:
                    ax.scatter(*obs['raw_coco_uv'][fi,omitted].T,s=35,marker='x',c='#999999',linewidths=1.3)
                gear = packets[mode]['equipment_uv'][fi]
                for foot in range(2):
                    for boundary in (gear[foot,14:31],gear[foot,31:48]):
                        ax.plot(boundary[:,0],boundary[:,1],color='#6df783',lw=.8)
                err = np.linalg.norm(uv[selected]-obs['uv'][fi,selected],axis=-1)
                ax.set_title(f'{mode} | {point/12800:.2f}s | PTS {point}\nselected median {np.median(err):.1f}px',fontsize=10)
                ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.tick_params(labelsize=7)
        fig.suptitle('Frozen camera | cyan=model, orange=selected, pink x=changed original, gray x=excluded | development',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.98));dest=output/f'compare_source_{group_index}.png';fig.savefig(dest,dpi=140);plt.close(fig)
        artifacts.append(dict(path=str(dest),kind='exact_source_static',pts=list(examples),crop='one source_bbox crop per frame, identical across modes'))
        fig = plt.figure(figsize=(15,4.8*len(examples)))
        for row,point in enumerate(examples):
            fi = int(np.searchsorted(pts,point));origin = packets['reference']['P_world_mid'][fi]
            cloud = np.concatenate([packets[m]['native22_world_mid'][fi] for m in MODES])
            mins=cloud.min(0);maxs=cloud.max(0);centre=(mins+maxs)/2;extent=max(float(np.max(maxs-mins)),1.7)*.60
            for col,mode in enumerate(MODES):
                ax=fig.add_subplot(len(examples),3,row*3+col+1,projection='3d')
                native=packets[mode]['native22_world_mid'][fi];gear=packets[mode]['gear_world_mid'][fi]
                for j in range(1,22):
                    line=native[[parents[j],j]];ax.plot(*line.T,color='#1677bb',lw=2)
                for foot in range(2):
                    for line in (gear[foot,14:31],gear[foot,31:48]):ax.plot(*line.T,color='#2c9b46',lw=1.2)
                P=packets[mode]['P_world_mid'][fi];G=P.copy();G[2]=0
                ax.scatter(*P,c='#b61b26',s=30);ax.plot([P[0],G[0]],[P[1],G[1]],[P[2],G[2]],c='#777777',ls='--',lw=.8)
                ax.scatter(*G,c='#555555',s=15)
                ax.set_xlim(centre[0]-extent,centre[0]+extent);ax.set_ylim(centre[1]-extent,centre[1]+extent);ax.set_zlim(min(0,mins[2]-.1),maxs[2]+.1)
                ax.set_box_aspect((2*extent,2*extent,maxs[2]+.1-min(0,mins[2]-.1)))
                ax.view_init(elev=15,azim=-55);ax.set_xlabel('X m',fontsize=8);ax.set_ylabel('Y m',fontsize=8);ax.set_zlabel('Z m',fontsize=8)
                ax.tick_params(labelsize=7);ax.set_title(f'{mode} | {point/12800:.2f}s | PTS {point}',fontsize=10)
        fig.suptitle('Identical world axes per source frame | anatomical skeleton + rigid blade boundaries | conditional metres, development',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.98));dest=output/f'compare_3d_{group_index}.png';fig.savefig(dest,dpi=140);plt.close(fig)
        artifacts.append(dict(path=str(dest),kind='three_dimensional_static',pts=list(examples),axes='identical per source frame across all modes'))
    return artifacts


def acceptance_assessment(packets, masks, output):
    """Diagnose this saved candidate, without inventing independent accuracy."""
    full, legacy = packets['full'], packets['legacy']
    active = masks['all_windows'];pts = full['pts']
    height = np.min(full['blade_world_mid'][...,2],axis=2)
    rows=[]
    for name, mask in masks.items():
        ids=np.flatnonzero(mask);worst=int(ids[np.argmin(np.min(height[mask],axis=1))])
        rows.append(dict(window=name,frames=int(mask.sum()),
            frames_with_any_blade_below_plane=int(np.count_nonzero(np.any(height[mask]<-1e-6,axis=1))),
            frames_with_any_blade_more_than_1cm_below=int(np.count_nonzero(np.any(height[mask]<-.01,axis=1))),
            lowest_saved_sample_m=float(np.min(height[mask])),worst_source_pts=int(pts[worst]),
            worst_source_seconds=float(pts[worst]/12800)))
    delta=full['P_world_mid']-legacy['P_world_mid']
    difference=dict(distance_m=stats(np.linalg.norm(delta[active],axis=1)),
        meaning='difference between two fitted models, NOT a measured error or confidence interval')
    camera_path=Path(output)/'camera_pose.npz'
    if camera_path.is_file():
        camera=load(camera_path);np.testing.assert_array_equal(camera['pts'],pts)
        ray=full['P_world_mid']-camera['centre']*float(full['metres_per_scene_unit'])
        ray/=np.linalg.norm(ray,axis=1,keepdims=True)
        signed=np.sum(delta*ray,axis=1)
        difference.update(along_camera_ray_absolute_m=stats(np.abs(signed[active])),
            perpendicular_to_ray_m=stats(np.linalg.norm((delta-signed[:,None]*ray)[active],axis=1)))
    return dict(verdict='not_ready_for_3d_accuracy_acceptance',
        verdict_zh='当前结果不能验收为准确的三维人体动作与轨迹；已完成的是可运行工程与开发观测上的二维拟合改善。',
        engineering='implementation and playback checked separately; does not validate physical accuracy',
        pixel_evidence='improvement on development observations used for fitting and selection; not an independent final test',
        geometry='saved blade geometry penetrates the frozen native ice plane',
        plane_penetration=rows,
        plane_count_definition='Z of saved lower-edge samples under current conditional scene scale; 1 micrometre avoids roundoff. 1 cm is an additional descriptive count, not a pass tolerance. No contact labels inferred.',
        between_fitted_models=difference,
        independent_3d_accuracy='unknown: no independent 3D ground truth or frozen final validation set',
        visual_review_role='source identity and visible anomalies only; user is not asked to judge invisible depth by eye')


def report_text(metrics):
    assessment=metrics['acceptance_assessment']
    overall=next(row for row in assessment['plane_penetration'] if row['window']=='all_windows')
    difference=assessment['between_fitted_models']['distance_m']
    lines=['S01 五窗口全身对照（开发材料）','',assessment['verdict_zh'],
           f"自动几何检查：{overall['frames']} 帧中，{overall['frames_with_any_blade_more_than_1cm_below']} 帧至少一侧保存刀刃样点低于冻结冰面超过 1 cm；最深 {-overall['lowest_saved_sample_m']*100:.1f} cm，原片 {overall['worst_source_seconds']:.3f}s。1 cm 只作描述性计数，不是验收容差。",
           f"两份拟合的骨盆位置差异：legacy 与 full 的距离中位数 {difference['median']*100:.1f} cm，P95 {difference['p95']*100:.1f} cm；这是解之间的差异，不是实测误差或置信区间。",
           '播放器工程检查不能证明三维准确；当前无需用户凭肉眼判断深度或认定接冰通过。','',
           '同一相机、同一观测台账、同一 observations.use 分母。reference 为旧 J1 在当前相机重投影；原旧相机 UV 另列，不能作拟合改善。',
           '误差单位为原生 1920×1080 像素；三维为当前冻结场景尺度下的条件米制。以下为 median / P95 / max。','']
    for group in GROUPS:
        lines.append(group+':')
        for mode in MODES:
            s=metrics['body_pixels'][mode]['adopted']['all_windows']['groups'][group]
            fmt=lambda x:'缺失' if x is None else f'{x:.2f}'
            lines.append(f'  {mode}: n={s["n"]}, missing={s["missing_n"]}; '+ ' / '.join(fmt(s[k]) for k in ('median','p95','max')))
    lines += ['', '鞋/刀（原有源曲线顶点到同语义有限边界；n=0 即无数据）：']
    for kind,row in metrics['footwear']['grouped']['all_windows'].items():
        values=[]
        for mode in MODES:
            s=row['modes'][mode];values.append(f'{mode} n={s["n"]} median={s["median"]:.2f} P95={s["p95"]:.2f}' if s['n'] else f'{mode} n=0')
        lines.append('  '+kind+': '+'; '.join(values))
    worst=metrics['body_pixels']['full']['adopted']['all_windows']['groups']['all']
    legacy_worst=metrics['body_pixels']['legacy']['adopted']['all_windows']['groups']['all']
    lines += ['', f"尾部残差：全窗口 max 从 legacy {legacy_worst['max']:.2f}px 到 full {worst['max']:.2f}px；PTS {worst['worst']['pts']}，关节 {worst['worst']['joint']}。",
              'w2、w4、w5 无原有刀曲线观测（n=0），不能由鞋轮廓贴合推称刀刃定位通过。']
    lines += ['', '关键关联点（采用点残差 reference / legacy / full；另列 full 对原点残差）：']
    for row in metrics['key_frames']:
        changed=np.linalg.norm(np.asarray(row['adopted_uv'])-np.asarray(row['original_coco_uv']),axis=-1)>1e-5
        for joint in np.flatnonzero(changed):
            adopted=' / '.join(f"{row['modes'][m]['adopted_residual_px'][joint]:.1f}" for m in MODES)
            original=row['modes']['full']['original_residual_px'][joint]
            lines.append(f"  {row['seconds']:.2f}s j{joint}: {adopted}px；full 对原点 {original:.1f}px")
    lines += ['', '求解终止：']
    for mode,windows in metrics['solver_windows'].items():
        reasons={row.get('stopping_reason','unknown') for row in windows.values()}
        converged=sum(bool(row.get('numerically_converged',False)) for row in windows.values())
        lines.append(f"  {mode}: "+'; '.join(sorted(reasons))+f'；数值梯度收敛 {converged}/5 窗。终止条件不等于准确性通过。')
    lines += ['', '状态变化（相对 reference）：']
    for mode in ('legacy','full'):
        geo=metrics['geometry']['modes'][mode];g=geo['groups']['all_windows']
        lines.append(f'  {mode}: P 位移 P95={g["P_delta_norm_m"]["p95"]:.4f}m; root Q 角变 P95={g["root_Q_change_deg"]["p95"]:.2f}°; 骨长时间范围最大={geo["max_bone_temporal_range_m"]:.3g}m')
    sample=next(row for row in metrics['key_frames'] if row['pts']==461056)
    lines += ['', '36.020s（PTS 461056）两侧保存刀刃样点的最低 Z／相对冻结冰面：']
    for mode in MODES:
        heights=sample['modes'][mode]['blade_sample_minimum_height_m']
        lines.append(f'  {mode}: {heights[0]:+.4f} / {heights[1]:+.4f}m')
    lines.append('负值是当前模型穿入冻结冰面，未强制贴地；这是条件尺度下的几何失败证据，不是实测接触或真实刀侧判断。')
    ledger=metrics['sources']['ledger']
    lines += ['', f"{ledger['changed_points']} 个局部点修正、{ledger.get('excluded_points',0)} 个待确认点排除，原始点和候选残差均保留；不把重绑、删点、换相机或单位换算当作拟合改善。原始资格全集另见 body_pixels.original_original_denominator，历史失败见 history/observations_v1。",
           f"分组指标相对 reference 或 legacy 的数值倒退共 {len(metrics['numeric_regressions'])} 项，逐条列于 metrics.json.numeric_regressions；不只报告总体收益。",
           '全部五窗口用于开发；没有独立最终检查集或独立三维真值。像素残差、骨长恒定、平滑及数值收敛均不等于真实三维准确；静态样例不等于完整镜头连续验收。',
           '物理左右脚身份、刀侧、接冰状态仍保留来源不确定性。P/G 分别是解剖骨盆及垂直投影，不是实测重心/接触点。',
           '报告只读取已保存结果，未重新优化。未播放、生成或嵌入任何视频。','']
    return '\n'.join(lines)


def review(root, output, figures=True):
    root,output=Path(root),Path(output)
    required=[output/m/'result.npz' for m in MODES]+[output/'observations.npz',output/'ledger.json']
    absent=[str(p) for p in required if not p.is_file()]
    if absent:
        raise FileNotFoundError('Exports are not ready; no report generated: '+', '.join(absent))
    packets={mode:load(output/mode/'result.npz') for mode in MODES}
    obs=load(output/'observations.npz');ledger=read_json(output/'ledger.json');pts=obs['pts']
    if np.any(np.diff(pts)<=0):raise ValueError('Source PTS must be strictly increasing')
    for mode,z in packets.items():
        np.testing.assert_array_equal(z['pts'],pts)
        np.testing.assert_array_equal(z['source_pts'],pts)
        np.testing.assert_array_equal(z['time_base'],[1,12800])
        np.testing.assert_array_equal(z['source_lower_body_inferred'],obs['lower_body_inferred'])
        np.testing.assert_equal(z['metres_per_scene_unit'],packets['reference']['metres_per_scene_unit'])
        if 'observation_ledger_id' in z and str(z['observation_ledger_id'])!=sha256(output/'ledger.json'):
            raise ValueError('Result observation ledger identity mismatch: '+mode)
    if ledger.get('observation_array_sha256')!=sha256(output/'observations.npz'):
        raise ValueError('Observation array does not match ledger SHA256')
    masks=window_masks(pts);use=np.asarray(obs['use'],bool)
    for mode in ('legacy','full'):
        np.testing.assert_array_equal(packets[mode]['window_mask'],masks['all_windows'])
    if packets['reference']['window_mask'].any():raise ValueError('Reference should have no optimized frames')
    camera_bindings={mode:json.loads(str(z['camera_binding_json'])) for mode,z in packets.items()}
    reference_camera=camera_bindings['reference']
    for mode,binding in camera_bindings.items():
        if binding['state_sha256']!=reference_camera['state_sha256'] or binding.get('camera_changed_during_fit'):
            raise ValueError('Camera binding differs across fitting modes: '+mode)
    original_use=np.asarray(load(root/'input.npz')['source_qualified'],bool)
    body={mode:dict(adopted=body_statistics(z['COCO_uv'],obs['uv'],use,masks,pts),
                    original_same_denominator=body_statistics(z['COCO_uv'],obs['raw_coco_uv'],use,masks,pts),
                    original_original_denominator=body_statistics(z['COCO_uv'],obs['raw_coco_uv'],original_use,masks,pts)) for mode,z in packets.items()}
    key_frames=[]
    for point in sorted(set(KEY_PTS)|{int(c['pts']) for c in ledger['changes']}):
        fi=int(np.searchsorted(pts,point))
        if fi>=len(pts) or pts[fi]!=point:raise ValueError(f'Key source PTS missing: {point}')
        key_frames.append(dict(pts=point,seconds=point/12800,lower_body_inferred=bool(obs['lower_body_inferred'][fi]),
            use=use[fi].tolist(),original_coco_uv=obs['raw_coco_uv'][fi].tolist(),adopted_uv=obs['uv'][fi].tolist(),
            source_id=obs['source_id'][fi].tolist(),semantic_index=obs['semantic_index'][fi].tolist(),
            modes={mode:dict(blade_sample_minimum_height_m=np.min(z['blade_world_mid'][fi,...,2],axis=1).tolist(),
                             original_residual_px=np.linalg.norm(z['COCO_uv'][fi]-obs['raw_coco_uv'][fi],axis=-1).tolist(),
                             adopted_residual_px=np.linalg.norm(z['COCO_uv'][fi]-obs['uv'][fi],axis=-1).tolist()) for mode,z in packets.items()}))
    historical_path=root/'next_round_20261029/J1_s2/result.npz'
    historical=None
    if historical_path.is_file():
        old=load(historical_path);np.testing.assert_array_equal(old['pts'],pts)
        historical=dict(path=str(historical_path),sha256=sha256(historical_path),
            warning='DIFFERENT HISTORICAL CAMERA; not a fitting improvement comparator',
            adopted=body_statistics(old['COCO_uv'],obs['uv'],use,masks,pts))
    metrics=dict(schema='rinkspace.fullbody-development-review.v1',status='development_only_no_independent_3d_ground_truth',
        source_time_base=[1,12800],source_frame_denominator=len(pts),window_seconds=WINDOWS,
        source_window_frames={name:int(mask.sum()) for name,mask in masks.items()},
        lower_body_inferred_frames={name:int(np.asarray(obs['lower_body_inferred'])[mask].sum()) for name,mask in masks.items()},
        denominator='one observations.use mask and the explicitly defined common five windows for all modes; missing predictions stay missing',
        groups=GROUPS,old_reference_current_camera=True,unit_change_is_improvement=False,camera_bindings=camera_bindings,
        sources=dict(ledger_path=str(output/'ledger.json'),ledger_sha256=sha256(output/'ledger.json'),ledger=ledger,
                     exports={mode:dict(path=str(output/mode/'result.npz'),sha256=sha256(output/mode/'result.npz')) for mode in MODES}),
        body_pixels=body,key_frames=key_frames,historical_camera_separate=historical,
        footwear=footwear_statistics(root,packets,pts,masks),geometry=geometric_statistics(root,packets,pts,masks,output),
        solver_windows={mode:{f'w{i+1}':read_json(output/mode/f'w{i+1}'/'result.json') for i in range(5)} for mode in ('legacy','full')},
        limitations=['All requested windows are development; no independent final check or 3D ground truth.',
                     'Static examples do not establish continuous full-shot visual acceptance.',
                     'Pelvis, vertical projection, blade boundary, and actual contact/centre-of-mass are distinct.',
                     'Assistant-only source association review is not human acceptance.',
                     'Engineering sigma values and raw source-specific detector scores are not calibrated probabilities.'])
    regressions=[]
    for baseline in ('reference','legacy'):
        for window in masks:
            for group in GROUPS:
                before=body[baseline]['adopted'][window]['groups'][group]
                after=body['full']['adopted'][window]['groups'][group]
                for statistic in ('median','p95','max'):
                    if before[statistic] is not None and after[statistic] is not None and after[statistic]>before[statistic]:
                        regressions.append(dict(domain='body_pixels',baseline=baseline,mode='full',window=window,group=group,statistic=statistic,before_px=before[statistic],after_px=after[statistic]))
    for baseline in ('reference','legacy'):
        for window,groups in metrics['footwear']['grouped'].items():
            for kind,row in groups.items():
                before=row['modes'][baseline];after=row['modes']['full']
                for statistic in ('median','p95','max'):
                    if before[statistic] is not None and after[statistic] is not None and after[statistic]>before[statistic]:
                        regressions.append(dict(domain='footwear',baseline=baseline,mode='full',window=window,group=kind,statistic=statistic,before_px=before[statistic],after_px=after[statistic]))
    metrics['numeric_regressions']=regressions
    metrics['acceptance_assessment']=acceptance_assessment(packets,masks,output)
    metrics['static_figures']=[]
    if figures:
        source=find_source(root);metrics['sources']['video']=source
        parents=load(root/'compact_smplx.npz')['parents'][:22].astype(int)
        metrics['static_figures']=make_figures(root,output,packets,obs,pts,parents,source)
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(jsonable(metrics),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    (output/'results.txt').write_text(report_text(metrics),encoding='utf8')
    return metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,help='original body_s01_20261029 run directory')
    parser.add_argument('--output',required=True,help='round directory containing reference/legacy/full exports')
    parser.add_argument('--no-figures',action='store_true',help='only compute statistics; no source decoding')
    args=parser.parse_args(argv)
    metrics=review(args.root,args.output,figures=not args.no_figures)
    print(json.dumps(dict(metrics=str(Path(args.output)/'metrics.json'),report=str(Path(args.output)/'results.txt'),figures=len(metrics['static_figures'])),ensure_ascii=False))


if __name__=='__main__':
    main()
