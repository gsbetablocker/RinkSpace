"""Select existing six-ray candidates using a disclosed weak motion prior.

This does not change UV, candidate XYZ, camera, articulation or contact. A
sparse nearest-neighbour graph makes the computation approximate; its full
edges and node costs are saved. Minimum motion is a working assumption, not
evidence that the chosen depth branch is true.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
import scipy
from .pelvis_footwear_prior import current_incam_root_prior

WEIGHTS = np.asarray([2., 2., .5, .5, .25, .25])
NEIGHBOURS = 8

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in (z.files if keys is None else keys)}

def edge_clock(actual_pts, mapping_path):
    """Official 50Hz only for verified endpoints, otherwise native display."""
    wanted = set(map(int,actual_pts)); rows = {}
    with Path(mapping_path).open(encoding='utf-8-sig') as f:
        for line in f:
            x = json.loads(line)
            required={'full_pts','match_status','accepted_official_pts'}
            if not required.issubset(x): raise ValueError('Source time mapping required keys absent')
            p = x.get('full_pts')
            if p is not None and int(p) in wanted: rows[int(p)] = x
    if set(rows)!=wanted: raise ValueError('Canonical source endpoints absent from original time mapping')
    for x in rows.values():
        if x['match_status']=='unique_pixel_match' and x['accepted_official_pts'] is None:
            raise ValueError('Unique source match lacks an accepted official timestamp')
    # Source schema is checked rather than guessing an alternate timestamp.
    dt=[]; origin=[]
    for a,b in zip(actual_pts[:-1],actual_pts[1:]):
        ra,rb=rows.get(int(a),{}),rows.get(int(b),{})
        oa,ob=ra.get('accepted_official_pts'),rb.get('accepted_official_pts')
        valid=(ra.get('match_status')=='unique_pixel_match' and rb.get('match_status')=='unique_pixel_match'
               and oa is not None and ob is not None and int(ob)>int(oa))
        if valid:
            dt.append((int(ob)-int(oa))/12800.)
            origin.append('official50_unique_pixel_endpoints_1_12800')
        else:
            dt.append((int(b)-int(a))/15360.)
            origin.append('native_canonical_presentation_approximation_1_15360')
    dt=np.asarray(dt,float)
    if np.any(dt<=0) or not np.isfinite(dt).all(): raise ValueError('Positive source edge times required')
    if len(actual_pts)==149:
        statuses=[x['match_status'] for x in rows.values()]
        official=np.asarray(origin)=='official50_unique_pixel_endpoints_1_12800'
        if statuses.count('unique_pixel_match')!=99 or statuses.count('no_close_official_pixel_match')!=50:
            raise ValueError('Actual tail original mapping qualification differs from 99 unique / 50 unmatched')
        if int(official.sum())!=97 or int((~official).sum())!=51 or not np.allclose(dt[official],.02,rtol=0,atol=1e-15):
            raise ValueError('Actual tail source-clock edge closure differs from 97 official20ms / 51 presentation edges')
    return dt,np.asarray(origin),[rows.get(int(p)) for p in actual_pts]

def select_profile(xyz, branch, eligible, offsets, actual_pts, dt, cut_ids, output):
    """DP on saved nearest edges plus a best-prefix fallback, without caps."""
    selected=np.full(len(actual_pts),-1,np.int64); layers=[]; features=[]
    for i in range(len(actual_pts)):
        ids=np.flatnonzero(eligible[offsets[i]:offsets[i+1]])+offsets[i]
        layers.append(ids)
        features.append((xyz[ids].astype(float)*np.sqrt(WEIGHTS/WEIGHTS.sum())[None,:,None]).reshape(len(ids),18))
    layer_offsets=np.r_[0,np.cumsum([len(x) for x in layers])].astype(np.int64)
    count=int(layer_offsets[-1]); width=NEIGHBOURS+1
    prev_global=np.full((count,width),-1,np.int64)
    edge_cost=np.full((count,width),np.nan,float)
    cumulative=np.full(count,np.inf,float)
    best_prev=np.full(count,-1,np.int64)
    alternatives=[]; segments=[]
    start=None
    def finish(end):
        nonlocal start
        if start is None: return
        rows=cumulative[layer_offsets[end]:layer_offsets[end+1]]
        best=int(np.argmin(rows)); terminal=layers[end][best]
        def path(last):
            out=np.full(len(actual_pts),-1,np.int64); current=int(last)
            for f in range(end,start-1,-1):
                out[f]=current
                local=int(np.searchsorted(layers[f],current))
                current=int(best_prev[layer_offsets[f]+local])
            return out
        selected[start:end+1]=path(terminal)[start:end+1]
        for b in np.unique(branch[layers[end]]):
            ids=np.flatnonzero(branch[layers[end]]==b)
            local=ids[np.argmin(rows[ids])]
            p=path(layers[end][local])
            alternatives.append(dict(segment=[start,end],terminal_branch=int(b),cost=float(rows[local]),excess_cost=float(rows[local]-rows[best]),indices=p))
        segments.append([start,end]); start=None
    for i,ids in enumerate(layers):
        if not len(ids):
            finish(i-1)
            continue
        if start is not None and cut_ids[i]!=cut_ids[i-1]: finish(i-1)
        lo,hi=layer_offsets[i:i+2]
        if start is None:
            start=i; cumulative[lo:hi]=0.
            continue
        previous=layers[i-1]; costs=cumulative[layer_offsets[i-1]:layer_offsets[i]]
        k=min(NEIGHBOURS,len(previous))
        _,nearest=cKDTree(features[i-1]).query(features[i],k=k,workers=1)
        nearest=np.asarray(nearest).reshape(len(ids),k)
        parents=np.c_[nearest,np.full(len(ids),int(np.argmin(costs)),np.int64)]
        delta=features[i][:,None]-features[i-1][parents]
        motion=np.einsum('ijk,ijk->ij',delta,delta)/dt[i-1]
        trial=costs[parents]+motion
        winning=np.argmin(trial,axis=1)
        slots=np.arange(len(ids))
        cumulative[lo:hi]=trial[slots,winning]
        best_prev[lo:hi]=previous[parents[slots,winning]]
        prev_global[lo:hi,:k+1]=previous[parents]
        edge_cost[lo:hi,:k+1]=motion
        if (i+1)%30==0: print(f'profile graph {output.stem}: {i+1}/{len(layers)}',flush=True)
    finish(len(layers)-1)
    nodes=np.concatenate(layers) if layers else np.empty(0,np.int64)
    np.savez_compressed(output,node_global_indices=nodes,layer_offsets=layer_offsets,
                        predecessor_global_indices=prev_global,edge_motion_prior_cost=edge_cost,
                        node_best_cumulative_cost=cumulative,node_best_predecessor_global=best_prev,
                        actual_pts=actual_pts,edge_dt_seconds=dt,point_weights=WEIGHTS,
                        alternative_candidate_indices=np.stack([x['indices'] for x in alternatives]) if alternatives else np.empty((0,len(actual_pts)),np.int64),
                        alternative_path_cost=np.asarray([x['cost'] for x in alternatives]),
                        alternative_path_excess_cost=np.asarray([x['excess_cost'] for x in alternatives]))
    return selected,segments,[{k:v for k,v in x.items() if k!='indices'} for x in alternatives]

def run(family_manifest, cameras_path, geometry_path, coco_path, native_manifest,
        mapping_path, output, cut_id='main9', source_reach_mask_path=None):
    out=Path(output)
    if out.exists(): raise FileExistsError('Preserve existing working selection')
    out.mkdir(parents=True)
    manifest=read(family_manifest); candidate_path=Path(manifest['output']['path'])
    if digest(candidate_path)!=manifest['output']['sha256']: raise ValueError('Candidate family binding changed')
    # All complete families remain in this unchanged upstream file. Only the
    # arrays required by selection are materialized; root/depth arrays are not
    # silently rewritten to carry selected values.
    f=arrays(candidate_path,['xyz6_m','depths_m','source_unit_rays','COCO_hipmean_P_m','branch_id','gear_profile_eligible',
        'candidate_source_projection_domain_valid','exposure_offsets','branch_offsets','actual_pts',
        'source_six_original_native_uv','source_six_original_score','source_six_original_qualified',
        'current_camera25','optical_C_m','model_joint0_from_COCO_hipmean_offset_radius_m',
        'source_clock_valid','source_near_ice_condition_applied'])
    c=arrays(cameras_path,['pts','camera25','exposure_representative_pts'])
    g=arrays(geometry_path); local=arrays(coco_path,['actual_pts','cocoreg_lower6_rootlocal_model_m'])
    actual=f['actual_pts']; cut_ids=np.full(len(actual),cut_id)
    if not np.array_equal(actual,g['actual_pts']) or not np.array_equal(actual,local['actual_pts']):
        raise ValueError('Body-local, family and actual source clocks differ')
    camera_rows=np.searchsorted(c['pts'],actual)
    if not np.array_equal(c['pts'][camera_rows],actual) or not np.array_equal(c['camera25'][camera_rows],f['current_camera25']):
        raise ValueError('Actual candidate physical camera changed')
    dt,time_origin,time_rows=edge_clock(actual,mapping_path)
    # Validity is provenance/domain, not a residual-based source deletion.
    clock_good=f['source_clock_valid'] & f['source_six_original_qualified'].all(1)
    layer_good=np.repeat(clock_good,np.diff(f['exposure_offsets']))
    source_reach=np.ones_like(f['gear_profile_eligible'],dtype=bool)
    reach_binding=None
    if source_reach_mask_path is not None:
        reach=arrays(source_reach_mask_path)
        ids=np.asarray(reach['full_global_node_index'],np.int64)
        observed=np.asarray(reach['source_blade_region_eligible'],bool)
        if (not np.array_equal(reach['actual_pts'],actual)
                or observed.shape!=(len(ids),3)
                or np.any(ids<0) or np.any(ids>=len(source_reach))
                or np.any(np.diff(ids)<=0)):
            raise ValueError('Source blade reach and unchanged full family indices differ')
        # This compact view must cover every otherwise selectable node. A
        # missing row is not permission to accept an unchecked candidate.
        covered=np.zeros(len(source_reach),bool);covered[ids]=True
        base_valid=(f['gear_profile_eligible'].any(1)
                    &f['candidate_source_projection_domain_valid']&layer_good)
        if np.any(base_valid&~covered):
            raise ValueError('Source reach packet omits selectable family nodes')
        source_reach[ids]=observed
        reach_binding=dict(path=str(source_reach_mask_path),sha256=digest(source_reach_mask_path),
                           selected_nodes_stay_in_original_family=True,
                           original_source_observations_changed=False)
    selections={}; profiles={}; boundary={}; ranges={}; graph_bindings=[]
    for col,label in [(1,'nominal9cm'),(0,'equipment8cm'),(2,'equipment10cm')]:
        mask=(f['gear_profile_eligible'][:,col]&f['candidate_source_projection_domain_valid']
              &layer_good&source_reach[:,col])
        graph=out/f'candidate_graph_{label}.npz'
        pick,segments,alternatives=select_profile(f['xyz6_m'],f['branch_id'],mask,f['exposure_offsets'],actual,dt,cut_ids,graph)
        selections[label]=pick; flags=np.zeros(len(actual),bool); minimum=[]; maximum=[]
        for i,p in enumerate(pick):
            a,b=map(int,f['exposure_offsets'][i:i+2]); ids=np.flatnonzero(mask[a:b])+a
            q=f['COCO_hipmean_P_m'][ids]
            minimum.append(q.min(0) if len(q) else [np.nan]*3)
            maximum.append(q.max(0) if len(q) else [np.nan]*3)
            if p>=0:
                assert mask[p] and a<=p<b
                branch=int(f['branch_id'][p]); left,right=f['branch_offsets'][i,branch:branch+2]
                flags[i]=(p==left or p==right-1 or not mask[p-1] or not mask[p+1])
        boundary[label]=flags; ranges[label]=(np.asarray(minimum),np.asarray(maximum))
        good=pick>=0; values=f['COCO_hipmean_P_m'][pick[good]]
        steps=np.linalg.norm(np.diff(values[:,:2],axis=0),axis=1)
        profiles[label]=dict(lift_m=[.08,.09,.1][col],selected_actual_count=int(good.sum()),
            selected_family_boundary_count=int(flags.sum()),maximum_adjacent_proxy_G_step_m=float(steps.max()) if len(steps) else None,
            maximum_step_actual_pts=actual[np.argmax(steps):np.argmax(steps)+2].tolist() if len(steps) else None,
            last_selected_candidate_index=int(pick[-1]),last_proxy_P_m=values[-1].tolist() if len(values) else None,
            alternatives=alternatives,segments=segments,graph=dict(path=str(graph),sha256=digest(graph)))
        graph_bindings.append(profiles[label]['graph'])
    nominal=selections['nominal9cm']; good=nominal>=0
    P=np.full((len(actual),3),np.nan); six=np.full((len(actual),6,3),np.nan)
    # Canonical selected nodes are source optical C + stored Euclidean
    # depths * original double-precision source unit rays. xyz6_m is only
    # the immutable float32 graph/plot representation of these same nodes.
    six[good]=f['optical_C_m'][good,None]+f['depths_m'][nominal[good],:,None].astype(float)*f['source_unit_rays'][good]
    P[good]=six[good,:2].mean(1)
    joint0=P.copy(); Q=[]; incam=[]; origins=[]
    scale=float(g['body_scale_nominal']); mean_local=local['cocoreg_lower6_rootlocal_model_m'][:,:2].astype(float).mean(1)
    for i in range(len(actual)):
        Rincam,origin=current_incam_root_prior(g,i,c)
        q=cv2.Rodrigues(f['current_camera25'][i,3:6])[0].T@Rincam
        Q.append(q);incam.append(Rincam);origins.append(origin)
        if good[i]:joint0[i]-=scale*(q@mean_local[i])
    Q=np.asarray(Q); incam=np.asarray(incam)
    G=P.copy();G[:,2]=0.;jG=joint0.copy();jG[:,2]=0.
    G[~good]=np.nan;jG[~good]=np.nan
    raw=read(native_manifest); native=np.asarray([x['pts'] for x in raw['frames']],np.int64)
    nr=np.searchsorted(c['pts'],native)
    if not np.array_equal(c['pts'][nr],native): raise ValueError('Native camera clock absent')
    reps=c['exposure_representative_pts'][nr]; ar=np.searchsorted(actual,reps)
    if not np.array_equal(actual[ar],reps): raise ValueError('Native aliases absent from selected actual clock')
    lower,upper=ranges['nominal9cm']; radius=f['model_joint0_from_COCO_hipmean_offset_radius_m']
    trace=dict(pts=native,actual_pts=actual,exposure_representative_pts=reps,
        selected_candidate_global_index_per_actual=nominal,selected_candidate_global_index=nominal[ar],
        selected_branch_id=np.where(good,f['branch_id'][np.maximum(nominal,0)],255).astype(np.uint8)[ar],
        source_valid=good[ar],unique_world_known=np.zeros(len(native),bool),unique_world_G_assumed_m=np.full((len(native),3),np.nan),
        blade_contact_valid=np.zeros(len(native),bool),camera25=c['camera25'][nr],camera_C_assumed_m=c['camera25'][nr,:3]*manifest['metres_per_scene_unit'],
        conditional_proxy_P_assumed_m=P[ar],conditional_proxy_G_assumed_m=G[ar],
        estimated_model_joint0_P_assumed_m=joint0[ar],estimated_model_joint0_G_assumed_m=jG[ar],
        selected_model_COCO_six_world_assumed_m=six[ar],
        current_root_Q_world=Q[ar],current_root_prior_R_incam=incam[ar],root_prior_origin=np.asarray(origins)[ar],
        source_six_original_native_uv=f['source_six_original_native_uv'][ar],source_six_original_score=f['source_six_original_score'][ar],
        source_six_original_qualified=f['source_six_original_qualified'][ar],
        candidate_proxy_P_min_m=lower[ar],candidate_proxy_P_max_m=upper[ar],candidate_proxy_G_min_xy_m=lower[ar,:2],candidate_proxy_G_max_xy_m=upper[ar,:2],
        candidate_joint0_G_conservative_outer_min_xy_m=(lower[:,:2]-radius[:,None])[ar],
        candidate_joint0_G_conservative_outer_max_xy_m=(upper[:,:2]+radius[:,None])[ar],
        model_joint0_offset_radius_m=radius[ar],selected_family_boundary=boundary['nominal9cm'][ar],
        source_near_ice_condition_applied=f['source_near_ice_condition_applied'][ar],
        edge_dt_seconds=dt,edge_time_origin=time_origin)
    if reach_binding is not None:
        trace['source_blade_reach_condition_satisfied']=np.asarray([
            bool(p>=0 and source_reach[p,1]) for p in nominal])[ar]
    for label,pick in selections.items():
        trace[f'{label}_selected_candidate_global_index_per_actual']=pick
        trace[f'{label}_selected_family_boundary_per_actual']=boundary[label]
    path=out/'conditional_pelvis_working_trace.npz';np.savez_compressed(path,**trace)
    (out/'actual_source_clock_rows.json').write_text(json.dumps(dict(mapping=dict(path=str(mapping_path),sha256=digest(mapping_path)),actual_pts=actual.tolist(),rows=time_rows,edge_time_origin=time_origin.tolist(),edge_dt_seconds=dt.tolist()),ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(schema='rinkspace.existing-six-ray-nodes-weak-motion-working-selection.v1',
        native_frames=len(native),actual_exposures=len(actual),nominal_profile='nominal9cm',profiles=profiles,
        edge_clock_origin_counts={str(x):int(np.count_nonzero(time_origin==x)) for x in np.unique(time_origin)},
        motion_prior=dict(point_weights=WEIGHTS.tolist(),cost='sum_j normalized_weight_j * norm(X_j(t)-X_j(t-1))^2 / actual_edge_dt',
            minimum_motion_is_engineering_prior_not_validation=True,no_hard_velocity_threshold=True,
            no_branch_number_penalty=True,no_acceleration_smoothing=True,no_candidate_coordinates_changed=True),
        approximate_graph=dict(nearest_predecessors=NEIGHBOURS,best_prefix_fallback=True,all_edges_saved=True,
            global_optimum_over_complete_candidate_graph_claim=False,numpy_version=np.__version__,scipy_version=scipy.__version__),
        native_joint0_offset_direction='raw network incam root orientation rebound via current camera; source-clock SLERP between true raw brackets; not old fitted body/foot Q or historical world R',
        body_scale_applied_once=scale,unknown_offset_orientation_full_radius_envelope_retained=True,
        source_candidate_family_arrays_unchanged=True,unknown_source_not_filled=True,cross_cut_edges=False,
        selected_node_coordinates='optical_C_m + original source_unit_rays * stored candidate depths; same global indices, no fitted translation',
        sparse_graph_features='immutable float32 xyz6_m; source-ray canonical geometry reconstructed after selecting the same node',
        old_fixed_pose_footwear_Q_used=False,coarse_circle_fit_filter_or_hard_acceptance_gate=False,
        full_SMPLX_pose_solved=False,
        selected_six_points_bound_to_one_complete_body_pose=False,
        native_joint0_is_conditional_offset_binding_not_full_body_fit=True,
        source_rays_and_proxy_lengths_are_conditions_not_measurements=True,range_is_sampled_family_not_confidence_interval=True,
        joint0_outer_envelope_is_conservative_not_exact_attainability=True,validated_speed_reported=False,
        accuracy_accepted=False,blade_contact=False,unique_true_world_known=False,
        source_blade_reach_binding=reach_binding,
        trace=dict(path=str(path),sha256=digest(path)),graph_bindings=graph_bindings,
        bindings=[dict(path=str(p),sha256=digest(p)) for p in [family_manifest,candidate_path,cameras_path,geometry_path,coco_path,native_manifest,mapping_path,Path(__file__)]],
        limitations=['Minimum-motion branch may be physically wrong or static even if source motion permits several branches; full candidate ranges and alternatives remain available.',
            'Six source-ray points with proxy segment lengths are not a complete SMPL-X pose. Root-offset binding does not make the torso, legs and footwear jointly consistent with those points or the image.',
            'Sparse graph is a reproducible computational approximation, not the global minimum on all possible pairwise edges.',
            'Exact image proxy rays, model proxy lengths, approximate stature and near-ice reach bounds remain engineering assumptions; pixel/shape uncertainty is not marginalized.',
            'Working native joint0 offset uses image-conditioned old-K root-direction prior, whose error can affect the point; its radius envelope remains separate.',
            'Official50 timing is used only where both canonical endpoints have unique pixel matches; other edge timing is approximate presentation timing.',
            'The 24.950 held exposure retains its broad unconditioned height family. No contact label or support equation is silently added.'])
    if reach_binding is not None:
        report['bindings'].append(reach_binding)
        report['limitations'].append('Visible blade regions constrain only exposures with existing source observations. Passing their necessary reach condition does not determine foot orientation or validate the continuous pelvis path.')
    (out/'working_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),profiles={k:{a:b for a,b in v.items() if a not in ['alternatives','graph']} for k,v in profiles.items()},edge_clock=report['edge_clock_origin_counts']),ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for name in ['family_manifest','cameras','geometry','coco_packet','native_manifest','time_mapping','output']:p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--cut-id',default='main9')
    p.add_argument('--source-reach-mask');a=p.parse_args()
    run(a.family_manifest,a.cameras,a.geometry,a.coco_packet,a.native_manifest,a.time_mapping,a.output,a.cut_id,a.source_reach_mask)

if __name__=='__main__':main()
