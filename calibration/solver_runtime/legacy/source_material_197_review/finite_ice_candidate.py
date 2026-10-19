"""Explicit candidate admission of three unchanged original motion FIT rows.

No default, source packet, kernel, or accepted class is patched. Call only on
an independently loaded physical candidate state, after checkpoint profiles
are restored and before source actors are created. The native/NumPy existing
finite-plane operator and every original finite-extent guard remain in use.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2,numpy as np

from calibration.solver_runtime.config import data_module_dir
HERE=data_module_dir(__file__)
BASE=HERE.parent
DEFAULT_PACKET=HERE/'three_original_ice_rows_source_QUALIFIED_CANDIDATE_NOT_INSTALLED.json'
SOURCE_MANIFEST=BASE/'prepared_full8503_1791035775270492900/complete_original_sources/low_source/low_source_factory/original_source_adapter_inputs/input_manifest.json'
EXPECTED_RAW=(82721,82723,82735)
EXPECTED_BUNDLE=(113623,113625,113636)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(value.dtype.str.encode('ascii'));h.update(str(value.shape).encode('ascii'));h.update(value.view(np.uint8))
    return h.hexdigest()

def authenticated_physical_chart_scope(state,q):
    """Accept only the two explicit original-source physical candidate charts.

    A statistical tail is never removed here. The caller authenticates and
    separates that tail, then passes either all38 or mount+all38 physical q.
    This checks chart ownership and indices before any material mutation.
    """
    from all_installation_optics_candidate.all_installation_optics_graph import InstallationOpticsChart,_lens_signature
    from z190_physical_camera import SharedCalibration
    from z190_continuous_source_bundle import SCENE_UNIT_M
    q=np.asarray(q)
    if q.dtype!=np.dtype('float64') or q.shape not in ((34204,),(34213,)) or not np.isfinite(q).all():
        raise ValueError('Explicit finite float64 all38 or authenticated mount+all38 physical q required')
    count=len(q);mount=count==34213
    ids=list(state.ids);charts=state.installation_optics;optics=state.optics_contract
    if len(ids)!=3 or len(set(ids))!=3 or list(charts)!=ids or optics.get('installation_order')!=ids:
        raise ValueError('Original ordered three physical installation chart owners required')
    old_count=34204 if mount else 34195
    optical_slots=list(range(old_count,old_count+9))
    if (optics.get('schema')!='rinkspace.all-installation-shared-optics-candidate.v1' or
        optics.get('original_parameter_count')!=old_count or optics.get('parameter_indices')!=optical_slots or
        not np.array_equal(state.optics_parameter_indices,optical_slots) or
        optics.get('all_29_shared_and_q11_preserved') is not True or
        optics.get('source_rows_sigma_roles_and_held_unchanged') is not True):
        raise ValueError('Explicit nine lens columns must follow their original authenticated chart')
    geometry=np.asarray(state.geometry_parameter_indices)
    if (not np.array_equal(geometry,[0,1,2,3,4,34189,34190,34191,34192,34193,34194]) or
        state.patterns.start!=34180):
        raise ValueError('Original q11 geometry and all nine shared material coordinates must retain their indices')
    mount_contract=None
    if mount:
        from installation_mount_candidate import mount_graph
        chart_class=mount_graph.MountOpticsChart
        mount_contract=getattr(state,'mount_contract',None)
        if not isinstance(mount_contract,dict):raise ValueError('Explicit mount contract required for34213 physical q')
        required=dict(schema='rinkspace.shared-constant-nonzero-installation-mount-candidate.v1',
            original_parameter_count=34195,extra_physical_shared_parameters=9,additional_prior_rows=0,
            one_lever_shared_over_all_cuts=True,zoom_dependence=False,per_frame_translation=False,
            prior_hardware_working_search_ranges_not_measurements=True,
            original_source_pixels_sigmas_loss_roles_and_held_unchanged=True,measured_lever_claim=False)
        if any(mount_contract.get(k)!=v for k,v in required.items()):
            raise ValueError('Known original-source constant mount chart schema/semantics required')
        mount_slots=list(range(34195,34204))
        if (mount_contract.get('parameter_indices')!=mount_slots or
            mount_contract.get('installation_order')!=ids or
            mount_contract.get('parameter_units')!='metre increments from authenticated base camera-local lever' or
            not np.array_equal(state.mount_parameter_indices,mount_slots) or
            optics.get('mount_contract')!=mount_contract):
            raise ValueError('Exactly nine authenticated metre mount coordinates before the nine lens coordinates required')
        chart_module_sha=sha(mount_graph.__file__)
    else:
        chart_class=InstallationOpticsChart;mount_slots=[]
        if getattr(state,'mount_contract',None) is not None or getattr(state,'mount_parameter_indices',None) is not None:
            raise ValueError('A mount declaration cannot be attached to the old all38 physical chart')
        import all_installation_optics_candidate.all_installation_optics_graph as chart_module
        chart_module_sha=sha(chart_module.__file__)
    for ordinal,name in enumerate(ids):
        chart=charts[name];slots=tuple(range(old_count+3*ordinal,old_count+3*ordinal+3))
        spec=optics['installations'][name]
        if (type(chart) is not chart_class or chart.installation_id!=name or
            chart.original_parameter_count!=old_count or chart.indices!=slots or
            spec.get('schema')!='rinkspace.installation-shared-optics-candidate.v1' or
            spec.get('installation_id')!=name or spec.get('original_parameter_count')!=old_count or
            spec.get('parameter_indices')!=list(slots) or spec.get('physical_column_scales')!=[1.,1.,.001] or
            spec.get('physical_derivative_steps')!=list(chart.physical_steps) or
            len(chart.physical_steps)!=3 or not np.isfinite(chart.physical_steps).all() or min(chart.physical_steps)<=0.):
            raise ValueError('Exact known namespace/class and physical lens chart indices/scales required')
        base=SharedCalibration.from_manifest(spec['base_calibration'])
        members=[b for b in state.blocks if b.bundle.calibration.installation_id==name]
        if not members or any(b.optics_chart is not chart or
            _lens_signature(b.optics_base_calibration)!=_lens_signature(base) for b in members):
            raise ValueError('Every original source block must use its one current physical installation chart')
        if mount:
            slots_mount=tuple(range(34195+3*ordinal,34195+3*ordinal+3))
            if (chart.mount_indices!=slots_mount or
                mount_contract['parameter_indices_by_installation'].get(name)!=list(slots_mount) or
                not np.array_equal(mount_contract['original_levers_m'].get(name),np.asarray(base.lever_xyz)*SCENE_UNIT_M)):
                raise ValueError('Per-installation original lever and explicit mount indices changed')
    return dict(schema='rinkspace.reviewed-material-authenticated-physical-chart.v1',
        mode='constant_mount9_plus_all_installation_optics9' if mount else 'all_installation_optics9',
        physical_core_parameter_count=count,original_parameter_count=34195,
        original_shared_parameter_count=29,physical_shared_parameter_count=47 if mount else 38,
        inherited_statistical_shared_parameter_count=3,required_complete_fresh_shared_gate_parameter_count=50 if mount else 41,
        chart_namespace=chart_class.__module__,chart_class=chart_class.__name__,chart_module_sha256=chart_module_sha,
        installation_order=ids,mount_parameter_indices=mount_slots,optics_parameter_indices=optical_slots,
        original_source_and_geometry_indices_preserved=True,statistical_tail_accepted_or_truncated_here=False)

def checked_packet(packet=DEFAULT_PACKET):
    from z190_continuous_source_bundle import SOURCE_SHA,HARD_SCENE_SHA
    path=Path(packet);p=read(path)
    if (p['source_sha256']!=SOURCE_SHA or p['scene_sha256']!=HARD_SCENE_SHA or
        p['source_review_status']!='source_native_material_checked_no_global_Lane_id_claim'):
        raise ValueError('Reviewed original event/material authority changed')
    rows=p['qualified_rows']
    if tuple(r['original_global_index'] for r in rows)!=EXPECTED_RAW or tuple(r['global_bundle_index'] for r in rows)!=EXPECTED_BUNDLE:
        raise ValueError('Exactly the three explicitly reviewed original FIT identities required')
    authority=p['source_motion_graph'];rawpath=Path(authority['path'])
    if sha(rawpath)!=authority['sha256']:raise ValueError('Original motion packet changed')
    if sha(p['source_material_review']['path'])!=p['source_material_review']['sha256']:
        raise ValueError('Source-only material ledger changed')
    for frame in p['source_native_frames']:
        if sha(frame['raw_png'])!=frame['raw_png_sha256']:raise ValueError('Original source PNG bytes changed')
        pixels=cv2.imread(frame['raw_png'])
        if pixels is None or hashlib.sha256(pixels.tobytes()).hexdigest()!=frame['native_bgr_sha256']:
            raise ValueError('Original native source pixels changed')
    with np.load(rawpath,allow_pickle=False) as raw:
        for row in rows:
            pair=int(row['original_pair_index']);g=int(row['original_global_index'])
            if g!=int(raw['offsets'][pair])+int(row['original_local_index']):raise ValueError('Original row identity changed')
            if (row['split']!='fit' or row['source_role']!='original_motion_retained' or
                row['physical_surface_identity']!='native_visible_ice_plane_z0' or
                row['expectedMaterial']!='native_visible_ice_plane_z0' or
                not row['both_end_source_material_reviewed'] or
                row.get('global_Lane_or_ring_or_world_XY_identity_inferred') is not False or
                row.get('root_source_pixel_review',{}).get('both_endpoints_checked') is not True):
                raise ValueError('Only the explicitly reviewed original FIT ice material may change')
            if len(row['source_endpoint_material_checks'])!=2 or not all(
                c['qualified'] and not c['clock_reflection_nearby'] and not c['person_nearby'] for c in row['source_endpoint_material_checks']):
                raise ValueError('Both original endpoints require static material review')
            if (raw['pair_pts'][pair].astype(int).tolist()!=row['native_pair_pts'] or
                not np.array_equal(raw['xy0'][g],row['xy0']) or not np.array_equal(raw['xy1'][g],row['xy1']) or
                float(raw['fb_error_px'][g])!=row['fb_error_px']):
                raise ValueError('Original PTS, UV or FB changed')
    return path,p

def _changed_row_kernel_inputs(block,q,indices,*,jacobian=False):
    bundle=block.bundle;local=block.local(q)
    cache=bundle.cache(local,jacobian);vv,rr,dr,cc,ds,dv=cache
    indices=np.asarray(indices,int);a,b=bundle.ia[indices],bundle.ib[indices]
    cal=bundle.calibration
    k=dict(a=bundle.a[indices],b=bundle.b[indices],fa=np.exp(vv[a,3]),fb=np.exp(vv[b,3]),
        ra=rr[a],rb=rr[b],da=dr[a] if jacobian else None,db=dr[b] if jacobian else None,
        ca=cc[a],cb=cc[b],dca_shared=ds[a] if jacobian else None,dcb_shared=ds[b] if jacobian else None,
        dca_view=dv[a] if jacobian else None,dcb_view=dv[b] if jacobian else None,
        pp=np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]]),k1=cal.k1[0],
        planes=bundle.finite_planes[indices],jacobian=jacobian,
        sigma=bundle.motion_sigma.reshape(-1,2)[indices],
        profile_weight=None if bundle.profile_weights is None else bundle.profile_weights[indices])
    return k,cache

def source_plane_geometry(block,q,indices):
    """Check the exact current source ray and its target cheirality/footprint.

    The observed target-ray intersection is report-only. Existing trial guards
    own the source intersection, hard footprint and target projection domain.
    No agreement of two fitted XY intersections is required or constructed.
    """
    from z190_continuous_source_bundle import radial_inverse,SCENE_UNIT_M
    k,cache=_changed_row_kernel_inputs(block,q,indices)
    scene=block.bundle.scene
    polygon=np.asarray(scene['ice_boundary']['points_xyz'],float)[:,:2].astype(np.float32)
    reports=[]
    for j,g in enumerate(indices):
        xy,_=radial_inverse((k['a'][j:j+1]-k['pp'])/k['fa'][j],k['k1'])
        ray=k['ra'][j].T@np.r_[xy[0],1.]
        den=ray[2]
        if abs(den)<1e-10:raise ValueError('Reviewed ice source ray tangent to ice plane')
        depth=-k['ca'][j,2]/den;world=k['ca'][j]+depth*ray
        margin=cv2.pointPolygonTest(polygon,tuple(world[:2]),True)
        target=k['rb'][j]@(world-k['cb'][j])
        if depth<=0 or margin<0 or target[2]<=0:raise ValueError('Reviewed ice source outside current positive hard-ice domain')
        if abs(world[2])>1e-12:raise ValueError('Exact ice source intersection changed Z0')
        xy1,_=radial_inverse((k['b'][j:j+1]-k['pp'])/k['fb'][j],k['k1'])
        ray1=k['rb'][j].T@np.r_[xy1[0],1.]
        td=-k['cb'][j,2]/ray1[2];tw=k['cb'][j]+td*ray1
        reports.append(dict(global_bundle_index=int(g),source_axial_depth_m=float(depth*SCENE_UNIT_M),
            source_ray_ice_world_xyz_scene=world.tolist(),source_footprint_margin_m=float(margin*SCENE_UNIT_M),
            target_projection_axial_depth_m=float(target[2]*SCENE_UNIT_M),
            observed_target_ray_ice_world_xyz_scene_report_only=tw.tolist(),
            observed_target_ray_depth_m_report_only=float(td*SCENE_UNIT_M),
            observed_target_ray_footprint_margin_m_report_only=float(cv2.pointPolygonTest(polygon,tuple(tw[:2]),True)*SCENE_UNIT_M)))
    return reports

def install_three_reviewed_motion_ice(state,packet=DEFAULT_PACKET,*,expected_old_finite_count=468,
                                     source_manifest=SOURCE_MANIFEST):
    """Return a checkpoint/manifest contract after atomic instance-only wiring.

    state.q must be its authenticated 34204 or34213 physical core (statistical
    tail belongs to the caller's authenticated separate chart). Old weights are
    first restored. Only these three initial weights are then recomputed under
    the finite operator; all other original frozen profile bytes are retained.
    The ordinary full profile update in the original solve loop is unchanged.
    """
    from z190_joint_source_solver import make_profile_update,_frozen_profile_token
    if getattr(state,'source_actors',None) is not None:raise ValueError('Material domain must be installed before any source actor is constructed')
    if getattr(state,'qualified_original_motion_ice_contract',None) is not None:raise ValueError('Material candidate already installed')
    path,p=checked_packet(packet);rows=p['qualified_rows']
    low=[b for b in state.blocks if b.name=='low']
    if len(low)!=1:raise ValueError('One complete original low source block required')
    block=low[0];bundle=block.bundle;q=np.asarray(state.q)
    physical_chart=authenticated_physical_chart_scope(state,q)
    aliases=state.original.aliases
    historical=read(source_manifest)
    cut5=next(r for r in historical['source_motion'] if r['cut']==5)
    if cut5['rawsha256']!=p['source_motion_graph']['sha256'] or Path(cut5['rawpath']).resolve()!=Path(p['source_motion_graph']['path']).resolve():
        raise ValueError('Source adapter motion graph differs from reviewed source')
    if sha(cut5['selected_rows'])!=cut5['selected_sha256']:raise ValueError('Original retained source selection changed')
    with np.load(cut5['selected_rows'],allow_pickle=False) as selected:
        ids=selected['original_global_row_indices']
        for row in rows:
            if np.count_nonzero(ids==row['original_global_index'])!=1:raise ValueError('Reviewed original row is not uniquely retained')
    if len(bundle.a)!=435185 or bundle.motion_rows!=870370:raise ValueError('Complete low original plus existing persistent motion ledger required')
    if not np.array_equal(bundle.finite_mask,np.linalg.norm(bundle.finite_planes[:,:3],axis=1)>0.):raise ValueError('Existing plane/mask disagree')
    old_finite=int(bundle.finite_mask.sum())
    if old_finite!=expected_old_finite_count or len(bundle.finite_records)!=old_finite:
        raise ValueError('Audited existing finite source inventory changed')
    indices=[];records=[]
    for row in rows:
        matched=np.flatnonzero(np.all(bundle.a==row['xy0'],axis=1)&np.all(bundle.b==row['xy1'],axis=1))
        if len(matched)!=1 or int(matched[0])!=row['global_bundle_index']:raise ValueError('Reviewed source row unique full-bundle identity changed')
        j=int(matched[0]);pts=[bundle.stamps[bundle.ia[j]],bundle.stamps[bundle.ib[j]]]
        if pts!=[aliases[int(t)] for t in row['native_pair_pts']] or pts!=row['canonical_pair_pts']:
            raise ValueError('Original exposure aliases changed')
        if not np.array_equal(bundle.motion_sigma.reshape(-1,2)[j],row['original_sigma_native_px']):raise ValueError('Original source sigma changed')
        if np.any(bundle.finite_planes[j]!=0.) or bundle.finite_mask[j]:raise ValueError('Reviewed original row was already finite')
        indices.append(j)
        records.append(dict(global_bundle_index=j,cut=5,original_pair_index=row['original_pair_index'],
            original_local_index=row['original_local_index'],original_global_index=row['original_global_index'],
            source_native_pair_pts=row['native_pair_pts'],canonical_pair_pts=pts,xy0=row['xy0'],xy1=row['xy1'],
            surface_kind='ice_plane',physical_surface_identity='native_visible_ice_plane_z0',
            expectedMaterial='native_visible_ice_plane_z0',material_identity=row['material_identity'],
            source_role='original_motion_retained',split='fit',review=dict(path=str(path),sha256=sha(path)),
            admission='both_end_native_static_ice_paint_source_review',match_precision_not_measured=True))
    before_token=_frozen_profile_token(make_profile_update(state.blocks))
    old_planes=bundle.finite_planes;old_mask=bundle.finite_mask;old_records=bundle.finite_records
    old_weights=bundle.profile_weights;old_cal=bundle.calibration
    if old_weights is None or old_weights.shape!=(len(bundle.a),2):raise ValueError('Restore the complete inherited frozen profile before material admission')
    old_changed=old_weights[indices].copy();plane=old_planes.copy();plane[indices]=[0.,0.,1.,0.]
    try:
        bundle.finite_planes=plane;bundle.finite_mask=np.linalg.norm(plane[:,:3],axis=1)>0.
        bundle.finite_records=list(old_records)+records
        geometry=source_plane_geometry(block,q,indices)
        k,cache=_changed_row_kernel_inputs(block,q,indices)
        bundle.validate_finite_extents(cache)
        values=bundle.motion_kernel(**k);residual=values[0] if isinstance(values,tuple) else values
        normalized=np.asarray(residual).reshape(-1,2)/k['sigma']
        updated=np.minimum(1.,2./np.maximum(np.abs(normalized),1e-12))
        if not np.isfinite(updated).all():raise ValueError('New finite source profile not finite')
        weights=old_weights.copy();weights[indices]=updated;bundle.profile_weights=weights
        remaining=np.ones(len(weights),bool);remaining[indices]=False
        if not np.array_equal(weights[remaining],old_weights[remaining]):raise ValueError('Untouched inherited frozen profiles changed')
        if int(bundle.finite_mask.sum())!=old_finite+3 or len(bundle.finite_records)!=old_finite+3:raise ValueError('Explicit finite count must increase by exactly three')
        contract=dict(schema='rinkspace.original-motion-three-reviewed-ice-plane-candidate.v1',
            source_packet=dict(path=str(path),sha256=sha(path)),source_motion_graph=p['source_motion_graph'],
            original_selected_rows=dict(path=cut5['selected_rows'],sha256=cut5['selected_sha256']),
            source_material_review=p['source_material_review'],original_motion_raw_indices=list(EXPECTED_RAW),
            global_bundle_indices=indices,old_finite_count=old_finite,candidate_finite_count=old_finite+3,
            source_rows_added=0,source_rows_removed=0,source_row_order_UV_sigma_FB_roles_unchanged=True,
            existing_finite_planes_records_and_held_unchanged=True,new_planes=[[0.,0.,1.,0.]]*3,
            physical_core_q_sha256=hashlib.sha256(np.ascontiguousarray(q).view(np.uint8)).hexdigest(),
            physical_core_parameter_count=len(q),authenticated_physical_chart=physical_chart,
            original_frozen_profile_token=before_token,candidate_frozen_profile_token=_frozen_profile_token(make_profile_update(state.blocks)),
            changed_row_old_profile=old_changed.tolist(),changed_row_new_profile=updated.tolist(),
            changed_row_old_profile_invalidated_and_recomputed=True,
            all_other_inherited_frozen_profile_bytes_unchanged=True,
            standard_full_profile_update_in_original_solver_preserved=True,
            source_geometry_checks=geometry,source_depth_kernel_domain_unchanged=True,
            no_world_XY_anchor_or_Lane_id_inferred=True,held_not_used_for_fit_or_qualification=True,
            current_surface_qualification_is_not_match_precision=True,
            Ad20_curved_rows_not_assigned_any_plane=True,
            complete_fresh_shared41_gate_required=len(q)==34204,
            complete_fresh_shared50_gate_required=len(q)==34213,
            complete_fresh_shared_gate_required=True,
            required_complete_fresh_shared_gate_parameter_count=physical_chart['required_complete_fresh_shared_gate_parameter_count'],
            complete_gate_passed=False,
            installation_must_precede_source_actor_construction=True,production_default_changed=False,
            whole_cut_accuracy_accepted=False)
        bundle.qualified_original_motion_ice_contract=contract
        state.qualified_original_motion_ice_contract=contract
        return contract
    except Exception:
        bundle.finite_planes=old_planes;bundle.finite_mask=old_mask;bundle.finite_records=old_records
        bundle.profile_weights=old_weights;bundle.calibration=old_cal
        raise

def checkpoint_arrays(contract):
    if contract['schema']!='rinkspace.original-motion-three-reviewed-ice-plane-candidate.v1':raise ValueError('Explicit material contract required')
    return dict(qualified_original_motion_ice_contract_json=np.array(json.dumps(contract,ensure_ascii=False)),
        qualified_original_motion_ice_global_bundle_indices=np.asarray(contract['global_bundle_indices'],np.int64),
        qualified_original_motion_ice_raw_indices=np.asarray(contract['original_motion_raw_indices'],np.int64))
