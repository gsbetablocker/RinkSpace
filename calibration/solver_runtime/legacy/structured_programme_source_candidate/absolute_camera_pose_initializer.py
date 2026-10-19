"""Apply an explicit source-qualified camera starting estimate to an exact q.

The original v1 estimate changes named nonreference exposure RF coordinates.
An explicit v2 estimate may also initialize one existing installation pivot Z
and its reference views. Derived view estimates remain initialization values,
never observations or a replacement for the complete physical/source objective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
POSE_SCHEMA = 'rinkspace.source-qualified-camera-pose-initializer.v1'
PHYSICAL_SCHEMA = 'rinkspace.source-qualified-physical-camera-initializer.v2'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _q_sha(q):
    return hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def verified_camera_pose_estimate(state, q, packet_path):
    """Verify source, old values, and exact allowed slots; return a q copy."""
    path = Path(packet_path).resolve()
    packet = json.loads(path.read_text(encoding='utf8'))
    parent = np.asarray(q, dtype=np.float64)
    physical = packet.get('schema') == PHYSICAL_SCHEMA
    if (packet.get('schema') not in (POSE_SCHEMA, PHYSICAL_SCHEMA)
            or packet.get('role') != 'initializer_only'
            or packet.get('accuracy_accepted') is not False
            or packet.get('source_video_sha256') != SOURCE_SHA
            or packet.get('time_base') != '1/15360'
            or parent.ndim != 1 or not np.isfinite(parent).all()
            or _q_sha(parent) != packet.get('parent_q_sha256')):
        raise ValueError('Exact parent and explicit same-source camera initializer required')
    actual_pts = packet.get('actual_pts')
    if (not isinstance(actual_pts, list) or not actual_pts
            or any(type(p) is not int for p in actual_pts)
            or len(set(actual_pts)) != len(actual_pts)):
        raise ValueError('Unique actual nonreference exposure PTS required')
    allowed = []
    installation = packet.get('installation_id') if physical else None
    pivot_slot = None
    if physical and (not isinstance(installation, str) or not installation
            or packet.get('derived_view_estimates_are_not_source_measurements') is not True
            or packet.get('shared_world_Pxy_lever_and_optics_unchanged') is not True):
        raise ValueError('Explicit same-installation Pz/RF initialization scope required')
    for pts in actual_pts:
        if pts not in state.bypts:
            raise ValueError('Initializer exposure is absent from current source')
        block = state.bypts[pts]
        if not physical and pts == int(block.bundle.reference_pts):
            raise ValueError('Initializer cannot change a fixed reference exposure')
        if physical:
            if block.bundle.calibration.installation_id != installation:
                raise ValueError('Shared pivot initialization cannot cross installations')
            slots = np.asarray(block.xyz_slots)
            if slots.shape != (3,) or slots.dtype.kind not in 'iu':
                raise ValueError('Existing installation XYZ parameter slots required')
            current_pivot_slot = int(slots[2])
            if pivot_slot is not None and pivot_slot != current_pivot_slot:
                raise ValueError('One physical installation must share one pivot Z slot')
            pivot_slot = current_pivot_slot
        start = int(block.start + 4 * block.bundle.index[pts])
        allowed.extend(range(start, start + 4))
    if physical:
        if (type(packet.get('existing_shared_pivot_Z_parameter_index')) is not int
                or packet['existing_shared_pivot_Z_parameter_index'] != pivot_slot):
            raise ValueError('Initializer must name the existing shared pivot Z slot')
        allowed.append(pivot_slot)
    evidence = packet.get('source_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Real source evidence required for a camera initializer')
    covered = set()
    for record in evidence:
        native = record.get('pts')
        if (type(native) is not int or not 229114112 <= native < 231705600
                or native not in state.original.aliases
                or int(state.original.aliases[native]) not in actual_pts
                or _sha(record['path']) != record.get('sha256')):
            raise ValueError('Initializer evidence has changed or belongs to another exposure')
        covered.add(int(state.original.aliases[native]))
    if physical:
        anchors = packet.get('anchor_actual_pts')
        if (not isinstance(anchors, list) or not anchors
                or any(type(p) is not int for p in anchors)
                or len(set(anchors)) != len(anchors)
                or covered != set(anchors)):
            raise ValueError('Every declared source anchor must have unchanged real evidence')
    elif covered != set(actual_pts):
        raise ValueError('Every initialized exposure must have its own source evidence')
    delta = packet['delta_npz']
    if _sha(delta['path']) != delta['sha256']:
        raise ValueError('Camera initializer values changed')
    with np.load(delta['path'], allow_pickle=False) as saved:
        if set(saved.files) != {'parameter_indices', 'original_values', 'candidate_values'}:
            raise ValueError('Initializer must contain only explicit old/new camera slots')
        indices = saved['parameter_indices'].copy()
        old = saved['original_values'].copy()
        new = saved['candidate_values'].copy()
    if (indices.dtype.kind not in 'iu' or indices.ndim != 1 or not len(indices)
            or len(set(map(int, indices))) != len(indices)
            or not set(map(int, indices)).issubset(set(allowed))
            or old.shape != indices.shape or new.shape != indices.shape
            or not np.isfinite(old).all() or not np.isfinite(new).all()
            or not np.array_equal(parent[indices], old)):
        raise ValueError('Initializer would change protected state or does not match original values')
    candidate = parent.copy()
    candidate[indices] = new
    if physical and (pivot_slot not in indices or candidate[pivot_slot] <= 0.):
        raise ValueError('Existing shared pivot Z must be explicit and remain above ice')
    changed = np.flatnonzero(candidate != parent)
    report = dict(
        schema='rinkspace.explicit-source-camera-initialization.v2' if physical else 'rinkspace.explicit-source-camera-pose-initialization.v1',
        initializer_packet=dict(path=str(path), sha256=_sha(path)),
        parent_q_sha256=_q_sha(parent), initialized_q_sha256=_q_sha(candidate),
        actual_pts=actual_pts, changed_parameter_indices=changed.tolist(),
        source_evidence=evidence, source_measurements_modified=False,
        shared_pivot_lever_optics_world_or_other_exposures_changed=physical,
        shared_pivot_Z_initialized=physical,
        installation_id=installation,
        shared_world_Pxy_lever_and_optics_changed=False,
        derived_view_estimates_are_not_source_measurements=physical,
        parent_used_readonly=True, initializer_only=True, accuracy_accepted=False,
        fresh_complete_source_profiles_and_jacobian_required=True)
    return candidate, report


def apply_absolute_camera_pose_initializer(state, q, packet_path):
    from installation_mount_candidate.mount_graph import validate_mounts
    candidate, report = verified_camera_pose_estimate(state, q, packet_path)
    validate_mounts(state, candidate)
    report['physical_mount_domain_validated'] = True
    return candidate, report


def reinitialize_invalid_qualified_source_views(state,q,*,protected_actual_pts=()):
    """Restore original finite-source ray domains after a shared-world edit.

    This is a numerical RF4 starting estimate on already qualified source
    pixels. It preserves all shared geometry/hardware/lens values and source
    rows. Every RF coordinate remains active in the following full solve.
    """
    import cv2
    from scipy.optimize import minimize
    from z190_continuous_source_bundle import radial_inverse
    from installation_mount_candidate.mount_graph import validate_mounts
    parent=np.asarray(q,float);candidate=parent.copy();protected=set(map(int,protected_actual_pts));repairs=[]
    core_count=int(state.structured_programme_coordinate_chart.core_count)
    for block in state.blocks:
        bundle=block.bundle
        if not bundle.finite_records:continue
        local=block.local(candidate[:core_count]);cal=bundle.current_calibration(local)
        poly=np.asarray(bundle.scene['ice_boundary']['points_xyz'],float)[:,:2].astype(np.float32)
        top=float(np.max(np.asarray(bundle.scene['board_surface']['top_edge_xyz'],float)[:,2]))
        groups={}
        for rec in bundle.finite_records:
            j=int(rec['global_bundle_index']);i=int(bundle.ia[j]);groups.setdefault(int(bundle.stamps[i]),[]).append(rec)
        for pts,records in groups.items():
            slots=block.start+4*bundle.index[pts]+np.arange(4)
            initial=candidate[slots].copy()
            def domains(rf,*,interior=False):
                rotation=cv2.Rodrigues(np.asarray(rf[:3],float))[0]
                centre=np.asarray(cal.pivot_xyz,float)+rotation.T@np.asarray(cal.lever_xyz,float)
                pp=np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]])
                focal=np.exp(rf[3]);values=[];facts=[]
                for rec in records:
                    j=int(rec['global_bundle_index']);xy,_=radial_inverse((bundle.a[j:j+1]-pp)/focal,cal.k1[0])
                    ray=rotation.T@np.r_[xy[0],1.];normal,d=bundle.finite_planes[j,:3],bundle.finite_planes[j,3]
                    den=float(normal@ray)
                    if not np.isfinite(den)or abs(den)<1e-10:
                        return np.full(sum(2 if r['surface_kind']=='ice_plane'else 5 for r in records),-1e6),[]
                    distance=-float(normal@centre+d)/den;world=centre+distance*ray
                    values.append(distance-1e-8)
                    if rec['surface_kind']=='ice_plane':
                        margin=float(cv2.pointPolygonTest(poly,tuple(map(float,world[:2])),True))
                        # A small initialization-only interior prevents the
                        # original coordinate FD from straddling this domain.
                        values.append(margin-(.01 if interior else 0.))
                        detail=dict(ice_signed_margin_scene_units=margin)
                    else:
                        start,end=np.asarray(rec['extent_start_xy']),np.asarray(rec['extent_end_xy']);delta=end-start
                        station=float((world[:2]-start)@delta/(delta@delta))
                        height_margin=.01 if interior else 0.
                        station_margin=1e-5 if interior else 0.
                        values.extend((float(world[2])+1e-6-height_margin,top+1e-6-float(world[2])-height_margin,
                            station+1e-6-station_margin,1.+1e-6-station-station_margin))
                        detail=dict(wall_station=station,wall_height_scene_units=float(world[2]))
                    facts.append(dict(original_point_index=j,actual_pts=pts,surface_kind=rec['surface_kind'],
                        original_source_uv=bundle.a[j].tolist(),ray_depth_scene_units=distance,world_scene_units=world.tolist(),**detail))
                return np.asarray(values,float),facts
            before,before_facts=domains(initial)
            if np.isfinite(before).all()and np.min(before)>=0.:continue
            if pts in protected:raise ValueError('A protected current manual exposure has an invalid original finite source domain: '+str(pts))
            scale=np.array([.01,.01,.01,.02])
            def changed(x):return initial+scale*np.asarray(x,float)
            optimized=minimize(lambda x:.5*float(np.asarray(x)@np.asarray(x)),np.zeros(4),jac=lambda x:np.asarray(x,float),
                method='SLSQP',bounds=[(None,None)]*3+[((float(np.log(100.))-initial[3])/scale[3],(float(np.log(50000.))-initial[3])/scale[3])],
                constraints=[dict(type='ineq',fun=lambda x:domains(changed(x),interior=True)[0])],
                # x is conditioned by .01 rad / .02 logf. The original
                # float32 ice polygon needs a real RF perturbation, rather
                # than SLSQP's default sub-quantization camera displacement.
                options=dict(ftol=1e-10,maxiter=160,eps=1e-4))
            proposed=changed(optimized.x);after,after_facts=domains(proposed)
            if not np.isfinite(proposed).all()or not np.isfinite(after).all()or np.min(after)<0.:
                raise ValueError('Original finite-source RF4 initialization did not recover '+block.name+' PTS '+str(pts)+': '+str(optimized.message))
            candidate[slots]=proposed
            repairs.append(dict(block=block.name,actual_pts=pts,view_parameter_indices=slots.tolist(),original_RF4=initial.tolist(),
                initialized_RF4=proposed.tolist(),optimizer_status=str(optimized.message),before=before_facts,after=after_facts,
                initialization_only=True,original_source_UV_planes_extents_roles_and_sigma_unchanged=True))
        bundle.validate_finite_extents(bundle.cache(block.local(candidate[:core_count]),False))
    changed=np.flatnonzero(candidate!=parent);allowed={k for r in repairs for k in r['view_parameter_indices']}
    if not set(changed.tolist()).issubset(allowed):raise ValueError('Finite-source pose initializer changed a shared or unrelated parameter')
    validate_mounts(state,candidate[:core_count])
    return candidate,dict(schema='rinkspace.original-qualified-finite-source-view-domain-initialization.v1',
        parent_q_sha256=_q_sha(parent),initialized_q_sha256=_q_sha(candidate),repairs=repairs,
        changed_parameter_indices=changed.tolist(),shared_world_pivots_mounts_PP_and_Brown_unchanged=True,
        current_manual_exposures_preserved=sorted(protected),source_rows_UV_sigma_roles_planes_and_extents_unchanged=True,
        derived_camera_views_are_initializers_not_observations=True,all_RF_coordinates_active_in_full_solver=True,
        initialization_only_ice_and_wall_height_interior_scene_units=.01,
        initialization_only_wall_station_interior=1e-5,original_finite_hard_domains_unchanged=True,
        fresh_complete_source_profiles_and_J_required=True,accuracy_accepted=False)
