"""Admit source-bound material candidates as queries, before RGB remeasurement."""
from copy import deepcopy
import hashlib
import numpy as np
from .vision_geometry import _same_frame


def admit_material_candidates(rgb, binding, candidates, geometry, *, existing_guides=(), exclusion=None):
    _same_frame(binding, candidates["source_binding"])
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    if candidates.get("source_RGB_sha256", candidates.get("RGB_sha256")) != digest:
        raise ValueError("Material candidates refer to another current RGB image")
    geometry.assert_unchanged()
    shape = rgb.shape[:2]
    excluded = np.zeros(shape, bool) if exclusion is None else np.asarray(exclusion)
    if excluded.dtype != np.bool_ or excluded.shape != shape:
        raise ValueError("Candidate admission needs the current native boolean exclusion mask")
    proposed = {row["id"]: row for row in candidates.get("guides", [])}
    accepted, pending, rejected, eligible = [], [], [], []
    paired_verification=[];material_frame=None
    association_states = {"human_reference_initialized_source_instance_correspondence",
                          "observed_RGB_material_layer_under_human_named_parent"}
    machine_states={'verified_RGB_material_initialized_source_instance_correspondence',
                    'observed_RGB_material_layer_under_machine_named_parent'}
    for row in candidates.get("source_shape_readings", []):
        identity = row.get("id")
        def reject(reason, unresolved=False):
            (pending if unresolved else rejected).append({"candidate_id": identity, "reason": reason,
                "physical_curve_id": row.get("physical_curve_id"), "source_evidence_preserved": True})
        _same_frame(binding, row["source_binding"])
        proposal = proposed.get(identity)
        definition = geometry.lookup(row.get("physical_curve_id"))
        if proposal is None or not row.get("physical_identity_verified") or not row.get("identity_authority"):
            reject("missing_independent_physical_identity_authority", True);continue
        association = proposal.get("source_identity_association", {})
        human=(proposal.get('human_confirmed_instance_identity_initialized') is True and
               association.get('status') in association_states)
        machine=(proposal.get('source_instance_identity_initialized') is True and
                 association.get('status') in machine_states)
        seed=(proposal.get('human_confirmed_identity_source_binding') if human else
              proposal.get('source_identity_source_binding') if machine else None)
        if not seed:
            reject("unresolved_source_identity_competition", True);continue
        if any(seed.get(k) != binding.get(k) for k in ("source_sha256", "time_base", "shot_id", "event_id", "rig_id", "installation_id")):
            reject("identity_authority_belongs_to_another_source_or_installation");continue
        if definition is None or definition["layer"] != "ice":
            reject("no_retained_native_material_path", True);continue
        paired_supports=[];paired_signals=set()
        if machine:
            from .vision_material_reader import material_readout_identity,paired_material_support
            proofs=row['identity_authority']
            if not isinstance(proofs,list):proofs=[proofs]
            verified=[]
            for proof in proofs:
                anchor=geometry.lookup(proof.get('native_path_id'))
                if (proof.get('kind')!='current_RGB_verified_material_identity_for_shape' or
                    proof.get('current_native_scene')!=geometry.refs[0] or
                    proof.get('human_pixel_acceptance') is not False or
                    anchor is None or anchor.get('layer')!='ice' or
                    anchor.get('label',anchor['id']).rsplit(':',1)[0]!=definition.get('label',definition['id']).rsplit(':',1)[0]):continue
                evidence=proof.get('source_readout_evidence',{})
                members=evidence.get('identity_by_source_index',{})
                old=proof.get('source_binding',{})
                if old!=seed or int(old['source_pts'])>int(binding['source_pts']):continue
                reading=dict(native_path_id=proof['native_path_id'],support=dict(actual_readout_evidence=evidence))
                if len(members)>=2 and all(material_readout_identity(reading,old,
                    proof['decoded_native_RGB_sha256'],int(i),member['actual_source_uv']) is not None
                    for i,member in members.items()):verified.append(proof)
            if not verified:
                reject('machine_shape_identity_has_no_current_native_source_material_proof',True);continue
            if association.get('requires_two_sided_center_reread') is True:
                for proof in verified:
                    if geometry.lookup(proof['native_path_id'])['id']!=definition['id']:continue
                    members=proof['source_readout_evidence']['identity_by_source_index'].values()
                    supports=[paired_material_support(member) for member in members]
                    if len(supports)>=2 and all(support is not None for support in supports):
                        paired_supports.extend(supports)
                        paired_signals.update(member.get('material_signal') for member in members)
        expected = "painted_stroke_center_axis" if definition["measurement"] == "centreline" else "filled_disc_boundary"
        paired_navigation=bool(machine and expected=='painted_stroke_center_axis' and len(paired_supports)>=2
            and association.get('correspondence_basis')=='actual_paired_material_support'
            and association.get('source_peak_is_center_measurement') is False
            and paired_signals in ({'red'},{'blue'},{'dark'}))
        if association.get('requires_two_sided_center_reread') and not paired_navigation:
            reject('paired_actual_material_support_for_center_query_missing',True);continue
        if not paired_navigation and (row.get("physical_role") != expected or row.get("native_material_role_matches") is not True):
            reject("material_role_requires_independent_resolution", True);continue
        if row.get("fitted_or_flow_pixels_used_as_measurements") is not False:
            reject("candidate_has_no_direct_RGB_measurement_provenance");continue
        uv = np.asarray(row["actual_source_uv"], float)
        indices = np.asarray(row["original_source_profile_indices"])
        normals = np.asarray(proposal.get("normals", []), float)
        if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2 or not np.isfinite(uv).all() or
                indices.shape != (len(uv),) or indices.dtype.kind not in "iu" or np.any(indices < 0) or
                normals.ndim != 2 or normals.shape[1] != 2 or np.any(indices >= len(normals))):
            reject("incomplete_current_RGB_query_support");continue
        competing = np.asarray(row.get("source_quality_flags", {}).get("competing_same_sign_material_transition_peaks", []))
        if competing.shape != (len(uv),) or competing.dtype != np.bool_:
            reject("unresolved_material_transition_competition", True);continue
        paired_by_index={}
        if paired_navigation:
            if association.get('paired_profile_source_binding')!=binding:
                reject('paired_material_profiles_need_current_exposure_correspondence',True);continue
            for item in association.get('matched_material_profiles',[]):
                index=int(item['profile_index'])
                if index in paired_by_index:paired_by_index[index]=None
                else:paired_by_index[index]=item
            supported=np.array([paired_by_index.get(int(i)) is not None and
                np.array_equal(paired_by_index[int(i)]['actual_source_interface_uv'],point)
                for i,point in zip(indices,uv)],bool)
            if not supported.all():
                pending.append(dict(candidate_id=identity,physical_curve_id=row.get('physical_curve_id'),
                    reason='edge_profiles_outside_actual_paired_material_identity',
                    original_source_profile_indices=indices[~supported].tolist(),source_evidence_preserved=True))
            uv,indices,competing=uv[supported],indices[supported],competing[supported]
            if len(uv)<2:continue
        _,inverse,counts=np.unique(indices,return_inverse=True,return_counts=True)
        competing=competing|(counts[inverse]>1)
        if competing.any():
            pending.append(dict(candidate_id=identity,physical_curve_id=row.get('physical_curve_id'),
                reason='unresolved_material_transition_profiles',
                original_source_profile_indices=indices[competing].tolist(),source_evidence_preserved=True))
            # One ambiguous scan cannot erase a separate readable arc. Keep
            # original candidates intact; only clear profiles become queries.
            uv,indices=uv[~competing],indices[~competing]
            if len(uv)<2:continue
        pair_evidence=None
        if paired_navigation:
            from .vision_source_shapes import _MaterialProfileFrame,read_source_stroke_midlines
            from .vision_ice_source_guides import _candidate_excluded
            profiles=[paired_by_index[int(i)] for i in indices]
            q=np.asarray([p['query_uv'] for p in profiles],float)
            n=np.asarray([p['normal'] for p in profiles],float)
            windows=np.asarray([p['interval_px'] for p in profiles],float)
            if np.any(windows[:,0]>-6.) or np.any(windows[:,1]<6.):
                reject('paired_material_side_search_context_incomplete',True);continue
            if material_frame is None:material_frame=_MaterialProfileFrame(rgb,binding,digest)
            signal=next(iter(paired_signals))
            pair_evidence=read_source_stroke_midlines(material_frame.rgb,binding=binding,
                search_points_uv=q,normals=n,
                entering_search_intervals_px=np.c_[windows[:,0],np.zeros(len(q))],
                leaving_search_intervals_px=np.c_[np.zeros(len(q)),windows[:,1]],
                material_signal_rgb_weights={'dark':None,'red':[1.,-1.,0.],'blue':[-1.,0.,1.]}[signal],
                paint_signal_increases=signal!='dark',expected_RGB_sha256=digest,_validated_frame=material_frame)
            clear=[]
            for i,reading in enumerate(pair_evidence['readings']):
                sides=np.asarray([reading[key]['actual_source_transition_candidate_uv'] for key in
                    ('actual_RGB_entering_interface','actual_RGB_leaving_interface')])
                available=reading['disposition']=='measured_paired_stroke_midline'
                clear.append(available and not _candidate_excluded(excluded,sides.mean(0),sides,n[i]))
            clear=np.asarray(clear,bool)
            paired_verification.append(dict(candidate_id=identity,original_source_profile_indices=indices.copy(),
                current_RGB_two_sided_reading=pair_evidence,source_evidence_preserved=True))
            if not clear.all():
                pending.append(dict(candidate_id=identity,physical_curve_id=row.get('physical_curve_id'),
                    reason='current_RGB_two_material_sides_unresolved_or_excluded',
                    original_source_profile_indices=indices[~clear].tolist(),
                    paired_verification_index=len(paired_verification)-1,source_evidence_preserved=True))
            uv,indices=uv[clear],indices[clear]
            if len(uv)<2:continue
        xy = np.rint(uv).astype(np.int64)
        inside = (xy[:,0] >= 0) & (xy[:,0] < shape[1]) & (xy[:,1] >= 0) & (xy[:,1] < shape[0])
        if not inside.all() or excluded[xy[:,1],xy[:,0]].any():
            reject("candidate_current_support_is_occluded_or_excluded");continue
        normal = normals[indices]
        length = np.linalg.norm(normal, axis=1)
        if not np.isfinite(normal).all() or np.any(length <= 0):
            reject("candidate_has_no_valid_profile_normals");continue
        order = np.argsort(indices, kind="stable")
        uv, normal, indices = uv[order], normal[order]/length[order,None], indices[order]
        segments = np.cumsum(np.r_[0, np.diff(indices) != 1]).astype(np.int64)
        query_id = definition["id"]+"/source_candidate/"+str(identity)
        authority = dict(kind="current_RGB_candidate_admission", source_binding=deepcopy(binding),
            current_native_scene=deepcopy(geometry.refs[0]), prior_identity_authority=deepcopy(row["identity_authority"]),
            source_identity_association=deepcopy(association), decoded_native_RGB_sha256=digest,
            physical_identity_from_model_projection=False, human_pixel_acceptance=False)
        if pair_evidence is not None:
            authority['current_RGB_two_sided_material_verification']=pair_evidence
        guide = dict(id=definition["id"], query_id=query_id, kind="ice_paint", points_uv=uv,
            normals=normal, segment_ids=segments, source_binding=deepcopy(binding),
            origin="source_annotation_search_hint", identity_confirmed=True, identity_authority=authority,
            physical_role=expected, original_source_profile_indices=indices,
            source_group_id="candidate:"+query_id, search_radius_px=float(proposal.get("search_radius_px", 18.)),
            candidate_navigation_only=True, accepted_source_UV_modified=False,
            source_exclusion_mask=excluded.copy(), source_exclusion_binding=deepcopy(binding))
        if expected == "painted_stroke_center_axis":guide["signal"] = proposal.get("signal", "dark_band")
        if paired_navigation:
            guide['signal']={'dark':'dark_band','red':'red_excess','blue':'blue_excess'}[next(iter(paired_signals))]
            profiles=[paired_by_index[int(i)] for i in indices]
            # Only current, individually matched interfaces can search for a
            # center. Use the observed paint pair's normal and window rather
            # than a crossing candidate curve's tangent or a global radius.
            guide['source_material_interface_query_uv']=uv.copy()
            guide['points_uv']=np.asarray([p['query_uv'] for p in profiles],float)
            guide['normals']=np.asarray([p['normal'] for p in profiles],float)
            guide['material_profile_intervals_px']=np.asarray([p['interval_px'] for p in profiles],float)
            guide['source_boundary_requires_native_center_reread']=True
        from .vision_navigation import _uncovered_query
        guide = _uncovered_query(guide, existing_guides, geometry=geometry)
        if guide is None:
            reject("already_measured_source_query_support");continue
        eligible.append((identity, definition["id"], guide))
    groups={}
    for item in eligible:groups.setdefault(item[1],[]).append(item)
    for path_id,items in groups.items():
        actual_material_fragments=(len({item[0] for item in items})==len(items) and all(
            item[2]['identity_authority']['source_identity_association'].get('correspondence_basis') in
                ('actual_signed_transition_support','actual_paired_material_support')
            for item in items))
        if len(items)>1 and not actual_material_fragments:
            pending.extend(dict(candidate_id=item[0],reason='multiple_candidates_compete_for_native_path',native_path_id=path_id) for item in items)
            continue
        from .vision_navigation import _uncovered_query
        for identity,_,guide in items:
            remainder=_uncovered_query(guide,[*existing_guides,*accepted],geometry=geometry)
            if remainder is not None:accepted.append(remainder)
            else:rejected.append(dict(candidate_id=identity,reason='duplicate_actual_material_support',source_evidence_preserved=True))
    bound, report = geometry.bind(accepted)
    if report.get("unresolved") or len(bound) != len(accepted):
        raise ValueError("Candidate admission lost its retained native path")
    return dict(accepted_current_query_guides=bound, pending_identity=pending, rejected=rejected,
        paired_material_verification=paired_verification,
        source_binding=deepcopy(binding), decoded_native_RGB_sha256=digest, current_native_scene=geometry.refs[0],
        candidate_UV_used_as_final_measurements=False, actual_RGB_reread_required=True)
