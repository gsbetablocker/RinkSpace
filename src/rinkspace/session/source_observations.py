"""Source facts, identity proposals and current input export are separate stages."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
import hashlib, json, re
import numpy as np


@dataclass
class SourceCandidate:
    candidate_id: str
    shot_id: str
    native_pts: int
    source_ref: dict
    original_uv: list
    source_feature_id: str
    actual_uv: list | None = None
    normal: list | None = None
    localization: str = 'unverified'
    identity: str = 'unresolved'
    native_path_id: str | None = None
    measurement: str | None = None
    source_group_id: str | None = None
    review: dict = field(default_factory=dict)
    withdrawal: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    origin: str = 'legacy_import'


def _list(x):
    return x.detach().cpu().tolist() if hasattr(x,'detach') else np.asarray(x).tolist()


def current_definition(geometry, name):
    # Explicit spelling aliases only. No historical coordinates or artwork are imported.
    if name == 'board.contact':name='board.bottom'
    if name == 'ice.C0:outer':name='ice.C0:centreline'
    return geometry.lookup(name)


def _matches_semantics(definition, support):
    if definition is None:return False
    if definition['layer']=='board':
        text=str(support.get('definition',support.get('measurement_definition','')))
        return ('blue' in text and ('upper' in text or 'top' in text)) if definition['label']=='board.top' else ('yellow' in text and ('lower' in text or 'contact' in text))
    text=str(support.get('definition', support.get('measurement_definition','')))
    if definition['layer']=='advertisement':return False  # actual retained print correspondence is separate
    center=definition.get('measurement')=='centreline'
    if center:return 'midpoint' in text or 'midline' in text or 'center' in text or 'centreline' in text
    return 'material_boundary' in text or 'material_interface' in text or 'ring' in text


def _board_frame_semantics(resolved,support,shot,pts,selected_row=None):
    """Read the typed frame-level material proof of a selected original row."""
    context=resolved.get('source_frame_context') or {}
    schema=context.get('schema')
    if schema not in ('rinkfit.dense_native_board_true_material_boundaries_source_frame.v1',
            'rinkfit.dense_native_board_source_frame.v1'):return None
    if context.get('native_pts')!=pts or context.get('shot_id')!=shot or context.get('time_base')!=support.get('time_base'):
        raise ValueError('Board frame material evidence belongs to another exposure')
    if (not {'board.top','board.contact'}.issubset(context.get('material_roles',[])) or
        len(context.get('decoded_native_BGR_sha256',''))!=64 or
        support.get('pixel_support',{}).get('current_pixel_profile_reread') is not True):return None
    if support.get('feature_id') not in ('board.top','board.contact'):return None
    if context.get('source_video_sha256') and context['source_video_sha256']!=support.get('source_video_sha256'):
        raise ValueError('Board frame material evidence belongs to another source')
    selected_definition=None
    if schema=='rinkfit.dense_native_board_true_material_boundaries_source_frame.v1':
        if (context.get('paired_blue_cap_and_white_board_body_semantics') is not True or
            not (context.get('yellow_contact_from_current_global_band_not_accumulated_extrapolation') is True or
                 context.get('yellow_contact_from_current_native_material_not_reflection') is True)):return None
    else:
        # This retained generation records semantics on its selected row and
        # the exact UV/profile on the referenced frame row. Neither fact may
        # be discarded merely because they are stored at different levels.
        selected_row=selected_row or {}
        qualification=context.get('local_material_requalification',{})
        selected_definition=support.get('definition',support.get('measurement_definition',
            selected_row.get('measurement_definition','')))
        text=str(selected_definition).lower()
        material_definition=(('blue' in text and 'upper' in text and 'vertical board' in text)
            if support['feature_id']=='board.top' else ('yellow' in text and 'lower' in text))
        if (not material_definition or qualification.get('actual_current_RGB_reread') is not True or
            qualification.get('original_anchored_source_UV_preserved') is not True or
            selected_row.get('source_UV_is_direct_measurement') is not True or
            support.get('pixels_measured_in_this_frame') is not True or
            support.get('model_or_world_projection_used') is not False):return None
    definition=('blue_band_upper_material_entry' if support['feature_id']=='board.top'
                else 'yellow_band_lower_contact_material_exit')
    return dict(kind='authenticated_original_board_material_frame',source_frame=context,
        source_provenance=resolved['provenance'],definition=definition,source_board_view='front_face',
        selected_source_measurement_definition=selected_definition,
        original_source_UV_rewritten=False,human_pixel_acceptance=False)


def normalize_source(envelope, resolver, geometry):
    """Retain originals even when an evidence/identity check cannot admit them."""
    row=envelope.source_payload
    if 'packet' in row:return list(_packet_candidates(envelope,geometry))
    if 'uv' not in row:return []
    name=row.get('feature_id','');ref=row.get('source_pixel_or_annotation_ref',{})
    c=SourceCandidate(row.get('source_point_id',row.get('selected_row_id')),envelope.shot_id,
        envelope.native_pts,dict(envelope.source_ref,support_ref=ref),list(row['uv']),name,
        review=dict(original_user_confirmed=row.get('user_confirmed'),human_scope_extended=False),
        evidence=dict(selected_row_id=row.get('selected_row_id'),source_kind=row.get('source_kind'),
            latest_qualification_refs=envelope.latest_qualification_refs))
    try:resolved=resolver.resolve(ref);support=resolved['payload']
    except (FileNotFoundError,ValueError,KeyError,IndexError) as exc:
        c.evidence['support_gap']=str(exc);return [c]
    c.evidence['support_provenance']=resolved['provenance']
    if support.get('native_pts') != c.native_pts or support.get('shot_id',c.shot_id)!=c.shot_id:
        raise ValueError('Source support belongs to another exposure')
    uv=support.get('uv')
    if uv is None or not np.array_equal(np.asarray(uv,float),np.asarray(c.original_uv,float)):
        c.evidence['support_gap']='selected UV differs from referenced source record';return [c]
    definition=current_definition(geometry,name)
    if definition:c.native_path_id=definition['id'];c.measurement=definition.get('measurement','contour')
    c.source_group_id=support.get('visible_span_id',support.get('segment_id'))
    c.evidence.update(source_support_status=support.get('status'),
        source_definition=support.get('definition',row.get('measurement_definition')),
        source_identity_navigation=support.get('navigation'),
        original_source_group_id=c.source_group_id,
        original_independence=support.get('samples_independent'),
        physical_world_correspondence_issued=row.get('physical_world_correspondence_issued'),
        original_source_row_id=support.get('source_row_id',support.get('point_id')))
    pixel=support.get('pixel_support',{})
    normal=pixel.get('normal')
    if normal is None and pixel.get('position_interval_endpoints_uv'):
        ends=np.asarray(pixel['position_interval_endpoints_uv']);normal=(ends[1]-ends[0]).tolist()
    if normal is not None:
        normal=np.asarray(normal,float);length=np.linalg.norm(normal)
        if normal.shape==(2,) and np.isfinite(length) and length>0:c.normal=(normal/length).tolist()
    direct=(row.get('source_UV_is_direct_measurement') is True and
        support.get('source_UV_is_direct_measurement',True) is not False and
        support.get('source_uv_from_model',False) is False and
        support.get('model_or_world_projection_used',False) is False and
        support.get('pixels_measured_in_this_frame',True) is not False)
    material=(pixel.get('current_pixel_profile_reread') is True or bool(pixel.get('side_interface_quartiles_uv'))
        or pixel.get('full_plus_minus_3px_profile_outside_current_SAM2_mask') is True
        or pixel.get('complete_plus_minus_3px_RGB_profile_qualified') is True)
    manual=support.get('source_kind')=='preserved_user_manual_native_image_click' and support.get('original_UV_unchanged') is True
    if manual:
        # Authenticate the exact historical annotation; this does not approve an entire frame.
        original=resolver.resolve(dict(file=support['source_file'],sha256=support['source_sha256'],json_pointer=support['source_pointer']))
        manual=np.array_equal(original['payload'],c.original_uv)
        if manual:c.review.update(human_source_ref=original['provenance'],scope='exact_original_annotation_point')
    if direct and (material or manual):
        c.actual_uv=list(c.original_uv);c.localization='measured'
    board_evidence=_board_frame_semantics(resolved,support,c.shot_id,c.native_pts,row)
    semantics=dict(support,definition=board_evidence['definition']) if board_evidence else support
    if board_evidence:c.evidence['source_board_material_frame']=board_evidence
    if definition and not name.startswith('unbound.') and (manual or _matches_semantics(definition,semantics)):
        if definition['layer'] in ('ice','board'):
            c.identity='named_source_candidate'
    c.evidence['current_semantics_match']=bool(definition and _matches_semantics(definition,semantics))
    c.evidence['pixel_support']=pixel
    if support.get('named_entity_status') or support.get('world_entity_binding')=='unbound_not_for_physical_fit':
        c.identity='unresolved';c.evidence['identity_gap']=support.get('named_entity_status','explicitly_unbound')
    return [c]


def _packet_board_semantics(envelope, feature, segment, definition):
    """Read typed same-exposure board materials from the archived packet schema.

    The packet's fitted display vertices are not used. Its paired blue-cap and
    yellow-body definitions authorize a fresh material query, not new pixels,
    advertisement identity, or human acceptance.
    """
    if not definition or definition['layer']!='board':return None
    packet=envelope.source_payload['packet'];source=packet.get('source',{})
    if (packet.get('schema')!='rinkspace.single-frame-source-review.v1' or
        packet.get('source_pts')!=envelope.native_pts or source.get('native_pts')!=envelope.native_pts or
        source.get('original_PTS_verified') is not True or source.get('PNG_roundtrip_lossless_verified') is not True or
        packet.get('observation_coordinates_from_projection') is not False):return None
    guard=packet.get('board_material_ownership_guard',{})
    paired_guard=(guard.get('source')=='current_measured_blue_cap_top_and_yellow_body_bottom' and
        guard.get('used_only_for_exclusion_not_observations') is True)
    bound_source=(packet.get('current_exact_source_binding_and_mask_verified') is True and
        packet.get('current_source_only_readout') is True and bool(packet.get('source_binding')) and
        packet['source_binding']==source.get('source_binding'))
    if not (paired_guard or bound_source):return None
    roles={'board_top':'blue_cap_upper_body_edge',
           'board_contact':'yellow_kickplate_body_lower_edge_reflection_excluded'}
    def typed(f,s):
        return (f.get('physical_surface')=='board' and f.get('review_layer') in roles and
            s.get('native_material_definition')==roles[f['review_layer']] and
            s.get('measurement_origin')=='current_native_whole_material_joint_curve' and
            s.get('current_interval_local_colour_support_count',0)>0 and
            len(s.get('source_native_readout_uv',[]))>=2 and
            not f.get('current_source_identity_review_rejected') and
            not s.get('current_source_identity_review_rejected'))
    if not typed(feature,segment):return None
    present={f['review_layer'] for f in packet.get('features',[]) for s in f.get('segments',[]) if typed(f,s)}
    if present!=set(roles):return None
    label='board.top' if feature['review_layer']=='board_top' else 'board.bottom'
    if definition['label']!=label:return None
    digest=source.get('decoded_BGR_sha256');video_sha=source.get('source_video_sha256')
    if not all(isinstance(v,str) and len(v)==64 for v in (digest,video_sha)):return None
    context=dict(schema=packet['schema'],native_pts=envelope.native_pts,shot_id=envelope.shot_id,
        time_base=source['time_base'],source_video_sha256=video_sha,decoded_native_BGR_sha256=digest,
        material_roles=['board.top','board.contact'])
    return dict(kind='authenticated_original_board_material_frame',source_frame=context,
        source_provenance=envelope.source_ref,definition=roles[feature['review_layer']],
        source_board_view='front_face',original_source_UV_rewritten=False,human_pixel_acceptance=False,
        original_fitted_representatives_used=False,current_RGB_material_reread_required=True,
        packet_material_context='paired_board_ownership_guard' if paired_guard else 'exact_source_binding_and_mask')


def _packet_candidates(envelope, geometry):
    packet=envelope.source_payload['packet']
    for fi,f in enumerate(packet.get('features',[])):
        name=f['id'];definition=current_definition(geometry,name)
        yield from _packet_direct_readouts(envelope,fi,f,definition)
        for si,segment in enumerate(f.get('segments',[])):
            board_evidence=_packet_board_semantics(envelope,f,segment,definition)
            evidence=segment.get('source_evidence',[])
            native=segment.get('source_native_readout_uv')
            if native is not None:
                points=[(uv,uv,{},j) for j,uv in enumerate(native)]
                origin='source_native_readout_uv'
            else:
                points=[(uv,(evidence[j].get('actual_native_readout_uv') if j<len(evidence) else
                    uv if segment.get('pixels_measured_in_this_frame') is True else None),
                    evidence[j] if j<len(evidence) else {},j) for j,uv in enumerate(segment.get('uv',[]))]
                origin='source_evidence/actual_native_readout_uv'
            for original,actual,e,j in points:
                c=SourceCandidate(f"{envelope.shot_id}/{envelope.native_pts}/{name}/{si}/{origin}/{j}",
                    envelope.shot_id,envelope.native_pts,
                    dict(envelope.source_ref,json_pointer=f'/features/{fi}/segments/{si}',actual_uv_field=origin,source_point_index=j),
                    original,name,actual_uv=actual,
                    normal=e.get('profile_normal'), localization='measured' if actual is not None else 'inferred',
                    source_group_id=segment.get('source_evidence_group_id',f.get('source_evidence_group_id',
                        f'{envelope.shot_id}/{envelope.native_pts}/{name}/segment/{si}')),
                    review=dict(original_user_confirmed=f.get('user_confirmed'),human_scope_extended=False),
                    evidence=dict(source_identity=f.get('physical_identity_evidence'),
                        identity_navigation_origin=f.get('identity_navigation_origin'),
                        original_allowed_for_fit=f.get('allowed_for_fit'),original_point_ids=segment.get('point_ids'),
                        shared_estimate=any(segment.get(k) is True for k in ('representative_points_share_line_estimate','representative_points_share_circle_family_estimate')),
                        precise_source_support=e,withdrawals=packet.get('unresolved_source_regions',[]),
                        material_signal=f.get('material_signal'),material_transition=f.get('material_transition')))
                if definition:
                    c.native_path_id=definition['id'];c.measurement=definition.get('measurement','contour')
                    if definition['layer']=='board' and native is not None:
                        c.identity='named_source_candidate'
                        c.evidence['current_semantics_match']=_matches_semantics(definition,
                            {'definition':segment.get('native_material_definition','')})
                        if board_evidence:c.evidence['source_board_material_frame']=board_evidence
                    if definition['layer']=='ice' and f.get('physical_identity_evidence') and actual is not None:
                        c.identity='named_source_candidate'
                        c.evidence['current_semantics_match']=_matches_semantics(definition,{'definition':e.get('measurement_origin',segment.get('measurement_origin',''))})
                if f.get('current_source_identity_review_rejected') or segment.get('current_source_identity_review_rejected'):
                    c.identity='rejected';c.withdrawal={'reason':'original_identity_review_rejected'}
                yield c


def _packet_direct_readouts(envelope, feature_index, feature, definition):
    """Recover stored RGB profiles, never the fitted line/circle representatives.

    Each support keeps the original correlated family and exact source pointer.
    Historical physical names remain candidates until current association.
    """
    packet=envelope.source_payload['packet'];name=feature['id']
    for field in ('source_line_fit','source_circle_fit'):
        family=feature.get(field) or {}
        for j,raw in enumerate(family.get('direct_readouts',[])):
            if 'uv' not in raw:continue
            evidence=raw.get('source_evidence') or {}
            uv=np.asarray(raw['uv'],float)
            sides=np.asarray(evidence.get('actual_source_sides_uv',[]),float)
            # The archived exporter rounds these original UV to four decimals.
            # This is an integrity check of that encoding, not an accuracy gate.
            matches=(uv.shape==(2,) and sides.shape==(2,2) and
                np.isfinite(uv).all() and np.isfinite(sides).all() and
                np.max(np.abs(sides.mean(axis=0)-uv))<=0.000051)
            direct=(family.get('current_frame_only') is True and
                feature.get('projected_uv_used_as_observations') is False and
                feature.get('source_evidence_measured_in_this_frame') is True and
                evidence.get('actual_source_support_outside_person_and_broadcast_masks') is True and matches)
            semantic=(definition is not None and definition['layer']=='ice' and
                ((definition.get('measurement')=='centreline' and feature.get('measurement_operator')=='paint_center') or
                 (definition.get('measurement')=='contour' and feature.get('measurement_operator')=='material_interface')))
            c=SourceCandidate(f'{envelope.shot_id}/{envelope.native_pts}/{name}/{field}/direct_readouts/{j}',
                envelope.shot_id,envelope.native_pts,
                dict(envelope.source_ref,json_pointer=f'/features/{feature_index}/{field}/direct_readouts/{j}',
                    actual_uv_field='uv',source_point_index=j),list(raw['uv']),name,
                actual_uv=list(raw['uv']) if direct else None,
                normal=evidence.get('profile_normal'),localization='measured' if direct else 'unverified',
                source_group_id=feature.get('source_evidence_group_id',evidence.get('source_evidence_group_id',
                    f'{envelope.shot_id}/{envelope.native_pts}/{name}/{field}')),
                review=dict(original_user_confirmed=feature.get('user_confirmed'),human_scope_extended=False),
                evidence=dict(precise_source_support=evidence,source_family_ref=field,
                    original_source_record=raw,current_semantics_match=semantic,
                    source_identity=feature.get('physical_identity_evidence'),
                    identity_navigation_origin=feature.get('identity_navigation_origin',feature.get('navigation_origin')),
                    direct_profile_roundtrip_verified=bool(matches),
                    original_allowed_for_fit=feature.get('allowed_for_fit'),
                    shared_estimate=True,material_signal=feature.get('material_signal'),
                    material_transition=feature.get('material_transition')))
            if definition:
                c.native_path_id=definition['id'];c.measurement=definition.get('measurement','contour')
            if direct and semantic and feature.get('physical_identity_evidence'):
                c.identity='named_source_candidate'
            rejection=next((r for r in packet.get('unresolved_source_regions',[])
                if r.get('feature_id')==name and r.get('current_source_identity_review_rejected') is True),None)
            if feature.get('current_source_identity_review_rejected') or rejection:
                c.identity='rejected';c.withdrawal=rejection or dict(reason='original_identity_review_rejected')
            yield c


def source_queries(candidates, binding, geometry, *, radius=18.):
    """Saved direct support can anchor search; unknown identity remains unknown."""
    hw=binding.get('native_hw')
    # Legacy source-only callers may lack an image binding. Do not invent its
    # dimensions; current programme bindings always carry the decoded size.
    height=width=np.inf
    if hw is not None:
        if len(hw)!=2 or any(type(v) is not int or v<=0 for v in hw):
            raise ValueError('Source query binding needs positive native height/width')
        height,width=hw
    groups=defaultdict(list)
    for c in candidates:
        if c.native_path_id and not c.withdrawal:
            # A front-face proof applies only to the source run that supplied
            # it. Do not extend it to an unclassified or top-deck run sharing
            # the same continuous native board path.
            view=(c.evidence.get('source_board_material_frame') or {}).get('source_board_view')
            groups[(c.native_path_id,view)].append(c)
    result=[]
    def query_run(candidate):
        if candidate.source_group_id:
            return candidate.source_group_id
        # Original manual points often have no statistical group ID. Their
        # authenticated annotation segment still supplies query connectivity.
        # This does not create an observation group or change stored source UV.
        reference=(candidate.review.get('human_source_ref') or {}).get('reference',{})
        pointer=reference.get('json_pointer','')
        match=re.fullmatch(r'(.*/segments/\d+)/uv/\d+',pointer)
        if match:
            return 'annotation_query/'+json.dumps([reference.get('file',reference.get('path')),
                reference.get('sha256'),match[1]],separators=(',',':'))
        return candidate.candidate_id
    def source_order(candidate, ordinal):
        reference=(candidate.review.get('human_source_ref') or {}).get('reference') or candidate.source_ref.get('support_ref',{})
        pointer=reference.get('json_pointer','')
        tail=pointer.rsplit('/',1)[-1]
        if tail.isdigit():return int(tail)
        return reference.get('jsonl_line_0based',candidate.source_ref.get('source_point_index',ordinal))
    for (identity,_view),rows in groups.items():
        uv=np.asarray([c.actual_uv if c.actual_uv is not None else c.original_uv for c in rows],float)
        good=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        rows=[c for c,k in zip(rows,good) if k];uv=uv[good]
        if len(uv)<2:continue
        # Keep per-source runs. Do not connect unrelated dense row ordering.
        order=sorted(range(len(rows)),key=lambda k:(query_run(rows[k]),source_order(rows[k],k),k))
        rows=[rows[k] for k in order];uv=uv[order]
        ids={};segments=[]
        for c in rows:
            key=query_run(c);ids.setdefault(key,len(ids));segments.append(ids[key])
        normal=np.asarray([c.normal if c.normal is not None else [0.,0.] for c in rows])
        known=np.linalg.norm(normal,axis=1)>0
        if not known.all():
            for group in set(segments):
                loc=np.flatnonzero(np.asarray(segments)==group)
                if len(loc)<2:continue
                tangent=np.gradient(uv[loc],axis=0);n=np.column_stack((-tangent[:,1],tangent[:,0]))
                n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
                normal[loc[~known[loc]]]=n[~known[loc]]
        g=dict(id=identity,candidate_kind='saved_source_readout',points_uv=uv,normals=normal,
            segment_ids=segments,source_binding=binding,origin='source_annotation_search_hint',
            identity_confirmed=False,source_candidates=rows,search_radius_px=radius,
            source_group_id=f"saved/{binding['shot_id']}/{binding['source_pts']}/{identity}",
            original_dependence_groups_per_point=[c.source_group_id for c in rows])
        g['query_connectivity_source']='original_dependence_group_or_authenticated_annotation_segment'
        g['query_connectivity_changes_original_observation_groups']=False
        board=[c.evidence['source_board_material_frame'] for c in rows if c.actual_uv is not None
            and c.evidence.get('current_semantics_match') and c.evidence.get('source_board_material_frame')]
        if board:
            hashes={p['source_frame']['decoded_native_BGR_sha256'] for p in board}
            if len(hashes)!=1:raise ValueError('One board query refers to different original images')
            g.update(source_board_view='front_face',source_board_view_evidence=dict(
                kind='authenticated_original_board_material_frame',source_binding=binding,
                decoded_native_BGR_sha256=next(iter(hashes)),source_frame_evidence=board[0],
                human_pixel_acceptance=False))
        extents=np.full(len(rows),float(radius))
        for k,c in enumerate(rows):
            if c.actual_uv is None or c.evidence.get('current_semantics_match') is not True:continue
            pixel=c.evidence.get('pixel_support') or c.evidence.get('precise_source_support') or {}
            quantiles=np.asarray(pixel.get('side_interface_quartiles_uv',[]),float)
            if quantiles.shape==(2,3,2) and np.isfinite(quantiles).all() and np.linalg.norm(normal[k])>0:
                offsets=(quantiles-np.asarray(c.actual_uv))@normal[k]
                # Recorded gradient support can be wider than the default
                # query. Include its measured blur plus the existing reader's
                # ten-pixel paint shoulder; do not relax any readout gate.
                extents[k]=max(extents[k],float(np.abs(offsets).max()+np.ptp(offsets,axis=1).max()+10.))
        g['material_profile_half_extent_px_per_point']=extents
        g['material_profile_extent_source']='original_same_exposure_interface_quartiles_and_reader_shoulder'
        materials=[]
        for c in rows:
            if c.actual_uv is None or c.evidence.get('current_semantics_match') is not True:continue
            pixel=c.evidence.get('pixel_support') or c.evidence.get('precise_source_support') or {}
            signal=pixel.get('material_signal',c.evidence.get('material_signal'))
            transition=pixel.get('material_transition',c.evidence.get('material_transition'))
            if signal in ('dark','red','blue') or transition in ('dark_to_light','light_to_dark'):
                materials.append((c,signal,transition))
        signals={s for _,s,_ in materials if s in ('dark','red','blue')}
        transitions={t for _,_,t in materials if t in ('dark_to_light','light_to_dark')}
        if len(signals)==1:g['signal']={'dark':'dark_band','red':'red_excess','blue':'blue_excess'}[next(iter(signals))]
        if len(transitions)==1:g['material_outward_transition']=next(iter(transitions))
        if materials:
            g['source_material_definition_evidence']=dict(source_binding=binding,
                same_exposure_saved_RGB_profiles=True,definition_from_native_style=False,
                source_records=[dict(id=c.candidate_id,source_ref=c.source_ref) for c,_,_ in materials],
                conflicting_signals=len(signals)>1,conflicting_transitions=len(transitions)>1)
        result.append(g)
    return geometry.bind(result)[0]


def qualify_measurement(context, reading, association):
    """A named query needs an actual, unmasked, current-image readout to qualify."""
    binding=context['binding'];identity=reading.get('native_path_id')
    uv=np.asarray(_list(reading.get('actual_source_uv',[])),float).reshape(-1,2)
    idx=np.asarray(_list(reading.get('source_indices',[])),int)
    evidence=reading.get('support',{}).get('actual_readout_evidence') or {}
    precise=set(evidence.get('precise_source_indices',[]))
    guide=reading.get('guide',{});normals=np.asarray(_list(guide.get('normals',[])),float)
    original_groups=reading.get('original_dependence_groups_per_point')
    if original_groups is not None and len(original_groups)!=len(_list(guide.get('uv',[]))):
        raise ValueError('Original dependence groups must align with the source query')
    out=[]
    from urllib.parse import quote
    query_id=reading.get('query_id')
    query_token='/query/'+quote(str(query_id),safe='') if query_id is not None else ''
    for i,point in enumerate(uv):
        j=int(idx[i]);confirmed=bool(association.get('identity_confirmed'));authority=association.get('identity_authority')
        graph_identity=evidence.get('identity_by_source_index',{}).get(str(j))
        identity_gap=None
        if graph_identity is not None and not confirmed:
            from ..vision_material_reader import material_readout_identity
            proof=material_readout_identity(reading,binding,context['rgb_sha256'],j,point)
            if proof is not None:confirmed=True;authority=proof
            else:identity_gap='material_identity_proof_does_not_match_this_exact_readout'
        c=SourceCandidate(
            f"rgb/{binding['source_sha256']}/{binding['source_pts']}/{identity}{query_token}/{j}",
            binding['shot_id'],binding['source_pts'],dict(source_binding=binding,decoded_RGB_sha256=context['rgb_sha256'],query_id=query_id),
            point.tolist(),reading.get('expected_id',identity),actual_uv=point.tolist(),
            normal=normals[j].tolist() if len(normals)>j else None,
            localization='measured',identity='confirmed' if confirmed else 'unresolved',
            native_path_id=identity,measurement=reading.get('native_measurement'),
            source_group_id=(original_groups[j] if original_groups is not None and original_groups[j]
                else reading.get('source_group_id')),origin='automatic',
            review=dict(status='machine_result_pending_scoped_visual_review',human_scope_extended=False),
            evidence=dict(method=reading.get('support',{}).get('method'),profile_index=j,
                precise=j in precise,identity_authority=authority,identity_gap=identity_gap,
                original_dependence_groups_per_point=reading.get('original_dependence_groups_per_point')))
        out.append(c)
    return out


def feature_registry(geometry):
    from ..domain.print_assets import resolve_print_curve
    bindings={}
    for ref,assets in geometry.scene_assets:
        for identity in ref.get('instance_ids',[ref.get('instance_id')]):
            for row in geometry.current_paths.values():
                if row.get('instance_id')==identity:
                    bindings[row['id']]=resolve_print_curve(assets,identity,row['template_path_id'])
    records=[]
    for row in geometry.current_paths.values():
        record=dict(id=row['id'],path_id=row['id'],label=row['label'],entity_id=row['entity_id'],
            kind='finite_curve',surface='ice' if row['layer']=='ice' else 'board',identity='confirmed',
            measurement=row.get('measurement','contour'),has_station_parameter=True)
        if row['id'] in bindings:
            record['print_binding']=bindings[row['id']].as_record()
            record['label']=bindings[row['id']].path_id
        if row.get('native_kind')==7:
            record.update(native_kind=7,station_domain=row['station_domain'])
        else:record['support']=row.get('support') or [0.,1.]
        records.append(record)
    return records,bindings


def select_source_observations(measurements, by_feature, noise, purpose, *, image_size=(1920, 1080)):
    """Select original correlated evidence without independently counting rereads."""
    width,height=image_size
    if type(width) is not int or type(height) is not int or width<=0 or height<=0:
        raise ValueError('Source image dimensions must be positive integers')
    chosen={};eligibility=[];eligible={};explicit_dependence=set()
    explicit_groups=list((noise or {}).get('groups',[]))
    explicit_members={oid for g in explicit_groups for oid in g['observation_ids']}
    for c in measurements:
        if c.candidate_id in explicit_members and c.source_group_id:
            explicit_dependence.add((c.shot_id,c.native_pts,c.source_group_id))
        reason=None
        if c.withdrawal:reason='withdrawn_source'
        elif c.localization!='measured' or c.actual_uv is None:reason='no_direct_source_measurement'
        elif c.identity!='confirmed':reason='unresolved_source_identity'
        elif c.origin=='legacy_import' and c.evidence.get('current_semantics_match') is not True:reason='original_measurement_needs_current_native_material_reread'
        elif c.native_path_id not in by_feature:reason='no_current_native_path'
        elif c.normal is None and c.candidate_id not in explicit_members and not c.evidence.get('source_junction_normals'):reason='no_calibrated_W_or_actual_profile_normal'
        elif by_feature[c.native_path_id]['kind']=='finite_curve' and c.evidence.get('station_initial') is None:reason='no_compatible_navigation_station'
        elif not (0<=c.actual_uv[0]<width and 0<=c.actual_uv[1]<height):reason='outside_original_image'
        if reason:eligibility.append(dict(id=c.candidate_id,reason=reason));continue
        eligible[c.candidate_id]=c
        group=c.source_group_id or c.candidate_id
        key=(c.shot_id,c.native_pts,group)
        # Selection uses source precision evidence, never proximity to the model.
        pixel=c.evidence.get('pixel_support',{})
        score=(bool(c.review.get('human_source_ref')),bool(c.evidence.get('precise')),
            -float(pixel.get('interface_IQR_px',float('inf'))))
        old=chosen.get(key)
        if old is None or score>old[0]:chosen[key]=(score,c)
    complete=[];weighted=set()
    for g in explicit_groups:
        ids=list(g['observation_ids'])
        if len(ids)!=len(set(ids)) or weighted.intersection(ids):raise ValueError('Duplicated original W member')
        if not set(ids)<=eligible.keys():
            eligibility.append(dict(group_id=g['id'],reason='original_correlated_group_not_complete',
                missing_or_unqualified_members=sorted(set(ids)-eligible.keys()),original_group_preserved=True))
            continue
        if purpose not in g['allowed_use']:raise ValueError('Original W does not permit requested use')
        w=np.asarray(g['weights'],dtype='<f8').reshape(-1)
        if w.size!=int(g['rank'])*2*len(ids) or not np.isfinite(w).all():raise ValueError('Original W shape differs')
        if hashlib.sha256(w.tobytes()).hexdigest()!=g['matrix_sha256']:raise ValueError('Original W hash differs')
        complete.append(dict(g,weights=w));weighted.update(ids)
    rows=[value[1] for key,value in chosen.items() if value[1].candidate_id not in explicit_members
        and key not in explicit_dependence]
    rows.extend(eligible[oid] for oid in sorted(weighted))
    rows.sort(key=lambda c:(c.shot_id,c.native_pts,c.native_path_id,c.candidate_id))
    return rows,complete,eligibility


def export_input(measurements, features, source_index, noise, purpose, out):
    """Export current input; retain unselected facts and original groups in its ledger.

    An existing W is consumed only with its complete member IDs. Otherwise one
    real profile per original dependence group gets declared normal-only W.
    """
    from .dataset import write_json,write_array,digest
    from ..domain.contracts import Source,Feature,Observation,NoiseGroup,validate_dataset
    from ..domain.scene_assets import snapshot_scene
    from ..domain.runtime_rigs import snapshot_runtime
    if purpose not in ('camera_navigation','geometry_fit'):raise ValueError('Explicit dataset purpose required')
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    geometry=source_index['geometry'];job=source_index['job']
    registry,bindings=features
    by_feature={x['id']:x for x in registry}
    rows,complete,eligibility=select_source_observations(measurements,by_feature,noise,purpose,
        image_size=(job['source']['width'],job['source']['height']))
    if purpose!='camera_navigation' and rows and not noise:
        raise ValueError('Engineering navigation weights cannot enter final geometry fitting')
    # This task's source-only selected rows expressly carry no independent W.
    # Engineering rank-one W is allowed only for camera navigation.
    source=job['source'];clock=job['source_clock'];pts=sorted(set(source_index['decoded_pts']))
    src=Source(source['id'],source['sha256'],source['width'],source['height'],clock['time_base_num'],clock['time_base_den'],frozenset(pts))
    fobjects={f['id']:Feature(f['id'],f['kind'],f['surface'],f['identity'],f['measurement'],
        tuple(f.get('parents',())),f.get('branch'),f['has_station_parameter'],f['label'],bindings.get(f['id']),f.get('junction_relation')) for f in registry}
    navigation=Path(source_index['navigation_dir'])
    snapshot_scene(geometry.scene_path,out)
    rigs,shots,_=snapshot_runtime(navigation/'rigs.json',navigation/'shots.json',out)
    shot_lookup={s['id']:i for i,s in enumerate(shots['shots'])};feature_lookup={f['id']:i for i,f in enumerate(registry)}
    groups=[];weights=[];members=[];objects=[];metadata=[];states=[]
    row_lookup={c.candidate_id:i for i,c in enumerate(rows)};group_for={}
    scale=3.0 # Declared engineering source-pixel scale, not calibrated covariance.
    def add_group(record,ids,w):
        w=np.asarray(w,dtype='<f8').reshape(-1)
        g=dict(record,member_start=len(members),member_count=len(ids),weight_start=len(weights),
            matrix_sha256=hashlib.sha256(w.tobytes()).hexdigest(),model='square_root_information')
        for oid in ids:group_for[oid]=len(groups)
        groups.append(g);members.extend(row_lookup[oid] for oid in ids);weights.extend(w)
    for g in complete:
        add_group({k:v for k,v in g.items() if k not in ('weights','observation_ids')},g['observation_ids'],g['weights'])
    for c in rows:
        if c.candidate_id in group_for:continue
        if purpose!='camera_navigation':raise ValueError('Uncalibrated profile cannot enter final geometry fitting')
        normals=np.asarray(c.evidence.get('source_junction_normals',[c.normal]),float)
        if (normals.ndim!=2 or normals.shape[1]!=2 or not np.isfinite(normals).all() or
            np.any(np.linalg.norm(normals,axis=1)==0)):
            raise ValueError('Invalid actual source profile normal: '+c.candidate_id)
        normals=normals/np.linalg.norm(normals,axis=1,keepdims=True)
        gid='navigation/'+hashlib.sha256(c.candidate_id.encode()).hexdigest()[:24]
        add_group(dict(id=gid,rank=len(normals),allowed_use=['camera_navigation'],
            weight_semantics='engineering_navigation_weight',scale_unit='native_source_px',calibrated=False,
            navigation_scale_px=scale,scale_source_ref='configured_unit_scale_3_native_source_px',
            original_dependence_group_id=c.source_group_id,selection='one_actual_profile_no_duplicate_display_samples'),
            [c.candidate_id],normals/scale)
    for i,c in enumerate(rows):
        gid=groups[group_for[c.candidate_id]]['id']
        quality='human_confirmed' if c.review.get('human_source_ref') else 'machine_verified'
        obj=Observation(c.candidate_id,src.source_id,src.sha256,c.native_pts,tuple(c.actual_uv),c.native_path_id,
            by_feature[c.native_path_id]['measurement'],c.origin,'measured',quality,'provenance/'+c.candidate_id,
            noise_group_id=gid)
        objects.append(obj)
        metadata.append(dict(id=obj.observation_id,feature_id=obj.feature_id,measurement=obj.measurement,
            source_id=src.source_id,source_sha256=src.sha256,time_base=f'{src.time_base_num}/{src.time_base_den}',
            pts=c.native_pts,shot_id=c.shot_id,uv=c.actual_uv,origin=c.origin,source_evidence='measured',quality=quality,
            active=True,noise_group_id=gid,provenance_id=obj.provenance_id,provenance=vars(c)))
        if by_feature[c.native_path_id]['kind']=='finite_curve':
            states.append(dict(observation_id=c.candidate_id,local_id='station/'+c.candidate_id,
                path_id=c.native_path_id,value=float(c.evidence['station_initial']),
                origin='nearest_current_native_path_initialization_for_already_identified_source_UV',
                source_UV_modified=False))
    ng={g['id']:NoiseGroup(g['id'],tuple(rows[members[j]].candidate_id for j in
        range(g['member_start'],g['member_start']+g['member_count'])),'arrays/noise_weights.npy',
        g['matrix_sha256'],'square_root_information') for g in groups}
    validation=validate_dataset({src.source_id:src},fobjects,tuple(objects),ng)
    n=len(rows)
    periodic=np.array([by_feature[c.native_path_id].get('native_kind')==7 for c in rows],bool)
    curves=np.array([by_feature[c.native_path_id]['kind']=='finite_curve' for c in rows],bool)
    arrays=dict(uv=np.array([c.actual_uv for c in rows],dtype='<f8').reshape(n,2),pts=np.array([c.native_pts for c in rows],dtype='<i8'),
        source_index=np.zeros(n,dtype='<i4'),shot_index=np.array([shot_lookup[c.shot_id] for c in rows],dtype='<i4'),
        feature_index=np.array([feature_lookup[c.native_path_id] for c in rows],dtype='<i4'),kind=curves.astype('u1'),
        active=np.ones(n,dtype='u1'),split=np.zeros(n,dtype='u1'),noise_group_index=np.array([group_for[c.candidate_id] for c in rows],dtype='<i4'),
        covariance=np.zeros((n,3),dtype='<f8'),support=np.array([[0.,0.] if p or not curve else by_feature[c.native_path_id]['support'] for c,p,curve in zip(rows,periodic,curves)],dtype='<f8').reshape(n,2),
        support_kind=periodic.astype('u1'),local_index=np.where(curves,np.arange(n),-1).astype('<i8'),
        observation_ids=np.asarray([c.candidate_id.encode() for c in rows],dtype='S'+str(max([len(c.candidate_id.encode()) for c in rows]+[1]))))
    refs={key:write_array(out,key,value) for key,value in arrays.items()}
    with (out/'observations.jsonl').open('w',encoding='utf-8') as f:
        for record in metadata:f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    write_json(out/'observation_arrays.json',dict(arrays=refs,metadata_ref='observations.jsonl'))
    write_json(out/'features.json',dict(features=registry))
    write_json(out/'sources.json',dict(sources=[dict(id=src.source_id,sha256=src.sha256,width=src.width,height=src.height,
        time_base_num=src.time_base_num,time_base_den=src.time_base_den,registered_path=source['registered_path'],
        pts_ref=write_array(out,'decoded_source_pts',pts,'<i8'))]))
    write_json(out/'noise_groups.json',dict(groups=groups,members_ref=write_array(out,'noise_members',members,'<i8'),
        weights_ref=write_array(out,'noise_weights',weights,'<f8')))
    write_json(out/'provenance.json',dict(purpose=purpose,source_index_ref=source_index['index_ref'],
        candidate_ledger_ref=source_index['candidate_ledger_ref'],geometry_source_refs=geometry.refs,
        original_source_UV_PTS_unchanged=True,human_acceptance_not_extended=True))
    write_json(out/'eligibility.json',dict(not_exported=eligibility,all_source_candidates_retained_in=source_index['candidate_ledger_ref']))
    write_json(out/'navigation_local_state.json',dict(schema='rinkspace.navigation_local_state.v1',rows=states))
    revision=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in refs.items()},sort_keys=True).encode()).hexdigest()
    write_json(out/'input.json',dict(schema='rinkspace.input.v1',dataset_id='okada_mei_sp/source_programme',revision=revision,
        scene_ref='scene.json',rigs_ref='rigs.json',shots_ref='shots.json',sources_ref='sources.json',
        observations_ref='observation_arrays.json',features_ref='features.json',noise_groups_ref='noise_groups.json',provenance_ref='provenance.json',
        model_configuration=dict(measurement_policy_id=geometry.scene['measurement_policy_id'],allowed_use=[purpose],
            engineering_weights_are_calibrated_covariance=False),validation=validation))
    return out/'input.json'
