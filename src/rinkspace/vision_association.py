"""One dispatcher for retained source, temporal, reference and search candidates."""
from copy import deepcopy
import numpy as np
from .vision_geometry import _same_frame


def associate(context, candidates, native_geometry, feedback=None):
    binding=context['binding'];chosen=[];pending=[];evidence=[]
    for incoming in candidates:
        g=dict(incoming);definition=native_geometry.lookup(g.get('id'))
        if definition is None:
            pending.append(dict(id=g.get('id'),reason='not_a_retained_current_native_path'));continue
        identity=definition['id'];g['id']=identity
        if g.get('source_binding'):_same_frame(binding,g['source_binding'])
        g['source_binding']=binding
        kind=g.get('candidate_kind','existing_query')
        if kind=='native_projection':
            g.update(identity_confirmed=False,identity_authority=None)
            priority=0
        elif kind=='saved_source_readout':
            rows=g.pop('source_candidates',[])
            supported=[c for c in rows if c.localization=='measured' and c.identity=='named_source_candidate'
                and not c.withdrawal and c.native_path_id==identity]
            semantically_bound=[c for c in supported if
                (definition['layer']=='board' and c.evidence.get('current_semantics_match') is True)
                or c.review.get('human_source_ref')
                or c.evidence.get('independent_source_identity_verified') is True]
            # Exact stored source identity is distinct from source pixel approval.
            # Print names need the current retained-template correspondence layer.
            if definition['layer'] in ('ice','board') and semantically_bound:
                g.update(identity_confirmed=True,identity_authority=dict(
                    kind='authenticated_existing_source_material_identity',
                    source_binding=deepcopy(binding),current_native_path_id=identity,
                    source_records=[dict(id=c.candidate_id,source_ref=c.source_ref) for c in semantically_bound],
                    checked_current_measurement_definition=definition.get('measurement'),
                    independent_identity_basis='exact_original_human_named_point_or_typed_physical_board_material',
                    identity_from_projection_alone=False,human_pixel_acceptance=False))
                priority=3
            else:
                g.update(identity_confirmed=False,identity_authority=None);priority=1
                pending.append(dict(id=identity,reason='source_material_retained_identity_needs_correspondence',
                    actual_source_records=len(supported)))
        else:
            authority=g.get('identity_authority')
            confirmed=g.get('identity_confirmed') is True and bool(authority)
            # Existing reference/instance modules own their verified matches. A
            # template ID or projection string by itself is never an authority.
            if definition['layer']=='advertisement' and confirmed:
                from .vision_instances import HUMAN_IDENTITIES, source_correspondence_identity_verified
                a=authority if isinstance(authority,dict) else {}
                physical=definition['instance_id'].rsplit('/',1)[-1]
                human=(a.get('kind') in HUMAN_IDENTITIES and bool(a.get('reference')))
                anchor=a.get('query_source_binding') or binding
                if anchor.get('source_pts')==binding.get('source_pts'):
                    confirmed=human or source_correspondence_identity_verified(a,binding,physical)
                else:
                    # Authenticate the original match at its actual exposure. A
                    # current flow-supported lineage is search identity only;
                    # the current UV still requires a fresh RGB profile read.
                    prior=g.get('hint_previous_source_binding') or {}
                    lineage=(g.get('origin')=='temporal_source_search_hint' and
                        all(anchor.get(k)==binding.get(k) for k in
                            ('source_sha256','time_base','shot_id','installation_id')) and
                        all(prior.get(k)==binding.get(k) for k in
                            ('source_sha256','time_base','shot_id','installation_id')) and
                        int(anchor['source_pts'])<=int(prior.get('source_pts',-1))<int(binding['source_pts']))
                    flow=g.get('temporal_flow_supported')
                    flow_ok=bool(flow is not None and np.asarray(flow.detach().cpu() if hasattr(flow,'detach') else flow).any())
                    confirmed=bool(lineage and flow_ok and
                        (human or source_correspondence_identity_verified(a,anchor,physical)))
                    g['verified_instance_lineage']=confirmed
                    if confirmed:
                        g['identity_lineage_evidence']=dict(anchor_source_binding=deepcopy(anchor),
                            previous_source_binding=deepcopy(prior),current_source_binding=deepcopy(binding),
                            flow_is_measurement=False,current_RGB_reread_required=True)
            g['identity_confirmed']=bool(confirmed)
            priority=4 if confirmed else 1
        points=np.asarray(g.get('points_uv',[])) if not hasattr(g.get('points_uv'),'detach') else g['points_uv'].detach().cpu().numpy()
        if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():
            pending.append(dict(id=identity,reason='no_finite_source_search_support'));continue
        chosen.append((priority,g))
        evidence.append(dict(id=identity,kind=kind,identity_confirmed=g.get('identity_confirmed',False),
            query_only=True,priority=priority,old_source_UV_modified=False))
    from .vision_navigation import _uncovered_query, attach_native_navigation_context
    retained=[]
    for priority,g in sorted(chosen,key=lambda row:-row[0]):
        remainder=_uncovered_query(g,retained,geometry=native_geometry)
        if remainder is None:continue
        remainder.setdefault('query_id',f"{remainder['id']}::associated_run:{len(retained)}")
        retained.append(remainder)
    projections=[g for _,g in chosen if g.get('candidate_kind')=='native_projection']
    retained=attach_native_navigation_context(retained,projections,binding=binding,geometry=native_geometry)
    queries,report=native_geometry.bind(retained)
    return dict(queries=queries,unresolved=pending,evidence=evidence,native_binding=report)


def extend_feedback_scope(context,new_entity_ids):
    from .vision_feedback import begin_read
    existing=set(context.get('feedback_entity_ids',()))
    existing.update(new_entity_ids);context['feedback_entity_ids']=sorted(existing)
    context['feedback']=begin_read(context['binding'],[dict(id=i) for i in sorted(existing)])
    return context['feedback']


def associate_candidates(context, candidates, native_geometry, evidence=None):
    """Shared dispatcher; candidate-specific proof remains on each query."""
    return associate(context,candidates,native_geometry,feedback=evidence)
