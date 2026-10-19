"""One named F3 observation on current, source-qualified overlapping support.

The old row3 name stays in the navigation display. Its model carrier is not an
independent competitor at a qualified measured F3 station. No whole-extent
physical alias equation, endpoint, pixel, or generalized dark-material exemption
is introduced.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from .vision_ice_model_context_identity import SameStateIceModelContextIdentity,source_paint_pixel_accuracy_check
from .vision_programme_ice_review import _candidate_context_excluded

F3='house_connection_F3_R3:segment0'
DISPLAY_ROW3='curling_centerline_row3:segment0'

def _same_navigation_carrier(first,second):
    """Check existing numeric linear carriers at floating arithmetic roundoff.

    This is a representation check for two existing search paths, not a scene
    geometry constraint or a source-distance tolerance. Different parallel
    carriers remain different even when their images are close.
    """
    first,second=np.asarray(first,np.float64),np.asarray(second,np.float64)
    if first.ndim!=2 or second.ndim!=2 or first.shape[1:]!=(3,) or second.shape[1:]!=(3,) or min(len(first),len(second))<2:return False
    if not np.isfinite(first).all() or not np.isfinite(second).all():return False
    vector=first[-1]-first[0];length=np.linalg.norm(vector)
    if length==0:return False
    unit=vector/length;both=np.vstack([first,second]);relative=both-first[0]
    perpendicular=relative-np.outer(relative@unit,unit)
    # Subtraction, norm, normalization, and a 3-term projection incur only
    # floating arithmetic error. The bound scales with the stored coordinates;
    # it is not a user-selected physical/material closeness allowance.
    scale=max(1.,float(np.max(np.abs(both))),length)
    roundoff=np.spacing(scale)*(4*both.shape[1]+4)
    return bool(np.max(np.linalg.norm(perpendicular,axis=1))<=roundoff)

class ConfirmedSingleGraySupportModelContextIdentity(SameStateIceModelContextIdentity):
    def __init__(self,*args,visible_F3_authority,**kwargs):
        super().__init__(*args,**kwargs)
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        path,_=authenticate_source_reference(visible_F3_authority)
        fact=json.loads(Path(path).read_text(encoding='utf8'))
        if (fact.get('schema')!='rinkspace.user-confirmed-visible-source-material-identity.v1'
                or fact.get('confirmed_material_id')!=F3
                or fact.get('identity_navigation_only') is not True
                or fact.get('single_canonical_entity_single_loss_required_on_measured_support') is not True
                or fact['source_binding']['source_sha256']!=self.source_sha256
                or fact['source_binding']['time_base']!=self.time_base):
            raise ValueError('The exact confirmed visible F3 source identity is required')
        self.visible_F3_authority=dict(visible_F3_authority)
        self._single_gray_support_by_candidate={}
        self.latest_single_gray_support_certificate=None

    def _source_context_output_curve_allowed(self,definition):
        return definition['curve_id']!=DISPLAY_ROW3

    def _source_field_order_competitors(self,rgb,measurement,mask,ci,chosen,neighbor_identities):
        result=super()._source_field_order_competitors(rgb,measurement,mask,ci,chosen,neighbor_identities)
        if self.inventory[ci]['curve_id']!=F3:return result
        self._single_gray_support_by_candidate={};self.latest_single_gray_support_certificate=None
        names=[v['curve_id'] for v in self.inventory]
        if DISPLAY_ROW3 not in names:return result
        sibling=names.index(DISPLAY_ROW3)
        if not _same_navigation_carrier(self.paths[ci]['xyz'],self.paths[sibling]['xyz']):return result
        chosen=np.asarray(chosen,np.int64);query=np.asarray(measurement['candidate_query'])
        qcurve=np.asarray(measurement['query_curve']);flags=np.asarray(measurement['candidate_flags'])
        fine=measurement.get('candidate_side_quantiles_uv')
        if fine is None:return result
        fine=np.asarray(fine,np.float64);q50=fine[chosen,:,1]
        complete=np.isfinite(fine[chosen]).all((1,2,3));width=np.linalg.norm(q50[:,1]-q50[:,0],axis=1)
        qualified=complete&(width>0)&((flags[chosen]&3)==0)&(qcurve[query[chosen]]==ci)
        for local,candidate in enumerate(chosen):
            if not qualified[local]:continue
            normal=(q50[local,1]-q50[local,0])/width[local];center=q50[local].mean(0)
            if any(_candidate_context_excluded(mask,p,normal) for p in [center,*q50[local]]):qualified[local]=False
        quality=source_paint_pixel_accuracy_check(rgb,measurement,chosen[qualified],self.inventory[ci],check_path=False)
        qualified&=np.isin(chosen,quality['accepted_candidate_indices'])
        for local,candidate in enumerate(chosen):
            if qualified[local]:
                result[local,neighbor_identities[local]==sibling]=False
                self._single_gray_support_by_candidate[int(candidate)]=dict(
                    original_candidate_index=int(candidate),source_query_index=int(query[candidate]),
                    actual_q50_uv=q50[local].mean(0).tolist(),actual_q50_sides_uv=q50[local].tolist())
        if self._single_gray_support_by_candidate:
            self.latest_single_gray_support_certificate=dict(
                kind='confirmed_single_F3_material_current_qualified_source_support',
                source_binding=dict(measurement['source_binding']),visible_F3_human_authority=self.visible_F3_authority,
                canonical_curve_id=F3,model_display_only_sibling=DISPLAY_ROW3,
                current_qualified_measured_support=list(self._single_gray_support_by_candidate.values()),
                existing_model_linear_navigation_carrier_roundoff_verified=True,
                every_other_Lane_house_or_material_competitor_retained=True,
                source_profile_station_uniqueness_scope_and_own_RGB_two_sides_required=True,
                unknown_physical_extent_and_unseen_endpoints_not_inferred=True,
                whole_curve_physical_alias_equation=False,model_UV_used_as_observation=False,
                source_thresholds_or_pixel_accuracy_gate_relaxed=False)
        return result

    def _source_field_order_identity_evidence(self,measurement,ci,indices):
        if self.inventory[ci]['curve_id']!=F3 or self.latest_single_gray_support_certificate is None:return {}
        evidence=dict(self.latest_single_gray_support_certificate)
        evidence['current_qualified_measured_support']=[self._single_gray_support_by_candidate[int(i)]
            for i in indices if int(i) in self._single_gray_support_by_candidate]
        return dict(source_single_physical_gray_support=evidence)
