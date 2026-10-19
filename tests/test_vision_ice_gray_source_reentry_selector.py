import copy
import numpy as np

from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector
from rinkspace.vision_ice_gray_source_reentry_selector import CurrentSourceGrayReentrySelector
from rinkspace.vision_ice_single_gray_support_identity import ConfirmedSingleGraySupportModelContextIdentity,F3
from test_vision_ice_single_gray_support_identity import context,gray_source,literal


def prepared(cls,tmp_path):
    c=context(ConfirmedSingleGraySupportModelContextIdentity,tmp_path,gray=True)
    selector=cls(c.inventory,source_sha256=c.source_sha256,time_base=c.time_base,model_context_identity=c)
    for pts,shift in [(10,0),(20,2)]:
        rgb,m=gray_source(pts,shift);mask=np.zeros(rgb.shape[:2],bool)
        selected=selector.consume(rgb,m,mask,source_ice_scope_mask=~mask)
    assert selected['rows'] and F3 in selector.tracks
    return selector


def stale(selector):
    selector.tracks[F3]['actual_pts']=10
    selector.tracks[F3]['actual_uv']=np.asarray(selector.tracks[F3]['actual_uv'])+[0.,80.]
    selector.tracks[F3]['navigation_uv']=selector.tracks[F3]['actual_uv']


def test_same_current_tracks_are_bitexact_and_context_advances_once(tmp_path):
    old=prepared(TemporalIceIdentitySelector,tmp_path)
    new=prepared(CurrentSourceGrayReentrySelector,tmp_path)
    rgb,m=gray_source(30,4);mask=np.zeros(rgb.shape[:2],bool)
    a=old.consume(rgb,copy.deepcopy(m),mask,source_ice_scope_mask=~mask)
    b=new.consume(rgb,copy.deepcopy(m),mask,source_ice_scope_mask=~mask)
    assert literal(a)==literal(b)
    assert literal(old.model_context_identity.previous_tracks)==literal(new.model_context_identity.previous_tracks)
    assert not new.latest_current_source_context_reentry
    assert new.model_context_identity.previous_binding['actual_pts']==30


def test_actual_two_image_seed_recovers_stale_gray_without_changing_pixels(tmp_path):
    old=prepared(TemporalIceIdentitySelector,tmp_path);new=prepared(CurrentSourceGrayReentrySelector,tmp_path)
    stale(old);stale(new)
    rgb,m=gray_source(30,4);mask=np.zeros(rgb.shape[:2],bool);before=literal(m)
    a=old.consume(rgb,copy.deepcopy(m),mask,source_ice_scope_mask=~mask)
    b=new.consume(rgb,m,mask,source_ice_scope_mask=~mask)
    assert not a['rows'] and len(b['rows'])==1
    assert literal(m)==before
    assert np.array_equal(b['rows'][0]['uv'],m['candidate_side_quantiles_uv'][:,:,1].mean(1))
    assert b['rows'][0]['temporal_evidence']['previous_actual_pts']==20
    r=new.latest_current_source_context_reentry[0]
    assert r['stale_actual_pts']==10 and r['previous_gray_actual_pts']==20
    assert r['own_two_image_RGB_seed_qualified'] and not r['stale_pixels_used_as_LK_support']


def test_wrong_current_locator_or_pending_seed_cannot_remove_stale_track(tmp_path):
    for corrupt in ('locator','competition'):
        new=prepared(CurrentSourceGrayReentrySelector,tmp_path);stale(new)
        real=new.model_context_identity.consume
        def changed(*args,**kwargs):
            result=real(*args,**kwargs)
            evidence=result['seeds'][0]['source_identity_navigation_evidence']
            if corrupt=='locator':evidence['current_actual_query_indices'][0]+=1
            else:evidence['competing_actual_material_interpretations_rejected']=False
            return result
        new.model_context_identity.consume=changed
        rgb,m=gray_source(30,4);mask=np.zeros(rgb.shape[:2],bool)
        result=new.consume(rgb,m,mask,source_ice_scope_mask=~mask)
        assert not result['rows'] and not new.latest_current_source_context_reentry
        assert new.tracks[F3]['actual_pts']==10
