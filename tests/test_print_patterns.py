import cv2
import numpy as np
import pytest
from rinkspace.session.print_patterns import PrintPattern, sample, project, registration_geometry_check


def reference():
    image=np.full((80,240,3),[150,60,140],np.uint8)
    paths=[]
    for name,x in [('A',20),('O',90),('M',160)]:
        box=np.array([[x,20],[x+40,20],[x+40,60],[x,60],[x,20]],float)
        cv2.rectangle(image,(x,20),(x+40,60),(230,230,230),-1)
        paths.append(dict(path_id='Ad1.print.'+name,pixels=box.tolist()))
    hole=np.array([[101,31],[119,31],[119,49],[101,49],[101,31]],float)
    cv2.rectangle(image,(101,31),(119,49),(150,60,140),-1)
    paths.append(dict(path_id='Ad1.print.O_inner',pixels=hole.tolist()))
    return PrintPattern(dict(paths=paths),image),image


def test_one_merged_white_band_cannot_confer_three_letter_identities():
    pattern,image=reference()
    cv2.rectangle(image,(20,20),(200,60),(230,230,230),-1)
    resolved,evidence=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert not resolved
    assert len(next(iter(evidence['merged_components'].values())))==3


def test_inner_contour_needs_a_current_visible_hole():
    pattern,image=reference()
    resolved,_=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert 'Ad1.print.O_inner' in resolved
    cv2.rectangle(image,(101,31),(119,49),(230,230,230),-1)
    resolved,_=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert 'Ad1.print.O' in resolved and 'Ad1.print.O_inner' not in resolved


def test_occluded_letter_cannot_supply_a_component():
    pattern,image=reference();mask=np.zeros(image.shape[:2],np.uint8);mask[:,85:136]=255
    resolved,_=pattern.resolved_glyphs(image,mask,np.eye(3))
    assert 'Ad1.print.A' in resolved and 'Ad1.print.O' not in resolved


def test_local_duplicate_fragments_are_not_a_complete_semantic_reference():
    pattern,image=reference()
    with pytest.raises(ValueError,match='unique path identities'):
        PrintPattern(dict(paths=[pattern.paths[0],pattern.paths[0]]),image)


def test_foreground_cut_does_not_resolve_a_merged_letter():
    pattern,image=reference()
    cv2.rectangle(image,(20,20),(200,60),(230,230,230),-1)
    mask=np.zeros(image.shape[:2],np.uint8);mask[:,65:155]=255
    resolved,evidence=pattern.resolved_glyphs(image,mask,np.eye(3))
    assert not resolved
    assert evidence['path_checks']['Ad1.print.M']=='current_component_cut_by_foreground'


def test_reference_topology_supports_split_arcs_and_non_suffix_hole_names():
    old,image=reference()
    paths=[dict(p)for p in old.paths]
    outer=np.array(paths[0]['pixels'])
    paths[0]=dict(path_id='Ad1.logo.arc_left',pixels=outer[:3].tolist())
    paths.append(dict(path_id='Ad1.logo.arc_right',pixels=outer[2:].tolist()))
    paths[1]['path_id']='Ad1.word.O_part00'
    paths[3]['path_id']='Ad1.word.O_part01'
    pattern=PrintPattern(dict(paths=paths),image)
    resolved,_=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert {'Ad1.logo.arc_left','Ad1.logo.arc_right','Ad1.word.O_part00','Ad1.word.O_part01'}<=resolved
    cv2.rectangle(image,(101,31),(119,49),(230,230,230),-1)
    resolved,_=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert 'Ad1.word.O_part00' in resolved and 'Ad1.word.O_part01' not in resolved


def test_dark_print_has_the_same_material_topology_and_missing_hole_guard():
    old,image=reference();image=255-image
    pattern=PrintPattern(dict(paths=old.paths,ink_polarity='dark'),image)
    resolved,_=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert len(resolved)==4
    cv2.rectangle(image,(101,31),(119,49),(25,25,25),-1)
    resolved,_=pattern.resolved_glyphs(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert 'Ad1.print.O' in resolved and 'Ad1.print.O_inner' not in resolved


def test_observed_blur_can_join_narrow_reference_parts_without_dropping_all_edges():
    image=np.zeros((80,180,3),np.uint8);paths=[]
    for i,x in enumerate((20,64,108)):
        cv2.rectangle(image,(x,20),(x+40,60),(230,230,230),-1)
        paths.append(dict(path_id=f'Ad1.print.{i}',semantic_glyph_id='one_split_print_mark',
            pixels=[[x,20],[x+40,20],[x+40,60],[x,60],[x,20]]))
    pattern=PrintPattern(dict(paths=paths),image)
    actual=cv2.blur(image,(9,1))
    resolved,_=pattern.resolved_glyphs(actual,np.zeros(actual.shape[:2],np.uint8),np.eye(3),horizontal_blur_px=9)
    assert resolved=={p['path_id']for p in paths}


def test_blurred_print_exposure_change_preserves_real_components_and_hole_guard():
    old,image=reference()
    # A single photographic exposure/contrast change is shared by the print.
    actual=np.rint(cv2.blur(image.astype(float),(9,1))*.65+42).astype(np.uint8)
    resolved,evidence=old.resolved_glyphs(actual,np.zeros(actual.shape[:2],np.uint8),
        np.eye(3),horizontal_blur_px=9)
    assert {'Ad1.print.A','Ad1.print.O','Ad1.print.M','Ad1.print.O_inner'}<=resolved
    cv2.rectangle(actual,(101,31),(119,49),(192,192,192),-1)
    resolved,_=old.resolved_glyphs(actual,np.zeros(actual.shape[:2],np.uint8),
        np.eye(3),horizontal_blur_px=9)
    assert 'Ad1.print.O_inner' not in resolved


def test_foreground_can_split_one_real_letter_into_two_visible_components():
    pattern,image=reference();mask=np.zeros(image.shape[:2],np.uint8)
    mask[:,107:113]=255
    resolved,evidence=pattern.resolved_glyphs(image,mask,np.eye(3))
    assert 'Ad1.print.O' in resolved
    assert evidence['path_checks']['Ad1.print.O']=='separate_visible_current_ink_components'
    group=next(k for k,names in evidence['reference_component_paths'].items() if 'Ad1.print.O' in names)
    visible=evidence['component_visibility_checks'][group]
    assert len(visible['current_components'])==2
    assert visible['reference_cut_by_foreground'] and visible['visible_shape_iou']==1.


def test_partial_visible_hole_still_rejects_a_filled_current_hole():
    pattern,image=reference();mask=np.zeros(image.shape[:2],np.uint8)
    mask[15:35,85:111]=255
    resolved,evidence=pattern.resolved_glyphs(image,mask,np.eye(3))
    assert {'Ad1.print.O','Ad1.print.O_inner'}<=resolved
    assert evidence['path_checks']['Ad1.print.O_inner']=='current_visible_hole_separated'
    cv2.rectangle(image,(101,31),(119,49),(230,230,230),-1)
    resolved,evidence=pattern.resolved_glyphs(image,mask,np.eye(3))
    assert 'Ad1.print.O' in resolved and 'Ad1.print.O_inner' not in resolved
    assert evidence['path_checks']['Ad1.print.O_inner']=='current_hole_unresolved'


def test_partial_visibility_does_not_admit_a_visible_merged_letter_band():
    pattern,image=reference();mask=np.zeros(image.shape[:2],np.uint8)
    cv2.rectangle(image,(20,20),(200,60),(230,230,230),-1)
    mask[15:35,15:35]=255
    resolved,evidence=pattern.resolved_glyphs(image,mask,np.eye(3))
    assert not resolved and evidence['merged_components']
    assert evidence['path_checks']['Ad1.print.O']=='current_ink_merges_distinct_letters'


def test_partial_visible_arcs_use_actual_profiles_without_foreground_output():
    pattern,image=reference();mask=np.zeros(image.shape[:2],np.uint8)
    actual=cv2.GaussianBlur(image,(0,0),.8)
    actual=cv2.warpAffine(actual,np.array([[1.,0.,.65],[0.,1.,.35]]),
        (image.shape[1],image.shape[0]),borderMode=cv2.BORDER_REPLICATE)
    mask[15:35,85:111]=255;actual[mask>0]=(20,240,10)
    registration=dict(status='source_pattern_registered',H=np.eye(3).tolist(),shared_horizontal_blur_px=1)
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',
        shot_id='partial_visible_test',native_hw=list(image.shape[:2]))
    rows,evidence=pattern.read_paths(actual,mask,registration,binding)
    measured={r['path_id'].split('::')[1] for r in rows}
    assert {'Ad1.print.O','Ad1.print.O_inner'}<=measured
    assert evidence['accepted']>0
    expanded=cv2.dilate(mask,np.ones((7,7),np.uint8))
    for row in rows:
        q=np.array(row['search_uv']);n=np.array(row['normal']);offsets=row['actual_normal_quantiles']
        support=q+np.linspace(offsets[0]-1,offsets[2]+1,33)[:,None]*n
        assert not np.any(sample(expanded,support)>0)
        assert row['current_source_RGB_measured'] and not row['model_UV_used_as_measurements']
    assert any(np.linalg.norm(np.array(r['uv'])-r['search_uv'])>.1 for r in rows)


def test_positive_four_corner_area_does_not_hide_a_horizon_and_reversed_print():
    H=np.array([[1.,0,0],[0,1.,0],[-.015,0,1.]])
    roi=np.array([[0.,0.],[100.,100.]])
    corners=np.array([[0,0],[100,0],[100,100],[0,100]],float)
    area=cv2.contourArea(project(H,corners).astype(np.float32),oriented=True)
    assert .25*10000<area<4*10000  # The old gate accepts this impossible warp.
    points=np.array([[80,20],[95,20],[95,40],[80,40]],float)
    check=registration_geometry_check(H,np.eye(3),points,warp_roi=roi)
    assert not check['valid']
    assert {'warp_ROI_horizon','shape_orientation_not_preserved'}<=set(check['reasons'])
    assert not check['registered_shape']['horizon_crosses_domain']
    assert check['relative_shape']['local_determinant_range'][1]<0


def test_finite_orientation_preserving_perspective_and_anisotropy_are_distinct():
    points=np.array([[20,20],[60,20],[60,40],[20,40]],float);roi=[[0,0],[100,80]]
    H=np.array([[1.1,.1,3],[.02,.9,2],[.0001,.0002,1.]])
    check=registration_geometry_check(H,np.eye(3),points,warp_roi=roi)
    assert check['valid'] and check['relative_shape']['local_determinant_range'][0]>0
    # No arbitrary anisotropy cutoff is smuggled into the no-fold proof.
    check=registration_geometry_check(np.diag([.01,50.,1.]),np.eye(3),points,warp_roi=roi)
    assert check['valid'] and check['relative_shape']['local_anisotropy_max']>1000


def test_every_registration_blur_candidate_must_pass_finite_domain_gate(monkeypatch):
    pattern,image=reference();H=np.array([[1.,0,0],[0,1.,0],[-.006,0,1.]],np.float32)
    calls=[]
    def impossible_ecc(*args):
        calls.append(1);return .95,H.copy()
    monkeypatch.setattr(cv2,'findTransformECC',impossible_ecc)
    result=pattern.register(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert result['status']=='registration_failed' and len(calls)==8
    assert len(result['geometry_attempts'])==8
    assert all(not c['valid'] and 'registered_shape_horizon' in c['reasons']
        for c in result['geometry_attempts'])


def test_finite_registered_print_keeps_its_real_material_support(monkeypatch):
    pattern,image=reference()
    monkeypatch.setattr(cv2,'findTransformECC',lambda *args:(.95,np.eye(3,dtype=np.float32)))
    result=pattern.register(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert result['status']=='source_pattern_registered'
    assert result['registration_geometry']['valid']
    assert len(result['geometry_attempts'])==8 and all(c['valid'] for c in result['geometry_attempts'])


def test_blur_cannot_merge_distinct_clean_glyphs_into_shared_identity():
    image=np.zeros((80,180,3),np.uint8);paths=[]
    for i,x in enumerate((20,64,108)):
        cv2.rectangle(image,(x,20),(x+40,60),(230,230,230),-1)
        paths.append(dict(path_id=f'Ad1.letter.{i}',pixels=[[x,20],[x+40,20],[x+40,60],[x,60],[x,20]]))
    pattern=PrintPattern(dict(paths=paths),image);actual=cv2.blur(image,(9,1))
    resolved,evidence=pattern.resolved_glyphs(actual,np.zeros(image.shape[:2],np.uint8),np.eye(3),horizontal_blur_px=9)
    assert not resolved and len(evidence['reference_component_paths'])==3
    assert len(next(iter(evidence['merged_components'].values())))==3


def test_holes_do_not_inflate_identity_count_and_one_complete_badge_can_register(monkeypatch):
    old,image=reference();paths=[old.paths[1],old.paths[3]]
    pattern=PrintPattern(dict(paths=paths),image)
    monkeypatch.setattr(cv2,'findTransformECC',lambda *args:(.95,np.eye(3,dtype=np.float32)))
    result=pattern.register(image,np.zeros(image.shape[:2],np.uint8),np.eye(3))
    assert result['status']=='source_pattern_registered'
    assert len(result['supported_letters'])==1 and len(result['supported_paths'])==2
    assert result['semantic_identity_check']['reference_material_groups']==1
    assert result['semantic_identity_check']['holes_are_not_additional_glyphs']


def test_bound_source_contour_topology_survives_reference_RGB_bridges(tmp_path):
    old,target=reference();source=target.copy();paths=[dict(p) for p in old.paths]
    cv2.rectangle(source,(60,39),(160,41),(230,230,230),-1)
    source_sha='b'*64;asset=tmp_path/'source_contours.json';declared=[]
    for p in paths:
        hole=p['path_id'].endswith('_inner')
        declared.append(dict(id=p['path_id'],closed=True,
            role='letter_hole_boundary' if hole else 'printed_ink_external_boundary',
            parent_contour_key='Ad1.print.O' if hole else None,
            shape_provenance=dict(source_native_boundary=p['pixels'][:-1],source_frame=dict(sha256=source_sha))))
        p['proof']=dict(kind='current_native_shape_source',asset=str(asset))
    import json
    asset.write_text(json.dumps(dict(templates=[dict(paths=declared)])),encoding='utf-8')
    pattern=PrintPattern(dict(paths=paths,image_sha256=source_sha),source)
    assert len(set(pattern.reference_path_components.values()))==3
    assert pattern.reference_path_components['Ad1.print.O']==pattern.reference_path_components['Ad1.print.O_inner']
    assert 'Ad1.print.O_inner' in pattern.reference_holes
    resolved,evidence=pattern.resolved_glyphs(target,np.zeros(target.shape[:2],np.uint8),np.eye(3))
    assert {p['path_id'] for p in paths}<=resolved
    assert len(evidence['source_declared_topology_evidence'])==4
    assert all(v['atlas_repeats_explicitly_closed_source_endpoint']
        for v in evidence['source_declared_topology_evidence'].values())


def test_reused_registration_cannot_read_a_now_merged_multiglyph_word():
    pattern,image=reference();cv2.rectangle(image,(20,20),(200,60),(230,230,230),-1)
    registration=dict(status='source_pattern_registered',H=np.eye(3).tolist(),shared_horizontal_blur_px=1)
    rows,evidence=pattern.read_paths(image,np.zeros(image.shape[:2],np.uint8),registration,
        dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='reused_bad_word'))
    assert not rows and evidence['reason']=='current_source_semantic_identity_unresolved'
    assert evidence['structure']['merged_components']


def test_reused_registered_status_cannot_bypass_finite_geometry_check():
    pattern,image=reference()
    registration=dict(status='source_pattern_registered',
        H=[[1.,0.,0.],[0.,1.,0.],[-.006,0.,1.]],shared_horizontal_blur_px=1)
    rows,evidence=pattern.read_paths(image,np.zeros(image.shape[:2],np.uint8),registration,
        dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='reused_fold'))
    assert not rows and evidence['reason']=='current_registration_geometry_invalid'
    assert 'registered_shape_horizon' in evidence['geometry']['reasons']
