"""Saved model diagnostics on exact PTS; no GUI, fitting or source inference."""
import json

import numpy as np
import pytest

from rinkspace.viewer3d.blade_review import BladeReview,diagnostic_lines,overlay_primitives,skate_axis_primitives


def review_asset(tmp_path, **extra):
    values=dict(pts=np.array([1000,1256],dtype=np.int64),
        test_blade_uv=np.arange(24,dtype=float).reshape(2,2,3,2),
        test_boot_uv=np.arange(16,dtype=float).reshape(2,2,2,2),
        test_axis_rate_rad_s=np.array([[np.nan,np.nan],[.5,.75]]))
    values.update(extra)
    np.savez_compressed(tmp_path/'projection.npz',**values)
    document=dict(schema='rinkspace.blade-player-review.v1',projection_npz='projection.npz',
        variants={'candidate':'test'},time_base=[1,12800],
        source_visibility=[dict(pts=1256,foot=0,blade_axis_observable=False,
            contact_state='unknown',material_state='unknown',orientation_unknown_reason='接触/材料 unknown')])
    path=tmp_path/'review.json';path.write_text(json.dumps(document),encoding='utf8')
    return BladeReview(path,time_base=(1,12800))


def model_diagnostics():
    return dict(test_pitch_degrees=np.array([[1.,2.],[3.,4.]]),
        test_roll_degrees=np.array([[5.,6.],[7.,8.]]),
        test_runner_min_x_fraction=np.array([[.1,.2],[.3,.4]]),
        test_station_height_m=np.array([[[.01,.02,.03],[.04,.05,.06]],[[.07,.08,.09],[.10,.11,.12]]]),
        test_runner_minus_pick_m=np.array([[-.01,.02],[-.03,.04]]))


def test_old_asset_keeps_original_fields_and_unknown_text(tmp_path):
    review=review_asset(tmp_path);current=review.at('candidate',1256)
    assert set(current['model'])=={'blade_uv','boot_uv','axis_rate_rad_s'}
    assert not any(row.get('evidence_kind')=='model_body_projection'
                   for row in overlay_primitives(current,include_model_body=True))
    lines=diagnostic_lines(current)
    assert len(lines)==4
    assert '原片刀轴：unknown' in lines[1]
    assert '原片刀轴：unknown' in lines[3]
    assert not any('刀纵倾' in line or '原始参考面高度' in line for line in lines)


def test_optional_model_arrays_keep_exact_pts_and_source_unknown(tmp_path):
    values=model_diagnostics();review=review_asset(tmp_path,**values)
    current=review.at('candidate',1256)
    for key,value in values.items():
        np.testing.assert_array_equal(current['model'][key.removeprefix('test_')],value[1])
    missing=review.at('candidate',1255)
    assert missing['model'] is None and missing['source_visibility']==[]
    assert not any('刀纵倾' in line for line in diagnostic_lines(missing))
    source=current['source_visibility'][0]
    assert source['contact_state']==source['material_state']=='unknown'
    lines=diagnostic_lines(current)
    text='\n'.join(lines)
    assert '旧pitch诊断 3.00°' in text and '旧roll诊断 7.00°' in text
    assert '非钢片角度测量' in text
    assert 'runner最低点 0.300（0后1前）' in text
    assert 'rear/mid/front 7.00/8.00/9.00 cm' in text
    assert 'runner-pick相对高度 -3.00 cm' in text
    assert '原始诊断，非真实穿冰判据' in text
    assert '原片刀轴：unknown' in text and '接触/材料 unknown' in text
    # New model diagnostics do not alter native projection primitives.
    old=dict(current,model={key:value for key,value in current['model'].items()
                           if key in ('blade_uv','boot_uv','axis_rate_rad_s')})
    before,after=overlay_primitives(old),overlay_primitives(current)
    assert len(before)==len(after)
    for a,b in zip(before,after):
        assert a.keys()==b.keys()
        for key in a:
            if key=='points':np.testing.assert_array_equal(a[key],b[key])
            else:assert a[key]==b[key]


def test_optional_arrays_are_independent(tmp_path):
    review=review_asset(tmp_path,test_roll_degrees=np.full((2,2),12.))
    text='\n'.join(diagnostic_lines(review.at('candidate',1256)))
    assert '旧roll诊断 12.00°' in text
    assert '刀纵倾' not in text and '原始参考面高度' not in text


@pytest.mark.parametrize('key',model_diagnostics())
def test_bad_optional_shape_is_rejected(tmp_path,key):
    with pytest.raises(ValueError,match=key):
        review_asset(tmp_path,**{key:np.zeros((2,))})


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_nonfinite_optional_model_diagnostic_is_rejected(tmp_path,value):
    with pytest.raises(ValueError,match='test_pitch_degrees'):
        review_asset(tmp_path,test_pitch_degrees=np.full((2,2),value))


def mode_diagnostics():
    result = {('test_' + name):np.broadcast_to(np.eye(3)[axis],(2,2,3)).copy()
              for axis,name in enumerate(('forward_world','lateral_world','up_world'))}
    result['test_ice_normal_world']=result['test_up_world'].copy()
    for name in ('fore_aft_pitch_degrees','edge_bank_degrees','total_lean_degrees','plane_normal_to_ice_normal_degrees'):
        result['test_'+name]=np.array([[1.,2.],[3.,4.]])
    result.update(test_runner_u=np.array([[.1,.9],[.25,.75]]),
                  test_edge_mode=np.full((2,2),'flat_or_unknown'),
                  test_contact_mode=np.full((2,2),'runner_interior'),
                  test_mode_conflict=np.array([[False,False],[True,False]],dtype=bool))
    return result


def test_saved_axes_modes_keep_exact_pts_and_distinguish_source_unknown(tmp_path):
    values=mode_diagnostics();review=review_asset(tmp_path,**values)
    current=review.at('candidate',1256)
    for key,value in values.items():
        np.testing.assert_array_equal(current['model'][key.removeprefix('test_')],value[1])
    assert review.at('candidate',1255)['model'] is None
    assert review.model_array('candidate','ice_normal_world',[1000,1255]) is None
    np.testing.assert_array_equal(review.model_array('candidate','forward_world',[1256,1000]),values['test_forward_world'][[1,0]])
    text='\n'.join(diagnostic_lines(current))
    assert '模型世界向量（非源测量）' in text and 'F [1.000,0.000,0.000]' in text
    assert 'fore-aft pitch 3.00°' in text and 'edge bank 3.00°' in text
    assert 'total lean 3.00°' in text and 'plane-normal / ice-normal 3.00°' in text
    assert 'runner_u 0.250000' in text and 'mode_conflict true' in text
    assert 'edge_mode flat_or_unknown' in text and 'contact_mode runner_interior' in text
    assert '源edge/contact：unknown/unknown' in text


@pytest.mark.parametrize('key',mode_diagnostics())
def test_new_optional_shapes_are_strict(tmp_path,key):
    with pytest.raises(ValueError,match=key):review_asset(tmp_path,**{key:np.zeros((2,))})


@pytest.mark.parametrize('key,value',[
    ('test_forward_world',np.full((2,2,3),'1')),
    ('test_runner_u',np.zeros((2,2),dtype=bool)),
    ('test_edge_mode',np.full((2,2),'unknown',dtype='S8')),
    ('test_contact_mode',np.full((2,2),'')),
    ('test_mode_conflict',np.zeros((2,2),dtype=np.int64)),
    ('test_total_lean_degrees',np.full((2,2),np.nan)),
    ('test_ice_normal_world',np.full((2,2,3),np.inf)),
])
def test_new_optional_dtype_and_finite_validation(tmp_path,key,value):
    with pytest.raises(ValueError,match=key):review_asset(tmp_path,**{key:value})


def test_axis_glyphs_use_saved_transform_and_existing_axis_scale_without_mutation():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    state=dict(skate_R=np.broadcast_to(R,(2,2,3,3)).copy(),skate_origin=np.arange(12.).reshape(2,2,3))
    state['axes']=np.stack((state['skate_origin'][:,:,None,:],state['skate_origin'][:,:,None,:]+[.2,0.,0.]),axis=-2)
    before={key:value.copy() for key,value in state.items()}
    normal=np.broadcast_to([0.,0.,1.],(2,2,3))
    arrows=skate_axis_primitives(state,unit_ratio=.5,ice_normal_world=normal)
    assert len(arrows)==8
    for arrow in arrows:
        foot=arrow['foot'];direction={'F':R[:,0],'L':R[:,1],'U':R[:,2],'iceN':normal[0,foot]}[arrow['label']]
        np.testing.assert_allclose(arrow['points'][:,0],state['skate_origin'][:,foot])
        np.testing.assert_allclose(arrow['points'][:,1],state['skate_origin'][:,foot]+.2*direction)
        assert 'foot %d'%foot in arrow['name'] and 'not source contact' in arrow['name']
    assert len(skate_axis_primitives(state))==6
    for key in state:np.testing.assert_array_equal(state[key],before[key])


def test_optional_body_projection_uses_each_variants_own_pixels_and_preserves_source(tmp_path):
    new = np.arange(68., dtype=float).reshape(2, 17, 2)
    old = new+500.
    review_asset(tmp_path, test_COCO_uv=new, old_COCO_uv=old,
        old_blade_uv=np.zeros((2, 2, 3, 2)), old_boot_uv=np.zeros((2, 2, 2, 2)),
        old_axis_rate_rad_s=np.array([[np.nan, np.nan], [0., 0.]]))
    path = tmp_path/'review.json'
    document = json.loads(path.read_text(encoding='utf8'))
    document['variants']['comparison'] = 'old'
    document['source_structures'] = [dict(pts=1256, foot=0, features=dict(
        boot_outline=dict(points_uv=[[3., 4.], [5., 6.]], status='observed')))]
    path.write_text(json.dumps(document), encoding='utf8')
    review = BladeReview(path)
    for variant, expected in [('candidate', new), ('comparison', old)]:
        current = review.at(variant, 1256)
        np.testing.assert_array_equal(current['model']['COCO_uv'], expected[1])
        disabled = overlay_primitives(current)
        assert not any(p.get('evidence_kind') == 'model_body_projection' for p in disabled)
        enabled = overlay_primitives(current, include_model_body=True)
        body = [p for p in enabled if p.get('evidence_kind') == 'model_body_projection']
        bones = {p['model_joint_indices'] for p in body}
        assert {(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)} <= bones
        for primitive in body:
            np.testing.assert_array_equal(primitive['points'], expected[1, list(primitive['model_joint_indices'])])
            assert primitive['kind'] == 'line' and primitive['color'] == (.72, .3, 1., 1.)
        source_before = [p for p in disabled if p.get('evidence_kind') == 'source_structure']
        source_after = [p for p in enabled if p.get('evidence_kind') == 'source_structure']
        assert len(source_before) == len(source_after) == 1
        np.testing.assert_array_equal(source_before[0]['points'], source_after[0]['points'])
        assert source_before[0]['color'] == source_after[0]['color']
    for variant, pts in [('missing', 1256), ('candidate', 1255)]:
        assert not any(p.get('evidence_kind') == 'model_body_projection'
                       for p in overlay_primitives(review.at(variant, pts), include_model_body=True))
