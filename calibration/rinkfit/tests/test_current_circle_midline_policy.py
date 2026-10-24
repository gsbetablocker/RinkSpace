import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from rinkfit_joint import circle_source_identity as circles
from rinkfit_joint import native_ice_identity_gate_v1 as identity
from rinkfit_joint import native_ice_recovery as recovery
from rinkfit_joint.dense_joint_proofs import _ref


def pair():
    def side(uv, transition):
        return dict(actual_source_uv=uv, material_transition=transition,
                    disposition='measured_material_transition', source_flags=dict(unresolved=False))
    return dict(disposition='measured_paired_stroke_midline', source_flags=dict(unresolved=False),
                intensity_median_or_darkest_peak_used_as_center=False, actual_source_uv=[10., 20.],
                actual_RGB_entering_interface=side([10., 18.], 'light_to_dark'),
                actual_RGB_leaving_interface=side([10., 22.], 'dark_to_light'))


def test_only_both_measured_material_sides_can_grant_C0_midline():
    row = dict(material_label='ice.C0:middle', source_uv=[10., 20.], actual_paired_midline_reading=pair())
    np.testing.assert_array_equal(circles.require_c0_midline_reading(row), [10., 20.])
    for change in ('missing_side', 'single_edge_uv', 'unresolved_side', 'wrong_transition'):
        bad = copy.deepcopy(row)
        if change == 'missing_side': bad['actual_paired_midline_reading'].pop('actual_RGB_leaving_interface')
        elif change == 'single_edge_uv': bad['source_uv'] = [10., 18.]
        elif change == 'unresolved_side': bad['actual_paired_midline_reading']['actual_RGB_leaving_interface']['source_flags']['unresolved'] = True
        else: bad['actual_paired_midline_reading']['actual_RGB_leaving_interface']['material_transition'] = 'light_to_dark'
        with pytest.raises(ValueError): circles.require_c0_midline_reading(bad)
    assert row['source_uv'] == [10., 20.]


def test_A_circles_cannot_grant_current_fit_even_with_identity():
    for name in ['A1', 'A2', 'A3', 'A4']:
        original = dict(target_id='ice.'+name+':outer', uv=[10., 20.], source_fit_eligible=True)
        result = circles.qualify_circle_profile_row(original, identity_evidence=dict(confirmed=True))
        assert not result['source_fit_eligible'] and not result['source_geometry_eligible']
        assert result['uv'] == original['uv'] and original['source_fit_eligible']
        with pytest.raises(ValueError):
            circles.require_c0_midline_reading(dict(material_label='ice.'+name+':outer', actual_paired_midline_reading=pair()))


def test_C0_identity_alone_does_not_turn_a_single_edge_into_a_midline(monkeypatch):
    monkeypatch.setattr(circles, '_validate_identity', lambda *a: True)
    row = dict(target_id='ice.C0:outer', uv=[10., 18.], source_fit_eligible=True)
    result = circles.qualify_circle_profile_row(row, identity_evidence=dict(confirmed=True))
    assert result['physical_instance_qualification'] == 'source_circle_identity_confirmed'
    assert not result['source_fit_eligible'] and 'measurement_operator_resolved' not in result
    import rinkspace.vision_source_shapes as reader
    monkeypatch.setattr(reader, 'read_source_stroke_midlines', lambda *a, **k: dict(readings=[pair()], source_binding={}, RGB_sha256='rgb'))
    measured = circles.read_confirmed_circle_midlines(None, rows=[row], identity_evidence=[{}])['rows'][0]
    assert measured['source_fit_eligible'] and measured['target_id'] == 'ice.C0:middle'
    assert measured['source_uv'] == [10., 20.] and measured['original_source_uv'] == [10., 18.]
    assert row['uv'] == [10., 18.]


def test_recovery_without_independent_review_stops_before_loading_any_packet(tmp_path):
    with pytest.raises(ValueError, match='Separate positive physical'):
        recovery.recover(packet='absent', qualification='absent', weights='absent', original_weights='absent',
            evidence='absent', base_frames_by_stream={}, output=tmp_path/'not_issued', stream_prefix='unused')
    assert not (tmp_path/'not_issued').exists()


def descriptor(tmp_path, name, value):
    value = np.asarray(value); path = tmp_path/name; value.tofile(path)
    return dict(_ref(path), dtype=value.dtype.str, shape=list(value.shape))


def test_selected_compact_raw_IDs_do_not_require_full_domain_strings(tmp_path):
    ids = [b'only/a', b'only/b']; blob = b''.join(ids); (tmp_path/'ids.u8').write_bytes(blob)
    offsets = descriptor(tmp_path, 'offsets.u64', np.array([0, len(ids[0]), len(blob)], '<u8'))
    reg = dict(row_start=4307920, count=2, source_id_offsets=offsets, source_id_utf8=_ref(tmp_path/'ids.u8'))
    doc = dict(row_count=4307922, source_row_ids=['prefix_only'], native_ad_source_bindings=[reg])
    assert identity.selected_packet_source_ids(doc, tmp_path, [0, 4307921]) == {0:'prefix_only', 4307921:'only/b'}
    with pytest.raises(ValueError, match='outside raw row'): identity.selected_packet_source_ids(doc, tmp_path, [-1])


def test_actual_recovery_calls_positive_gate_with_compact_ID_before_any_eligible_write(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path/name; path.write_text(json.dumps(value), encoding='utf8'); return path
    source_id = 'dense_native_source/s04/256/F2:outer/7'
    (tmp_path/'id.u8').write_bytes(source_id.encode())
    reg = dict(row_start=1, count=1, source_id_utf8=_ref(tmp_path/'id.u8'),
               source_id_offsets=descriptor(tmp_path, 'offsets.u64', np.array([0,len(source_id)],'<u8')),
               source_store_ordinal=descriptor(tmp_path, 'ordinal.u64', np.array([7],'<u8')))
    arrays = {k:descriptor(tmp_path,k+'.bin',a) for k,a in dict(admission=np.array([1,0],'u1'),
        fit_role=np.array([1,0],'u1'), native_pts=np.array([0,256],'<i8'), shot_index=np.array([0,3],'<u2'),
        material_index=np.array([0,149],'<u4'), path=np.array([0,0],'<u4'), uv=np.array([[0,0],[10,20]],'<f8')).items()}
    packet=save('packet.json',dict(row_count=2,arrays=arrays,source_row_ids=['prefix'],dense_source_binding=reg))
    q=save('qualification.json',dict(source_packet=_ref(packet))); w=save('weights.json',dict(source_packet=_ref(packet)))
    old=save('original_weights.json',dict(files={k:descriptor(tmp_path,k,np.array(v,dtype=dtype))for k,v,dtype in
        [('weights.f64',[1,1],'<f8'),('active.u8',[1,1],'u1'),('information_class.u8',[1,1],'u1')]}))
    proof=dict(row=1,source_row_id=source_id,source_UV=[10.,20.],native_pts=256,material_index=149,
               physical_surface='ice',adjacent_native_exposure_support=1,source_span_distance_px=0.)
    with gzip.open(tmp_path/'proofs.gz','wt',encoding='utf8') as f:f.write(json.dumps(proof)+'\n')
    dummy=save('source.json',{})
    wanted=descriptor(tmp_path,'wanted.u64',np.array([1],'<u8'))
    evidence=save('evidence.json',dict(schema='rinkfit.native_ice_source_restoration_evidence.v1',source_UV_changed=False,
        source_model_residual_used=False,reader=_ref(dummy),per_row_evidence=_ref(tmp_path/'proofs.gz'),source_anchors=_ref(dummy),
        source_span_matches=_ref(dummy),source_files=[],restored_packet_rows=wanted,restored_rows=1))
    calls=[]
    def refuse(review, packet_ref, wanted, doc, actual, proofs, *, source_ids):
        calls.append(source_ids); assert source_ids[1] == source_id and actual['path'][1] == 0
        raise ValueError('independent_gate_refused')
    monkeypatch.setattr(identity,'require_physical_recovery_review',refuse)
    output=tmp_path/'not_issued'
    with pytest.raises(ValueError,match='independent_gate_refused'):
        recovery.recover(packet=packet,qualification=q,weights=w,original_weights=old,evidence=evidence,
            base_frames_by_stream={},output=output,stream_prefix='unused',physical_identity_review='explicit')
    assert len(calls)==1 and not output.exists()
