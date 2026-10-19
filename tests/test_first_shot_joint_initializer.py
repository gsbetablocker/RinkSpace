from types import SimpleNamespace
import json

import numpy as np
import pytest

from calibration.program_tracking.first_shot_joint_initializer import (
    write_programme_prefix_initializer, authenticate_prefix_parameters)


def parent():
    q=np.arange(7687,dtype=np.float64)/31
    aux={'programme_parameter_count':np.array([7674],np.int64),
         'static_venue_initial_camera_state':np.linspace(-3.,1.,13),
         'native_pts':np.array([256256,256512],np.int64)}
    aux.update({'original_chart_'+str(i):np.array([i+.01],np.float64)for i in range(12)})
    return dict(q=q,aux=aux,identity={'binding':{'path':'unchanged','sha256':'known'}},
        reference={'path':'complete_accepted.npz','sha256':'full'},
        values={'damping':np.array(.003),'next_line_search_fraction':np.array(.125)})


def test_prefix_initializer_keeps_every_aux_and_never_claims_accepted_resume(tmp_path):
    value=parent();ref=write_programme_prefix_initializer(value,tmp_path/'prefix.npz')
    with np.load(ref['path'],allow_pickle=False)as saved:
        np.testing.assert_array_equal(saved['q'].view(np.uint64),value['q'][:7674].view(np.uint64))
        for key,array in value['aux'].items():np.testing.assert_array_equal(saved['aux__'+key],array)
        assert len([k for k in saved.files if k.startswith('aux__')])==15
        assert not {'r','J','scales','quadratic'}&set(saved.files)
        identity=json.loads(str(saved['identity_json']))
        assert identity['accepted_state_claimed'] is False
        assert identity['accepted_complete_stage_parent']==value['reference']
        assert identity['original_wide13_not_discarded_from_complete_new_stage'] is True
    # The historical auxiliary camera is distinct from the latest accepted
    # camera; neither may be silently substituted for the other.
    assert not np.array_equal(value['aux']['static_venue_initial_camera_state'],value['q'][7674:])


def test_preliminary_parameter_auth_rejects_one_bit_and_requires_later_source_auth():
    value=parent();owner=SimpleNamespace(start=value['q'][:7674].copy(),
        checkpoint_arrays=lambda:{'native_pts':value['aux']['native_pts'].copy()},
        seed_receipt={'initial_damping':.003,'initial_line_search_fraction':.125})
    proof=authenticate_prefix_parameters(owner,value)
    assert proof['full_old_source_authentication_still_required_after_explicit_group_quarantine']
    owner.start[22]=np.nextafter(owner.start[22],np.inf)
    with pytest.raises(ValueError,match='programme prefix'):
        authenticate_prefix_parameters(owner,value)


def test_complete_review_media_export_uses_existing_owner_and_exact_native_clock(tmp_path):
    from calibration.solver_runtime import install
    install()
    from calibration.program_tracking.first_shot_source_transition import reference
    from calibration.program_tracking.first_shot_joint_stage_runtime import export_complete_review_inputs
    packet=tmp_path/'source.json'
    source=dict(native_pts=256256,actual_pts=256256,programme_frame_index_1based=1,
        native_frame_index_1based=1002,
        time_base='1/12800',source_time_seconds=20.02,image_path='original.png',png_sha256='image-sha',
        mask_path='original-mask.png',mask_sha256='mask-sha')
    packet.write_text(json.dumps({'source':source}),encoding='utf8')
    frame=dict(index=1,native_pts=256256,source_point_count=1,source_packet=reference(packet),
        rows=[dict(target={'kind':'ice_curve','curve_id':'C:segment0'},source_point_ids=['C/0'],
            uv=np.array([[123.,456.]]),sigma=np.array([1.7]))])
    owner=SimpleNamespace(bundle=SimpleNamespace(frames=[frame],
        whitening=[SimpleNamespace(count=2)],ledgers=[{'actual':True}]),
        scales=np.array([1.7,1.7]),quadratic=np.zeros(2,bool),formal_offsets=np.array([0,2]),
        checkpoint_arrays=lambda:{'native_pts':np.array([256256],np.int64)})
    inventory,ref=export_complete_review_inputs(owner,tmp_path/'out')
    np.testing.assert_array_equal(inventory['source_uv'],[[123.,456.]])
    receipt=json.loads(open(ref['path'],encoding='utf8').read())
    assert receipt['frames'][0]['image']=={'path':'original.png','sha256':'image-sha'}
    assert receipt['frames'][0]['source_packet']==reference(packet)
    assert receipt['another_source_bundle_constructed'] is False
    owner.checkpoint_arrays=lambda:{'native_pts':np.array([256512],np.int64)}
    with pytest.raises(ValueError,match='native clock'):
        export_complete_review_inputs(owner,tmp_path/'bad-clock')


def test_legacy_media_missing_png_digest_is_verified_by_original_decoded_pixels(tmp_path):
    import cv2,hashlib
    from calibration.program_tracking.first_shot_joint_stage_runtime import review_media_record,sha
    from calibration.program_tracking.first_shot_source_transition import reference
    pixels=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    path=tmp_path/'original.png';path.write_bytes(cv2.imencode('.png',pixels)[1].tobytes())
    source=dict(native_pts=256256,programme_frame_index_1based=1,native_frame_index_1based=1002,
        time_base='1/12800',image_path=str(path),decoded_BGR_sha256=sha(pixels),
        mask_path='original-mask.png',mask_sha256='unchanged-mask')
    packet=tmp_path/'source.json';packet.write_text(json.dumps({'source':source}),encoding='utf8')
    frame=dict(index=1,native_pts=256256,source_packet=reference(packet))
    record=review_media_record(frame)
    assert record['png_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert record['native_hw']==[3,4]
    assert record['native_frame_index_0based']==1001
    assert record['source_time_seconds']==20.02
    assert 'png_sha256'in record['derived_metadata_fields']
    cv2.imwrite(str(path),pixels+1)
    with pytest.raises(ValueError,match='decoded_BGR'):
        review_media_record(frame)


def test_effective_review_keeps_native_q_physical_and_canonical_checkpoint_separate(tmp_path):
    from scipy.sparse import eye
    from calibration.program_tracking.first_shot_joint_stage_runtime import export_effective_review,sha
    from calibration.program_tracking.first_shot_safe_frame_export import _encode,_decode
    from calibration.program_tracking.first_shot_source_transition import reference
    q=np.arange(8,dtype=float);qe=q.copy();qe[2]=99.
    calls=[]
    owner=SimpleNamespace(programme_count=5,physical_stage=object(),
        effective_q_and_chain=lambda value:(calls.append(value.copy())or(qe.copy(),eye(8,format='csr'))),
        checkpoint_arrays=lambda:{'unchanged':np.array([12.])},checkpoint_identity=lambda:{'chart':'explicit'})
    def physical_export(stage,value,folder):
        assert stage is owner.physical_stage
        np.testing.assert_array_equal(value,qe)
        for kind in ('programme','static'):
            path=folder/'programme/frames.json'if kind=='programme'else folder/'static.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            native_q=value[:5]if kind=='programme'else value
            arrays={};encoded=_encode({'q':native_q.copy()},arrays)
            array_path=path.parent/('arrays.npz'if kind=='programme'else 'static_arrays.npz')
            np.savez(array_path,**arrays)
            path.write_text(json.dumps(dict(schema='safe-native',arrays=reference(array_path),
                payload=encoded,q_sha256=sha(native_q))),encoding='utf8')
        return {'programme':{},'wide_groups':[]}
    result=export_effective_review(owner,q,tmp_path/'review',physical_export=physical_export)
    assert len(calls)==1
    for path in (result['programme']['manifest']['path'],result['static']['path']):
        receipt=json.loads(open(path,encoding='utf8').read())
        with np.load(receipt['arrays']['path'],allow_pickle=False)as saved:
            payload=_decode(receipt['payload'],{key:saved[key]for key in saved.files})
        native_q=qe[:5]if receipt['schema']=='safe-native'else qe
        np.testing.assert_array_equal(payload['q'],native_q)
        np.testing.assert_array_equal(payload['canonical_q_augmented'],q)
        np.testing.assert_array_equal(payload['physical_q_augmented'],qe)
        assert receipt['q_sha256']==sha(native_q)
        assert receipt['canonical_q_sha256']==sha(q)
        assert receipt['physical_q_sha256']==sha(qe)
