"""An inactive optical extension is proved over support, never dropped."""
from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
import pytest

from rinkspace.compat.joint_camera import decode_joint_camera, import_joint_camera


def model():
    extension={'schema':'rinkfit.source_supported_wide_brown_hinge.v1','power':2,
               'activation_log_f_native':8.,'coefficients':[.6,1.7,0.,0.]}
    return {'scene_block':0,'blocks':[{'values':[0.]*5},{'values':[0.]*15},
        {'values':[0.]*12}]+[{'values':[0.,0.,z]} for z in (8.,9.,9.,8.)]+
        [{'values':[.1,.2]}],
        'rigs':{'main':{'head_block':1,'optics_block':2,
                       'camera':{'brown_wide_extension':extension}}},
        'shots':{'s04':{'rig_id':'main','control_blocks':[3,4,5,6],
            'reference_angles_block':7,'time_origin_pts':100,
            'scope':{'first_pts':100,'stop_pts_exclusive':200},
            'camera':{'n_controls':4,'dependent_control':1}}}}


def test_inactive_hinge_keeps_definition_and_entire_support_proof(monkeypatch):
    m=model();saved=deepcopy(m)
    decoder=Mock(return_value=({'reference_dependency':{}},
                              np.array([b['values'] for b in m['blocks'][3:7]])))
    monkeypatch.setattr('rinkspace.compat.joint_camera.decode_camera',decoder)
    camera,controls=decode_joint_camera(m,'s04',None)
    assert m==saved
    assert 'brown_wide_extension' not in decoder.call_args.args[0]['camera']
    assert camera['original_camera_definition']['brown_wide_extension']==m['rigs']['main']['camera']['brown_wide_extension']
    assert camera['inactive_extension_proofs'][0]['allocated_log_f_lower_bound']==8.
    np.testing.assert_array_equal(controls[:,2],[8.,9.,9.,8.])


@pytest.mark.parametrize('change', ['active','unknown','nonfinite','wrong_count'])
def test_active_or_unknown_hinge_never_reaches_base_decoder(monkeypatch,change):
    m=model();extension=m['rigs']['main']['camera']['brown_wide_extension']
    if change=='active':m['blocks'][3]['values'][2]=7.99
    elif change=='unknown':extension['schema']='unknown'
    elif change=='nonfinite':extension['coefficients'][0]=float('nan')
    else:extension['coefficients'].pop()
    decoder=Mock()
    monkeypatch.setattr('rinkspace.compat.joint_camera.decode_camera',decoder)
    with pytest.raises(ValueError):decode_joint_camera(m,'s04',None)
    decoder.assert_not_called()


def test_packed_controls_read_explicit_triplets_instead_of_stale_aliases(monkeypatch):
    m=model()
    packed=[[.3,.4,8.2],[.5,.6,8.4],[.7,.8,8.6],[.9,1.,8.8]]
    m['blocks'].append({'values':np.asarray(packed).ravel().tolist()})
    m['shots']['s04']['control_coordinates']=[{'block':8,'offset':i*3} for i in range(4)]
    def decode(local,core):
        values=np.array([b['values'] for b in local['blocks'][1:]])
        np.testing.assert_array_equal(values,packed)
        return {'reference_dependency':{}},values
    monkeypatch.setattr('rinkspace.compat.joint_camera.decode_camera',decode)
    camera,controls=decode_joint_camera(m,'s04',None)
    np.testing.assert_array_equal(controls[:,2],np.array(packed)[:,2])
    assert camera['inactive_extension_proofs'][0]['allocated_log_f_lower_bound']==8.2
    assert camera['control_storage_import']['legacy_alias_blocks_used'] is False


@pytest.mark.parametrize('addresses', [[{'block':3,'offset':1}]*4,
    [{'block':3,'offset':0}]*4,[{'block':3,'offset':0}]])
def test_invalid_packed_addresses_are_rejected_before_projection(monkeypatch,addresses):
    m=model();m['shots']['s04']['control_coordinates']=addresses
    decoder=Mock();monkeypatch.setattr('rinkspace.compat.joint_camera.decode_camera',decoder)
    with pytest.raises(ValueError):decode_joint_camera(m,'s04',None)
    decoder.assert_not_called()


@pytest.fixture
def saved_camera_source(tmp_path):
    m=model()
    m['source_clock']={'source_video_sha256':'a'*64,'time_base':[1,12800],
                       'image_size':[1920,1080],'frame_step_pts':256}
    scope={'source':{'id':'official','sha256':'a'*64,'width':1920,'height':1080},
           'source_clock':{'time_base_num':1,'time_base_den':12800,'native_pts_step':256},
           'required_shot_ids':['s04'],
           'shots':[{'id':'s04','rig_id':'main','processing':{
               'first_pts':100,'stop_pts_exclusive':200}}]}
    def save():
        model_path,scope_path=tmp_path/'model.json',tmp_path/'scope.json'
        model_path.write_text(json.dumps(m),encoding='utf8')
        scope_path.write_text(json.dumps(scope),encoding='utf8')
        return model_path,scope_path
    return m,scope,save


@pytest.mark.parametrize('defect',[
    'different_video','different_clock','different_pixel_grid','different_pts_step',
    'missing_source_clock','missing_sha','missing_clock','missing_size','missing_pts_step',
    'bool_clock','fractional_interval','different_rig',
])
def test_import_cannot_relabel_wrong_or_incomplete_source(saved_camera_source,monkeypatch,tmp_path,defect):
    m,scope,save=saved_camera_source
    clock=m['source_clock']
    if defect=='different_video':clock['source_video_sha256']='b'*64
    elif defect=='different_clock':clock['time_base']=[1,1000]
    elif defect=='different_pixel_grid':clock['image_size']=[1280,720]
    elif defect=='different_pts_step':clock['frame_step_pts']=512
    elif defect=='missing_source_clock':m.pop('source_clock')
    elif defect=='missing_sha':clock.pop('source_video_sha256')
    elif defect=='missing_clock':clock.pop('time_base')
    elif defect=='missing_size':clock.pop('image_size')
    elif defect=='missing_pts_step':clock.pop('frame_step_pts')
    elif defect=='bool_clock':clock['time_base']=[True,12800]
    elif defect=='fractional_interval':m['shots']['s04']['scope']['first_pts']=100.5
    elif defect=='different_rig':m['shots']['s04']['rig_id']='secondary'
    model_path,scope_path=save()
    native=Mock();decoder=Mock()
    monkeypatch.setattr('rinkspace.compat.joint_camera.NativeCore',native)
    monkeypatch.setattr('rinkspace.compat.joint_camera.decode_joint_camera',decoder)
    out=tmp_path/'imported'
    with pytest.raises(ValueError):
        import_joint_camera(model_path,'s04',scope_path,'unused-library',out)
    native.assert_not_called();decoder.assert_not_called()
    assert not out.exists()


def test_valid_import_preserves_controls_and_records_metadata_check_only(saved_camera_source,monkeypatch,tmp_path):
    m,scope,save=saved_camera_source
    # Equal rational clocks preserve native PTS; no time-coordinate conversion.
    m['source_clock']['time_base']=[2,25600]
    model_path,scope_path=save()
    controls=np.arange(12,dtype='<f8').reshape(4,3)
    camera={'shot_id':'s04','rig_id':'main','reference_pts':100,
            'first_pts':100,'stop_pts_exclusive':200,'controls_ref':'controls.npy'}
    native=Mock();decoder=Mock(return_value=(camera,controls.copy()))
    monkeypatch.setattr('rinkspace.compat.joint_camera.NativeCore',native)
    monkeypatch.setattr('rinkspace.compat.joint_camera.decode_joint_camera',decoder)
    out=tmp_path/'imported'
    receipt=import_joint_camera(model_path,'s04',scope_path,'unused-library',out)
    written=json.loads((out/'camera.json').read_text())
    np.testing.assert_array_equal(np.load(out/'controls.npy',allow_pickle=False),controls)
    assert written['source_id']=='official'
    assert written['time_base_num']==1 and written['time_base_den']==12800
    proof=written['origin']['source_binding']
    assert proof['model_time_base']==[2,25600]
    assert proof['scope_time_base']==[1,12800]
    assert proof['saved_metadata_match'] is True
    assert proof['source_video_redecoded'] is False
    assert written['origin']['geometry_or_source_rows_imported'] is False
    assert receipt['fit_executed'] is False
    decoder.assert_called_once()
