"""An inactive optical extension is proved over support, never dropped."""
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from rinkspace.compat.joint_camera import decode_joint_camera


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
