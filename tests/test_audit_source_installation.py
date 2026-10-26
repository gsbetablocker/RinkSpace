from copy import deepcopy
import pytest
from rinkspace.source_binding import canonical_source_binding
from rinkspace.vision_geometry import _same_frame


def fixture():
    scope = dict(event_id='event', source=dict(sha256='a'*64, width=20, height=10),
        source_clock=dict(time_base_num=1, time_base_den=1000, native_pts_step=10),
        rig_bindings=dict(ids={'main':'long_side', 'end':'low_end'}), shots=[
            dict(id='s01', rig_id='main', processing=dict(first_pts=100, stop_pts_exclusive=200)),
            dict(id='s02', rig_id='end', processing=dict(first_pts=200, stop_pts_exclusive=300))])
    binding = dict(source_sha256='a'*64, source_pts=100, time_base='1/1000', shot_id='s01', native_hw=[10,20])
    return scope, binding


def test_legacy_event_installation_is_explicit_read_only_mapping():
    scope,binding=fixture();binding['installation_id']='event';before=deepcopy(binding)
    with pytest.raises(ValueError,match='installation'):canonical_source_binding(binding,scope)
    current=canonical_source_binding(binding,scope,allow_legacy_event_installation=True)
    assert binding==before
    assert (current['event_id'],current['rig_id'],current['installation_id'])==('event','main','long_side')
    assert current['installation_mapping']['historical_packet_modified'] is False


@pytest.mark.parametrize('key,value', [('rig_id','end'),('event_id','other'),('installation_id','low_end'),
    ('source_sha256','b'*64),('source_pts',201),('source_pts',101),('time_base','1/50'),('native_hw',[20,10])])
def test_conflicting_registered_binding_is_rejected(key,value):
    scope,binding=fixture();binding[key]=value
    with pytest.raises(ValueError):canonical_source_binding(binding,scope)


def test_frame_comparison_rejects_installation_and_rig_conflicts():
    scope,binding=fixture();current=canonical_source_binding(binding,scope)
    for field in ('event_id','rig_id','installation_id'):
        with pytest.raises(ValueError):_same_frame(current,dict(current,**{field:'other'}))
