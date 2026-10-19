import numpy as np
import pytest
from rinkspace.geometry import PlaneMap
from rinkspace.motion import kinematics, split_segments, window_sensitivity


def rows(n=151):
    return [dict(t_s=i*.02,x=i*.02*3,y=i*.02*4,sigma=.01,contact="on_ice",status="accepted",
                 video_role="live",shot_id="shot1",map_id="m",entity_id="right",point_definition="contact",epoch="live") for i in range(n)]


def plane():return PlaneMap(np.eye(3),"metric_assumed","m",map_id="m")


def test_straight_speed_no_finite_turn_radius():
    r=kinematics(rows(),plane())
    assert len(r)>0
    assert all(x['speed']==pytest.approx(5) for x in r)
    assert all(x['radius'] is None for x in r)
    assert all(x['geometry_level']=='metric_assumed' for x in r)


def test_local_quadratic_acceleration():
    data=rows()
    for r in data:r.update(x=2*r['t_s']**2,y=0)
    out=kinematics(data,plane())
    assert all(r['speed']==pytest.approx(4*r['t_s'],abs=1e-8) for r in out)


def test_circle_radius():
    data=rows()
    for r in data:r.update(x=8*np.cos(.35*r['t_s']),y=8*np.sin(.35*r['t_s']))
    out=kinematics(data,plane())
    assert np.median([r['speed'] for r in out])==pytest.approx(2.8,rel=.005)
    assert np.median([r['radius'] for r in out])==pytest.approx(8,rel=.01)


@pytest.mark.parametrize("field,value", [("entity_id","left"),("shot_id","shot2"),("map_id","another"),("point_definition","shoe_tip")])
def test_no_splicing_across_changed_definition(field,value):
    d=rows(40)
    for r in d[20:]:r[field]=value
    seg=split_segments(d)
    assert [len(s) for s in seg]==[20,20]


@pytest.mark.parametrize("field,value",[("contact","unknown"),("contact","airborne"),("video_role","replay"),("status","candidate")])
def test_missing_or_replay_breaks_track(field,value):
    d=rows(40);d[20][field]=value
    assert [len(s) for s in split_segments(d)]==[20,19]


def test_gap_breaks_derivatives():
    d=rows(40)
    for r in d[20:]:r['t_s']+=.5
    assert len(split_segments(d))==2


def test_duplicate_pts_rejected():
    d=rows();d[3]['t_s']=d[2]['t_s']
    with pytest.raises(ValueError,match="increasing"):split_segments(d)


def test_projective_derivatives_rejected():
    with pytest.raises(ValueError,match="Projective"):kinematics(rows(),PlaneMap(np.eye(3)))


def test_window_sensitivity_is_available():
    r=window_sensitivity(rows(),plane())
    assert r['samples']
    assert max(x['speed_spread_over_median'] for x in r['samples'])<1e-10


def test_endpoints_omitted():
    d=rows();r=kinematics(d,plane(),half_window_s=.24)
    assert r[0]['t_s']>=.24-1e-9 and r[-1]['t_s']<=d[-1]['t_s']-.24+1e-9
