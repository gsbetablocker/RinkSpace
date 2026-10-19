"""Physical camera25 must be converted to the original conditioned J15."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/diagnostics'))
import joint_absolute_source_derivatives_20261011 as derivatives
from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState


@pytest.mark.parametrize('changed', [False, True])
def test_public_camera_map_matches_conditioned_state_and_actual_source_pixels(changed):
    unit=.025; at=100
    cal=SharedCalibration('test',(10.,-20.,4.),lever_xyz=(.1,-.6,.3),
        k1=(.2,), principal_x_fraction=(.5,), principal_y_fraction=(.5,),axial=(0.,.1,.03))
    reference=np.zeros(13);reference[9:12]=[.02,-.03,.01];reference[12]=np.log(1100.)
    def camera(value,p):
        current=replace(cal,pivot_xyz=tuple(np.asarray(cal.pivot_xyz)+(value[:3]-reference[:3])/unit),
            lever_xyz=tuple(np.asarray(cal.lever_xyz)+(value[3:6]-reference[3:6])/unit),
            principal_x_fraction=(cal.principal_x_fraction[0]+(value[6]-reference[6])/1920.,*cal.principal_x_fraction[1:]),
            principal_y_fraction=(cal.principal_y_fraction[0]+(value[7]-reference[7])/1080.,*cal.principal_y_fraction[1:]),
            k1=(cal.k1[0]+.001*(value[8]-reference[8]),*cal.k1[1:]))
        return current.camera25(ViewState(tuple(value[9:12]),float(value[12]-np.log(1920.))))
    block=SimpleNamespace(start=9,xyz_slots=np.arange(3),optics_chart=SimpleNamespace(
        indices=np.arange(6,9),mount_indices=np.arange(3,6)),
        bundle=SimpleNamespace(index={at:0},calibration=cal,camera=camera))
    c=SimpleNamespace(q=reference.copy(),arrays={'q':reference.copy()},provider=SimpleNamespace(scene_unit_m=unit),
        state=SimpleNamespace(bypts={at:block},evaluation_q=lambda q:np.asarray(q)))
    value=reference.copy()
    if changed:value+=np.r_[np.full(6,.001),[1.,-.7,2.],[.001,-.002,.003,.02]]
    columns,mapping=derivatives.camera_map(c,at,value=value)
    from projection import to_state
    def conditioned(q):
        cam=camera(q,at)
        return np.r_[to_state(cam),cam[11:15]]
    for ordinal,column in enumerate(columns):
        h=1e-4 if column in (6,7,8) else 1e-7 if column in (9,10,11) else 1e-6
        delta=np.zeros(13);delta[column]=h
        actual=(conditioned(value+delta)-conditioned(value-delta))/(2*h)
        np.testing.assert_allclose(mapping[:,ordinal],actual,rtol=2e-6,atol=3e-9)
    from z190_source_camera_diagnostic import absolute_value_jac15
    row=dict(id='point_source',world=np.array([[0.,0.,20.],[1.,1.,20.]]),
        uv=np.zeros((2,2)),sigma=np.ones(2),point_correspondence=True)
    _,J15=absolute_value_jac15(camera(value,at),[row],True)
    direction=np.sin(np.arange(13)+1)*1e-6
    fd=(absolute_value_jac15(camera(value+direction,at),[row],False)
        -absolute_value_jac15(camera(value-direction,at),[row],False))/2
    np.testing.assert_allclose(J15@mapping@direction,fd,rtol=2e-6,atol=2e-9)
