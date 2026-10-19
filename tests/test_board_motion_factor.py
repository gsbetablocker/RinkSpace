import numpy as np
import pytest
from calibration.program_tracking.board_motion_factor import evaluate_board_motion
from calibration.program_tracking.first_shot_source_kernel import _interfaces
from calibration.program_tracking.projection import from_state,project_fast

def problem():
 _interfaces()
 from calibration.program_tracking.reference_board_pair import ReferenceBoardPair
 bottom=np.array([[-10.,-10.,0.],[10.,-10.,0.],[10.,10.,0.],[-10.,10.,0.]])
 scene={'ice_boundary':{'points_xyz':bottom},'board_surface':{'bottom_edge_xyz':bottom,'top_edge_xyz':bottom+np.array([0,0,4.])}}
 ca=np.zeros(25);ca[:3]=[0,0,3.];ca[3]=np.deg2rad(95);ca[6:8]=np.log([1200,1210]);ca[8:10]=[960,540];ca[10]=.4;ca[11:15]=[.01,-.005,.0001,-.0002]
 cb=ca.copy();cb[:3]+=[.05,-.03,.01];cb[4]=.002;cb[6:8]+=.001
 world=np.array([[-2,10,1.5],[0,10,2.],[2,10,2.5]])
 u=project_fast(ca,world)[0];v=project_fast(cb,world)[0]
 surface=ReferenceBoardPair(u,v,scene,np.arange(4),ca,cb)
 return surface,ca,cb,u,v,world

def conditioned(c):return np.r_[c[:3]/100,c[3:8],c[8:11]/1000,c[11:15]]

def test_exact_value_and_both_camera_derivatives():
 surface,ca,cb,u,v,w=problem();r,ja,jb,world=evaluate_board_motion(surface,ca,cb,u,v)
 np.testing.assert_allclose(r,0,atol=1e-9);np.testing.assert_allclose(world,w,atol=1e-9)
 for side,cam,jac in ((0,ca,ja),(1,cb,jb)):
  x=conditioned(cam)
  for k in range(15):
   d=np.zeros(15);d[k]=1e-6
   hi,lo=from_state(x+d,cam),from_state(x-d,cam)
   ra=evaluate_board_motion(surface,hi if side==0 else ca,hi if side==1 else cb,u,v,jacobian=False)[0]
   rb=evaluate_board_motion(surface,lo if side==0 else ca,lo if side==1 else cb,u,v,jacobian=False)[0]
   np.testing.assert_allclose((ra-rb)/2e-6,jac[:,k],rtol=2e-5,atol=2e-5)

def test_individual_row_cameras_and_inputs_preserved():
 surface,ca,cb,u,v,w=problem();aa=np.tile(ca,(len(u),1));bb=np.tile(cb,(len(u),1));aa[:,4]=[-.0001,0,.0001];bb[:,4]+=[-.0001,0,.0001];before=[x.copy()for x in (aa,bb,u,v)];r,ja,jb,world=evaluate_board_motion(surface,aa,bb,u,v)
 same=evaluate_board_motion(surface,aa,bb,u,v,jacobian=False)
 np.testing.assert_array_equal(r,same[0]);np.testing.assert_array_equal(world,same[3]);assert ja.shape==jb.shape==(6,15)
 for x,y in zip((aa,bb,u,v),before):np.testing.assert_array_equal(x,y)

def test_first_unverified_wall_and_target_occlusion_reject():
 surface,ca,cb,u,v,w=problem();surface._allowed[:]=False
 with pytest.raises(ValueError,match='first_hit_outside_verified_print_arc'):evaluate_board_motion(surface,ca,cb,u,v)
 surface._allowed[:]=True;outside=cb.copy();outside[:3]=[0,-20,3]
 with pytest.raises(ValueError,match='occluded'):evaluate_board_motion(surface,ca,outside,u,v)


@pytest.mark.parametrize('case',('valid','occluded','unverified_before_bad_inverse','bad_inverse','unverified_before_scalar_error'))
def test_batch_preserves_complete_results_and_failure_order(case,monkeypatch):
 from calibration.program_tracking import board_motion_factor as factor
 surface,ca,cb,u,v,w=problem();aa=np.tile(ca,(len(u),1));bb=np.tile(cb,(len(u),1))
 if case=='occluded':bb[0,:3]=[0,-20,3]
 if case.startswith('unverified'):surface._allowed[:]=False
 if 'bad_inverse'in case:
  aa[1,11]=-10.;u[1]=[10000.,10000.]
 if 'scalar_error'in case:aa[1,6:8]=-1000.
 def outcome():
  try:return ('value',factor.evaluate_board_motion(surface,aa,bb,u,v))
  except Exception as error:return ('error',type(error),str(error))
 with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
  actual=outcome()
  with monkeypatch.context()as original:
   original.setattr(factor,'_world_rays',lambda cameras,pixels,tolerance:(None,None))
   expected=outcome()
 assert actual[0]==expected[0]
 if actual[0]=='error':assert actual[1:]==expected[1:]
 else:
  for a,b in zip(actual[1],expected[1],strict=True):np.testing.assert_array_equal(a,b)
