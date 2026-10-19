"""Independent zero, FD, initializer and spawn-worker zoom regressions."""
from copy import copy,deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json,pickle,multiprocessing
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np,pytest
from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
from calibration.program_tracking.first_shot_fit import _CameraState,_init_source,_source_chunk,load_parameter_initializer
from calibration.program_tracking.first_shot_zoom_camera import PARAMETERS,append_zoom_initializer
from calibration.program_tracking.first_shot_row_camera import prepare_frame_row_camera
from calibration.program_tracking.first_shot_row_source import evaluate_row_timed_frame,row_timed_raw
from calibration.program_tracking.first_shot_source_kernel import _interfaces
from calibration.program_tracking.complete_official_fit_dataset import camera_fit_state
from calibration.program_tracking.official_static_surface_dataset import reference

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014')

@pytest.fixture(scope='module')
def zoom_states(tmp_path_factory):
 if not (BASE/'revised_inputs_rs_v5_mount/binding.json').exists() or not (BASE/'solver_rs_v5_mount/latest_accepted_state.npz').exists():
  pytest.skip('Real RinkSpace native50 binding/checkpoint fixture is not installed')
 old=FirstShotPhysicalState.from_binding(BASE/'revised_inputs_rs_v5_mount/binding.json')
 with np.load(BASE/'solver_rs_v5_mount/latest_accepted_state.npz',allow_pickle=False)as z:oldq=z['q'].copy()
 folder=tmp_path_factory.mktemp('zoom_real_state');config=dict(schema='rinkspace.first-shot-shared-zoom-camera.v1',source_video_sha256=old.source['sha256'],source_clock=old.source_clock,parameters=list(PARAMETERS),driver='final_effective_domain_log_focal_ratio',per_exposure_extra_parameters=0,manufacturer_calibrated_coefficients=False,reference_domain_log_focal_ratio=float(oldq[old.view_columns(1200)[3]]))
 p=folder/'zoom.json';p.write_text(json.dumps(config),encoding='utf8');binding=deepcopy(old.binding);binding['shared_zoom_camera']=reference(p);bp=folder/'binding.json';bp.write_text(json.dumps(binding),encoding='utf8');state=FirstShotPhysicalState.from_binding(bp);q,proof=append_zoom_initializer(state,oldq)
 return old,state,oldq,q

def test_zero_slopes_preserve_existing_complete_state_and_projection(zoom_states):
 old,state,oldq,q=zoom_states
 assert old.count==5168 and state.count==5172
 np.testing.assert_array_equal(state.zoom_parameter_indices,np.arange(5168,5172))
 assert q[:5168].tobytes()==oldq.tobytes()
 np.testing.assert_array_equal(state.initial_q[:5168],old.initial_q)
 np.testing.assert_array_equal(state.world_parameter_indices,old.world_parameter_indices)
 for n in [1,259,516,1200,1239,1248]:
  assert state.camera25(q,n).tobytes()==old.camera25(oldq,n).tobytes()
  before=prepare_frame_row_camera(old,oldq,n);after=prepare_frame_row_camera(state,q,n)
  for y in [0.,211.25,800.5,1079.]:
   np.testing.assert_array_equal(after.camera25_at_row(y),before.camera25_at_row(y))

def nonzero(zoom_states):
 _,state,_,base=zoom_states;q=base.copy();q[state.zoom_parameter_indices]=[18.,-11.,6.,.003];return state,q

def test_central_complete_analytic_chain_has_zoom_and_logf_derivatives(zoom_states):
 state,q=nonzero(zoom_states)
 for n in [259,1200,1248]:
  camera,cols,J=state.camera15_mapping(q,n);fd=np.empty_like(J)
  for k,col in enumerate(cols):
   h=1e-6 if col in state.view_columns(n)else 1e-5;p=q.copy();m=q.copy();p[col]+=h;m[col]-=h
   fd[:,k]=(camera_fit_state(state.camera25(p,n))-camera_fit_state(state.camera25(m,n)))/(2*h)
  np.testing.assert_allclose(J,fd,rtol=3e-7,atol=3e-8)
  assert cols[-4:].tolist()==state.zoom_parameter_indices.tolist()
  assert np.linalg.norm(J[:,list(cols).index(state.view_columns(n)[3])])>0

def test_row_chain_and_vector_values_include_neighbor_zoom_driver(zoom_states):
 state,q=nonzero(zoom_states)
 for n in [1,1200,1248]:
  ctx=prepare_frame_row_camera(state,q,n);rows=np.array([0.,233.25,800.5,1079.]);batch=ctx.camera25_at_rows(rows)
  for j,y in enumerate(rows):
   camera,cols,J=ctx.camera15_mapping(y);np.testing.assert_allclose(batch[j],camera,rtol=1e-13,atol=2e-11)
   fd=np.empty_like(J)
   for k,col in enumerate(cols):
    h=1e-6 if state.view_start<=col<state.view_stop else 1e-5;p=q.copy();m=q.copy();p[col]+=h;m[col]-=h
    cp=prepare_frame_row_camera(state,p,n).camera25_at_row(y);cm=prepare_frame_row_camera(state,m,n).camera25_at_row(y)
    fd[:,k]=(camera_fit_state(cp)-camera_fit_state(cm))/(2*h)
   np.testing.assert_allclose(J,fd,rtol=3e-6,atol=4e-8)

def test_initializer_and_worker_pickle_keep_all_prior_values(zoom_states,tmp_path):
 _,state,oldq,q=zoom_states
 # Source material restarts are separate from parameter-chart extension.
 light=copy(state);light.material_revision_proof=None
 archive=BASE/'solver_rs_v5_mount/latest_accepted_state.npz';out,proof=load_parameter_initializer(SimpleNamespace(state=light),archive,reference(archive)['sha256'])
 assert out[:5168].tobytes()==oldq.tobytes()
 assert proof['appended_shared_zoom_camera']['existing_coordinates_byte_exact']
 assert proof['appended_shared_row_timing']is None
 worker=pickle.loads(pickle.dumps(_CameraState(state)))
 np.testing.assert_array_equal(worker.zoom_parameter_indices,state.zoom_parameter_indices)
 assert worker.zoom_configuration==state.zoom_configuration
 for n in [1,1200,1248]:np.testing.assert_array_equal(worker.camera25(q,n),state.camera25(q,n))

@dataclass
class DenseWhitening:
 matrix:np.ndarray
 def apply(self,value):return self.matrix@value

def synthetic_factor(zoom_states):
 full,q=nonzero(zoom_states);state=copy(full);state.world_parameter_indices=np.array([0,1]);state.world_parameter_count=2;n=1200;camera=state.camera25(q,n);r=cv2.Rodrigues(camera[3:6])[0];K=np.array([[np.exp(camera[6]),camera[10],camera[8]],[0,np.exp(camera[7]),camera[9]],[0,0,1.]])
 def seen(uv,depth):
  ray=cv2.undistortPoints(np.asarray(uv,float).reshape(-1,1,2),K,camera[11:25]).reshape(-1,2);return camera[:3]+(np.c_[ray,np.ones(len(ray))]*depth)@r
 world={'line':seen([[500.,220.],[1300.,900.]],20.),'points':seen([[700.,300.],[1200.,820.]],18.)};dw={k:np.zeros(a.shape+(2,))for k,a in world.items()}
 for k in dw:dw[k][:,0,0]=1.;dw[k][:,2,1]=1.
 project=_interfaces()[0];line=project(camera,world['line'],with_jacobian=False)[0];points=project(camera,world['points'],with_jacobian=False)[0];uv=.25*line[[0,1]]+.75*line[[1,0]]+[[1.,-2.],[-1.5,2.]]
 frame=dict(index=n,source_point_count=4,rows=[dict(_target_key='line',target=dict(kind='ice_curve'),point_correspondence=False,uv=uv),dict(_target_key='points',target=dict(kind='ad_corner'),point_correspondence=True,uv=points+[[1.,1.],[-1.,-1.]])]);w=DenseWhitening(np.array([[1.,.2,.3,0,0,0,.1,0],[0,1,0,.4,0,.1,0,0],[.1,0,1,0,.2,0,0,.2],[0,.2,0,1,0,.1,.3,0],[0,0,.2,.1,1,.3,.1,.4]]));return state,q,world,dw,frame,w

def test_correlated_source_full_FD_and_real_spawn_worker(zoom_states):
 state,q,world,dw,frame,w=synthetic_factor(zoom_states);r,J=evaluate_row_timed_frame(state,q,frame,w,world,dw)
 def worlds(qq):return {k:a+np.einsum('ijk,k->ij',dw[k],qq[:2]-q[:2])for k,a in world.items()}
 cols=np.r_[0,1,state.zoom_parameter_indices,state.view_columns(1199),state.view_columns(1200),state.view_columns(1201),state.row_readout_index];fd=np.empty((len(r),len(cols)))
 for k,col in enumerate(cols):
  h=1e-6 if state.view_start<=col<state.view_stop else 1e-5;p=q.copy();m=q.copy();p[col]+=h;m[col]-=h;fd[:,k]=(w.apply(row_timed_raw(state,p,frame,worlds(p)))-w.apply(row_timed_raw(state,m,frame,worlds(m))))/(2*h)
 np.testing.assert_allclose(J[:,cols].toarray(),fd,rtol=3e-5,atol=4e-5)
 with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),initializer=_init_source,initargs=(_CameraState(state),[frame],[w]))as pool:rr,JJ,receipt=pool.submit(_source_chunk,(q,world,dw,[0])).result()
 np.testing.assert_array_equal(rr,r);np.testing.assert_allclose(JJ.toarray(),J.toarray(),rtol=0,atol=0)


def test_zero_zoom_preserves_legacy_zero_tails_without_local_materials():
 from calibration.program_tracking.first_shot_zoom_camera import with_zoom_calibration
 from calibration.program_tracking.z190_physical_camera import SharedCalibration,ViewState
 state=SimpleNamespace(zoom_parameter_indices=np.arange(4),scene_unit_working_m=.3,zoom_configuration={'reference_domain_log_focal_ratio':1.2})
 old=SharedCalibration('synthetic',(1.,-3.,4.),lever_xyz=(.05,-.1,.2),principal_x_fraction=(.48,),principal_y_fraction=(.55,),k1=(.02,),axial=(0.,0.,0.))
 new=with_zoom_calibration(state,np.zeros(4),old)
 for f in [.3,1.2,1.6]:
  view=ViewState((.1,-.3,.2),f)
  assert old.camera25(view).tobytes()==new.camera25(view).tobytes()


def test_append_zoom_keeps_all_old_coordinates_without_local_materials():
 state=SimpleNamespace(zoom_parameter_indices=np.arange(5168,5172),zoom_reference={'fixture':'synthetic-zero-extension'})
 old=np.linspace(-.4,.6,5168,dtype=np.float64);new,proof=append_zoom_initializer(state,old)
 assert new[:5168].tobytes()==old.tobytes()
 np.testing.assert_array_equal(new[5168:],np.zeros(4))
 assert proof['existing_coordinates_byte_exact']


def test_optical_envelope_checks_terminal_rows_without_local_materials():
 from calibration.program_tracking.first_shot_zoom_camera import zoom_mount_envelope
 from calibration.program_tracking.z190_physical_camera import SharedCalibration
 state=SimpleNamespace(view_start=0,view_stop=12,scene_unit_working_m=.3,zoom_configuration={'reference_domain_log_focal_ratio':0.},mount_absolute_lower=np.array([-.05,-.2,-.11]),mount_absolute_upper=np.array([.05,-.1,.11]))
 q=np.zeros(12);q[[3,7,11]]=[0.,0.,1.]
 calibration=SharedCalibration('synthetic',(0.,0.,1.),lever_xyz=(0.,-.5,0.),q_reference=0.,axial=(0.,1/3))
 result=zoom_mount_envelope(state,q,calibration)
 assert result['valid']is False
 # Native axial values stay in [.0,.1] m; the supported last half-interval
 # reaches .15 m, so checking only native knots would wrongly pass.
 np.testing.assert_allclose(result['axial_displacement_range_working_m'],[0.,.15],atol=1e-15)


class SyntheticZoomMount:
 """Use the production two-coordinate chart without any local data files."""
 view_start=0
 view_stop=12
 scene_unit_working_m=.3
 zoom_parameter_indices=np.arange(128,132)
 zoom_configuration={'reference_domain_log_focal_ratio':.25}
 mount_absolute_lower=np.array([-.05,-.2,-.11])
 mount_absolute_upper=np.array([.05,-.1,.11])
 base_lever_z_working_m=.025

 def calibration(self,q):
  from calibration.program_tracking.z190_physical_camera import SharedCalibration
  return SharedCalibration('synthetic',(0.,0.,1.),
   lever_xyz=(0.,-.5,(self.base_lever_z_working_m+q[123])/self.scene_unit_working_m),
   q_reference=self.zoom_configuration['reference_domain_log_focal_ratio'],
   axial=(0.,q[self.zoom_parameter_indices[3]]/self.scene_unit_working_m))


def synthetic_mount_state(focals=(0.,0.,1.)):
 state=SyntheticZoomMount();q=np.linspace(-.004,.009,132)
 q[[3,7,11]]=focals;q[123]=-.11-state.base_lever_z_working_m
 q[state.zoom_parameter_indices]=0.
 return state,q


def independent_mount_faces(state,q):
 # Directly check every native optical centre and both supported half-intervals,
 # plus the mechanical reference. No production polygon helper is reused.
 f=q[state.view_start+3:state.view_stop:4]
 x=np.r_[0.,f-state.zoom_configuration['reference_domain_log_focal_ratio'],
  1.5*f[0]-.5*f[1]-state.zoom_configuration['reference_domain_log_focal_ratio'],
  1.5*f[-1]-.5*f[-2]-state.zoom_configuration['reference_domain_log_focal_ratio']]
 A=np.c_[np.ones(len(x)),x]
 return np.r_[A,-A],np.r_[np.full(len(x),state.mount_absolute_lower[2]),
  np.full(len(x),-state.mount_absolute_upper[2])]


def check_mount_range(state,q):
 from calibration.program_tracking.first_shot_zoom_camera import zoom_mount_envelope
 cal=state.calibration(q);lever=np.asarray(cal.lever_xyz)*state.scene_unit_working_m
 assert np.all(lever>=state.mount_absolute_lower-1e-12)
 assert np.all(lever<=state.mount_absolute_upper+1e-12)
 assert zoom_mount_envelope(state,q,cal)['valid']


def test_mount_domain_contains_real_zero_zoom_boundaries(zoom_states):
 from calibration.program_tracking.first_shot_zoom_camera import zoom_mount_linear_constraints,recover_zoom_mount_trial
 _,state,_,base=zoom_states
 selected=np.r_[state.zoom_parameter_indices,123,121,122]
 for edge in [state.mount_absolute_lower[2],state.mount_absolute_upper[2]]:
  q=base.copy();q[123]+=edge-state.calibration(q).lever_xyz[2]*state.scene_unit_working_m
  domain=zoom_mount_linear_constraints(state,q,selected)
  assert domain['matrix'].shape==(6,len(selected))
  assert np.all(domain['lower']<=2e-15)
  assert np.any(abs(domain['lower'])<2e-15)
  check_mount_range(state,q)
  result=recover_zoom_mount_trial(state,q,q)
  assert result['q'].tobytes()==q.tobytes()
  assert result['report']['modified_parameter_indices']==[]


@pytest.mark.parametrize('focals,slope,lever',[
 ((0.,0.,1.),.4,.02),       # The terminal maximum exceeds every native knot.
 ((0.,0.,1.),-.4,-.02),
 ((0.,1.,1.),.4,.02),       # The first terminal minimum is the active face.
 ((1.,0.,0.),-.4,-.02),
 ((2.,-.5,.2),.4,.2),       # Focal extrema change order from the parent.
 ((-.5,2.,.2),-.4,-.2),
 ((2.,2.,2.),-.06,.2),     # Optical centres fit, but the mechanical reference does not.
 ((-2.,-2.,-2.),.06,.2),
])
def test_mount_recovery_is_nearest_two_coordinate_projection(focals,slope,lever):
 from scipy.optimize import minimize,LinearConstraint
 from calibration.program_tracking.first_shot_zoom_camera import recover_zoom_mount_trial,zoom_mount_linear_constraints
 state,parent=synthetic_mount_state();proposed=parent.copy()
 proposed[[3,7,11]]=focals;proposed[123]=lever-state.base_lever_z_working_m
 proposed[131]=slope;before=proposed.copy();parent_before=parent.copy()
 answer=recover_zoom_mount_trial(state,parent,proposed);q=answer['q']
 assert proposed.tobytes()==before.tobytes() and parent.tobytes()==parent_before.tobytes()
 unchanged=np.ones(len(q),bool);unchanged[[123,131]]=False
 assert q[unchanged].tobytes()==proposed[unchanged].tobytes()
 assert set(answer['report']['modified_parameter_indices'])<=set([123,131])
 assert answer['report']['original_source_cost_and_complete_physical_guard_required']
 check_mount_range(state,q)
 # An independent constrained optimizer establishes closest-point optimality.
 A,b=independent_mount_faces(state,proposed);start=np.array([lever,slope])
 oracle=minimize(lambda v:.5*np.sum((v-start)**2),[0.,0.],jac=lambda v:v-start,
  constraints=LinearConstraint(A,b,np.inf),method='SLSQP',
  options={'ftol':1e-14,'maxiter':100})
 assert oracle.success,oracle.message
 actual=np.array([state.calibration(q).lever_xyz[2]*state.scene_unit_working_m,q[131]])
 np.testing.assert_allclose(actual,oracle.x,rtol=0,atol=2e-9)
 # The callback must use the candidate's focal range, including swapped extrema.
 selected=np.array([131,22,123,45]);domain=zoom_mount_linear_constraints(state,proposed,selected)
 delta=q[selected]-proposed[selected]
 assert np.all(domain['matrix']@delta>=domain['lower']-2e-14)


def test_mount_domain_recovery_reaches_real_streaming_source_acceptance(tmp_path):
 from scipy.sparse import csr_matrix
 from calibration.solver_runtime import install
 from calibration.program_tracking.first_shot_zoom_camera import zoom_mount_linear_constraints,recover_zoom_mount_trial
 install()
 import z190_joint_source_solver as solver
 from z190_streaming_source_normal import SourceRowBlock
 state,initial=synthetic_mount_state();active=np.array([123,131,3,7,11])
 target=initial[active].copy();target[:2]=[.11-state.base_lever_z_working_m,-.4]
 target[2:]=[0.,1.,2.]
 mask=np.ones(len(active),bool);scales=np.ones(len(active))
 J=csr_matrix((np.ones(len(active)),(np.arange(len(active)),active)),shape=(len(active),len(initial)))
 recovered=[];accepted=[];cost_states=[]
 def source_blocks(q,derivative):
  if not derivative:cost_states.append(q.copy())
  yield SourceRowBlock(q[active]-target,J if derivative else None,mask,scales,
   dict(family='complete_zoom_mount_source_regression',original_start=0,original_stop=len(active)))
 def guard(q):
  try:check_mount_range(state,q)
  except AssertionError:raise ValueError('Original mechanical and all-row optical ranges required')
  return True
 def recover(parent,proposed):
  answer=recover_zoom_mount_trial(state,parent,proposed);recovered.append((proposed.copy(),answer))
  return answer
 def checkpoint(q,event):
  if event['checkpoint_kind']=='accepted':
   accepted.append(q.copy());return dict(stop_branch=True,reason='One complete source accepted after coupled mount recovery')
 def unused(*args):raise AssertionError('Use the original streaming source contract')
 fixed=np.setdiff1d(np.arange(len(initial)),active)
 lo=state.mount_absolute_lower[2]-state.base_lever_z_working_m
 hi=state.mount_absolute_upper[2]-state.base_lever_z_working_m
 result=solver.solve_joint_source(unused,initial,mask,scales,tmp_path,guard,
  source_block_factory=source_blocks,cache_native_source_replay=True,reuse_source_jacobian=True,
  global_schur_count=2,global_schur_indices=[123,131],fixed_parameter_indices=fixed,
  shared_parameter_bounds={123:(lo,hi)},schur_backtracking=True,
  shared_linear_constraints=lambda q,selected:zoom_mount_linear_constraints(state,q,selected),
  trial_feasibility_recovery=recover,checkpoint_callback=checkpoint,verbose=False)
 assert result.status=='stopped_by_source_review'
 assert len(accepted)==len(result.history)==1 and result.history[0]['accepted']
 assert result.history[0]['next_cost']<result.history[0]['cost']
 assert recovered and any(a['report']['modified_parameter_indices']for _,a in recovered)
 check_mount_range(state,result.q)
 assert result.q[fixed].tobytes()==initial[fixed].tobytes()
 assert any(q.tobytes()==result.q.tobytes()for q in cost_states)
 trial=result.history[0]['global_schur_trial_records'][-1]
 proof=trial['trial_feasibility_recovery']['complete_state_recovery_identity']
 assert proof['declared_fixed_values_preserved'] and proof['recovered_state_shape_and_finite_values_valid']
 assert set(proof['changed_complete_state_parameter_indices'])<=set([123,131])
