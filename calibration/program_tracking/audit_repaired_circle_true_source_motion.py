from calibration.program_tracking import diagnose_first_shot_temporal_motion as qa
import cv2,numpy as np
original=qa.distance_to_feature
def exact_circle_distance(uv,feature):
 distance,common=original(uv,feature)
 geometry=feature.get('geometric_interpretation',{})
 if 'center_uv' not in geometry:return distance,common
 ellipse=(geometry['center_uv'],geometry['axes_px'],geometry['angle_degrees']);nearest=cv2.getClosestEllipsePoints(ellipse,np.asarray(uv,np.float32)).reshape(-1,2)
 # A fixed angular pad can admit over100px of unseen arc on a flat ellipse.
 # Compare only the actually supported source arc; endpoint slack is in pixels.
 angle=np.deg2rad(ellipse[2]);rotation=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]);radii=np.maximum(np.asarray(ellipse[1])*.5,.5)
 polar=(nearest-np.asarray(ellipse[0]))@rotation.T/radii;phi=np.arctan2(polar[:,1],polar[:,0]);common=np.zeros(len(uv),bool)
 for segment in feature['segments']:
  observed=np.asarray(segment['uv']);local=(observed-np.asarray(ellipse[0]))@rotation.T/radii;theta=np.unwrap(np.arctan2(local[:,1],local[:,0]))
  for turn in (-2*np.pi,0.,2*np.pi):common|=(phi+turn>=theta.min())&(phi+turn<=theta.max())
  common|=np.linalg.norm(nearest-observed[0],axis=1)<=15
  common|=np.linalg.norm(nearest-observed[-1],axis=1)<=15
 return np.linalg.norm(nearest.astype(float)-uv,axis=1),common
qa.distance_to_feature=exact_circle_distance
feature_list=qa.feature_list
qa.feature_list=lambda packet:[f for f in feature_list(packet) if f.get('source_circle_fit')]
qa.supported_motion(qa.DATA/'repair_20261014/ice_candidates','circle_true_euclidean_current')
a=qa.read(qa.OUT/'circle_true_euclidean_current_supported_motion.json');print(a['summary']);print(a['records'][:15])
