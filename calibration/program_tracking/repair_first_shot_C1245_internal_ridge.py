"""Remove one reviewed internal colour ridge and refit the same native C band."""
import cv2,numpy as np
from copy import deepcopy
from .extract_first_shot_ice import (DATA,ROOT,Extractor,NativeProfiles,read,write,sha,
 current_broadcast_exclusion,current_ice_visible_domain)
from .repair_first_shot_local_line_visibility import frozen_digest

RD=DATA/'repair_20261014';OUT=RD/'temporal_diagnostics'

def main():
 index=1245;dest=RD/f'wide_line_candidates/{index:06}.json';packet=read(dest);before=deepcopy(packet)
 old=next(f for f in packet['features']if f['target_id']=='ice.C:segment0');fit=old['source_line_fit']
 bad=np.asarray([715.8212,665.3535]);removed=[r for r in fit['direct_readouts']if np.linalg.norm(np.asarray(r['uv'])-bad)<.01]
 rows=[r for r in fit['direct_readouts']if np.linalg.norm(np.asarray(r['uv'])-bad)>=.01]
 if len(removed)!=1 or len(rows)!=6:raise ValueError('C1245 source readout inventory changed')
 source=packet['source'];bgr=cv2.imread(source['image_path'])
 if sha(source['image_path'])!=source['png_sha256']:raise ValueError('Immutable source changed')
 broadcast,_=current_broadcast_exclusion(bgr);outside,_=current_ice_visible_domain(index)
 class ReviewedCurrentRows(NativeProfiles):
  def sample(self,path,guide,radius=28,spacing=48):
   return deepcopy(rows),dict(query_count=len(rows),measured_count=len(rows),
    operator='same_current_native_two_side_readouts_reused_after_exact_internal_ridge_rejection')
 native=ReviewedCurrentRows(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside)
 model=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014';extractor=Extractor(RD,model)
 path=dict(next(p for p in extractor.paths if p['curve_id']=='C:segment0'))
 path['_line_identity_policy']=old.get('native_static_line_identity_policy',{})
 a=np.asarray(fit['point_origin']);v=np.asarray(fit['unit_tangent']);t=(np.asarray([r['uv']for r in rows])-a)@v
 # Restore the original source query basis; extent only bounds already
 # measured support. No current pixel is synthesized from the guide.
 guide=np.linspace(a,a+v*(max(10.,float(t.max())+1)),257)
 feature,_,_=extractor.line(path,guide,native);feature['navigation_origin']=old['navigation_origin']
 if feature['source_line_fit']['actual_native_readout_count']!=6:raise ValueError('Good source band observations unexpectedly lost')
 feature['current_internal_ridge_rejection']=dict(
  receipt_path=str(OUT/'C1245_internal_ridge_repair_receipt.json'),
  removal_reason='upper local red ridge is internal to the visibly wider physical red band, not both exterior paint/ice interfaces',
  same_current_source_good_material_readouts_retained=6,no_new_observations_from_projection=True)
 others=[f for f in packet['features']if f['target_id']!='ice.C:segment0']
 packet['features']=others+[feature];packet['point_count']=sum(f['point_count']for f in packet['features'])
 packet['per_feature_coverage']=[r for r in packet['per_feature_coverage']if r['target_id']!='ice.C:segment0']+[
  dict(target_id='ice.C:segment0',points=feature['point_count'],navigation_origin=feature['navigation_origin'],native_material_diagnostic=feature['extraction_diagnostic'],missing_reason=None)]
 write(dest,packet)
 marked=bgr.copy()
 for r in rows:
  sides=np.rint(r['source_evidence']['actual_source_sides_uv']).astype(int);uv=np.rint(r['uv']).astype(int)
  cv2.line(marked,tuple(sides[0]),tuple(sides[1]),(0,255,255),1)
  for q in sides:cv2.circle(marked,tuple(q),3,(0,255,255),-1)
  cv2.circle(marked,tuple(uv),3,(0,0,255),-1)
 crop=np.hstack([bgr[590:1080,480:820],marked[590:1080,480:820]])
 image=OUT/'C1245_after_internal_ridge_rejection.jpg';cv2.imwrite(str(image),crop,[cv2.IMWRITE_JPEG_QUALITY,97])
 receipt=dict(index=index,before_sha256=frozen_digest(before),before_sha256_kind='canonical_packet_object',after_packet_sha256=sha(dest),source_sha256=sha(source['image_path']),
  removed_actual_readout=removed[0],retained_current_native_readouts=rows,previous_feature=old,current_feature=feature,
  non_C_features_preserved_exactly=others==[f for f in before['features']if f['target_id']!='ice.C:segment0'],
  actual_readout_count_before=7,actual_readout_count_after=6,wide_packet_only=True,original_source_masks_models_and_ice_candidates_unchanged=True,
  revised_native_raw_and_two_flank_view=str(image),native_visual_review_completed=False)
 write(OUT/'C1245_internal_ridge_repair_receipt.json',receipt)
 print(feature['point_count'],len(rows),sha(dest),flush=True)

if __name__=='__main__':main()
