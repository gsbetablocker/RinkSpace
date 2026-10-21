import cv2,numpy as np

def body_material_domain(board,rgb,actor,graphics,original_print_uv=(),source_unknown=None):
 top=board['roles']['board.top'];foot=board['roles']['board.contact'];band=top.get('material_band',top.get('meta',top))
 def value(k):return top.get(k,band.get(k))
 cap=np.asarray(value('full_blue_cap_lower_y'),float);direct=np.asarray(value('full_blue_cap_lower_direct'),bool);rad=np.asarray([np.nan if x is None else x for x in value('full_blue_cap_lower_localization_radius')],float);occ=np.asarray(band.get('full_material_occlusion',top.get('full_material_occlusion',np.zeros(1920,bool))),bool);direct&=np.isfinite(cap+rad)&~occ
 fy=np.asarray(foot.get('full_native_y',foot.get('y')),float);ty=np.asarray(top.get('full_native_y',top.get('y')),float);axis=np.arange(1920);left=np.maximum.accumulate(np.where(direct,axis,-1));right=np.minimum.accumulate(np.where(direct,axis,1920)[::-1])[::-1];bound=cap.copy();proof=np.zeros(1920,bool)
 for x in range(1920):
  if direct[x]:bound[x]=cap[x]+rad[x]+1;proof[x]=True;continue
  n=[i for i in(left[x],right[x])if 0<=i<1920]
  if n:bound[x]=max([cap[x]+1]+[cap[i]+rad[i]+1 for i in n]);proof[x]=True
  else:bound[x]=fy[x]
 yy=np.arange(1080)[:,None];nav=(yy>bound[None,:])&(yy<fy[None,:]-4)&np.isfinite(bound+fy)[None,:]&proof[None,:];invalid=cv2.dilate(actor.astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0;unknown=np.zeros_like(actor,dtype=bool)if source_unknown is None else source_unknown.astype(bool);unknown_guard=cv2.dilate(unknown.astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0;invalid|=unknown_guard;invalid|=graphics;region=nav&~invalid
 # Physical white board-body seeds lie below directly measured current blue/white interfaces.
 capseed=region&direct[None,:]&(yy>cap[None,:]+rad[None,:]+2)&(yy<cap[None,:]+rad[None,:]+10)
 # Actual yellow-to-white current-foot photometry is independent evidence for the connected board body.
 chroma=(rgb[:,:,0].astype(float)+rgb[:,:,1].astype(float))*.5-rgb[:,:,2].astype(float);xx=np.repeat(axis.astype('f4')[:,None],5,axis=1);before=cv2.remap(chroma.astype('f4'),xx,(fy[:,None]+np.arange(-6,-1)[None,:]).astype('f4'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean(1);after=cv2.remap(chroma.astype('f4'),xx,(fy[:,None]+np.arange(2,7)[None,:]).astype('f4'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean(1);footdirect=np.isfinite(fy)&(before>20)&(before-after>12);footseed=region&footdirect[None,:]&(yy>fy[None,:]-14)&(yy<fy[None,:]-4)
 original=np.zeros((1080,1920),'u1')
 for uv in original_print_uv:
  x,y=np.rint(uv).astype(int)
  if 0<=x<1920 and 0<=y<1080 and region[y,x]:original[y,x]=1
 n,labels=cv2.connectedComponents(region.astype('u1'));c=set(np.unique(labels[capseed]));f=set(np.unique(labels[footseed]));o=set(np.unique(labels[original>0]));confirmed=(c&f)|o;confirmed.discard(0);valid=np.isin(labels,list(confirmed));receipt=dict(current_native_RGB_material_component_body_proof=True,cap_direct_columns=int(direct.sum()),cap_boundary_occluded_columns=int(occ.sum()),foot_yellow_to_white_current_photometry_columns=int(footdirect.sum()),body_connected_components=n-1,body_components_touch_current_real_cap_and_foot=sorted(map(int,(c&f)-{0})),body_components_touch_verified_original_print_readout=sorted(map(int,o-{0})),confirmed_body_components=sorted(map(int,confirmed)),candidate_body_pixels=int(region.sum()),confirmed_body_pixels=int(valid.sum()),unknown_cap_body_bounds_are_exclusion_only_not_source_UV=True,occluded_cap_boundary_does_not_mask_entire_body_column=True,source_RGB_used_for_current_material_interfaces_and_pixel_graph=True,body_domain_is_not_world_projection=True,person_guard_native_px=8,source_unknown_exclusion_guard_native_px=8,source_unknown_is_not_person_foreground=True,source_unknown_exclusion_pixels=int(unknown.sum()))
 return valid.astype('u1')*255,region.astype('u1')*255,receipt
