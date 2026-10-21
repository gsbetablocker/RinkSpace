import cv2,numpy as np

def qualified_direct_body(board,qualified,person,unknown,graphics):
 top=board['roles']['board.top'];foot=board['roles']['board.contact'];band=top['material_band'];cap=np.asarray(band['full_blue_cap_lower_y'],float);direct=np.asarray(band['full_blue_cap_lower_direct'],bool);rad=np.asarray([np.nan if x is None else x for x in band['full_blue_cap_lower_localization_radius']],float);occ=np.asarray(band['full_material_occlusion'],bool);direct&=np.isfinite(cap+rad)&~occ;fy=np.asarray(foot['full_native_y'],float);axis=np.arange(1920);left=np.maximum.accumulate(np.where(direct,axis,-1));right=np.minimum.accumulate(np.where(direct,axis,1920)[::-1])[::-1];bound=cap.copy();proof=np.zeros(1920,bool)
 for x in range(1920):
  if direct[x]:bound[x]=cap[x]+rad[x]+1;proof[x]=True;continue
  n=[i for i in(left[x],right[x])if 0<=i<1920]
  if n:bound[x]=max([cap[x]+1]+[cap[i]+rad[i]+1 for i in n]);proof[x]=True
  else:bound[x]=fy[x]
 yy=np.arange(1080)[:,None];region=(yy>bound[None,:])&(yy<fy[None,:]-4)&np.isfinite(bound+fy)[None,:]&proof[None,:];bad=cv2.dilate(person.astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0;bad|=cv2.dilate(unknown.astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0;bad|=graphics;region&=~bad
 capseed=region&direct[None,:]&(yy>cap[None,:]+rad[None,:]+2)&(yy<cap[None,:]+rad[None,:]+10)
 visible=np.zeros(1920,bool)
 for sp in qualified['roles']['board.contact']['spans']:
  a,z=map(int,sp['x_range']);visible[a:z+1]=True
 footdirect=np.asarray(foot['full_direct'],bool)&visible;footseed=region&footdirect[None,:]&(yy>fy[None,:]-14)&(yy<fy[None,:]-4);n,labels=cv2.connectedComponents(region.astype('u1'));c=set(np.unique(labels[capseed]));f=set(np.unique(labels[footseed]));confirmed=(c&f)-{0};valid=np.isin(labels,list(confirmed))
 return valid.astype('u1')*255,dict(actual_current_source_qualified_contact_direct_columns=int(footdirect.sum()),actual_current_cap_direct_columns=int(direct.sum()),body_components_touch_current_real_cap_and_verified_qualified_direct_contact=sorted(map(int,confirmed)),current_RGB_direct_support_from_frozen_board_qualified_not_private_yellow_threshold=True,board_contact_weak_or_occluded_not_promoted_to_direct=True,current_final_person_unknown_and_UI_full_profile_domain=True,source_UV_not_issued_by_this_domain_classification=True)
