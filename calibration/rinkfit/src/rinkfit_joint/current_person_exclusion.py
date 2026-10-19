"""Same-exposure body/reflection exclusion shared by source and frozen QA."""
import hashlib
import numpy as np

def current_person_exclusion_mask(rgb,boxes,*,native_pts,graphics_mask=None,current_board_context=None):
 """Return (mask, receipt); contact pixels are from this exposure only.

 A frozen model may supply the explicit current object search window; its
 projected coordinates never become contact readings. No temporal pixel
 transport or fresh detector inference is performed here.

 Original same-exposure box bodies retain the four-native-pixel margin.
 Reflection columns require current material contacts bracketing that box;
 without object context or actual local contacts, only body/graphics masks
 are returned. Spectator boxes do not erase the board below them. The numerical margin is implementation
 conditioning, not an additional user fact or a calibrated mask uncertainty.
 """
 from .board_outer_source import source_outer_boundary_rows
 rgb=np.asarray(rgb);h,w=rgb.shape[:2];rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();mask=np.zeros((h,w),bool);verified=[]
 if graphics_mask is None:
  mask[:min(170,h),:min(620,w)]=True;mask[min(900,h):,min(1600,w):]=True
 else:
  g=np.asarray(graphics_mask,bool)
  if g.shape!=mask.shape:raise ValueError('Current graphics mask shape differs')
  mask|=g
 for box in boxes:
  binding=box.get('source_binding') or {}
  if binding.get('source_pts')!=int(native_pts) or binding.get('decoded_native_rgb_sha256')!=rgbsha:raise ValueError('Person box is not bound to this actual exposure')
  q=np.asarray(box['box_xyxy'],float);a=np.maximum([0,0],np.floor(q[:2]-4).astype(int));end=np.minimum([w,h],np.ceil(q[2:]+4).astype(int));mask[a[1]:end[1],a[0]:end[0]]=True;verified.append((q,a,end))
 if current_board_context is not None:
  if int(current_board_context['native_pts'])!=int(native_pts)or current_board_context['actual_native_RGB_sha256']!=rgbsha:raise ValueError('Exclusion context is not this actual exposure')
  initial,certificates=source_outer_boundary_rows(rgb,mask,current_board_context=current_board_context)
 else:initial,certificates=[],[]
 contacts=np.asarray([r['current_uv'] for r in initial if r['material_role']=='board.contact'],float).reshape(-1,2);extended=[];decisions=[];contact_y=float(np.median(contacts[:,1]))if len(contacts)else None
 for j,(q,a,end)in enumerate(verified):
  decision=dict(box_index=j,reflection_extended=False)
  # The global median is diagnostic only. Perspective can place a spectator
  # below that median while the actual board foot in their column is lower.
  left=contacts[(contacts[:,0]<a[0])&(contacts[:,0]>=a[0]-128)]if len(contacts)else contacts
  right=contacts[(contacts[:,0]>=end[0])&(contacts[:,0]<=end[0]+128)]if len(contacts)else contacts
  if len(left)and len(right):
   L=left[np.argmax(left[:,0])];R=right[np.argmin(right[:,0])];upper_envelope=float(max(L[1],R[1]));decision.update(actual_contact_bracket_uv=[L.tolist(),R.tolist()],foreground_contact_y_conservative=upper_envelope)
   if float(q[3])>upper_envelope+8:
    mask[end[1]:h,a[0]:end[0]]=True;extended.append(j);decision.update(reflection_extended=True,status='box_bottom_below_both_current_actual_contact_neighbors')
   else:decision['status']='current_local_contact_does_not_establish_ice_foreground'
  else:decision['status']='current_local_contact_bracket_unavailable; body_only_mask_preserved'
  decisions.append(decision)
 receipt=dict(schema='rinkfit.same_current_body_reflection_exclusion.v4',native_pts=int(native_pts),decoded_native_RGB_sha256=rgbsha,original_same_exposure_boxes=len(verified),body_margin_native_px=4,foreground_actual_contact_margin_native_px=8,actual_contact_source_rows=len(contacts),actual_contact_median_y=contact_y,actual_contact_median_is_not_foreground_threshold=True,actual_contact_reading_attempted=current_board_context is not None,actual_contact_status='actual_current_material_support'if len(contacts)else'context_not_supplied_no_actual_contact_reflection_extension'if current_board_context is None else'context_supplied_but_actual_contact_unavailable',foreground_local_contact_neighbor_limit_px=128,reflection_foreground_decisions=decisions,reflection_columns_extended_only_for_foreground_boxes=extended,spectator_columns_extended=False,model_or_other_exposure_used_to_create_contact_pixels=False,navigation_context_schema=current_board_context.get('schema') if current_board_context is not None else None,navigation_context_is_not_contact_measurement=True,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),source_operator_version=certificates[0]['configuration']['version'] if certificates else None,source_reflection_test='actual current box lower extent below this exposure measured physical yellow LOWER board-contact material',implementation_conditions_not_user_facts=True)
 return mask,receipt
