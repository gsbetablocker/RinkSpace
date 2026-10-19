"""Only the five already admitted Ad14 corner-interval exposures.

Raw native identity/support qualifies observations, not conditional old WORLD.
The historical image-to-world role assignment is preserved explicitly; it is
not silently inferred from a candidate projection or matching brand.
"""
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
from functools import lru_cache
import pickle,json,hashlib
import numpy as np
B=_runtime_data_root()
PACKET=B/'shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
PACKET_SHA='ca5179e55bb1fd92b48e232ab1367ba583e41b1181b9849fa92d3df795942f78'
AUTHORITY=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/low5_repair/secondary6_Ad14_native_source_v1/secondary6_Ad14_native_corners_QUALIFIED_FROZEN.json')
AUTHORITY_SHA='34f60c5fbd7c7e3650f4b0b152c47c43575b3bdbd7b7ad12ba027c748371e798'
ROLES=('left_top','right_top','right_bottom','left_bottom')
IMAGE=('TL','TR','BR','BL')
SCHEMA='rinkspace.original-five-Ad14-native-corner-interval-assigned-role-certificate.v1'
@lru_cache(maxsize=1)
def _original():
    raw=PACKET.read_bytes();authority_raw=AUTHORITY.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PACKET_SHA or hashlib.sha256(authority_raw).hexdigest()!=AUTHORITY_SHA:raise ValueError('Original Ad14 source authority or immutable programme packet changed')
    data=pickle.loads(raw);authority=json.loads(authority_raw)
    if authority['physical_ad_id']!=14 or authority['identity_authority']['kind']!='explicit_user_confirmation' or authority['source_only_worldbinding_unassigned']is not True:raise ValueError('Original user source identity and unassigned-WORLD authority required')
    rows={r['source_native_pts']:{} for r in authority['source_rows']}
    for point in authority['source_rows']:rows[point['source_native_pts']][point['image_corner_clockwise']]=point
    originals={}
    for item in data['programme_absolute_rows']:
        r=item['source_row']
        if r.get('physical_ad_id')!=14:continue
        pts=int(item['group_actual_pts'])
        if r['id'] in originals:raise ValueError('Ambiguous original Ad14 assigned source row')
        if tuple(r['physical_world_corner_roles'])!=ROLES or r['image_corner_to_world_role']!=dict(zip(IMAGE,ROLES)):raise ValueError('Original admitted Ad14 assigned role mapping changed')
        originals[r['id']]=(r,pts,item['collection_role'])
    if len(originals)!=5 or len(rows)!=5:raise ValueError('Exactly four FIT and one independent HELD original Ad14 source exposures required')
    return originals,rows,authority

def qualify_ad14_assigned_roles(row,*,actual_pts=None):
    originals,raw,authority=_original();item=originals.get(row.get('id'))
    if item is None:raise ValueError('No authority for arbitrary Ad14 source row; only the five original admitted exposures')
    original,pts,collection=item
    if actual_pts is not None and int(actual_pts)!=pts:raise ValueError('Original qualified Ad14 actual exposure changed')
    for key in('physical_ad_id','point_correspondence','curve_observation_kind','absolute_support_volume_quadratic','pts','split','physical_world_corner_roles','image_corner_to_world_role'):
        if row.get(key)!=original.get(key):raise ValueError('Original Ad14 source identity/role/operator changed: '+key)
    for key in('uv','sigma','point_support_bounds_xy_px','source_native_pts'):
        if not np.array_equal(np.asarray(row[key]),np.asarray(original[key])):raise ValueError('Original Ad14 native support or engineering sigma changed: '+key)
    if np.shape(row['world'])!=(4,3) or not np.isfinite(np.asarray(row['world'],float)).all():raise ValueError('Four finite assigned model corners required')
    for j,image in enumerate(IMAGE):
        point=raw[pts][image]
        if point['split']!=original['split'] or point['physical_ad_id']!=14 or point['source_actual_pts']!=pts or not point['corner_manually_qualified'] or not point['source_support_manually_qualified'] or point['model_projection_used_to_generate_or_select_uv'] or point['predicted_corner_or_projected_ROI_used']:raise ValueError('Original source-only Ad14 pixel authority differs')
        if not np.array_equal(row['uv'][j],point['uv']) or not np.array_equal(row['point_support_bounds_xy_px'][j],point['native_support_bounds_xy_px']) or not np.array_equal(np.repeat(row['sigma'][j],2),point['sigma_native_px']):raise ValueError('Ad14 source point must match original native authority exactly')
    return np.array([0.,1.,1.,0.]),np.array([1.,1.,0.,0.]),dict(schema=SCHEMA,source_id=row['id'],actual_pts=pts,original_role=collection,physical_ad_id=14,
        source_authority_path=str(AUTHORITY),source_authority_sha256=AUTHORITY_SHA,original_source_packet_sha256=PACKET_SHA,
        original_source_World_sha256=hashlib.sha256(np.asarray(original['world'],float).tobytes()).hexdigest(),current_input_World_sha256=hashlib.sha256(np.asarray(row['world'],float).tobytes()).hexdigest(),
        world_literal_is_conditional_model_not_measured_material=True,world_literal_membership_used_to_reject_source_identity=False,
        image_corner_to_world_role=dict(zip(IMAGE,ROLES)),assigned_world_roles_authority='original immutable admitted programme source integration; raw source clockwise labels alone do not assign WORLD',
        original_native_UV_sigma_support_operator_and_FIT_HELD_exact=True,source_identity_authority=authority['identity_authority'],no_new_pixel_no_candidate_projection_no_generic_brand_admission=True)

def main():
    import copy
    originals,_,_=_original();accepted=[];negative=[]
    for original,pts,_ in originals.values():
        _,_,certificate=qualify_ad14_assigned_roles(original,actual_pts=pts);accepted.append(certificate)
    row=copy.deepcopy(next(iter(originals.values()))[0]);row['world']=np.asarray(row['world'])+np.array([.2,-.1,.03]);qualify_ad14_assigned_roles(row)
    for field in('uv','sigma','split','id','image_corner_to_world_role'):
        bad=copy.deepcopy(next(iter(originals.values()))[0])
        if field in('uv','sigma'):bad[field]=np.asarray(bad[field],float)+.01
        elif field=='image_corner_to_world_role':bad[field]=dict(bad[field],TL='right_top')
        else:bad[field]='unqualified-change'
        try:qualify_ad14_assigned_roles(bad)
        except ValueError:negative.append(field)
        else:raise RuntimeError('Illegal Ad14 identity/support mutation admitted')
    result=dict(schema=SCHEMA+'.actual-small-audit',status='PASS',five_original_rows=accepted,conditional_world_change_admitted_without_pixel_change=True,rejected_fields=negative,physical_world_roles_not_new_measurement=True)
    (B/'structured_programme_source_candidate/actual_ad14_source_role_certificate_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(dict(status='PASS',rows=len(accepted),rejected=negative)),flush=True)
if __name__=='__main__':main()
