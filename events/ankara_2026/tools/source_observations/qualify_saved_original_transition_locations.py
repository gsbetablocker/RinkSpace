"""Qualify saved original FIT transition locations; no pixel/model evaluation.

Each scalar location and observed UV is independently addressable in an
immutable pre-existing measurement JSON.  This adds no observations, changes
no UV/sigma/support and never promotes HELD or interprets a support midpoint
as a measured location.  Existing correction records are excluded wholesale.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
HISTORY = DATA / 'history/work_sessions/20261002_01a0fd26-33ba-7873-9aa9-bbd9d2ca1328'
H = DATA / 'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
ORIGINAL = DATA / 'calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
ORIGINAL_SHA256 = 'ca5179e55bb1fd92b48e232ab1367ba583e41b1181b9849fa92d3df795942f78'
CORRECTION = H / 'source_pixel_review_additional/board_top_contact_material_semantics/original_TOP_RGB_measured_anchor_revision/original_ice_and_TOP_actual_RGB_corrected_source_WHOLE_SOURCE_REVIEW36_20261011.json'
PACKET = ROOT / 'events/ankara_2026/observations/source_packets/original_saved_transition_locations_20261011/packet.json'
RECEIPT = H / 'source_pixel_review_additional/original_saved_transition_locations/qualification_receipt_20261011.json'
SCHEMA = 'rinkspace.original-saved-transition-location.v1'

# Only exact original UV=q50 formats with existing physical-source admission.
# The index list identifies source facts, never residual magnitude or outcome.
CLASSES = (
    dict(indices=(5593,5594,5595,5596), source='low5_repair/ad20_source_intervals_v1/source_rows.json',
         sha='4fcdefcad362aa536a8f3b8a78185acef09c903d74b5e5f36f615f9fb45ef1de',
         kind='blue_chroma_half_plateau_q50', quantile='q10_q50_q90_native',
         producers=('low5_repair/ad20_interval_source.py',)),
    dict(indices=(5620,5621,5629,5630), source='low5_repair/F2_inner_source_intervals_v2/source_rows.json',
         sha='6038b88d25527fb93aec7c7753e7560b3a619c00a3864d929cb99f16c8d6f99c',
         kind='axis_intensity_or_chroma_half_plateau_q50', quantile='q10_q50_q90_native',
         producers=('low5_repair/F2_inner_source.py',)),
    dict(indices=(5589,5591), source='low5_repair/user_major_F1_Lane1_source_v2/source_rows.json',
         sha='7e56b85f8c93f7c806c1e7a8db23f6ddc7aa757eca39a9bd30572a9ffea5216c',
         kind='gray_outer_ring_half_plateau_q50', quantile='q10_q50_q90_native',
         producers=('low5_repair/joint_variable_source_audit_v1/major_source_packet_v2.py',)),
    dict(indices=(5475,5476,5477,5478,5479), source='ice_alignment_review/ad7_source_boundaries_qualified.json',
         sha='9cf126825314962e65ac9885cc3ef704eeb15c43a8197810d3b07a858a02a2fb',
         kind='signed_gray_falling_gradient_mass_q50', quantile='source_transition_q10_q50_q90',
         producers=('ice_alignment_review/read_ad7_late_source.py','ice_alignment_review/qualify_ad7_source_boundary.py'), nested=True),
    dict(indices=(6680,6681,6682,6683), source='ice_alignment_review/main9_ice_source/main9_R3_R4_external_boundary_packet_qualified.json',
         sha='3ebda4a39392ec6ecc2d4bf4e5c0e2d1db919dc91a89a83370efa61e9d41f1de',
         kind='signed_outer_paint_gradient_mass_q50', quantile='q10_q50_q90_y',
         producers=('ice_alignment_review/main9_ice_source/read_R3_R4_outer_source.py',)),
    dict(indices=(6684,6685), source='main9_board_ice_key_source_v1/core_key_visible_board_edges_source.json',
         sha='8f10051947755b546a7313e11fa5ae1b82f4e90d88f422b67375a3b9634016a4',
         kind='signed_print_border_gradient_mass_q50', quantile='source_transition_q10_q50_q90', producers=()),
    dict(indices=(1997,), source='ice_alignment_review/main1_ice_source/main1_F3_external_boundary_packet_qualified.json',
         sha='e88744ae926813d0e491f975ae380ebc1fed2dae196837c05ed2ec4844e8ff8d',
         kind='signed_gray_outer_boundary_gradient_mass_q50', quantile='q10_q50_q90_y',
         producers=('ice_alignment_review/main1_ice_source/finalize_f3_source_packet.py',)),
    dict(indices=(1998,2000), source='ice_alignment_review/main1_ice_source/main1_F3_extent_Ad7_source_packet_qualified.json',
         sha='758c96a024a8c8d09d5bf92946d5d6bd8cf498ab242bd4023888e55ffe2e4175',
         kind='signed_external_boundary_gradient_mass_q50', quantile='q10_q50_q90_coordinate',
         producers=('ice_alignment_review/main1_ice_source/read_f3_extent_ad7_source.py',)),
    dict(indices=(5159,5160,5161,5167,5168,5169,5170,5187,5188,5189,5190),
         source='source_solver_audit/secondary_core_ice_current_review_v1/new_cut2_ring_lane_full_source_gradient_qualified_v2.json',
         sha='7faaf672bcfd1ad41ebf4221318c893ae7041b4971a1f7df71a095097494ec0e',
         kind='full_signed_outer_paint_gradient_mass_q50', quantile='q10_q50_q90_full_source_gradient_axis',
         producers=('source_solver_audit/qualify_full_source_support_packets.py',), nested=True),
)
INDICES = tuple(sorted(index for item in CLASSES for index in item['indices']))


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v)for v in value]
    return value


def canonical(value):
    return json.dumps(plain(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


def reference(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition,message):
    if not condition:raise ValueError(message)


def walk(value,path=()):
    if isinstance(value,dict):
        yield list(path),value
        for key,item in value.items():yield from walk(item,(*path,key))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from walk(item,(*path,index))


def at_path(value,path):
    for key in path:value=value[key]
    return value


def source_points(source,row,item,definition):
    """Return source paths to each literal observed UV and original estimator."""
    points=[]
    if definition.get('nested'):
        at=int(item['group_actual_pts'])
        records=[(i,r)for i,r in enumerate(source['records'])
                 if int(r.get('canonical_pts',r['source_pts']))==at]
        physical=row.get('physical_curve_id')
        ids=row.get('physical_ids',[])
        if physical is None and len(ids)==1 and isinstance(ids[0],str):physical=ids[0]
        if physical:records=[(i,r)for i,r in records if r.get('physical_id')==physical]
        require(len(records)==1,'One authenticated source exposure/material record required')
        record_index,record=records[0]
        key='source_qualified_rows'if 'source_qualified_rows'in record else'rows'
        for uv,normal in zip(row['uv'],row['normal']):
            matches=[(i,p)for i,p in enumerate(record[key])
                     if np.array_equal(p['source_uv'],uv)and np.array_equal(p['source_normal_uv'],normal)]
            require(len(matches)==1,'Source UV/normal must have unique admitted measurement')
            index,point=matches[0]
            path=['records',record_index,key,index]
            points.append((path,point,path+['source_uv']))
    else:
        def measured_uv(value):
            if 'uv'in value:return value['uv']
            if 'samples'in value and all('source_uv'in p for p in value['samples']):
                return [p['source_uv']for p in value['samples']]
            return None
        def contains_original(value):
            source_uv=measured_uv(value)
            if source_uv is None:return False
            source_uv=np.asarray(source_uv,float)
            return source_uv.ndim==2 and source_uv.shape[1:]==(2,)and all(
                int(np.all(source_uv==uv,axis=1).sum())==1 for uv in np.asarray(row['uv'],float))
        candidates=[(path,v)for path,v in walk(source)
                    if v.get('id')==row['id']and contains_original(v)
                    and int(v.get('pts',-1))==int(row['pts'])]
        require(len(candidates)==1,'One exact original source row is required')
        path,source_row=candidates[0]
        require(source_row.get('split')=='fit','Independent source row must retain FIT role')
        key='source_profiles'if 'source_profiles'in source_row else'samples'
        all_uv=np.asarray(measured_uv(source_row),float)
        require(len(source_row[key])==len(all_uv),'Original source profiles and source UV count differ')
        for uv in row['uv']:
            matches=np.flatnonzero(np.all(all_uv==uv,axis=1))
            require(len(matches)==1,'Unique literal original UV/profile correspondence required')
            index=int(matches[0]);point=source_row[key][index]
            observed=path+[key,index,'source_uv']if 'source_uv'in point else path+['uv',index]
            points.append((path+[key,index],point,observed))
    return points


def qualify_row(index,item,definition,source,source_ref,producer_refs):
    row=item['source_row'];uv=np.asarray(row['uv'],float);normal=np.asarray(row['normal'],float)
    require(item['collection_role']=='FIT'and row.get('split')=='fit','HELD/development-role promotion forbidden')
    require(uv.ndim==2 and uv.shape[1:]==(2,)and normal.shape==uv.shape and np.isfinite(uv).all(),
            'Finite original measured UVs/normals required')
    require('normal_interval'in row,'Existing optical interval operator required')
    points=source_points(source,row,item,definition);bindings=[];measurements=[]
    for p,(path,point,observed_path)in enumerate(points):
        require(normal[p].tolist()in ([1.,0.],[0.,1.]),'Only explicitly axis-measured formats are certified')
        axis=int(np.argmax(normal[p]));quantile=definition['quantile'];q=np.asarray(point[quantile],float)
        if 'source_normal_uv'in point:
            require(np.array_equal(point['source_normal_uv'],normal[p]),'Source measured axis and original normal differ')
        if 'axis'in point or 'quantile_axis'in point:
            require(point.get('axis',point.get('quantile_axis'))==('x'if axis==0 else'y'),
                    'Source scan axis and original normal differ')
        require(q.shape==(3,)and np.isfinite(q).all(),'Original three estimator quantiles required')
        location_path=path+[quantile,1];fixed_path=observed_path+[1-axis]
        observed=at_path(source,observed_path)
        require(np.array_equal(observed,uv[p]),'Source receipt UV differs from original packet')
        require(float(at_path(source,location_path))==float(uv[p,axis]),'Original UV is not the saved estimator location')
        require(float(at_path(source,fixed_path))==float(uv[p,1-axis]),'Original scan fixed coordinate changed')
        for flag in ('source_selection_uses_model_or_camera','source_measurement_uses_camera',
                     'source_pixel_measurement_uses_camera','physical_material_point_correspondence'):
            require(point.get(flag,False)is False,'Model-selected/material-point-only reading is ineligible')
        bindings.append(dict(point_index=p,measured_axis=axis,observed_uv_path=observed_path,
            location_path=location_path,fixed_coordinate_path=fixed_path))
        measurements.append(dict(point_index=p,observed_uv=plain(observed),measured_axis=axis,
            location=float(q[1]),q10_q50_q90=plain(q),source_measurement_path=path,
            source_measurement_canonical_json_sha256=hashlib.sha256(canonical(point)).hexdigest()))
    require(len(bindings)==len(uv),'Every original source point needs independent location authority')
    # Whole rows without producer scripts still have an immutable saved
    # estimator receipt containing exact q50, UV and reviewed material identity.
    # Do not pretend the historical producer script was found for those rows.
    return dict(original_packet_row_index=index,
        locator={k:plain(item[k])for k in ('block','collection_role','group_actual_pts','group_order','row_order')},
        original_source_row_sha256=hashlib.sha256(canonical(row)).hexdigest(),source_row_id=row['id'],
        point_indices=list(range(len(uv))),source_reference=source_ref,producer_references=producer_refs,
        measured_uv=plain(uv),estimator_kind=definition['kind'],source_measurements=measurements,
        measurement_bindings=bindings,original_normal=plain(normal),
        original_optical_support_px=plain(row['normal_interval']),original_stored_sigma=plain(row['sigma']),
        source_operator_attributes={k:plain(row.get(k))for k in ('curve_observation_kind','scanline_fixed_axis',
            'scanline_branch','must_preserve_upper_branch','point_correspondence')},
        producer_script_available=bool(producer_refs),
        original_source_UV_sigma_support_roles_and_physical_identity_unchanged=True,
        soft_estimator_location_is_not_calibrated_noise_or_exact_latent_edge=True)


def save_new(path,value):
    path=Path(path);data=(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
    if path.exists():require(path.read_bytes()==data,'Frozen output already exists with different bytes: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def build(packet_path=PACKET,receipt_path=RECEIPT):
    original_ref=reference(ORIGINAL);require(original_ref['sha256']==ORIGINAL_SHA256,'Original absolute packet changed')
    source_rows=pickle.loads(ORIGINAL.read_bytes())['programme_absolute_rows']
    correction=json.loads(CORRECTION.read_text(encoding='utf8'));correction_indices={r['original_packet_row_index']for r in correction['records']}
    require(len(set(INDICES))==len(INDICES),'Source classes overlap')
    records=[];source_refs=[]
    for definition in CLASSES:
        path=HISTORY/definition['source'];source_ref=reference(path)
        require(source_ref['sha256']==definition['sha'],'Original saved estimator receipt changed')
        source=json.loads(path.read_text(encoding='utf8'));source_refs.append(source_ref)
        producers=[dict(**reference(HISTORY/p),role='original_source_estimator_producer')for p in definition['producers']]
        for index in definition['indices']:
            require(index not in correction_indices,'Current correction overlaps this source estimator')
            records.append(qualify_row(index,source_rows[index],definition,source,source_ref,producers))
    records.sort(key=lambda r:r['original_packet_row_index'])
    require(len(records)==35 and sum(len(r['point_indices'])for r in records)==177,'Frozen source-only scope changed')
    exclusions=dict(current_original_correction_records='all current correction row indices excluded, including already-corrected paint centres and TOP measurements',
        support_union_extent_indices=[1999,5162,5309,5322,5624,5625,5633,5634],
        unchanged_old_UV_not_receipt_q50_indices=[5176,5177,5178,5179],
        half_plateau_line_intersection_not_direct_q50_indices=[2012],
        source_producer_or_independent_JSON_not_located_indices=[5637,5638,5639,5640,2882,2885],
        manual_or_target_region_support='original nearboard49 and secondary64 support declarations are not converted into exact positions',
        HELD='No HELD observation is revised or promoted')
    receipt=dict(schema=SCHEMA+'.qualification-receipt',original_absolute_packet=original_ref,
        reproducer_reference=reference(__file__),current_correction_reference=reference(CORRECTION),
        source_references=source_refs,records=[dict(original_packet_row_index=r['original_packet_row_index'],
            source_row_id=r['source_row_id'],points=len(r['point_indices']),estimator_kind=r['estimator_kind'],
            source_reference=r['source_reference'])for r in records],exclusions=exclusions,
        all_saved_estimator_locations_equal_original_UV_bitexact=True,
        qualification_based_on_original_receipts_and_admitted_physical_source_identity=True,
        no_model_residual_used_for_qualification=True,source_images_decoded=0,source_images_written=0,
        no_new_UV_normal_sigma_support_geometry_role_clock_or_FIT_rows=True,
        source_qualifications_are_soft_estimator_locations_not_accuracy_acceptance=True)
    save_new(receipt_path,receipt)
    packet=dict(schema=SCHEMA,original_absolute_packet=original_ref,receipt_reference=reference(receipt_path),
        records=records,all_original_observations_unchanged=True,
        current_correction_reference=reference(CORRECTION),source_reference_paths_are_preexisting_measurement_JSON=True,
        active_configuration_changed=False,accuracy_accepted=False)
    save_new(packet_path,packet)
    return dict(packet=reference(packet_path),receipt=reference(receipt_path),rows=len(records),
        points=sum(len(r['point_indices'])for r in records),estimator_kinds=dict(Counter(r['estimator_kind']for r in records)),
        source_images_decoded=0,source_images_written=0,new_J_or_fit=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,default=PACKET);parser.add_argument('--receipt',type=Path,default=RECEIPT)
    options=parser.parse_args();print(json.dumps(build(options.packet,options.receipt),ensure_ascii=False))


if __name__=='__main__':main()
