"""Actual source-anchor UV for a shared frozen-shape print-model candidate.

All UV, sigma, exposure, and FIT/HELD roles are inherited. Normalized material
coordinates are a q0/source-derived geometry initializer, never measured XYZ.
The frozen v1, original absolute packet, q0, and runtime remain untouched.
"""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import hashlib,json,os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from scripts.diagnostics.verify_current_native_primitives_20261007 import state_camera_adapter
from rinkspace.vision_camera_initialization import current_complete_source_geometry

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT=BASE/'diagnostics/irregular_six_ads_source_revisit_20261007'
PARENT=OUT/'native_board_tracks_plus_six_irregular_ads_source_FIT_HELD_v1.json'
STATE=BASE/'calibration/runs/solver_acceleration_20261003/okada_global_geometry_boardtracks_midtime_absolute_20261007/accepted_states/696fd93d920fe5ebd3590d707ed03196587e779a48f661fd9124df6c31b50351/complete_state_UNACCEPTED.npz'
AUX=BASE/'calibration/runs/solver_acceleration_20261003/user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json'
Q0_SHA='696fd93d920fe5ebd3590d707ed03196587e779a48f661fd9124df6c31b50351'
SIX={8,9,10,21,22,23}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(np.asarray(value,dtype=np.float64).tobytes()).hexdigest()
def pixel_key(actual,number,uv):
    return int(actual),int(number),np.asarray(uv,dtype=np.float64).tobytes()

def main():
    parent_bytes=PARENT.read_bytes();packet=json.loads(parent_bytes)
    arrays=dict(np.load(STATE,allow_pickle=False));q0=arrays['q'].copy()
    assert array_sha(q0)==Q0_SHA
    identity=json.loads((STATE.parent/'state_identity.json').read_text(encoding='utf8'))
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    provider,_,_=current_complete_source_geometry(STATE,AUX)
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor,current_far_board_points,board_material_coordinates,points_from_board_material
    from structured_programme_source_candidate.paired_programme_chart import WORLD_INDICES
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    original=NativeBoardTrackFactor(state,PARENT,len(q0));border=original._border(q0)
    course,_,_=original.geometry.print_course_binding(q0[WORLD_INDICES])
    unit=float(course.scene_unit_m)
    def camera(native):
        actual=int(state.original.aliases[int(native)]);block=state.bypts[actual]
        return block.bundle.camera(block.local(q0),actual)
    def material(native,uv):
        xyz=current_far_board_points(camera(native),np.asarray(uv,float),border)
        return board_material_coordinates(xyz,border)
    templates=[];template_for={};counterpart_maps={}
    for pair in packet['shared_print_templates']:
        canonical=int(pair['canonical_advertisement_id']);counterpart=int(pair['counterpart_advertisement_id'])
        canonical_item=next(x for x in packet['source_conditioned_print_contours']if x['advertisement_id']==canonical)
        fid0,perimeter=material(pair['canonical_anchor_pts'],[pair['canonical_fiducial_uv']])
        fid1,_=material(pair['counterpart_anchor_pts'],[pair['counterpart_fiducial_uv']])
        fid0,fid1=fid0[0],fid1[0]
        def unwrap(m):
            m=np.asarray(m,float).copy();m[:,0]=fid0[0]+(m[:,0]-fid0[0]+perimeter/2)%perimeter-perimeter/2;return m
        contours=[]
        for contour in canonical_item['external_contours']:
            m,_=material(pair['canonical_anchor_pts'],contour['uv'])
            contours.append((deepcopy(contour),unwrap(m)))
        all_material=np.concatenate([m for _,m in contours]);lo=all_material.min(axis=0);size=all_material.max(axis=0)-lo
        assert np.all(size>0)
        fid_norm=(fid0-lo)/size
        counter_datum=fid1-fid_norm*size
        for contour,m in contours:contour['initialization']=dict(material_coordinates_normalized=((m-lo)/size).tolist(),derived_not_source_measurement=True)
        fit_pair_rows=[r for r in packet['rows']if r.get('shared_print_pair_id')==pair['id']and r['split']=='fit']
        u=np.concatenate([np.asarray(r['uv0'],float)for r in fit_pair_rows]);v=np.concatenate([np.asarray(r['uv1'],float)for r in fit_pair_rows])
        H,_=cv2.findHomography(v,u,0)
        if H is None or len(u)<4:raise ValueError('Actual cross-instance FIT points cannot initialize the canonical material')
        exact={np.asarray(b,dtype=np.float64).tobytes():a for a,b in zip(u,v)}
        mapping=dict(matrix=H.tolist(),source_number=counterpart,target_number=canonical,
            source_native_pts=pair['counterpart_anchor_pts'],target_native_pts=pair['canonical_anchor_pts'],
            input_cross_instance_FIT_row_ids=[r['id']for r in fit_pair_rows],FIT_points=len(u),
            mapped_canonical_UV_is_geometry_initializer_not_new_source_measurement=True,
            HELD_points_not_used_to_compute_H=True,source_only_projective_mapping_not_world_ground_truth=True)
        init=dict(q0_sha256=Q0_SHA,state=dict(path=str(STATE),sha256=sha(STATE)),
            camera_archive=dict(path=identity['camera_path'],sha256=sha(identity['camera_path'])),
            conditional_working_metres_per_scene_unit=unit,metric_scale_is_not_independently_verified=True,
            canonical_bbox_station_z_m=[*(lo*unit).tolist(),*(size*unit).tolist()],
            canonical_fiducial_material_coordinates_normalized=fid_norm.tolist(),
            instance_datum_station_zmin_m={str(canonical):(lo*unit).tolist(),str(counterpart):(counter_datum*unit).tolist()},
            shared_width_height_m=(size*unit).tolist(),source_only_counterpart_to_canonical_H=mapping,
            frozen_normalized_shape_is_source_and_q0_derived_not_measured_world=True,
            source_fiducials_remain_initialization_only_excluded_from_FIT_HELD=True,
            source_H_is_an_approximate_initialization_not_a_hard_world_correspondence_prior=True)
        template=dict(id=pair['id'],canonical_advertisement_number=canonical,counterpart_advertisement_number=counterpart,
            shared_dimensions_authority=deepcopy(pair['shared_dimensions_authority']),
            external_contours=[c for c,_ in contours],initialization=init,
            geometry_binding='frozen_normalized_source_shape_shared_W_H_independent_instance_station_zmin.v1')
        templates.append(template)
        for number in(canonical,counterpart):template_for[number]=(template,lo.copy(),size.copy(),fid0.copy(),perimeter)
        counterpart_maps[counterpart]=(H,exact)
    def coordinates(row):
        number=int(row['advertisement_id']);template,lo,size,fid,perimeter=template_for[number]
        raw=np.asarray(row['uv0'],float)
        if number==template['canonical_advertisement_number']:
            mapped=raw.copy();meaning=['actual canonical source UV']*len(raw)
        else:
            H,exact=counterpart_maps[number]
            mapped=cv2.perspectiveTransform(raw[None],H)[0];meaning=[]
            for i,uv in enumerate(raw):
                actual=exact.get(np.asarray(uv,dtype=np.float64).tobytes())
                if actual is not None:mapped[i]=actual;meaning.append('same actual matched canonical FIT pixel')
                else:meaning.append('source-only FIT H inferred canonical geometry pixel; not a source observation')
        m,_=material(next(p['canonical_anchor_pts']for p in packet['shared_print_templates']if p['id']==template['id']),mapped)
        m[:,0]=fid[0]+(m[:,0]-fid[0]+perimeter/2)%perimeter-perimeter/2
        return dict(template_id=template['id'],q0_sha256=Q0_SHA,material_coordinates_normalized=((m-lo)/size).tolist(),
            canonical_geometry_UV=mapped.tolist(),canonical_geometry_UV_role=meaning,
            all_material_coordinates_derived_initialization_not_source_measurement=True)
    # Original target endpoints already have residuals. Do not create another
    # source-anchor residual for an identical measured printed-material pixel.
    already_target={}
    for row in packet['rows']:
        if row['split']!='fit' or row.get('advertisement_id')not in SIX:continue
        number=int(row.get('counterpart_advertisement_id',row['advertisement_id']))
        actual=state.original.aliases[int(row['target_pts'])]
        for index,uv in enumerate(row['uv1']):already_target[pixel_key(actual,number,uv)]=(row['id'],index)
    groups=defaultdict(list);seen_fit={};seen_held={};dedup=[];old_new=[]
    explicit=[r for r in packet['rows']if r.get('advertisement_id')in SIX]
    for row in explicit:row['initialization']=coordinates(row)
    for split in('fit','holdout'):
        for row in explicit:
            if row['split']!=split:continue
            number=int(row['advertisement_id']);native=int(row['source_pts']);actual=int(state.original.aliases[native])
            sigma=np.broadcast_to(np.asarray(row['sigma_px'],float),(len(row['uv0']),2))
            for index,uv in enumerate(row['uv0']):
                key=pixel_key(actual,number,uv);previous=already_target.get(key)
                if previous is None:previous=seen_fit.get(key)
                if previous is None and split=='holdout':previous=seen_held.get(key)
                if previous is not None:
                    dedup.append(dict(row_id=row['id'],point_index=index,split=split,already_scored_or_selected=previous,
                        original_target_row_and_source_role_retained=True));continue
                (seen_fit if split=='fit'else seen_held)[key]=(row['id'],index)
                groups[(split,number,native,actual,tuple(sigma[index]))].append((row,index))
    anchors=[]
    for (split,number,native,actual,sigma),members in groups.items():
        first=members[0][0]
        anchors.append(dict(id=f'Ad{number}_actual_source_anchor_{native}_{split}',split=split,
            advertisement_number=number,pts=native,native_pts=native,actual_pts=actual,time_base='1/15360',
            source_raw_frame=deepcopy(first['source_frame']),uv=[r['uv0'][i]for r,i in members],sigma_px=list(sigma),
            original_source_point_ledger=[dict(row_id=r['id'],source_field='uv0',point_index=i,original_split=r['split'])for r,i in members],
            source_role='actual printed-material point at original source exposure',
            source_UV_sigma_and_split_inherited_without_pixel_rewrite=True,
            initialization=dict(template_id=first['initialization']['template_id'],q0_sha256=Q0_SHA,
                material_coordinates_normalized=[r['initialization']['material_coordinates_normalized'][i]for r,i in members],
                derived_material_coordinates_are_not_world_observations=True),
            holdout_role='development source holdout; pre-split source screening and shape review occurred'if split=='holdout'else None))
    for row in explicit:
        template,_,size,_,_=template_for[int(row['advertisement_id'])]
        datum=np.asarray(template['initialization']['instance_datum_station_zmin_m'][str(row['advertisement_id'])])/unit
        m=datum+np.asarray(row['initialization']['material_coordinates_normalized'])*size
        pair=original.shared_print_templates.get(str(row.get('shared_print_pair_id')))
        if pair is not None:
            counterpart=pair['counterpart_advertisement_id']
            datum=np.asarray(template['initialization']['instance_datum_station_zmin_m'][str(counterpart)])/unit
            m=datum+np.asarray(row['initialization']['material_coordinates_normalized'])*size
        new_world=points_from_board_material(m,border)
        new_prediction,_,good=project_central_valid_full_brown(camera(row['target_pts']),new_world)
        assert np.all(good)
        old_prediction=original._evaluate(q0,next(r for r in original.rows+original.held if r['id']==row['id']))[1]
        old_errors=old_prediction-np.asarray(row['uv1']);new_errors=new_prediction-np.asarray(row['uv1'])
        old_new.append(dict(id=row['id'],split=row['split'],points=len(old_errors),
            old_q0_native_error_median=float(np.median(np.linalg.norm(old_errors,axis=1))),
            new_q0_native_error_median=float(np.median(np.linalg.norm(new_errors,axis=1))),
            prediction_change_max_px=float(np.linalg.norm(old_prediction-new_prediction,axis=1).max()),
            original_UV_sigma_clock_split_unchanged=True,old_dynamic_source_conditioned_shape_operator_replaced=True))
    packet['parent_source_packet']=dict(path=str(PARENT),sha256=sha(PARENT))
    packet['parent_source_admission']=packet.pop('source_admission')
    packet['frozen_source_print_templates']=templates
    packet['source_anchor_observations']=anchors
    packet['source_anchor_contract']=dict(schema='rinkspace.actual-source-print-anchor-observations.v1',
        q0_sha256=Q0_SHA,initializer_is_not_source_WORLD_or_new_pixel_measurement=True,
        anchor_source='Original explicit-six FIT/HELD uv0 endpoints only',
        original_rows_UV_sigma_time_and_FIT_HELD_preserved=True,
        source_anchor_residual_family_requires_same_current_material_producer_as_target_and_renderer=True,
        target_endpoint_already_scored_and_same_actual_pixel_anchor_development_duplicates_not_added=True,
        no_original_source_row_or_HELD_target_removed=True,excluded_initialization_fiducials_not_admitted=True,
        deduplication_key='actual exposure + physical advertisement number + exact float64 UV bytes',
        deduplicated_original_source_point_ledger=dedup,
        source_anchor_FIT_points=sum(len(r['uv'])for r in anchors if r['split']=='fit'),
        source_anchor_development_HELD_points=sum(len(r['uv'])for r in anchors if r['split']=='holdout'),
        HELD_not_promoted_to_FIT=True,source_only_H_fit_rows_only=True)
    packet['review_pending_before_production_admission']=True
    for old,new in zip(json.loads(parent_bytes)['rows'],packet['rows']):
        original_fields={k:v for k,v in new.items()if k!='initialization'}
        assert old==original_fields,'Original source row field changed'
    path=OUT/'native_board_tracks_six_frozen_normalized_shapes_source_anchors_v2_CANDIDATE.json'
    path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report=dict(schema='rinkspace.six-print-frozen-shape-q0-source-contract-review.v1',
        packet=dict(path=str(path),sha256=sha(path)),q0_sha256=Q0_SHA,camera_archive_error=camera_error,
        original_v1_unchanged=True,source_anchor_contract=packet['source_anchor_contract'],
        q0_target_operator_before_after=old_new,no_runtime_model_changed=True,no_solver_started=True,
        final_runtime_complete_q_FD_and_same_producer_render_review_pending=True)
    rp=OUT/'six_frozen_normalized_shapes_source_anchor_q0_review_20261008.json'
    rp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert PARENT.read_bytes()==parent_bytes
    print(json.dumps(dict(packet=str(path),sha256=sha(path),review=str(rp),anchor_fit=packet['source_anchor_contract']['source_anchor_FIT_points'],anchor_development_held=packet['source_anchor_contract']['source_anchor_development_HELD_points'])))

if __name__=='__main__':main()
