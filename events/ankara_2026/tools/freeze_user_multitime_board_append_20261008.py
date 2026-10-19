"""Finalize explicitly viewed source-only additions without altering parents."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from measure_user_multitime_board_edges_20261008 import OUT,PARENT,ref,read,write_new


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority',required=True);a=ap.parse_args();assert a.authority.strip()
    pending=OUT/'board_parent49_plus_Ad4_Ad5_multitime_actual_RGB_v2_SOURCE_REVIEW_PENDING.json'
    source=read(pending);parent=read(PARENT);rows=source['rows'][49:]
    assert len(parent['rows'])==49 and source['rows'][:49]==parent['rows'] and sum(len(r['uv'])for r in rows)==65
    qa_parent=OUT/'Ad4_Ad5_multitime_actual_native_edges_source_QA_v1.json';qa=read(qa_parent)
    assert qa['newFIT_points']==65 and qa['new_scalar_rows']==130
    images={}
    for r in rows:
        path=Path(r['source_raw_frame']);assert ref(path)['sha256']==r['source_file_sha256']
        if path not in images:
            image=cv2.imread(str(path));assert image.shape==(1080,1920,3)
            images[path]=hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        assert images[path]==r['source_native_RGB_sha256']
        assert r['uv']==[m['actual_source_uv']for m in r['measured_RGB_sources']]
        assert r['normal']==[m['source_search_normal']for m in r['measured_RGB_sources']]
        assert r['normal_interval_px']==[m['observed_normal_interval_px']for m in r['measured_RGB_sources']]
        assert r['sigma_px']==[max(4.,float(np.ptp(m['observed_normal_interval_px']))/2.)for m in r['measured_RGB_sources']]
        assert all(m['disposition']=='measured_material_transition'and not any(m['source_flags'].values())for m in r['measured_RGB_sources'])
        assert r['point_correspondence'] is False and r['curve_observation_kind']=='normal_interval' and r['split']=='fit'
    for pts in sorted({r['pts']for r in rows}):
        path=OUT/f'native_{pts}_actual_SOURCE_edges_review.png';assert path.exists()
    refined=OUT/'native_230320640_Ad5_left_full_blur_actual_SOURCE_review.png'
    qa.update(status='all_actual_source_RGB_and_native_marker_rows_visually_qualified',
        source_qualification_not_complete_until_full_blur_marker_view=False,
        human_actual_material_and_role_review=dict(reviewer='ad_evidence',
            all65_native_source_q50_points_and_q10_q90_support_markers_viewed=True,
            no_point_on_person_UI_ice_or_wrong_print=True,
            Ad4_ANKARA_white_skater_blue_JGP_group_external_left_right_scope=True,
            Ad5_centre_blue_white_green_graphic_only=True,
            refined_full_blur_left_marker=ref(refined),all_source_scanline_context_not_fixed_Z=True,
            blur_interval_widths_inherited_in_source_sigma=True,subpixel_covariance_accuracy_not_certified=True),
        authority=a.authority,parent_source_QA=ref(qa_parent),source_append_pack_not_installed_or_fitted=True)
    qa['source_native_RGB_marker_review']['full_blur_left_marker_review_pending']=False
    qpath=OUT/'Ad4_Ad5_multitime_actual_native_edges_source_QA_FINAL_v1.json';write_new(qpath,qa)
    packet=deepcopy(source);packet['source_append_user_multitime']['source_QA']=ref(qpath)
    packet.update(source_marker_review_pending_full_blur_left=False,
        source_native_RGB_and_material_roles_qualified=True,new_source_measurement_authority=a.authority,
        source_admission_QA=ref(qpath),source_measurement_candidate=ref(pending),solver_target_updated_here=False)
    assert packet['rows']==source['rows'] and packet['rows'][:49]==parent['rows']
    path=OUT/'board_parent49_plus_Ad4_Ad5_multitime_actual_RGB_ADMITTED_v1.json';write_new(path,packet)
    print(json.dumps(dict(packet=ref(path),source_QA=ref(qpath),parent_rows_exact=49,
        appended_rows=len(rows),newFITpoints=65,newFITscalar=130,newHELDpoints=0)),flush=True)


if __name__=='__main__':main()
