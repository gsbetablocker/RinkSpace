"""Small native RGB review of the largest stored absolute-source discrepancies.

No projection, solver or inferred correspondence is used in the source image.
"""
from pathlib import Path
import hashlib, json, pickle
import av, cv2, numpy as np

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
VIDEO = BASE/'media/originals/oxXu1avhrvs.mp4'
PACKET = BASE/'calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
OUT = BASE/'diagnostics/external_expert_20261007/all_original_absolute_source_audit/native_rgb_review'


def main():
    cv2.setNumThreads(1)
    rows = pickle.loads(PACKET.read_bytes())['programme_absolute_rows']
    selections = [
        (229217280, [1046], [1270, 275, 1730, 460]),
        (229624064, [5253, 5254, 5255, 5256], [0, 370, 1920, 760]),
        (229535232, [5499, 5500, 5501, 5502], [0, 220, 730, 430]),
        (230507264, [5554], [1450, 300, 1920, 590]),
        (229823744, [11347, 11348], [0, 370, 1920, 690]),
        (230199296, [5599, 5600], [0, 230, 1920, 465]),
        (230557184, [2885], [0, 980, 260, 1080]),
        (229702144, [10396, 10397, 10398], [0, 580, 1920, 1080]),
        (229290752, [2568], [650, 720, 1920, 1080]),
    ]
    # Locate the main1 source by its actual/native identity, not by a guessed index.
    selections[0] = (229217280, [i for i, r in enumerate(rows) if r['source_row']['id']=='known_ad_corners' and np.array_equal(r['source_row'].get('source_native_pts'),229217280)], selections[0][2])
    OUT.mkdir(parents=True, exist_ok=True)
    evidence=[]
    with av.open(str(VIDEO)) as c:
        stream = c.streams.video[0]; stream.thread_count=1
        for pts, indices, box in selections:
            c.seek(pts-15360, stream=stream, backward=True, any_frame=False)
            frame=next(f for f in c.decode(stream) if f.pts==pts)
            bgr=frame.to_ndarray(format='bgr24'); marked=bgr.copy()
            records=[]
            for index in indices:
                item=rows[index]; r=item['source_row']
                for p in np.asarray(r['uv']):
                    cv2.drawMarker(marked,tuple(np.rint(p).astype(int)),(0,0,255),cv2.MARKER_CROSS,9,1)
                records.append(dict(original_packet_row_index=index, actual_pts=item['group_actual_pts'], collection_role=item['collection_role'], id=r['id'], source_uv=np.asarray(r['uv']).tolist(), physical_ids=r.get('physical_ids'), source_origin=r.get('origin')))
            x0,y0,x1,y1=box
            raw=OUT/f'{pts}_source_RGB_crop.png'; overlay=OUT/f'{pts}_stored_source_UV_crop.png'
            cv2.imwrite(str(raw), bgr[y0:y1,x0:x1]); cv2.imwrite(str(overlay), marked[y0:y1,x0:x1])
            evidence.append(dict(decoded_native_pts=pts, decoded_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(), source_time_seconds=pts/15360, crop_xyxy=box, raw_crop=str(raw), stored_source_marker_crop=str(overlay), records=records))
    labels=BASE/'diagnostics/external_expert_20261007/source_label_probe/source_label_probe.json'
    contact=[]
    for source in json.loads(labels.read_text(encoding='utf-8'))['results']:
        candidates=[]
        for point in source['observations']:
            y=np.asarray(point['profile_y']);signal=np.asarray(point['yellow_contrast']);uv=point['original_uv']
            grad=np.gradient(signal);window=(y>=uv[1]-16)&(y<=uv[1]+16)
            yy=int(y[window][np.argmin(grad[window])])
            candidates.append(dict(index=point['index'],stored_native_uv=uv,raw_yellow_to_ice_transition_candidate_y=yy,candidate_minus_stored_y_px=yy-uv[1]))
        contact.append(dict(pts=source['pts'],decoded_bgr_sha256=source['decoded_bgr_sha256'],profiles_source=str(labels),candidates=candidates,RGB_review='Stored native points are on the lower visible yellow kickplate transition, not its upper white-to-yellow transition. The gradient candidates differ by 0 to -4 px; blur and the physical bottom envelope remain separate from gradient extrema.',automatic_gradient_is_not_ground_truth=True,original_sigma_px=7,source_pixel_correction_admitted=False))
    report=dict(schema='rinkspace.original-absolute-native-rgb-review.v1', video=str(VIDEO), original_packet=str(PACKET), original_packet_sha256=hashlib.sha256(PACKET.read_bytes()).hexdigest(), evidence=evidence, projection_not_used=True, source_not_modified=True, physical_identity_not_confirmed_by_marker=True,
        independent_RGB_review_notes=[
            dict(native_pts=229217280,actual_pts=229217024,physical_source_role='ad3:left_top',finding='Source point is on the actual upper-left external purple print transition; the 48.9 px model error is not explained by picking a rail or inner text.'),
            dict(native_pts=229624064,actual_pts=229624064,physical_source_role='ad1 external four corners',finding='All four saved source points lie on the actual external ISU ALL SKATING rectangular print. Current left X errors +56.2/+51.1 versus right -1.7/-4.7 px imply a width/pose conflict, not a common horizontal translation.'),
            dict(native_pts=229535232,actual_pts=229534976,physical_source_role='ad3:right_bottom and actual print right/top/bottom edges',finding='External purple print is visible with optical blur. Saved edge samples lie in its transition; the saved right-bottom corner differs somewhat from the central right-edge profile within existing blur. Original sigma and source support must remain in assessment.'),
            dict(native_pts=230507264,actual_pts=230507264,physical_source_role='advertisement7.left_outer_boundary',finding='All nine stored points lie in the visible purple print external left transition. The transition is broad; the original asymmetric blur support already represents this uncertainty. A 52 px raw model offset exceeds that original support.'),
        ],contact_source_review=contact,local_checks_not_full_programme_accuracy=True)
    path=OUT/'original_absolute_native_source_review.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(path),crops=[r['raw_crop'] for r in evidence])))


if __name__=='__main__':main()
