"""Original-RGB identity audit for retained low5 rows labelled ad4.

Only inspection material is written. No original row, split, geometry, or
runtime manifest is changed; saved labels are proposals until source review.
"""
from pathlib import Path
import hashlib,json,pickle
import av,cv2

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT=BASE/'diagnostics/external_expert_20261007/all_original_absolute_source_audit/native_rgb_review/low5_Ad4_17_identity'
PACKET=BASE/'calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
PTS=sorted(set([229961216+6144*i for i in range(10)]+[229945856,230022144,230066432,230123008,230186240,230249472,230312704,230338304]))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parent_bytes=PACKET.read_bytes();rows=pickle.loads(parent_bytes)['programme_absolute_rows']
    selected=[(i,r)for i,r in enumerate(rows)if r['group_actual_pts']==229961216
              and r['source_row'].get('id','').startswith('ad4:')]
    assert len(selected)==3 and all(r['collection_role']=='HELD'for _,r in selected)
    evidence=[]
    with av.open(str(BASE/'media/originals/oxXu1avhrvs.mp4'))as c:
        stream=c.streams.video[0];stream.thread_count=1
        for pts in PTS:
            c.seek(pts-15360,stream=stream,backward=True,any_frame=False)
            frame=next(f for f in c.decode(stream)if f.pts==pts)
            bgr=frame.to_ndarray(format='bgr24');path=OUT/f'raw_native_{pts}.png';cv2.imwrite(str(path),bgr)
            evidence.append(dict(native_pts=pts,path=str(path),decoded_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    image=cv2.imread(str(OUT/'raw_native_229961216.png'));marked=image.copy()
    for _,record in selected:
        source=record['source_row'];assert source['source_native_pts']==229961216
        for u,v in source['uv']:
            cv2.circle(marked,(int(round(u)),int(round(v))),3,(255,0,255),1,cv2.LINE_AA)
    for bgr,suffix in[(image,'RGB'),(marked,'stored_UV')]:
        cv2.imwrite(str(OUT/f'old_ad4_native_229961216_{suffix}_left_border_crop.png'),bgr[420:650,210:750])
    assert evidence[next(i for i,x in enumerate(evidence)if x['native_pts']==229961216)]['decoded_BGR_sha256']==selected[0][1]['source_row']['source_decoded_bgr24_sha256']
    assert PACKET.read_bytes()==parent_bytes
    report=dict(schema='rinkspace.old-low5-Ad4-Ad17-original-RGB-inspection.v1',
        actual_and_native_pts=229961216,original_packet_sha256=hashlib.sha256(parent_bytes).hexdigest(),
        raw_source_decoded_hash_matches_original=True,source_and_split_unchanged=True,
        projection_used=False,evidence=evidence,
        old_rows=[dict(original_packet_index=i,collection_role=r['collection_role'],source={k:v for k,v in r['source_row'].items()if k not in['world','sigma']})for i,r in selected],
        physical_instance_identity_review_pending=True,samples_not_complete_shot_accuracy_validation=True)
    path=OUT/'old_low5_Ad4_17_source_identity_inspection.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:x.tolist())+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(path),frames=len(evidence))))

if __name__=='__main__':main()
