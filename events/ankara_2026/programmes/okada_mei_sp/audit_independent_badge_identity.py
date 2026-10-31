"""Freeze a failed badge package and export source-only physical identity review.

Appearance agreement is not physical-instance identity. This command does not
relabel points, produce a replacement fit package, change camera state, or fit.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import cv2,numpy as np
from capture_independent_badge import exact_frames


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('root','input','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();source=a.input.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    index=json.loads((root/'index.json').read_text('utf8'));original=Path(index['source']['resolved_path'])
    if sha(original)!=index['source']['actual_sha256']:raise ValueError('Actual original source changed')
    rows_file=source/'qualified/s07/source_rows.jsonl';records=[r for line in rows_file.open(encoding='utf8')if(r:=json.loads(line))['fit_eligible']]
    pts=[r['pts']for r in records]
    if len(pts)!=73 or len(set(pts))!=len(pts):raise ValueError('Expected unchanged 73-frame failed package')
    lookup={r['pts']:r for r in records};ranges=[]
    for r in records:
        pt=r['pts']
        if ranges and pt==ranges[-1]['last_pts']+256:ranges[-1]['last_pts']=pt;ranges[-1]['frames']+=1;ranges[-1]['rows']+=len(r['rows']);ranges[-1]['pts'].append(pt)
        else:ranges.append(dict(first_pts=pt,last_pts=pt,frames=1,rows=len(r['rows']),pts=[pt]))
    approved={1446656,1446912,1447168};targets=sorted({1568000,*[g[k]for g in ranges for k in('first_pts','last_pts')]})
    out.mkdir(parents=True);frozen=out/'FAILED_DO_NOT_INGEST_snapshot';shutil.copytree(source/'source_packets',frozen)
    original_hashes={str(p.relative_to(source/'source_packets')):sha(p)for p in(source/'source_packets').rglob('*')if p.is_file()}
    for rel,digest in original_hashes.items():
        if sha(frozen/rel)!=digest:raise ValueError('Failed packet byte snapshot differs')
    (out/'DO_NOT_INGEST.json').write_text(json.dumps(dict(status='physical_instance_identity_unverified',original_failed_package=str(source/'source_packets/s07'),frozen_forensic_snapshot=str(frozen),same_artwork_is_not_instance_identity=True,only_current_explicitly_approved_PTS=sorted(approved),no_new_fit_package_created=True,no_relabeling_or_camera_change=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    images=exact_frames(original,targets);proof_dir=out/'source_review';proof_dir.mkdir();proofs={}
    for pt in targets:
        image=images[pt];r=lookup[pt]
        if hashlib.sha256(image.tobytes()).hexdigest()!=r['source_BGR_sha256']:raise ValueError('Exact PTS source BGR differs from frozen row evidence')
        raw_path=proof_dir/f's07_pts{pt}_original.png';cv2.imwrite(str(raw_path),image)
        proof=image.copy()
        for q in r.get('pending_rows',[]):cv2.circle(proof,tuple(np.rint(q['uv']).astype(int)),2,(0,120,255),1,cv2.LINE_AA)
        for number,q in enumerate(r['rows']):
            cv2.circle(proof,tuple(np.rint(q['uv']).astype(int)),4,(0,255,0),1,cv2.LINE_AA)
            cv2.putText(proof,str(number),tuple(np.rint(q['uv']).astype(int)+[5,-3]),0,.38,(0,255,0),1,cv2.LINE_AA)
            for side,c in zip(np.array(q['actual_source_quantiles']),[(255,100,0),(255,0,255)]):cv2.line(proof,tuple(np.rint(side[0]).astype(int)),tuple(np.rint(side[2]).astype(int)),c,1,cv2.LINE_AA)
        cv2.putText(proof,f'S07 PTS {pt} | original label Ad12 UNVERIFIED | check full board order | NO FIT',(18,35),0,.65,(0,220,255),2,cv2.LINE_AA)
        overlay=proof_dir/f's07_pts{pt}_identity_HOLD_candidates.jpg';cv2.imwrite(str(overlay),proof,[cv2.IMWRITE_JPEG_QUALITY,95])
        proofs[pt]=dict(pts=pt,source_time_seconds=pt/12800.,original_source=str(raw_path),original_source_file_sha256=sha(raw_path),source_BGR_sha256=r['source_BGR_sha256'],candidate_proof=str(overlay),selected_rows=len(r['rows']),pending_candidate_rows=len(r.get('pending_rows',[])),physical_instance_status='explicitly_approved_pilot'if pt in approved else'UNVERIFIED_DO_NOT_FIT',stored_path_ids=sorted({q['path_id']for q in r['rows']}))
    for g in ranges:
        g.update(first_source_time_seconds=g['first_pts']/12800.,last_source_time_seconds=g['last_pts']/12800.,first_source_proof=proofs[g['first_pts']],last_source_proof=proofs[g['last_pts']],explicitly_approved_PTS=[p for p in g['pts']if p in approved],unverified_physical_instance_PTS=[p for p in g['pts']if p not in approved])
    report=dict(status='HOLD_PHYSICAL_INSTANCE_IDENTITY',source_video=str(original),source_video_sha256=index['source']['actual_sha256'],original_failed_package=str(source/'source_packets/s07'),original_qualified_source=str(rows_file),original_qualified_source_sha256=sha(rows_file),forensic_snapshot=str(frozen),forensic_snapshot_file_sha256=original_hashes,qualified_frame_count_in_failed_input=len(records),rows_in_failed_input=sum(len(r['rows'])for r in records),all_73_qualified_PTS=pts,only_current_explicitly_approved_PTS=sorted(approved),unverified_PTS=[p for p in pts if p not in approved],intervals=ranges,source_proofs=proofs,physical_instance_not_determined_by_pattern_similarity=True,camera_or_native_projection_not_changed=True,points_not_relabelled=True,new_fit_package_created=False,fit_executed=False,adapter=str(Path(__file__).resolve()),adapter_sha256=sha(__file__))
    (out/'identity_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['Ad12 failed package: PHYSICAL INSTANCE IDENTITY HOLD. Original bytes retained.',f'73 frames / {report["rows_in_failed_input"]} rows; only three pilot PTS explicitly approved.','']
    for i,g in enumerate(ranges,1):lines.append(f'{i}. PTS {g["first_pts"]}..{g["last_pts"]} | {g["first_source_time_seconds"]:.2f}..{g["last_source_time_seconds"]:.2f}s | {g["frames"]} frames / {g["rows"]} rows | endpoints in source_review/')
    lines.extend(['',f'Priority 1568000 original: {proofs[1568000]["original_source"]}',f'Priority 1568000 candidates: {proofs[1568000]["candidate_proof"]}'])
    (out/'intervals.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(out/'identity_review.json'),intervals=str(out/'intervals.txt'),forensic_snapshot=str(frozen),frames=len(pts),endpoint_source_frames=len(targets),priority_1568000=proofs[1568000],ranges=[{k:g[k]for k in('first_pts','last_pts','frames','rows')}for g in ranges]),ensure_ascii=True),flush=True)


if __name__=='__main__':main()