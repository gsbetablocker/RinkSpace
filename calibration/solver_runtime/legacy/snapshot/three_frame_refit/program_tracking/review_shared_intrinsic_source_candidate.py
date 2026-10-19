"""Review exact native frames of a camera-only candidate on one fixed scene.

All held feature rows remain outside fitting. The saved images use the complete
fixed rink inventory and actual advertisement geometry, with no video output.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

from main7_source_camera_candidate import arrays, errors, sha
from render_result import SceneRenderer, decode_selected, PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
import cv2, numpy as np


def source_time(pts):
    milliseconds=round(pts*1000/15360)
    hours,milliseconds=divmod(milliseconds,3600000)
    minutes,milliseconds=divmod(milliseconds,60000)
    seconds,milliseconds=divmod(milliseconds,1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}'


def load_rows(path):
    data=json.loads(path.read_text(encoding='utf8'))
    for rows in data.values():
        for row in rows:
            for name in ('uv','world','sigma','normal','normal_interval'):
                if name in row:row[name]=np.asarray(row[name],float)
    return data


def run(a):
    if a.output.exists():raise FileExistsError('Keep historical review images; use a new output directory')
    a.output.mkdir(parents=True)
    scene=json.loads(a.scene.read_text(encoding='utf8'))
    candidate=arrays(a.candidate);seed=arrays(a.seed)
    if not np.array_equal(candidate['pts'],seed['pts']):raise ValueError('Seed and candidate must cover identical native PTS')
    camera={int(p):c for p,c in zip(candidate['pts'],candidate['camera25'])}
    before={int(p):c for p,c in zip(seed['pts'],seed['camera25'])}
    rows=load_rows(a.rows);held=rows['held_rows']
    shots=json.loads(a.shots.read_text(encoding='utf8'))['shots']
    def cut(p):
        matches=[i for i,s in enumerate(shots) if s['start_pts']<=p<s['end_pts']]
        if len(matches)!=1:raise ValueError(f'Ambiguous cut at {p}')
        return matches[0]
    selections={int(r['pts']):['held absolute feature']for r in held}
    for c in ([] if a.no_focal_samples else sorted({cut(p)for p in camera})):
        stamps=np.array([p for p in camera if cut(p)==c])
        f=np.array([np.exp(np.mean(camera[p][6:8]))for p in stamps])
        for quantile,label in ((0.,'wide'),(.5,'medium'),(1.,'narrow')):
            p=int(stamps[np.argmin(np.abs(f-np.quantile(f,quantile)))])
            selections.setdefault(p,[]).append(f'cut {c} {label} effective focal sample')
    for p in [int(x)for x in a.extra_pts.split(',')if x]:
        if p not in camera:raise ValueError(f'Camera candidate omitted requested source PTS {p}')
        selections.setdefault(p,[]).append('requested source frame')
    for p in [int(x)for x in a.neighbours.split(',')if x]:
        for offset in (-256,0,256):
            if p+offset in camera:selections.setdefault(p+offset,[]).append('three adjacent native frames')
    renderer=SceneRenderer(scene);masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    checks=[]
    for split,group in rows.items():
        for row in group:
            p=int(row['pts']);checks.append(dict(pts=p,cut=cut(p),id=row['id'],split=row['split'],
                source_origin=row.get('origin'),seed=errors(before[p],[row])[0],candidate=errors(camera[p],[row])[0]))
    images=[];thumbnails=[]
    for p,bgr in decode_selected(a.source,sorted(selections)):
        try:mask,mask_status=masks.get(p,require_sam2=True)
        except ValueError:mask=None;mask_status='no confirmed source mask; no box hiding used'
        title=f'source {source_time(p)} | PTS {p} | cut {cut(p)} | camera candidate / unverified'
        overlay,metrics=renderer.draw(bgr,camera[p],mask,title=title)
        earlier,_=renderer.draw(bgr,before[p],mask,title=f'source {source_time(p)} | {a.reference_label} / unverified')
        observed=bgr.copy()
        for row in [r for r in held if int(r['pts'])==p]:
            for uv in row['uv']:
                if np.isfinite(uv).all():cv2.circle(observed,tuple(np.rint(uv).astype(int)),3,(25,40,255),-1,cv2.LINE_AA)
        files={}
        for label,image in (('original_native',bgr),('held_source_pixels',observed),('full_model_overlay',overlay)):
            filename=a.output/f'{p}_{label}.jpg';cv2.imwrite(str(filename),image,[cv2.IMWRITE_JPEG_QUALITY,94]);files[label]=str(filename)
        comparison=np.vstack([cv2.resize(earlier,(1280,720),interpolation=cv2.INTER_AREA),cv2.resize(overlay,(1280,720),interpolation=cv2.INTER_AREA)])
        filename=a.output/f'{p}_seed_vs_camera_candidate.jpg';cv2.imwrite(str(filename),comparison,[cv2.IMWRITE_JPEG_QUALITY,94]);files['comparison']=str(filename)
        thumb=cv2.resize(overlay,(640,360),interpolation=cv2.INTER_AREA)
        cv2.putText(thumb,f'{source_time(p)} cut {cut(p)}',(8,348),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(thumb,f'{source_time(p)} cut {cut(p)}',(8,348),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        thumbnails.append(thumb)
        images.append(dict(pts=p,cut=cut(p),source_time=source_time(p),selection_reasons=selections[p],files=files,
            native_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),mask_status=mask_status,
            candidate_camera25=camera[p].tolist(),render_metrics=metrics))
    for start in range(0,len(thumbnails),8):
        subset=thumbnails[start:start+8]
        while len(subset)%2:subset.append(np.full((360,640,3),245,np.uint8))
        board=np.vstack([np.hstack(subset[i:i+2])for i in range(0,len(subset),2)])
        cv2.imwrite(str(a.output/f'contact_sheet_{start//8+1}.jpg'),board,[cv2.IMWRITE_JPEG_QUALITY,93])
    report=dict(accuracy_accepted=False,no_video_generated=True,whole_cuts_not_visually_validated=True,
        review_scope='all supplied held feature keys, requested native frames/neighbours, optional effective-focal extremes/median',
        reference_label=a.reference_label,
        held_independence_limit='New absolute features excluded from camera fitting; same frames remain in the source motion graph.',
        geometry_fixed_sha256=sha(a.scene),input_files={str(p):sha(p)for p in (a.scene,a.candidate,a.seed,a.rows,a.shots)},
        fixed_model_path_counts=renderer.path_counts,advertisement_geometry='actual solved AdGeometry on same finite board',
        source_uv_changed=False,feature_checks=checks,images=images)
    (a.output/'source_review_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(images=len(images),held_keys=len(set(int(r['pts'])for r in held)),output=str(a.output))),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','scene','candidate','seed','rows','shots','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--neighbours',default='')
    parser.add_argument('--extra-pts',default='')
    parser.add_argument('--no-focal-samples',action='store_true')
    parser.add_argument('--reference-label',default='prepared numerical seed')
    run(parser.parse_args())
