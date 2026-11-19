"""Exclude source-flow patch supports touching existing source foreground candidates."""
from collections import defaultdict
import gzip
import json
import cv2
import numpy as np
from shared import DATA, OUT, arrays, read, save, sha, write
from motion_tracks import packet


def run():
    cv2.setNumThreads(1)
    root=DATA/'runs/native_programme_20261031/source_person_mask_audit'
    ledger=root/'all9_frame_bindings.jsonl.gz';summary=read(root/'all9_summary.json')
    with gzip.open(ledger,'rt',encoding='utf8') as stream:
        records=[json.loads(line) for line in stream]
    by_pts={int(r['pts']):r for r in records};assert len(by_pts)==len(records)==8329
    setup=read(OUT/'prepared/setup.json');base=arrays(OUT/'prepared/baseline.npz')
    folder=OUT/'source_flow_visibility';folder.mkdir(exist_ok=True);reports={}
    # The inherited LK window is 19x19 at half resolution: 9*2 pixels
    # either side, plus source-pixel interpolation support rounded outwards.
    radius=19;kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
    for name,s in setup['shots'].items():
        if s['program']!='okada':continue
        assert summary['source_video_sha256']==s['source_sha256']
        path=packet(name,s);a=arrays(path);pts=base[name+'_pts']
        np.testing.assert_array_equal(a['pts'],pts)
        for pt in pts:assert int(pt) in by_pts and by_pts[int(pt)]['shot_id']==s['shot']
        reason=np.zeros(len(a['uv']),np.uint8);queries=defaultdict(list)
        for flag,index,uv in [(1,a['index'],a['uv']),
            (2,a['index']-a['lag'],a['left_uv']),
            (4,a['start'],a['anchor_uv']),
            (8,a['index']-2,a['prev2_uv'])]:
            good=np.isfinite(uv).all(1)&(index>=0)&(index<len(pts))
            for frame in np.unique(index[good]):
                ix=np.flatnonzero(good&(index==frame))
                queries[int(pts[frame])].append((flag,ix,uv[ix]))
        checked=[];empty=[]
        for pt,pieces in sorted(queries.items()):
            r=by_pts[pt];mask_path=r['mask_path']
            if sha(mask_path)!=r['mask_sha256']:raise ValueError('Changed source foreground file: '+mask_path)
            mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            assert mask.shape==(1080,1920)
            if not mask.any():empty.append(pt)
            covered=cv2.dilate((mask!=0).astype(np.uint8),kernel)
            for flag,ix,uv in pieces:
                xy=np.rint(uv).astype(int)
                valid=(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
                blocked=~valid
                if not mask.any():blocked[:]=True
                else:blocked[valid]|=covered[xy[valid,1],xy[valid,0]]!=0
                reason[ix[blocked]]|=flag
            checked.append(dict(pts=pt,path=mask_path,sha256=r['mask_sha256'],kind=r['mask_kind']))
        save(folder/(name+'.npz'),keep=reason==0,reason=reason,pts=pts,
            source_packet_sha256=sha(path),source_sha256=s['source_sha256'])
        write(folder/(name+'.json'),dict(source_masks=checked,empty_mask_pts=empty,
            track_packet=str(path),track_packet_sha256=sha(path),source_sha256=s['source_sha256'],
            rows=len(reason),excluded=int((reason!=0).sum()),retained=int((reason==0).sum())))
        reports[name]=dict(rows=len(reason),excluded=int((reason!=0).sum()),retained=int((reason==0).sum()))
        print(name,reports[name],flush=True)
    for case in read(OUT/'okada_flow_source/manifest.json'):
        assert not arrays(folder/(case['shot']+'.npz'))['keep'][case['source_row']]
    write(folder/'binding.json',dict(source_index=str(ledger),source_index_sha256=sha(ledger),
        source_video_sha256=summary['source_video_sha256'],shots=reports,
        support_radius_native_px=radius,
        policy='Exclude a flow row if any available actual source endpoint, anchor, or prev2 patch intersects the source foreground/unknown exclusion candidate. Empty masks are unknown, not clear background.',
        source_uv_or_raw_track_changed=False,camera_or_residual_used_for_admission=False,
        mask_accuracy_accepted=False,background_clearance_proven=False,
        source_review=str(OUT/'okada_flow_source/manifest.json')))


if __name__=='__main__':run()
