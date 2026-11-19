"""Repair isolated bilateral detector flips using bidirectional source motion."""
import argparse
import json
from pathlib import Path

import numpy as np

from source_review import digest
from calibration.program_tracking.body_tracking.body_source_revision import revise_body_observations


def repair(observations, flow, output):
    old = dict(np.load(observations))
    tracking = dict(np.load(flow))
    np.testing.assert_array_equal(old['pts'],tracking['pts'])
    uv, pts = old['uv'],old['pts']
    ids = old['model_joint_ids'].copy()
    torso = np.linalg.norm(uv[:,5:7].mean(1)-uv[:,11:13].mean(1),axis=1)
    tolerance = np.maximum(8.,.12*np.minimum(torso[:-1],torso[1:]))
    same = tracking['shot'][:-1]==tracking['shot'][1:]
    revisions=[]
    for left in (5,7,9,11,13,15):
        pair=np.array([left,left+1])
        forward=tracking['forward'][:-1,pair]
        backward=tracking['backward'][1:,pair]
        direct=np.stack((np.linalg.norm(forward-uv[1:,pair],axis=-1),
                         np.linalg.norm(backward-uv[:-1,pair],axis=-1)),axis=1)
        crossed=np.stack((np.linalg.norm(forward-uv[1:,pair[::-1]],axis=-1),
                          np.linalg.norm(backward-uv[:-1,pair[::-1]],axis=-1)),axis=1)
        preferred=crossed.mean(2)<direct.mean(2)
        best=np.where(preferred[:,:,None],crossed,direct)
        other=np.where(preferred[:,:,None],direct,crossed)
        reliable=(best.max(2)<tolerance[:,None]) & ((other-best).min(2)>tolerance[:,None])
        for side,(direction,cut) in enumerate([('forward',slice(None,-1)),('backward',slice(1,None))]):
            reliable[:,side] &= (tracking[direction+'_fb'][cut,pair]<3.5).all(1)
            reliable[:,side] &= (tracking[direction+'_photo'][cut,pair]<32.).all(1)
        # One reliable direction suffices; a reliable reverse track must agree.
        valid=same & reliable.any(1) & ~(reliable.all(1)&(preferred[:,0]!=preferred[:,1]))
        cross=np.where(reliable[:,0],preferred[:,0],preferred[:,1])
        edges=np.flatnonzero(valid)
        if not len(edges):
            continue
        components=np.split(edges,np.flatnonzero(np.diff(edges)>1)+1)
        prior=old['model_joint_ids'][:,left]!=left
        for edges in components:
            frames=np.arange(edges[0],edges[-1]+2)
            relative=np.r_[False,np.logical_xor.accumulate(cross[edges])]
            options=[]
            for offset in (False,True):
                guess=relative^offset
                # Both flanks and all previous explicit reassignments remain
                # fixed. No model projection chooses anatomical identity.
                if (guess[0]!=prior[frames[0]] or guess[-1]!=prior[frames[-1]]
                        or np.any(prior[frames] & ~guess)):
                    continue
                options.append((int(np.count_nonzero(guess!=prior[frames])),guess))
            if len(options)!=1:
                continue
            count,guess=options[0]
            if not count or count*2>=len(frames):
                continue
            changed=frames[guess!=prior[frames]]
            if not old['use'][changed[:,None],pair].all():
                continue
            for frame in changed:
                before=ids[frame,pair].copy()
                ids[frame,pair]=pair[::-1] if guess[frame-frames[0]] else pair
                revisions.append(dict(pts=int(pts[frame]),pair=pair.tolist(),
                    before=before.tolist(),after=ids[frame,pair].tolist(),
                    source_track_flanks_pts=pts[frames[[0,-1]]].tolist(),
                    provenance='inferred from reciprocal adjacent source pixels; no model input'))
    assignments=[dict(source_pts=r['pts'],joint_pairs=[r['pair']],
                      reason=r['provenance'],evidence_pts=r['source_track_flanks_pts'])
                 for r in revisions]
    if assignments:
        new,_=revise_body_observations(old,assignments,[],
            dict(source_flow_sha256=digest(flow),inferred_from_source_pixels=True))
    else:
        new={k:v.copy() for k,v in old.items()}
    np.testing.assert_array_equal(new['model_joint_ids'],ids)
    new['source_motion_track_revision_json']=np.array(json.dumps(dict(
        parent=dict(path=str(observations),sha256=digest(observations)),
        source_flow=dict(path=str(flow),sha256=digest(flow)),
        method='reciprocal native-source patch correspondence; closed components preserve both flanks and prior explicit repairs',
        source_uv_sigma_use_unchanged=True,model_projection_used=False,
        inferred_correspondences=True,changes=revisions),sort_keys=True))
    np.savez_compressed(output,**new)
    for key in ('pts','uv','sigma_px','use'):
        np.testing.assert_array_equal(new[key],old[key])
    print(json.dumps(dict(output=str(output),changed_pairs=len(revisions),
        per_shot={str(s):sum(tracking['shot'][np.searchsorted(pts,r['pts'])]==s for r in revisions)
                  for s in np.unique(tracking['shot'])},
        examples=[dict(seconds=r['pts']/12800,pair=r['pair']) for r in revisions[::max(1,len(revisions)//35)]]),default=int),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observations','flow','output'):
        p.add_argument('--'+name,type=Path,required=True)
    repair(**vars(p.parse_args()))
