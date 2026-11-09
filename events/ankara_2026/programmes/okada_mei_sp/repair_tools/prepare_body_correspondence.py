import argparse
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment
from calibration.program_tracking.body_tracking.programme_skate_replay import save, write
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load

def main():
    p=argparse.ArgumentParser()
    for key in ('observations','review','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    r=json.loads(a.review.read_text(encoding='utf8'))
    if (r['schema']!='rinkspace.body-source-correspondence-review.v1'
            or digest(a.observations)!=r['parent_observations_sha256']
            or digest(r['source_review'])!=r['source_review_sha256']):
        raise ValueError('Unbound source correspondence review')
    data=load(a.observations)
    pts=data['pts']
    assignment=body_joint_assignment(data).copy()
    use=data['use'].copy()
    records=[]
    for row in r['actions']:
        stamp=row['pts']
        i=int(np.searchsorted(pts,stamp))
        if int(pts[i])!=stamp:
            raise ValueError('Correspondence change outside the exact clock')
        for j in row['swap_pairs']:
            if j not in (13,15) or assignment[i,j:j+2].tolist()!=[j,j+1]:
                raise ValueError('Requires explicit identity on the original knee/ankle pair')
            assignment[i,j:j+2]=[j+1,j]
        for j in row['disable_slots']:
            if j not in (13,14,15,16) or not use[i,j]:
                raise ValueError('Can only remove an admitted knee/ankle slot')
            use[i,j]=False
        records.append(dict(row,source_uv=data['uv'][i,13:17].tolist(),
                            new_model_ids=assignment[i,13:17].tolist(),new_use=use[i,13:17].tolist()))
    identity=np.arange(17)
    record=dict(kind='source_observation_correspondence',source_measurements_unchanged=True,
                source_pts=pts[np.any(assignment!=identity,axis=1)].tolist(),
                parent_provenance=json.loads(str(data['body_joint_assignment_provenance_json'])),
                review_path=str(a.review.resolve()),review_sha256=digest(a.review),records=records)
    revised=dict(data,model_joint_ids=assignment,use=use,
                 body_joint_assignment_provenance_json=np.array(json.dumps(record,ensure_ascii=False,sort_keys=True)))
    body_joint_assignment(revised)
    a.output.mkdir()
    filename=a.output/'observations.npz'
    save(filename,**revised)
    for k in ('uv','sigma_px','pts'):
        np.testing.assert_array_equal(revised[k],data[k])
    write(a.output/'contract.json',dict(path=str(filename.resolve()),sha256=digest(filename),
          parent_observations_sha256=digest(a.observations),source_video_sha256=r['source_video_sha256'],
          time_base=r['time_base'],review_path=str(a.review.resolve()),review_sha256=digest(a.review),
          changed_exposures=len(records),rebound_rows=int(np.sum(assignment!=data['model_joint_ids'])),
          excluded_rows=int(np.sum(data['use'] & ~use)),records=records))

if __name__=='__main__':
    main()
