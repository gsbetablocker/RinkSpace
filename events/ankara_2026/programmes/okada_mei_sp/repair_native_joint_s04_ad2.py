"""Apply only individually source-reviewed S04 Ad2 letter identities."""
from pathlib import Path
import argparse
import collections
import shutil
import numpy as np
import prepare_native_joint_shots as p


def main(out):
    root=p.RUN;out.mkdir(exist_ok=True)
    review=p.read(p.HERE/'s04_ad2_identity_review_20261027.json')
    manifest=p.read(root/'joint_manifest.json');mapping=p.read(root/'mapping.json')
    assert len({c['observation_id'] for c in review['changes']})==len(review['changes'])
    original=p.npz(root/'point_observations.npz');points={k:v.copy() for k,v in original.items()}
    assert p.digest(root/'point_observations.npz')==manifest['point_input_sha256']
    state=p.npz(root/'joint_repaired/accepted_pixels_latest.npz')
    model=p.SceneParameterModel(p.read(p.prep.SCENE),p.prep.SCENE,optimize_scene=True)
    p.validate_joint_snapshot(state,model.initial_state.layout,mapping,manifest,points)
    lookup={str(v):i for i,v in enumerate(points['observation_ids'])};changed=[];applied=[]
    s04=next(s for s in manifest['shots'] if s['id']=='s04')
    for c in review['changes']:
        i=lookup[c['observation_id']];assert s04['row_begin']<=i<s04['row_stop']
        assert int(points['pts'][i])==c['pts'] and np.array_equal(points['uv'][i],c['uv'])
        assert p.digest(c['source_evidence'])==c['source_evidence_sha256']
        current=mapping['path_ids'][int(points['path_indices'][i])]
        if current==c['after']:continue
        assert current==c['before']
        points['path_indices'][i]=mapping['path_ids'].index(c['after']);changed.append(i);applied.append(c)
    changed=np.array(changed);assert len(changed)>0 and len(np.unique(changed))==len(changed)
    assert np.array_equal(np.flatnonzero(points['path_indices']!=original['path_indices']),np.sort(changed))
    for k in points:
        if k!='path_indices':np.testing.assert_array_equal(points[k],original[k])
        np.testing.assert_array_equal(points[k][:s04['row_begin']],original[k][:s04['row_begin']])
    np.savez(out/'point_observations.npz',**points);old_sha=manifest['point_input_sha256']
    manifest['point_input_sha256']=p.digest(out/'point_observations.npz')
    manifest['source_identity_repairs']=[*manifest.get('source_identity_repairs',[]),dict(review=str(p.HERE/'s04_ad2_identity_review_20261027.json'),review_sha256=p.digest(p.HERE/'s04_ad2_identity_review_20261027.json'),rows=len(changed),old_input_sha256=old_sha)]
    fixed={k:v.copy() for k,v in state.items()};fixed['stations'][changed]=.5
    fixed['point_input_sha256']=np.array(manifest['point_input_sha256'])
    p.bind_joint_snapshot(fixed,model.initial_state.layout,mapping,manifest)
    p.validate_joint_snapshot(fixed,model.initial_state.layout,mapping,manifest,points)
    for k in ('rig','controls','scene_values'):np.testing.assert_array_equal(fixed[k],state[k])
    np.savez(out/'initial_pixels.npz',**fixed);p.write(out/'joint_manifest.json',manifest)
    for name in ('point_observations.npz','initial_pixels.npz','joint_manifest.json'):
        prior=out/('before_'+name)
        if prior.exists():raise ValueError('This exact repair already has an installation history')
        shutil.copy2(root/name,prior)
    shutil.copy2(root/'joint_repaired/accepted_pixels_latest.npz',out/'before_pixels.npz')
    for name in ('point_observations.npz','initial_pixels.npz','joint_manifest.json'):
        tmp=root/(name+'.pending');shutil.copy2(out/name,tmp);tmp.replace(root/name)
    p.write(out/'source_review.json',review)
    receipt=dict(rows=len(points['pts']),rebound_rows=len(changed),removed_rows=0,original_uv_pts_ids_preserved=True,
        all_other_point_fields_unchanged=True,s01_all_fields_unchanged=True,rig_scene_controls_unchanged=True,
        new_input_sha256=manifest['point_input_sha256'],before_input_sha256=old_sha,
        counts=dict(collections.Counter(c['before']+' -> '+c['after'] for c in applied)),
        point_weights_or_error_ranges_added=False)
    p.write(out/'replacement_receipt.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=p.RUN/'ad2_identity_repair')
    main(parser.parse_args().output)
