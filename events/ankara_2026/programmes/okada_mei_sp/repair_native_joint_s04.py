"""Withdraw two source-verified S04 identity failures; retain original package."""
import json,gzip
import numpy as np
import prepare_native_joint_shots as p

def main():
    r=p.RUN;m=p.read(r/'joint_manifest.json');z=p.npz(r/'point_observations.npz');v=p.npz(r/'s04_reseeded_initial.npz')
    if m.get('s04_source_repair'):raise ValueError('Repair already applied')
    mapping=p.read(r/'mapping.json');paths=np.array(mapping['path_ids'])[z['path_indices']]
    s04=z['pts']>=745984
    # Source frames show the early atlas-6 tracks on ISU-SKATING.COM, not
    # ANKARA. Its later real ANKARA observations at 72.90--73.70 remain.
    ad=s04&(z['pts']<=857600)&(np.char.find(paths,'Ad4.ANKARA.native440.boundary_')>=0)
    # Consecutive source frames show these samples moving through blank ice,
    # streaks and the skater/reflection, not the declared continuous A1 arc.
    ice=s04&(paths=='demo_initial/A1:outer')&(z['pts']>=945408)&(z['pts']<=947456)
    bad=ad|ice;keep=~bad
    if (int(ad.sum()),int(ice.sum()))!=(1267,56):raise ValueError('Unexpected repair scope; review source evidence again')
    repair=r/'s04_source_repairs.jsonl.gz'
    with gzip.open(repair,'wt',encoding='utf8') as f:
        for i in np.flatnonzero(bad):
            record=dict(action='exclude_from_current_fit_keep_original_source',observation_id=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),
                original_joint_row=int(i),pts=int(z['pts'][i]),uv=z['uv'][i].tolist(),path_id=str(paths[i]),
                reason='ANKARA identity falsely tracked onto ISU-SKATING.COM; exact letter correspondence not established' if ad[i] else 'no persistent A1 curve at source samples; blank ice/streak/reflection reads',
                evidence=str(r/'review'/('atlas6_span.png' if ad[i] else 'a1_continuous.png')),criterion='actual consecutive source image identity; not fitted residual magnitude')
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    original=r/'imported_point_observations.npz'
    if original.exists():raise FileExistsError(original)
    (r/'point_observations.npz').replace(original)
    np.savez(r/'point_observations.npz',**{k:a[keep] for k,a in z.items()})
    sha=p.digest(r/'point_observations.npz');v['stations']=v['stations'][keep];v['source_rows']=z['source_rows'][keep];v['point_input_sha256']=np.array(sha)
    np.savez(r/'s04_repaired_initial.npz',**v)
    m['imported_point_input_sha256']=m['point_input_sha256'];m['point_input_sha256']=sha;m['shots'][1]['row_stop']-=int(bad.sum())
    m['s04_source_repair']=dict(path=str(repair),sha256=p.digest(repair),ads_excluded=int(ad.sum()),ice_excluded=int(ice.sum()),
        correct_later_ANKARA_rows_retained=3929,s01_observations_changed=0,original_UV_of_retained_rows_changed=0,parameter_values_changed=0,
        evidence=[str(r/'review/atlas6_span.png'),str(r/'review/a1_continuous.png')])
    p.write(r/'joint_manifest.json',m);print(json.dumps(m['s04_source_repair']),flush=True)

if __name__=='__main__':main()
