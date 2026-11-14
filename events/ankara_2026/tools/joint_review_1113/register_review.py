"""Register the common numeric candidate, per-programme bindings, and local review."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json

import joint_inputs as ji
from joint_inputs import OUT,np
from rinkspace.path_policy import guard_project_write
import solve_joint as sj
import verify_joint as verify
from inspect_geometry import changes,compare

NAME='joint_review_1113'
REPO=ji.ROOT
EVENT=REPO/'events/ankara_2026'
PREP=OUT/'prepared_guarded'
CAND=OUT/'final'
VID=OUT/'continuous'
SOURCE=OUT/'reserved_source'

def fingerprint(path):
    return ji.pin(Path(path).resolve())

PRIOR_MANIFEST={
    'jin_shuxian_sp':'205fe9bcc76fb54893c0b627f306fd35d03d9fb1392574bb729841f49a5cbd13',
    'okada_mei_sp':'544b7243caf194ae15c3beed67dda707c1e08520a715e7d78fd6e3eb5696b3a8'}
ALLOWED={}


def file_path(x):
    p=guard_project_write(x)
    if p.exists() and fingerprint(p)['sha256']!=ALLOWED.get(p):
        raise FileExistsError(p)
    return p


def verified_draft():
    for prog,sha in PRIOR_MANIFEST.items():
        folder=EVENT/'programmes'/prog
        if prog=='okada_mei_sp':
            folder=folder/'fit_v11'
        path=guard_project_write(folder/(NAME+'.json'))
        ref=ji.pin(path,sha)
        row=ji.read(path)
        ALLOWED[path]=ref['sha256']
        for key in ('geometry','camera'):
            p=guard_project_write(row[key]['path'])
            ALLOWED[p]=ji.check(row[key])['sha256']

def install_npz(path,arrays):
    p=file_path(path)
    if p.is_file():
        with np.load(p,allow_pickle=False) as z:
            if z.files==list(arrays) and all(np.array_equal(z[k],v) for k,v in arrays.items()):
                return fingerprint(p)
    staged=file_path(p.with_name(p.stem+'.staged.npz'))
    with staged.open('xb') as f:
        np.savez_compressed(f,**arrays)
    with np.load(staged,allow_pickle=False) as z:
        for k,v in arrays.items():
            np.testing.assert_array_equal(z[k],v)
    staged.replace(p)
    return fingerprint(p)

def install_json(path,value):
    p=file_path(path)
    staged=file_path(p.with_name(p.stem+'.staged.json'))
    ji.write(staged,value)
    if ji.read(staged)!=value:
        raise ValueError('Candidate registration did not round-trip')
    staged.replace(p)
    return fingerprint(p)

def summarize(m):
    total={}
    for prog in ('jin','okada'):
        data={'before':[],'after':[],'schedule':0,'checked':0}
        root=SOURCE/(prog+'.json')
        if not root.exists():continue
        for name,f in ji.read(root).items():
            if not isinstance(f,dict):continue
            rows=ji.read(SOURCE/(name+'.json'))['rows']
            data['schedule']+=f['scheduled'];data['checked']+=f['actual_qualified_frames']
            for r in rows:
                data['before'].extend(r['baseline'])
                data['after'].extend(r['baseline'] if name in ('jin_s10','jin_s18','jin_s24') else r['review'])
        total[prog]=dict(scheduled_frames=data['schedule'],qualified_frames=data['checked'],
                         before=ji.stats(data['before']),after=ji.stats(data['after']),
                         selected_without_S10_S18_S24_using_baseline_errors=True)
    return total

def new_counts(shot_rows):
    rows=Counter();frames=0
    for path in shot_rows:
        for f in ji.lines(path):
            if f['fold']!=0:continue
            selected=Counter()
            for r in f['groups']:
                if r['origin']=='fresh':
                    selected[r['material']]+=len(r['uv'])
            rows.update(selected)
            frames+=bool(selected)
    return dict(fresh_rows=sum(rows.values()),fresh_exposures=frames,by_material=dict(rows))

def key_stats(table,label,fold,frames):
    a=table['baseline'][fold].get(label)
    b=table['review'][fold].get(label)
    if not a or not b:return None
    return dict(rows=a['rows'],frames=a['frames'],before=a['p90'],after=b['p90'],
                unit='native_pixel',stat='p90')

def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--out',default=NAME)
    v=ap.parse_args()
    if v.out==NAME:
        verified_draft()
    sj.INPUT=PREP
    m,core,b,series=sj.load()
    r=verify.load_checkpoint(CAND/'review.npz')
    a=ji.context()
    old=ji.ParameterState(m['scene_layout'],b['scene_values'])
    state=ji.ParameterState(m['scene_layout'],r['scene_values'])
    native=ji.NativeSceneGPU(m['scene_path'],core,parameter_state=state)
    base=ji.NativeSceneGPU(m['scene_path'],core,parameter_state=old)
    scene_changes=changes(old,state)
    if len(scene_changes)!=5 or any(not q['block'].startswith('installation/') for q in scene_changes):
        raise ValueError('Unexpected scene changes')
    world_changes=compare(base,native)
    if world_changes['removed_paths'] or world_changes['added_paths'] or len(native.paths)!=922:
        raise ValueError('Shared candidate changed the rink inventory')
    if not np.array_equal(base.scene['board'],native.scene['board']) or not np.array_equal(
            base.scene['shared_layout'],native.scene['shared_layout']):
        raise ValueError('Ice or boards changed')
    cmp=ji.read(CAND/'comparison.json')
    expected=cmp['geometry_sha256']
    if ji.numeric_hash(state.values)!=expected:raise ValueError('Candidate reference changed')
    reports={}
    for prog in ('jin','okada'):
        root=VID/prog
        v0=ji.read(root/'review.json')
        if v0['geometry_sha256']!=expected:raise ValueError('Video was generated with different geometry')
        receipts={p.name.split('_')[1]:ji.read(p) for p in sorted(root.glob('*_receipt.json'))}
        if set(receipts)!=set(k.split('_')[1] for k in v0['shots']):raise ValueError('Review missing an entire shot')
        proof=verify.load_checkpoint(Path(v0['numeric_state']['path']))
        if prog=='jin':
            root2=OUT/'rechecked/jin'
            p2=ji.read(root2/'review.json')
            if p2['geometry_sha256']!=expected:
                raise ValueError('Updated Jin clips do not match the selected candidate')
            proof2=verify.load_checkpoint(Path(p2['numeric_state']['path']))
            for sid in ('s10','s18'):
                receipts[sid]=ji.read(root2/f'jin_{sid}_receipt.json')
                for suffix in ('_rig','_controls'):
                    np.testing.assert_array_equal(proof2['jin_'+sid+suffix],r['jin_'+sid+suffix])
            root3=OUT/'rechecked_spin/jin'
            p3=ji.read(root3/'review.json')
            if p3['geometry_sha256']!=expected or p3['numeric_state']['sha256']!=fingerprint(CAND/'review.npz')['sha256']:
                raise ValueError('S24 clip does not use the selected candidate')
            receipts['s24']=ji.read(root3/'jin_s24_receipt.json')
        if prog=='okada':
            root4=OUT/'rechecked_motion/okada'
            p4=ji.read(root4/'review.json')
            if p4['geometry_sha256']!=expected or p4['numeric_state']['sha256']!=fingerprint(CAND/'review.npz')['sha256']:
                raise ValueError('Okada S04 clip does not use the selected candidate')
            receipts['s04']=ji.read(root4/'okada_s04_receipt.json')
        for sid in receipts:
            if prog=='okada' and sid=='s04':continue
            name=prog+'_'+sid
            if prog=='jin' and sid in ('s10','s18','s24'):continue
            for suffix in ('_rig','_controls'):
                np.testing.assert_array_equal(proof[name+suffix],r[name+suffix])
        reports[prog]=receipts
    metrics=cmp['aggregate']
    counts=new_counts(PREP.glob('jin_*_decisions.jsonl'))
    controls={k:{'change':float(np.max(abs(r[k+'_controls']-b[k+'_controls']))),
                'rig_change':float(np.max(abs(r[k+'_rig']-b[k+'_rig'])))}
              for k in m['shots']}
    results=dict(shared_scene_sha256=expected,
                 shared_path_count=922, geometry_changes=scene_changes,
                 affected_world_paths=world_changes['moved_paths'],source_point_summary=counts,
                 reserved=summarize(m),separate_interval_summary={
                     p:{f:{w:metrics[v][p+'_audit'][f][w] for w in ('rows','frames','balanced_rms','balanced_p90')}
                         for f in ('all','board','ice','ads')} for p in ('jin','okada') for v in ('review',)},
                 checkpoints={k: fingerprint(OUT/'guarded'/f'{k}.npz') for k in ('motion','camera','joint')},
                 previous_geometry_value_sha256=ji.numeric_hash(old.values),
                 original_frame_motion_check=fingerprint(CAND/'movement.json'),
                 actual_prior_programme_comparison=fingerprint(CAND/'actual_previous.json'),
                 changed_camera_shots={k:x for k,x in controls.items() if x['change']>1.e-10 or x['rig_change']>1.e-10},
                 monitored_shots=reports,whole_programme_candidate_acceptance=False)
    for p in ('jin','okada'):
        results['separate_interval_summary'][p]['baseline']={f:{w:metrics['baseline'][p+'_audit'][f][w]
                for w in ('rows','frames','balanced_rms','balanced_p90')} for f in ('all','board','ice','ads')}
    results['local_feature_checks']={}
    for name,features in {'jin_s03':('C','Ad8','Ad9'),'jin_s04':('goal_line_left','Ad4'),
            'jin_s10':('B2','R1','R2','board.top'),'jin_s13':('Ad1','lane3'),
            'jin_s14':('A4','F1','F2','lane1','lane2','goal_line_left','board.top'),
            'jin_s15':('Ad13','Ad14'),'jin_s18':('C','Ad4'),'jin_s21':('Ad17','Ad18','Ad19','C','B2'),
            'jin_s22':('board.top','R1','R2'),'jin_s24':('Ad26','B2','C'),
            'okada_s05':('Ad17','Ad18','Ad19','A4'),'okada_s08':('goal_line_left',)}.items():
        results['local_feature_checks'][name]={f:key_stats(cmp['shot'][name],f,'local',None) for f in features}
    geom=install_npz(EVENT/'scene/field'/('shared_'+v.out+'.npz'),dict(
                    scene_values=state.values,scene_values_sha256=np.asarray(expected),
                    scene_file_sha256=np.asarray(ji.pin(native.scene_path)['sha256']),
                    layout_revision=np.asarray(state.layout['layout_revision'])))
    geom={**geom,'value_sha256':expected,'layout_revision':state.layout['layout_revision'],
          'structure':fingerprint(native.scene_path),'native_library':fingerprint(core.library_path)}
    receipt=CAND/'receipt.json'
    ji.write(receipt,results)
    result= fingerprint(receipt)
    # Store one programme-specific compact state each. Independent time
    # blocks preserve the original native source clock and every rig row.
    files={}
    for prog,pid in (('jin','jin_shuxian_sp'),('okada','okada_mei_sp')):
        directory=EVENT/'programmes'/pid
        if prog=='okada':directory=directory/'fit_v11'
        names=[k for k,s in m['shots'].items() if s['program']==prog]
        times=[b[k+'_times'] for k in names]
        pts=[b[k+'_pts'] for k in names]
        npz=install_npz(directory/(v.out+'_camera.npz'),dict(
            shot_ids=np.asarray([k.split('_')[1] for k in names]),
            camera_indices=np.asarray([m['shots'][k]['camera_index'] for k in names],np.int32),
            control_dt=np.asarray([m['shots'][k]['dt'] for k in names]),
            first_control_time=np.asarray([m['shots'][k]['first'] for k in names]),
            control_offsets=np.r_[0,np.cumsum([len(r[k+'_controls']) for k in names])],
            source_offsets=np.r_[0,np.cumsum(list(map(len,pts)))],
            source_pts=np.concatenate(pts),local_exposure_seconds=np.concatenate(times),
            controls=np.vstack([r[k+'_controls'] for k in names]),
            rig_by_source_pts=np.vstack([r[k+'_rig'] for k in names]),
            scene_values_sha256=np.asarray(expected)))
        manifest=dict(schema='rinkspace.programme_joint_scene_review.v1',event_id='ankara_2026',programme_id=pid,
            review_status='review_candidate_local_improvements',user_visual_acceptance=False,
            source=fingerprint(m['shots'][names[0]]['source']),
            source_native_PTS=dict(first=int(pts[0][0]),last=int(pts[-1][-1]),time_base='1/12800',
                                   exposures=sum(map(len,pts))),
            geometry=geom,camera=npz,camera_model='native rolling row',actual_shot_counts={k.split('_')[1]:len(z) for k,z in zip(names,pts)},
            previous_binding=a['ji_ref'] if prog=='jin' else a['original_ref'],
            current_unified_baseline=fingerprint(PREP/'baseline.npz'), prepared_source={
                'registry':fingerprint(PREP/'input_summary.json'),
                'per_shot_uv':{k.split('_')[1]:fingerprint(PREP/(k+'_decisions.jsonl')) for k in names}},
            comparison=fingerprint(CAND/'comparison.json'),
            comparison_to_previous_accepted_binding=fingerprint(CAND/'actual_previous.json'),
            verification=result,
            direct_reserved_source_diagnostic=fingerprint(SOURCE/(prog+'.json')) if (SOURCE/(prog+'.json')).exists() else None,
            clips={k:dict(file=d['video'],start=d['source_first_seconds'],end=d['source_last_seconds'],
                           exact_frames=d['decoded_frames'],overlay_PTS_valid=d['native_PTS_equal'])
                   for k,d in reports[prog].items()})
        files[prog]=install_json(directory/(v.out+'.json'),manifest)
    print(json.dumps(dict(geometry=geom['path'],bindings=files,verification=result,
                          stats=results['separate_interval_summary'],source=counts,reserved=results['reserved']),
                     ensure_ascii=False,indent=2))
if __name__=='__main__':main()
