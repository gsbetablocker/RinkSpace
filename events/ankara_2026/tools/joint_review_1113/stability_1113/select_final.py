"""Commit evaluated stability corrections to a new programme review state."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from camera_audit import HERE,PREP,ji,sj,vj,setup,native_scene
from compare import compare
from refit import rig_seed
from measure import stat


SELECTION={
's04':('composed','fixed mount and ice/advertising held checks'),
's06':('composed','fixed mount and improved worst three-frame motion'),
's07':('composed','local oscillation near 63.1, earlier original pan retained'),
's10':('composed','local drift near initial pan and 83–85 seconds'),
's12':('composed','fixed mount with reduced 160 ms drift'),
's13':('composed','local cut-head correction, advertising and lane holdout'),
's14':('composed','local correction preserves left goal, lanes and advertisements'),
's16':('composed','reduced source displacement error around cut head'),
's17':('composed','fixed mount and improved ice holdout'),
's18':('composed','local fix supported by source motion and Ad4'),
's20':('mount','fixed mount with improved ice and advertising'),
's22':('fixed_option','late pan, retaining existing early source motion'),
's24':('composed','local overshoot and 183-second focal transient'),
}

STAGED=HERE/'selected.npz'


def layer_aggregate(dirs,meta,history,series_path):
    values={}
    for material in ('board','ice'):
        for term in ('adjacent','jitter','lag8'):
            key=material+'_'+term+'_all'
            z=[]
            for k in meta['shots']:
                if k.startswith(dirs):
                    f=series_path/(k+'_evaluation.npz')
                    p=vj.load_checkpoint(f if f.exists() else HERE/'baseline_motion'/(k+'.npz'))
                    z.extend(p[key].tolist())
            values[material+'_'+term]=stat(np.array(z))
    return values


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--options',action='store_true')
    a=p.parse_args()
    m,core,hist,state=setup()
    native=native_scene(m,core,state)
    cache=HERE/'baseline_abs'
    out=HERE/'adopted'
    out.mkdir(exist_ok=True)
    plan=SELECTION if not a.config else ji.read(Path(a.config))
    if a.options:
        for sid in ('s22','s24'):
            key='jin_'+sid
            d=vj.shot_from_inputs(key,m,hist)
            d.rigs=state[key+'_rig']
            r=rig_seed(d,d.rigs,'rigid',None)
            rr=vj.load_checkpoint(HERE/'composed'/(key+'.npz'))
            data=HERE/'fixed_option'
            data.mkdir(exist_ok=True)
            sj.atomic_arrays(data/(key+'.npz'),controls=rr['controls'],rig=r,scene_values=rr['scene_values'],pts=d.pts)
        return
    results={}
    tested={}
    for sid,(folder,why) in plan.items():
        name='jin_'+sid
        f=HERE/folder/(name+'.npz')
        d=vj.shot_from_inputs(name,m,hist)
        d.controls=state[name+'_controls']
        d.rigs=state[name+'_rig']
        rows=vj.load_checkpoint(PREP/(name+'.npz'))
        tracks=vj.load_checkpoint(HERE/'source_tracks'/(name+'.npz'))
        fixed=vj.load_checkpoint(f)
        np.testing.assert_array_equal(fixed['pts'],d.pts)
        np.testing.assert_array_equal(fixed['scene_values'],state['scene_values'])
        z={name+'_controls':fixed['controls'],name+'_rig':fixed['rig']}
        check=compare(d,core,native,rows,tracks,state,z,out,cache/(name+'.npz'))
        assert check['camera']['nonpositive_focal_rows']==0 and check['camera']['brown_unproven_frames']==0
        assert all(check['absolute'][k]['missing_after']<=check['absolute'][k]['missing_before']+2 for k in ('ice','ads','rail') if k in check['absolute'])
        results[name]=z
        tested[name]=dict(purpose=why,candidate=ji.pin(f),results=check)
        print(name,folder,'validated',flush=True)
    b=dict(state)
    for k,z in results.items():b.update(z)
    np.testing.assert_array_equal(b['scene_values'],state['scene_values'])
    report=dict(
        mode='whole-source review, source-track fit and rolling-row camera repair',
        scene_unchanged=True,geometry_sha256=ji.numeric_hash(b['scene_values']),
        frame_count_jin=int(sum(len(hist[k+'_pts']) for k in m['shots'] if k.startswith('jin_'))),
        frame_count_okada=int(sum(len(hist[k+'_pts']) for k in m['shots'] if k.startswith('okada_'))),
        compared_reference='Okada original 1102 fixed per-installation cameras',
        full_scan=ji.pin(HERE/'source_motion_before.json'),
        full_precheck_camera=ji.pin(HERE/'camera_before.json'),
        full_programme={
            'jin_before':layer_aggregate('jin',m,hist,HERE/'baseline_motion'),
            'jin_after':layer_aggregate('jin',m,hist,out),
            'okada_1102':layer_aggregate('okada',m,hist,HERE/'baseline_motion'),
        },
        physical_scope='finite proper native orientations, focal positivity, Brown central injectivity lower bound at five readout samples per exposure, and variable mounting audit',
        same_shot_flow_scope='Independent direct original texture tracking without fitting predictions as optical-flow initials. Programme comparisons have different view and qualified feature distributions.',
        selection=plan,changes=tested)
    sj.write_json(HERE/'selection.json',report)
    sj.atomic_arrays(STAGED,**b)
    print('Saved',len(results),STAGED,flush=True)


if __name__=='__main__':
    main()
