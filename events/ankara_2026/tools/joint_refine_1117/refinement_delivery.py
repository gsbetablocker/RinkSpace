"""Construct and verify one numerical model for the Ankara source-clock review."""
from argparse import ArgumentParser
import numpy as np
from lane_end_evidence import BASE, WORK
from review import camera_state
from shared import PREPARED, PROGRAMMES, arrays, numeric_sha, read, save, sha, write
from source_bundle import flow, native_run, relative


def seed():
    common=relative(WORK/'camera_assembly');state=camera_state(common)
    original=state['scene_values'].copy();values=original.copy();geometry={}
    for label in ('ends','widths','R2','R3','F24','R4_radii'):
        path=WORK/('bundle_'+label)/'state.npz';v=arrays(path)['scene_values']
        ix=np.flatnonzero(v!=original)
        if np.any(values[ix]!=original[ix]):raise ValueError(('overlapping coordinate',label))
        values[ix]=v[ix];geometry[label]=dict(path=str(path),sha256=sha(path),columns=ix.tolist())
    state['scene_values']=values;state['scene_values_sha256']=numeric_sha(values)
    source={}
    physical={'jin_s08':'physical','jin_s13':'optical','jin_s15':'physical','jin_s16':'mount_trials',
        'jin_s18':'physical','jin_s19':'optical','jin_s23':'physical','jin_s24':'physical','chen_s20':'optical'}
    details={'okada_s02','jin_s10','nayeon_s02','nayeon_s04','chen_s14'}
    for name in read(PREPARED/'setup.json')['shots']:
        if name in physical:stage=physical[name]
        elif name in details:stage='fine'
        elif (WORK/'polish_pilot'/name/'report.json').exists():stage='polish_pilot'
        elif (WORK/'polish'/name/'report.json').exists():stage='polish'
        else:stage='cameras'
        path=WORK/stage/name/'state.npz';record=arrays(path)
        for k in ('_controls','_rigs','_pts','_times','_active_folds'):
            if k in ('_pts','_times'):np.testing.assert_array_equal(state[name+k],record[name+k])
            state[name+k]=record[name+k]
        source[name]=dict(stage=stage,path=str(path),sha256=sha(path))
    save(WORK/'release_seed/state.npz',**state)
    write(WORK/'release_seed/inputs.json',dict(geometry=geometry,cameras=source,
        role='combined initialization; cameras re-optimized on the audited original observations'))


def finish_cameras(programmes,parent):
    for name,s in read(PREPARED/'setup.json')['shots'].items():
        if s['program'] not in programmes:continue
        extra=['--photo-bindings',WORK.parent/'repair/prepared_v2/nayeon_photo_bindings.npz'] if s['program']=='nayeon' else []
        native_run(WORK/'release_cameras'/name,parent,[
            '--mode','camera','--shots',name,'--stride','8','--unit-weights','--station-seed-dir',WORK/'evaluation/camera',
            *flow(.78),*extra])


def assemble_final(parent,overrides,output='selected'):
    state=camera_state(parent);shots={}
    for name in read(PREPARED/'setup.json')['shots']:
        p=WORK/overrides.get(name,'release_cameras/'+name)
        data=arrays(p/'state.npz')
        for key in ('scene_values',name+'_pts',name+'_times'):np.testing.assert_array_equal(state[key],data[key])
        for suffix in ('_rigs','_controls','_active_folds'):state[name+suffix]=data[name+suffix]
        shots[name]=dict(parent=relative(p),state_sha256=sha(p/'state.npz'))
    out=WORK/output;save(out/'state.npz',**state)
    write(out/'inputs.json',dict(scene_parent=parent,scene_numeric_sha256=numeric_sha(state['scene_values']),
        prepared=str(PREPARED),shots=shots))


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('task',choices=['seed','cameras','assemble'])
    ap.add_argument('--parent',default='lane_motion_1118/release_seed')
    ap.add_argument('--programmes',nargs='+',default=list(PROGRAMMES))
    ap.add_argument('--overrides');ap.add_argument('--output',default='selected');args=ap.parse_args()
    if args.task=='seed':seed()
    elif args.task=='cameras':finish_cameras(args.programmes,args.parent)
    else:assemble_final(args.parent,read(args.overrides) if args.overrides else {},args.output)
