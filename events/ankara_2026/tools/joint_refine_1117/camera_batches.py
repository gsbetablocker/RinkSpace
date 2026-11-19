"""Serial full-source camera batches; one main journal, no observation thinning."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
from shared import OUT, PREPARED, arrays, read, save, sha, write


def run(args):
    folder=OUT/args.out;folder.mkdir(parents=True,exist_ok=False)
    main=folder/'solve.jsonl'
    def log(event,**values):
        with main.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**values),ensure_ascii=False)+'\n')
    setup=read(PREPARED/'setup.json')
    names=[name for name,s in setup['shots'].items() if s['program'] in args.programmes and (not args.shots or name in args.shots)]
    if not names:raise ValueError('No selected source shots')
    groups=read(args.groups_json)['groups'] if args.groups_json else [dict(id=name,shots=[name]) for name in names]
    grouped=[name for group in groups for name in group['shots']]
    if len(grouped)!=len(set(grouped)) or set(grouped)!=set(names):
        raise ValueError('Calibration groups must partition exactly the selected source shots')
    options=args.options[1:] if args.options[:1]==['--'] else args.options
    if any(x in options for x in ['--out','--parent','--mode','--shots','--programmes']):
        raise ValueError('Batch selection and parent cannot be overridden by stage options')
    log('batch_start',shots=names,parent=args.parent,prepared=str(PREPARED),options=options,
        policy='every admitted observation retained; independent shots run serially to fit device memory')
    parent=args.parent;receipts=[]
    initial=arrays(OUT/parent/'state.npz')
    for group in groups:
        name=group['id'];selected=group['shots']
        stage=str(Path(args.out)/name)
        command=[sys.executable,str(Path(__file__).with_name('solve.py')),'--out',stage,
                 '--mode','camera','--parent',parent,'--shots',*selected,*options]
        environment=dict(os.environ,RINKSPACE_JOINT_MAIN_LOG=str(main))
        log('batch_shot_start',shot=name,stage=stage,log=str(OUT/stage/'solve.jsonl'))
        print('START',name,str(main),flush=True)
        with (folder/(name+'_console.txt')).open('x',encoding='utf-8') as output:
            process=subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT)
        if process.returncode:
            log('batch_shot_failed',shot=name,returncode=process.returncode)
            raise RuntimeError(f'{name} solver failed; last successful parent: {parent}')
        state=arrays(OUT/stage/'state.npz')
        np.testing.assert_array_equal(state['scene_values'],initial['scene_values'])
        receipt=dict(shot=name,shots=selected,stage=stage,state_sha256=sha(OUT/stage/'state.npz'),report=read(OUT/stage/'report.json'))
        receipts.append(receipt);parent=stage
        completed=[shot for r in receipts for shot in r['shots']]
        write(folder/'progress.json',dict(completed=completed,pending=[n for n in names if n not in completed],current_parent=parent))
        log('batch_shot_finish',shot=name,state_sha256=receipt['state_sha256'])
    final=arrays(OUT/parent/'state.npz')
    for name,s in setup['shots'].items():
        if name in names:continue
        for suffix in ('_pts','_times','_controls','_rigs'):
            np.testing.assert_array_equal(final[name+suffix],initial[name+suffix])
    save(folder/'state.npz',**final)
    write(folder/'batch_report.json',dict(parent=args.parent,shots=names,stages=receipts,
        unchanged_unselected_cameras=True,unchanged_shared_geometry=True,
        user_visual_acceptance=False,state_sha256=sha(folder/'state.npz')))
    log('batch_finish',state=str(folder/'state.npz'),shots=names)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--parent',required=True)
    parser.add_argument('--programmes',nargs='+',required=True)
    parser.add_argument('--shots',nargs='+')
    parser.add_argument('--groups-json',help='Explicit source installation groups; each group is solved together')
    parser.add_argument('options',nargs='...')
    run(parser.parse_args())
