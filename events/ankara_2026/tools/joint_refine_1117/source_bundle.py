"""Bundle the four original-source inputs and refit supported camera and ice coordinates."""
from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
import time
import numpy as np
from shared import OUT, PREPARED, arrays, read, save, write
from lane_end_evidence import BASE, ENDS, WORK
from review import camera_state, native_for


def relative(path):
    return Path(path).relative_to(OUT).as_posix()


def native_run(folder,parent,options):
    if (folder/'report.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    executable=Path(__file__).with_name('solve.py')
    cmd=[sys.executable,'-B',str(executable),'--out',relative(folder),'--parent',parent,'--all-development',
         '--station-seed',BASE+'/full_qa',*map(str,options)]
    write(folder/'invocation.json',dict(argv=cmd,prepared=str(PREPARED)))
    with (folder/'driver.txt').open('w',encoding='utf-8') as stream:
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)


def assemble():
    setup=read(PREPARED/'setup.json')['shots'];parent=relative(WORK/'end_geometry')
    state=camera_state(parent);sources={}
    for name in setup:
        p=WORK/'cameras'/name
        while not (p/'report.json').exists():time.sleep(3)
        current=camera_state(relative(p))
        for suffix in ('_controls','_rigs','_active_folds'):
            state[name+suffix]=current[name+suffix]
        sources[name]=relative(p)
    save(WORK/'camera_assembly/state.npz',**state)
    write(WORK/'camera_assembly/inputs.json',dict(parent=parent,shots=sources))


def flow(gain):
    return ['--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
        '--texture-dir',WORK/'texture','--rebind-texture','--triplet-scale','3.8','--flow-gain',str(gain)]


def paint(branch):
    setup=read(PREPARED/'setup.json')['shots']
    parent=relative(WORK/'camera_assembly')
    base=['--mode','joint','--unit-weights','--stride','9','--station-seed-dir',WORK/'evaluation/camera',
          '--scene-blocks','scene/ankara_2026.field/ice_tolerance',*flow(.65)]
    if branch=='ends':
        native_run(WORK/'bundle_ends',parent,[*base,'--camera-shots','jin_s22','chen_s07',
            '--passive-paths',*ENDS,'--ice-coordinates',
            *[f'lane_right_{key}/{i}' for key in ('position','slope') for i in range(4)]])
    elif branch=='widths':
        native=native_for(parent,'jin')
        paths=[k for k,v in native.paths.items() if v['label'].startswith('ice.lane') and ':edge' in v['label']]
        native_run(WORK/'bundle_widths',parent,[*base,'--camera-shots','jin_s22','chen_s18',
            '--passive-paths',*paths,'--ice-coordinates',
            *[f'lane_width_at_{end}/{i}' for end in ('F','R') for i in range(4)]])
    elif branch=='F24':
        native=native_for(parent,'jin')
        paths=[k for k,v in native.paths.items() if v['label'].split(':')[0] in ('ice.F2','ice.F4')]
        native_run(WORK/'bundle_F24',parent,[*base,'--camera-shots','jin_s18','--passive-paths',*paths,
            '--ice-coordinates','F_column_position/1','F_column_position/3',
            *[f'house_ring_radius/{4*i+j}' for i in (1,3) for j in (2,3)]])
    elif branch.startswith('R'):
        number=int(branch[1:]);house=number+3
        native=native_for(parent,'jin')
        paths=[key for key,v in native.paths.items() if v.get('label','').split(':')[0]=='ice.'+branch]
        if len(paths)<3:raise ValueError((branch,paths))
        # The row alignment, column position, and four radii are separate for every house.
        # Unreliable centre and inner paint identities keep their inherited radius.
        active={1:'nayeon_s08',2:'chen_s07',3:'chen_s07',4:'okada_s03'}[number]
        native_run(WORK/('bundle_'+branch),parent,[*base,'--camera-shots',active,
            '--passive-paths',*paths,'--ice-coordinates',f'R_column_position/{number-1}',
            f'FR_row_alignment/{number-1}',f'house_ring_radius/{4*house+2}',f'house_ring_radius/{4*house+3}'])
    else:raise ValueError(branch)


def polish(names,stage,parent=None):
    for name in names:
        if parent:
            seed=parent
        else:
            p=WORK/'cameras'/name
            while not (p/'report.json').exists():time.sleep(3)
            seed=relative(p)
        extra=['--photo-bindings',OUT/'repair/prepared_v2/nayeon_photo_bindings.npz'] if name.startswith('nayeon') else []
        native_run(WORK/stage/name,seed,['--mode','camera','--shots',name,'--unit-weights',
            '--station-seed-dir',WORK/'evaluation/camera','--stride','9',*flow(.65),*extra])


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['assemble','paint','polish'])
    ap.add_argument('--branches',nargs='+');ap.add_argument('--shots',nargs='+')
    ap.add_argument('--stage',default='polish');ap.add_argument('--parent')
    args=ap.parse_args()
    if args.mode=='assemble':assemble()
    elif args.mode=='paint':
        for branch in args.branches:paint(branch)
    else:polish(args.shots or list(read(PREPARED/'setup.json')['shots']),args.stage,args.parent)
