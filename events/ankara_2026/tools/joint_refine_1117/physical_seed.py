"""Repair an inadmissible inherited installation with a same-shot fixed seed.

The selected rig is an existing source-era numerical state, not a measurement.
Source pixels and motion clocks are unchanged; a source fit is required next.
"""
from shared import OUT, arrays, read, save, sha, write
from review import native_for
import numpy as np


def create(parent='joint_board_free', target='physical_source_seed'):
    state=arrays(OUT/parent/'state.npz')
    native=native_for(parent,'jin');b=np.asarray(native.scene['board'])
    setup=read(OUT/'prepared/setup.json');changes={}
    for name,s in setup['shots'].items():
        rig=state[name+'_rigs'];pivot=rig[:,15:18]
        u=np.maximum(0.,(abs(pivot[:,0])-(b[0]-b[2]))/b[2])
        v=np.maximum(0.,(abs(pivot[:,1])-(b[1]-b[3]))/b[3])
        valid=(u**b[4]+v**b[4]>=1.)&(pivot[:,2]>0)&(np.linalg.norm(rig[:,18:21],axis=1)>1.e-12)
        valid&=(pivot[:,0]<0) if s['camera_index']==2 else (pivot[:,1]<=-b[1])
        if valid.all():continue
        candidates=np.flatnonzero(valid)
        if not len(candidates):raise ValueError('No same-shot admissible installation seed: '+name)
        centre=np.median(pivot,axis=0)
        chosen=candidates[np.argmin(np.linalg.norm(pivot[candidates]-centre,axis=1))]
        state[name+'_rigs']=np.repeat(rig[chosen:chosen+1],len(rig),axis=0)
        changes[name]=dict(original_invalid_exposures=int((~valid).sum()),
            same_shot_selected_exposure=int(chosen),source_pts=int(state[name+'_pts'][chosen]),
            chosen_pivot=pivot[chosen].tolist(),
            policy='one existing physically admissible rig for the complete shot; fit only actual source observations next',
            legacy_state_sha256=sha(OUT/parent/'state.npz'))
    save(OUT/target/'state.npz',**state)
    write(OUT/target/'migration.json',dict(parent=parent,changes=changes,
        source_pixels_unchanged=True,source_clock_unchanged=True,seed_is_not_an_accuracy_claim=True))
    print(changes)


if __name__=='__main__':create()
