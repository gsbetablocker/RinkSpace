"""Compose continuous, source-checked corrections on the published source clock."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from camera_audit import HERE,ji,sj,vj,setup
from refit_local import quadratic_halo


def blend_weights(x,lo,hi,width):
    left=np.ones_like(x) if lo is None else np.clip((x-lo)/width,0.,1.)
    right=np.ones_like(x) if hi is None else np.clip((hi-x)/width,0.,1.)
    return np.sin(np.minimum(left,right)*np.pi*.5)**2


# Overrides at programme time. Named source folds and fresh static tracks are
# checked again after merging, because blends change the exact camera.
PLAN={
    's04': ('rigid_local',None),
    's06': ('rigid_local',None),
    's07': ('local',[(62.82,63.45,.13,1.)]),
    's10': ('local',[(None,78.74,.12,1.),(83.16,85.50,.18,1.),(88.68,None,.12,1.)]),
    's12': ('rigid_local',None),
    's13': ('local',[(None,106.30,.16,1.),(112.06,113.86,.17,.90)]),
    's14': ('local',[(122.54,None,.16,1.)]),
    's16': ('local',[(None,128.42,.17,1.),(131.72,None,.14,1.)]),
    's17': ('rigid_local',None),
    's18': ('local',[(136.78,138.06,.13,1.)]),
    's20': ('rigid_local',None),
    's21': ('motion4',[(148.72,None,.17,.65)]),
    's22': ('local',[(152.76,None,.19,1.)]),
    's24': ('local',[(171.12,171.98,.13,1.),(172.06,178.83,.21,.7),(182.94,183.64,.14,1.)])
}


def combine(d,c,r,folder,window):
    q=vj.load_checkpoint(HERE/folder/(d.name+'.npz'))
    np.testing.assert_array_equal(d.pts,q['pts'])
    candidate=q['controls']
    rr=q['rig']
    if window is None:
        return candidate.copy(),rr.copy()
    np.testing.assert_array_equal(r,rr)
    raw=d.first+d.dt*np.arange(len(c))-d.times[0]+d.pts[0]/12800.
    strength=np.maximum.reduce([blend_weights(raw,lo,hi,w)*a for lo,hi,w,a in window])
    return c+strength[:,None]*(candidate-c),r.copy()


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--folder',default='composed')
    p.add_argument('--configuration')
    a=p.parse_args()
    m,core,hist,base=setup()
    out=HERE/a.folder
    out.mkdir(exist_ok=True)
    plan=PLAN if not a.configuration else ji.read(Path(a.configuration))
    for k,(folder,window) in plan.items():
        name='jin_'+k
        shot=vj.shot_from_inputs(name,m,hist)
        c,r=combine(shot,base[name+'_controls'],base[name+'_rig'],folder,window)
        sj.atomic_arrays(out/(name+'.npz'),controls=c,rig=r,scene_values=base['scene_values'],pts=shot.pts)
    sj.write_json(out/'composition.json',plan)


if __name__=='__main__':
    main()
