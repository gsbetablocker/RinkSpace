from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from spin_model import (JOB, r, pts, config, curve, clock, doc, state as refit_state, anatomy_values,
                        ratio, camera, assembly, moving_projection, PARENT)
from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, evaluate_curve
from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface, OrientedFaceWitness, witnesses_packet
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from experiments.skate_video_v1.inputs import atomic_json, read
from experiments.skate_video_v1.motion_review.scoped_fit import geometry, save_candidate


GUARD = load(JOB/'parent_replay/scoped_guard_baseline.npz')
STAGE = load(JOB/'parent_replay/solve/candidate.npz')
SCAN = load(doc['programme_structure']['surface_witnesses'])
GEOM = CollisionSurface(r,doc)
GUARD_CFG = config['guard']
ICE_TOL = config['ice']['tolerance_m']
FIELDS = ('sample','vertex','face_vertices','bary','normal','initial_gap','face_a','face_b')
global_floor = read(JOB/'parent_replay/final_metrics.json')['physical_diagnostics']['surface_intersection']['worst_signed_gap_m']
allowance = GEOM.contact_allowance


class Check:
    def __init__(self, idx):
        self.idx=idx
        w=clock[np.searchsorted(pts,config['focus_pts'][idx])]
        ctrl=np.flatnonzero((curve.knots[:-4]<w[1]) & (curve.knots[4:]>w[0]))
        self.ctrl=ctrl
        a,b=curve.knots[[ctrl[0],ctrl[-1]+4]]
        self.native=(clock>a)&(clock<b)
        self.n=np.flatnonzero(self.native)
        self.angle_native=self.native[:-1]|self.native[1:]
        self.step_rows=np.flatnonzero(self.angle_native)
        self.rot_times=clock[self.step_rows[0]:self.step_rows[-1]+2]
        self.ice_times=STAGE['physics_times'][(STAGE['physics_times']>a)&(STAGE['physics_times']<b)]
        names,h0=component_ice(r.binding,refit_state(curve,self.ice_times))
        self.ice_names=names
        self.radii=np.asarray([ICE_TOL.get(k,ICE_TOL['solid']) for k in names])
        self.ice_baseline=np.maximum(-h0-self.radii,0).max(0)
        self.left=int(np.searchsorted(SCAN['times'],a,side='right'))
        self.right=int(np.searchsorted(SCAN['times'],b,side='left'))
        self.surface_times=SCAN['times'][self.left:self.right]
        selected=(SCAN['sample']>=self.left)&(SCAN['sample']<self.right)
        self.rows={k:SCAN[k][selected] for k in FIELDS}
        self.rows['sample']-=self.left
        self.before=None
        self.body_poses = refit_state(curve,clock[self.n])
        self.baseline_ankle = np.maximum(0.,GUARD['anatomy'][self.n])
        self.skate=[]
        roles=load(doc['whole_skate_source']['path'])
        for foot in (0,1):
            keep=(STAGE['whole_skate_match_feet']==foot) & self.native[STAGE['whole_skate_match_frames']]
            if keep.any():
                indices=STAGE['whole_skate_match_frames'][keep]
                sel=(STAGE['whole_skate_match_vertices'][keep],STAGE['whole_skate_match_fraction'][keep])
                uv=STAGE['whole_skate_match_target'][keep]
                times=(clock[indices,None]+camera.readout_times(indices)[:,None]*np.linspace(-.5,.5,5)).ravel()
                cap=GUARD['silhouette'][indices,foot]+GUARD_CFG['maximum_new_trusted_source_sigmas']*roles['sigma_px'][indices,foot]
                self.skate.append((foot,indices,sel,uv,times,cap))

    def surface_gap(self,V,pack):
        i=pack['sample']
        point=V[i,pack['vertex']]
        faces=V[i[:,None],pack['face_vertices']]
        sign=GEOM.response_signs(np.c_[pack['face_a'],pack['face_b']])[:,0]
        return OrientedFaceWitness.from_geometry(point,faces,pack['bary'],sign).gap

    def fixed_checks(self,current):
        out=[]
        A=evaluate_curve(current,self.rot_times)['rotations']
        cur=np.rad2deg(np.linalg.norm(_log(A[:-1].swapaxes(-1,-2)@A[1:]),axis=-1)).max(0)
        cap=GUARD['angle'][self.angle_native].max(0)+GUARD_CFG['world_angular_step_budget_deg']
        for j in np.flatnonzero(cur>cap+1.e-7):
            out.append(('joint_step',int(j),float(cur[j]),float(cap[j])))
        actual=anatomy_values(refit_state(current,clock[self.n]),self.n)
        peak=actual.max(0)-self.baseline_ankle.max(0)
        mean=np.maximum(actual-self.baseline_ankle,0).mean(0)
        for name,value,cap in (('anatomy_peak',peak,GUARD_CFG['anatomy_peak_budget']),
                               ('anatomy_mean',mean,GUARD_CFG['anatomy_new_component_budget'])):
            for j,k in zip(*np.nonzero(value>cap+1.e-7),strict=True):
                out.append((name,[int(j),int(k)],float(value[j,k]),cap))
        names,h=component_ice(r.binding,refit_state(current,self.ice_times))
        ic=np.maximum(-h-self.radii,0).max(0)
        for j,k in zip(*np.nonzero(ic>self.ice_baseline+GUARD_CFG['component_height_numerical_m']+1.e-9),strict=True):
            out.append(('ice_mm',[int(j),names[k]],float(ic[j,k]*1000),float((self.ice_baseline[j,k]+GUARD_CFG['component_height_numerical_m'])*1000)))
        for foot,indices,sel,uv,times,cap in self.skate:
            state=refit_state(current,times)
            xyz=assembly.points(state,foot,sel,(state['R'],state['O']))
            image=moving_projection(camera,xyz.reshape(len(indices),5,-1,3),indices)['uv']
            dist=np.median(np.linalg.norm(image-uv,axis=-1),axis=1)
            for j in np.flatnonzero(dist>cap+1.e-7):
                out.append(('outline_px',[int(foot),float(pts[indices[j]]/12800.)],float(dist[j]),float(cap[j])))
        return out

    def surface(self,current,refresh=True):
        def vertices(c):
            return GEOM.vertices(refit_state(c,self.surface_times))
        if self.before is None:
            self.before=vertices(curve)
        trial=vertices(current)
        fixed=self.surface_gap(trial,self.rows)
        out=[]
        if fixed.min()<global_floor-1.e-6:
            sample_ids=np.unique(self.rows['sample'][fixed<global_floor-1.e-6])
            for k in sample_ids:
                ind=np.flatnonzero(self.rows['sample']==k)
                j=ind[np.argmin(fixed[ind])]
                f,g=GEOM.parts[[self.rows['face_a'][j],self.rows['face_b'][j]]]
                out.append(('old_surface', [str(GEOM.part_names[f]),str(GEOM.part_names[g]),float(self.surface_times[self.rows['sample'][j]]+pts[0]/12800.)],
                            float(fixed[j]),global_floor))
        if not refresh:
            return out
        def scan(item):
            i,V=item
            witness,_=GEOM.scan_sample(V,self.surface_times[i])
            selected={}
            for v in witness:
                pair=tuple(sorted(map(int,GEOM.parts[[v[5],v[6]]])))
                if pair not in selected or selected[pair][4]>v[4]:
                    selected[pair]=v
            return [(i,w) for w in selected.values()]
        with ThreadPoolExecutor(max_workers=8) as pool:
            records=[v for rows in pool.map(scan,enumerate(trial)) for v in rows]
        fresh=witnesses_packet(records)
        together={k:np.r_[self.rows[k],fresh[k]] for k in FIELDS}
        p=np.sort(GEOM.parts[np.c_[together['face_a'],together['face_b']]],axis=1)
        h0,h1=(self.surface_gap(v,together) for v in (self.before,trial))
        for left,right in np.unique(p,axis=0):
            keep=(p[:,0]==left)&(p[:,1]==right)
            old,new=float(h0[keep].min()),float(h1[keep].min())
            floor=min(old,0.)-allowance
            if new < floor:
                j=np.flatnonzero(keep)[np.argmin(h1[keep])]
                sec=float(self.surface_times[together['sample'][j]]+pts[0]/12800.)
                out.append(('fresh_surface',[str(GEOM.part_names[left]),str(GEOM.part_names[right]),sec],new,floor))
        return out


def check(idx, file, refresh, fraction=1.):
    p=load(file)
    trial=curve_from_packet(p)
    worker=Check(idx)
    if fraction != 1.:
        rp=curve.position_controls+fraction*(trial.position_controls-curve.position_controls)
        rr=curve.rotation_controls.copy()
        from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp
        changed=worker.ctrl
        rr[changed]=rr[changed]@_exp(fraction*_log(rr[changed].swapaxes(-1,-2)@trial.rotation_controls[changed]))
        trial=geometry(curve,rp,rr)
    return worker.fixed_checks(trial)+worker.surface(trial,refresh=refresh)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--fraction', type=float, default=1.)
    parser.add_argument('--fixed', action='store_true')
    args = parser.parse_args()
    errors = check(0, JOB / 'proposals' / (args.name + '.npz'), not args.fixed, args.fraction)
    print(errors, flush=True)
    atomic_json(JOB / f'{args.name}_check_{args.fraction:.2f}.json', errors)
