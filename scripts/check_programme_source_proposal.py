"""Check raw and weighted source contour derivatives on the saved common state."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.source_skate_structure_factors import FullSkateStructureFactors

def main():
    p=argparse.ArgumentParser()
    for key in ('project','prepared','source-evidence','parameter','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        root,camera,_,doc,_=load_reference(a.project,a.prepared,a.source_evidence);saved=load(a.parameter)
        ref=absolute_reference(root,curve_from_packet(saved),root=a.prepared);ref.relative_skate_curve=shoe_curve_from_packet(saved)
        ref.binding=copy.copy(root.binding);ref.binding.reference=ref;ref.binding.x=saved['x'][:17].copy();ref.binding._packet=None;ref.binding._parameter_key=None
        state=SequenceState(ref,(ref.times[0],ref.times[-1]))
        problem=SimpleNamespace(reference=ref,binding=ref.binding,camera=camera,
            body=SimpleNamespace(state=state,tau=float(camera.p['rig'][37])),size=state.size+17)
        document=copy.deepcopy(doc['full_source_skate_structures'])
        document['rows']=[r for r in document['rows']if r['pts']in (1052416,1065216)]
        factor=FullSkateStructureFactors(problem,document);x=np.zeros(state.size)
        first=factor.evaluate(x,True);J=first['jac'][:,17:];raw=factor.proposal_pixel_jacobian[:,17:]
        def points(value):return np.concatenate([v['model_closest_uv']for v in value['metrics']['visible_runner_contours']]).ravel()
        rng=np.random.default_rng(1108);checks=[]
        for family in ('all','root','leg','shoe'):
            direction=rng.normal(size=state.size)
            if family!='all':
                direction[:]=0
                if family=='shoe':direction[state.body_size:]=rng.normal(size=state.shoe_size)
                else:
                    columns=np.arange(3)if family=='root'else np.r_[np.arange(6,24),np.arange(27,36)]
                    ii=(np.arange(state.m)[:,None]*69+columns).ravel();direction[ii]=rng.normal(size=len(ii))
            direction/=np.linalg.norm(direction);h=2e-5
            plus=factor.evaluate(x+h*direction,False);minus=factor.evaluate(x-h*direction,False)
            for name,analytic,numeric in (
                ('raw',raw@direction,(points(plus)-points(minus))/(2*h)),
                ('weighted',J@direction,(plus['residual']-minus['residual'])/(2*h))):
                checks.append(dict(family=family,kind=name,relative_error=float(np.linalg.norm(analytic-numeric)/max(np.linalg.norm(numeric),1e-12)),
                    maximum_absolute_error=float(np.max(abs(analytic-numeric)))))
        prediction=factor.proposal_pixel_prediction;radius=np.array([np.sqrt(v['region_shape_px2'][0,0])for v in factor.factor_rows])
        inside=np.linalg.norm(prediction-factor.source_uv,axis=1)<=radius
        rows=np.c_[2*np.flatnonzero(inside),2*np.flatnonzero(inside)+1].ravel()
        result=dict(parameter=dict(path=str(a.parameter),sha256=digest(a.parameter)),source_pts=sorted({r['pts']for r in document['rows']}),
            source_points=len(prediction),inside_band_points=int(inside.sum()),inside_weighted_J_max=float(np.max(abs(J[rows].data),initial=0)),
            inside_raw_J_max=float(np.max(abs(raw[rows].data),initial=0)),checks=checks)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result),flush=True)
        assert all(v['relative_error']<2e-4 for v in checks)
        assert result['inside_band_points']>0 and result['inside_weighted_J_max']==0 and result['inside_raw_J_max']>0
if __name__=='__main__':main()
