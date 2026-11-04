from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation,RotationSpline
from calibration.program_tracking.body_tracking.skate_integration import build
from calibration.program_tracking.body_tracking.exmr_prior import load,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1];base=Path(a.base_root);out=Path(a.output)
    ref,_,_,_=build(project,base,out/'collar_diagnostic/shared')
    ref.binding.x=load(project/'events/ankara_2026/programmes/okada_mei_sp/skate_s01_1103.npz')['x'][:17].copy()
    ref.binding.enable_collar_constraints();results={}
    for label in ['gvhmr_fixed','gvhmr_joint1','exmr_prefoot','exmr_postfoot','exmr_final']:
        case='gvhmr' if label.startswith('gvhmr') else ('exmr_final' if label=='exmr_postfoot' else label)
        initial=load(out/'common'/case/'initial.npz');pts=initial['pts'];local=initial['local_R']
        if label=='gvhmr_joint1':local=load(out/'common/gvhmr/fitted/result.npz')['local_R_mid']
        if label=='exmr_postfoot':
            raw=load(out/'exmr_postfoot.npz');t=(raw['pts']-pts[0])/12800.;q=(pts-pts[0])/12800.
            rotations=Rotation.from_rotvec(raw['body_pose'].reshape(-1,3)).as_matrix().reshape(-1,21,3,3)
            local=np.stack([RotationSpline(t,Rotation.from_matrix(rotations[:,j]))(q).as_matrix() for j in range(21)],axis=1)
        _,native,_,frames=ref.body.forward_rotations(local,with_frames=True)
        samples=ref.binding.surface.sample_vertices(native,frames,ref.binding.sample_ids,double_precision=True)
        collar=ref.binding.collar_metrics(native,frames,samples)
        unknown=[]
        for i,foot in zip(*np.nonzero(collar['unknown'])):
            unknown.append(dict(source_pts=int(pts[i]),source_seconds=float(pts[i]/12800),foot=int(foot),
                reason=str(collar['unknown_reason'][i,foot]),sample_count=int(collar['sample_count'][i,foot])))
        valid=~collar['unknown'];breach=collar['breach_solver'][valid]*ref.binding.ratio
        results[label]=dict(unknown_sections=len(unknown),unknown_exposures=len({x['source_pts'] for x in unknown}),
            unknown_rows=unknown,valid_collar_breach_max_m=float(breach.max()) if breach.size else None,
            unknown_is_not_measured_penetration=True,world_root_cannot_change_this_body_local_test=True)
        print(json.dumps(dict(label=label,unknown_sections=len(unknown))),flush=True)
    write_json(out/'collar_diagnostic.json',dict(mode='read_only_no_optimization',cases=results,
        caution='Template/skin section availability, not independent real boot or ice truth. Domain surrogate is not physical penetration.'))

if __name__=='__main__':main()