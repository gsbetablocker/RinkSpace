"""Same-ledger review of completed S05 candidates; no optimization or alignment."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .spin_continuous_review import describe,region_measure,PROGRAMME
from .spin_continuous_refit import restore_absolute
from .stage_quality import error_summary
from .absolute_motion_curve import _log


def trajectory(packet):
    pts=packet['source_pts'];t=pts/12800.;out={}
    channels={'pelvis':packet['P_world_mid'],'com_mass_proxy':packet['com_world_mid'],
        'support_blade_material_midpoint':packet['blade_material_world_mid'][:,1,1],
        'support_current_lowest_geometry':packet['skate_witness_world_mid'][:,1]}
    for name,(a,b) in {'whole':(76.,95.),'entry':(80.8,82.4),'rotation':(84.,86.),
                       'held':(88.,92.),'supported_episode':(86.8,92.46),'release':(92.4,93.2)}.items():
        mask=(t>=a)&(t<=b);ix=np.flatnonzero(mask);record={}
        for kind,X in channels.items():
            q=X[ix];speed=np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1)/np.diff(t[ix])
            record[kind]=dict(xy_extent_m=np.ptp(q[:,:2],axis=0).tolist(),
                horizontal_interval_speed_p95_m_s=float(np.quantile(speed,.95)),
                maximum_horizontal_20ms_displacement_m=float(np.max(speed*.02)),
                minimum_reference_height_m=float(q[:,2].min()),maximum_reference_height_m=float(q[:,2].max()))
        out[name]=record
    out['interpretation']={'pelvis':'joint centre; can orbit and is not COM',
        'com_mass_proxy':'full segment COM with assumed masses, not a measured COM',
        'support_blade_material_midpoint':'fixed template midpoint, not actual contact',
        'support_current_lowest_geometry':'may migrate/switch on rocker and edge; its interval speed is not material velocity',
        'raw_height':'nominal reference plane, not direct actual contact truth'}
    return out


def run(project,base,run_root,parameter,output,previous=None):
    project,base,root,parameter,output=map(lambda p:Path(p).resolve(),(project,base,run_root,parameter,output))
    if output.exists():raise FileExistsError('Preserve existing review: '+str(output))
    summary=json.loads((root/'run_summary.json').read_text(encoding='utf-8'))
    if Path(summary['parameter']).resolve()!=parameter:raise ValueError('Completed parameter identity mismatch')
    obs_path=root/'common/exmr_hand_nofoot/observations.npz';obs=load(obs_path)
    doc_path=root/'source_evidence.json';doc=json.loads(doc_path.read_text(encoding='utf-8'))
    pts=np.arange(972800,1216001,256,dtype=np.int64);np.testing.assert_array_equal(obs['pts'],pts)
    definitions={}
    original=base/'spin_continuous_1105'
    frozen_observations=load(original/'common/exmr_hand_nofoot/observations.npz')
    for label,stage in [('A','source_stage'),('B','flight_stage')]:
        definitions[label]=json.loads((original/stage/'run_summary.json').read_text(encoding='utf-8'))
    if previous:definitions['previous_continuation']=json.loads((Path(previous)/'run_summary.json').read_text(encoding='utf-8'))
    definitions['new']=summary
    output.mkdir(parents=True)
    results={};plot_packets={}
    for label,definition in definitions.items():
        parameter_path=Path(definition['parameter']);path=Path(definition['result']);packet=load(path)
        np.testing.assert_array_equal(packet['source_pts'],pts)
        if label=='new':
            for key,expected in [('source_uv_native',obs['uv']),('source_sigma_px',obs['sigma_px']),
                                 ('source_observation_use',obs['use']),('source_observation_objective_use',obs['use'])]:
                np.testing.assert_array_equal(packet[key],expected)
            if str(packet['source_observation_ledger_sha256'])!=digest(obs_path):raise ValueError('Exported ledger SHA mismatch')
        ref,camera,info,_=restore_absolute(project,base,output/(label+'_geometry'),parameter_path)
        frames=np.arange(len(pts));state=ref.at(ref.times)
        np.testing.assert_allclose(state['P']*ref.binding.ratio,packet['P_world_mid'],rtol=0,atol=2e-10)
        np.testing.assert_allclose(state['Q'],packet['Q_world_mid'],rtol=0,atol=2e-10)
        np.testing.assert_allclose(state['local_R'],packet['local_R_mid'],rtol=0,atol=2e-10)
        raw_error=np.linalg.norm(packet['COCO_uv']-frozen_observations['uv'],axis=-1)
        raw_mask=frozen_observations['use'];raw_clock=np.broadcast_to(pts[:,None],raw_mask.shape)
        results[label]=dict(frozen_detector_diagnostic=error_summary(raw_error[raw_mask],raw_clock[raw_mask]),
            frozen_detector_diagnostic_is_not_truth=True,parameter=str(parameter_path),parameter_sha256=digest(parameter_path),
            result=str(path),result_sha256=digest(path),source_frames=len(pts),
            body_same_revised_ledger=describe(packet,obs),source_regions=region_measure(ref,camera,doc),
            trajectory=trajectory(packet),saved_state_exact=True)
        plot_packets[label]=packet
    report=dict(schema='rinkspace.spin-continuation-review.v1',observations=str(obs_path),
        observations_sha256=digest(obs_path),source_evidence=str(doc_path),source_evidence_sha256=digest(doc_path),
        source_time_base=[1,12800],source_frames=len(pts),actual_admitted_body_points=int(obs['use'].sum()),
        same_revised_ledger_for_all_variants=True,training_evidence_not_independent_validation=True,
        unchanged_A_B_not_refitted=True,variants=results,accuracy_accepted=False,
        note='Region residual can be zero anywhere inside a finite development region. Neither this nor smaller pelvis motion certifies correct shoe/contact anatomy.')
    write_json(output/'same_evidence_comparison.json',report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for label,p in plot_packets.items():
        sec=p['source_pts']/12800.;mask=(sec>=86.8)&(sec<=92.46)
        axs[0,0].plot(p['P_world_mid'][:,0],p['P_world_mid'][:,1],label=label)
        X=p['skate_witness_world_mid'][mask,1];axs[0,1].plot(X[:,0],X[:,1],label=label)
        X=p['blade_material_world_mid'][mask,1,1];axs[1,0].plot(sec[mask],X[:,0],label=label)
        Q=p['Q_world_mid'];omega=_log(Q[1:]@Q[:-1].swapaxes(-1,-2))/.02
        axs[1,1].plot(sec[1:],omega[:,2],label=label)
    for ax,title in zip(axs.flat,['Pelvis XY, all source frames','Current support geometric witness XY, 86.8-92.46s',
                                  'Fixed blade material midpoint X, 86.8-92.46s','Actual root world omega_z, rad/s']):
        ax.set_title(title);ax.grid(alpha=.2)
    axs[0,0].axis('equal');axs[0,1].axis('equal');axs[0,0].legend();fig.tight_layout()
    fig.savefig(output/'motion_comparison.png',dpi=150);plt.close(fig)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['project','base-root','run-root','parameter','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--previous',type=Path);a=p.parse_args(argv)
    r=run(a.project,a.base_root,a.run_root,a.parameter,a.output,a.previous)
    print(json.dumps(dict(report=str(a.output/'same_evidence_comparison.json'),variants=list(r['variants'])),ensure_ascii=False))
if __name__=='__main__':main()
