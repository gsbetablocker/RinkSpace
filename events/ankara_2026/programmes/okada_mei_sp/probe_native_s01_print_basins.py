"""Isolated native LM comparisons on the same original print observations.

Only the selected existing shape block and its curve stations vary. This is
an explanatory experiment, never a replacement for the production state.
"""
from pathlib import Path
import importlib.util, io, json, time
import numpy as np
spec=importlib.util.spec_from_file_location('prep',Path(__file__).with_name('prepare_native_s01.py'))
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.domain.print_assets import sample_template_paths
from rinkspace.session.gpu_geometry import NativeSceneGPU


def run():
    out=prep.RUN/'print_shape_audit';target=out/'basin_probe';target.mkdir(exist_ok=True)
    with np.load(io.BytesIO((out/'audited_state.npz').read_bytes())) as f:saved={k:f[k] for k in f.files}
    with np.load(prep.RUN/'point_observations.npz') as f:z={k:f[k] for k in f.files}
    input_hash=prep.digest(prep.RUN/'point_observations.npz');snapshot_hash=prep.digest(prep.RUN/'accepted_pixels_latest.npz')
    assert str(saved['point_input_sha256'])==input_hash
    paths=np.asarray(prep.read(prep.RUN/'mapping.json')['path_ids'])[z['path_indices']]
    scene=prep.read(prep.SCENE);model=SceneParameterModel(scene,prep.SCENE,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,saved['scene_values']);core=NativeCore(prep.LIBRARY)
    camera={**prep.camera_for_snapshot(prep.read(prep.CAMERA),saved),'rig':saved['rig'].tolist()}
    cases=[(8,'Ad8.print.glyph_W.boundary.0000'),(8,'Ad8.print.glyph_N.boundary.0000'),
           (10,'Ad10.print.glyph_1_S.boundary.0000'),(12,'Ad12.shield_print_component019')]
    previous=prep.read(target/'report.json') if (target/'report.json').exists() else None
    if previous is not None and previous['input_sha256']!=input_hash:
        raise ValueError('Existing diagnostic belongs to different observations')
    records=[] if previous is None else previous['records'];started=time.monotonic()
    with (target/'fit_progress.log').open('a',encoding='utf-8',buffering=1) as log:
        def emit(record):
            record['elapsed_s']=round(time.monotonic()-started,3)
            log.write(json.dumps(record,ensure_ascii=True)+'\n')
        for n,pid in cases:
            ref,asset=next((r,a) for r,a in read_scene_assets(scene,prep.SCENE) if r['advertisement_number']==n)
            template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
            group=next(g for g in template['internal_geometry']['groups'] if pid in g['path_ids'])
            block=state.blocks[group['parameter_block_id']]
            indices=np.asarray(block['independent_local_indices'])+block['offset']
            rows=np.flatnonzero(np.isin(paths,['print_instance/Ad'+str(n)+'::'+p for p in group['path_ids']]))
            q=len(indices);dirs=np.zeros((len(state.values),q));dirs[indices,np.arange(q)]=.002
            local=np.full(len(rows),-1,np.int32);free=np.flatnonzero(z['kind'][rows]==1);local[free]=np.arange(len(free))
            candidates=[]
            for seed_name in ('accepted_shape','reset_warp','reset_group'):
                existing=next((r for r in records if r['path']==pid and r['seed']==seed_name),None)
                if existing is not None:
                    candidates.append(existing)
                    continue
                values=state.values.copy();local_indices=block['independent_local_indices']
                reset=[] if seed_name=='accepted_shape' else [i for i in local_indices if seed_name=='reset_group' or i>=4]
                for i in reset:values[block['offset']+i]=model.initial_state.values[block['offset']+i]
                candidate=ParameterState(state.layout,values)
                native=NativeSceneGPU(prep.SCENE,core,parameter_state=candidate)
                with native.resident(camera=camera,controls=saved['controls'],path_ids=paths[rows].tolist(),
                        stations=saved['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows],scene_directions=dirs) as resident:
                    resident.configure_pixel_linearization(directions=np.zeros((len(saved['controls']),3,q)),
                        local_ids=local,local_scales=np.full(len(free),.001))
                    resident.configure_solver(lower=np.zeros(len(free)),upper=np.ones(len(free)),
                        device_budget_bytes=resident.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,
                        damping_metric='jacobi',initial_lambda=.001)
                    refined=resident.refine_curve_stations();initial=resident.evaluate('cuda')
                    initial_cost=float(.5*np.sum(initial['values'][:,9:11]**2))
                    emit(dict(event='begin',ad=n,path=pid,seed=seed_name,rows=len(rows),cost=initial_cost))
                    def progress(step):
                        s=step['summary']
                        if s['accepted']:resident.refine_curve_stations()
                        ev=resident.evaluate('cuda');errors=np.linalg.norm(ev['values'][:,9:11],axis=1)
                        emit(dict(event='step',ad=n,seed=seed_name,step=s['trial_count'],
                            cost=float(.5*np.dot(errors,errors)),max_px=float(errors.max()),
                            above20=int(np.count_nonzero(errors>20)),accepted=s['accepted']))
                        return not (target/'stop_requested').exists()
                    result=resident.solve(progress=progress)
                    after=resident.export_state();evaluation=resident.evaluate('cuda')
                mask=np.ones(len(values),bool);mask[indices]=False
                assert np.array_equal(after['scene_values'][mask],state.values[mask])
                assert np.array_equal(after['controls'],saved['controls']) and np.array_equal(after['rig'],saved['rig'])
                errors=np.linalg.norm(evaluation['values'][:,9:11],axis=1)
                name=pid.replace('.','_').replace('/','_')+'_'+seed_name
                np.savez(target/(name+'.npz'),**after,source_rows=z['source_rows'][rows],point_input_sha256=input_hash)
                final_state=ParameterState(state.layout,after['scene_values'])
                shapes=sample_template_paths(asset,template['id'],core,np.exp(final_state[template['size_class_id']]),parameter_state=final_state)
                record=dict(ad=n,path=pid,seed=seed_name,rows=len(rows),initial_cost=initial_cost,
                    final_cost=float(.5*np.dot(errors,errors)),max_px=float(errors.max()),
                    parameter_values=final_state[group['parameter_block_id']].tolist(),result=result,
                    shapes=shapes,state_file=str(target/(name+'.npz')))
                records.append(record);candidates.append(record);emit({k:v for k,v in record.items() if k not in ('shapes','result')})
                prep.write(target/'report.json',dict(input_sha256=input_hash,records=records,
                    production_state_changed=False,observations_changed=False,interpretation='Existing native local shape/station subproblem from different seeds; not a global solve or full-shot acceptance'))
                print(json.dumps({k:v for k,v in record.items() if k not in ('shapes','result')},ensure_ascii=True),flush=True)
                if (target/'stop_requested').exists():return
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(3,1,figsize=(13,7),layout='constrained')
            for ax,record in zip(axes,candidates):
                for path in record['shapes']:
                    xy=np.asarray(path['native_template_xy'])
                    ax.plot(xy[:,0],xy[:,1],color='red' if path['id'] in group['path_ids'] else '#999999',lw=.65)
                ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis();ax.set_title(f'{record["seed"]} | same {len(rows)} source points | cost {record["final_cost"]:.3f}')
            fig.suptitle(pid);fig.savefig(target/(pid.replace('.','_').replace('/','_')+'_seeds.png'),dpi=120);plt.close(fig)
    assert prep.digest(prep.RUN/'point_observations.npz')==input_hash
    assert prep.digest(prep.RUN/'accepted_pixels_latest.npz')==snapshot_hash


if __name__=='__main__':run()
