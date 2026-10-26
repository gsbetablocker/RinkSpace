"""Inspect every native print template before/after fitting, without camera projection."""
from pathlib import Path
import argparse, hashlib, importlib.util, io, itertools, json
import numpy as np

spec=importlib.util.spec_from_file_location('prep',Path(__file__).with_name('prepare_native_s01.py'))
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.print_assets import sample_template_paths
from rinkspace.domain.print_parameters import parameter_values
from rinkspace.domain.scene_assets import read_scene_assets


def proper_crossings(a,b,same=False):
    """Strict segment crossings; shared endpoints/collinear contacts are separate."""
    a=np.asarray(a);b=np.asarray(b);u=a[1:]-a[:-1];v=b[1:]-b[:-1]
    delta=b[None,:-1,:]-a[:-1,None,:]
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    den=cross(u[:,None,:],v[None,:,:]);ok=abs(den)>1e-18
    s=np.divide(cross(delta,v[None,:,:]),den,out=np.zeros_like(den),where=ok)
    t=np.divide(cross(delta,u[:,None,:]),den,out=np.zeros_like(den),where=ok)
    ok&=(s>1e-9)&(s<1-1e-9)&(t>1e-9)&(t<1-1e-9)
    if same:
        i,j=np.indices(ok.shape);ok&=j>i+1
    i,j=np.nonzero(ok)
    return [dict(segment_a=int(x),segment_b=int(y),uv=(a[x]+s[x,y]*u[x]).tolist()) for x,y in zip(i,j)]


def run(snapshot,out):
    out.mkdir(parents=True,exist_ok=True)
    blob=snapshot.read_bytes();(out/'audited_state.npz').write_bytes(blob)
    with np.load(io.BytesIO(blob),allow_pickle=False) as s:saved={k:s[k] for k in s.files}
    core=NativeCore(prep.LIBRARY);scene=prep.read(prep.SCENE);model=SceneParameterModel(scene,prep.SCENE,optimize_scene=True)
    states={'native_initial':model.initial_state,'fitted':ParameterState(model.initial_state.layout,saved['scene_values'])}
    with np.load(prep.RUN/'point_observations.npz',allow_pickle=False) as z:
        indices=z['path_indices'];pts=z['pts']
    mapping=prep.read(prep.RUN/'mapping.json');observed=np.asarray(mapping['path_ids'])[indices]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=[];all_shapes={}
    for reference,assets in read_scene_assets(scene,prep.SCENE):
        template=next(t for t in assets['templates'] if t['id']==reference['template_id'])
        paths={p['id']:p for p in template['paths']};evaluated={};checks={}
        for label,state in states.items():
            values=parameter_values(assets,state);dimensions=np.exp(values[template['size_class_id']])
            shapes=sample_template_paths(assets,template['id'],core,dimensions,parameter_state=state)
            evaluated[label]={p['id']:p for p in shapes};crossed=[];self_crossed=[]
            for a in shapes:
                crosses=proper_crossings(a['material_uv'],a['material_uv'],True)
                if crosses:self_crossed.append(dict(path=a['id'],crossings=crosses))
            for a,b in itertools.combinations(shapes,2):
                crosses=proper_crossings(a['material_uv'],b['material_uv'])
                if crosses:crossed.append(dict(paths=[a['id'],b['id']],crossings=crosses,
                    parameter_blocks=[a['parameter_block_ids'],b['parameter_block_ids']]))
            checks[label]=dict(self_crossings=self_crossed,path_pair_crossings=crossed,dimensions=dimensions.tolist())
        old_pairs={tuple(p['paths']) for p in checks['native_initial']['path_pair_crossings']}
        new_pairs=[p for p in checks['fitted']['path_pair_crossings'] if tuple(p['paths']) not in old_pairs]
        old_self={p['path'] for p in checks['native_initial']['self_crossings']}
        new_self=[p for p in checks['fitted']['self_crossings'] if p['path'] not in old_self]
        number=reference['advertisement_number'];instances=reference.get('instance_ids',[reference.get('instance_id')])
        activity={instance:int(np.count_nonzero(np.char.startswith(observed,instance+'::'))) for instance in instances}
        fig,axes=plt.subplots(3,1,figsize=(14,8),layout='constrained')
        palette=plt.get_cmap('tab20')
        for k,(label,shapes) in enumerate(evaluated.items()):
            ax=axes[k]
            for j,(identity,p) in enumerate(shapes.items()):
                q=np.asarray(p['native_template_xy']);ax.plot(q[:,0],q[:,1],lw=.75,color=palette(j%20))
            ax.set_title(label+' | native template physical aspect, no camera or drawing joins')
            ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis();ax.grid(alpha=.15)
        for label,color in [('native_initial','#888888'),('fitted','#d14725')]:
            for p in evaluated[label].values():
                q=np.asarray(p['material_uv']);axes[2].plot(q[:,0],q[:,1],lw=.65,color=color,alpha=.7)
        axes[2].set_title('Normalized material chart | gray: initial | red: fitted')
        axes[2].invert_yaxis();axes[2].grid(alpha=.15)
        fig.suptitle(f'Ad{number} | new crossing pairs {len(new_pairs)} | new self-crossing paths {len(new_self)}')
        image=out/f'Ad{number:02d}_native_shapes.png';fig.savefig(image,dpi=125);plt.close(fig)
        record=dict(template_id=template['id'],advertisement=number,instances=instances,observation_counts=activity,
            path_count=len(paths),native_shape_group_count=len(template.get('internal_geometry',{}).get('groups',[])),
            new_crossing_pairs=new_pairs,new_self_crossing_paths=new_self,stages=checks,image=str(image))
        if template.get('material_domain',{}).get('kind')=='physical_unit_panel':
            q=np.vstack([p['material_uv'] for p in evaluated['fitted'].values()])
            record['physical_panel_domain']=dict(kind='physical_unit_panel',bounds=[q.min(0).tolist(),q.max(0).tolist()],
                max_outside_material=float(max(0.,-q.min(),q.max()-1.)),
                check='Native material coordinates, before installation and camera; no source-pixel allowance')
        report.append(record);all_shapes[str(number)]=evaluated
        print(json.dumps(dict(ad=number,paths=len(paths),observations=activity,new_crossing_pairs=len(new_pairs),new_self_crossing_paths=len(new_self)),ensure_ascii=True),flush=True)
    prep.write(out/'native_shapes.json',all_shapes)
    result=dict(schema='rinkspace.native_print_topology_audit.v1',snapshot_sha256=hashlib.sha256(blob).hexdigest(),
        scene_sha256=prep.digest(prep.SCENE),point_input_sha256=prep.digest(prep.RUN/'point_observations.npz'),
        templates=len(report),instances=sum(len(r['instances']) for r in report),records=report,
        cross_check='Strict non-endpoint segment crossings in native material coordinates; geometric intersection is evidence to inspect, not an automatic source exclusion.',
        parameters_changed=False,source_points_changed=False,display_vertices_reordered=False)
    prep.write(out/'report.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scene-path',type=Path);parser.add_argument('--run-dir',type=Path);a=parser.parse_args()
    if a.scene_path is not None:prep.SCENE=a.scene_path.resolve(strict=True)
    if a.run_dir is not None:prep.RUN=a.run_dir.resolve(strict=True)
    run(a.snapshot,a.out)
