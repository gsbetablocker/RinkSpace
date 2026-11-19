"""Source-testable release of Ad16 dimensions, retaining the same artwork."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import numpy as np
from shared import EVENT,FIELD,LIBRARY,OUT,NativeCore,NativeSceneGPU,ParameterState,SceneParameterModel,arrays,read,save,sha,write
from review import sample


def create(parent,target_name='ad16_size_seed'):
    if (OUT/target_name/'state.npz').exists():raise ValueError('Preserve the existing candidate: '+target_name)
    old=arrays(OUT/parent/'state.npz');original=Path(str(old.get('scene_path',FIELD)))
    doc=read(original);before=SceneParameterModel(doc,original,optimize_scene=True)
    previous=ParameterState(before.initial_state.layout,old['scene_values'])
    ref=next(x for x in doc['advertisement_assets'] if x['advertisement_number']==3)
    asset=read(ref['path']);source=str(ref['path'])
    instance=next(x for x in asset['instances'] if x['id']=='print_instance/Ad16')
    block=deepcopy(next(x for x in asset['parameter_blocks'] if x['id']==instance['size_block_id']))
    previous_size=block['id'];block['id']='size/Ad16_free_1117'
    block['values']=previous[previous_size].tolist()
    block['previous_equal_size_authority']=block.pop('authority',{})
    block['authority']=dict(kind='current_user_requested_geometry_constraint_ablation',
        status='source_comparison_candidate_not_metric_fact',paired_template_retained=True)
    asset['parameter_blocks'].append(block)
    instance.update(size_block_id=block['id'],size_class_ref=block['id'],size_binding_policy='independent_instance')
    template=next(t for t in asset['templates'] if t['id']==instance['template_id'])
    template['shared_size_instance_ids']=['print_instance/Ad3']
    asset['independent_dimension_trial']=dict(parent=source,reason='Repeated Nayeon Ad16 right-edge residual with other three edges close',
        actual_size_difference_not_presumed=True,original_source_uv_unchanged=True)
    target=EVENT/'scene/ad3/free_ad16_1117.json'
    if not target.exists():write(target,asset)
    ref.update(path=str(target),sha256=sha(target),shared_size_block_id=previous_size,
        independently_sized_instance_ids=['print_instance/Ad16'])
    doc['revision']='free_metric_and_Ad16_dimension_ablation_1117'
    doc['independent_print_size_trial']=asset['independent_dimension_trial']
    field=EVENT/'scene/field/free_ad16_1117.json'
    if not field.exists():write(field,doc)
    else:doc=read(field)
    model=SceneParameterModel(doc,field,optimize_scene=True)
    values=model.initial_state.values.copy()
    for key,b in model.initial_state.blocks.items():
        source_key=previous_size if key==block['id'] else key
        values[b['offset']:b['offset']+b['size']]=previous[source_key]
    state=ParameterState(model.initial_state.layout,values)
    core=NativeCore(LIBRARY)
    a=NativeSceneGPU(original,core,parameter_state=previous);b=NativeSceneGPU(field,core,parameter_state=state)
    first=sample(a);second=sample(b)
    assert [p['id'] for p in first['cuts']]==[p['id'] for p in second['cuts']]
    delta=float(np.max(abs(first['xyz']-second['xyz'])))
    assert delta<1.e-12
    old.update(scene_values=values,scene_path=str(field),layout_revision=state.layout['layout_revision'])
    save(OUT/target_name/'state.npz',**old)
    write(OUT/target_name/'migration.json',dict(parent=parent,scene=str(field),stored_values=len(values),
        unchanged_initial_world_vertices_max_error=delta,scope='only Ad16 dimension equality released; shared normalized artwork retained'))
    print('Ad16 explicit size trial:',len(values),'stored values, migration delta',delta)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--parent',required=True)
    ap.add_argument('--out',default='ad16_size_seed');args=ap.parse_args();create(args.parent,args.out)
