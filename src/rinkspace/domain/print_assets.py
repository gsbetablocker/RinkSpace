"""Print identity and shared parameter references, without pseudo-observations."""
from __future__ import annotations
from copy import deepcopy
import math


class PrintBindingError(ValueError):
    def __init__(self,code,**context):
        super().__init__(code);self.code=code;self.context=context


def _registry(records):
    result={}
    for record in records:
        if not record.get('id') or record['id'] in result:
            raise PrintBindingError('duplicate_or_missing_print_identity')
        result[record['id']]=record
    return result


def resolve_instance(assets,instance_id):
    """Resolve the actual shared objects, not duplicated parameter values."""
    instances=_registry(assets['instances']);templates=_registry(assets['templates'])
    blocks=_registry(assets.get('parameter_blocks',[]))
    if instance_id not in instances:raise PrintBindingError('unknown_print_instance',instance_id=instance_id)
    instance=instances[instance_id]
    try:
        template=templates[instance['template_id']]
        size=blocks[instance['size_block_id']]
        placement=blocks[instance['placement_block_id']]
    except KeyError as exc:
        raise PrintBindingError('missing_print_parameter_reference',instance_id=instance_id,reference=str(exc)) from exc
    if template['size_class_id']!=size['id'] or size['kind']!='print_size' or size['size']!=2:
        raise PrintBindingError('invalid_shared_print_size',instance_id=instance_id)
    if placement['kind']!='print_placement' or placement['size']!=3:
        raise PrintBindingError('invalid_print_placement',instance_id=instance_id)
    return instance,template,size,placement


def native_instance_parameters(assets,instance_id):
    """Native kind5 parameters; unknown identity-only placements fail clearly."""
    instance,template,size,placement=resolve_instance(assets,instance_id)
    if placement['values'] is None:
        raise PrintBindingError('unlocalized_print_instance',instance_id=instance_id,block_id=placement['id'])
    if size['values'] is None:
        raise PrintBindingError('uninitialized_print_size',instance_id=instance_id,block_id=size['id'])
    if len(size['values'])!=2 or len(placement['values'])!=3:
        raise PrintBindingError('invalid_print_parameter_values',instance_id=instance_id)
    try:
        width,height=map(math.exp,size['values']);values=[*map(float,placement['values']),width,height]
    except (OverflowError,TypeError,ValueError) as exc:
        raise PrintBindingError('invalid_print_parameter_values',instance_id=instance_id) from exc
    if not all(math.isfinite(v) for v in values) or min(width,height)<=0:
        raise PrintBindingError('invalid_print_parameter_values',instance_id=instance_id)
    return values


def initialize_instance(assets,instance_id,parameters,*,provenance):
    """Consume an explicit installation seed; never infer size from a picture.

    This changes the shared size block and this instance's independent placement.
    Source rows, template coordinates and any other placement are untouched.
    """
    result=deepcopy(assets)
    instance,template,size,placement=resolve_instance(result,instance_id)
    values=list(map(float,parameters))
    if len(values)!=5 or not all(math.isfinite(v) for v in values) or min(values[3:])<=0:
        raise PrintBindingError('invalid_print_installation_seed',instance_id=instance_id)
    proposed_size=list(map(math.log,values[3:]));proposed_placement=values[:3]
    for block,proposed in ((size,proposed_size),(placement,proposed_placement)):
        if block['values'] is not None and block['values']!=proposed:
            raise PrintBindingError('print_initialization_conflicts_with_current_values',block_id=block['id'])
        block['values']=proposed
        block['status']='historical_installation_initial_not_refitted'
        block['initialization_provenance']=deepcopy(provenance)
    instance['placement_status']='continuous_board_installed_initial_not_refitted'
    instance['installation_provenance']=deepcopy(provenance)
    native_instance_parameters(result,instance_id)
    return result


def sample_instance(assets,instance_id,board,core,*,max_step=.0004):
    """One native print chart and BoardCourse path for every template contour.

    Dense sampling is along each material edge before bending, so edges crossing
    course joins are not replaced by spatial chords. Closed/hole metadata and
    open supports remain the template's own. Samples are not observations.
    """
    import numpy as np
    if not math.isfinite(max_step) or max_step<=0:
        raise ValueError('positive print sampling step required')
    instance,template,size,placement=resolve_instance(assets,instance_id)
    parameters=native_instance_parameters(assets,instance_id)
    shapes=[];maximum=0.
    for path in template['paths']:
        geometry=path.get('geometry')
        if geometry is None:
            vertices=np.asarray(path['material_uv'],dtype=float)
        else:
            if geometry.get('kind')!='unfolded_square' or 'material_uv' in path:
                raise PrintBindingError('unsupported_or_ambiguous_print_geometry',path_id=path['id'])
            vertices=np.asarray(core.print_square_vertices(parameters,geometry['center_uv'],
                geometry['side_over_panel_height']),dtype=float)
        if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<2 or not np.all(np.isfinite(vertices)):
            raise PrintBindingError('invalid_print_path',path_id=path['id'])
        if path.get('closed') and not np.array_equal(vertices[0],vertices[-1]):
            vertices=np.vstack([vertices,vertices[0]])
        lengths=np.linalg.norm(np.diff(vertices,axis=0)*parameters[3:],axis=1)
        stations=[]
        for i,length in enumerate(lengths):
            count=max(1,math.ceil(float(length)/max_step))
            stations.extend((i+j/count)/(len(vertices)-1) for j in range(count))
        stations.append(1.)
        sz=np.asarray(core.sample_print_chart(parameters,vertices,stations))
        xyz=np.asarray(core.sample_path(board,5,parameters,vertices,stations))
        check=np.asarray(core.board_points(board,sz[:,0],sz[:,1]))
        maximum=max(maximum,float(np.max(abs(xyz-check))))
        shapes.append({key:deepcopy(path[key]) for key in
            ('id','closed','role','parent_contour_key','boundary_side','logical_labels','display_fill','geometry') if key in path}|
            {'stations':stations,'sz':sz.tolist(),'xyz':xyz.tolist(),
             'sample_role':'model_drawing_support_not_source_observation'})
    if maximum>1e-12:
        raise PrintBindingError('native_print_chart_board_disagreement',maximum=maximum)
    return {'instance_id':instance_id,'template_id':template['id'],
        'surface_id':instance['scene_surface_id'],'parameters':parameters,
        'size_block_id':size['id'],'placement_block_id':placement['id'],'paths':shapes,
        'chart_board_max_abs':maximum,'native_entity_kind':5,
        'fit_executed':False,'new_source_observations':0,
        'dimensions_status':'historical_installation_initial_not_refitted'}


def attach_paired_instance(assets,primary_instance_id,paired_instance_id,size_block_id,*,authority):
    """Register one independent, unlocalized instance of the same print.

    Templates and log-size parameters are shared by reference. No image rows,
    equality factors, initialization from diagrams, or perimeter offsets occur.
    """
    result=deepcopy(assets);instances=_registry(result['instances']);templates=_registry(result['templates'])
    if primary_instance_id not in instances:raise PrintBindingError('unknown_primary_print_instance')
    primary=instances[primary_instance_id];template=templates[primary['template_id']]
    blocks=result.setdefault('parameter_blocks',[]);by_id=_registry(blocks)
    def ensure(block):
        if block['id'] in by_id:
            existing=by_id[block['id']]
            if any(existing.get(k)!=block[k] for k in ('kind','size','parameterization')):
                raise PrintBindingError('conflicting_print_parameter_block',block_id=block['id'])
            return
        blocks.append(block);by_id[block['id']]=block
    ensure({'id':size_block_id,'kind':'print_size','size':2,'coordinates':['log_width','log_height'],
        'parameterization':'log_positive','unit':'rink_length','values':None,'status':'uninitialized',
        'authority':deepcopy(authority),'values_are_not_display_aspect_ratio':True})
    template['size_class_id']=size_block_id
    for identity in (primary_instance_id,paired_instance_id):
        label=identity.rsplit('/',1)[-1];placement_id=f'installation/{label}/shared_placement'
        ensure({'id':placement_id,'kind':'print_placement','size':3,
            'coordinates':['station_s','height_z','rotation_sz'],'parameterization':'unwrapped_s_z_rotation',
            'units':['rink_length','rink_length','radian'],'values':None,'status':'unlocalized'})
        if identity not in instances:
            instance={'id':identity,'entity_id':'entity/'+label,'template_id':template['id'],
                'scene_surface_id':primary.get('scene_surface_id','entity/board.continuous'),
                'identity_status':'user_confirmed','placement_status':'identity_only_unlocalized',
                'source_status':'no_instance_source_pixels','source_observation_ids':[],
                'identity_authority':deepcopy(authority)}
            result['instances'].append(instance);instances[identity]=instance
        instance=instances[identity]
        if instance['template_id']!=template['id']:
            raise PrintBindingError('conflicting_paired_print_template',instance_id=identity)
        if instance.get('size_block_id',size_block_id)!=size_block_id:
            raise PrintBindingError('conflicting_paired_print_size',instance_id=identity)
        instance['size_block_id']=size_block_id;instance['size_class_ref']=size_block_id
        instance['placement_block_id']=placement_id
    template['shared_size_instance_ids']=list(dict.fromkeys(template.get('shared_size_instance_ids',[])+[primary_instance_id,paired_instance_id]))
    for identity in (primary_instance_id,paired_instance_id):resolve_instance(result,identity)
    result['counts']['print_instances']=len(result['instances'])
    result['counts']['shared_size_parameter_blocks']=sum(b['kind']=='print_size' for b in blocks)
    return result
