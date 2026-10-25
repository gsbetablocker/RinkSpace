"""Print identity and shared parameter references, without pseudo-observations."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
from collections.abc import Mapping
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


def validate_template_topology(template):
    """Current template relationships use the same full IDs as their paths.

    Historical short keys are converted by the import adapter, never guessed
    while rendering or compiling a current template. Pixel coordinates and
    contour geometry are not changed by this check.
    """
    paths = _registry(template['paths'])
    for identity, path in paths.items():
        parent = path.get('parent_contour_key')
        if path.get('role', '').endswith('hole_boundary') and parent is None:
            raise PrintBindingError('missing_print_hole_parent', path_id=identity)
        if parent is not None:
            if parent not in paths or parent == identity:
                raise PrintBindingError('invalid_print_parent_reference', path_id=identity, parent_id=parent)
            if paths[parent].get('role', '').endswith('hole_boundary'):
                raise PrintBindingError('print_hole_parent_is_hole', path_id=identity, parent_id=parent)
        for child in path.get('separate_inner_contour_keys', []):
            if child not in paths or paths[child].get('parent_contour_key') != identity:
                raise PrintBindingError('invalid_print_child_reference', path_id=identity, child_id=child)
    return paths


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
    validate_template_topology(template)
    if template['size_class_id']!=size['id'] or size['kind']!='print_size' or size['size']!=2:
        raise PrintBindingError('invalid_shared_print_size',instance_id=instance_id)
    if placement['kind']!='print_placement' or placement['size']!=3:
        raise PrintBindingError('invalid_print_placement',instance_id=instance_id)
    return instance,template,size,placement



# These roles describe actual materials, including an open visible part of an
# ink boundary. They do not describe paint-width centrelines or image texture.
PRINT_CONTOUR_ROLES = frozenset({
    'letter_material_boundary', 'letter_outer_boundary', 'letter_hole_boundary',
    'print_material_boundary', 'panel_outer_boundary', 'logo_outer_boundary',
    'logo_hole_boundary', 'printed_ink_external_boundary', 'printed_ink_open_boundary_arc',
})


@dataclass(frozen=True)
class PrintCurveBinding:
    """An identified instance of one template material boundary, not pixels.

    Obtain bindings from resolve_print_curve. Serialized declarations are checked
    against the current scene assets again at prepare/load, never trusted as a
    role override. Shared size/placement remain references in those assets.
    """
    instance_id: str
    template_id: str
    template_path_id: str
    role: str
    parent_contour_key: str | None = None

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in
               (self.instance_id, self.template_id, self.template_path_id)):
            raise PrintBindingError('invalid_print_curve_identity')
        if self.role not in PRINT_CONTOUR_ROLES:
            raise PrintBindingError('unsupported_print_measurement_role', role=self.role)
        if self.parent_contour_key is not None and (
                not isinstance(self.parent_contour_key, str) or not self.parent_contour_key
                or self.parent_contour_key == self.template_path_id):
            raise PrintBindingError('invalid_print_curve_parent', path_id=self.template_path_id)
        if self.role.endswith('hole_boundary') and self.parent_contour_key is None:
            raise PrintBindingError('missing_print_hole_parent', path_id=self.template_path_id)

    @property
    def path_id(self):
        return self.instance_id+'::'+self.template_path_id

    def as_record(self):
        return asdict(self)


def resolve_print_curve(assets, instance_id, template_path_id):
    """Resolve a real path and its role/topology through the actual instance.

    An unlocalized paired identity can have a binding without acquiring geometry
    or observations. Localization, source visibility, quality and statistics are
    separate requirements; resolving a material identity does not activate it.
    """
    instance, template, _, _ = resolve_instance(assets, instance_id)
    if instance.get('scene_surface_id') != 'entity/board.continuous':
        raise PrintBindingError('print_curve_requires_board_surface', instance_id=instance_id)
    paths = validate_template_topology(template)
    if template_path_id not in paths:
        raise PrintBindingError('unknown_print_template_path', instance_id=instance_id,
                                path_id=template_path_id)
    path = paths[template_path_id]
    if path.get('template_id', template['id']) != template['id']:
        raise PrintBindingError('print_path_template_mismatch', path_id=template_path_id)
    return PrintCurveBinding(instance_id, template['id'], template_path_id,
                             path.get('role'), path.get('parent_contour_key'))


def resolve_feature_print_bindings(features, scene_assets, *, active_feature_ids=()):
    """Verify feature declarations against only the scene's selected assets.

    No Ad-number/label inference and no blanket board-surface exception. Actual
    canonical path IDs also catch a print feature that omits its binding.
    """
    available = {}
    instance_entities = {}
    unlocalized = set()
    active = set(active_feature_ids)
    for ref, assets in scene_assets:
        selected = ref.get('instance_ids', [ref['instance_id']] if 'instance_id' in ref else [])
        for instance_id in selected:
            instance, template, size, placement = resolve_instance(assets, instance_id)
            instance_entities[instance_id] = instance.get('entity_id', 'entity/'+instance_id.rsplit('/', 1)[-1])
            if size['values'] is None or placement['values'] is None:
                unlocalized.add(instance_id)
            for path in template['paths']:
                binding = resolve_print_curve(assets, instance_id, path['id'])
                if binding.path_id in available:
                    raise PrintBindingError('duplicate_scene_print_path', path_id=binding.path_id)
                available[binding.path_id] = binding
    result = {}
    records = features.items() if isinstance(features, Mapping) else (
        (feature['id'], feature) for feature in features)
    for identity, feature in records:
        declaration = feature.get('print_binding')
        path_id = feature.get('path_id', identity)
        is_print = (path_id in available or declaration is not None or feature.get('native_kind') == 5
                    or any(key in feature for key in ('instance_id', 'template_id', 'template_path_id'))
                    or feature.get('role') in PRINT_CONTOUR_ROLES
                    or feature.get('canonical_geometry_id') in available
                    or feature.get('entity_id') in instance_entities.values())
        if not is_print:
            continue
        if not isinstance(declaration, Mapping):
            raise PrintBindingError('missing_print_curve_binding', feature_id=identity)
        binding = available.get(path_id)
        if binding is None:
            raise PrintBindingError('unregistered_print_feature_path', feature_id=identity, path_id=path_id)
        if dict(declaration) != binding.as_record():
            raise PrintBindingError('print_feature_binding_mismatch', feature_id=identity, path_id=path_id)
        if (feature.get('surface') != 'board' or feature.get('kind', 'finite_curve') != 'finite_curve'
                or feature.get('native_kind', 5) != 5
                or feature.get('station_domain', {}).get('kind') == 'unbounded_periodic'):
            raise PrintBindingError('invalid_print_feature_geometry', feature_id=identity)
        if identity in active and binding.instance_id in unlocalized:
            raise PrintBindingError('active_print_instance_is_unlocalized', feature_id=identity,
                                    instance_id=binding.instance_id)
        for key, expected in (('entity_id', instance_entities[binding.instance_id]),
                              ('scene_surface_id', 'entity/board.continuous'),
                              ('canonical_geometry_id', binding.path_id),
                              ('instance_id', binding.instance_id), ('template_id', binding.template_id),
                              ('template_path_id', binding.template_path_id), ('role', binding.role),
                              ('parent_contour_key', binding.parent_contour_key)):
            if key in feature and feature[key] != expected:
                raise PrintBindingError('print_feature_binding_mismatch', feature_id=identity, field=key)
        result[identity] = binding
    return result


def native_instance_parameters(assets,instance_id,*,parameter_state=None,_values=None):
    """Native kind5 parameters; unknown identity-only placements fail clearly."""
    instance,template,size,placement=resolve_instance(assets,instance_id)
    from .print_parameters import parameter_values
    values_by_id = parameter_values(assets, parameter_state) if _values is None else _values
    placement_values = values_by_id[placement['id']]
    size_values = values_by_id[size['id']]
    if placement_values is None:
        raise PrintBindingError('unlocalized_print_instance',instance_id=instance_id,block_id=placement['id'])
    if size_values is None:
        raise PrintBindingError('uninitialized_print_size',instance_id=instance_id,block_id=size['id'])
    if len(size_values)!=2 or len(placement_values)!=3:
        raise PrintBindingError('invalid_print_parameter_values',instance_id=instance_id)
    try:
        width,height=map(math.exp,size_values);values=[*map(float,placement_values),width,height]
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


def _print_context(assets, instance_id, parameter_state):
    from .print_parameters import parameter_values, validate_print_parameters
    instance,template,size,placement=resolve_instance(assets,instance_id)
    validate_print_parameters(assets)
    values=parameter_values(assets,parameter_state)
    parameters=native_instance_parameters(assets,instance_id,_values=values)
    internal=template.get('internal_geometry')
    return {'instance':instance,'template':template,'size':size,'placement':placement,
            'values':values,'parameters':parameters,'internal':internal,
            'groups':{} if internal is None else _registry(internal['groups'])}


def _print_seed(path, context, core):
    import numpy as np
    geometry=path.get('geometry')
    if geometry is None:
        vertices=np.asarray(path.get('material_uv'),dtype=float)
    else:
        if geometry.get('kind')!='unfolded_square' or 'material_uv' in path:
            raise PrintBindingError('unsupported_or_ambiguous_print_geometry',path_id=path['id'])
        internal=context['internal']
        if internal is None:
            centre=geometry['center_uv'];side=geometry['side_over_panel_height']
        else:
            binding=internal['path_bindings'][path['id']]
            if binding['kind']!='unfolded_square':
                raise PrintBindingError('print_square_lost_analytic_binding',path_id=path['id'])
            q=context['values'][binding['parameter_block_id']]
            centre=q[:2]
            try:side=math.exp(float(q[2]))
            except OverflowError as exc:raise PrintBindingError('invalid_print_square_state',path_id=path['id']) from exc
            if not math.isfinite(side) or side<=0:
                raise PrintBindingError('invalid_print_square_state',path_id=path['id'])
        vertices=np.asarray(core.print_square_vertices(context['parameters'],centre,side),dtype=float)
    if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<2 or not np.all(np.isfinite(vertices)):
        raise PrintBindingError('invalid_print_path',path_id=path['id'])
    if path.get('closed') and not np.array_equal(vertices[0],vertices[-1]):
        vertices=np.vstack([vertices,vertices[0]])
    return vertices


def _positive_bounds(vertices):
    import numpy as np
    lo=np.min(vertices,axis=0).copy();hi=np.max(vertices,axis=0).copy()
    span=hi-lo
    for axis in (0,1):
        if span[axis]<=0:
            padding=max(float(np.max(span)),1.)/2
            lo[axis]-=padding;hi[axis]+=padding
    return [*lo,*hi]


def evaluate_print_path(assets,instance_id,path_id,board,core,stations,*,
                        parameter_state=None,jacobian=True,_context=None):
    """Evaluate one current material curve and its actual shared dependencies.

    Shape, instance installation and log-size columns come from the same native
    call. Source UV/PTS and station identity are not modified. Jacobians here
    are geometry XYZ derivatives, not a complete camera/factor/GPU Jacobian.
    """
    import numpy as np
    context=_print_context(assets,instance_id,parameter_state) if _context is None else _context
    paths={p['id']:p for p in context['template']['paths']}
    if path_id not in paths:
        raise PrintBindingError('unknown_print_template_path',path_id=path_id)
    path=paths[path_id];vertices=_print_seed(path,context,core)
    parameters=context['parameters'];internal=context['internal'];binding=None
    if internal is not None:binding=internal['path_bindings'][path_id]
    shape_block=None;shape_key=None
    if binding is not None and binding['kind']=='shape_group':
        group=context['groups'][binding['group_id']];shape_block=group['parameter_block_id']
        data=core.sample_deformed_print_path(board,parameters,vertices,stations,
            group['reference_bounds'],context['values'][shape_block],jacobian=jacobian)
        shape_key='d_xyz_d_shape'
    elif binding is not None and binding['kind']=='unfolded_square':
        shape_block=binding['parameter_block_id']
        if not hasattr(core,'sample_print_square_path'):
            raise PrintBindingError('native_print_square_state_capability_unavailable',path_id=path_id)
        data=core.sample_print_square_path(board,parameters,context['values'][shape_block],stations,jacobian=jacobian)
        shape_key='d_xyz_d_square'
    elif binding is not None and binding['kind']=='rectangle_side':
        group=context['groups'][binding['group_id']];shape_block=group['parameter_block_id']
        if not hasattr(core,'sample_print_rectangle_path'):
            raise PrintBindingError('native_print_rectangle_state_capability_unavailable',path_id=path_id)
        data=core.sample_print_rectangle_path(board,parameters,context['values'][shape_block],
            binding['side'],stations,reverse=binding['reverse'],jacobian=jacobian)
        shape_key='d_xyz_d_rectangle'
    elif binding is not None and binding['kind']!='panel_chart_gauge':
        raise PrintBindingError('unknown_print_internal_path_kind',path_id=path_id)
    elif jacobian:
        # The identity warp shares the native installed-path derivative. Its
        # shape columns are not registered as a panel/frozen-path dependency.
        data=core.sample_deformed_print_path(board,parameters,vertices,stations,
            _positive_bounds(vertices),[0.]*10,jacobian=True)
    else:
        sz=np.asarray(core.sample_print_chart(parameters,vertices,stations),dtype=float)
        xyz=np.asarray(core.sample_path(board,5,parameters,vertices,stations),dtype=float)
        seed=np.asarray(core.sample_print_chart([0.,0.,0.,1.,1.],vertices,stations),dtype=float)
        data={'material_uv':seed,'sz':sz,'xyz':xyz,'status':np.zeros(len(stations),dtype=np.int32)}
    data=dict(data)
    failures=np.flatnonzero(np.asarray(data.get('status',[])))
    if len(failures):
        row=int(failures[0])
        raise PrintBindingError('native_print_path_domain_failure',path_id=path_id,station_row=row,
                                status=int(data['status'][row]))
    ids=[context['placement']['id'],context['size']['id']]
    if shape_block is not None:ids.append(shape_block)
    data['dependency_block_ids']=ids
    data['path_id']=path_id;data['instance_id']=instance_id
    data['geometry_parameterization']=None if binding is None else binding['kind']
    if jacobian:
        ji=np.asarray(data['d_xyz_d_installation'],dtype=float)
        by_block={context['placement']['id']:ji[:,:,:3],
                  context['size']['id']:ji[:,:,3:]*np.asarray(parameters[3:])[None,None,:]}
        if shape_block is not None:by_block[shape_block]=np.asarray(data[shape_key],dtype=float)
        data['jacobians_by_block']=by_block
        data['jacobians']=by_block
        data['station_jacobian']=data.get('d_xyz_d_station')
    return data


def _drawing_stations(path,vertices,context,core,max_step):
    """Choose drawing support in the unchanged seed vertex-index coordinate."""
    import numpy as np
    base=np.linspace(0.,1.,len(vertices));internal=context['internal']
    binding=None if internal is None else internal['path_bindings'][path['id']]
    breaks=list(base)
    if binding is not None and binding['kind']=='shape_group':
        group=context['groups'][binding['group_id']]
        bounds=group['reference_bounds']
        for axis in (0,1):
            for fraction in (.25,.5,.75):
                knot=bounds[axis]+fraction*(bounds[axis+2]-bounds[axis])
                for j,(left,right) in enumerate(zip(vertices[:-1],vertices[1:])):
                    delta=right[axis]-left[axis]
                    if delta:
                        t=(knot-left[axis])/delta
                        if 0.<t<1.:breaks.append((j+t)/(len(vertices)-1))
    breaks=np.unique(breaks)
    # Exact native material points are sufficient for drawing density: rotation
    # preserves lengths in the unfolded plane and BoardCourse handles bending.
    if binding is not None and binding['kind']=='shape_group':
        group=context['groups'][binding['group_id']]
        seeds=np.asarray(core.sample_print_chart([0.,0.,0.,1.,1.],vertices,breaks),dtype=float)
        q=core.deform_print_points(seeds,group['reference_bounds'],context['values'][group['parameter_block_id']])
        changed=np.asarray(q['material_uv'],dtype=float)
    else:
        changed=np.asarray(core.sample_print_chart([0.,0.,0.,1.,1.],vertices,breaks),dtype=float)
    lengths=np.linalg.norm(np.diff(changed,axis=0)*context['parameters'][3:],axis=1)
    stations=[]
    for j,length in enumerate(lengths):
        count=max(1,math.ceil(float(length)/max_step))
        stations.extend(float(breaks[j]+(breaks[j+1]-breaks[j])*k/count) for k in range(count))
    stations.append(1.)
    return stations


def sample_instance(assets,instance_id,board,core,*,max_step=.0004,stations_by_path=None,parameter_state=None):
    """Sample the current state through one native chart and BoardCourse.

    The frozen seed chart supplies stable finite station coordinates. New shape
    blocks deform that chart in native, before installation; they never become
    independent coordinates for each dense drawing sample or each video frame.
    """
    import numpy as np
    if not math.isfinite(max_step) or max_step<=0:
        raise ValueError('positive print sampling step required')
    context=_print_context(assets,instance_id,parameter_state)
    instance,template,size,placement=(context[k] for k in ('instance','template','size','placement'))
    requested={} if stations_by_path is None else stations_by_path
    if not isinstance(requested,dict):raise ValueError('print sampling stations must be keyed by path identity')
    unknown=set(requested)-{p['id'] for p in template['paths']}
    if unknown:raise PrintBindingError('unknown_print_sampling_path',path_ids=sorted(unknown))
    shapes=[];maximum=0.
    for path in template['paths']:
        vertices=_print_seed(path,context,core)
        if path['id'] in requested:
            explicit=np.asarray(requested[path['id']],dtype=float)
            if (explicit.ndim!=1 or not len(explicit) or not np.isfinite(explicit).all()
                    or np.any(explicit<0.) or np.any(explicit>1.) or np.any(np.diff(explicit)<0.)):
                raise PrintBindingError('invalid_print_sampling_stations',path_id=path['id'])
            stations=explicit.tolist()
        else:stations=_drawing_stations(path,vertices,context,core,max_step)
        data=evaluate_print_path(assets,instance_id,path['id'],board,core,stations,
                                 jacobian=False,_context=context)
        sz=np.asarray(data['sz']);xyz=np.asarray(data['xyz'])
        check=np.asarray(core.board_points(board,sz[:,0],sz[:,1]))
        if len(xyz):maximum=max(maximum,float(np.max(abs(xyz-check))))
        shapes.append({key:deepcopy(path[key]) for key in
            ('id','closed','role','parent_contour_key','boundary_side','logical_labels','display_fill','geometry') if key in path}|
            {'stations':stations,'sz':sz.tolist(),'xyz':xyz.tolist(),
             'parameter_block_ids':data['dependency_block_ids'],
             'sample_role':'model_drawing_support_not_source_observation'})
    if maximum>1e-12:raise PrintBindingError('native_print_chart_board_disagreement',maximum=maximum)
    return {'instance_id':instance_id,'template_id':template['id'],'surface_id':instance['scene_surface_id'],
        'parameters':context['parameters'],'size_block_id':size['id'],'placement_block_id':placement['id'],
        'paths':shapes,'chart_board_max_abs':maximum,'native_entity_kind':5,
        'internal_geometry_schema':None if context['internal'] is None else context['internal']['schema'],
        'parameter_state_consumed':parameter_state is not None,
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


def print_parameter_dependencies(assets, instance_id, path_id):
    from .print_parameters import print_parameter_dependencies as dependencies
    return dependencies(assets, instance_id, path_id)
