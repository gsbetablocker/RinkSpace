"""Shared print-internal parameter definitions and state binding.

This compiler records structural ownership, not source observations. Numerical
shape evaluation remains in the native core; current template vertices are the
immutable reference chart, not an alternate optimized state.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
import re

import numpy as np

from .print_assets import PrintBindingError, _registry, validate_template_topology

SCHEMA = 'rinkspace.print_internal_geometry.v1'
SHAPE_KIND = 'monotone_axis_warp.v1'
SHAPE_COORDINATES = ['translate_u', 'translate_v', 'log_scale_u', 'log_scale_v',
                     'u_log_interval_0', 'u_log_interval_1', 'u_log_interval_2',
                     'v_log_interval_0', 'v_log_interval_1', 'v_log_interval_2']
BLOCK_KINDS = frozenset({'print_size', 'print_placement', 'print_shape', 'print_square', 'print_rectangle'})


def _finite_vector(value, size, identity):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PrintBindingError('invalid_print_parameter_values', block_id=identity) from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise PrintBindingError('invalid_print_parameter_values', block_id=identity)
    return result


def parameter_values(assets, parameter_state=None):
    """Return checked current block values, never filling a missing state ID.

    A complete shared State can contain other scene/camera blocks. Every block
    declared by this asset must still occur in that State, including explicit
    null placements of unlocalized identities. Reading never changes activity.
    """
    if parameter_state is not None and not isinstance(parameter_state, Mapping):
        raise PrintBindingError('print_parameter_state_must_be_mapping')
    blocks = _registry(assets.get('parameter_blocks', []))
    values = {}
    for identity, block in blocks.items():
        if block.get('kind') not in BLOCK_KINDS:
            raise PrintBindingError('unsupported_print_parameter_kind', block_id=identity,
                                    kind=block.get('kind'))
        size = block.get('size')
        expected = {'print_size': 2, 'print_placement': 3, 'print_shape': 10, 'print_square': 3, 'print_rectangle': 4}[block['kind']]
        if type(size) is not int or size != expected:
            raise PrintBindingError('invalid_print_parameter_size', block_id=identity)
        if parameter_state is not None and identity not in parameter_state:
            raise PrintBindingError('missing_current_print_state_block', block_id=identity)
        value = block.get('values') if parameter_state is None else parameter_state[identity]
        reference = block.get('values')
        if reference is None:
            if value is not None:
                raise PrintBindingError('unlocalized_print_state_cannot_invent_values', block_id=identity)
            values[identity] = None
            continue
        if value is None:
            raise PrintBindingError('initialized_print_state_is_null', block_id=identity)
        current = _finite_vector(value, size, identity)
        initial = _finite_vector(reference, size, identity)
        independent = block.get('independent_local_indices', list(range(size)))
        if (not isinstance(independent, list) or len(set(independent)) != len(independent)
                or any(type(i) is not int or not 0 <= i < size for i in independent)):
            raise PrintBindingError('invalid_print_tangent_layout', block_id=identity)
        frozen = sorted(set(range(size))-set(independent))
        if frozen and np.any(current[frozen] != initial[frozen]):
            raise PrintBindingError('inactive_print_coordinate_changed', block_id=identity,
                                    local_indices=frozen)
        if block.get('active') is False and np.any(current != initial):
            raise PrintBindingError('inactive_print_block_changed', block_id=identity)
        values[identity] = current.copy()
    return values


def _stable_id(prefix, template_id, path_ids):
    encoded = '\n'.join(sorted(path_ids)).encode('utf-8')
    return prefix+'/'+template_id+'/'+hashlib.sha256(encoded).hexdigest()[:20]


def _qualified_label(value):
    # Short display labels such as 'A' or 'L' identify a glyph class, not an
    # installed occurrence. Only fully qualified current material identities
    # can join disconnected parts automatically during this one-time compiler.
    if not isinstance(value, str) or not ('.' in value or '/' in value):
        return None
    return re.sub(r'_{1,2}part[0-9]+(?=/|$)', '', value)


def _group_indices(paths, group_overrides):
    parent = {identity: identity for identity in paths}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def join(a, b):
        if a not in parent or b not in parent:
            raise PrintBindingError('unknown_print_shape_group_path', path_id=a if a not in parent else b)
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    labels = {}
    for identity, path in paths.items():
        if path.get('parent_contour_key') is not None:
            join(identity, path['parent_contour_key'])
        if path.get('part_of_original_path_id') is not None:
            join(identity, path['part_of_original_path_id'])
        supplied = path.get('shape_group_identity')
        if supplied is not None:
            if not isinstance(supplied, str) or not supplied:
                raise PrintBindingError('invalid_print_shape_group_identity', path_id=identity)
            keys = ['explicit:'+supplied]
        else:
            keys = []
            for value in path.get('logical_labels', []):
                key = _qualified_label(value)
                if key is not None:
                    keys.append(key)
            key = _qualified_label(identity)
            if key is not None:
                keys.append(key)
        for key in keys:
            if key in labels:
                join(identity, labels[key])
            else:
                labels[key] = identity
    for override in group_overrides:
        if not isinstance(override, (list, tuple)) or len(override) < 2:
            raise PrintBindingError('invalid_print_shape_group_override')
        for identity in override[1:]:
            join(override[0], identity)
    groups = {}
    for identity in paths:
        groups.setdefault(root(identity), []).append(identity)
    return list(groups.values())


def _shape_bounds(records):
    points = []
    for path in records:
        q = np.asarray(path.get('material_uv'), dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2 or len(q) < 2 or not np.isfinite(q).all():
            raise PrintBindingError('invalid_print_shape_reference', path_id=path['id'])
        points.append(q)
    q = np.vstack(points)
    lo, hi = q.min(0), q.max(0)
    span = hi-lo
    nonzero = span > np.finfo(float).eps*max(1., float(np.max(abs(q))))*32
    reference_span = span.copy()
    for axis in (0, 1):
        if not nonzero[axis]:
            reference_span[axis] = max(float(np.max(span)), 1.)
            lo[axis] -= reference_span[axis]/2
            hi[axis] += reference_span[axis]/2
    # Affine-only curves do not gain six nuisance changes of station. Such
    # coordinates move points along the same straight/rectangular material
    # set and would be gauges against observation station variables.
    centred = q-q.mean(0)
    rank = np.linalg.matrix_rank(centred, tol=max(1., float(np.max(abs(q))))*1e-12)
    no_holes = not any(p.get('parent_contour_key') for p in records)
    tol = max(1., float(np.max(abs(q))))*1e-12
    actual_lo, actual_hi = q.min(0), q.max(0)
    on_box = np.all(np.any(np.isclose(q, actual_lo, atol=tol, rtol=0.) |
                             np.isclose(q, actual_hi, atol=tol, rtol=0.), axis=1))
    corners = [np.array([u, v]) for u in (actual_lo[0], actual_hi[0]) for v in (actual_lo[1], actual_hi[1])]
    has_corners = all(np.min(np.linalg.norm(q-corner, axis=1)) <= tol for corner in corners)
    affine_only = rank <= 1 or (no_holes and on_box and has_corners)
    independent = [0, 1]+([2] if nonzero[0] else [])+([3] if nonzero[1] else [])
    if not affine_only:
        independent.extend(range(4, 10))
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])], independent, affine_only



def _rectangle_families(paths):
    """Find four explicitly named sides, never infer a rectangle from pixels."""
    families = {}
    sides = {'top_edge', 'right_vertical', 'bottom_edge', 'left_vertical'}
    for identity,path in paths.items():
        side=path.get('boundary_side')
        if side not in sides or path.get('role')=='panel_outer_boundary':
            continue
        base=identity.rsplit('/path',1)[0]
        suffix='.'+side
        if not base.endswith(suffix):
            raise PrintBindingError('rectangle_side_lacks_shared_identity', path_id=identity)
        families.setdefault(base[:-len(suffix)],{})[side]=identity
    for identity, sides_by_name in families.items():
        if set(sides_by_name)!=sides:
            raise PrintBindingError('incomplete_print_rectangle_sides', group_id=identity)
    return families


def _apply_template_gauge(template, blocks):
    internal=template['internal_geometry']
    if any(b['kind']=='panel_chart_gauge' for b in internal['path_bindings'].values()):
        internal['coordinate_gauge']={'kind':'unit_panel_chart','source_measurement':False,
            'reason':'Instance size and independent placement carry whole-print pose and dimensions'}
        return
    groups=internal['groups']
    if not groups:
        raise PrintBindingError('print_template_has_no_coordinate_reference', template_id=template['id'])
    rectangles=[g for g in groups if g['parameterization']=='unfolded_rectangle.v1']
    candidates=rectangles or groups
    reference=sorted(candidates,key=lambda g:(-((g['reference_bounds'][2]-g['reference_bounds'][0])*            (g['reference_bounds'][3]-g['reference_bounds'][1])),g['id']))[0]
    block=blocks[reference['parameter_block_id']]
    independent=[i for i in block['independent_local_indices'] if i>=4]
    block['independent_local_indices']=independent;block['tangent_size']=len(independent)
    block['active']=bool(independent)
    block['gauge_fixed_local_indices']=list(range(4))
    reference['whole_print_coordinate_reference']=True
    internal['coordinate_gauge']={'kind':'reference_group_layout','group_id':reference['id'],
        'parameter_block_id':block['id'],'fixed_local_indices':list(range(4)),
        'source_measurement':False,
        'reason':'Remove template-wide translation/scale gauge; shared size and independent instance placement retain those freedoms'}


def parameterize_print_assets(assets, *, group_overrides=()):
    """Compile explicit shared internal geometry without changing any pixels.

    Overrides are lists of existing path IDs whose same-material ownership was
    already established by the caller. They are stored as groups and are never
    guessed again while consuming a numerical state.
    """
    result = deepcopy(assets)
    templates = _registry(result['templates'])
    existing = [t.get('internal_geometry') is not None for t in templates.values()]
    if any(existing):
        if not all(existing):
            raise PrintBindingError('partially_parameterized_print_asset')
        validate_print_parameters(result)
        return result
    blocks = _registry(result.get('parameter_blocks', []))
    for block in blocks.values():
        if block.get('kind') not in ('print_size', 'print_placement'):
            raise PrintBindingError('unsupported_initial_print_parameter_kind', block_id=block['id'])
        # Rebuilding artwork groups must preserve explicit installation/size
        # constraints, including an instance's fixed board-parallel rotation.
        active = block.get('values') is not None and block.get('active', True)
        independent = (block.get('independent_local_indices', list(range(block['size'])))
                       if active else [])
        block.update(active=bool(active), tangent_size=len(independent),
                     independent_local_indices=independent)
    for template in templates.values():
        paths = validate_template_topology(template)
        bindings, groups = {}, []
        ordinary = {}
        rectangle_ids=set()
        for family, sides in _rectangle_families(paths).items():
            side_points={side:np.asarray(paths[identity]['material_uv'],dtype=float) for side,identity in sides.items()}
            left=float(side_points['left_vertical'][:,0].mean());right=float(side_points['right_vertical'][:,0].mean())
            top=float(side_points['top_edge'][:,1].mean());bottom=float(side_points['bottom_edge'][:,1].mean())
            if not (left<right and top<bottom):
                raise PrintBindingError('invalid_print_rectangle_reference', group_id=family)
            identities=list(sides.values());group_id=_stable_id('print_group',template['id'],identities)
            block_id=_stable_id('print_rectangle',template['id'],identities)
            block={'id':block_id,'kind':'print_rectangle','size':4,
                'coordinates':['centre_u','centre_v','log_width_uv','log_height_uv'],
                'parameterization':'unfolded_rectangle.v1','values':[(left+right)/2,(top+bottom)/2,math.log(right-left),math.log(bottom-top)],
                'active':True,'tangent_size':4,'independent_local_indices':[0,1,2,3],
                'owner_template_id':template['id']}
            result['parameter_blocks'].append(block);blocks[block_id]=block
            groups.append({'id':group_id,'parameter_block_id':block_id,'path_ids':identities,
                'parameterization':'unfolded_rectangle.v1','reference_bounds':[left,top,right,bottom],
                'initialization':'arithmetic_mean_of_each_declared_side_not_a_fit',
                'original_path_vertices_preserved':True,'source_observations_created':0,
                'preserved_structure':['four_straight_sides','common_corners','opposite_sides_parallel','right_angles']})
            side_codes={'top_edge':0,'right_vertical':1,'bottom_edge':2,'left_vertical':3}
            for side,identity in sides.items():
                q=side_points[side]
                if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or not np.isfinite(q).all():
                    raise PrintBindingError('invalid_print_rectangle_seed_side',path_id=identity)
                axis=0 if side in ('top_edge','bottom_edge') else 1
                direction=1 if side in ('top_edge','right_vertical') else -1
                reverse=bool((q[-1,axis]-q[0,axis])*direction<0)
                bindings[identity]={'kind':'rectangle_side','group_id':group_id,'side':side_codes[side],'reverse':reverse}
                rectangle_ids.add(identity)
        for identity, path in paths.items():
            if identity in rectangle_ids:
                continue
            geometry = path.get('geometry')
            if path.get('role') == 'panel_outer_boundary':
                if geometry is not None:
                    raise PrintBindingError('ambiguous_print_panel_geometry', path_id=identity)
                bindings[identity] = {'kind': 'panel_chart_gauge'}
            elif geometry is not None:
                if geometry.get('kind') != 'unfolded_square' or 'material_uv' in path:
                    raise PrintBindingError('unsupported_print_internal_geometry', path_id=identity)
                centre = _finite_vector(geometry.get('center_uv'), 2, identity)
                side = geometry.get('side_over_panel_height')
                if not isinstance(side, (int, float)) or not math.isfinite(side) or side <= 0:
                    raise PrintBindingError('invalid_print_square_reference', path_id=identity)
                block_id = _stable_id('print_square', template['id'], [identity])
                block = {'id':block_id, 'kind':'print_square', 'size':3,
                         'coordinates':['centre_u','centre_v','log_side_over_panel_height'],
                         'parameterization':'unfolded_square.v1',
                         'values':[float(centre[0]),float(centre[1]),math.log(side)],
                         'active':True,'tangent_size':3,'independent_local_indices':[0,1,2],
                         'owner_template_id':template['id']}
                if block_id in blocks:
                    raise PrintBindingError('duplicate_print_internal_block', block_id=block_id)
                result['parameter_blocks'].append(block);blocks[block_id]=block
                bindings[identity]={'kind':'unfolded_square','parameter_block_id':block_id}
            else:
                ordinary[identity]=path
        selected_overrides = []
        for override in group_overrides:
            hit = set(override).intersection(paths)
            if hit:
                if not set(override).issubset(ordinary):
                    raise PrintBindingError('print_group_override_crosses_template_or_analytic_geometry')
                selected_overrides.append(override)
        for identities in _group_indices(ordinary, selected_overrides):
            bounds, independent, affine_only = _shape_bounds([ordinary[i] for i in identities])
            block_id = _stable_id('print_shape', template['id'], identities)
            group_id = _stable_id('print_group', template['id'], identities)
            block = {'id':block_id,'kind':'print_shape','size':10,
                     'coordinates':list(SHAPE_COORDINATES),'parameterization':SHAPE_KIND,
                     'values':[0.]*10,'active':True,'tangent_size':len(independent),
                     'independent_local_indices':independent,'owner_template_id':template['id']}
            if block_id in blocks:
                raise PrintBindingError('duplicate_print_internal_block', block_id=block_id)
            result['parameter_blocks'].append(block);blocks[block_id]=block
            groups.append({'id':group_id,'parameter_block_id':block_id,'path_ids':identities,
                           'reference_bounds':bounds,'parameterization':SHAPE_KIND,
                           'affine_only_material_set':bool(affine_only),
                           'preserved_structure':['path_identity','path_order','closed_or_open_support',
                                                 'parent_hole_ownership','axis_aligned_straight_edges'],
                           'source_observations_created':0})
            for identity in identities:
                bindings[identity]={'kind':'shape_group','group_id':group_id}
        template['internal_geometry']={'schema':SCHEMA,'groups':groups,'path_bindings':bindings,
            'panel_chart_gauge':'unit_material_rectangle_with_instance_size_and_placement',
            'shape_scope':'shared_template_not_per_instance_or_per_frame',
            'drawing_vertices_are_not_free_parameters':True,'source_values_rewritten':False}
        _apply_template_gauge(template,blocks)
    result.setdefault('counts',{})['internal_shape_groups']=sum(len(t['internal_geometry']['groups']) for t in templates.values())
    result['counts']['internal_parameter_blocks']=sum(b['kind'] in ('print_shape','print_square','print_rectangle') for b in blocks.values())
    validate_print_parameters(result)
    return result


def parameterize_print_layout(assets, layouts):
    """Compile explicit artwork-component layout, preserving all source paths.

    ``layouts`` maps template IDs to a complete list of named material groups.
    Every ordinary path must be named exactly once, including holes and parts.
    The existing native shape evaluator carries only a shared positive affine
    map for each group. Its six nonlinear warp coordinates remain zero. This
    is an explicit model change, never inferred from labels or observation UV.
    Analytic panels/rectangles/squares retain their existing construction.
    """
    source = deepcopy(assets)
    templates = _registry(source['templates'])
    if set(layouts) != set(templates):
        raise PrintBindingError('incomplete_print_component_layout')
    overrides = []
    definitions = {}
    for identity, template in templates.items():
        paths = validate_template_topology(template)
        rectangles = {p for sides in _rectangle_families(paths).values() for p in sides.values()}
        ordinary = {p for p, record in paths.items()
                    if p not in rectangles and record.get('role') != 'panel_outer_boundary'
                    and record.get('geometry') is None}
        assigned = set(); names = set()
        for definition in layouts[identity]:
            name = definition.get('id'); members = definition.get('path_ids')
            if (not isinstance(name, str) or not name or name in names
                    or not isinstance(members, list) or not members
                    or len(set(members)) != len(members)
                    or not set(members) <= ordinary or assigned.intersection(members)
                    or not isinstance(definition.get('evidence'), dict)
                    or not definition['evidence']):
                raise PrintBindingError('invalid_print_component_layout', template_id=identity)
            if type(definition.get('fixed_in_template_chart', False)) is not bool:
                raise PrintBindingError('invalid_print_component_activity', group_id=name)
            names.add(name); assigned.update(members)
            for path in members:
                parent = paths[path].get('parent_contour_key')
                if parent is not None and parent not in members:
                    raise PrintBindingError('print_component_splits_parent_hole', path_id=path)
            if len(members)>1: overrides.append(members)
            definitions[frozenset(members)] = deepcopy(definition)
        if assigned != ordinary:
            raise PrintBindingError('incomplete_print_component_paths',
                                    template_id=identity, path_ids=sorted(ordinary-assigned))
        template.pop('internal_geometry', None)
    source['parameter_blocks'] = [b for b in source['parameter_blocks']
                                 if b['kind'] in ('print_size', 'print_placement')]
    result = parameterize_print_assets(source, group_overrides=overrides)
    blocks = _registry(result['parameter_blocks'])
    found = set()
    for template in result['templates']:
        for group in template['internal_geometry']['groups']:
            if group['parameterization'] != SHAPE_KIND: continue
            key = frozenset(group['path_ids'])
            if key not in definitions:
                raise PrintBindingError('implicit_print_component_merge', path_ids=sorted(key))
            definition = definitions[key]; found.add(key)
            block = blocks[group['parameter_block_id']]
            independent = ([] if definition.get('fixed_in_template_chart', False)
                           else [i for i in block['independent_local_indices'] if i < 4])
            block.update(independent_local_indices=independent,
                         tangent_size=len(independent), active=bool(independent),
                         activity_reason='explicit_shared_artwork_component_layout')
            group.update(component_identity=definition['id'],
                         component_evidence=definition['evidence'],
                         nonlinear_warp_enabled=False,
                         fixed_in_template_chart=definition.get('fixed_in_template_chart', False),
                         preserved_structure=['shared_positive_axis_affine_map',
                             'relative_contour_geometry_and_spacing', 'parent_hole_ownership',
                             'closed_or_open_support', 'source_vertex_order'])
        template['internal_geometry']['layout_policy'] = {
            'schema':'rinkspace.print_component_layout.v1',
            'nonlinear_glyph_warp':False,
            'groups':deepcopy(layouts[template['id']]),
            'whole_installation_and_size_remain_separate':True,
            'reference_artwork_is_not_a_measured_physical_size':True}
    if found != set(definitions):
        raise PrintBindingError('missing_print_layout_component')
    validate_print_parameters(result)
    return result


def validate_print_parameters(assets):
    """Validate structural state ownership without re-reading source evidence."""
    blocks = _registry(assets.get('parameter_blocks', []))
    used = set()
    for template in assets['templates']:
        internal = template.get('internal_geometry')
        if internal is None:
            continue  # Explicitly historical/static asset; no hidden new block.
        if internal.get('schema') != SCHEMA:
            raise PrintBindingError('unsupported_print_internal_geometry_schema', template_id=template['id'])
        paths = validate_template_topology(template)
        groups = _registry(internal.get('groups', []))
        domain=template.get('material_domain')
        if domain is not None:
            boundary=domain.get('boundary_path_ids',[])
            if (domain.get('kind')!='physical_unit_panel'
                    or not isinstance(domain.get('evidence'),dict) or not domain['evidence']
                    or internal.get('coordinate_gauge',{}).get('kind')!='unit_panel_chart'
                    or len(boundary)!=4 or len(set(boundary))!=4
                    or any(p not in paths or paths[p].get('role')!='panel_outer_boundary' for p in boundary)):
                raise PrintBindingError('invalid_physical_print_panel_domain',template_id=template['id'])
        layout_policy = internal.get('layout_policy')
        if layout_policy is not None:
            if (layout_policy.get('schema') != 'rinkspace.print_component_layout.v1'
                    or layout_policy.get('nonlinear_glyph_warp') is not False):
                raise PrintBindingError('invalid_print_layout_policy', template_id=template['id'])
        bindings = internal.get('path_bindings')
        if not isinstance(bindings, Mapping) or set(bindings) != set(paths):
            raise PrintBindingError('incomplete_print_internal_path_bindings', template_id=template['id'])
        grouped = set()
        for group in groups.values():
            if group.get('parameterization')=='unfolded_rectangle.v1':
                block=blocks.get(group.get('parameter_block_id'),{})
                if block.get('kind')!='print_rectangle' or block.get('size')!=4 or block.get('owner_template_id')!=template['id']:
                    raise PrintBindingError('invalid_print_rectangle_state_block',group_id=group['id'])
                identities=group.get('path_ids',[])
                side_set=set()
                for identity in identities:
                    binding=bindings.get(identity,{})
                    if identity not in paths or identity in grouped or binding.get('kind')!='rectangle_side' or binding.get('group_id')!=group['id']:
                        raise PrintBindingError('invalid_print_rectangle_path_binding',path_id=identity)
                    if type(binding.get('side')) is not int or binding['side'] not in range(4) or type(binding.get('reverse')) is not bool:
                        raise PrintBindingError('invalid_print_rectangle_side',path_id=identity)
                    side_set.add(binding['side']);grouped.add(identity)
                if len(identities)!=4 or side_set!=set(range(4)):
                    raise PrintBindingError('incomplete_print_rectangle_state',group_id=group['id'])
                used.add(block['id'])
                continue
            block_id = group.get('parameter_block_id')
            block = blocks.get(block_id, {})
            if (block.get('kind') != 'print_shape' or block.get('size') != 10
                    or block.get('owner_template_id') != template['id']
                    or group.get('parameterization') != SHAPE_KIND):
                raise PrintBindingError('invalid_print_internal_shape_block', group_id=group['id'])
            if layout_policy is not None:
                declarations = [g for g in layout_policy.get('groups', [])
                                if g.get('id') == group.get('component_identity')]
                if (len(declarations) != 1
                        or set(declarations[0]['path_ids']) != set(group['path_ids'])
                        or declarations[0]['evidence'] != group.get('component_evidence')
                        or group.get('nonlinear_warp_enabled') is not False
                        or any(i >= 4 for i in block['independent_local_indices'])
                        or any(v != 0. for v in block['values'][4:])
                        or (declarations[0].get('fixed_in_template_chart', False)
                            and block['independent_local_indices'])):
                    raise PrintBindingError('print_layout_freedom_contract_changed', group_id=group['id'])
            bounds = _finite_vector(group.get('reference_bounds'), 4, group['id'])
            if not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
                raise PrintBindingError('degenerate_print_shape_bounds', group_id=group['id'])
            identities = group.get('path_ids')
            if not isinstance(identities,list) or not identities or len(set(identities)) != len(identities):
                raise PrintBindingError('invalid_print_shape_group_paths', group_id=group['id'])
            for identity in identities:
                if identity not in paths or identity in grouped or bindings[identity] != {'kind':'shape_group','group_id':group['id']}:
                    raise PrintBindingError('inconsistent_print_shape_group_path', path_id=identity)
                path = paths[identity]
                if path.get('geometry') is not None or path.get('role') == 'panel_outer_boundary':
                    raise PrintBindingError('analytic_print_path_in_general_warp', path_id=identity)
                q = np.asarray(path.get('material_uv'), dtype=float)
                if (q.ndim != 2 or q.shape[1] != 2 or len(q)<2 or not np.isfinite(q).all()
                        or np.any(q < bounds[:2]-1e-12) or np.any(q > bounds[2:]+1e-12)):
                    raise PrintBindingError('print_reference_outside_group_bounds', path_id=identity)
                grouped.add(identity)
            used.add(block_id)
        for identity,path in paths.items():
            binding = bindings[identity]
            if binding.get('kind') == 'shape_group':
                if identity not in grouped:
                    raise PrintBindingError('absent_print_shape_group', path_id=identity)
                parent = path.get('parent_contour_key')
                if parent is not None and bindings[parent] != binding:
                    raise PrintBindingError('print_hole_shape_is_not_shared_with_parent', path_id=identity)
            elif binding.get('kind') == 'rectangle_side':
                if identity not in grouped or path.get('boundary_side') is None:
                    raise PrintBindingError('absent_print_rectangle_group',path_id=identity)
            elif binding.get('kind') == 'panel_chart_gauge':
                if path.get('role') != 'panel_outer_boundary' or 'geometry' in path:
                    raise PrintBindingError('invalid_print_panel_gauge_binding', path_id=identity)
            elif binding.get('kind') == 'unfolded_square':
                block = blocks.get(binding.get('parameter_block_id'),{})
                if (path.get('geometry',{}).get('kind') != 'unfolded_square'
                        or block.get('kind') != 'print_square' or block.get('owner_template_id') != template['id']):
                    raise PrintBindingError('invalid_print_square_state_binding', path_id=identity)
                used.add(block['id'])
            else:
                raise PrintBindingError('unknown_print_internal_path_kind', path_id=identity)
    declared={b['id'] for b in blocks.values() if b.get('kind') in ('print_shape','print_square','print_rectangle')}
    if used != declared:
        raise PrintBindingError('unreferenced_or_missing_print_internal_block', block_ids=sorted(used^declared))
    parameter_values(assets)
    return {'parameter_blocks':len(blocks),'internal_blocks':len(used),'source_values_rewritten':False}


def resolve_path_parameters(assets, template, path, parameter_state=None):
    """Return current values and explicit typed path owner for a native call."""
    validate_print_parameters(assets)
    values = parameter_values(assets, parameter_state)
    internal = template.get('internal_geometry')
    if internal is None:
        return values, None, None
    binding = internal['path_bindings'][path['id']]
    group = None
    if binding['kind'] in ('shape_group','rectangle_side'):
        group = next(g for g in internal['groups'] if g['id']==binding['group_id'])
    return values, binding, group


def print_parameter_dependencies(assets, instance_id, path_id):
    """Static dependencies include unlocalized identities without inventing values."""
    from .print_assets import resolve_instance
    _,template,size,placement=resolve_instance(assets,instance_id)
    paths=validate_template_topology(template)
    if path_id not in paths:
        raise PrintBindingError('unknown_print_template_path',path_id=path_id)
    validate_print_parameters(assets)
    result=[placement['id'],size['id']]
    internal=template.get('internal_geometry')
    if internal is not None:
        binding=internal['path_bindings'][path_id]
        if binding['kind'] in ('shape_group','rectangle_side'):
            group=next(g for g in internal['groups'] if g['id']==binding['group_id'])
            result.append(group['parameter_block_id'])
        elif binding['kind']=='unfolded_square':result.append(binding['parameter_block_id'])
    return result
