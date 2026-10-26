"""Shared scene parameter layout and immutable numeric state.

Static definitions and source observations are not candidate values. All print
instances consume this one ID/offset layout; no advertisement solver lives here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .scene_assets import read_scene_assets, scene_content_identity

LAYOUT_SCHEMA = 'rinkspace.parameter_layout.v1'
LOCAL_LAYOUT_SCHEMA = 'rinkspace.parameter_layout.v2'
STATE_LAYOUT = 'shared_scene_parameters.v1'


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def compile_scene_parameters(scene, scene_path, *, local_charts=None, optimize_scene=False):
    """Compile present numeric blocks only; never activate observations.

    Board/layout activity is explicit. The rink-length gauge and declared
    dependent coordinates remain fixed; prints share their existing blocks.
    Unlocalized blocks retain None and allocate no invented numeric values.
    """
    from .print_parameters import validate_print_parameters
    blocks, values, identities = [], [], set()

    def add(record, initial, target, *, active=False):
        identity = record['id']
        if identity in identities:
            raise ValueError('Duplicate shared parameter block: '+identity)
        identities.add(identity)
        size = record['size']
        if type(size) is not int or size <= 0:
            raise ValueError('Invalid parameter block size: '+identity)
        indices = record.get('independent_local_indices', list(range(size)))
        if (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < size for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid independent parameter coordinates: '+identity)
        available = initial is not None
        enabled = bool(record.get('active', active)) and available
        numeric = None if not available else np.asarray(initial, dtype='<f8')
        if numeric is not None and (numeric.shape != (size,) or not np.isfinite(numeric).all()):
            raise ValueError('Nonfinite/incomplete parameter values: '+identity)
        independent = indices if enabled else []
        block = {key: deepcopy(value) for key, value in record.items()
                 if key not in ('values', 'offset', 'active', 'original_block')}
        block.update(id=identity, size=size, offset=len(values) if available else None,
            available=available, active=enabled, independent_local_indices=independent,
            tangent_size=len(independent), target=deepcopy(target),
            constant_values=([None if i in independent else float(numeric[i]) for i in range(size)]
                             if available else None))
        if available:
            values.extend(numeric.tolist())
        blocks.append(block)

    scene_id = scene.get('scene_id', 'scene')
    if 'board' in scene:
        add(dict(id='scene/'+scene_id+'/board', kind='board_shape', size=len(scene['board']),
                 **({'independent_local_indices':list(range(1,len(scene['board'])))} if optimize_scene else {})),
            scene['board'], {'kind':'board'}, active=optimize_scene)
    layout = scene.get('shared_layout')
    if layout is not None:
        claimed = set()
        for record in scene.get('parameter_blocks', []):
            start, size = record['offset'], record['size']
            support = set(range(start, start+size))
            if start < 0 or start+size > len(layout) or claimed.intersection(support):
                raise ValueError('Overlapping/out-of-range shared scene parameter block')
            claimed.update(support)
            descriptor = {**record, 'id':'scene/'+scene_id+'/'+record['id'], 'kind':'shared_geometry'}
            add(descriptor, layout[start:start+size], {'kind':'shared_layout', 'offset':start}, active=optimize_scene)
        if claimed != set(range(len(layout))):
            raise ValueError('Shared scene layout must register every stored coordinate')
    shared_assets = {}
    for reference, assets in read_scene_assets(scene, scene_path):
        # Disjoint scene references may select instances from the same stored
        # template. Its shared size/shape blocks must be allocated once, with
        # the union of selected installations determining placement activity.
        identity = reference['sha256']
        if identity not in shared_assets:
            shared_assets[identity] = (assets, set())
        shared_assets[identity][1].update(reference.get('instance_ids', [reference.get('instance_id')]))
    for assets, selected in shared_assets.values():
        validate_print_parameters(assets)
        allowed_placements = {i['placement_block_id'] for i in assets['instances'] if i['id'] in selected}
        for record in assets.get('parameter_blocks', []):
            descriptor = record
            if record['kind'] == 'print_placement' and record['id'] not in allowed_placements:
                # An asset's complete values contract remains readable in an
                # instance subset. Omitted placements retain existing values
                # (or None), but acquire no active optimization coordinates.
                descriptor = {**record, 'active':False}
            add(descriptor, record.get('values'), {'kind':'print', 'asset_template_ids':
                [t['id'] for t in assets['templates']]}, active=record['kind'].startswith('print_'))
    definition = {'schema':LAYOUT_SCHEMA, 'scene_content_revision':scene_content_identity(scene, scene_path),
        'blocks':blocks, 'size':len(values), 'scope':'shared_scene_geometry',
        'source_observations_changed':False}
    if optimize_scene:definition['optimize_scene']=True
    definition['layout_revision'] = _identity(definition)
    state = ParameterState(definition, np.asarray(values, dtype='<f8'))
    if local_charts is not None:
        _check_scene_charts(scene, scene_path, local_charts)
        state, _ = migrate_parameter_state(state, local_charts, station_bindings=(), core=None)
    return state


class ParameterState(Mapping):
    """Immutable view of one validated common float64 state vector."""
    def __init__(self, layout, values):
        if not isinstance(layout, dict) or layout.get('schema') not in (LAYOUT_SCHEMA, LOCAL_LAYOUT_SCHEMA):
            raise ValueError('Unknown scene parameter layout')
        raw = np.asarray(values)
        if (raw.dtype.str != '<f8' or raw.ndim != 1 or not raw.flags.c_contiguous
                or not np.isfinite(raw).all() or len(raw) != layout.get('size')):
            raise ValueError('State values must be finite contiguous float64 with exact layout size')
        expected = {k:v for k,v in layout.items() if k != 'layout_revision'}
        if layout.get('layout_revision') != _identity(expected):
            raise ValueError('Parameter layout content identity differs')
        registry, claimed = {}, set()
        local_layout = layout['schema'] == LOCAL_LAYOUT_SCHEMA
        tangent_offset = 0
        for block in layout['blocks']:
            identity, size = block.get('id'), block.get('size')
            if not isinstance(identity, str) or not identity or identity in registry:
                raise ValueError('Duplicate/missing state parameter identity')
            if type(size) is not int or size <= 0:
                raise ValueError('Invalid state parameter block size')
            offset = block.get('offset')
            independent = block.get('independent_local_indices')
            if (not isinstance(independent, list) or len(set(independent)) != len(independent)
                    or any(type(i) is not int or not 0 <= i < size for i in independent)
                    or (not local_layout and block.get('tangent_size') != len(independent))):
                raise ValueError('Invalid state tangent layout: '+identity)
            if not block.get('available'):
                if offset is not None or block.get('active') or independent:
                    raise ValueError('Unlocalized parameters cannot acquire values or active columns')
            else:
                if type(offset) is not int or offset < 0 or offset+size > len(raw):
                    raise ValueError('State parameter offset is out of bounds')
                support = set(range(offset, offset+size))
                if claimed.intersection(support):
                    raise ValueError('State parameter blocks overlap')
                claimed.update(support)
                constants = block.get('constant_values')
                if not isinstance(constants, list) or len(constants) != size:
                    raise ValueError('State constant-coordinate layout is incomplete')
                for i, value in enumerate(constants):
                    if i in independent:
                        if value is not None:
                            raise ValueError('Active coordinate declared constant')
                    elif value is None or raw[offset+i] != value:
                        raise ValueError('Constrained/frozen parameter coordinate changed: '+identity)
            if local_layout:
                chart = block.get('local_chart')
                tangent_size = block.get('tangent_size')
                if type(tangent_size) is not int or tangent_size < 0 or block.get('tangent_offset') != tangent_offset or block.get('ambient_size') != size:
                    raise ValueError('Invalid ambient/tangent layout: '+identity)
                if chart is None:
                    expected_map = np.eye(size)[:, independent]
                    if block.get('chart_revision') is not None:
                        raise ValueError('Selection block has an unexpected chart revision')
                else:
                    from .shape_chart import validate_shape_local_chart
                    validate_shape_local_chart(chart)
                    if chart['parameter_block_id'] != identity or chart['ambient_size'] != size or block.get('chart_revision') != chart['revision'] or not block['active']:
                        raise ValueError('Shape chart ownership/activity mismatch')
                    expected_map = np.asarray(chart['d_ambient_d_local'])
                    if set(range(4))-set(independent) != set(chart['fixed_affine_indices']) or not set(range(4,10)) <= set(independent):
                        raise ValueError('Shape chart changes existing coordinate gauge')
                    canonical = raw[offset:offset+size]-np.asarray(chart['ambient_origin'])
                    projected = expected_map @ np.linalg.lstsq(expected_map, canonical, rcond=None)[0]
                    if not np.allclose(projected, canonical, rtol=0., atol=2e-12):
                        raise ValueError('Noncanonical shape state requires explicit station migration')
                expansion = np.asarray(block.get('d_ambient_d_local'), dtype=float)
                if expected_map.shape != (size, tangent_size) or expansion.shape != expected_map.shape or not np.array_equal(expansion, expected_map):
                    raise ValueError('Invalid local expansion map: '+identity)
                tangent_offset += tangent_size
            registry[identity] = deepcopy(block)
        if local_layout and (layout.get('tangent_size') != tangent_offset or layout.get('ambient_size') != len(raw)):
            raise ValueError('Incomplete common tangent layout')
        if claimed != set(range(len(raw))):
            raise ValueError('State contains unregistered numeric coordinates')
        self._layout = deepcopy(layout)
        self._values = np.array(raw, dtype='<f8', order='C', copy=True)
        self._values.setflags(write=False)
        def freeze(value):
            if isinstance(value, dict):
                return MappingProxyType({k:freeze(v) for k,v in value.items()})
            if isinstance(value, list):
                return tuple(freeze(v) for v in value)
            return value
        self.blocks = freeze(registry)

    @property
    def layout(self):
        return deepcopy(self._layout)

    @property
    def values(self):
        result = self._values.view()
        result.setflags(write=False)
        return result

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, identity):
        block = self.blocks[identity]
        if not block['available']:
            return None
        return self.values[block['offset']:block['offset']+block['size']]

    def active_columns(self, block_ids=None):
        wanted = set(self.blocks) if block_ids is None else set(block_ids)
        unknown = wanted-set(self.blocks)
        if unknown:
            raise ValueError('Unknown requested active parameter IDs: '+str(sorted(unknown)))
        if self._layout['schema'] == LOCAL_LAYOUT_SCHEMA:
            return np.asarray([b['tangent_offset']+i for b in self.blocks.values() if b['id'] in wanted
                for i in range(b['tangent_size'])], dtype='<i8')
        return np.asarray([b['offset']+i for b in self.blocks.values() if b['id'] in wanted
            and b['active'] for i in b['independent_local_indices']], dtype='<i8')

    def expansion(self, identity):
        block = self.blocks[identity]
        if self._layout['schema'] == LOCAL_LAYOUT_SCHEMA:
            return np.asarray(block['d_ambient_d_local'], dtype=float).reshape(block['size'], block['tangent_size'])
        return np.eye(block['size'])[:, list(block['independent_local_indices'])]

    def local_values(self, identity):
        block = self.blocks[identity]
        if not block['available']:
            return np.empty(0)
        chart = block.get('local_chart')
        if chart is None:
            return self[identity][list(block['independent_local_indices'])].copy()
        ambient = self[identity]-np.asarray(chart['ambient_origin'])
        return np.linalg.lstsq(self.expansion(identity), ambient, rcond=None)[0]

    def retract_block(self, identity, tangent_delta):
        block = self.blocks[identity]
        delta = np.asarray(tangent_delta, dtype=float)
        if delta.shape != (block['tangent_size'],) or not np.isfinite(delta).all():
            raise ValueError('Invalid block tangent delta')
        if not block['available']:
            if delta.size:
                raise ValueError('Cannot update unlocalized parameters')
            return self
        return self.with_blocks({identity:self[identity]+self.expansion(identity) @ delta})

    def retract(self, tangent_delta):
        if self._layout['schema'] != LOCAL_LAYOUT_SCHEMA:
            raise ValueError('Compile the explicit tangent layout before a global retraction')
        delta = np.asarray(tangent_delta, dtype=float)
        if delta.shape != (self._layout['tangent_size'],) or not np.isfinite(delta).all():
            raise ValueError('Global tangent delta has wrong size or nonfinite values')
        return self.with_blocks({identity:self[identity]+self.expansion(identity) @
            delta[block['tangent_offset']:block['tangent_offset']+block['tangent_size']]
            for identity, block in self.blocks.items() if block['available']})

    def with_blocks(self, replacements):
        """Create a numeric candidate, respecting the compiled active set/gauge."""
        values = self.values.copy()
        for identity, proposed in replacements.items():
            if identity not in self.blocks:
                raise ValueError('Unknown candidate parameter block: '+identity)
            block = self.blocks[identity]
            candidate = np.asarray(proposed, dtype='<f8')
            if not block['available'] or candidate.shape != (block['size'],):
                raise ValueError('Candidate block is unlocalized or has the wrong size: '+identity)
            values[block['offset']:block['offset']+block['size']] = candidate
        return ParameterState(self.layout, values)

    def apply_scene(self, scene):
        """Extract current scene values; geometry remains evaluated by native."""
        result = deepcopy(scene)
        for identity, block in self.blocks.items():
            target = block['target']
            if target['kind'] == 'board':
                result['board'] = self[identity].tolist()
            elif target['kind'] == 'shared_layout':
                start = target['offset']
                result['shared_layout'][start:start+block['size']] = self[identity].tolist()
        return result


def load_parameter_state(state_path, *, expected_scene=None, expected_scene_path=None):
    """Load only the current layout and values; no provenance/history replay."""
    from ..session.dataset import load_array, read_json, relative
    path = Path(state_path).resolve(strict=True)
    state = read_json(path)
    if state.get('schema') != 'rinkspace.state.v1' or state.get('layout_family') != STATE_LAYOUT:
        raise ValueError('Expected a shared scene parameter state')
    layout = read_json(relative(path.parent, state['parameter_layout_ref']))
    if state.get('layout_revision') != layout.get('layout_revision') or state.get('parameter_blocks') != layout.get('blocks'):
        raise ValueError('State ID/offset layout differs from its parameter layout')
    desc = state.get('values_ref')
    if not isinstance(desc, dict):
        raise ValueError('Current state requires a typed values array reference')
    values = load_array(path.parent, desc, verify=True)
    result = ParameterState(layout, values)
    if expected_scene is not None:
        charts = ({b['id']:b['local_chart'] for b in result.layout['blocks'] if b.get('local_chart') is not None}
                  if result.layout['schema'] == LOCAL_LAYOUT_SCHEMA else None)
        actual = compile_scene_parameters(expected_scene, expected_scene_path, local_charts=charts, optimize_scene=result.layout.get("optimize_scene",False))
        if actual.layout != result.layout:
            raise ValueError('State parameter layout differs from the current scene definition')
    if state.get('station_bindings_ref') is not None:
        load_parameter_stations(path, parameter_state=result)
    return result


def write_parameter_state(root, parameter_state, *, scene_ref='scene.json', prepared_revision='unbound',
                          shots=(), required_shot_ids=(), selected_shot_ids=(), name='state.json',
                          station_bindings=None, layout_name='parameter_layout.json'):
    """Serialize values/layout into the same state.v1 used by review/future solve.

    This is a parameter snapshot, not a solver checkpoint or an optimization run.
    Per-observation stations/noise and approval remain owned by prepared inputs.
    """
    from ..session.dataset import write_array, write_json
    from ..path_policy import guard_project_write
    root = Path(root)
    if Path(name).name != name or not name.endswith('.json'):
        raise ValueError('State snapshot name must be one JSON filename')
    if Path(layout_name).name != layout_name or not layout_name.endswith('.json'):
        raise ValueError('Layout snapshot name must be one JSON filename')
    values_name = 'state_'+Path(name).stem
    for relative_path in (layout_name, 'arrays/'+values_name+'.npy', name):
        guard_project_write(root/relative_path)
    if (root/name).exists() or (root/'arrays'/(values_name+'.npy')).exists():
        raise FileExistsError('State snapshots never overwrite existing numeric candidates')
    layout_path = root/layout_name
    if layout_path.exists():
        existing = json.loads(layout_path.read_text(encoding='utf-8-sig'))
        if existing != parameter_state.layout:
            raise ValueError('State snapshot folder contains a different parameter layout')
    else:
        write_json(layout_path, parameter_state.layout)
    values_ref = write_array(root, values_name, parameter_state.values, '<f8')
    selected, required = list(selected_shot_ids), list(required_shot_ids)
    state = {'schema':'rinkspace.state.v1', 'state_id':'parameters_'+_identity({
        'layout':parameter_state.layout['layout_revision'], 'values':values_ref['sha256']})[:20],
        'layout_family':STATE_LAYOUT, 'layout_revision':parameter_state.layout['layout_revision'],
        'scene_revision':parameter_state.layout['scene_content_revision'],
        'prepared_revision':prepared_revision, 'scene_ref':scene_ref,
        'parameter_layout_ref':layout_name, 'parameter_blocks':parameter_state.layout['blocks'],
        'values_ref':values_ref, 'local_values_ref':None, 'local_state_status':'owned_by_prepared_inputs_not_invented',
        'shots':deepcopy(list(shots)), 'required_shot_ids':required, 'selected_shot_ids':selected,
        'scope_status':'complete' if required and set(required)==set(selected) else 'subset',
        'origin':{'kind':'current_scene_parameters_without_optimization'},
        'fit_executed':False, 'cuda_executed':False, 'user_approval':'not_reviewed'}
    if station_bindings is not None:
        from ..session.dataset import digest
        station_name = 'stations_'+Path(name).stem+'.json'
        guard_project_write(root/station_name)
        if (root/station_name).exists():
            raise FileExistsError('Station snapshots never overwrite existing candidates')
        _validate_station_bindings(parameter_state, station_bindings)
        write_json(root/station_name, list(station_bindings))
        state['station_bindings_ref'] = {'path':station_name, 'sha256':digest(root/station_name)}
        state['local_state_status'] = 'explicit_path_station_snapshot'
    write_json(root/name, state)
    return state


class SceneParameterModel:
    """Present scene topology plus one common parameter layout for all prints."""
    def __init__(self, scene, scene_path, *, local_charts=None, optimize_scene=False):
        self.scene, self.scene_path = deepcopy(scene), Path(scene_path).resolve()
        self.initial_state = compile_scene_parameters(scene, scene_path, local_charts=local_charts, optimize_scene=optimize_scene)
        self.assets = read_scene_assets(scene, scene_path)
        self.paths = {}
        for reference, assets in self.assets:
            templates = {t['id']:t for t in assets['templates']}
            selected = reference.get('instance_ids', [reference.get('instance_id')])
            for instance in assets['instances']:
                if instance['id'] not in selected:
                    continue
                for path in templates[instance['template_id']]['paths']:
                    identity = instance['id']+'::'+path['id']
                    if identity in self.paths:
                        raise ValueError('Duplicate compiled print feature path')
                    self.paths[identity] = (assets, instance['id'], path['id'])
        self.factor_plan = FactorPlan(self.initial_state,
            {identity:self.dependencies(identity) for identity in self.paths},
            scope='print_geometry_dependencies_not_complete_camera_factors')

    def dependencies(self, feature_path_id):
        from .print_parameters import print_parameter_dependencies
        if feature_path_id not in self.paths:
            raise ValueError('Unregistered print feature path: '+feature_path_id)
        assets, instance_id, path_id = self.paths[feature_path_id]
        return print_parameter_dependencies(assets, instance_id, path_id)

    def evaluate_print(self, feature_path_id, stations, core, *, state=None, jacobian=True):
        """Evaluate XYZ and explicitly aligned derivatives of this State.

        ``jacobians_by_block`` retains the native stored-coordinate layout.
        For assembly, ``active_jacobians_by_block`` has exactly one last-axis
        column per global index in ``parameter_columns``. Frozen blocks have
        zero active columns; this is geometry linearization, not a factor J.
        """
        from .print_assets import evaluate_print_path
        current = self.initial_state if state is None else state
        if current.layout != self.initial_state.layout:
            raise ValueError('Candidate and compiled scene parameter layouts differ')
        assets, instance_id, path_id = self.paths[feature_path_id]
        result = evaluate_print_path(assets, instance_id, path_id,
            current.apply_scene(self.scene)['board'], core, stations,
            parameter_state=current, jacobian=jacobian)
        result['parameter_columns'] = self.factor_plan.columns(feature_path_id)
        if jacobian:
            result['active_jacobians_by_block'] = self.factor_plan.pullback(
                feature_path_id, result['jacobians_by_block'])
        result['layout_revision'] = current.layout['layout_revision']
        return result



def _check_scene_charts(scene, scene_path, charts):
    from .shape_chart import compile_shape_local_chart, shape_group_definition
    remaining = set(charts)
    for _, assets in read_scene_assets(scene, scene_path):
        for template in assets['templates']:
            internal = template.get('internal_geometry', {})
            for group in internal.get('groups', []):
                identity = group['parameter_block_id']
                if identity not in charts:
                    continue
                chart = charts[identity]
                definition = shape_group_definition(template, group,
                    fixed_material_points=chart['definition'].get('fixed_material_points', []))
                if compile_shape_local_chart(definition, internal['coordinate_gauge']) != chart:
                    raise ValueError('Local chart does not describe the current complete group: '+identity)
                remaining.discard(identity)
    if remaining:
        raise ValueError('Unknown scene chart blocks: '+str(sorted(remaining)))


def migrate_parameter_state(state, charts, *, station_bindings, core):
    """Explicitly canonicalize ambient state and its supplied auxiliary stations.

    The caller must supply all existing station bindings (an explicit empty
    sequence declares there are none). Source records/UV/PTS/W are not inputs
    to be rewritten. Binding fields other than station values are copied.
    """
    from .shape_chart import encode_shape_state, expand_shape_state, migrate_shape_stations, validate_shape_local_chart
    if set(charts)-set(state.blocks):
        raise ValueError('Unknown local chart block')
    layout, values = state.layout, state.values.copy()
    if layout['schema'] == LOCAL_LAYOUT_SCHEMA:
        current_charts = {b['id']:b['local_chart'] for b in layout['blocks'] if b.get('local_chart') is not None}
        if any(identity in charts and charts[identity] != chart for identity, chart in current_charts.items()):
            raise ValueError('Replacing an established chart requires a new explicit migration')
        charts = {**current_charts, **charts}
    offset = 0
    for block in layout['blocks']:
        identity, size = block['id'], block['size']
        chart = charts.get(identity)
        if chart is not None:
            validate_shape_local_chart(chart)
            if not block['active'] or block['kind'] != 'print_shape' or chart['parameter_block_id'] != identity:
                raise ValueError('Chart cannot activate or change a block kind')
            encoded = encode_shape_state(chart, state[identity])
            values[block['offset']:block['offset']+size] = expand_shape_state(chart, encoded)
            expansion = np.asarray(chart['d_ambient_d_local'])
        else:
            expansion = np.eye(size)[:, block['independent_local_indices']]
        block.update(ambient_size=size, tangent_offset=offset, tangent_size=expansion.shape[1],
            local_chart=deepcopy(chart), chart_revision=None if chart is None else chart['revision'],
            d_ambient_d_local=expansion.tolist())
        offset += expansion.shape[1]
    layout.update(schema=LOCAL_LAYOUT_SCHEMA, ambient_size=len(values), tangent_size=offset)
    layout.pop('layout_revision')
    layout['layout_revision'] = _identity(layout)
    migrated = ParameterState(layout, values)
    bindings = deepcopy(list(station_bindings))
    seen = set()
    for binding in bindings:
        identity = binding['block_id']
        if binding['id'] in seen:
            raise ValueError('Duplicate auxiliary station identity')
        seen.add(binding['id'])
        if identity in charts:
            if core is None:
                raise ValueError('Native geometry is required to migrate existing stations')
            binding['stations'] = migrate_shape_stations(charts[identity], binding['path_id'],
                binding['stations'], state[identity], migrated[identity], core).tolist()
    return migrated, bindings



def _validate_station_bindings(state, bindings):
    seen = set()
    for binding in bindings:
        identity = binding.get('id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Auxiliary stations require unique stable IDs')
        seen.add(identity)
        block = state.blocks.get(binding.get('block_id'))
        if block is None or not block['available']:
            raise ValueError('Auxiliary station has no available block')
        stations = np.asarray(binding.get('stations'), dtype=float)
        if stations.ndim != 1 or not np.isfinite(stations).all() or np.any((stations < 0) | (stations > 1)):
            raise ValueError('Auxiliary station outside finite support')
        chart = block.get('local_chart')
        if chart is not None and binding.get('path_id') not in chart['path_ids']:
            raise ValueError('Auxiliary station path differs from local chart')


def load_parameter_stations(state_path, *, parameter_state=None):
    from ..session.dataset import read_json, relative, digest
    path = Path(state_path).resolve(strict=True)
    record = read_json(path)
    reference = record.get('station_bindings_ref')
    if reference is None:
        return None
    station_path = relative(path.parent, reference['path'])
    if digest(station_path) != reference['sha256']:
        raise ValueError('Auxiliary station snapshot digest differs')
    bindings = read_json(station_path)
    state = load_parameter_state(path) if parameter_state is None else parameter_state
    _validate_station_bindings(state, bindings)
    return bindings


class FactorPlan:
    """One typed dependency/column plan, shared by geometry and future factors.

    This compiles layout maps only. It does not invent observations, whitening,
    camera derivatives, normal equations or a solver lifecycle.
    """
    def __init__(self, parameter_state, dependencies, *, scope='declared_dependencies'):
        self._state = parameter_state
        self._dependencies = deepcopy(dependencies)
        for identity, blocks in self._dependencies.items():
            if not isinstance(identity, str) or not identity or len(set(blocks)) != len(blocks) or set(blocks)-set(parameter_state.blocks):
                raise ValueError('Invalid factor dependencies: '+str(identity))
        self.layout_revision = parameter_state.layout['layout_revision']
        self.scope = scope

    def columns(self, factor_id):
        return {identity:self._state.active_columns([identity]) for identity in self._dependencies[factor_id]
            if self._state.blocks[identity]['available']}

    def pullback(self, factor_id, ambient_jacobians):
        result = {}
        for identity in self.columns(factor_id):
            jacobian = np.asarray(ambient_jacobians[identity], dtype=float)
            if jacobian.shape[-1] != self._state.blocks[identity]['size']:
                raise ValueError('Ambient Jacobian width differs from its block')
            result[identity] = jacobian @ self._state.expansion(identity)
        return result

    def to_dict(self):
        state = self._state
        used = {identity for blocks in self._dependencies.values() for identity in blocks}
        blocks = []
        for identity, block in state.blocks.items():
            if identity not in used:
                continue
            blocks.append({'id':identity, 'ambient_size':block['size'], 'ambient_offset':block['offset'],
                'tangent_offset':block.get('tangent_offset'), 'tangent_size':block['tangent_size'],
                'chart_revision':block.get('chart_revision'), 'columns':state.active_columns([identity]).tolist(),
                'd_ambient_d_local':state.expansion(identity).tolist(), 'available':block['available']})
        result = {'schema':'rinkspace.factor_plan.v1', 'layout_revision':self.layout_revision,
            'column_space':'packed_tangent' if state.layout['schema'] == LOCAL_LAYOUT_SCHEMA else 'legacy_ambient_selection',
            'scope':self.scope, 'blocks':blocks, 'factors':deepcopy(self._dependencies),
            'source_rows_or_groups_changed':False}
        result['revision'] = _identity(result)
        return result
