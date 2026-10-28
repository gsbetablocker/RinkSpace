"""Current dataset persistence. No historical producers or numerical models."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


class DatasetError(ValueError):
    def __init__(self, code, message, context=None):
        super().__init__(message)
        self.code, self.message, self.context = code, message, context or {}


def require(condition, message, **context):
    if not condition:
        raise DatasetError("invalid_input", message, context)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(root, value):
    """Manifest references are bounded; registered external assets use AssetStore."""
    root = Path(root).resolve()
    value = Path(value)
    require(not value.is_absolute(), "Manifest reference must be relative", path=str(value))
    path = (root / value).resolve()
    require(path.is_relative_to(root), "Manifest reference escapes its asset root", path=str(value))
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_array(root, name, values, dtype=None):
    root = Path(root)
    array = np.asarray(values, dtype=dtype)
    require(not array.dtype.hasobject, "Pickled/object arrays are forbidden", name=name)
    if array.dtype.kind in "iufc":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    require(array.dtype.kind not in "fc" or np.isfinite(array).all(), "Array contains nonfinite values", name=name)
    path = root / "arrays" / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {"path": path.relative_to(root).as_posix(), "dtype": array.dtype.str,
            "shape": list(array.shape), "sha256": digest(path)}


def load_array(root, desc, *, verify=False):
    path = relative(root, desc["path"])
    if verify:
        require(digest(path) == desc["sha256"], "Array content differs from current revision", path=str(path))
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(not array.dtype.hasobject and array.flags.c_contiguous, "Array must be C contiguous without objects", path=str(path))
    require(array.dtype.str == desc["dtype"] and list(array.shape) == desc["shape"],
            "Array shape/dtype differs from manifest", path=str(path))
    return array


@dataclass(frozen=True)
class PreparedReceipt:
    manifest_path: str
    dataset_id: str
    revision: str
    selected_shot_ids: tuple[str, ...]
    scope_status: str


def validate_current_noise_groups(groups, members, weights, arrays):
    """Validate current packed correlation bindings without coercing indices.

    Returns views of each complete group's members and whitening values. A
    rank-zero engineering-only group keeps zero statistical weight; numerical
    covariance estimation and source adjudication are not performed here.
    """
    require(members.dtype.str == '<i8' and members.ndim == 1,
            'Correlation members must be a native int64 vector')
    require(weights.dtype.str == '<f8' and weights.ndim == 1,
            'Correlation weights must be a float64 vector')
    require(isinstance(groups, list) and all(isinstance(g, dict) and isinstance(g.get('id'), str)
            and g['id'] for g in groups), 'Correlation groups require stable identities')
    require(len({g['id'] for g in groups}) == len(groups), 'Duplicate correlation group identity')
    n = len(arrays['pts'])
    indices = arrays.get('noise_group_index')
    splits = arrays.get('split')
    require(indices is not None and indices.dtype.str == '<i4' and indices.shape == (n,),
            'Correlation indices must match the current observation layout')
    require(splits is not None and splits.dtype == np.dtype('u1') and splits.shape == (n,),
            'Correlation split binding is missing or malformed')
    require(np.all((indices >= -1) & (indices < len(groups))), 'Invalid current correlation group index')
    claimed = np.zeros(n, dtype='u1')
    result = []
    for ordinal, group in enumerate(groups):
        fields = ('member_start', 'member_count', 'rank', 'weight_start')
        require(all(type(group.get(k)) is int and group[k] >= 0 for k in fields),
                'Correlation offsets, counts and rank must be nonnegative integers', group_id=group['id'])
        start, count, rank, begin = (group[k] for k in fields)
        require(count > 0 and start + count <= len(members) and begin + rank*count*2 <= len(weights),
                'Correlation support or weights are outside their arrays', group_id=group['id'])
        rr = members[start:start+count]
        require(len(np.unique(rr)) == count and np.all((rr >= 0) & (rr < n)),
                'Malformed correlation support', group_id=group['id'])
        require(not np.any(claimed[rr]), 'Correlation groups overlap', group_id=group['id'])
        claimed[rr] = 1
        require(np.all(indices[rr] == ordinal), 'Correlation membership is not reciprocal', group_id=group['id'])
        require(len(np.unique(splits[rr])) == 1, 'Correlated evidence closure crosses data split', group_id=group['id'])
        w = weights[begin:begin+rank*count*2]
        require(np.isfinite(w).all(), 'Malformed group whitening matrix', group_id=group['id'])
        result.append((rr, w))
    require(np.all(claimed[indices >= 0] == 1), 'Observation references absent correlation support')
    return result


def validate_current_stations(features, arrays, rows):
    """Validate auxiliary material identities and their evidence closure.

    A shared local index denotes the same material coordinate across exposures.
    It cannot silently join different curves, disjoint finite supports or a
    fitting row with a held-out row. No local state or source row is rewritten.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = len(arrays['pts'])
    local = arrays['local_index']
    support = arrays['support']
    require(local.dtype.str == '<i8' and local.shape == (n,),
            'Station identities must be a native int64 vector')
    require(support.dtype.str == '<f8' and support.shape == (n, 2),
            'Station support must be a float64 [N,2] array')
    require(np.isfinite(support[rows]).all(), 'Nonfinite station support')
    fixed = rows[arrays['kind'][rows] == 0]
    curves = rows[arrays['kind'][rows] == 1]
    require(np.all(local[fixed] == -1) and np.all(support[fixed] == 0),
            'Fixed material points cannot acquire per-observation stations')
    require(np.all(local[curves] >= 0), 'Finite curves require nonnegative station identities')
    require(np.all(support[curves, 0] <= support[curves, 1]), 'Reversed finite curve support')
    support_kind = arrays.get('support_kind', np.zeros(n, dtype='u1'))
    # Include the complete identity closure if a selected row owns the station.
    # Selecting one shot cannot conceal a conflicting split in another shot.
    order = np.argsort(local, kind='stable')
    sorted_local = local[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_local)) + 1, n]
    starts, stops = boundaries[:-1], boundaries[1:]
    shared = (stops-starts) > 1
    starts, stops = starts[shared], stops[shared]
    selected = np.isin(sorted_local[starts], local[curves])
    for start, stop in zip(starts[selected], stops[selected]):
        identity = sorted_local[start]
        members = order[start:stop]
        feature_keys = {features[int(i)].get('canonical_geometry_id',
                        features[int(i)].get('path_id', features[int(i)]['id']))
                        for i in arrays['feature_index'][members]}
        require(len(feature_keys) == 1, 'Shared station refers to different feature paths',
                local_index=int(identity))
        kinds = np.unique(support_kind[members])
        require(len(kinds) == 1, 'Shared station mixes finite and periodic coordinates',
                local_index=int(identity))
        if kinds[0] == 0:
            require(np.isfinite(support[members]).all() and
                    np.max(support[members, 0]) <= np.min(support[members, 1]),
                    'Shared station has incompatible finite supports', local_index=int(identity))
        if 'split' in arrays:
            require(len(np.unique(arrays['split'][members])) == 1,
                    'Shared material evidence closure crosses data split', local_index=int(identity))


def validate_current_observation_row(record, row, arrays, *, sources=None, shots=None, noise_groups=None):
    """Bind explicit current metadata to its numerical row without remeasurement.

    Historical information nested under provenance is left untouched. Missing
    optional metadata is not a new approval decision; an explicit pending,
    rejected or synthetic declaration cannot be contradicted by an active bit.
    """
    def value(name, field):
        require(name in arrays and row < len(arrays[name]),
                'Observation metadata lacks its current numerical binding', source_row=row, field=field)
        return arrays[name][row]

    def agrees(field, expected):
        require(record[field] == expected, 'Observation metadata differs from its current numerical row',
                source_row=row, field=field)

    def registry_record(index_name, registry, field):
        index = int(value(index_name, field))
        require(registry is not None and 0 <= index < len(registry),
                'Observation metadata lacks its current registry binding', source_row=row, field=field)
        return registry[index]

    if 'id' in record and 'observation_ids' in arrays:
        identity = value('observation_ids', 'id')
        agrees('id', identity.decode('utf-8') if isinstance(identity, bytes) else str(identity))
    if 'pts' in record:
        require(type(record['pts']) is int, 'Observation metadata PTS must be a native integer', source_row=row)
        agrees('pts', int(value('pts', 'pts')))
    for field in ('uv', 'source_uv'):
        if field in record:
            supplied = np.asarray(record[field])
            require(supplied.shape == (2,) and supplied.dtype.kind in 'iuf' and
                    np.isfinite(supplied).all() and np.array_equal(supplied, value('uv', field)),
                    'Observation metadata pixels differ from its current numerical row', source_row=row, field=field)
    source_fields = ('source_id', 'source_sha256', 'time_base', 'time_base_num', 'time_base_den')
    if any(field in record for field in source_fields):
        source = registry_record('source_index', sources, 'source_id')
        for field, source_key in (('source_id', 'id'), ('source_sha256', 'sha256'),
                                  ('time_base_num', 'time_base_num'), ('time_base_den', 'time_base_den')):
            if field in record:
                if field.startswith('time_base_'):
                    require(type(record[field]) is int, 'Source time base metadata must use integers',
                            source_row=row, field=field)
                agrees(field, source[source_key])
        if 'time_base' in record:
            try:
                supplied_clock = Fraction(record['time_base'])
                expected_clock = Fraction(source['time_base_num'], source['time_base_den'])
            except (TypeError, ValueError, ZeroDivisionError):
                raise DatasetError('invalid_input', 'Invalid source time base metadata', {'source_row': row}) from None
            require(supplied_clock == expected_clock, 'Observation metadata time base differs from its source',
                    source_row=row)
    if 'shot_id' in record:
        agrees('shot_id', registry_record('shot_index', shots, 'shot_id')['id'])
    if 'noise_group_id' in record:
        index = int(value('noise_group_index', 'noise_group_id'))
        require(index >= -1, 'Invalid current noise group index', source_row=row)
        if index == -1:
            expected = None
        else:
            require(noise_groups is not None and index < len(noise_groups),
                    'Observation metadata lacks its current noise group', source_row=row)
            expected = noise_groups[index]['id']
        agrees('noise_group_id', expected)
    raw_active = value('active', 'active')
    require(isinstance(raw_active, (int, np.integer)) and not isinstance(raw_active, (bool, np.bool_))
            and raw_active in (0, 1), 'Current observation activity must be integer zero or one', source_row=row)
    active = raw_active == 1
    for field in ('active', 'active_for_fit'):
        if field in record:
            require(type(record[field]) is bool, 'Observation activity metadata must be boolean',
                    source_row=row, field=field)
            agrees(field, active)
    if active:
        if 'origin' in record:
            require(record['origin'] in ('human', 'automatic', 'legacy_import'),
                    'Synthetic or unknown-origin observation cannot enter fitting', source_row=row)
        if 'source_evidence' in record:
            require(record['source_evidence'] == 'measured',
                    'Mathematical or unavailable evidence cannot enter fitting', source_row=row)
        if 'quality' in record:
            legacy = record['quality'] == 'existing_confirmed_source' and record.get('origin') == 'legacy_import'
            require(record['quality'] in ('human_confirmed', 'machine_verified') or legacy,
                    'Pending or rejected observation cannot enter fitting', source_row=row)


def iter_current_observations(path, features, feature_indices, *, mask=None, print_bindings=None,
                              arrays=None, sources=None, shots=None, noise_groups=None):
    """Stream current row semantics and preserve the exact selected JSON lines.

    This checks declarations against current numeric bindings. It does not
    inspect provenance files, source frames, historical receipts or producers.
    """
    from ..domain.measurement import validate_observation_binding
    count = 0
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row, line in enumerate(stream):
            require(row < len(feature_indices), 'Observation metadata has extra rows')
            count += 1
            if mask is not None and not mask[row]:
                continue
            index = int(feature_indices[row])
            require(0 <= index < len(features), 'Row feature index is outside current registry', source_row=row)
            binding = (print_bindings or {}).get(features[index]['id'])
            record = json.loads(line)
            validate_observation_binding(record, features[index], row, print_binding=binding)
            if arrays is not None:
                validate_current_observation_row(record, row, arrays, sources=sources,
                                                 shots=shots, noise_groups=noise_groups)
            yield line
    require(count == len(feature_indices), 'Observation metadata row count differs from arrays')



def verify_prepared_dependencies(root, manifest, *, purpose):
    """Bind a current prepared revision to the bytes its consumers will read.

    Only current snapshot dependencies are inspected. Provenance links to
    historical production, original media and retired states are not followed.
    Older structural fixtures without content identities retain their format.
    """
    expected = manifest.get('dependency_revision')
    if expected is None:
        return
    require(isinstance(expected, dict), 'Malformed prepared dependency identity')
    documents = {
        'scope': manifest.get('scope_ref'), 'rigs': 'rigs.json', 'shots': 'shots.json',
        'features': manifest.get('features_ref'), 'provenance': manifest.get('provenance_ref'),
        'observations': manifest.get('observations_ref'), 'queries': manifest.get('query_constraints_ref'),
        'parameter_layout': manifest.get('parameter_layout_ref'),
        'factor_plan': manifest.get('factor_plan_ref'),
        'parameter_dependencies': manifest.get('parameter_dependencies_ref'),
    }
    inspection_differences = []
    for name, reference in documents.items():
        if expected.get(name) is None:
            continue
        matches = reference is not None and digest(relative(root, reference)) == expected[name]
        # Older prepare wrote the original scope file's hash but serialized its
        # snapshot again. Whitespace/BOM differences are compatible only if the
        # full scope semantics still reconstruct the saved revision below; do
        # not read the original producer or replace its saved dependency hash.
        if name == 'scope' and not matches:
            continue
        # An explicit review may inspect conflicting source metadata. This does
        # not permit changed numeric inputs/cameras, or authorize those rows for
        # fitting; surface the sidecar mismatch in the returned receipt.
        if name == 'observations' and purpose == 'review' and not matches:
            inspection_differences.append(name)
            continue
        require(matches,
                'Prepared current dependency differs from its revision', dependency=name)
    for reference, identity in expected.get('rig_assets', {}).items():
        require(digest(relative(root, reference)) == identity,
                'Prepared current camera dependency differs from its revision', dependency=reference)
    if 'scene' in expected:
        from ..domain.scene_assets import scene_content_identity
        path = relative(root, manifest['scene_ref'])
        require(scene_content_identity(read_json(path), path) == expected['scene'],
                'Prepared current scene differs from its revision')
    # The embedded registries are what numerical consumers use. Matching a
    # sidecar hash alone would not detect an edited in-manifest registry.
    for table in ('rigs', 'shots', 'features'):
        if expected.get(table) is not None:
            document = read_json(relative(root, documents[table]))
            require(manifest.get(table) == document.get(table, []),
                    'Prepared registry differs from its current snapshot', registry=table)
    if expected.get('initial_parameters') is not None:
        state = read_json(relative(root, manifest['initial_state_ref']))
        desc = state['values_ref']
        require(desc['sha256'] == expected['initial_parameters'],
                'Prepared initial parameter identity changed')
        load_array(root, desc, verify=True)
    # Reconstruct the same content identity written by prepare. Do not accept
    # updated hash descriptors, source clocks or activity under an old revision.
    scope = read_json(relative(root, manifest['scope_ref']))
    require(manifest['required_shot_ids'] == scope['required_shot_ids'],
            'Prepared required shots differ from the current scope snapshot')
    scope_status = ('complete' if set(manifest['selected_shot_ids']) == set(scope['required_shot_ids'])
                    else 'subset')
    require(manifest['scope_status'] == scope_status,
            'Prepared scope status differs from the current scope snapshot')
    identity = hashlib.sha256(json.dumps({
        'input': manifest['input_revision'], 'selected': manifest['selected_shot_ids'],
        'scope': scope, 'dependencies': expected, 'sources': manifest['sources'],
        'shots': manifest['shots'], 'noise_groups': manifest['noise_groups'],
        'model_configuration': manifest.get('model_configuration', {}),
        'arrays': {k: v['sha256'] for k, v in manifest['arrays'].items()},
    }, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == manifest['revision'], 'Prepared manifest differs from its content revision')
    if inspection_differences:
        manifest['review_identity_differences'] = inspection_differences


def load_prepared(prepared_dir, *, purpose="fit", core=None):
    """Current-format structural load; does not rehash historical assets."""
    root = Path(prepared_dir).resolve()
    manifest = read_json(root / "manifest.json")
    require(manifest.get("schema") == "rinkspace.prepared.v1", "Unknown prepared format")
    verify_prepared_dependencies(root, manifest, purpose=purpose)
    # These are the current prepared inputs, not historical producers. Reading
    # altered bytes under an unchanged revision would invalidate every later
    # state/prepared identity check, including read-only source overlays.
    arrays = {key: load_array(root, desc, verify=True) for key, desc in manifest["arrays"].items()}
    for source in manifest.get('sources', []):
        load_array(root, source['pts_ref'], verify=True)
    require(purpose in {"fit", "review", "camera_navigation"}, "Unknown prepared consumer purpose")
    allowed = manifest.get('model_configuration', {}).get('allowed_use')
    if allowed is not None and purpose != 'review':
        require(purpose in allowed or (purpose == 'fit' and 'geometry_fit' in allowed),
                'Prepared engineering weights do not authorize this consumer purpose')
    if purpose in ("fit", "camera_navigation"):
        from .prepare import validate_current_feature_contracts
        if 'noise_groups' in manifest:
            require('noise_group_members' in arrays and 'noise_group_weights' in arrays,
                    'Current correlation groups require their numerical arrays')
            validate_current_noise_groups(manifest['noise_groups'], arrays['noise_group_members'],
                                          arrays['noise_group_weights'], arrays)
        scene = read_json(relative(root, manifest["scene_ref"]))
        from ..domain.print_assets import resolve_feature_print_bindings
        from ..domain.scene_assets import read_scene_assets
        active_indices = np.unique(arrays['feature_index'][arrays['active'] != 0])
        active_ids = {manifest['features'][int(i)]['id'] for i in active_indices}
        print_bindings = resolve_feature_print_bindings(manifest['features'],
            read_scene_assets(scene, relative(root, manifest['scene_ref'])), active_feature_ids=active_ids)
        validate_current_feature_contracts(manifest, scene, manifest["features"], arrays, np.arange(len(arrays["pts"])),
                                          print_bindings=print_bindings, core=core)
        if manifest.get('initial_state_ref'):
            from ..domain.parameters import load_parameter_state
            parameter_state = load_parameter_state(relative(root, manifest['initial_state_ref']),
                                 expected_scene=scene, expected_scene_path=relative(root, manifest['scene_ref']))
            if manifest.get('factor_plan_ref'):
                from ..domain.parameters import SceneParameterModel, FactorPlan, LOCAL_LAYOUT_SCHEMA
                layout = parameter_state.layout
                charts = ({b['id']:b['local_chart'] for b in layout['blocks'] if b.get('local_chart') is not None}
                          if layout['schema'] == LOCAL_LAYOUT_SCHEMA else None)
                model = SceneParameterModel(scene, relative(root, manifest['scene_ref']), local_charts=charts)
                dependencies = {f['id']:model.dependencies(f.get('path_id', f['id']))
                    for f in manifest['features'] if f.get('print_binding') is not None}
                expected = FactorPlan(parameter_state, dependencies,
                    scope='prepared_print_geometry_dependencies_not_complete_camera_factors').to_dict()
                require(read_json(relative(root, manifest['factor_plan_ref'])) == expected,
                        'Prepared factor plan differs from the common parameter layout')
        elif any(t.get('internal_geometry') is not None for _,a in read_scene_assets(
                 scene, relative(root, manifest['scene_ref'])) for t in a['templates']):
            raise DatasetError('invalid_input', 'Current parameterized scene requires its prepared initial State')
        if manifest.get('observations_ref'):
            for _ in iter_current_observations(relative(root, manifest['observations_ref']),
                                               manifest['features'], arrays['feature_index'], print_bindings=print_bindings,
                                               arrays=arrays, sources=manifest.get('sources'), shots=manifest.get('shots'),
                                               noise_groups=manifest.get('noise_groups')):
                pass
    return manifest, arrays
