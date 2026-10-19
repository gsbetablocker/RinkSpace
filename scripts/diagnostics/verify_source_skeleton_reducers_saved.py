"""One existing prepared-object read plus light original reducer checks."""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import pickle
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np

from calibration.solver_runtime import install
from calibration.solver_runtime import source_state_skeleton_cache_runtime as skeleton
from calibration.solver_runtime.source_assembly_gc_runtime import defer_automatic_source_gc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opt = parser.parse_args()
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    install_mappingproxy_pickle(); install_geometry_unpickle()
    from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import PairedLocationsSharedSizesGeometry
    from benchmark_harness import N, restore_local_priors
    import z190_joint_shared_F_height_source as shared
    module = shared.module_at(N/'ice_alignment_review/main9_full_source_factory.py')
    sys.modules[module.__name__] = module

    original = PairedLocationsSharedSizesGeometry.__new__(PairedLocationsSharedSizesGeometry)
    values = np.arange(65537., dtype=float).reshape(-1,1)
    original.__dict__.update(material=values, source_alias=values,
        _derived_course_center_cache={'current':(values,values)},
        _scene_cache={'current':values}, _value_cache={'current':values})
    root = SimpleNamespace(q=np.asarray([1.]), geometry=original)
    output = io.BytesIO()
    skeleton._OriginalSkeletonPickler(output, root).dump(root)
    restored = pickle.loads(output.getvalue())
    assert restored.q is None and type(restored.geometry) is PairedLocationsSharedSizesGeometry
    assert restored.geometry._derived_course_center_cache == restored.geometry._scene_cache == restored.geometry._value_cache == {}
    assert restored.geometry.material is restored.geometry.source_alias
    assert np.array_equal(restored.geometry.material, values)
    assert original._derived_course_center_cache['current'][0] is values

    with defer_automatic_source_gc('saved_prepared_reducer_shape_check'):
        with opt.prepared.open('rb') as stream:
            state = pickle.load(stream)
        restore_local_priors(state)
        pending = [(state,'state')]
        seen, kinds, eligible = set(), Counter(), Counter()
        errors, unsupported_lambdas, known_prior_lambdas = [], [], 0
        probe = skeleton._OriginalSkeletonPickler(io.BytesIO(), state)
        while pending:
            value, path = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, (dict, MappingProxyType)):
                pending.extend((item,path+'.'+str(key)) for key,item in value.items())
            elif isinstance(value, (tuple,list,set,frozenset)):
                pending.extend((item,path+'['+str(i)+']') for i,item in enumerate(value))
            elif isinstance(value, np.ndarray):
                continue
            elif hasattr(value, '__dict__') and not isinstance(value,type) and not callable(value):
                name = type(value).__module__+'.'+type(value).__qualname__
                kinds[name] += 1
                own = vars(value)
                prior = own.get('priors')
                if callable(prior) and getattr(prior,'__name__',None) == '<lambda>':
                    if own.get('_isolated_cache_heightless_prior') is True:
                        known_prior_lambdas += 1
                    else:
                        unsupported_lambdas.append(path+'.priors')
                try:
                    reduced = probe.reducer_override(value)
                    if reduced is not NotImplemented:
                        eligible[name+':'+str(len(reduced))] += 1
                except Exception as exc:
                    errors.append(dict(path=path, error=type(exc).__name__+': '+str(exc)))
                pending.extend((item,path+'.'+str(key)) for key,item in own.items()
                    if key != 'priors' or own.get('_isolated_cache_heightless_prior') is not True)
            elif callable(value) and getattr(value,'__name__',None) == '<lambda>':
                unsupported_lambdas.append(path)
    report = dict(prepared=str(opt.prepared.resolve()), prepared_objects_loaded_once=True,
        paired_original_two_item_dict_restore_roundtrip_pass=True,
        original_large_array_bytes_and_aliases_preserved=True,
        current_derived_caches_cleared_without_live_mutation=True,
        visited_objects=len(seen), object_types=dict(kinds), eligible_reducers=dict(eligible),
        original_known_prior_lambdas=known_prior_lambdas,
        unsupported_lambdas=unsupported_lambdas, reducer_errors=errors,
        no_programme_graph_construction_source_evaluation_or_solver=True,
        all_checked_pass=not errors and not unsupported_lambdas,
        helper_sha256=skeleton._sha(skeleton.__file__))
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(opt.output), all_pass=report['all_checked_pass'],
        visited_objects=len(seen), known_priors=known_prior_lambdas,
        reducer_errors=errors, unsupported_lambdas=unsupported_lambdas)))


if __name__ == '__main__':
    main()
