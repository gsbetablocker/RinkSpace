"""Refit a complete rig group after a frozen source-only surface review.

The camera model and prior targets are the original v2 definitions. Saved v2
states supply an initial estimate only. Source UV stays preserved in the source
packets; the separate review adapter records every excluded/unknown row.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time

import fit_continuous_rig_camera as base
import numpy as np
from continuous_camera_solver import solve, write_json


MODEL_FILES = ('continuous_rig_camera.py', 'continuous_camera_model.py',
               'continuous_brown_adapter.py', 'continuous_brown_board_pair.py',
               'projection.py', 'native_board_pair.py', 'reference_board_pair.py',
               'surface_pair.py', 'continuous_exposure_clock.py')


def _projection_additions_only(directory, expected_hash):
    """Accept only a reviewed, hash-verified projection source transition."""
    candidates=(directory/'code_at_start'/'projection.py',
        directory/'code_dependencies'/'calibration'/'program_tracking'/'projection.py')
    saved=next((p for p in candidates if p.is_file() and base.sha(p)==expected_hash),None)
    if saved is None:
        raise ValueError('No hash-verified old projection source for warm compatibility')
    from solver_acceleration_compatibility import reviewed_source_change
    reviewed = reviewed_source_change('projection.py', saved,
        Path(__file__).with_name('projection.py'), expected_hash)
    if reviewed is not None:
        return reviewed
    raise ValueError('Projection source transition has not been reviewed')


def warm_states(directory, model, initial, data, aliases, *, state_filename='states.npz',
                reference_metadata=None, allow_projection_additions=False):
    if state_filename not in ('states.npz', 'accepted_states.npz'):
        raise ValueError('Warm state must be a completed or atomically saved solver payload')
    phase=json.loads((directory/'phase_00.json').read_text(encoding='utf8'))
    if (phase['schema'] not in ('rinkspace.shared-mechanical-centre-brown-camera.v1',
                              'rinkspace.shared-rig-source-surface-review.v1',
                              'rinkspace.shared-rig-original-source-calibration.v1')
            or phase['camera_id']!=data['camera_id']
            or phase['source_sha256']!=data['diag']['source_sha256']
            or phase['model_sha256']!=data['diag']['model_sha256']):
        raise ValueError('Warm model/source/camera identity differs')
    compatibility={}
    for name in MODEL_FILES:
        if phase['code_sha256'][name]!=base.sha(Path(__file__).with_name(name)):
            from solver_acceleration_compatibility import reviewed_source_change
            candidates=(directory/'code_at_start'/name,
                directory/'code_dependencies'/'calibration'/'program_tracking'/name)
            saved=next((p for p in candidates if p.is_file()
                        and base.sha(p)==phase['code_sha256'][name]),None)
            reviewed=(reviewed_source_change(name,saved,Path(__file__).with_name(name),
                       phase['code_sha256'][name]) if saved is not None else None)
            if reviewed is not None:
                compatibility[name]=reviewed
                continue
            if name=='projection.py' and allow_projection_additions:
                compatibility[name]=_projection_additions_only(directory,phase['code_sha256'][name])
                continue
            raise ValueError('Warm physical-state definition changed: '+name)
    prior_directory=directory
    if phase['schema']!='rinkspace.shared-mechanical-centre-brown-camera.v1':
        # The reviewed run exported its WARM start as initial_cameras. Its
        # unchanged prior targets were verified against the original v2 run.
        provenance=phase['warm_estimate']
        parent_directory=Path(provenance['directory'])
        if base.sha(parent_directory/'phase_00.json')!=provenance['phase_sha256']:
            raise ValueError('Original v2 prior provenance changed')
        prior_directory=Path(provenance.get('original_prior_directory',parent_directory))
        parent=json.loads((prior_directory/'phase_00.json').read_text(encoding='utf8'))
        if (parent['schema']!='rinkspace.shared-mechanical-centre-brown-camera.v1'
                or any(parent[k]!=phase[k] for k in ('camera_id','source_sha256','model_sha256'))):
            raise ValueError('Reviewed warm estimate has no original v2 prior source')
    with np.load(prior_directory/'initial_cameras.npz',allow_pickle=False) as z:
        pts=np.asarray(sorted(model.frames),np.int64)
        if not np.array_equal(z['pts'],pts) or not np.array_equal(
                z['exposure_representative_pts'],[aliases[int(p)] for p in pts]):
            raise ValueError('Warm exposure identity differs')
        rebuilt=np.asarray([model.camera25(int(p),initial) for p in pts])
        initialization_difference=float(np.max(np.abs(rebuilt-z['camera25'])))
        if initialization_difference>1e-9:
            raise ValueError('Original v2 prior initializer was not reproduced')
    payload=(directory/state_filename).read_bytes()
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        if z['states'].shape!=(len(initial),11) or not np.isfinite(z['states']).all():
            raise ValueError('Warm states are not finite Vector11 nodes')
        incoming={int(k):v.copy() for k,v in zip(z['keys'],z['states'])}
    if len(incoming)!=len(initial) or incoming.keys()!=initial.keys():
        raise ValueError('Warm graph nodes differ')
    expected_knots={str(k):dict(keys=list(s.control_keys),pts=list(s.knot_pts))
                    for k,s in model.shot_splines.items()}
    if phase['shot_knots']!=expected_knots:
        raise ValueError('Warm centre spline definition differs')
    if phase['zoom_reference']!=model.lens_zoom_reference:
        raise ValueError('Warm lens zoom reference differs')
    if phase['schema']=='rinkspace.shared-rig-original-source-calibration.v1':
        if reference_metadata is None:
            raise ValueError('Source-calibrated warm state requires reference identity checks')
        for name in ('new_graph_nodes','reference_pts','reference_image_sha256',
                     'reference_sources','source_video_sha256',
                     'current_programme_scene_sha256','source_curve_pixel_count'):
            if phase['reference_calibration'][name]!=reference_metadata[name]:
                raise ValueError('Warm source-reference identity differs: '+name)
    for name in ('mount_keys','offset_keys','offset_slope_keys','brown_keys',
                 'brown_slope_keys','lens_base_keys','lens_slope_keys'):
        if phase[name]!=getattr(model,name):
            raise ValueError('Warm shared model node binding differs: '+name)
    return incoming,dict(directory=str(directory.resolve()),
        original_prior_directory=str(prior_directory.resolve()),
        state_filename=state_filename,
        states_sha256=hashlib.sha256(payload).hexdigest(),
        phase_sha256=base.sha(directory/'phase_00.json'),
        definition_compatibility=compatibility,
        prior_initializer_camera_max_absolute_difference=initialization_difference,
        role='initial_estimate_only; v2 baseline prior targets unchanged')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',required=True)
    parser.add_argument('--surface-review',type=Path,required=True)
    parser.add_argument('--warm-group',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    # Imported only by this opt-in entry; the running original v2 is unchanged.
    import continuous_source_surface_review as review
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(base.TMP.resolve()):
        raise ValueError('Use a fresh programme TMP output directory')
    output.mkdir(parents=True)
    tick=time.perf_counter()
    data,pairs,inputs=base.prepare_inputs(output,args.camera)
    aliases,alias_meta=base.load_exposure_aliases(data)
    clock,clock_meta,valid_pairs=base.build_exposure_clock(data,aliases)
    model,prior_initial,baseline,init_meta=base.make_rig(data,aliases,clock,1.,True)
    # Reproduce the original v2 initializer using the original input packet;
    # a changed first pair must not silently move the first-pose prior target.
    adapted=base.adapt_rig_initializer(model,prior_initial,baseline,data['scene'],
                                       base.extended_native_pairs(pairs))
    data,pairs,review_metadata=review.apply_source_surface_review(
        data,pairs,args.surface_review)
    pairs=base.extended_native_pairs(pairs)
    # Build all original v2 priors BEFORE replacing the starting estimate.
    prior_copy={k:v.copy() for k,v in prior_initial.items()}
    graph,counts=base.build_graph(data,pairs,model,prior_initial,clock,valid_pairs,.25)
    initial,warm_meta=warm_states(args.warm_group,model,prior_initial,data,aliases)
    if any(not np.array_equal(prior_initial[k],prior_copy[k]) for k in prior_copy):
        raise RuntimeError('Warm start changed original v2 prior targets')
    validity=base.preflight(pairs,model,initial)
    write_json(output/'initial_source_geometry.json',validity)
    if validity['invalid_pair_count']:
        raise ValueError('Preserved reviewed source pixels have invalid initial board geometry')
    previous=json.loads((args.warm_group/'phase_00.json').read_text(encoding='utf8'))
    phase={**previous,'schema':'rinkspace.shared-rig-source-surface-review.v1',
        'source_inputs':inputs,'source_surface_review':review_metadata,
        'warm_estimate':warm_meta,'aliases':alias_meta,'exposure_clock':clock_meta,
        'initializer':init_meta,'initializer_adaptation':adapted,
        'initial_geometry':validity,'factor_counts':counts,
        'prior_targets':'original v2 baseline initializer; never moved to warm fitted states',
        'source_uv_changed':False,'full_programme_accepted':False}
    code_names=list(dict.fromkeys([*previous['code_sha256'],
        Path(__file__).name,Path(review.__file__).name]))
    phase['code_sha256']={name:base.sha(Path(__file__).with_name(name)) for name in code_names}
    write_json(output/'phase_00.json',phase)
    snapshot=output/'code_at_start'
    snapshot.mkdir()
    for name in code_names:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    repository=Path(__file__).resolve().parents[2]
    dependencies={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:
            continue
        path=Path(filename).resolve()
        if path.suffix=='.py' and path.is_relative_to(repository) and path.is_file():
            relative=path.relative_to(repository)
            dependencies[str(relative)]=base.sha(path)
            target=output/'code_dependencies'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    write_json(output/'code_dependencies.json',dict(
        scope='loaded project module files captured when graph was ready',
        sha256=dependencies))
    base.export(model,initial,data,aliases,output/'initial_cameras.npz')
    print(json.dumps(dict(stage='model_ready',camera=args.camera,frames=len(model.frames),
        factors=len(graph._pending),seconds=time.perf_counter()-tick)),flush=True)
    if args.preflight_only:
        return
    solved,trace=solve(graph,initial,output,max_iterations=args.iterations)
    report=base.export(model,solved,data,aliases,output/'shot_cameras.npz')
    report.update(camera_id=args.camera,shot_indices=list(data['shot_indices']),
        source_sha256=data['diag']['source_sha256'],model_sha256=data['diag']['model_sha256'],
        phase_sha256=base.sha(output/'phase_00.json'),seconds=time.perf_counter()-tick,
        accepted=False,scope='Complete same-camera group after frozen source-only surface review')
    write_json(output/'summary.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
