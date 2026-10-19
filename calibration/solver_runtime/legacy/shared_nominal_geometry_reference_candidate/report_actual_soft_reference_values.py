"""Strict actual-value report from one whole q/profile checkpoint; no graph.

Initial mode authenticates the inherited event then extends it with the exact
declared zero K20 chart. Latest mode requires matching whole-q/profile tokens
in actual programme, wide and scene reports before recording any values.
"""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,io,json,sys,time
import numpy as np

BASE=_runtime_data_root()
sys.path[:0]=[str(_runtime_source_root),str(_runtime_program_source_root)]
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import CorrectedSharedRinkNumericGeometryLanesAdPairs
from shared_ice_hockey_geometry_candidate.shared_ice_hockey_numeric_geometry import SharedIceHockeyNumericGeometry
from shared_ice_hockey_geometry_candidate.hockey_source_chart import HockeyCoordinateChart
from shared_nominal_geometry_reference_candidate.soft_nominal_geometry_source import EngineeringReferenceConfig,SoftNominalGeometryFactor
from z190_joint_source_solver import _frozen_profile_token


EXPECTED_SCHEMA='rinkspace.complete-original-source-shared-hockey20-soft-reference18-guarded-wide25.v3'
REPORT_NAMES=('latest_actual_source_review.json','latest_actual_wide_source_review.json','latest_candidate_scene_UNACCEPTED.json')


class CheckpointProfiles:
    """Preserve checkpoint's original profile key ordering, dtype and bytes."""
    def __init__(self,z):
        self.z=z;self.keys=[k for k in z.files if k.startswith('profile__')]
        if len(self.keys)!=18:raise ValueError('All eighteen original profile arrays required')
    def iter_checkpoint_arrays(self):
        for key in self.keys:yield key[9:],self.z[key]


def sha(data):return hashlib.sha256(data).hexdigest()
def qtoken(q):return sha(np.ascontiguousarray(q).view(np.uint8))
def jsonbytes(data):return json.loads(data.decode('utf-8'))


def check_event(payload,expected,label):
    supplied=payload.get('source_event_state_contract',{})
    mismatch={k:dict(expected=v,actual=supplied.get(k))for k,v in expected.items()if supplied.get(k)!=v}
    if mismatch:raise ValueError('Whole-q/profile event mismatch in '+label+': '+json.dumps(mismatch))


def build_report(run,mode='latest',*,checkpoint=None,expected_manifest_schema=EXPECTED_SCHEMA):
    started=time.perf_counter();run=Path(run).resolve()
    if not run.is_relative_to(BASE.resolve()):raise ValueError('Own-task candidate path required')
    manifest_path=run/'input_manifest.json';manifest_bytes=manifest_path.read_bytes();manifest=jsonbytes(manifest_bytes)
    if manifest.get('schema')!=expected_manifest_schema:raise ValueError('Explicit expected input manifest schema mismatch')
    rc=manifest['soft_reference_contract']
    if rc.get('schema')!='rinkspace.soft-engineering-geometry-relations.v1':raise ValueError('Exact engineering reference contract schema required')
    config=EngineeringReferenceConfig(**rc['reference_configuration'])
    chartdata=manifest['current_coordinate_chart']
    chart=HockeyCoordinateChart(int(chartdata['new_geometry_parameter_count']))
    if chart.manifest()!=chartdata:raise ValueError('Exact authenticated shared hockey coordinate chart required')
    if manifest['world_parameter_indices']!=list(chart.world_indices):raise ValueError('Manifest whole world indices mismatch')
    if rc['parameter_count']!=chart.full_count or rc['world_parameter_indices']!=list(chart.world_indices):
        raise ValueError('Reference contract actual full-count/world ownership mismatch')
    # Validate this module's actual runtime against the running candidate's
    # locked input authority, without changing those running bytes.
    runtime=Path(__file__).parent/'soft_nominal_geometry_source.py'
    refs=[r for r in manifest['imported_new_runtime_source_files']if Path(r['path']).resolve()==runtime.resolve()]
    if len(refs)!=1 or refs[0]['sha256']!=sha(runtime.read_bytes()):raise ValueError('Locked soft reference runtime mismatch')
    before={};parent_event=None
    if mode=='initial':
        seed=manifest['complete_current_initial_event'];path=Path(seed['path']).resolve()
        if checkpoint is not None and Path(checkpoint).resolve()!=path:raise ValueError('Initial checkpoint must be the exact inherited manifest authority')
        data=path.read_bytes()
        if sha(data)!=seed['sha256']:raise ValueError('Inherited complete initial checkpoint hash mismatch')
        parent_event=seed['source_state_contract']
        report_dir=path.parent
        before={name:(report_dir/name).read_bytes()for name in REPORT_NAMES}
    elif mode=='latest':
        path=(run/'solver/latest_accepted_state.npz')if checkpoint is None else Path(checkpoint).resolve()
        if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Own-task checkpoint required')
        before={name:(run/name).read_bytes()for name in REPORT_NAMES}
        # Read once: a concurrent atomic checkpoint replacement cannot mix q
        # with profiles from a different Zip file after this byte snapshot.
        data=path.read_bytes()
    else:raise ValueError('Mode must be initial or latest')
    with np.load(io.BytesIO(data),allow_pickle=False)as z:
        profiles=CheckpointProfiles(z)
        token=_frozen_profile_token(profiles)
        q=z['q'].copy()
        observed=dict(q_dtype=q.dtype.str,q_shape=list(q.shape),q_sha256=qtoken(q),frozen_profile_sha256=token)
        if mode=='initial':
            if q.shape!=(34253,) or any(parent_event.get(k)!=v for k,v in observed.items()):
                raise ValueError('Actual inherited q/profile differs from manifest event')
            for name,payload in before.items():check_event(jsonbytes(payload),observed,'inherited '+name)
            q=chart.extend_full_q(q)
        else:
            chart.full_q(q)
            for name,payload in before.items():check_event(jsonbytes(payload),observed,name)
            if not np.array_equal(z['aux__current_shared_world_geometry_parameter_indices'],chart.world_indices):
                raise ValueError('Checkpoint actual world ownership mismatch')
            if json.loads(z['aux__current_soft_geometry_reference_contract_json'].item())!=rc:
                raise ValueError('Checkpoint and input reference contracts differ')
        if 'aux__shared_scene_json' not in z.files:raise ValueError('Actual original geometry scene authority required')
        base=json.loads(z['aux__shared_scene_json'].item())
        profile_keys=list(profiles.keys)
        elapsed=float(z['elapsed_seconds']) if 'elapsed_seconds'in z.files else None
    # A moving latest source report is not silently accepted. Caller may
    # retry after the normal checkpoint/review pair has completed.
    directory=path.parent if mode=='initial' else run
    if any((directory/name).read_bytes()!=payload for name,payload in before.items()):
        raise ValueError('Source event reports changed during snapshot; retry a coherent event')
    if manifest_path.read_bytes()!=manifest_bytes:raise ValueError('Input manifest changed during numerical audit')
    geometry=SharedIceHockeyNumericGeometry(CorrectedSharedRinkNumericGeometryLanesAdPairs(base,.31353729994266055),
        include_fr_ratio_deltas=chart.hockey_parameter_count==23)
    factor=SoftNominalGeometryFactor(geometry,chart.world_indices,chart.full_count,config=config)
    if factor.contract!=rc:raise ValueError('Reconstructed actual engineering factor contract differs from input manifest')
    if factor.row_count!=manifest['soft_reference_row_count']:raise ValueError('Actual reference row count differs from running source')
    value=factor.evaluate_world(q[list(chart.world_indices)],True)
    rows=[]
    for metadata,raw,residual,scale,j in zip(value.rows,value.raw,value.residual,value.scales,value.jacobian):
        row=dict(metadata)
        actual=float(raw+row['nominal'])
        if row['id']=='FR_button_minimum_radius':
            actual=factor._circle('F1_center',q[list(chart.world_indices)])[1][0]
        row.update(actual_value=actual,nominal_value=float(row['nominal']),
            raw_error=float(actual-row['nominal']),objective_raw_error=float(raw),
            normalized_error=float(residual),working_sigma=float(scale),
            quadratic_cost=float(.5*residual*residual),active_world_local_columns=np.flatnonzero(j).tolist())
        rows.append(row)
    actual_event=dict(q_dtype=q.dtype.str,q_shape=list(q.shape),q_sha256=qtoken(q),frozen_profile_sha256=token)
    primary=jsonbytes(before['latest_actual_source_review.json'])
    # The original solver event may contain full 34k trial vectors. Preserve
    # it in the authority review and its hash, never duplicate/print it here.
    original_event=primary.get('event') or {}
    event={key:original_event[key]for key in ('iteration','accepted','cost','next_cost','trial',
        'damping','failure_reason','checkpoint_kind','stage','checkpoint','elapsed_seconds')if key in original_event}
    return dict(schema='rinkspace.actual-whole-q-profile-soft-reference-numeric-report.v1',status='PASS',
        mode=mode,input_manifest=dict(path=str(manifest_path),sha256=sha(manifest_bytes),schema=manifest['schema']),
        checkpoint=dict(path=str(path),sha256=sha(data),original_q_shape=observed['q_shape'],elapsed_seconds=elapsed),
        source_event_state_contract=actual_event,inherited_source_event_state_contract=observed if mode=='initial'else None,
        initial_zero_K20_extension_exactly_preserves_original_complete_q=(mode=='initial'),
        copied_whole_q_not_shared_subset=True,all_original_profile_array_keys=profile_keys,
        profile_token_formula='direct original z190_joint_source_solver._frozen_profile_token; v1 key/dtype/shape/rawbytes order',
        reference_contract_exact_input_and_checkpoint_match=True,
        same_event_source_review_tokens_verified=True,source_review_event=event,
        source_review_token_match_scope='inherited complete34253 before exact declared zero extension'if mode=='initial'else'actual current whole-q/profile',
        full_solver_event_not_duplicated=True,
        source_review_files={name:dict(path=str(directory/name),sha256=sha(payload))for name,payload in before.items()},
        working_reference_config=rc['reference_configuration'],world_parameter_indices=list(chart.world_indices),
        actual_world_parameter_values=q[list(chart.world_indices)].tolist(),
        reference_row_count=len(rows),reference_rows=rows,total_engineering_reference_cost=value.report['cost'],
        reference_report=value.report,reference_values_are_not_source_alignment_accuracy=True,
        actual_boundary_bounds_conditional_m=factor.actual_boundary_bounds_m.tolist(),
        rink_60x30_only_nominal_report_reference=True,actual_boundary_60x30_transport_installed=False,
        prepared_graph_loaded=False,source_preparation_or_solver_run=False,source_UV_roles_sigma_profiles_modified=False,
        helper_source_sha256=sha(Path(__file__).read_bytes()),locked_factor_source_sha256=sha(runtime.read_bytes()),
        seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=('initial','latest'),default='latest')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-manifest-schema',default=EXPECTED_SCHEMA)
    args=parser.parse_args()
    report=build_report(args.run,args.mode,checkpoint=args.checkpoint,expected_manifest_schema=args.expected_manifest_schema)
    output=args.output
    if output is None:
        output=_runtime_data_module_dir(__file__)/'actual_numeric_reports'/(
            args.run.name+'_'+args.mode+'_'+report['source_event_state_contract']['q_sha256'][:12]+'_'+
            report['source_event_state_contract']['frozen_profile_sha256'][:12]+'_'+report['checkpoint']['sha256'][:8]+'_'+
            report['helper_source_sha256'][:8]+'.json')
    output=output.resolve()
    if not output.is_relative_to(BASE.resolve()):parser.error('Own-task report output required')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        previous=json.loads(output.read_text(encoding='utf-8'))
        for key in ('source_event_state_contract','reference_rows','checkpoint','input_manifest'):
            if previous[key]!=report[key]:raise ValueError('Existing output is a different actual event; choose a new report path')
        # Preserve the earlier immutable capture of the same event.
    else:output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(output),status=report['status'],mode=args.mode,
        total_reference_cost=report['total_engineering_reference_cost'],reference_rows=report['reference_row_count'],
        q_sha256=report['source_event_state_contract']['q_sha256'],
        profile_sha256=report['source_event_state_contract']['frozen_profile_sha256'],seconds=report['seconds']),indent=2))


if __name__=='__main__':main()
