"""Light original225 review from actual material owner and canonical cameras.

No programme graph, actors, source admission, geometry fitting or UV recovery.
The scene JSON alone cannot reproduce original finite source material support.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from calibration.solver_runtime.event_factories import owned_source_path
from collections import Counter
from pathlib import Path
import hashlib,json,pickle,importlib,importlib.util,sys
import numpy as np

BASE=_runtime_data_root()
PACKET=BASE/'shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
PACKET_SHA256='ca5179e55bb1fd92b48e232ab1367ba583e41b1181b9849fa92d3df795942f78'
SOURCE_SHA256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
EVALUATOR_FIXTURE=BASE/'real_absolute_fixture.pkl'
EVALUATOR_FIXTURE_SHA256='7ce8916dad783924fc7e8073573d06088d98838326e8129843aafecc08f08279'
MAIN9_ADAPTER=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/ice_alignment_review/main9_typed_source_adapter.py')
MAIN9_ADAPTER_SHA256='457d3adab1c3b7c282d22dcd908259a56256d80e22c80b2c21628f84d7c32b39'
SCHEMA='rinkspace.original225-native-source-review.v1'
OBSERVATION_FIELDS=('uv','sigma','normal','normal_interval','point_support_bounds_xy_px',
    'id','pts','split','point_correspondence','curve_correspondence','curve_observation_kind',
    'must_preserve_upper_branch','scanline_fixed_axis','scanline_branch','extent_side',
    'clip_boundary_x','clip_blur_allowance_px','source_uv_is_image_boundary_context_only')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numeric_sha(value):
    a=np.asarray(value)
    if a.dtype.hasobject:raise TypeError('Numeric source array required')
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()

def observation_signature(row):
    signature={}
    for k in OBSERVATION_FIELDS:
        if k not in row:signature[k]={'present':False};continue
        v=row[k]
        if isinstance(v,(list,tuple,np.ndarray)):
            signature[k]={'present':True,'numeric_sha256':numeric_sha(v)}
        else:signature[k]={'present':True,'value':v.item()if isinstance(v,np.generic)else v}
    return signature

def load_original_review_rows(packet=PACKET,*,expected_sha256=PACKET_SHA256):
    packet=Path(packet)
    if sha(packet)!=expected_sha256:raise ValueError('Immutable actual source packet SHA mismatch')
    obj=pickle.loads(packet.read_bytes())
    rows=[x for x in obj['programme_absolute_rows']if x['collection_role']=='HELD']
    counts=(len(rows),sum(len(x['source_row']['uv'])for x in rows),
        len({int(x['group_actual_pts'])for x in rows}))
    if counts!=(225,3495,81):raise ValueError('Original225 source inventory changed: '+str(counts))
    return rows

def load_original_absolute_evaluators():
    """Resolve actual per-block source operators from the frozen audit fixture.

    That small fixture explicitly records the original callable per block.
    Importing the private typed adapter starts no motion preparation/solve.
    """
    if sha(EVALUATOR_FIXTURE)!=EVALUATOR_FIXTURE_SHA256:
        raise ValueError('Original per-block absolute evaluator fixture changed')
    declarations={}
    for entry in pickle.loads(EVALUATOR_FIXTURE.read_bytes()):
        value=(entry['absolute_evaluator_module'],entry['absolute_evaluator_name'])
        block=entry['block']
        if block in declarations and declarations[block]!=value:
            raise ValueError('Original per-block evaluator identities disagree')
        declarations[block]=value
    expected={'main9':('main9_typed_source_adapter','typed_absolute_value_jac15'),
        'secondary':('z190_secondary_original_ring_source_qualification','qualified_original_ring_absolute_value_jac15'),
        **{b:('z190_source_camera_diagnostic','absolute_value_jac15')for b in('main1','main7','main4','low')}}
    if declarations!=expected:raise ValueError('Original six-block evaluator identities changed')
    calls={};authorities={}
    for block,(name,attribute)in declarations.items():
        if name=='main9_typed_source_adapter':
            adapter=owned_source_path(name)
            if sha(adapter)!=MAIN9_ADAPTER_SHA256:raise ValueError('Original main9 typed adapter changed')
            if name not in sys.modules:
                spec=importlib.util.spec_from_file_location(name,adapter)
                module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        module=importlib.import_module(name);calls[block]=getattr(module,attribute)
        if not callable(calls[block]):raise TypeError('Original absolute source evaluator must be callable')
        authorities[block]=dict(module=name,callable=attribute,path=str(module.__file__),sha256=sha(module.__file__))
    return calls,authorities

def original_pixel_error_report(camera,row,block,source_errors):
    """Display the matching original typed residual, not a generic substitute."""
    if(block=='main9'and row.get('id')=='C'and row.get('scanline_fixed_axis')=='y'
            and row.get('scanline_branch')=='unique_x'):
        module=importlib.import_module('main9_typed_source_adapter')
        raw=module.horizontal_unique_x_value_jac15(camera,row,False).reshape(-1,2)
        d=np.linalg.norm(raw,axis=1)
        return dict(pts=row['pts'],id=row['id'],split=row.get('split'),points=len(d),
            observation_kind='original_source_horizontal_scanline_unique_x',
            median_px=float(np.median(d)),p95_px=float(np.quantile(d,.95)),maximum_px=float(d.max()),
            source_y_role='Original source y selects scanline; y is not fitted',
            generic_nearest_curve_substitution_used=False)
    if(block=='secondary'and(row.get('curve_observation_kind')=='confirmed_lower_long_side_top_horizontal_support'
            or row.get('original_outer_source_qualification_operator'))):
        return dict(pts=row['pts'],id=row['id'],split=row.get('split'),points=len(row['uv']),
            observation_kind='original_per_block_typed_source_supported_operator',
            source_supported_pixel_metrics_in_parent_record=True,raw_generic_curve_distance_not_reported=True)
    display=dict(row)
    if 'split'not in display:display['split']=None
    return source_errors(camera,[display])[0]

def inventory(rows):
    splits=Counter(('absent'if 'split'not in x['source_row']else str(x['source_row']['split']))for x in rows)
    aliases=[dict(id=x['source_row']['id'],source_row_pts=int(x['source_row']['pts']),
        actual_pts=int(x['group_actual_pts']))for x in rows if int(x['source_row']['pts'])!=int(x['group_actual_pts'])]
    return dict(schema=SCHEMA,source_packet_path=str(PACKET),source_packet_sha256=PACKET_SHA256,
        source_rows=len(rows),source_UV_context_count=sum(len(x['source_row']['uv'])for x in rows),
        actual_exposures=len({int(x['group_actual_pts'])for x in rows}),original_split_counts=dict(splits),
        existing_source_exposure_aliases=aliases,
        source_context_UV_are_not_all_independent_point_correspondences=True,
        original225_sources_promoted_to_FIT=False,full_source_graph_or_actor_built=False,
        current_scene_JSON_alone_sufficient_for_original_finite_source_operator=False,
        required_inputs=['actual registered PairedLocationProgrammeGeometry owner',
            'same-state complete logical world (historical world66 prefix retained)',
            'same-state canonical physical camera NPZ'])

def evaluate_original_review_rows(binder,world66,camera_npz,*,rows=None,complete_q_sha256=None,absolute_evaluators=None):
    """Use the exact actual owner; all exposure aliases are explicit, no fill.

    Pending identity and previous development observations keep their status.
    Missing split fields get None only in the old display function's local
    record, and remain explicitly absent in this report and immutable packet.
    """
    from .paired_location_programme_geometry import PairedLocationProgrammeGeometry
    from z190_physical_camera import PhysicalCameraModel
    from z190_source_camera_diagnostic import source_errors
    from reference_board_pair import BoardIntersectionError
    if type(binder)is not PairedLocationProgrammeGeometry:
        raise TypeError('The actual registered programme material binder is required')
    world=np.asarray(world66,dtype=float)
    ng=int(binder.parameter_count)
    if world.shape!=(ng,)or not np.isfinite(world).all():raise ValueError('Same-state complete declared physical world required')
    binder.validate_parameters(world)
    rows=load_original_review_rows()if rows is None else rows
    if(len(rows),sum(len(x['source_row']['uv'])for x in rows))!=(225,3495):
        raise ValueError('All original225 rows must retain original UV support')
    with np.load(camera_npz,allow_pickle=False)as a:meta=json.loads(str(a['metadata_json']))
    if meta.get('source_sha256')!=SOURCE_SHA256 or meta.get('source_time_base')!='1/15360':
        raise ValueError('Wrong source video or native camera clock')
    model,pts,aliases,views=PhysicalCameraModel.load_npz(camera_npz)
    calls,authorities=load_original_absolute_evaluators()
    if absolute_evaluators is not None:
        if set(absolute_evaluators)!=set(calls):raise ValueError('The complete six original source block evaluators are required')
        for block,call in absolute_evaluators.items():
            if call is not calls[block]:raise ValueError('Actual source evaluator differs from original certified callable: '+block)
        calls=absolute_evaluators
    cameras={};result=[]
    for item in rows:
        original=item['source_row'];actual_pts=int(item['group_actual_pts'])
        original_signature=observation_signature(original)
        record=dict(block=item['block'],group_order=int(item['group_order']),row_order=int(item['row_order']),
            collection_role=item['collection_role'],id=original['id'],source_row_pts=int(original['pts']),
            source_native_pts=original.get('source_native_pts'),actual_pts=actual_pts,
            original_split_present='split'in original,original_source_split=original.get('split'),
            UV_count=len(original['uv']),observation_signature=original_signature,
            original_operator_attributes={k:original.get(k)for k in('curve_observation_kind',
                'scanline_fixed_axis','scanline_branch','must_preserve_upper_branch',
                'point_correspondence','extent_side')},
            original_source_authority=original.get('source_branch_operator_authority',original.get('origin')))
        # These calculations use the original operator and sigma, never a new
        # generic circle/line or an assumed model-to-image correspondence.
        phase='canonical_camera'
        try:
            if actual_pts not in aliases:raise ValueError('No explicit canonical camera at actual PTS '+str(actual_pts))
            representative=aliases[actual_pts]
            record['canonical_representative_pts']=int(representative)
            if representative not in cameras:cameras[representative]=model.camera25(views[representative])
            camera=cameras[representative]
            phase='current_material_binding'
            # The lightweight provider starts with the active producer rows.
            # Register this immutable HELD source's existing physical identity
            # before rebinding; it remains HELD and adds no solver observation.
            binder.register_row(original,actual_pts=actual_pts,
                allow_previously_qualified_held_ad2_representation=item['collection_role']=='HELD')
            rebound=binder.rebind(original,world)
            if observation_signature(rebound)!=original_signature:
                raise ValueError('Rebinding modified immutable source observation '+str(original.get('id')))
            binding=binder._binding(original)[1];identity=binding['contract']
            record.update(physical_identity=identity.get('physical_identity'),material_kind=binding['kind'])
            phase='original_source_operator'
            r=calls[item['block']](camera,[rebound],False)
            if r.shape!=(2*len(original['uv']),)or not np.isfinite(r).all():
                raise ValueError('Original source operator returned invalid residuals')
            pixels=(r*np.repeat(np.asarray(original['sigma']),2)).reshape(-1,2)
            distances=np.linalg.norm(pixels,axis=1)
            record.update(normalized_operator_rms=float(np.sqrt(np.mean(r*r))),
                original_operator_outside_support_median_px=float(np.median(distances)),
                original_operator_outside_support_p95_px=float(np.quantile(distances,.95)),
                original_operator_outside_support_maximum_px=float(distances.max()))
            phase='original_pixel_error_display'
            record['original_pixel_error_report']=original_pixel_error_report(camera,rebound,item['block'],source_errors)
            record['evaluation_status']='evaluated'
        except BoardIntersectionError as error:
            record.update(evaluation_status='invalid_projection',invalid_projection=True,
                error_phase=phase,error_type=type(error).__name__,reason=getattr(error,'reason',str(error)))
        except (ValueError,TypeError,KeyError,IndexError,FloatingPointError)as error:
            unsupported=(phase=='original_source_operator'and
                str(error)=='Expected explicitly source-identified vertical scanline branch')
            record.update(evaluation_status='unsupported_original_typed_operator'if unsupported else'row_evaluation_failed',
                evaluation_failed=True,error_phase=phase,error_type=type(error).__name__,
                original_operator_error=str(error),generic_operator_fallback_used=False)
        if observation_signature(original)!=original_signature:
            record.update(evaluation_status='immutable_source_guard_failed',evaluation_failed=True,
                error_phase='immutable_source_guard',error_type='ValueError',
                original_operator_error='Immutable original source row was modified')
        result.append(record)
    return dict(**inventory(rows),camera_npz_path=str(camera_npz),camera_npz_sha256=sha(camera_npz),
        complete_q_sha256=complete_q_sha256,world66_sha256=numeric_sha(world[:66]),
        world_parameter_count=ng,world_parameters_sha256=numeric_sha(world),
        historical_world66_summary_is_prefix=True,rows=result,
        original_absolute_evaluator_fixture_sha256=EVALUATOR_FIXTURE_SHA256,
        original_absolute_evaluators_by_block=authorities,
        evaluation_status_counts=dict(Counter(r['evaluation_status']for r in result)),
        complete_numeric_source_evaluation=all(r['evaluation_status']=='evaluated'for r in result),
        accuracy_accepted=False,
        evidence_policy='Original split retained; pending/development groups are not independent validation. F1 added-held lies in an already-fitted exposure.')

def compare_original_review_reports(before,after):
    """Compare identical source instances; never pool pending and held splits."""
    if before['source_packet_sha256']!=after['source_packet_sha256']:
        raise ValueError('Before/after immutable source packets differ')
    key=lambda r:(r['block'],r['group_order'],r['row_order'],r['actual_pts'],r['id'])
    old={key(r):r for r in before['rows']};new={key(r):r for r in after['rows']}
    if len(old)!=225 or old.keys()!=new.keys():raise ValueError('Original source instances changed')
    changes=[]
    for k,a in old.items():
        b=new[k]
        for field in('observation_signature','original_split_present','original_source_split','UV_count'):
            if a[field]!=b[field]:raise ValueError('Before/after source identity/support differs: '+str(k))
        d=dict(id=a['id'],actual_pts=a['actual_pts'],original_source_split=a['original_source_split'],
            original_split_present=a['original_split_present'],block=a['block'],
            before_status=a.get('evaluation_status','evaluated'),after_status=b.get('evaluation_status','evaluated'),
            before_invalid=bool(a.get('invalid_projection')),after_invalid=bool(b.get('invalid_projection')),
            before_evaluation_failed=bool(a.get('evaluation_failed')),after_evaluation_failed=bool(b.get('evaluation_failed')))
        if not any(d[f]for f in('before_invalid','after_invalid','before_evaluation_failed','after_evaluation_failed')):
            d.update(before_median_px=a['original_operator_outside_support_median_px'],
                after_median_px=b['original_operator_outside_support_median_px'],
                delta_median_px=b['original_operator_outside_support_median_px']-a['original_operator_outside_support_median_px'])
        else:d.update(before_error=a.get('original_operator_error',a.get('reason')),
            after_error=b.get('original_operator_error',b.get('reason')))
        changes.append(d)
    return dict(schema=SCHEMA+'.comparison',source_rows=225,source_packet_sha256=before['source_packet_sha256'],
        before_camera_sha256=before['camera_npz_sha256'],after_camera_sha256=after['camera_npz_sha256'],
        before_complete_q_sha256=before.get('complete_q_sha256'),after_complete_q_sha256=after.get('complete_q_sha256'),
        rows=changes,accuracy_accepted=False)
