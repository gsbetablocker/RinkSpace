"""Only same-PTS absolute source metadata for the authorized 55-exposure audit.

No projection, residual, source/truth J, frame decoding or model update.
"""
from pathlib import Path
import json
import pickle
import sys
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_saved_absolute_target_response_20261011 as a
from calibration.solver_runtime.source_point_qualification import load_source_point_qualification
np=a.np

PTS=[229376000,229582592,229604096,229702144,229953536,229961216,229987328,229988864,
229989632,229990400,229991936,229993472,229995008,229995264,230024192,230043904,
230094080,230112512,230141440,230146304,230148864,230201600,230231552,230254336,
230306816,230314496,230352896,230359040,230360576,230365184,230368512,230372096,
230382336,230390528,230564096,230571776,230579456,230587136,230594816,230602496,
230610176,230648576,230656256,230663936,230710016,230717696,230725376,230733056,
230740736,231678464,229611008,229612544,229614080,230156288,230708224]
PTS=sorted(PTS)
LOW=[229953536,229961216,230306816,230314496,230352896,230360576]
OUT=a.H/'joint55_exposure_existing_absolute_metadata_20261011.json'


def same_source_values(left,right):
    # The compact packet stores source lists as numeric arrays. Compare every
    # original value without confusing JSON integer/float spelling with a
    # changed source measurement.
    if isinstance(left,dict)and isinstance(right,dict):
        return set(left)==set(right)and all(same_source_values(left[k],right[k])for k in left)
    if isinstance(left,(list,tuple,np.ndarray))and isinstance(right,(list,tuple,np.ndarray)):
        return len(left)==len(right)and all(same_source_values(x,y)for x,y in zip(left,right))
    return bool(left==right)


def main():
    options=a.read(a.CONFIG)['source_options'];manifest=a.read(a.RUN/'input_manifest.json')
    directory=a.RUN/'accepted_states'/a.CHILD;identity=a.read(directory/'state_identity.json')
    state_path=directory/'complete_state_UNACCEPTED.npz'
    assert a.sha(state_path)==identity['state_npz_sha256']
    provider,world,proof=a.current_complete_source_geometry(state_path,options['--world-init'])
    assert a.sha(identity['camera_path'])==identity['camera_sha256']
    with np.load(identity['camera_path'],allow_pickle=False)as z:
        aliases=dict(zip(map(int,z['pts']),map(int,z['exposure_representative_pts'])))
    assert a.sha(a.PACKET)==a.PACKET_SHA256
    packet=pickle.loads(a.PACKET.read_bytes());originals=packet['programme_absolute_rows']
    keep={}
    for block in manifest['original_absolute_input_compaction']['blocks']:
        for row in block['row_maps']:
            keep[block['block'],row['group_order'],row['row_order']]=row['kept_original_point_indices']
    corrections={r['original_packet_row_index']:r for r in a.read(options['--original-ice-pixel-correction-source'])['records']}
    rows=[];binding_failures=[]
    for index,item in enumerate(originals):
        at=int(item['group_actual_pts'])
        if at not in PTS:continue
        row=item['source_row'];active=keep.get((item['block'],item['group_order'],item['row_order']),list(range(len(row['uv']))))
        try:
            provider.register_row(row,actual_pts=at,allow_previously_qualified_held_ad2_representation=item['collection_role']=='HELD')
            binding=provider._binding(row)[1]
        except Exception as error:
            binding_failures.append(dict(original_packet_row_index=index,id=row['id'],actual_pts=at,role=item['collection_role'],error=str(error)))
            continue
        rows.append(dict(family='original_absolute',packet_row_index=index,actual_pts=at,native_pts=int(row['pts']),
            block=item['block'],role=item['collection_role'],id=row['id'],source_points=len(row['uv']),active_point_count=len(active),
            active_original_point_indices=active,current_pixel_correction=index in corrections,
            physical_binding={k:binding[k]for k in('kind','name','ids','columns')if k in binding},
            original_operator={k:row[k]for k in('point_correspondence','curve_correspondence','curve_observation_kind',
                'scanline_fixed_axis','scanline_branch','extent_side')if k in row},
            original_normal_interval_present=row.get('normal_interval')is not None))
    source_refs=[]
    declarations=[('--additional-board-source','additional_real_native_board_edges'),
        ('--additional-ice-source','additional_real_native_ice_curves'),
        ('--additional-board-material-point-source','additional_real_native_board_material_points'),
        ('--additional-board-circle-source','additional_real_native_board_circles')]
    for key,family in declarations:
        path=Path(options[key]);data=a.read(path);source_refs.append(a.ref(path))
        qualification=load_source_point_qualification(path,data,source_family='additional_native_ice_curves')if key=='--additional-ice-source'else None
        for index,row in enumerate(data['rows']):
            native=int(row['pts']);at=aliases[native]
            if at not in PTS:continue
            assert int(row.get('actual_pts',at))==at
            active=np.flatnonzero(qualification.point_mask(row))if qualification is not None and row['split']=='fit'else np.arange(len(row['uv']))
            curve=row.get('curve',{});number=curve.get('advertisement_number',curve.get('advertisement_id'))
            logical=(provider.current.material_parameter_indices(number).tolist()if key=='--additional-board-source'and type(number)is int else None)
            rows.append(dict(family=family,packet_row_index=index,actual_pts=at,native_pts=native,
                role='FIT'if row['split']=='fit'else'HELD'if row['split']=='holdout'else row['split'],
                id=row['id'],source_points=len(row['uv']),active_point_count=len(active),active_original_point_indices=active,
                physical_binding=dict(feature_id=row.get('feature_id'),curve=curve,material_roles=row.get('material_roles'),
                    logical_print_geometry_columns=logical),
                original_operator={k:row[k]for k in('point_correspondence','curve_observation_kind','source_optical_normal_support_authorized',
                    'source_point_bounds_authorized')if k in row}))
    bypts=defaultdict(list)
    for row in rows:bypts[row['actual_pts']].append(row)
    # Global families include additional programme and external viewpoints.
    # Validate source-container identity and aliases here; do not re-admit
    # source PNGs or evaluate their cameras in this metadata-only audit.
    external=packet['external']
    external_pts=set(map(int,external['canonical_pts']))
    assert not external_pts.intersection(PTS)
    pattern_root=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/low5_repair/secondary6_Ad13_Ad12_native_patterns_v1')
    pattern_sources=[];programme_fit=set();programme_held=set()
    for ad,expected in ((12,'8bc7571bac78ebb6c3ca6fa2d21f9085ba43b8ac9277021e53883293f3d8e28f'),
                        (13,'214448266965e5ee14e8519c0fef8bac03bba356273c8acdaea17c00f84775e5')):
        path=pattern_root/f'secondary6_Ad{ad}_native_pattern_QUALIFIED_FROZEN.json'
        assert a.sha(path)==expected
        source=a.read(path);assert source['physical_ad_id']==ad
        assert source['schema']=='rinkspace.native-advert-pattern-source.v1'
        fit=[];held=[]
        for frame in source['frames']:
            native=int(frame['source_native_pts']);actual=int(frame['source_actual_pts'])
            assert aliases[native]==actual
            assert frame['split']in('fit','held')
            (fit if frame['split']=='fit'else held).append(actual)
        programme_fit.update(fit);programme_held.update(held)
        pattern_sources.append(dict(**a.ref(path),advertisement_id=ad,
            FIT_actual_pts=sorted(set(fit)),HELD_actual_pts=sorted(set(held))))
    external_pattern_reference=external['source_manifest']['source_authorities']['patterns']
    assert a.sha(external_pattern_reference['path'])==external_pattern_reference['sha256']
    external_pattern_source=a.read(external_pattern_reference['path'])
    assert same_source_values(external_pattern_source['rows'],external['row_sets']['pattern_source_records'])
    external_pattern_fit=set();external_pattern_held=set();excluded_pattern=0
    for row in external_pattern_source['rows']:
        if row['physical_id']=='Ad12':
            assert row['outside_actual38']is True and row['fit_admitted']is False
            excluded_pattern+=1;continue
        assert row['physical_id']=='Ad13'
        actual=int(row['source_actual_pts']);native=int(row['source_native_pts'])
        assert external['aliases'][native]==actual
        (external_pattern_held if row['split']=='held'else external_pattern_fit).add(actual)
    assert not (programme_fit|programme_held).intersection(PTS)
    tail=dict(external_main_to_low=dict(source_class=external['source_class'],external_source_pts=sorted(external_pts),
        same_programme_pts_intersection=[],global_source_still_in_full_target=True,
        actual_local_parameter_count=len(external['local_q']),actual_private_chart='38 independent RF4 plus 2 shared unknown-anchor log depths',
        row_set_counts={key:len(value)for key,value in external['row_sets'].items()},
        source_authorities=external['source_manifest']['source_authorities'],
        actual_factory=a.ref(a.ROOT/'calibration/solver_runtime/legacy/structured_programme_source_candidate/current_full_source.py')),
        strong_high_wide_source=dict(source_pts=56218880,same_programme_pts_intersection=[],
            separate_wide_camera_chart=True,global_source_still_in_full_target=True,
            declaration=a.ref(a.ROOT/'calibration/solver_runtime/legacy/wide_anchor_candidate/complete_wide_free_optics_source.py')),
        shared_board_patterns=dict(global_shared_pattern_factor=True,
            contains_additional_programme_and_external_camera_observations=True,
            programme_source_packets=pattern_sources,
            programme_FIT_actual_pts=sorted(programme_fit),programme_HELD_actual_pts=sorted(programme_held),
            programme_same_selected_pts_intersection=[],
            external_source_packet=external_pattern_reference,
            external_FIT_actual_pts=sorted(external_pattern_fit),external_HELD_actual_pts=sorted(external_pattern_held),
            excluded_pre_window_Ad12_source_rows=excluded_pattern,
            source_container_SHA_and_aliases_checked=True,RGB_sources_not_reauthenticated_by_this_metadata_audit=True,
            producer=a.ref(a.ROOT/'calibration/solver_runtime/legacy/structured_programme_source_candidate/current_full_source.py'),
            original_owner=a.ref(a.ROOT/'calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/z190_joint_shared_board_pattern_adapter.py')))
    # Direct centres duplicate existing source-container UV identities. They
    # remain actual objectives but add no independent RGB source evidence.
    centres=manifest.get('highest_direct_human_hard_truth',{})
    report=dict(schema='rinkspace.same55-exposure-absolute-source-metadata.v1',script=a.ref(__file__),
        config=a.ref(a.CONFIG),input_manifest=a.ref(a.RUN/'input_manifest.json'),
        original_packet=a.ref(a.PACKET),native_packet_references=source_refs,
        state=a.ref(state_path),camera=a.ref(identity['camera_path']),geometry_parameter_indices=proof['actual_world_parameter_indices'],
        selected_actual_pts=PTS,low_Ad4_selected_actual_pts=LOW,rows=rows,binding_failures=binding_failures,
        per_selected_actual_pts=[dict(actual_pts=pts,FIT_row_count=sum(r['role']=='FIT'and r['active_point_count']>0 for r in bypts[pts]),
            HELD_row_count=sum(r['role']=='HELD'for r in bypts[pts]),row_indices=[i for i,r in enumerate(rows)if r['actual_pts']==pts])for pts in PTS],
        global_tail_same_exposure_check=tail,
        current_direct_human_manifest_keys=list(centres),
        scope_limits=['Actual physical register binding and compaction/qualification are checked for original absolute and four native material/ice families only.',
            'Direct correlated centres, 86 hard guards and persistent/static source tracks are not counted as independent new absolute entities by this metadata summary.',
            'Native circle suffix and independent pattern placement columns must be included through their public actual factor during the subsequent local derivative audit.',
            'The three global tail families remain scene constraints despite having no added absolute observation on these selected programme PTS.',
            'Metadata and co-presence do not certify rank or physical truth.'],
        no_projection_residual_J_fit_decode_or_image=True)
    OUT.write_text(json.dumps(report,default=a.plain,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(path=str(OUT),sha256=a.sha(OUT),bytes=OUT.stat().st_size,rows=len(rows),binding_failures=len(binding_failures))))
    for pts in LOW+[230390528,230254336]:
        print(json.dumps(dict(actual_pts=pts,rows=[dict(family=r['family'],id=r['id'],role=r['role'],points=r['active_point_count'],
            physical_binding=r['physical_binding'])for r in bypts[pts]]),default=a.plain,ensure_ascii=False))


if __name__=='__main__':main()
