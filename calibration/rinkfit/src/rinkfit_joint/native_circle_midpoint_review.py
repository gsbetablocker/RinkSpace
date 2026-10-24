"""Prepare exact paired-paint source reviews for the measurement-revision API.

This adapter does not install a packet or claim that the existing single-edge
native guard accepts paint midpoints. UV history and source identity stay
explicit, and every requested midpoint is the arithmetic mean of two actual
material-side observations in one original exposure.
"""
from pathlib import Path
import json
import numpy as np
from .compact_identity_withhold_v2 import _array, _source_identities
from .dense_joint_proofs import _ref, _checked, _write


def _require_complete_paint_identity(item, base):
    """A local dark-profile pair is a candidate, never a full-paint identity."""
    proof = item.get('complete_paint_identity')
    if not isinstance(proof, dict) or proof.get('schema') != 'rinkfit.reviewed_complete_circle_paint_sides.v1':
        raise ValueError('Missing actual complete C0 paint-side identity; a local dark-band replay cannot grant qualification')
    if (proof.get('entity_id') != 'C0'
            or proof.get('full_paint_cross_section_confirmed') is not True
            or proof.get('same_physical_paint_stroke_confirmed') is not True
            or proof.get('internal_dark_band_used_as_complete_paint') is not False
            or proof.get('model_or_conic_used_to_issue_identity') is not False
            or proof.get('source_fit_eligible') is not True):
        raise ValueError('Unresolved complete paint material cannot issue a midpoint')
    refs = proof.get('actual_source_evidence_refs', ())
    if not refs:
        raise ValueError('Full paint-side identity requires actual source evidence')
    for ref in refs:
        evidence = _checked(ref, base)
        if evidence.suffix.lower() == '.json':
            record = json.loads(evidence.read_text(encoding='utf-8-sig'))
            if record.get('source_identity_finding_retracted') is True or record.get('qualification_status', '').startswith('withdrawn'):
                raise ValueError('Complete paint identity evidence was withdrawn')
    if item.get('native_pts') is not None and int(proof.get('native_pts', -1)) != int(item['native_pts']):
        raise ValueError('Complete paint-side identity binds a different exposure')
    return proof


def prepare_review(*, parent_packet, source_config, source_coverage, assignments,
                   physical_identity_evidence, evidence, output):
    """Prepare source Queries from explicit independently read complete sides.

    A replayable RGB profile cannot itself issue C0/A identity. Each assignment
    must also carry a reviewed complete-paint identity bound to this exposure.
    Previously withheld membership needs an explicit separate predecessor.
    """
    from .circle_source_identity import circle_name
    parent_packet = Path(parent_packet).resolve()
    parent = json.loads(parent_packet.read_text(encoding='utf-8-sig'))
    for ref in [source_config, source_coverage, *physical_identity_evidence, *evidence]:
        _checked(ref, parent_packet.parent)
    if not assignments or not physical_identity_evidence:
        raise ValueError('Actual source arc identity and complete paired material readings required')
    ix = [int(a['raw_row']) for a in assignments]
    if len(set(ix)) != len(ix): raise ValueError('Revised raw Query rows must be distinct')
    ids, _ = _source_identities(parent_packet, ix)
    columns = {k:_array(parent['arrays'][k]) for k in
               ('uv', 'native_pts', 'material_index', 'path', 'admission', 'fit_role', 'kind', 'alpha', 'alpha_bounds')}
    rows = []
    for assignment in assignments:
        proof = _require_complete_paint_identity(assignment, parent_packet.parent)
        i = int(assignment['raw_row'])
        if not (columns['admission'][i] == columns['fit_role'][i] == 1 and columns['kind'][i] == 0
                and columns['alpha'][i] == 0 and np.all(columns['alpha_bounds'][i] == 0)):
            raise ValueError('Use original eligible sliding source rows; membership recovery is separate')
        ref = assignment.get('source_paired_readings')
        if ref is None: raise ValueError('Local dark-band receipts cannot replace explicit complete paired-side readings')
        readings = json.loads(_checked(ref, parent_packet.parent).read_text(encoding='utf-8-sig'))
        if (readings.get('schema') != 'rinkspace.source-paired-material-stroke-midline.v1'
                or readings.get('world_camera_projection_input_used') is not False
                or readings.get('search_points_used_as_measurements') is not False):
            raise ValueError('Actual complete paired-side source readings required')
        binding = readings['source_binding']
        if (int(binding['source_pts']) != int(columns['native_pts'][i])
                or int(proof.get('native_pts', -1)) != int(binding['source_pts'])
                or proof.get('native_RGB_sha256') != readings['RGB_sha256']
                or proof.get('source_video_sha256') != binding['source_sha256']
                or ids[i] not in proof.get('source_row_ids', ())):
            raise ValueError('Complete paint identity does not bind the actual source Query/RGB/exposure')
        pair = readings['readings'][int(assignment['reading_index'])]
        if (pair['disposition'] != 'measured_paired_stroke_midline' or any(pair['source_flags'].values())
                or pair.get('intensity_median_or_darkest_peak_used_as_center') is not False):
            raise ValueError('Unresolved complete paint sides cannot become a midpoint')
        left = pair['actual_RGB_entering_interface']; right = pair['actual_RGB_leaving_interface']
        if any(side['disposition'] != 'measured_material_transition' or any(side['source_flags'].values())
               for side in (left, right)):
            raise ValueError('Both complete paint-to-ice interfaces must be independently observed')
        first = np.asarray(left['actual_source_transition_candidate_uv'], float)
        last = np.asarray(right['actual_source_transition_candidate_uv'], float)
        center = (first + last)*.5
        normal = np.asarray(pair['source_search_normal'], float)
        navigation = np.asarray(pair['source_search_point_uv'], float)
        if (not np.allclose(center, pair['actual_source_uv'], rtol=0, atol=1e-9)
                or not np.isfinite(center).all() or abs(normal@normal-1) > 1e-10):
            raise ValueError('Finite arithmetic midpoint of two actual source interfaces required')
        label = 'ice.C0:middle'
        if circle_name(label) != proof['entity_id']:
            raise ValueError('Complete paint identity does not match the finite-width circle material')
        profile = [float(np.linalg.norm(last-first)*.5),
            min(abs(float(left['actual_source_sided_material_delta'])), abs(float(right['actual_source_sided_material_delta']))),
            float((center-navigation)@normal), 0., -1.]
        replay = dict(reader='rinkspace.vision_source_shapes.read_source_stroke_midlines',
            entering_search_interval_px=left['source_search_normal_interval_px'],
            leaving_search_interval_px=right['source_search_normal_interval_px'],
            entering_context_interval_px=left['source_context_profile']['source_search_normal_interval_px'],
            leaving_context_interval_px=right['source_context_profile']['source_search_normal_interval_px'],
            tangent_half_width_px=2,paint_signal_increases=False,actual_flanks_uv=[first.tolist(),last.tolist()],
            actual_paired_material_profile=pair)
        rows.append(dict(raw_row=i,source_row_id=ids[i],native_pts=int(columns['native_pts'][i]),
            old_uv=columns['uv'][i].tolist(),current_uv=center.tolist(),navigation_uv=navigation.tolist(),
            normal=normal.tolist(),profile=profile,measurement_operator='paint_center',
            complete_paint_identity=proof,paint_midpoint_replay=replay,
            old_material_index=int(columns['material_index'][i]),new_material_index=int(columns['material_index'][i]),
            old_path=int(columns['path'][i]),new_path=int(columns['path'][i]),material_label=label,
            physical_identity_finding=proof['physical_identity_finding'],
            physical_identity_evidence=[*physical_identity_evidence,ref,*proof['actual_source_evidence_refs']],
            native_RGB_sha256=readings['RGB_sha256'],native_BGR_sha256=assignment['native_BGR_sha256'],
            source_query_remeasured=True,persistent_material_point_correspondence=False,
            source_model_projection_used=False,source_images_changed=False,original_source_row_retained=True))
    review = dict(schema='rinkfit.native_source_measurement_revision_review.v1',parent_packet=_ref(parent_packet),
        source_config=source_config,source_coverage=source_coverage,evidence=[*evidence,_ref(__file__)],rows=rows,
        source_model_projection_used=False,source_images_changed=False,measured_uv_changed=True,
        measurement_operator_extension='complete_paired_paint_midpoint_v2',
        source_query_identity_history_preserved=True,source_UV_is_warp_or_model_prediction=False,
        requires_explicit_membership_recovery_if_parent_is_withheld=True,
        existing_single_interface_native_guard_accepts_this_operator=False,quality_accepted=False)
    ref = _write(output, review)
    return dict(review=ref,revised_rows=len(rows),actual_source_midpoints=[r['current_uv'] for r in rows],
                packet_installed=False,native_paint_midpoint_operator_extension_required=True)


def validate_paint_center_review(*, parent_packet, review, scope_path, roles_path):
    """Replay every paint side from the exact official RGB, without installing.

    Returns the same checked structure as native_source_revision_v1.validate_review
    for its existing installer. Membership recovery is a separate predecessor.
    """
    import gzip,hashlib,cv2
    from .native_source import iter_scope_native_rgb
    from .circle_source_continuity import _footprint_clear
    from rinkspace.vision_source_shapes import read_source_stroke_midlines
    packet_path=Path(parent_packet).resolve(); review_path=Path(review).resolve()
    parent=json.loads(packet_path.read_text(encoding='utf-8-sig'));doc=json.loads(review_path.read_text(encoding='utf-8-sig'))
    if (doc.get('schema')!='rinkfit.native_source_measurement_revision_review.v1'
            or doc.get('source_model_projection_used') is not False
            or doc.get('source_images_changed') is not False or doc.get('measured_uv_changed') is not True
            or doc.get('parent_packet')!=_ref(packet_path) or not doc.get('rows')):
        raise ValueError('Exact paired-paint source generation required')
    dependencies=[doc['parent_packet'],doc['source_config'],doc['source_coverage'],*doc['evidence'],_ref(scope_path),_ref(roles_path),_ref(__file__)]
    config=json.loads(_checked(doc['source_config'],review_path.parent).read_text(encoding='utf-8-sig'))
    coverage_path=_checked(doc['source_coverage'],review_path.parent)
    if Path(config['source_coverage']).resolve()!=coverage_path:raise ValueError('Source coverage differs')
    scope=json.loads(Path(scope_path).read_text(encoding='utf-8-sig'))
    if scope['source']['sha256']!=config['source_video_sha256']:raise ValueError('Official source identity differs')
    coverage={}
    for line in gzip.open(coverage_path,'rt',encoding='utf8'):
        frame=json.loads(line);coverage[int(frame['native_pts'])]=frame
    columns={k:_array(parent['arrays'][k]) for k in ('uv','native_pts','material_index','path','admission','fit_role','kind','alpha','alpha_bounds')}
    indices=[int(r['raw_row'])for r in doc['rows']]
    if len(indices)!=len(set(indices)):raise ValueError('Duplicate paint revision Query')
    identities,_=_source_identities(packet_path,indices);wanted={int(r['native_pts'])for r in doc['rows']};by_pts={}
    for item in doc['rows']:
        proof=_require_complete_paint_identity(item, review_path.parent)
        i=int(item['raw_row']);pt=int(item['native_pts'])
        if (proof.get('native_RGB_sha256')!=item['native_RGB_sha256'] or identities[i]not in proof.get('source_row_ids',())):
            raise ValueError('Complete paint identity does not bind exact Query and RGB')
        if (item['measurement_operator']!='paint_center' or identities[i]!=item['source_row_id']
                or columns['uv'][i].tolist()!=item['old_uv'] or int(columns['native_pts'][i])!=pt
                or columns['admission'][i]!=1 or columns['fit_role'][i]!=1 or columns['kind'][i]!=0
                or columns['alpha'][i]!=0 or np.any(columns['alpha_bounds'][i]!=0)
                or int(columns['material_index'][i])!=item['old_material_index']
                or int(columns['path'][i])!=item['old_path']
                or item['new_material_index']!=item['old_material_index'] or item['new_path']!=item['old_path']):
            raise ValueError('Exact original eligible circle Query required; recover membership separately')
        for ref in item['physical_identity_evidence']:_checked(ref,review_path.parent);dependencies.append(ref)
        by_pts.setdefault(pt,[]).append(item)
    validated=[];frames=[]
    for pt,rgb,receipt in iter_scope_native_rgb(scope_path,wanted,roles_path=roles_path):
        source=coverage[pt];digest=receipt['decoded_native_BGR_sha256']
        if digest!=source['source_BGR_sha256']:raise ValueError('Actual native BGR differs')
        if any(v['complete_paint_identity'].get('source_video_sha256')!=receipt['source_video_sha256'] or v['native_RGB_sha256']!=receipt['decoded_native_RGB_sha256'] for v in by_pts[pt]):
            raise ValueError('Actual source identity differs from complete paint proof')
        mask=np.zeros(rgb.shape[:2],np.uint8)
        for ref in [source['person_mask'],source.get('unknown_mask')]:
            if ref is None:continue
            path=_checked(ref,coverage_path.parent);pixels=cv2.imread(str(path),0)
            if pixels is None or pixels.shape!=mask.shape:raise ValueError('Actual source exclusion mask differs')
            mask|=pixels;dependencies.append(ref)
        binding=dict(source_sha256=receipt['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id=by_pts[pt][0]['complete_paint_identity']['shot_id'],decoded_native_rgb_sha256=receipt['decoded_native_RGB_sha256'])
        for item in by_pts[pt]:
            replay=item['paint_midpoint_replay'];navigation=np.asarray(item['navigation_uv'],float);normal=np.asarray(item['normal'],float)
            if replay.get('reader') != 'rinkspace.vision_source_shapes.read_source_stroke_midlines':
                raise ValueError('A local dark-band replay cannot authenticate complete paint width')
            context = [*replay['entering_context_interval_px'], *replay['leaving_context_interval_px']]
            if not _footprint_clear(navigation,normal,mask==0,max(abs(float(v)) for v in context)+1,4):
                raise ValueError('Complete paint profile/background intersects excluded context')
            fresh=read_source_stroke_midlines(rgb,binding=binding,search_points_uv=[navigation],normals=[normal],
                entering_search_intervals_px=[replay['entering_search_interval_px']],
                leaving_search_intervals_px=[replay['leaving_search_interval_px']],
                entering_context_search_intervals_px=[replay['entering_context_interval_px']],
                leaving_context_search_intervals_px=[replay['leaving_context_interval_px']],
                tangent_half_width_px=replay['tangent_half_width_px'],paint_signal_increases=replay['paint_signal_increases'])
            pair=fresh['readings'][0]
            if pair['disposition']!='measured_paired_stroke_midline' or any(pair['source_flags'].values()):
                raise ValueError('Actual complete paint material profile ambiguous')
            left=pair['actual_RGB_entering_interface'];right=pair['actual_RGB_leaving_interface']
            flank0=np.asarray(left['actual_source_transition_candidate_uv']);flank1=np.asarray(right['actual_source_transition_candidate_uv']);uv=(flank0+flank1)*.5
            profile=np.array([np.linalg.norm(flank1-flank0)*.5,min(abs(left['actual_source_sided_material_delta']),abs(right['actual_source_sided_material_delta'])),float((uv-navigation)@normal),0.,-1.])
            if (item['native_BGR_sha256']!=digest or not np.allclose(uv,item['current_uv'],rtol=0,atol=1e-6)
                    or not np.allclose(profile,item['profile'],rtol=0,atol=1e-6)
                    or not np.allclose([flank0,flank1],replay['actual_flanks_uv'],rtol=0,atol=1e-6)):
                raise ValueError('Actual paired paint source replay differs')
            validated.append(dict(item,current_uv=uv.tolist(),profile=profile.tolist(),
                actual_RGB_profile_replay_max_abs_px=float(np.max(abs(uv-np.asarray(item['current_uv'])))),
                original_source_row_retained=True,paired_paint_midpoint_RGB_replayed=True))
        frames.append(dict(native_pts=pt,source_BGR_sha256=digest,person_mask=source['person_mask'],unknown_mask=source.get('unknown_mask'),reviewed_rows=len(by_pts[pt])))
    if len(validated)!=len(indices):raise ValueError('Incomplete actual source replay')
    dependencies.extend([_ref(Path(__file__).with_name('native_source.py')),_ref(Path(__file__).with_name('circle_source_continuity.py'))])
    import rinkspace.vision_source_shapes as shapes
    dependencies.append(_ref(shapes.__file__));unique={}
    for ref in dependencies:
        actual=_checked(ref,review_path.parent);unique[(str(actual).casefold(),ref['sha256'])]=dict(file=str(actual),sha256=ref['sha256'])
    return dict(parent=parent,review=doc,rows=sorted(validated,key=lambda r:r['raw_row']),frames=frames,actual_dependencies=list(unique.values()),actual_native_RGB_replayed=True,source_model_projection_used=False)
