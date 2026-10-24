"""Create new native paint Queries from independently confirmed full paint sides.

The old failed Queries are evidence links only. This producer never restores
those pixels, reuses their statistical information, or edits a source packet.
The existing native_surface_append_v2 consumes its output.
"""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import cv2
import numpy as np
from .dense_joint_proofs import _checked, _ref, _write
from .native_circle_midpoint_review import _require_complete_paint_identity
from .native_source import iter_scope_native_rgb
from .circle_source_continuity import _footprint_clear
from rinkspace.vision_source_shapes import read_source_stroke_midlines


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _plain(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {k:_plain(v)for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [_plain(v)for v in value]
    return value


def _replay_parameters(item):
    pair = item.get('actual_paired_material_profile')
    replay = item.get('paint_midpoint_replay', {})
    pair = pair or replay['actual_paired_material_profile']
    entering, leaving = (pair['actual_RGB_entering_interface'], pair['actual_RGB_leaving_interface'])
    return pair, dict(
        entering_search_intervals_px=[replay.get('entering_search_interval_px', entering['source_search_normal_interval_px'])],
        leaving_search_intervals_px=[replay.get('leaving_search_interval_px', leaving['source_search_normal_interval_px'])],
        entering_context_search_intervals_px=[replay.get('entering_context_interval_px', entering['source_context_profile']['source_search_normal_interval_px'])],
        leaving_context_search_intervals_px=[replay.get('leaving_context_interval_px', leaving['source_context_profile']['source_search_normal_interval_px'])],
        tangent_half_width_px=replay.get('tangent_half_width_px', 2),
        paint_signal_increases=replay.get('paint_signal_increases', entering['material_transition']=='dark_to_light'))


def prepare(*, readouts, scope, roles, masks, output, person_context_margin_px=20):
    """Replay complete sides and issue new source IDs, with an immutable ledger.

    readouts has confirmed rows carrying complete_paint_identity and each side's
    original search/context intervals. masks is the original person-mask ledger;
    full predicted boxes are excluded as unknown context, not asserted human.
    """
    readouts, scope, roles, masks = map(lambda p: Path(p).resolve(), (readouts, scope, roles, masks))
    document = _read(readouts)
    if document.get('schema') not in ('rinkfit.confirmed_native_paint_midpoints.v1', 'rinkfit.native_source_measurement_revision_review.v1'):
        raise ValueError('Explicit independently confirmed complete-paint readings required')
    from .circle_source_identity import circle_name, C0_CURRENT_LABEL, require_c0_midline_reading
    def item_entity(item):
        return circle_name(item.get('material_label', 'ice.'+item.get('complete_paint_identity', {}).get('entity_id', '')))
    excluded_policy = [dict(native_pts=x.get('native_pts'), source_row_id=x.get('source_row_id'), reason='A1_A4_removed_from_current_fitting_no_replacement') for x in document['rows'] if item_entity(x) in {'A1','A2','A3','A4'}]
    items = [x for x in document['rows'] if item_entity(x) not in {'A1','A2','A3','A4'}]
    if not items or any(x.get('measurement_operator', 'paint_center')!='paint_center' for x in items):
        raise ValueError('Missing complete paint measurements')
    by_pts = defaultdict(list)
    dependencies = [_ref(readouts), _ref(scope), _ref(roles), _ref(masks), _ref(__file__)]
    for item in items:
        proof = _require_complete_paint_identity(item, readouts.parent)
        if not proof.get('physical_identity_finding'):
            raise ValueError('A material profile is insufficient without a named source entity')
        dependencies.extend(proof['actual_source_evidence_refs'])
        by_pts[int(item['native_pts'])].append(item)
    coverage = {}
    with gzip.open(masks, 'rt', encoding='utf-8') as stream:
        for line in stream:
            entry=json.loads(line)
            if int(entry['native_pts']) in by_pts:
                if int(entry['native_pts']) in coverage:
                    raise ValueError('Ambiguous source-mask exposure')
                coverage[int(entry['native_pts'])]=entry
    if set(coverage)!=set(by_pts):
        raise ValueError('Every source exposure requires its original exclusion-mask binding')
    out=Path(output).resolve()
    if out.exists():
        raise FileExistsError('New source evidence must have a new output')
    out.mkdir(parents=True)
    rows, frames, excluded, identities=[], [], list(excluded_policy), set()
    source_info=_read(scope)['source']
    from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion
    import rinkspace.vision_source_shapes as shapes
    import rinkfit_joint.native_source as native
    dependencies.extend([_ref(shapes.__file__), _ref(native.__file__)])
    unique={(str(Path(ref['file']).resolve()).casefold(),ref['sha256']):ref for ref in dependencies}
    dependencies=list(unique.values())
    image_by_pixels={}
    for ref in dependencies:
        file=Path(ref['file'])
        if file.suffix.lower()!='.png':continue
        image=cv2.imread(str(file),cv2.IMREAD_COLOR)
        if image is not None:
            image_by_pixels[hashlib.sha256(image.tobytes()).hexdigest()]=file
    cv2.setNumThreads(1)
    for pt,rgb,receipt in iter_scope_native_rgb(scope, by_pts, roles_path=roles):
        cover=coverage[pt]
        if (cover['source_BGR_sha256']!=receipt['decoded_native_BGR_sha256'] or
                cover['source_RGB_sha256']!=receipt['decoded_native_RGB_sha256'] or
                cover['source_video_sha256']!=receipt['source_video_sha256']):
            raise ValueError('Source mask belongs to a different native exposure')
        maskref=dict(file=cover['mask_file'],sha256=cover['mask_sha256'])
        originalmask=_checked(maskref,masks.parent)
        mask=cv2.imread(str(originalmask),0)
        if mask is None or mask.shape!=rgb.shape[:2]:
            raise ValueError('Missing exact native exclusion mask')
        mask=mask.copy();dependencies.append(maskref)
        for box in cover.get('current_boxes', []):
            binding=box['source_binding']
            if binding['source_pts']!=pt or binding['decoded_native_rgb_sha256']!=receipt['decoded_native_RGB_sha256']:
                raise ValueError('Person context belongs to another source exposure')
            a,b,c,d=np.asarray(box['box_xyxy'],float)
            margin=person_context_margin_px
            mask[max(0,int(np.floor(b))-margin):min(mask.shape[0],int(np.ceil(d))+margin+1),
                 max(0,int(np.floor(a))-margin):min(mask.shape[1],int(np.ceil(c))+margin+1)]=255
        bgr=rgb[:,:,::-1].copy();graphics,_=current_broadcast_exclusion(bgr);mask[graphics]=255
        png=out/(str(pt)+'.png');maskfile=out/(str(pt)+'_mask.png')
        # Reuse an existing exact source image when the evidence already names it.
        png=image_by_pixels.get(receipt['decoded_native_BGR_sha256'],png)
        if not png.exists() and not cv2.imwrite(str(png),bgr):raise OSError('Cannot save source PNG')
        if not cv2.imwrite(str(maskfile),mask):raise OSError('Cannot save exclusion mask')
        features=[];frame_rows=[]
        for ordinal,item in enumerate(by_pts[pt]):
            proof=item['complete_paint_identity']
            if proof['native_RGB_sha256']!=receipt['decoded_native_RGB_sha256'] or proof['source_video_sha256']!=receipt['source_video_sha256']:
                raise ValueError('Complete-paint identity binds another RGB')
            prior,params=_replay_parameters(item)
            nav=np.asarray(item.get('navigation_uv',prior['source_search_point_uv']),float)
            normal=np.asarray(item.get('normal',prior['source_search_normal']),float)
            binding=dict(source_sha256=receipt['source_video_sha256'],source_pts=pt,time_base=receipt['time_base'],shot_id=receipt['shot_id'],decoded_native_rgb_sha256=receipt['decoded_native_RGB_sha256'])
            pair=read_source_stroke_midlines(rgb,binding=binding,search_points_uv=[nav],normals=[normal],**params)['readings'][0]
            if pair['disposition']!='measured_paired_stroke_midline' or any(pair['source_flags'].values()):
                raise ValueError('Confirmed full-paint profile no longer replays')
            side=[pair['actual_RGB_entering_interface'],pair['actual_RGB_leaving_interface']]
            flanks=np.array([x['actual_source_uv'] for x in side]);uv=flanks.mean(0)
            expected=item.get('current_uv',item.get('uv'))
            if expected is None or not np.allclose(uv,expected,rtol=0,atol=1e-6):raise ValueError('Source midpoint changed')
            extent=max(abs(v)for k in ('entering_context_search_intervals_px','leaving_context_search_intervals_px')for v in params[k][0])+1
            if not _footprint_clear(nav,normal,mask==0,extent,4):
                excluded.append(dict(native_pts=pt,node_id=item.get('node_id'),reason='unknown_person_or_graphics_context'));continue
            contrasts=[abs(x['actual_source_sided_material_delta'])for x in side]
            if min(contrasts)<4:
                excluded.append(dict(native_pts=pt,node_id=item.get('node_id'),reason='below_existing_native_material_contrast_gate',contrasts=contrasts));continue
            label=C0_CURRENT_LABEL;mi=int(item.get('material_index',item.get('new_material_index')));path=int(item.get('path',item.get('new_path',0)))
            identity=hashlib.sha256(json.dumps([receipt['source_video_sha256'],pt,label,path,uv.tolist(),normal.tolist()],separators=(',',':')).encode()).hexdigest()[:24]
            sid=item.get('source_row_id','native_paint_center/'+receipt['shot_id']+'/'+str(pt)+'/'+identity)
            if sid in identities:raise ValueError('Duplicate new source Query')
            identities.add(sid)
            quarters=[(nav+np.array(s['source_q10_q50_q90_normal_offsets_px'])[:,None]*normal).tolist()for s in side]
            profile=dict(status='current_native_material_width_midpoint',measurement_definition='arithmetic_width_midpoint_of_two_actual_material_edges',complete_both_material_sides_visible=True,actual_source_support_outside_person_and_broadcast_masks=True,current_uv=uv.tolist(),normal=normal.tolist(),two_side_interfaces_uv=flanks.tolist(),side_interface_quartiles_uv=quarters,signed_interface_IQR_contrast=contrasts,actual_paired_material_profile=pair)
            fi=len(features);features.append(dict(id=label,physical_surface='ice',measurement_operator='paint_center',complete_paint_identity=proof,segments=[dict(uv=[uv.tolist()],pixels_measured_in_this_frame=True)]))
            frame_rows.append(dict(source_dataset_id='native_complete_paint_midpoints',source_row_id=sid,group_id='native_paint/'+receipt['shot_id']+'/'+str(pt)+'/'+proof['entity_id'],frame_index=len(frames),feature_index=fi,segment_index=0,source_field='uv',source_point_index=0,source_uv=uv.tolist(),shot_id=receipt['shot_id'],native_pts=pt,material_index=mi,material_label=label,physical_surface='ice',allowed_paths=[path],active=True,inside_person_mask=False,engineering_E=2.,source_model_residual_used=False,original_UV_changed=False,current_native_profile=profile,previous_failed_query_is_same_physical_point=False,source_identity_review=_ref(readouts)))
        for measured_row in frame_rows:require_c0_midline_reading(measured_row)
        if not frame_rows:continue
        image_ref,mask_ref=_ref(png),_ref(maskfile)
        record=_write(out/(str(pt)+'.json'),dict(source=dict(native_pts=pt,source_video_sha256=receipt['source_video_sha256'],time_base=receipt['time_base'],png_sha256=image_ref['sha256'],mask_sha256=mask_ref['sha256'],decoded_native_BGR_sha256=receipt['decoded_native_BGR_sha256']),features=features))
        frames.append(dict(native_pts=pt,shot_id=receipt['shot_id'],frame_role='native_material_repair_training',source=record,image=image_ref,mask=mask_ref))
        rows.extend(frame_rows)
    if not rows:raise ValueError('No independent complete-paint Queries passed the existing native source gate')
    rowref,frameref=_write(out/'rows.json',_plain(rows)),_write(out/'frames.json',frames)
    evidence=dict(schema='rinkfit.native_surface_source_readout_evidence.v2',source_dataset_id='native_complete_paint_midpoints',source_model_residual_used=False,visual_surface_identity_review_completed=True,rows=rowref,frames=frameref,reader=_ref(__file__),source_profile_reader=_ref(shapes.__file__),review_images=list({(x['image']['file'],x['image']['sha256']):x['image']for x in frames}.values()),source_evidence=dependencies,source_video=dict(file=source_info['path'],sha256=source_info['sha256']),actual_native_RGB_and_both_paint_sides_replayed=True,source_row_count=len(rows),excluded_source_candidates=excluded,old_failed_queries_restored=False,source_noise_W_or_sigma_issued=False,quality_accepted=False)
    result=_write(out/'evidence.json',evidence)
    return dict(evidence=result,new_source_rows=len(rows),native_frames=len(frames),excluded_rows=len(excluded),source_packet_written=False)
