"""Stream source RGB rereads around registered human references, without a camera.

This event tool adapts existing SourceRinkReader, RAFT and material readers.
Derived pixels remain source hypotheses; human truth stays at its own exposure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import av
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from rinkspace.vision import SourceRinkReader, confirmed_source_guides, confirmed_exposure_groups
from rinkspace.vision_geometry import measure_boundary_profiles
from rinkspace.vision_source_topology import (
    _ordered_board_foot_profile_batches, source_material_transition_support,
    read_source_board_rail_material_profiles,
)

SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
MODEL_SHA = '295025ca423a5fd47385c41c21af15976bb4c5cd29dee723664bf6cfdf0920ea'
SHOT_RANGES = {'shot5': (229945856, 230369024), 'shot2': (229497600, 229671680),
               'shot6': (230369024, 230518016)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)): return plain(array(value).tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def emit(stream, value):
    stream.write(json.dumps(plain(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')


def normals(points, prior=None):
    points = np.asarray(points, float)
    tangent = np.gradient(points, axis=0) if len(points) > 1 else np.array([[1., 0.]])
    result = np.c_[-tangent[:, 1], tangent[:, 0]]
    lengths = np.linalg.norm(result, axis=1)
    result[lengths < 1e-9] = [0., 1.]
    result /= np.maximum(np.linalg.norm(result, axis=1), 1e-9)[:, None]
    if prior is None: result[result[:, 1] < 0] *= -1
    else: result[(result*np.asarray(prior)).sum(1) < 0] *= -1
    return result


def sample_field(field, points, nearest=False):
    h, w = field.shape[-2:]
    points = torch.as_tensor(points, device=field.device, dtype=torch.float32)
    grid = points*points.new_tensor([2/(w-1), 2/(h-1)])-1
    return F.grid_sample(field.float(), grid[None, :, None], mode='nearest' if nearest else 'bilinear',
                         align_corners=True)[0, :, :, 0].T.detach().cpu().numpy()


def sparse_original_points(points, spacing=8.):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 3: return points.copy(), np.arange(len(points))
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    selected = [0]
    for i in range(1, len(points)-1):
        if length[i]-length[selected[-1]] >= spacing: selected.append(i)
    if selected[-1] != len(points)-1: selected.append(len(points)-1)
    return points[selected].copy(), np.asarray(selected)


def make_seed(record, binding):
    complete = record['complete_human_stroke']
    ref = complete['reference']
    if sha(ref['path']) != ref['sha256']: raise ValueError('Human complete reference identity changed')
    original = json.loads(Path(ref['path']).read_text(encoding='utf8'))
    raw = record.get('_full_source_uv', original.get('user_stroke_raw_native_px'))
    # Ad frame sides and original Lane samples already have an explicit source
    # correspondence in the hard authority. No invented rectangle is sampled.
    points, selected = sparse_original_points(raw if raw is not None else record['uv'])
    curve = record['curve']
    kind = ('ice_contact' if curve['kind'] == 'ice_board_contact' else
            'board_top' if curve['kind'] == 'board_top' else
            'ad_boundary' if curve['kind'] == 'advertisement_boundary' else 'ice_paint')
    annotation = dict(feature_id=record['id'], group_id=record['id'], kind=kind,
        primitive='free_curve', uv=points.tolist(), physical_role=record['geometry_role'],
        state='user_confirmed_visual_supervision', authority=dict(human_authority=True,
            authority_record_id=record['id'], reference=ref, temporal_scope='exact_reference_exposure'),
        source_pts=binding['source_pts'], source_sha256=SOURCE_SHA,
        search_radius_px=24., annotation_closed=False)
    packet = confirmed_source_guides(dict(source_hash=SOURCE_SHA, source_pts=binding['source_pts'],
        time_base='1/15360', shot_id=binding['shot_id'], frame_size_wh=[1920, 1080],
        annotations=[annotation]), return_packet=True)
    if len(packet['guides']) != 1 or packet['unresolved']: raise ValueError('Typed human source guide was unresolved')
    guide = packet['guides'][0]
    nrm = np.asarray(record['_full_source_normal'])[selected] if '_full_source_normal' in record else normals(points)
    if kind in ('ice_contact', 'board_top'):
        nrm = nrm.copy();nrm[nrm[:, 1]<0] *= -1
    guide.update(points_uv=points, normals=nrm, source_binding=binding,
        authority_reference_pts=int(record['pts']), human_original_query_indices=selected,
            authority_record_id=record['id'], identity_confirmed=True)
    return guide


class NativeDecoder:
    def __init__(self, video):
        self.container = av.open(str(video))
        self.stream = self.container.streams.video[0]
        if Fraction(self.stream.time_base) != Fraction(1, 15360): raise ValueError('Source clock differs')
    def exact(self, pts):
        self.container.seek(int(pts), stream=self.stream, backward=True)
        for frame in self.container.decode(self.stream):
            if frame.pts is None: raise ValueError('Source frame has no PTS')
            if frame.pts == pts: return frame.to_ndarray(format='rgb24')
            if frame.pts > pts: break
        raise ValueError('Exact original frame missing: '+str(pts))
    def presentations(self, start, stop):
        self.container.seek(int(start), stream=self.stream, backward=True)
        for frame in self.container.decode(self.stream):
            if frame.pts is None: raise ValueError('Source frame has no PTS')
            if frame.pts < start: continue
            if frame.pts >= stop: break
            yield int(frame.pts),frame.to_ndarray(format='rgb24')
    def close(self): self.container.close()


def warm_material(reader, rgb, binding, points, directions):
    """Reuse the original yellow entry/platform/exit reader at source hints."""
    device = reader.vision.device
    pts = torch.as_tensor(points, device=device, dtype=torch.float32)
    nrm = torch.as_tensor(directions, device=device, dtype=torch.float32)
    probabilities = reader.vision.current_prediction['class_probabilities']
    profiles = measure_boundary_profiles(reader.vision.current_image,
        probabilities[:, :1]*0+.5, pts, nrm, class_index=0, search_radius=48., sample_step=.5,
        rgb_source_binding=binding, probability_source_binding=binding, projection_source_binding=binding,
        minimum_contrast=.008, maximum_edge_width=32.)
    h, w = rgb.shape[:2]
    uv = profiles['sampled_native_uv']
    grid = torch.stack((2*(uv[..., 0]+.5)/w-1, 2*(uv[..., 1]+.5)/h-1), -1)[None]
    sampled = F.grid_sample(probabilities, grid, align_corners=False)[0].permute(1, 2, 0)
    semantics = {name: sampled[..., list(reader.vision.classes).index(name)] for name in
        ('ice_contact', 'background', 'ice_paint', 'ad_boundary', 'shadow_or_reflection')}
    ordered = _ordered_board_foot_profile_batches(profiles,
        torch.zeros((h, w), device=device, dtype=torch.bool), np.full(len(points), h*.25), semantics)
    support = source_material_transition_support(profiles, ordered)
    offsets = profiles['rgb_gradient_offsets_native_px']
    entry = profiles['points_native_uv']+profiles['normals_unit']*offsets[ordered['entry_index'], None]
    exit_ = profiles['points_native_uv']+profiles['normals_unit']*ordered['selected_exit_offset_native_px'][:,None]
    missing = array(ordered['missing_same_band']).astype(bool)
    unique = array(ordered['usable']).astype(bool) & ~array(ordered['competing_same_band_pairs']).astype(bool)
    return dict(entry_uv=array(entry), lower_exit_uv=array(exit_), unique_lower_role=unique,
        missing_same_band=missing, competing_same_material_bands=array(ordered['competing_same_band_pairs']),
        lower_exit_index=array(ordered['exit_index']), entry_index=array(ordered['entry_index']),
        lower_normal_optical_support=array(support['normal_support_interval_native_px'])
            if 'normal_support_interval_native_px' in support else None,
        all_ordered_pair_indices=array(ordered['all_ordered_pair_indices_profile_entry_exit']),
        selected_band_width_px=array(ordered['selected_band_width_px']),
        selected_inside_warm_gamma=array(ordered['selected_inside_warm_gamma']),
        first_iceward_yellow_material_fall_evidence=ordered['first_iceward_yellow_material_fall_evidence'],
        observed_optical_fall_support=support,
        source_image_roles_are_hypotheses=True, current_RGB_reread=True,
        human_or_flow_search_pixels_used_as_measurements=False)


def blue_material(rgb, binding, points):
    points=np.asarray(points)
    result = read_source_board_rail_material_profiles(rgb, dict(source_binding=binding,
        RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), actual_source_top_uv=points),
        source_instances=(), search_radius_px=36, sample_spacing_px=1)
    selected=[int(np.flatnonzero(points[:,0]==x)[np.argmin(points[points[:,0]==x,1])])
        for x in np.unique(points[:,0])]
    return dict(entry_uv=result['actual_source_blue_rail_entry_uv'], internal_exit_uv=result['actual_source_blue_to_white_exit_uv'],
        unique_entry_role=result['primary_role_samples'], evidence=result['source_material_role_evidence'],
        queried_original_indices=selected,
        missing=result['missing_or_competing_source_roles'], human_or_temporal_navigation_used=True,
        current_RGB_reread=True, derived_source_hypothesis_not_direct_human_GT=True)


def read_frame(reader, rgb, binding, guides, *, old_rows=()):
    # Prediction is reset for each processed exposure, including reverse traversal.
    # RAFT below supplies an independently bound chronological pair; no reversed
    # recurrent PTS or false flow metadata is introduced into the existing reader.
    result = reader.read(rgb, binding, guides_2d=guides, reset=True, extract_source_components=False)
    binding = result['source_binding']
    outputs, next_guides = [], []
    for guide, reading in zip(guides, result['guided']['readings']):
        points = array(guide['points_uv']).copy()
        indices = array(reading.get('coherent_source_indices', reading['source_indices'])).astype(int)
        observed = array(reading.get('coherent_source_uv', reading['actual_source_uv']))
        row = dict(id=guide['id'], kind=guide['kind'], physical_role=guide['physical_role'],
            authority_reference_pts=guide['authority_reference_pts'], original_query_indices=guide['human_original_query_indices'],
            navigation_uv=points.copy(), observed_generic_RGB_edge_uv=observed, observed_query_indices=indices,
            direct_human_GT_input=binding['source_pts']==guide['authority_reference_pts'],
            derived_measurements_are_GT=False, generic_edge_is_not_automatically_same_physical_role=True,
            flow_supported=guide.get('temporal_flow_supported'), flow_fb_error_px=guide.get('flow_fb_error_px'),
            outframe_query_indices=np.flatnonzero((points[:, 0]<0)|(points[:, 0]>=1920)|(points[:, 1]<0)|(points[:, 1]>=1080)),
            missing_or_occluded_query_indices=np.setdiff1d(np.arange(len(points)), indices),
            occlusion_confirmed=False, invisible_source_points_filled=False)
        if guide['kind']=='ice_contact':
            material=warm_material(reader,rgb,binding,points,normals(points,guide['normals']))
            row['yellow_material_order']=material
            # Only observed unique material exits update a search; competing exits
            # remain measurements in the ledger, never silently chosen as truth.
            good=material['unique_lower_role']&~material['missing_same_band']
            points[good]=material['lower_exit_uv'][good]
            row['observed_same_role_uv']=material['lower_exit_uv'][good]
            row['observed_same_role_query_indices']=np.flatnonzero(good)
        elif guide['kind']=='board_top':
            material=blue_material(rgb,binding,points);row['blue_TOP_material_order']=material
            # The public blue reader uses source column subsampling; keep its
            # actual UV and query indices separately instead of inventing a bijection.
            row['observed_same_role_uv']=np.asarray(material['entry_uv'])[np.asarray(material['unique_entry_role'],bool)]
            row['observed_same_role_query_indices']=None
            for entry,ev,good in zip(material['entry_uv'],material['evidence'],material['unique_entry_role']):
                if good:
                    seed=np.asarray(ev['original_source_navigation_seed_uv']);i=int(np.argmin(np.linalg.norm(points-seed,axis=1)))
                    if np.linalg.norm(points[i]-seed)<1e-4:points[i]=entry
        else:
            # Painted-axis versus edge compatibility remains explicit; this
            # diagnostic never promotes a generic gradient to a fitted line.
            points[indices]=observed
            row['target_operator_compatibility']='raw_edge_candidates_only; no inferred painted_axis_GT'
        if binding['source_pts']==guide['authority_reference_pts']:
            row['human_reference_navigation_preserved_exactly']=True
            row['automatic_RGB_measured_edges_do_not_replace_reference_human_GT']=True
            points=array(guide['points_uv']).copy()
        new=dict(guide, points_uv=points, normals=normals(points,guide['normals']),source_binding=binding,
                 origin='temporal_source_search_hint')
        next_guides.append(new);outputs.append(row)
    old_checks=[]
    for old in old_rows:
        role=old['semantic_role'];uv=np.asarray(old['original_uv'],float)
        rec=dict(compact_packet_row_index=old['compact_packet_row_index'],block=old['block'],
            group_order=old['group_order'],row_order=old['row_order'],id=old['id'],
            collection_role=old['collection_role'],original_full_uv_float64_sha256=old['original_full_uv_float64_sha256'],
            source_native_pts=old.get('source_native_pts'),actual_pts=old['group_actual_pts'],
            original_uv=uv,original_sigma=old['original_sigma'],semantic_role=role,
            original_explicit_board_side=old.get('original_explicit_board_side'),
            old_far_fullperimeter_not_assumed_lower_long_side=True,points=[])
        if role in ('ice_board_contact','ice_contact'):
            material=warm_material(reader,rgb,binding,uv,normals(uv));rec['yellow_material_order']=material
            for i,p in enumerate(uv):
                delta=material['lower_exit_uv'][i]-p;upper=material['entry_uv'][i]-p
                outside=bool(np.any(p<0)or np.any(p>=[1920,1080]))
                unique=bool(material['unique_lower_role'][i]and not material['missing_same_band'][i]and not outside)
                internal=unique and np.linalg.norm(upper)<=2. and np.linalg.norm(delta)>4.
                rec['points'].append(dict(original_uv_index=i,fresh_edge_minus_old_uv=delta,
                    yellow_upper_minus_old_uv=upper,unique_material_support=unique,
                    competing=bool(material['competing_same_material_bands'][i]),
                    visibility='outframe'if outside else'visible_material_hypothesis'if unique else'unknown_or_occluded',
                    occlusion_confirmed=False,retire_candidate=internal,
                    reason='old_point_at_yellow_upper_internal_interface'if internal else'no_proven_retirement',
                    quarantine_candidate=not unique,
                    quarantine_reason='outframe'if outside else'competing_or_missing_material_identity'if not unique else None,
                    facet_verified_from_old_model_binding=False,original_FIT_HELD_role_preserved=True))
        elif role=='board_top':
            # The existing reader queries one min-y seed per native x column.
            # Repeat only the unqueried duplicate-column layer, retaining an
            # individual result even when no blue-band candidate exists.
            pending=np.arange(len(uv));layers=[]
            while len(pending):
                material=blue_material(rgb,binding,uv[pending]);layers.append(material)
                selected=np.asarray(material['queried_original_indices'],int)
                measured={int(ev['source_query_index']):(entry,exit_,ev,bool(good))
                    for entry,exit_,ev,good in zip(material['entry_uv'],material['internal_exit_uv'],material['evidence'],material['unique_entry_role'])}
                missing={int(row['query_index']):row['reason']for row in material['missing']}
                for query_index,local in enumerate(selected):
                    i=int(pending[local]);p=uv[i];outside=bool(np.any(p<0)or np.any(p>=[1920,1080]))
                    actual=measured.get(query_index);good=bool(actual is not None and actual[3]and not outside)
                    delta=np.asarray(actual[0])-p if actual else None
                    internal_delta=np.asarray(actual[1])-p if actual else None
                    internal=bool(good and np.linalg.norm(internal_delta)<=2. and np.linalg.norm(delta)>4.)
                    rec['points'].append(dict(original_uv_index=i,fresh_edge_minus_old_uv=delta,
                        blue_internal_exit_minus_old_uv=internal_delta,unique_material_support=good,
                        competing=bool(actual is not None and not actual[3]),
                        visibility='outframe'if outside else'visible_material_hypothesis'if good else'unknown_or_occluded',
                        missing_material_reason=missing.get(query_index),occlusion_confirmed=False,
                        retire_candidate=internal,reason='old_point_at_blue_to_white_internal_exit'if internal else'no_proven_retirement',
                        quarantine_candidate=not good,
                        quarantine_reason='outframe'if outside else missing.get(query_index,'competing_or_missing_material_identity')if not good else None,
                        facet_verified_from_old_model_binding=False,original_FIT_HELD_role_preserved=True))
                pending=np.delete(pending,selected)
            rec['blue_TOP_material_order_layers']=layers
            rec['points'].sort(key=lambda x:x['original_uv_index'])
            if [r['original_uv_index']for r in rec['points']]!=list(range(len(uv))):raise ValueError('Old TOP query coverage incomplete')
            rec['unresolved_original_uv_indices']=[]
        else:rec['status']='not_a_TOP_or_contact_role; retained_raw_index_for_other_QA'
        if role in ('board_top','ice_board_contact','ice_contact'):
            rec['physical_facet_identity']='unverified_by_this_material_reader; original_model_binding_is_history_only'
            rec['facet_quarantine_candidate']=True
            rec['facet_quarantine_requires_main_source_review']=True
        old_checks.append(rec)
    return binding,outputs,next_guides,old_checks


def transport(reader, previous_rgb, current_rgb, previous_binding, binding, guides):
    device=reader.vision.device
    def tensor(rgb):return torch.as_tensor(rgb.copy(),device=device).permute(2,0,1)[None].float()/255.
    before=tensor(previous_rgb);after=tensor(current_rgb)
    if previous_binding['source_pts']<binding['source_pts']:
        pair=reader.vision.flow.compute(before,after,analysis_hw=(384,640),source0=previous_binding,source1=binding)
        field,valid,error=pair.forward,pair.valid_forward,pair.fb_error_forward
    else:
        pair=reader.vision.flow.compute(after,before,analysis_hw=(384,640),source0=binding,source1=previous_binding)
        field,valid,error=pair.backward,pair.valid_backward,pair.fb_error_backward
    moved=[]
    for guide in guides:
        points=array(guide['points_uv']);change=sample_field(field,points)
        supported=sample_field(valid.float(),points,True)[:,0]>.5
        fb=sample_field(error,points)[:,0]
        moved.append(dict(guide,points_uv=points+change,source_binding=binding,origin='temporal_source_search_hint',
            temporal_flow_supported=supported,flow_fb_error_px=fb))
    del pair,field,valid,error,before,after
    return moved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--old-row-index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/rink_vision_candidate_11200_user_iceedge2.pt'))
    parser.add_argument('--shot',choices=tuple(SHOT_RANGES),action='append')
    parser.add_argument('--probe',action='store_true',help='Actual anchor and neighbouring exposure startup only; not full-shot evidence')
    opt=parser.parse_args();out=opt.output.resolve()
    if not out.is_relative_to(Path('D:/CodexData/RinkSpace_Data'))or out.exists():raise ValueError('New Data output required')
    authority=json.loads(opt.authority.read_text(encoding='utf8'))
    if sha(opt.authority)!='1c17877c4746c910c3e25353cc02ce9c53714ec49b2eb48525146a04a2e795f7':raise ValueError('Hard authority identity changed')
    if authority['source_video_sha256']!=SOURCE_SHA:raise ValueError('Human source SHA differs')
    old_index=json.loads(opt.old_row_index.read_text(encoding='utf8'))
    if sha(opt.old_row_index)!='aef268b0f9b046c7644f665eff32dc769fbf6237a798ab91a6bd5dc46833000f':raise ValueError('Old source index identity changed')
    out.mkdir(parents=True);torch.set_num_threads(2)
    reader=SourceRinkReader(ROOT/'configs/rink_vision.json',model_path=opt.model)
    if reader.vision.model_sha256!=MODEL_SHA:raise ValueError('Existing 11200 model identity differs')
    full_path=opt.authority.parent/'human_registered_full_stroke_hard_guard_v1.json'
    if sha(full_path)!='c5db59096204de9b6fd253521736a2255316170339d535fe34812f527beae868':raise ValueError('Full human trace metadata identity changed')
    full=json.loads(full_path.read_text(encoding='utf8'));npz=full['guard_arrays_reference']
    if sha(npz['path'])!='5abd423e11a6e1d28f0d69ac41498c8336300b1bad20fd6ef885354d6add6e51':raise ValueError('Full human source arrays identity changed')
    by_id={r['id']:r for r in authority['records']}
    with np.load(npz['path'],allow_pickle=False)as arrays:
        for r in full['records']:
            original=by_id[r['id']];prefix=r['array_prefix']
            original['_full_source_uv']=arrays[prefix+'_uv'].copy()
            original['_full_source_normal']=arrays[prefix+'_normal'].copy()
    mapping,clock=confirmed_exposure_groups(SOURCE_SHA,'1/15360',registry=reader.vision.exposure_registry)
    video=Path(reader.config['source_video']);decoder=NativeDecoder(video)
    summary=dict(schema='rinkspace.registered-human-source-continuous-RGB-reread.v1',status='running',
        authority_path=str(opt.authority),authority_sha256=sha(opt.authority),old_row_index=str(opt.old_row_index),
        old_row_index_sha256=sha(opt.old_row_index),source_video=str(video),source_sha256=SOURCE_SHA,
        source_video_rehashed_this_run=False,source_reference_frames_verified_against_original_RGB=True,
        model_path=str(opt.model),model_sha256=MODEL_SHA,clock=clock,probe=opt.probe,
        no_training=True,no_camera_or_world_or_model_projection_navigation=True,
        maximum_decoded_RGB_frames_retained=2,full_probabilities_or_flow_arrays_written=False,
        derived_source_hypotheses_promoted_to_GT=False,original_source_packets_modified=False,shots=[])
    summary['source_code_sha256']={str(p):sha(p)for p in (Path(__file__),ROOT/'src/rinkspace/vision.py',
        ROOT/'src/rinkspace/vision_geometry.py',ROOT/'src/rinkspace/vision_source_topology.py')}
    write(out/'run_manifest.json',summary);start=time.perf_counter()
    try:
        for shot in opt.shot or ['shot5','shot2','shot6']:
            lo,hi=SHOT_RANGES[shot];records=[r for r in authority['records']if lo<=r['actual_pts']<hi]
            if not records:raise ValueError('Shot lacks its direct human reference')
            displays=sorted(p for p in mapping if lo<=p<hi)
            aliases=defaultdict(list)
            for p in displays:aliases[int(mapping[p]['canonical'])].append(p)
            actual=sorted(aliases)
            if any(p<lo or p>=hi for p in actual):raise ValueError('Exposure alias crosses shot boundary')
            shot_rows=defaultdict(list)
            for r in old_index['records']:
                if r['shot']==shot:shot_rows[int(r['group_actual_pts'])].append(r)
            seed_rows=defaultdict(list)
            for r in records:seed_rows[int(r['actual_pts'])].append(r)
            first,last=min(seed_rows),max(seed_rows)
            coverage=Counter();done=set();report=dict(shot=shot,start_pts=lo,end_pts_exclusive=hi,
                native_presentations=len(displays),actual_exposures=len(actual),reference_pts=sorted(seed_rows),
                processed_actual_exposures=0,derived_reference_scope_only=True,
                expected_old_source_rows=sum(map(len,shot_rows.values())),
                expected_old_TOP_contact_points=sum(r['point_count']for values in shot_rows.values()for r in values
                    if r['semantic_role']in('board_top','ice_board_contact','ice_contact')),
                new_visible_unassociated_material_is_not_filled=True)
            summary['shots'].append(report)
            def binding(p):return dict(source_sha256=SOURCE_SHA,source_pts=int(p),time_base='1/15360',
                shot_id=shot,installation_id='source_classified_'+shot,native_hw=[1080,1920])
            def inject(p,rgb,guides):
                if p not in seed_rows:return guides
                byid={g['id']:g for g in guides}
                for r in seed_rows[p]:
                    if sha(r['source_raw_frame'])!=r['source_file_sha256']:raise ValueError('Reference PNG identity changed')
                    expected=np.asarray(Image.open(r['source_raw_frame']).convert('RGB'))
                    if not np.array_equal(rgb,expected):raise ValueError('Actual video reference differs from registered PNG')
                    byid[r['id']]=make_seed(r,binding(p))
                return list(byid.values())
            with gzip.open(out/(shot+'_measurements.jsonl.gz'),'wt',encoding='utf8')as stream:
                # Prepare later-reference Ad sides down to the first reference.
                # This remains source-derived navigation, never human GT there.
                prep=[first] if opt.probe else [p for p in actual if first<=p<=last][::-1]
                if opt.probe:last=first
                rgb=decoder.exact(last);p=last;guides=inject(p,rgb,[]);b=binding(p)
                for target in prep:
                    if target!=p:
                        current=decoder.exact(target);guides=transport(reader,rgb,current,b,binding(target),guides)
                        rgb=current;p=target;b=binding(p)
                    guides=inject(p,rgb,guides)
                    b,rows,guides,old=read_frame(reader,rgb,b,guides,old_rows=shot_rows.get(p,()))
                    if target!=first:emit(stream,dict(phase='later_reference_backward_navigation_preparation',source_pts=p,
                        actual_pts=p,display_pts=aliases[p],source_binding=b,features=rows,old_source_checks=old))
                anchor_guides=deepcopy(guides);anchor_binding=b
                phases=[('backward',[p for p in actual if p<=first][::-1]),('forward',[p for p in actual if p>first])]
                if opt.probe:phases=[('probe',[first,next(p for p in actual if p>first)])]
                for phase,targets in phases:
                    rgb=decoder.exact(first);guides=deepcopy(anchor_guides);b=anchor_binding;p=first
                    for target in targets:
                        if target!=p:
                            current=decoder.exact(target);guides=transport(reader,rgb,current,b,binding(target),guides)
                            rgb=current;p=target;b=binding(p)
                        guides=inject(p,rgb,guides)
                        b,rows,guides,old=read_frame(reader,rgb,b,guides,old_rows=shot_rows.get(p,()))
                        emit(stream,dict(phase=phase,source_pts=p,actual_pts=p,display_pts=aliases[p],source_binding=b,
                            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),features=rows,old_source_checks=old,
                            direct_human_reference=p in seed_rows,derived_source_hypotheses_not_GT=p not in seed_rows))
                        done.add(p);coverage['old_source_rows_checked']+=len(old)
                        for row in old:
                            coverage['old_TOP_contact_points_checked']+=len(row['points'])
                            coverage['old_point_retire_candidates']+=sum(bool(v['retire_candidate'])for v in row['points'])
                            coverage['old_point_material_quarantine_candidates']+=sum(bool(v['quarantine_candidate'])for v in row['points'])
                            coverage['old_TOP_contact_rows_unverified_facet']+=int(row.get('facet_quarantine_candidate',False))
                        for row in rows:
                            coverage['generic_RGB_edge_points']+=len(row['observed_generic_RGB_edge_uv'])
                            coverage['outframe_queries']+=len(row['outframe_query_indices'])
                            coverage['missing_or_occluded_queries']+=len(row['missing_or_occluded_query_indices'])
                        report.update(processed_actual_exposures=len(done),coverage=dict(coverage),wall_seconds=time.perf_counter()-start)
                        write(out/'progress.json',dict(shot=shot,phase=phase,current_actual_pts=p,report=report))
                        if len(done)%20==0:print(json.dumps(dict(stage='source_RGB_exposure_reread',shot=shot,
                            phase=phase,actual_pts=p,completed=len(done),expected=len(actual))),flush=True)
                report.update(complete_actual_exposure_coverage=done==set(actual),
                    missing_actual_pts=sorted(set(actual)-done),processed_display_presentations=sum(len(aliases[p])for p in done))
                report['complete_old_source_row_coverage']=coverage['old_source_rows_checked']==report['expected_old_source_rows']
                report['complete_old_TOP_contact_point_coverage']=coverage['old_TOP_contact_points_checked']==report['expected_old_TOP_contact_points']
                # Every displayed presentation retains its verified actual exposure
                # identity; repeated displays do not create new independent points.
                expected_displays=displays if not opt.probe else sorted(q for p in done for q in aliases[p])
                decode_lo=lo if not opt.probe else min(expected_displays)
                decode_hi=hi if not opt.probe else max(expected_displays)+1
                decoded=[]
                with gzip.open(out/(shot+'_all_presentations.jsonl.gz'),'wt',encoding='utf8')as clock_stream:
                    for native,pixels in decoder.presentations(decode_lo,decode_hi):
                        if native not in mapping:raise ValueError('Decoded native PTS lacks registered exposure proof')
                        decoded.append(native)
                        emit(clock_stream,dict(source_pts=native,actual_pts=mapping[native]['canonical'],
                        exposure_measurement_present=int(mapping[native]['canonical'])in done,
                        decoded_RGB_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                        exposure_proof_sha256=clock['proof_sha256'],new_independent_point=False))
                if decoded!=expected_displays:raise ValueError('Native decoded coverage differs from registered actual clock')
                report['all_native_RGB_presentations_decoded_and_clock_verified']=not opt.probe
                write(out/'run_manifest.json',summary)
        summary.update(status='complete_probe'if opt.probe else'complete_source_reread',
            complete_all_requested_shots=all(r['complete_actual_exposure_coverage']for r in summary['shots']),
            wall_seconds=time.perf_counter()-start,accuracy_accepted=False)
        write(out/'run_manifest.json',summary)
    except Exception as exc:
        summary.update(status='failed_preserved',error_type=type(exc).__name__,error=str(exc),wall_seconds=time.perf_counter()-start)
        write(out/'run_manifest.json',summary);raise
    finally:decoder.close()


if __name__=='__main__':main()
