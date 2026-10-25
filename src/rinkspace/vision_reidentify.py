"""Connect accepted source identities to existing RGB matching and temporal guides.

No new detector or world geometry: the existing source advertisement matcher
qualifies identity; its image correspondence only navigates current RGB reads.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from .vision_native import accepted_packet_guides, file_ref


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class AcceptedSourceReidentifier:
    def __init__(self, specification, native_geometry, *, device):
        from .vision_instances import SourceAdvertisementMatcher, SourcePersonBoxes, _add_existing_geometry_dependencies
        _add_existing_geometry_dependencies()
        from .camera_reuse_localization import ReuseConfig
        if native_geometry is None:
            raise ValueError('Reidentification requires the current native scene')
        self.spec = _load(specification)
        self.spec_reference = file_ref(specification)
        self.geometry = native_geometry
        self.last_search = None
        self.interval = Fraction(str(ReuseConfig().keyframe_every_s))
        self.person = SourcePersonBoxes(**self.spec['person'], device=device)
        self.references = {}
        self.ice_descriptors = {}
        cache = Path(self.spec['cache_directory'])
        cache.mkdir(parents=True, exist_ok=True)
        acceptance = _load(self.spec['acceptance'])
        paths = []
        for index in self.spec['reference_indices']:
            name = f'{index:06d}.json'
            path = Path(self.spec['packet_directory'])/name
            binding_path = Path(self.spec['binding_directory'])/name
            packet = _load(path)
            source = packet['source']
            if source['source_video_sha256'] != self.spec['source_sha256']:
                raise ValueError('Reference source differs from programme reidentification scope')
            binding = dict(source_sha256=source['source_video_sha256'], source_pts=packet['source_pts'],
                time_base=source['time_base'], shot_id=self.spec['reference_shot_id'], native_hw=source['native_hw'],
                installation_id=self.spec['event_id'])
            guides, reference = accepted_packet_guides(packet, _load(binding_path), acceptance,
                packet_path=path, binding_path=binding_path, acceptance_path=self.spec['acceptance'], source_binding=binding)
            source_guides = guides
            guides, _ = native_geometry.bind(source_guides)
            ads = {}
            for guide in source_guides:
                match = re.match(r'^Ad(\d+)\.', guide['id'])
                if match:
                    ads.setdefault(int(match[1]), []).extend(guide['points_uv'])
            entries = []
            for number, points in ads.items():
                uv = np.asarray(points)
                low, high = np.floor(uv.min(0)).astype(int), np.ceil(uv.max(0)).astype(int)
                if np.any(high-low <= 8):
                    continue
                entries.append(dict(advertisement_number=number, physical_identity_human_confirmed=True,
                    physical_identity_authority=reference,
                    components=[dict(component_id='accepted_native_paths', observed_kind='printed_pattern',
                        actual_source_navigation_roi=np.r_[low, high].tolist())]))
            if not entries:
                continue
            bank = dict(event_id=self.spec['event_id'], identity_catalog=self.spec['identity_catalog'],
                source=dict(binding, source_frame_path=source['image_path'], source_frame_sha256=source['png_sha256'],
                    decoded_native_rgb_sha256=source['decoded_RGB_sha256']),
                advertisements=entries,
                image_left_to_right_physical_order=[a['advertisement_number'] for a in sorted(entries,
                    key=lambda a:a['components'][0]['actual_source_navigation_roi'][0])])
            bank_path = cache/name
            bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding='utf-8')
            digest = file_ref(bank_path)['sha256']
            self.references[digest] = dict(guides=guides, reference=reference, source_binding=binding,
                source_exclusion_reference=dict(path=source['mask_path'],sha256=source['mask_sha256']))
            paths.append(bank_path)
        from .vision_native_references import current_native_source_banks
        native_paths, native_refs, self.native_bank_receipt = current_native_source_banks(
            native_geometry, cache_directory=cache/'current_native', identity_catalog=self.spec['identity_catalog'],
            event_id=self.spec['event_id'], source_sha256=self.spec['source_sha256'])
        paths.extend(native_paths);self.references.update(native_refs)
        if not paths:
            raise ValueError('No authenticated appearance references in the source bank')
        self.matcher = SourceAdvertisementMatcher(paths, event_id=self.spec['event_id'])

    def exclusions(self, rgb, binding):
        from .vision_source_topology import source_graphic_rectangle_candidates
        mask = np.zeros(rgb.shape[:2], bool)
        people = self.person.read(rgb, binding)
        graphics = source_graphic_rectangle_candidates(rgb, binding)
        # Person rectangles only prompt the current-frame segmenter. Their
        # empty interiors must never veto visible rink or printed material.
        for box in graphics.get('candidates', []):
            x0, y0, x1, y1 = box['box_xyxy']
            mask[max(0,int(np.floor(y0))):min(mask.shape[0],int(np.ceil(y1))),
                 max(0,int(np.floor(x0))):min(mask.shape[1],int(np.ceil(x1)))] = True
        return mask, people, graphics

    def continuous_reference_ice(self,rgb,binding,eligible,contexts,*,flow_engine,excluded):
        """Recover forgotten within-shot guides through the existing RAFT reader."""
        import torch
        from torch.nn import functional as F
        from .vision import _tensor
        from .vision_instances import _add_existing_geometry_dependencies
        _add_existing_geometry_dependencies()
        from .camera_reuse_localization import ReuseConfig
        if flow_engine is None:return [],dict(status='existing_flow_engine_unavailable')
        choices=[]
        for key,physical in eligible.items():
            source=self.references[key]['source_binding']
            delta=(int(binding['source_pts'])-int(source['source_pts']))*Fraction(binding['time_base'])
            if (source['shot_id']==binding['shot_id'] and source['source_sha256']==binding['source_sha256']
                and source['time_base']==binding['time_base'] and 0<delta<=Fraction(str(ReuseConfig().max_lk_gap_s))):
                choices.append((delta,key,physical))
        if not choices:return [],dict(status='no_close_reference_in_this_actual_shot',cross_cut_flow_used=False)
        _,key,physical=min(choices,key=lambda r:r[0]);reference=self.references[key];context=contexts[key]
        pair=flow_engine.compute(_tensor(np.ascontiguousarray(context['image'][:,:,::-1])[None],flow_engine.device)[:,0],
            _tensor(rgb[None],flow_engine.device)[:,0],analysis_hw=(384,640),source0=reference['source_binding'],source1=binding)
        authority=dict(kind='current_native_ice_source_correspondence',source_binding=dict(binding),
            query_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),reference=reference['reference'],
            reference_source_binding=reference['source_binding'],physical_context_authority=physical['identity_authority'],
            current_native_scene=self.geometry.refs[0],same_actual_shot_bidirectional_flow=pair.metadata,
            cross_cut_flow_used=False,reference_UV_used_as_measurement=False,human_query_pixel_acceptance=False)
        guides=[];h,w=rgb.shape[:2]
        for original in reference['guides']:
            if original['kind']!='ice_paint':continue
            points=torch.as_tensor(original['points_uv'],device=flow_engine.device,dtype=torch.float32)
            grid=points*points.new_tensor([2/(w-1),2/(h-1)])-1
            move=F.grid_sample(pair.forward,grid[None,:,None],align_corners=True)[0,:,:,0].T
            valid=F.grid_sample(pair.valid_forward.float(),grid[None,:,None],mode='nearest',align_corners=True)[0,0,:,0]>.5
            target=points+move;xy=torch.round(target).long();inside=(xy>=0).all(1)&(xy<xy.new_tensor([w,h])).all(1)
            valid&=inside
            ix=xy[inside].cpu().numpy();valid[inside]&=torch.as_tensor(~excluded[ix[:,1],ix[:,0]],device=valid.device)
            if int(valid.sum())<2:continue
            guide=dict(original,points_uv=target,source_binding=dict(binding),origin='source_annotation_search_hint',
                identity_authority=authority,identity_confirmed=True,temporal_flow_supported=valid,
                reference_UV_used_as_measurement=False,previous_source_binding=reference['source_binding'])
            if original.get('normals') is not None:
                normal=torch.as_tensor(original['normals'],device=points.device,dtype=points.dtype)
                tangent=torch.stack((normal[:,1],-normal[:,0]),1)
                tip=(points+tangent)*points.new_tensor([2/(w-1),2/(h-1)])-1
                tip_move=F.grid_sample(pair.forward,tip[None,:,None],align_corners=True)[0,:,:,0].T
                moved=tangent+tip_move-move;guide['normals']=torch.stack((-moved[:,1],moved[:,0]),1)
            for field in ('source_exclusion_mask','source_exclusion_binding','accepted_source_packet','confirmed_source_uv'):
                guide.pop(field,None)
            guides.append(guide)
        return guides,dict(status='ice_identity_recovered_from_actual_shot_RGB' if guides else 'ice_same_shot_flow_support_missing',
            guide_count=len(guides),authority=authority,cross_cut_flow_used=False)

    def ice_search(self, rgb, binding, matches, *, prediction, classes, exclusion=None, flow_engine=None):
        """Reuse ice-plane RGB correspondences after physical context matching."""
        import cv2
        from .camera_reuse_localization import descriptors, match_features, ReuseConfig
        from .vision_instances import source_correspondence_identity_verified
        probabilities=prediction['class_probabilities']
        if hasattr(probabilities,'detach'):probabilities=probabilities.detach().cpu().numpy()
        probabilities=np.asarray(probabilities)
        if probabilities.ndim==4:probabilities=probabilities[0]
        from calibration.program_tracking.source_board_measurements import measure_board_source_frame
        from .vision_programme_board_review import conservative_ice_exclusion
        excluded=np.zeros(rgb.shape[:2],bool) if exclusion is None else np.asarray(exclusion,bool)
        bgr=np.ascontiguousarray(rgb[:,:,::-1])
        board=measure_board_source_frame(bgr,excluded,dict(binding,actual_pts=int(binding['source_pts']),
            native_pts=int(binding['source_pts']),decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
        scope=~conservative_ice_exclusion(excluded,board['ice_scope_boundary_candidates'])
        current=descriptors(np.ascontiguousarray(rgb[:,:,::-1]),scope.astype(np.uint8)*255)
        eligible={}
        for row in matches['matches']:
            key=row.get('source_packet_sha256');physical=row.get('physical_id')
            reference=self.references.get(key,{})
            if (physical and any(g['kind']=='ice_paint' for g in reference.get('guides',[]))
                    and source_correspondence_identity_verified(row.get('identity_authority'),binding,physical)):
                if key not in eligible or row['inlier_count']>eligible[key]['inlier_count']:eligible[key]=row
        contexts={c['source_packet_sha256']:c for c in self.matcher.contexts}
        candidates=[];attempts=[]
        for key,physical in eligible.items():
            reference=self.references[key];context=contexts[key]
            ice=[g for g in reference['guides'] if g['kind']=='ice_paint']
            if key not in self.ice_descriptors:
                mask=np.zeros(context['image'].shape[:2],np.uint8)
                for guide in ice:
                    uv=np.asarray(guide['points_uv'],float);segments=np.asarray(guide['segment_ids'])
                    for segment in np.unique(segments):
                        part=np.rint(uv[segments==segment]).astype(np.int32)
                        if len(part)>1:cv2.polylines(mask,[part],False,255,36)
                image_ref=reference.get('source_exclusion_reference')
                if image_ref:
                    from .vision_native import file_ref
                    if file_ref(image_ref['path'])['sha256']!=image_ref['sha256']:raise ValueError('Accepted ice reference mask changed')
                    mask[cv2.imread(image_ref['path'],cv2.IMREAD_GRAYSCALE)>0]=0
                self.ice_descriptors[key]=descriptors(context['image'],mask)
            correspondence=match_features(self.ice_descriptors[key],current,ReuseConfig())
            if correspondence is None:
                attempts.append(dict(reference=key,status='insufficient_unambiguous_ice_RGB_correspondence'));continue
            left,right,H=correspondence
            projected=cv2.perspectiveTransform(left[None].astype(np.float32),H)[0]
            error=float(np.median(np.linalg.norm(projected-right,axis=1)))
            if error>ReuseConfig().max_fit_median_px:continue
            candidates.append((len(right),-error,key,H,left,right,physical))
        if not candidates:
            flow_guides,flow_receipt=self.continuous_reference_ice(rgb,binding,eligible,contexts,flow_engine=flow_engine,excluded=excluded)
            return flow_guides,dict(flow_receipt,attempts=attempts,current_ice_descriptor_count=len(current[0]),model_projection_used=False)
        _,_,key,H,left,right,physical=max(candidates,key=lambda r:(r[0],r[1]))
        reference=self.references[key];hull=cv2.convexHull(left.astype(np.float32))
        authority=dict(kind='current_native_ice_source_correspondence',source_binding=dict(binding),
            query_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),reference=reference['reference'],
            reference_source_binding=reference['source_binding'],physical_context_authority=physical['identity_authority'],
            inlier_reference_uv=left.tolist(),inlier_current_uv=right.tolist(),inlier_count=len(right),
            current_native_scene=self.geometry.refs[0],same_ice_plane_correspondence=True,
            reference_UV_used_as_measurement=False,human_query_pixel_acceptance=False)
        guides=[]
        for original in reference['guides']:
            if original['kind']!='ice_paint':continue
            uv=np.asarray(original['points_uv'],np.float32)
            query=cv2.perspectiveTransform(uv[None],H)[0]
            supported=np.array([cv2.pointPolygonTest(hull,tuple(map(float,p)),True)>=-float(original.get('search_radius_px',18.)) for p in uv])
            supported&=np.isfinite(query).all(1)&(query[:,0]>=0)&(query[:,0]<rgb.shape[1])&(query[:,1]>=0)&(query[:,1]<rgb.shape[0])
            if supported.sum()<2:continue
            ids=np.flatnonzero(supported);old_segments=np.asarray(original['segment_ids'])
            segments=np.cumsum(np.r_[False,(np.diff(ids)!=1)|(old_segments[ids[1:]]!=old_segments[ids[:-1]])])
            guide=dict(original,points_uv=query[ids],segment_ids=segments,source_binding=dict(binding),
                origin='source_annotation_search_hint',identity_confirmed=True,identity_authority=authority,
                reidentification_reference=reference['reference'],reference_UV_used_as_measurement=False,
                historical_world_geometry_imported=False)
            if original.get('normals') is not None:
                normal=np.asarray(original['normals'],np.float32)
                transformed=cv2.perspectiveTransform((uv+normal)[None],H)[0]-query
                transformed/=np.maximum(np.linalg.norm(transformed,axis=1,keepdims=True),1e-9)
                guide['normals']=transformed[ids]
            for field in ('accepted_source_packet','source_exclusion_mask','source_exclusion_binding','confirmed_source_uv'):
                guide.pop(field,None)
            for field in ('original_source_point_indices','original_dependence_groups_per_point'):
                if field in guide:guide[field]=[guide[field][i] for i in ids]
            guides.append(guide)
        return guides,dict(status='ice_identity_reacquired' if guides else 'ice_support_outside_reference',
            guide_count=len(guides),authority=authority,attempts=attempts,model_projection_used=False)

    def accepted_ad_anchors(self,guides,binding):
        from .vision_geometry import _same_frame
        from .vision_instances import HUMAN_IDENTITIES
        grouped={}
        for guide in guides:
            authority=guide.get('identity_authority',{})
            if guide.get('origin')!='user_confirmed_visual_supervision' or authority.get('kind') not in HUMAN_IDENTITIES:continue
            definition=self.geometry.lookup(guide['id'])
            if not definition or definition['layer']!='advertisement':continue
            _same_frame(binding,guide['source_binding'])
            physical=definition['instance_id'].rsplit('/',1)[-1]
            entry=grouped.setdefault(physical,dict(points=[],authority=authority))
            entry['points'].extend(np.asarray(guide['points_uv']).tolist())
        result=[]
        for physical,entry in grouped.items():
            uv=np.asarray(entry['points'],float)
            if min(np.ptp(uv,axis=0))<=8:continue
            result.append(dict(physical_id=physical,source_binding=dict(binding),
                identity_authority=entry['authority'],source_pixel_to_expected_identity_verified=True,
                box_xyxy=np.r_[uv.min(0),uv.max(0)].tolist(),source_pixels_are_new_boundary_measurement=False))
        return result

    def advertisement_guides(self,rgb,binding,matches,*,native_only=False):
        import cv2
        from .vision_instances import source_correspondence_identity_verified,HUMAN_IDENTITIES
        best={};physical_priority={};paths={}
        for row in matches['matches']:
            physical=row.get('physical_id');authority=row.get('identity_authority',{}) or {}
            human=row.get('source_pixel_to_expected_identity_verified') and authority.get('kind') in HUMAN_IDENTITIES and authority.get('reference')
            if not physical or not (human or source_correspondence_identity_verified(authority,binding,physical)):continue
            reference=self.references[row['source_packet_sha256']]
            for original in reference['guides']:
                definition=self.geometry.lookup(original['id'])
                if not definition or definition['layer']!='advertisement':continue
                native=original.get('identity_authority',{}).get('kind')=='current_native_registered_source_correspondence'
                if native_only and not native:continue
                instance=definition['instance_id'].rsplit('/',1)[-1]
                query_id='print_instance/'+physical+'::'+definition['template_path_id']
                target=self.geometry.lookup(query_id)
                if target is None or target.get('template_id')!=definition.get('template_id'):continue
                # Current physical identity may resolve the paired template,
                # but never replace it by another artwork or removed path.
                priority=(native,row['inlier_count'])
                key=(physical,target['id'])
                if key not in paths or priority>paths[key][0]:paths[key]=(priority,row,reference,original,query_id)
                if physical not in best or priority>physical_priority[physical]:
                    best[physical]=row;physical_priority[physical]=priority
        guides=[]
        for _,row,reference,original,query_id in paths.values():
            H=np.asarray(row['H_query_native_from_reference_native'],np.float64)
            uv=np.asarray(original['points_uv'],np.float32);query=cv2.perspectiveTransform(uv[None],H)[0]
            inside=np.isfinite(query).all(1)&(query[:,0]>=0)&(query[:,0]<rgb.shape[1])&(query[:,1]>=0)&(query[:,1]<rgb.shape[0])
            if not inside.any():continue
            ids=np.flatnonzero(inside);segments=np.asarray(original['segment_ids'])
            split=np.cumsum(np.r_[0,(np.diff(ids)!=1)|(segments[ids[1:]]!=segments[ids[:-1]])])
            guide=dict(original,id=query_id,points_uv=query[ids],segment_ids=split,source_binding=dict(binding),
                origin='source_annotation_search_hint',identity_confirmed=True,identity_authority=row['identity_authority'],
                reidentification_reference=reference['reference'],reidentification_previous_source_binding=reference['source_binding'],
                reference_UV_used_as_measurement=False,historical_world_geometry_imported=False,
                closed=bool(original.get('closed') and inside.all()))
            for field in ('accepted_source_packet','normals','source_exclusion_mask','source_exclusion_binding'):
                guide.pop(field,None)
            for field in ('original_source_point_indices','original_dependence_groups_per_point'):
                if field in guide:guide[field]=[guide[field][i] for i in ids]
            guides.append(guide)
        return guides,best

    def search(self, rgb, binding, *, reset=False, person_boxes=(), prediction=None, classes=(), exclusion=None, flow_engine=None):
        import cv2
        from .vision_instances import source_correspondence_identity_verified
        if binding['source_sha256'] != self.spec['source_sha256']:
            raise ValueError('Reidentification reference bank belongs to a different source')
        now = int(binding['source_pts'])*Fraction(binding['time_base'])
        if reset:
            self.last_search = None
        if self.last_search is not None and now-self.last_search < self.interval:
            return [], dict(status='within_shot_RGB_flow_tracking', source_binding=binding)
        self.last_search = now
        matches = self.matcher.read(rgb, dict(binding, event_id=self.spec['event_id']), person_boxes=person_boxes)
        guides,best=self.advertisement_guides(rgb,binding,matches)
        ice_guides,ice_receipt=self.ice_search(rgb,binding,matches,prediction=prediction,classes=classes,exclusion=exclusion,flow_engine=flow_engine)
        guides.extend(ice_guides);matches['ice_reidentification']=ice_receipt
        # Reuse current no-prior source topology for the continuous physical
        # board. Its typed pixels only navigate the paired material rereaders.
        from .vision_instances import automatic_source_region_prompts
        topology = automatic_source_region_prompts(rgb, binding,
            class_probabilities=prediction['class_probabilities'], classes=classes,
            probabilities_conditioned_on_geometry=False, person_boxes=person_boxes)
        for candidate in topology['boundary_topology_hypotheses']:
            if not candidate.get('preferred_region_search_hypothesis'):
                continue
            pixels = np.asarray([(x, y) for y, start, length in candidate['source_gradient_pixel_runs_y_x_length']
                for x in range(start, start+length)], np.float32)
            columns = np.unique(pixels[:,0])[::4]
            points = np.array([[x, np.median(pixels[pixels[:,0]==x,1])] for x in columns], np.float32)
            if len(points)<2:
                continue
            kind = candidate['kind']
            guide = dict(id='board.top' if kind=='board_top' else 'board.bottom', kind=kind,
                primitive='free_curve', physical_role='board_top_edge' if kind=='board_top' else 'ice_board_contact',
                points_uv=points, segment_ids=np.cumsum(np.r_[0,np.diff(columns)>4]), source_binding=binding,
                source_board_view='front_face', origin='source_annotation_search_hint',
                identity_confirmed=True, identity_authority=dict(kind='current_source_paired_board_materials',
                    source_binding=binding, reference=self.spec_reference, typed_pixel_is_measured_boundary=False),
                search_radius_px=18., source_group_id='current_source_paired_board_materials:'+kind,
                reference_UV_used_as_measurement=False)
            bound, _ = self.geometry.bind([guide])
            guides.extend(bound)
        matches['board_search'] = dict(hypothesis_count=len(topology['boundary_topology_hypotheses']),
            paired_candidate_count=len(topology['board_contact_pair_hypotheses']),
            mask_or_typed_edge_used_as_measurement=False)
        matches['current_native_reference_bank']=self.native_bank_receipt
        matches.update(status='source_identity_reacquired' if guides else 'source_identity_unresolved',
            guide_count=len(guides), native_path_ids=[g['native_geometry']['path_id'] for g in guides],
            verified_physical_ids=sorted(best), reference_bank=self.spec_reference,
            query_UV_from_reference=False, world_geometry_modified=False)
        return guides, matches
