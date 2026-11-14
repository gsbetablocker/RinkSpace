"""Single-decode, same-shot material links for source-seeded static identities.

Links are actual two-image RGB evidence, not camera/projection predictions.
Unidentified material nodes do not become observations until a real source
identity anchor reaches them through unambiguous material links. Only compact
readouts and descriptors remain after each image has been released.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .vision_ice_temporal_identity import (
    _current_readout_uncertainty, _measured_material_descriptors)


_LK_TRACKING_METHOD = 'bidirectional_actual_material_support_LK_and_RGB_context'


def _source_flow_raw_reference(reference):
    if (not isinstance(reference, dict) or not reference.get('path')
            or not isinstance(reference.get('sha256'), str) or len(reference['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in reference['sha256'])):
        raise ValueError('External source flow requires an exact raw file reference and SHA256')
    try:
        path = Path(reference['path']).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
    except (OSError, TypeError) as exc:
        raise ValueError('External source flow raw reference is unreadable') from exc
    if digest.hexdigest() != reference['sha256']:
        raise ValueError('External source flow raw reference SHA256 differs')
    return dict(path=str(path), sha256=reference['sha256'])


def _material_link_evidence(error, correlation, flow_receipt=None):
    evidence = dict(source_forward_backward_error_native_px=error, current_material_context_correlation=correlation,
        source_RGB_support_only=True, full_2D_homography_estimated=False,
        model_projection_or_fit_residual_used_for_identity=False)
    if flow_receipt is None:
        evidence.update(source_forward_backward_LK_error_px=error, source_RGB_matching_method=_LK_TRACKING_METHOD)
    else:
        evidence.update(source_RGB_matching_method=flow_receipt['method'],
            source_flow_correspondence_reference=flow_receipt['raw_reference'], source_flow_correspondence=flow_receipt)
    return evidence


def _source_support_flow_predictions(old, support, target, valid, error, previous_mask, current_mask):
    """Apply the same native support gates and normal-only motion to any flow."""
    centers, normals = old['uv'], old['normal']
    a, b = support.reshape(-1, 2), target.reshape(-1, 2)
    good = valid.reshape(-1).copy()
    error = error.reshape(-1)
    good &= np.isfinite(b).all(1) & (error <= 2.)
    for xy, mask in ((a, previous_mask), (b, current_mask)):
        pix = np.rint(np.where(np.isfinite(xy), xy, -1)).astype(int)
        inside = (pix>=0).all(1)&(pix<[mask.shape[1], mask.shape[0]]).all(1)
        good &= inside
        good[inside] &= ~mask[pix[inside, 1], pix[inside, 0]]
    good = good.reshape(-1, 3)
    delta = (b-a).reshape(-1, 3, 2)
    error = error.reshape(-1, 3)
    predictions = centers.copy()
    identified = np.zeros(len(centers), bool)
    errors = np.full(len(centers), np.inf)
    for i in np.flatnonzero(good.any(1)):
        accepted = good[i]
        # Duplicated interface points count once. A material boundary may
        # identify only its source normal; no full 2D transform is estimated.
        if not len(np.unique(support[i, accepted], axis=0)):
            continue
        movement = np.median(delta[i, accepted], axis=0)
        predictions[i] = centers[i]+normals[i]*float(movement@normals[i])
        identified[i] = True
        errors[i] = float(np.median(error[i, accepted]))
    return predictions, identified, errors


def _point_source_flow(previous, current, old, previous_mask, current_mask):
    """Bidirectional LK on measured material supports only; no background H."""
    centers, sides = old['uv'], old['sides']
    if not len(centers):
        return centers.copy(), np.zeros(0, bool), np.zeros(0, float)
    support = np.concatenate([centers[:,None],sides],axis=1).reshape(-1, 1, 2).astype(np.float32)
    forward, valid, _ = cv2.calcOpticalFlowPyrLK(previous, current, support, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    if forward is None:
        return centers.copy(), np.zeros(len(centers), bool), np.full(len(centers), np.inf)
    backward, reverse_valid, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    if backward is None:
        return centers.copy(), np.zeros(len(centers), bool), np.full(len(centers), np.inf)
    a, b = support[:, 0], forward[:, 0]
    error = np.linalg.norm(backward[:, 0]-a, axis=1)
    good = valid.ravel().astype(bool) & reverse_valid.ravel().astype(bool)
    return _source_support_flow_predictions(old, support.reshape(-1, 3, 2), b.reshape(-1, 3, 2),
        good.reshape(-1, 3), error.reshape(-1, 3), previous_mask, current_mask)


def _tracking_method_evidence(links, *, preserved=False):
    methods = sorted({link.get('source_RGB_matching_method', _LK_TRACKING_METHOD) for link in links})
    method = methods[0] if len(methods)==1 else 'mixed_actual_material_support_correspondence_and_RGB_context'
    return dict(source_RGB_tracking_method=('preserved_' if preserved else '')+method,
                source_RGB_tracking_methods=methods)


def _cuda_material_eligibility(old,now,before,after,predicted,device):
    """Source adjacency gates on CUDA; source UV and readouts stay untouched."""
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA source-material matching requested but unavailable')
    def tensor(value):return torch.as_tensor(np.asarray(value),dtype=torch.float64,device=device)
    with torch.inference_mode():
        movement=tensor(now['uv'][after])[:,None]-tensor(predicted[before])[None]
        normal=tensor(old['normal'][before]);newnormal=tensor(now['normal'][after])
        normal_distance=torch.abs(torch.sum(movement*normal[None],dim=-1))
        tangent=torch.stack([-normal[:,1],normal[:,0]],dim=1)
        tangent_distance=torch.abs(torch.sum(movement*tangent[None],dim=-1))
        agreement=torch.abs(newnormal@normal.T)
        # No TF32 or float32 threshold upgrades. Persisted descriptors retain
        # their original compact float16 actual-source readout values.
        correlation=tensor(now['descriptor'][after])@tensor(old['descriptor'][before]).T
        width=tensor(now['width'][after])[:,None];oldwidth=tensor(old['width'][before])[None]
        ratio=width/torch.clamp(oldwidth,min=1e-6)
        interfaces=(width<=1e-6)&(oldwidth<=1e-6)
        eligible=(normal_distance<=6.)&(tangent_distance<=32.)&(agreement>=.95)&(correlation>=.70)
        eligible &= interfaces|((ratio>=.65)&(ratio<=1.55))
        return eligible.cpu().numpy()


def _actual_material_links(previous, current, old, now, previous_mask, current_mask,*,device='cpu',source_flow_prediction=None):
    predicted, flow_good, errors = (_point_source_flow(previous, current, old, previous_mask, current_mask)
        if source_flow_prediction is None else source_flow_prediction)
    edges = []
    for curve in np.intersect1d(old['curve'], now['curve']):
        before = np.flatnonzero((old['curve']==curve)&flow_good)
        after = np.flatnonzero(now['curve']==curve)
        if not len(before) or not len(after):
            continue
        if str(device).startswith('cuda'):
            eligible=_cuda_material_eligibility(old,now,before,after,predicted,device)
        else:
            movement = now['uv'][after, None]-predicted[None, before]
            normal_distance = np.abs(np.einsum('nki,ki->nk', movement, old['normal'][before]))
            tangent = np.c_[-old['normal'][before, 1], old['normal'][before, 0]]
            tangent_distance = np.abs(np.einsum('nki,ki->nk', movement, tangent))
            normal_agreement = np.abs(now['normal'][after]@old['normal'][before].T)
            correlation = now['descriptor'][after].astype(float)@old['descriptor'][before].astype(float).T
            ratio = now['width'][after, None]/np.maximum(old['width'][None, before], 1e-6)
            interfaces = (now['width'][after, None]<=1e-6)&(old['width'][None, before]<=1e-6)
            eligible = (normal_distance<=6.)&(tangent_distance<=32.)&(normal_agreement>=.95)
            eligible &= (correlation>=.70)&(interfaces|((ratio>=.65)&(ratio<=1.55)))
        for local, node in enumerate(after):
            match = np.flatnonzero(eligible[local])
            if not len(match):
                continue
            if str(device).startswith('cuda'):
                matched=before[match];delta=now['uv'][node]-predicted[matched]
                nm=old['normal'][matched];tg=np.c_[-nm[:,1],nm[:,0]]
                corr=old['descriptor'][matched].astype(float)@now['descriptor'][node].astype(float)
                ratio=now['width'][node]/np.maximum(old['width'][matched],1e-6)
                interfaces=(now['width'][node]<=1e-6)&(old['width'][matched]<=1e-6)
                check=(np.abs(np.einsum('ij,ij->i',delta,nm))<=6.)&(np.abs(np.einsum('ij,ij->i',delta,tg))<=32.)
                check &= (np.abs(nm@now['normal'][node])>=.95)&(corr>=.70)&(interfaces|((ratio>=.65)&(ratio<=1.55)))
                match=match[check]
                if not len(match):continue
            # Neighboring stations on the same paint path are not different
            # entities. Competing material offsets at the same station are.
            source_offsets = np.einsum('ij,ij->i', old['uv'][before[match]]-predicted[before[match]], old['normal'][before[match]])
            if np.ptp(source_offsets)>6.:
                continue
            distance = np.linalg.norm(now['uv'][node]-predicted[before[match]], axis=1)
            parent = int(before[match[np.argmin(distance)]])
            corr=float(now['descriptor'][node].astype(float)@old['descriptor'][parent].astype(float))
            edges.append((parent, int(node), float(errors[parent]), corr))
    return edges


class IceSourceMaterialIdentityGraph:
    """Delayed source rows from unique same-material paths, within one shot."""
    def __init__(self, inventory, *, source_sha256, time_base, source_path=None,device='cpu',source_surface='ice'):
        if source_surface not in ('ice', 'board'):
            raise ValueError('Source material graph surface must be ice or board')
        self.inventory = [dict(row) for row in inventory]
        if source_surface == 'board' and any(row['operator'] != 'material_interface' for row in self.inventory):
            raise ValueError('Board material graphs require explicit material interfaces')
        self.source_surface = source_surface
        self.source_sha256, self.time_base, self.source_path = source_sha256, str(time_base),None if source_path is None else str(source_path)
        if device!='cpu' and not str(device).startswith('cuda'):raise ValueError('Source material matching device must be cpu or cuda')
        self.device=device
        self.frames = []
        self.previous_gray = self.previous_mask = None
        self.shot_id = None
        self.catalog = []

    @property
    def source_material_class(self):
        return 'static_board_material' if self.source_surface == 'board' else 'static_painted_ice'

    @staticmethod
    def _scope_mask(legacy, generic):
        if (legacy is None) == (generic is None):
            raise ValueError('Supply exactly one actual source material scope mask')
        return legacy if generic is None else generic

    def source_frame_state(self):
        """Snapshot stream structure without copying immutable pixel/node arrays."""
        return {key: list(value) if isinstance(value, list) else value
                for key, value in self.__dict__.items() if key != '_source_frame_checkpoint'}

    def restore_source_frame_state(self, state, checkpoint=None):
        self.__dict__.clear()
        self.__dict__.update({key: list(value) if isinstance(value, list) else value
                              for key, value in state.items()})
        self._source_frame_checkpoint = checkpoint

    def checkpoint_source_frame(self):
        """Keep the state before this exposure for a same-RGB query revision."""
        self._source_frame_checkpoint = self.source_frame_state()

    def _editable_frame(self, ordinal):
        frame = self.frames[ordinal]
        checkpoint = getattr(self, '_source_frame_checkpoint', None)
        if (checkpoint is not None and ordinal < len(checkpoint['frames'])
                and frame is checkpoint['frames'][ordinal]):
            # A new anchor can label old frames and mark their rows emitted.
            # Copy those mutable records before writing so rollback also
            # retracts the current exposure's backward evidence. Native node
            # arrays are never edited by propagation and remain shared.
            from copy import deepcopy
            frame = {key: deepcopy(value) if isinstance(value, (dict, list, set)) else value
                     for key, value in frame.items()}
            self.frames[ordinal] = frame
        return frame

    def source_support_candidate_indices(self, measurement, exclusion_mask, *, source_ice_scope_mask=None,source_scope_mask=None):
        """Candidate indices needing real blur support before RGB release.

        These are pending source material nodes, not admitted observations.
        Future anchors can reach any of them, so reading only today's admitted
        rows would leave earlier exposures without real gradient blur evidence.
        """
        source_ice_scope_mask=self._scope_mask(source_ice_scope_mask,source_scope_mask)
        mask=np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
        uv=np.asarray(measurement['candidate_uv'],float)
        pixels=np.rint(uv).astype(int)
        inside=(pixels>=0).all(1)&(pixels<[mask.shape[1],mask.shape[0]]).all(1)
        keep=(np.asarray(measurement['candidate_flags'])&3)==0
        keep &= inside
        keep[inside] &= ~mask[pixels[inside,1],pixels[inside,0]]
        curve=np.asarray(measurement['query_curve'])[np.asarray(measurement['candidate_query'])]
        return np.flatnonzero(keep)

    def _nodes(self, rgb, measurement, mask):
        q = np.asarray(measurement['candidate_query'])
        curve = np.asarray(measurement['query_curve'])[q]
        flags = np.asarray(measurement['candidate_flags'])
        uv = np.asarray(measurement['candidate_uv'], float)
        sides = np.asarray(measurement['candidate_sides'], float)
        keep = (flags&3)==0
        pix = np.rint(uv).astype(int)
        bounded = (pix>=0).all(1)&(pix<[mask.shape[1],mask.shape[0]]).all(1)
        keep &= bounded
        keep[bounded] &= ~mask[pix[bounded,1],pix[bounded,0]]
        from .vision_ice_model_context_identity import source_paint_pixel_accuracy_check
        source_hold=[];RGB_field=np.asarray(rgb).astype(np.float32)
        for ci,definition in enumerate(self.inventory):
            if definition['operator']!='paint_center':continue
            picks=np.flatnonzero(keep&(curve==ci))
            if not len(picks):continue
            checked=source_paint_pixel_accuracy_check(rgb,measurement,picks,definition,check_path=False,native_rgb_field=RGB_field)
            held=[r['original_candidate_index'] for r in checked['pending']]
            keep[held]=False;source_hold.extend(held)
        ids = np.flatnonzero(keep)
        operators = np.array([self.inventory[int(curve[i])]['operator'] for i in ids],dtype=str)
        centers = uv[ids].copy()
        painted = operators=='paint_center'
        centers[painted] = sides[ids[painted]].mean(1)
        width = np.linalg.norm(sides[ids,1]-sides[ids,0],axis=1)
        normal = np.asarray(measurement['query_normal'],float)[q[ids]].copy()
        nonzero = width>1e-6
        normal[nonzero] = (sides[ids[nonzero],1]-sides[ids[nonzero],0])/width[nonzero,None]
        descriptor, strength = _measured_material_descriptors(rgb,centers,sides[ids],normal)
        # An unresolved world role still has an actually read source band
        # hypothesis. Store/link it as pending material; never skip the path or
        # promote it into a boundary/center observation until a real role and
        # original source identity are confirmed.
        complete = strength>0
        ids, centers, width, normal, descriptor = ids[complete],centers[complete],width[complete],normal[complete],descriptor[complete]
        quantiles = measurement.get('candidate_side_quantiles_uv')
        result = dict(ids=ids.astype(np.int32),query_ids=q[ids].astype(np.int32),uv=centers,sides=sides[ids],normal=normal,
            source_pixel_accuracy_pending_candidate_ids=source_hold,
            width=width,curve=curve[ids].astype(np.int16),descriptor=descriptor.astype(np.float16),
            raw_uv=uv[ids],label={},parents={},children={},emitted=set())
        if quantiles is not None:
            result['quantiles'] = np.asarray(quantiles,float)[ids]
        return result

    def _source_flow_correspondence(self, correspondence, binding, digest, mask, *, previous_frame=None,
                                    previous_mask=None, allow_empty=False):
        before = previous_frame if previous_frame is not None else (self.frames[-1] if self.frames else None)
        if before is None or (not allow_empty and not len(before['uv'])):
            raise ValueError('External source flow requires previous actual material nodes')
        if before['binding']['shot_id'] != binding['shot_id']:
            raise ValueError('External source flow cannot cross an original source shot')
        fields = {'source_support_uv', 'target_support_uv', 'valid_FB', 'FB_error_native_px',
                  'previous_binding', 'current_binding', 'method', 'raw_reference'}
        if not isinstance(correspondence, dict) or set(correspondence) != fields:
            raise ValueError('External source flow needs the known correspondence fields')
        bindings = {}
        for key, actual, rgb_digest in (('previous_binding', before['binding'], before['RGB_sha256']),
                                        ('current_binding', binding, digest)):
            supplied = correspondence[key]
            expected = dict(source_sha256=self.source_sha256, actual_pts=actual['actual_pts'],
                native_pts=actual['native_pts'], shot_id=actual['shot_id'], time_base=self.time_base,
                decoded_native_RGB_sha256=rgb_digest)
            if not isinstance(supplied, dict) or any(supplied.get(k) != v for k, v in expected.items()):
                raise ValueError('External source flow '+key+' differs from the exact original RGB clock')
            if any(isinstance(supplied[k], (bool, np.bool_)) or not isinstance(supplied[k], (int, np.integer))
                   for k in ('actual_pts', 'native_pts')):
                raise ValueError('External source flow requires actual integer original source PTS')
            bindings[key] = expected
        shape = (len(before['uv']), 3, 2)
        coordinates = {}
        for key in ('source_support_uv', 'target_support_uv'):
            value = np.asarray(correspondence[key])
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError('External source flow '+key+' requires finite Nx3x2 source coordinates')
            coordinates[key] = value.astype(np.float64, copy=False)
        support = np.concatenate([before['uv'][:, None], before['sides']], axis=1)
        if not np.array_equal(coordinates['source_support_uv'], support):
            raise ValueError('External source flow supports differ from previous actual material readouts')
        valid = np.asarray(correspondence['valid_FB'])
        error = np.asarray(correspondence['FB_error_native_px'])
        if valid.shape != shape[:2] or valid.dtype.kind != 'b':
            raise ValueError('External source flow valid_FB requires an Nx3 boolean array')
        if (error.shape != shape[:2] or error.dtype.kind not in 'fiu'
                or not np.isfinite(error).all() or np.any(error < 0)):
            raise ValueError('External source flow FB_error_native_px requires finite nonnegative Nx3 errors')
        method = correspondence['method']
        if not isinstance(method, str) or not method.strip():
            raise ValueError('External source flow requires its actual matching method')
        reference = _source_flow_raw_reference(correspondence['raw_reference'])
        receipt = dict(method=method, raw_reference=reference,
            **bindings, material_nodes=len(before['uv']), support_points_per_node=3,
            source_support_matches_actual_material_readouts=True, flow_coordinates_used_as_source_measurements=False)
        prediction = _source_support_flow_predictions(before, support, coordinates['target_support_uv'],
            valid, error, self.previous_mask if previous_mask is None else previous_mask, mask)
        return prediction, receipt

    def consume(self, rgb, measurement, exclusion_mask, *, source_ice_scope_mask=None,source_scope_mask=None,
                selected_rows=(), source_binding=None, source_flow_correspondence=None):
        """Return newly reached rows; each row carries its own exact binding.

        The caller releases RGB after this call, normalizes delayed rows using
        the cached measurement at their actual PTS, and writes no PNG library.
        Pending graphs are discarded at a cut, never propagated between shots.
        Optional source flow binds both RGB exposures and the previous measured
        center/two-side supports; it navigates matching and supplies no readout.
        """
        binding = dict(source_binding or measurement['source_binding'])
        if getattr(self,'offline_readout_graph',False):
            raise ValueError('Restored no-image graph supports offline source-anchor replay, not new image-flow continuation')
        pts = int(binding.get('actual_pts',binding.get('source_pts')))
        digest = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr = hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest()
        if (measurement['decoded_native_RGB_sha256']!=digest or binding.get('source_sha256',binding.get('video_sha256'))!=self.source_sha256
                or str(binding['time_base'])!=self.time_base):
            raise ValueError('Identity graph requires the exact measured original RGB clock')
        binding.update(actual_pts=pts,native_pts=int(binding.get('native_pts',pts)),
            decoded_native_BGR_sha256=bgr,decoded_BGR_sha256=bgr)
        if self.source_path is not None:binding.setdefault('source_path',str(self.source_path))
        source_ice_scope_mask=self._scope_mask(source_ice_scope_mask,source_scope_mask)
        mask = np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        flow_prediction = flow_receipt = None
        if source_flow_correspondence is not None:
            flow_prediction, flow_receipt = self._source_flow_correspondence(
                source_flow_correspondence, binding, digest, mask)
        if binding['shot_id']!=self.shot_id:
            self.frames=[];self.previous_gray=self.previous_mask=None;self.shot_id=binding['shot_id']
        if self.frames and pts<=self.frames[-1]['pts']:
            raise ValueError('The compact material graph consumes one ordered original stream')
        frame = self._nodes(rgb,measurement,mask)
        frame.update(pts=pts,binding=binding,RGB_sha256=digest,
            candidate_local_source_clearance={int(i):v for i,v in measurement.get('candidate_local_source_clearance',{}).items()
                if int(i) in frame['ids']},
            readout_quantization_step_px=measurement.get('source_readout_quantization_step_px',1.),
            source_interpolation_grid_px=measurement.get('source_interpolation_grid_px'),
            measurement_method=measurement.get('source_measurement_method','actual_native_RGB_profiles'))
        ordinal=len(self.frames)
        frame['ordinal']=ordinal
        if flow_receipt is not None:
            frame['source_flow_correspondence_receipt'] = flow_receipt
        self.frames.append(frame)
        queue=deque()
        if ordinal:
            before=self._editable_frame(ordinal-1)
            options = {} if flow_prediction is None else dict(source_flow_prediction=flow_prediction)
            edges=_actual_material_links(self.previous_gray,gray,before,frame,self.previous_mask,mask,device=self.device,**options)
            for a,b,error,corr in edges:
                evidence = _material_link_evidence(error, corr, flow_receipt)
                frame['parents'][b]=(ordinal-1,a,evidence)
                before['children'].setdefault(a,[]).append((ordinal,b,evidence))
                if a in before['label']:
                    self._label(ordinal,b,before['label'][a],(ordinal-1,a,evidence),'forward',queue)
        selected=selected_rows.get('rows',()) if isinstance(selected_rows,dict) else selected_rows
        by_id={int(candidate):node for node,candidate in enumerate(frame['ids'])}
        for row in selected:
            if row.get('admission_status') not in ('admitted_source_material_center_temporal_identity','admitted_source_physical_boundary_temporal_identity'):
                continue
            for candidate in row.get('original_candidate_indices',row.get('source_candidate_indices',())):
                node=by_id.get(int(candidate))
                if node is None:continue
                proof=dict(physical_material_id=row['physical_material_id'],authority=row['source_seed_authority'],
                    anchor_actual_pts=int(row['source_seed_reference']['actual_pts']),
                    source_seed_reference=row['source_seed_reference'])
                self._label(ordinal,node,proof,None,'anchor',queue)
                frame['emitted'].add(node)  # Already delivered by the selector.
        reached=set()
        while queue:
            fi,node=queue.popleft();current=self.frames[fi];proof=current['label'][node]
            reached.add(fi)
            parent=current['parents'].get(node)
            if parent is not None:
                pfi,pnode,evidence=parent
                self._label(pfi,pnode,proof,(fi,node,evidence),'backward',queue)
            for nfi,nnode,evidence in current['children'].get(node,()):
                self._label(nfi,nnode,proof,(fi,node,evidence),'forward',queue)
        rows=[]
        for fi in sorted(reached):rows.extend(self._new_rows(self.frames[fi]))
        self.previous_gray,self.previous_mask=gray,mask
        catalog=dict(actual_pts=pts,shot_id=binding['shot_id'],decoded_BGR_sha256=bgr,
            decoded_RGB_sha256=digest,actual_material_nodes=int(len(frame['ids'])))
        self.catalog.append(catalog)
        return dict(rows=rows,source_catalog_entry=catalog,visual_review_complete=False,source_material_matching_device=self.device,
            source_material_adjacency_recorded=True,projected_UV_used_as_measurements=False)

    def _relink_source_mask(self, record, frame):
        if not isinstance(record, dict) or set(record) != {'source_binding', 'mask', 'raw_reference'}:
            raise ValueError('Saved source flow needs each original combined source mask and binding')
        actual = frame['binding']
        expected = dict(source_sha256=self.source_sha256, actual_pts=frame['pts'], native_pts=actual['native_pts'],
            shot_id=actual['shot_id'], time_base=self.time_base, decoded_native_RGB_sha256=frame['RGB_sha256'])
        binding = record['source_binding']
        if (not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected.items())
                or any(isinstance(binding[k], (bool, np.bool_)) or not isinstance(binding[k], (int, np.integer))
                       for k in ('actual_pts', 'native_pts'))):
            raise ValueError('Saved source flow mask differs from the original frame binding')
        mask = np.asarray(record['mask'])
        if mask.ndim != 2 or mask.dtype.kind != 'b' or not mask.size:
            raise ValueError('Saved source flow needs an original boolean combined image mask')
        if 'native_hw' in actual and tuple(actual['native_hw']) != mask.shape:
            raise ValueError('Saved source flow mask differs from the original native image grid')
        reference = _source_flow_raw_reference(record['raw_reference'])
        try:
            with np.load(reference['path'], allow_pickle=False) as saved:
                hw = saved['native_hw']
                packed = saved['mask_packbits']
                if (hw.shape != (2,) or hw.dtype.kind not in 'iu' or tuple(hw) != mask.shape
                        or packed.dtype != np.uint8 or packed.shape != ((mask.size+7)//8,)):
                    raise ValueError('Saved source flow mask cache has an invalid native grid or packed mask')
                original = np.unpackbits(packed, bitorder='little', count=mask.size).reshape(mask.shape).astype(bool)
        except (OSError, KeyError) as exc:
            raise ValueError('Saved source flow needs its original packed combined mask cache') from exc
        if not np.array_equal(mask, original):
            raise ValueError('Saved source flow mask differs from its preserved original combined mask')
        return mask, dict(source_binding=expected, raw_reference=reference, native_hw=list(mask.shape),
            combined_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())

    def relink_source_flow_correspondences(self, correspondences, *, frame_masks):
        """Build an independent saved-node graph from bound source flow and masks.

        Readonly source arrays remain shared. Only explicit original anchors
        seed new links; old propagated labels cannot become independent seeds.
        Both inputs are indexed sequences, allowing adjacent masks to stay lazy.
        """
        if not getattr(self, 'offline_readout_graph', False):
            raise ValueError('Source flow relinking requires a loaded compact original material graph')
        if len(correspondences) != max(0, len(self.frames)-1) or len(frame_masks) != len(self.frames):
            raise ValueError('Source flow relinking requires every saved adjacent frame pair and original mask')
        graph = IceSourceMaterialIdentityGraph(self.inventory, source_sha256=self.source_sha256,
            time_base=self.time_base, source_path=self.source_path, device=self.device,source_surface=self.source_surface)
        graph.shot_id = self.shot_id
        graph.catalog = deepcopy(self.catalog)
        graph.offline_readout_graph = True
        graph.compact_state_path = None
        anchors = []
        excluded = {'label', 'parents', 'children', 'emitted', 'preserved_source_row_proofs',
                    'source_flow_correspondence_receipt', 'source_flow_mask_receipt'}
        for fi, original in enumerate(self.frames):
            frame = {key: deepcopy(value) if isinstance(value, (dict, list, set)) else value
                     for key, value in original.items() if key not in excluded}
            frame.update(label={}, parents={}, children={}, emitted=set())
            graph.frames.append(frame)
            for node, proof in original['label'].items():
                if proof['direction'] == 'anchor':
                    anchors.append((fi, node, {k:deepcopy(v) for k,v in proof.items() if k not in ('neighbor', 'direction')}))
        before_mask = None
        old_edges = sum(len(f['parents']) for f in self.frames)
        empty_pairs = new_edges = 0
        methods = set()
        for fi, frame in enumerate(graph.frames):
            mask, mask_receipt = graph._relink_source_mask(frame_masks[fi], frame)
            frame['source_flow_mask_receipt'] = mask_receipt
            if fi:
                before = graph.frames[fi-1]
                if frame['pts'] <= before['pts'] or frame['binding']['shot_id'] != before['binding']['shot_id']:
                    raise ValueError('Saved source flow pairs must be ordered adjacent original frames in one shot')
                prediction, receipt = graph._source_flow_correspondence(correspondences[fi-1],
                    frame['binding'], frame['RGB_sha256'], mask,
                    previous_frame=before, previous_mask=before_mask, allow_empty=True)
                receipt['previous_combined_mask_reference'] = before['source_flow_mask_receipt']
                receipt['current_combined_mask_reference'] = mask_receipt
                methods.add(receipt['method'])
                frame['source_flow_correspondence_receipt'] = receipt
                edges = _actual_material_links(None, None, before, frame, before_mask, mask,
                    device=graph.device, source_flow_prediction=prediction)
                for a,b,error,correlation in edges:
                    evidence = _material_link_evidence(error, correlation, receipt)
                    frame['parents'][b] = (fi-1, a, evidence)
                    before['children'].setdefault(a, []).append((fi, b, evidence))
                new_edges += len(edges)
                empty_pairs += int(not len(before['ids']) or not len(frame['ids']))
            before_mask = mask
        queue = deque()
        for fi,node,proof in anchors:
            graph._label(fi, node, proof, None, 'anchor', queue)
        while queue:
            fi,node = queue.popleft()
            frame = graph.frames[fi]
            proof = frame['label'][node]
            if node in frame['parents']:
                nfi,nnode,evidence = frame['parents'][node]
                graph._label(nfi, nnode, proof, (fi,node,evidence), 'backward', queue)
            for nfi,nnode,evidence in frame['children'].get(node, ()):
                graph._label(nfi, nnode, proof, (fi,node,evidence), 'forward', queue)
        original_path = Path(self.compact_state_path)
        original_digest = hashlib.sha256()
        with original_path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                original_digest.update(block)
        graph.source_flow_relink_receipt = dict(
            original_graph_reference=dict(path=str(original_path), sha256=original_digest.hexdigest()),
            original_actual_source_adjacency_edges=old_edges, actual_source_adjacency_edges=new_edges,
            source_frames=len(graph.frames), adjacent_source_pairs=len(correspondences), zero_node_source_pairs=empty_pairs,
            source_RGB_matching_methods=sorted(methods),
            explicit_original_anchor_nodes=len(anchors), original_confirmed_nodes=sum(len(f['label']) for f in self.frames),
            confirmed_nodes=sum(len(f['label']) for f in graph.frames),
            old_propagated_labels_reused_as_anchors=False, source_RGB_decoded_again=False,
            source_quantiles_remeasured=False, original_node_arrays_reused=True,
            model_projection_used_as_source_pixel_truth=False, visual_review_complete=False)
        return graph

    def relink_changed_source_flow_correspondences(self, baseline, correspondences, *, frame_masks):
        """Relink only pairs touching added nodes; retain exact old actual edges.

        Every baseline readout must be unchanged. The supplied pair mapping
        must equal the adjacent pairs of all added-node frames. Scope may only
        expand while retaining the same authenticated original exclusion guard.
        No old propagated label becomes a new independent source anchor.
        """
        from collections.abc import Mapping
        if (not getattr(self, 'offline_readout_graph', False)
                or not getattr(baseline, 'offline_readout_graph', False)
                or len(self.frames) != len(baseline.frames)
                or len(frame_masks) != len(self.frames)
                or self.inventory != baseline.inventory
                or self.source_surface != baseline.source_surface
                or (self.source_sha256, self.time_base, self.shot_id) !=
                   (baseline.source_sha256, baseline.time_base, baseline.shot_id)):
            raise ValueError('Incremental source flow requires matching loaded source-node and baseline graphs')
        remap = []; changed_frames = []
        fields = ('query_ids', 'uv', 'sides', 'normal', 'width', 'curve', 'descriptor', 'raw_uv', 'quantiles')
        for fi, (current, old) in enumerate(zip(self.frames, baseline.frames)):
            if (current['pts'] != old['pts'] or current['binding'] != old['binding']
                    or current['RGB_sha256'] != old['RGB_sha256']):
                raise ValueError('Incremental source flow cannot change an original source frame binding')
            by_id = {int(value): i for i, value in enumerate(current['ids'])}
            if len(by_id) != len(current['ids']) or any(int(value) not in by_id for value in old['ids']):
                raise ValueError('Incremental source flow must retain every unique original candidate ID')
            locations = np.asarray([by_id[int(value)] for value in old['ids']], int)
            for key in fields:
                if (key in current) != (key in old) or (key in old and
                        not np.array_equal(current[key][locations], old[key], equal_nan=True)):
                    raise ValueError('Incremental source flow changed an original material readout: '+key)
            remap.append(locations)
            if len(current['ids']) != len(old['ids']): changed_frames.append(fi)
        changed_pairs = {pair for fi in changed_frames for pair in (fi-1, fi)
                         if 0 <= pair < len(self.frames)-1}
        if (not isinstance(correspondences, Mapping)
                or any(type(key) is not int for key in correspondences)
                or set(correspondences) != changed_pairs):
            raise ValueError('Changed source flow pairs must exactly match adjacent pairs of all added-node frames')
        graph = IceSourceMaterialIdentityGraph(self.inventory, source_sha256=self.source_sha256,
            time_base=self.time_base, source_path=self.source_path, device=self.device,source_surface=self.source_surface)
        graph.shot_id = self.shot_id; graph.catalog = deepcopy(self.catalog)
        graph.offline_readout_graph = True; graph.compact_state_path = None
        anchors = []
        excluded = {'label', 'parents', 'children', 'emitted', 'preserved_source_row_proofs',
                    'source_flow_correspondence_receipt', 'source_flow_mask_receipt'}
        for fi, original in enumerate(self.frames):
            frame = {key: deepcopy(value) if isinstance(value, (dict, list, set)) else value
                     for key, value in original.items() if key not in excluded}
            frame.update(label={}, parents={}, children={}, emitted=set()); graph.frames.append(frame)
            for node, proof in original['label'].items():
                if proof['direction'] == 'anchor':
                    anchors.append((fi, node, {k:deepcopy(v) for k,v in proof.items() if k not in ('neighbor','direction')}))
        previous_mask = None; actual_edges = reused_edges = expanded_masks = 0; methods = set()
        for fi, frame in enumerate(graph.frames):
            mask, mask_receipt = graph._relink_source_mask(frame_masks[fi], frame)
            prior = baseline.frames[fi].get('source_flow_mask_receipt')
            if prior is None: raise ValueError('Incremental source flow needs the baseline actual mask receipt')
            prior_reference = _source_flow_raw_reference(prior['raw_reference'])
            with np.load(prior_reference['path'], allow_pickle=False) as saved:
                old_mask = np.unpackbits(saved['mask_packbits'], bitorder='little', count=mask.size).reshape(mask.shape).astype(bool)
                old_meta = json.loads(str(saved['metadata_json'].item())) if 'metadata_json' in saved else {}
            if np.any(mask & ~old_mask):
                raise ValueError('Incremental source scope may expand only; old clear support cannot be newly excluded')
            if not np.array_equal(mask, old_mask):
                with np.load(mask_receipt['raw_reference']['path'], allow_pickle=False) as saved:
                    new_meta = json.loads(str(saved['metadata_json'].item())) if 'metadata_json' in saved else {}
                old_guard, new_guard = old_meta.get('original_guard'), new_meta.get('original_guard')
                if (not old_guard or not new_guard or old_guard.get('sha256') != new_guard.get('sha256')
                        or Path(old_guard['path']).resolve() != Path(new_guard['path']).resolve()):
                    raise ValueError('Expanded source scope requires the identical original source exclusion guard')
                guard_reference = _source_flow_raw_reference(old_guard)
                guard_image = cv2.imread(guard_reference['path'], cv2.IMREAD_GRAYSCALE)
                if guard_image is None or guard_image.shape != mask.shape or not mask[guard_image > 0].all():
                    raise ValueError('Expanded source scope cannot release original person or broadcast guard pixels')
                expanded_masks += 1
            frame['source_flow_mask_receipt'] = mask_receipt
            if fi:
                before = graph.frames[fi-1]
                if fi-1 in changed_pairs:
                    prediction, receipt = graph._source_flow_correspondence(correspondences[fi-1],
                        frame['binding'], frame['RGB_sha256'], mask, previous_frame=before,
                        previous_mask=previous_mask, allow_empty=True)
                    receipt['previous_combined_mask_reference'] = before['source_flow_mask_receipt']
                    receipt['current_combined_mask_reference'] = mask_receipt
                    frame['source_flow_correspondence_receipt'] = receipt; methods.add(receipt['method'])
                    edges = _actual_material_links(None, None, before, frame, previous_mask, mask,
                        device=graph.device, source_flow_prediction=prediction)
                    for a,b,error,correlation in edges:
                        evidence = _material_link_evidence(error, correlation, receipt)
                        frame['parents'][b] = (fi-1,a,evidence)
                        before['children'].setdefault(a,[]).append((fi,b,evidence))
                    actual_edges += len(edges)
                else:
                    old_frame = baseline.frames[fi]
                    if 'source_flow_correspondence_receipt' in old_frame:
                        frame['source_flow_correspondence_receipt'] = deepcopy(old_frame['source_flow_correspondence_receipt'])
                    for b,(pfi,a,evidence) in old_frame['parents'].items():
                        if pfi != fi-1: raise ValueError('Baseline actual source edge must connect adjacent frames')
                        pa, cb = int(remap[pfi][a]), int(remap[fi][b]); proof = deepcopy(evidence)
                        frame['parents'][cb] = (pfi,pa,proof)
                        before['children'].setdefault(pa,[]).append((fi,cb,proof))
                        methods.add(proof.get('source_RGB_matching_method',_LK_TRACKING_METHOD)); reused_edges += 1
            previous_mask = mask
        queue = deque()
        for fi,node,proof in anchors: graph._label(fi,node,proof,None,'anchor',queue)
        while queue:
            fi,node=queue.popleft();frame=graph.frames[fi];proof=frame['label'][node]
            if node in frame['parents']:
                nfi,nnode,evidence=frame['parents'][node]
                graph._label(nfi,nnode,proof,(fi,node,evidence),'backward',queue)
            for nfi,nnode,evidence in frame['children'].get(node,()):
                graph._label(nfi,nnode,proof,(fi,node,evidence),'forward',queue)
        def graph_ref(value):
            path=Path(value.compact_state_path)
            return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        graph.source_flow_relink_receipt=dict(original_graph_reference=graph_ref(baseline),
            extended_node_graph_reference=graph_ref(self),source_frames=len(graph.frames),
            original_material_nodes=sum(len(f['ids'])for f in baseline.frames),
            added_material_nodes=sum(len(f['ids'])for f in self.frames)-sum(len(f['ids'])for f in baseline.frames),
            changed_node_frame_ordinals=changed_frames,changed_pair_ordinals=sorted(changed_pairs),
            actual_relinked_source_pairs=len(changed_pairs),reused_source_pairs=max(0,len(self.frames)-1)-len(changed_pairs),
            actual_new_source_edges=actual_edges,reused_original_actual_source_edges=reused_edges,
            actual_source_adjacency_edges=actual_edges+reused_edges,source_scope_expanded_frames=expanded_masks,
            explicit_original_anchor_nodes=len(anchors),confirmed_nodes=sum(len(f['label'])for f in graph.frames),
            source_RGB_matching_methods=sorted(methods),old_propagated_labels_reused_as_anchors=False,
            source_RGB_decoded_again=False,source_quantiles_remeasured=False,old_readouts_exact=True,
            original_exclusion_guards_preserved=True,visual_review_complete=False)
        return graph

    def save_compact_state(self,path):
        """Persist this shot's actual material graph before a cut resets it."""
        path=Path(path)
        if path.exists():raise FileExistsError('Do not overwrite a preserved source material graph')
        path.parent.mkdir(parents=True,exist_ok=True)
        offsets=np.r_[0,np.cumsum([len(f['ids']) for f in self.frames])].astype(np.int64)
        def cat(key,shape,dtype):
            arrays=[np.asarray(f[key],dtype=dtype) for f in self.frames]
            return np.concatenate(arrays) if arrays else np.empty(shape,dtype)
        edges=[];edge_lookup={};edge_evidence_groups=[];edge_evidence_lookup={};edge_evidence_indices=[]
        for fi,frame in enumerate(self.frames):
            for node,(pfi,pnode,evidence) in frame['parents'].items():
                edge_lookup[(pfi,pnode,fi,node)]=len(edges)
                error = evidence.get('source_forward_backward_error_native_px', evidence.get('source_forward_backward_LK_error_px'))
                edges.append([pfi,pnode,fi,node,float(error),
                    float(evidence['current_material_context_correlation'])])
                extra = {k:v for k,v in evidence.items() if k not in (
                    'source_forward_backward_error_native_px', 'source_forward_backward_LK_error_px',
                    'current_material_context_correlation')}
                extra['legacy_LK_error_field'] = 'source_forward_backward_LK_error_px' in evidence
                key = json.dumps(extra, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                if key not in edge_evidence_lookup:
                    edge_evidence_lookup[key] = len(edge_evidence_groups)
                    edge_evidence_groups.append(extra)
                edge_evidence_indices.append(edge_evidence_lookup[key])
        groups=[];lookup={};labels=[]
        for fi,frame in enumerate(self.frames):
            for node,proof in frame['label'].items():
                authority=proof['authority']
                key=(proof['physical_material_id'],authority.get('path'),authority.get('sha256'),proof['anchor_actual_pts'],
                    id(proof['source_seed_reference']))
                if key not in lookup:
                    lookup[key]=len(groups)
                    groups.append({k:v for k,v in proof.items() if k not in ('neighbor','direction')})
                neighbor=proof['neighbor'];edge=-1
                if neighbor is not None:
                    nfi,nnode,_=neighbor
                    edge=edge_lookup[(fi,node,nfi,nnode) if proof['direction']=='backward' else (nfi,nnode,fi,node)]
                labels.append([fi,node,lookup[key],{'anchor':0,'forward':1,'backward':-1}[proof['direction']],edge,
                    int(node in frame['emitted'])])
        quantiles=[np.asarray(f.get('quantiles',np.full((len(f['ids']),2,3,2),np.nan)),float) for f in self.frames]
        frames=[dict(pts=f['pts'],binding=f['binding'],RGB_sha256=f['RGB_sha256'],
            has_quantiles='quantiles' in f,readout_quantization_step_px=f['readout_quantization_step_px'],
            source_interpolation_grid_px=f['source_interpolation_grid_px'],measurement_method=f['measurement_method'],
            candidate_local_source_clearance=f.get('candidate_local_source_clearance',{}),
            source_pixel_accuracy_pending_candidate_ids=f.get('source_pixel_accuracy_pending_candidate_ids',[]),
            source_pixel_accuracy_pending_nodes=sorted(f.get('source_pixel_accuracy_pending_nodes',()))) for f in self.frames]
        for info, frame in zip(frames, self.frames):
            for key in ('source_flow_correspondence_receipt', 'source_flow_mask_receipt'):
                if key in frame:info[key] = frame[key]
        metadata=dict(schema='rinkspace.compact-actual-source-material-adjacency.v1',source_sha256=self.source_sha256,
            time_base=self.time_base,source_path=self.source_path,shot_id=self.shot_id,inventory=self.inventory,
            source_surface=self.source_surface,
            frames=frames,authority_groups=groups,source_material_matching_device=self.device,
            source_RGB_images_saved=False,model_projection_used_as_source_pixel_truth=False,
            model_navigation_context_used_for_identity=any(g['source_seed_reference'].get('source_identity_navigation_evidence') is not None for g in groups),
            actual_source_edge_evidence_groups=edge_evidence_groups,
            visual_review_complete=False)
        if hasattr(self, 'source_flow_relink_receipt'):
            metadata['source_flow_relink_receipt'] = self.source_flow_relink_receipt
        preserved=[dict(frame_ordinal=fi,proof=proof) for fi,f in enumerate(self.frames)
            for proof in f.get('preserved_source_row_proofs',())]
        preserved_utf8=np.frombuffer(json.dumps(preserved,ensure_ascii=False,separators=(',',':')).encode('utf8'),np.uint8)
        np.savez_compressed(path,node_offsets=offsets,original_candidate_indices=cat('ids',(0,),np.int32),
            original_query_indices=cat('query_ids',(0,),np.int32),curve_indices=cat('curve',(0,),np.int16),
            uv=cat('uv',(0,2),np.float64),sides_uv=cat('sides',(0,2,2),np.float64),normals=cat('normal',(0,2),np.float64),
            widths=cat('width',(0,),np.float64),descriptors=cat('descriptor',(0,123),np.float16),raw_candidate_uv=cat('raw_uv',(0,2),np.float64),
            side_quantiles_uv=np.concatenate(quantiles) if quantiles else np.empty((0,2,3,2)),
            actual_source_edges=np.asarray(edges,np.float64).reshape(-1,6),confirmed_node_labels=np.asarray(labels,np.int64).reshape(-1,6),
            actual_source_edge_evidence_group_indices=np.asarray(edge_evidence_indices,np.int32),
            metadata_json=json.dumps(metadata,ensure_ascii=False,separators=(',',':')),
            preserved_source_row_proofs_utf8=preserved_utf8)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if getattr(self, 'offline_readout_graph', False):
            self.compact_state_path = str(path.resolve())
        return dict(path=str(path.resolve()),sha256=digest.hexdigest(),
            source_frames=len(frames),actual_material_nodes=int(offsets[-1]),actual_source_adjacency_edges=len(edges),
            confirmed_nodes=len(labels),visual_review_complete=False)

    @classmethod
    def load_compact_state(cls,path,*,device='cpu',source_surface=None):
        """Restore exact readouts/links for offline later source-anchor replay."""
        with np.load(path,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata_json'].item()))
            if metadata['schema']!='rinkspace.compact-actual-source-material-adjacency.v1':raise ValueError('Unknown source graph schema')
            saved_surface=metadata.get('source_surface','ice')
            if saved_surface not in ('ice','board'):
                raise ValueError('Saved source graph has an unknown physical surface')
            selected_surface=saved_surface if source_surface is None else source_surface
            if selected_surface != saved_surface and (len(saved['confirmed_node_labels'])
                    or metadata.get('authority_groups')
                    or ('preserved_source_row_proofs_utf8' in saved and
                        json.loads(saved['preserved_source_row_proofs_utf8'].tobytes().decode('utf8')))):
                raise ValueError('A source graph with identity proofs cannot change its physical surface')
            graph=cls(metadata['inventory'],source_sha256=metadata['source_sha256'],time_base=metadata['time_base'],
                source_path=metadata['source_path'],device=device,source_surface=selected_surface)
            graph.shot_id=metadata['shot_id']
            if 'source_flow_relink_receipt' in metadata:
                graph.source_flow_relink_receipt = metadata['source_flow_relink_receipt']
            keys={'ids':'original_candidate_indices','query_ids':'original_query_indices','curve':'curve_indices',
                'uv':'uv','sides':'sides_uv','normal':'normals','width':'widths','descriptor':'descriptors','raw_uv':'raw_candidate_uv'}
            # NpzFile does not cache decompressed members. Read each complete
            # field once, then retain read-only frame views of its one array.
            # Re-reading a multi-million-node member per frame is quadratic
            # decompression; copying each slice doubles the loaded pixel data.
            arrays={name:saved[name] for name in keys.values()}
            node_offsets=saved['node_offsets']
            quantiles=saved['side_quantiles_uv']
            edges=saved['actual_source_edges']
            edge_evidence_indices=(saved['actual_source_edge_evidence_group_indices']
                if 'actual_source_edge_evidence_group_indices' in saved.files else None)
            labels=saved['confirmed_node_labels']
            for value in arrays.values():value.flags.writeable=False
            quantiles.flags.writeable=False
            for fi,info in enumerate(metadata['frames']):
                lo,hi=node_offsets[fi:fi+2]
                frame={key:arrays[field][lo:hi] for key,field in keys.items()}
                frame.update(info,ordinal=fi,label={},parents={},children={},emitted=set())
                frame['candidate_local_source_clearance']={int(k):v for k,v in info.get('candidate_local_source_clearance',{}).items()}
                frame['source_pixel_accuracy_pending_nodes']=set(info.get('source_pixel_accuracy_pending_nodes',()))
                if info['has_quantiles']:frame['quantiles']=quantiles[lo:hi]
                graph.frames.append(frame)
                graph.catalog.append(dict(actual_pts=info['pts'],shot_id=info['binding']['shot_id'],
                    decoded_BGR_sha256=info['binding']['decoded_BGR_sha256'],decoded_RGB_sha256=info['RGB_sha256'],
                    actual_material_nodes=int(hi-lo)))
            def evidence(index):
                row=edges[index]
                if edge_evidence_indices is None:
                    return dict(source_forward_backward_LK_error_px=float(row[4]),
                        current_material_context_correlation=float(row[5]),source_RGB_support_only=True,
                        full_2D_homography_estimated=False,model_projection_or_fit_residual_used_for_identity=False)
                proof = dict(metadata['actual_source_edge_evidence_groups'][int(edge_evidence_indices[index])])
                legacy = proof.pop('legacy_LK_error_field')
                proof.update(source_forward_backward_error_native_px=float(row[4]),
                    current_material_context_correlation=float(row[5]))
                if legacy:proof['source_forward_backward_LK_error_px']=float(row[4])
                return proof
            for index,row in enumerate(edges):
                pfi,pnode,fi,node=map(int,row[:4]);proof=evidence(index)
                graph.frames[fi]['parents'][node]=(pfi,pnode,proof)
                graph.frames[pfi]['children'].setdefault(pnode,[]).append((fi,node,proof))
            for row in labels:
                fi,node,group,direction,edge,emitted=map(int,row)
                neighbor=None
                if edge>=0:
                    data=edges[edge];pfi,pnode,cfi,cnode=map(int,data[:4])
                    neighbor=(cfi,cnode,evidence(edge)) if direction==-1 else (pfi,pnode,evidence(edge))
                graph.frames[fi]['label'][node]={**metadata['authority_groups'][group],'neighbor':neighbor,
                    'direction':{0:'anchor',1:'forward',-1:'backward'}[direction]}
                if emitted:graph.frames[fi]['emitted'].add(node)
            if 'preserved_source_row_proofs_utf8' in saved.files:
                proofs=json.loads(saved['preserved_source_row_proofs_utf8'].tobytes().decode('utf8'))
                for record in proofs:
                    graph.frames[int(record['frame_ordinal'])].setdefault('preserved_source_row_proofs',[]).append(record['proof'])
        graph.offline_readout_graph=True
        graph.compact_state_path=str(Path(path).resolve())
        return graph

    def preserve_current_source_row_proofs(self,rows):
        """Opt in to retaining exact selector proof after ``consume``.

        This changes no node admission or identity. The caller supplies only
        current already admitted source rows. Their exact clock, authority,
        temporal evidence and readout domain are verified, then retained as
        lossless minimal evidence beside the existing source arrays.
        """
        from calibration.program_tracking.strict_absolute_source import _temporal_identity
        from calibration.program_tracking.source_identity_navigation import admissible_source_identity_context,GRID_CONTEXT_FIELDS
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        if not self.frames:raise ValueError('Preserve proof only after consuming its current actual source frame')
        frame=self.frames[-1];binding=frame['binding'];by_id={int(v):i for i,v in enumerate(frame['ids'])}
        selected=rows.get('rows',()) if isinstance(rows,dict) else rows;verified=[]
        fields=('actual_pts','native_pts','shot_id','source_sha256','time_base','decoded_native_RGB_sha256',
            'decoded_native_BGR_sha256','curve_id','feature_id','world_curve_role','physical_material_id',
            'static_material_confirmed','actual_source_RGB_measured','pixel_location_confirmed',
            'model_projection_or_fit_residual_used_for_identity','source_seed_authority','source_seed_reference',
            'temporal_evidence','admission_status','center_definition',*GRID_CONTEXT_FIELDS)
        for row in selected:
            if (row.get('actual_pts')!=frame['pts'] or row.get('native_pts')!=binding['native_pts']
                    or row.get('shot_id')!=binding['shot_id'] or row.get('source_sha256')!=self.source_sha256
                    or str(row.get('time_base'))!=self.time_base or row.get('decoded_native_RGB_sha256')!=frame['RGB_sha256']
                    or row.get('decoded_native_BGR_sha256')!=binding['decoded_BGR_sha256']
                    or row.get('actual_source_RGB_measured') is not True or row.get('static_material_confirmed') is not True
                    or row.get('pixel_location_confirmed') is not True or not admissible_source_identity_context(row)
                    or row.get('admission_status') not in ('admitted_source_material_center_temporal_identity','admitted_source_physical_boundary_temporal_identity')
                    or not _temporal_identity(row)):
                raise ValueError('Preserved selector proof requires its exact current admitted source binding and temporal identity')
            authority=row['source_seed_authority'];authenticate_source_reference(authority)
            seed=row['source_seed_reference']
            if (seed.get('curve_id')!=row['curve_id'] or seed.get('source_sha256')!=self.source_sha256
                    or str(seed.get('time_base'))!=self.time_base or seed.get('surface_and_physical_identity_confirmed') is not True
                    or seed.get('source_material_class')!=self.source_material_class
                    or seed.get('authority',{}).get('path')!=authority.get('path') or seed.get('authority',{}).get('sha256')!=authority.get('sha256')):
                raise ValueError('Preserved selector proof requires the authenticated original named paint seed')
            ev=row['temporal_evidence']
            if ev['kind'] in ('same_shot_actual_RGB_temporal_identity','actual_RGB_material_field_order_and_continuity_with_model_navigation',
                    'actual_RGB_material_field_order_and_continuity_with_native_source_guides'):
                old=next((f for f in self.frames if f['pts']==int(ev['previous_actual_pts'])),None)
                if (old is None or old['binding']['shot_id']!=ev['previous_shot_id']
                        or old['binding']['decoded_BGR_sha256']!=ev['previous_decoded_BGR_sha256']):
                    raise ValueError('Preserved selector continuity refers to an unauthenticated prior original frame')
            ids=np.asarray(row['original_candidate_indices'],np.int64)
            if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or any(int(v) not in by_id for v in ids):
                raise ValueError('Preserved selector proof needs existing unique exact material-node locators')
            nodes=np.asarray([by_id[int(v)] for v in ids],int)
            if any(self.inventory[int(frame['curve'][i])]['curve_id']!=row['curve_id'] for i in nodes):
                raise ValueError('Preserved selector proof cannot rename another source-material curve')
            uv=np.asarray(row['uv'],np.float64);sides=np.asarray(row['source_sides_uv'],np.float64)
            raw=np.array_equal(uv,frame['uv'][nodes]) and np.array_equal(sides,frame['sides'][nodes])
            q50=frame.get('quantiles',np.empty((0,2,3,2)));fine=False
            if len(q50):
                fine_sides=q50[nodes,:,1]
                fine=np.isfinite(fine_sides).all() and np.array_equal(sides,fine_sides) and np.array_equal(uv,fine_sides.mean(1))
            if not (raw or fine):raise ValueError('Preserved selector UV and material sides differ from exact source readouts')
            proof={name:deepcopy(row[name]) for name in fields if name in row}
            proof.update(original_candidate_indices=ids.tolist(),source_query_indices=frame['query_ids'][nodes].tolist(),
                source_readout_coordinate_domain='actual_gradient_q50_float64' if fine else 'raw_saved_candidate_readout',
                source_readout_uv_sha256=hashlib.sha256(uv.tobytes()).hexdigest(),
                source_readout_sides_sha256=hashlib.sha256(sides.tobytes()).hexdigest())
            verified.append(proof)
        frame.setdefault('preserved_source_row_proofs',[]).extend(verified)
        return dict(actual_pts=frame['pts'],preserved_current_source_rows=len(verified),node_admission_changed=False)

    def export_confirmed_source_rows(self,*,source_graph_reference=None,stream=False):
        """Recover proven saved rows after a writer failure, without new RGB.

        A saved selector continuation was historically labelled ``anchor``
        without retaining its original temporal proof. Such a label alone is
        insufficient. Export starts at hash-bound same-exposure physical seeds
        and follows only saved actual material edges through already confirmed
        nodes. Disconnected labels remain pending. ``stream=True`` returns an
        iterator of frame-sized rows instead of another large JSON scalar.
        """
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        if getattr(self,'offline_readout_graph',False):
            if (not source_graph_reference or str(Path(source_graph_reference['path']).resolve())!=self.compact_state_path):
                raise ValueError('Recovered source rows need the preserved graph SHA reference')
            authenticate_source_reference(source_graph_reference)
        offsets=np.r_[0,np.cumsum([len(f['ids']) for f in self.frames])].astype(np.int64)
        total=int(offsets[-1]);origin=np.full(total,-1,np.int64);previous=np.full(total,-1,np.int64)
        queue=deque();source_origins={};pending=[];validated={}
        def validate(proof):
            seed=proof['source_seed_reference'];authority=proof['authority']
            key=(proof['physical_material_id'],authority.get('path'),authority.get('sha256'),id(seed))
            if key not in validated:
                authenticate_source_reference(authority)
                valid=(seed.get('curve_id')==proof['physical_material_id']
                    and seed.get('source_sha256')==self.source_sha256 and str(seed.get('time_base'))==self.time_base
                    and seed.get('surface_and_physical_identity_confirmed') is True
                    and seed.get('source_material_class')==self.source_material_class
                    and seed.get('authority',{}).get('path')==authority.get('path')
                    and seed.get('authority',{}).get('sha256')==authority.get('sha256'))
                validated[key]=valid
            return validated[key]
        for fi,frame in enumerate(self.frames):
            for node,proof in frame['label'].items():
                # A propagated node can return to its seed exposure through
                # real adjacent-frame edges; it is still not an original root.
                if proof['direction']!='anchor':continue
                ci=int(frame['curve'][node]);definition=self.inventory[ci];seed=proof['source_seed_reference']
                if node in frame.get('source_pixel_accuracy_pending_nodes',()):continue
                # A grid context seed is a two-exposure source proof, not a
                # direct human anchor. Recover its original proof separately.
                if seed.get('source_identity_navigation_evidence') is not None:continue
                if (definition['curve_id']!=proof['physical_material_id'] or definition['operator']=='unresolved' or not validate(proof)):
                    continue
                if (int(seed.get('actual_pts',-1))!=frame['pts'] or seed.get('shot_id')!=frame['binding']['shot_id']
                        or int(seed.get('native_pts',-1))!=frame['binding']['native_pts']
                        or seed.get('decoded_native_RGB_sha256')!=frame['RGB_sha256']):
                    continue
                flat=int(offsets[fi]+node);origin[flat]=flat;queue.append((fi,int(node)))
                source_origins[flat]=proof
        # Future compact graphs can preserve the selector's exact original
        # temporal proof, independently of the smaller graph's adjacency.
        # Old graphs have no such member and keep the same pending outcome.
        from calibration.program_tracking.strict_absolute_source import _temporal_identity
        frame_lookup={f['pts']:f for f in self.frames}
        preserved_count=0
        for fi,frame in enumerate(self.frames):
            by_id={int(value):i for i,value in enumerate(frame['ids'])}
            for source_row in frame.get('preserved_source_row_proofs',()):
                binding=frame['binding'];authority=source_row['source_seed_authority']
                if (source_row.get('actual_pts')!=frame['pts'] or source_row.get('native_pts')!=binding['native_pts']
                        or source_row.get('shot_id')!=binding['shot_id'] or source_row.get('source_sha256')!=self.source_sha256
                        or str(source_row.get('time_base'))!=self.time_base
                        or source_row.get('decoded_native_RGB_sha256')!=frame['RGB_sha256']
                        or source_row.get('decoded_native_BGR_sha256')!=binding['decoded_BGR_sha256']
                        or source_row.get('actual_source_RGB_measured') is not True
                        or source_row.get('static_material_confirmed') is not True
                        or source_row.get('pixel_location_confirmed') is not True
                        or source_row.get('admission_status') not in ('admitted_source_material_center_temporal_identity','admitted_source_physical_boundary_temporal_identity')
                        or not _temporal_identity(source_row)):
                    raise ValueError('Preserved original temporal proof differs from its exact saved source binding')
                seed=source_row['source_seed_reference']
                proof=dict(physical_material_id=source_row['physical_material_id'],authority=authority,
                    source_seed_reference=seed,anchor_actual_pts=int(seed['actual_pts']))
                if not validate(proof):raise ValueError('Preserved original temporal proof lacks its authenticated source identity')
                ev=source_row['temporal_evidence']
                if ev['kind']=='same_shot_actual_RGB_temporal_identity':
                    previous_frame=frame_lookup.get(int(ev['previous_actual_pts']))
                    if (previous_frame is None or previous_frame['binding']['shot_id']!=ev['previous_shot_id']
                            or previous_frame['binding']['decoded_BGR_sha256']!=ev['previous_decoded_BGR_sha256']):
                        raise ValueError('Preserved original temporal proof refers to another previous source frame')
                candidates=np.asarray(source_row['original_candidate_indices'],np.int64)
                if len(np.unique(candidates))!=len(candidates) or any(int(v) not in by_id for v in candidates):
                    raise ValueError('Preserved original temporal proof has no exact saved candidate locator')
                nodes=np.asarray([by_id[int(v)] for v in candidates],int)
                if not np.array_equal(frame['query_ids'][nodes],source_row['source_query_indices']):
                    raise ValueError('Preserved original temporal proof source query locators changed')
                domain=source_row['source_readout_coordinate_domain']
                if domain=='actual_gradient_q50_float64':
                    if 'quantiles' not in frame:raise ValueError('Preserved actual q50 readout no longer exists')
                    read_sides=frame['quantiles'][nodes,:,1,:];read_uv=read_sides.mean(1)
                elif domain=='raw_saved_candidate_readout':
                    read_sides=frame['sides'][nodes];read_uv=frame['uv'][nodes]
                else:raise ValueError('Unknown preserved actual-source readout domain')
                if (not np.isfinite(read_uv).all() or not np.isfinite(read_sides).all()
                        or hashlib.sha256(np.ascontiguousarray(read_uv,np.float64).tobytes()).hexdigest()!=source_row['source_readout_uv_sha256']
                        or hashlib.sha256(np.ascontiguousarray(read_sides,np.float64).tobytes()).hexdigest()!=source_row['source_readout_sides_sha256']):
                    raise ValueError('Preserved original temporal proof actual UV or sides changed')
                proof['preserved_original_source_row_proof']=source_row
                proof['preserved_original_source_row_proof_sha256']=hashlib.sha256(json.dumps(source_row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                for node in nodes:
                    if int(node) in frame.get('source_pixel_accuracy_pending_nodes',()):continue
                    label=frame['label'].get(int(node))
                    if (label is None or label['physical_material_id']!=source_row['curve_id']
                            or self.inventory[int(frame['curve'][node])]['curve_id']!=source_row['curve_id']
                            or label['authority']['path']!=authority['path'] or label['authority']['sha256']!=authority['sha256']):
                        raise ValueError('Preserved temporal proof cannot create or rename a source-material node')
                    flat=int(offsets[fi]+node);origin[flat]=flat;previous[flat]=-1
                    source_origins[flat]=proof;queue.append((fi,int(node)));preserved_count+=1
        while queue:
            fi,node=queue.popleft();frame=self.frames[fi];flat=int(offsets[fi]+node)
            root=source_origins[int(origin[flat])]
            neighbor=[]
            if node in frame['parents']:
                nfi,nnode,evidence=frame['parents'][node];neighbor.append((nfi,nnode))
            neighbor.extend((nfi,nnode) for nfi,nnode,evidence in frame['children'].get(node,()))
            for nfi,nnode in neighbor:
                nflat=int(offsets[nfi]+nnode)
                if origin[nflat]>=0:continue
                target=self.frames[nfi];label=target['label'].get(nnode)
                if nnode in target.get('source_pixel_accuracy_pending_nodes',()):continue
                if (label is None or label['physical_material_id']!=root['physical_material_id']
                        or target['binding']['shot_id']!=frame['binding']['shot_id']
                        or self.inventory[int(target['curve'][nnode])]['curve_id']!=root['physical_material_id']):
                    continue
                origin[nflat]=origin[flat];previous[nflat]=flat;queue.append((nfi,int(nnode)))
        confirmed=sum(len(f['label']) for f in self.frames);reachable=int(np.count_nonzero(origin>=0))
        for fi,frame in enumerate(self.frames):
            lost=[node for node in frame['label'] if origin[int(offsets[fi]+node)]<0]
            if lost:pending.append(dict(actual_pts=frame['pts'],confirmed_labels_without_recoverable_source_proof=len(lost),
                status='saved_anchor_temporal_fields_missing_and_no_authenticated_seed_material_path'))

        def recovered_rows():
            for fi,frame in enumerate(self.frames):
                groups={};group_proofs={}
                for node in frame['label']:
                    flat=int(offsets[fi]+node);root=int(origin[flat])
                    if root<0:continue
                    proof=source_origins[root]
                    root_group=(proof['physical_material_id'],proof['authority']['path'],proof['authority']['sha256'],proof['anchor_actual_pts'],
                        proof.get('preserved_original_source_row_proof_sha256'))
                    group_proofs[root_group]=proof
                    neighbor=int(previous[flat]);nfi=-1 if neighbor<0 else int(np.searchsorted(offsets,neighbor,side='right')-1)
                    direction='anchor' if neighbor<0 else ('forward' if self.frames[nfi]['pts']<frame['pts'] else 'backward')
                    groups.setdefault((int(frame['curve'][node]),root_group,nfi,direction),[]).append(int(node))
                for (ci,root,nfi,direction),nodes in groups.items():
                    if len(nodes)<2:continue
                    proof=group_proofs[root];definition=self.inventory[ci];idx=np.asarray(nodes,int)
                    evidence=dict(kind='confirmed_original_source_identity_anchor' if direction=='anchor' else 'same_shot_actual_RGB_temporal_identity',
                        physical_material_id=definition['curve_id'],shot_id=frame['binding']['shot_id'],current_actual_pts=frame['pts'],
                        same_physical_entity_confirmed=True,current_frame_RGB_measured=True,
                        model_projection_or_fit_residual_used_for_identity=False,source_identity_authority=proof['authority'])
                    original_proof=proof.get('preserved_original_source_row_proof') if direction=='anchor' else None
                    if original_proof is not None:
                        evidence=deepcopy(original_proof['temporal_evidence'])
                    elif direction=='anchor':
                        evidence['source_identity_confirmed']=True
                    else:
                        neighbor=self.frames[nfi];links=[]
                        for node in nodes:
                            pflat=int(previous[int(offsets[fi]+node)]);pnode=pflat-int(offsets[nfi])
                            link=(frame['parents'][node][2] if direction=='forward' else neighbor['parents'][pnode][2])
                            links.append(link)
                        evidence.update(direction=direction,previous_shot_id=neighbor['binding']['shot_id'],previous_actual_pts=neighbor['pts'],
                            previous_decoded_BGR_sha256=neighbor['binding']['decoded_BGR_sha256'],previous_frame_RGB_measured=True,
                            source_RGB_tracking_confirmed=True,current_feature_semantics_remeasured=True,
                            actual_material_adjacency_evidence=links, **_tracking_method_evidence(links, preserved=True))
                    row=dict(actual_pts=frame['pts'],native_pts=frame['binding']['native_pts'],source_binding=dict(frame['binding']),
                        source_sha256=self.source_sha256,time_base=self.time_base,decoded_native_RGB_sha256=frame['RGB_sha256'],
                        decoded_native_BGR_sha256=frame['binding']['decoded_BGR_sha256'],shot_id=frame['binding']['shot_id'],
                        curve_id=definition['curve_id'],feature_id=definition['feature_id'],world_curve_role=definition['world_curve_role'],
                        physical_material_id=definition['curve_id'],static_material_confirmed=True,actual_source_RGB_measured=True,
                        pixel_location_confirmed=True,model_projection_or_fit_residual_used_for_identity=False,
                        original_candidate_indices=frame['ids'][idx].tolist(),source_candidate_indices=frame['ids'][idx].tolist(),
                        source_query_indices=frame['query_ids'][idx].tolist(),uv=frame['uv'][idx].tolist(),source_sides_uv=frame['sides'][idx].tolist(),
                        candidate_sides=frame['sides'][idx].tolist(),raw_candidate_uv=frame['raw_uv'][idx].tolist(),
                        actual_source_normal_uv=frame['normal'][idx].tolist(),
                        source_readout_quantization_step_px=frame['readout_quantization_step_px'],
                        source_interpolation_grid_px=frame['source_interpolation_grid_px'],
                        source_measurement_method=frame['measurement_method'],
                        center_definition='midpoint_of_two_current_actual_RGB_material_interfaces' if definition['operator']=='paint_center' else None,
                        source_seed_authority=proof['authority'],source_seed_reference=proof['source_seed_reference'],temporal_evidence=evidence,
                        admission_status='admitted_source_material_center_temporal_identity' if definition['operator']=='paint_center' else 'admitted_source_physical_boundary_temporal_identity',
                        model_projection_used_as_measurement=False,source_center_smoothed_or_interpolated=False,
                        recovered_from_preserved_actual_material_graph=source_graph_reference,
                        recovered_readout_coordinate_domain='raw_saved_candidate_readout')
                    if proof['source_seed_reference'].get('source_identity_navigation_evidence') is not None:
                        from .vision_ice_model_context_identity import attach_grid_context_evidence
                        attach_grid_context_evidence(row,proof['source_seed_reference'],source_binding=dict(source_sha256=self.source_sha256,
                            time_base=self.time_base,actual_pts=frame['pts'],native_pts=frame['binding']['native_pts'],
                            decoded_native_BGR_sha256=frame['binding']['decoded_BGR_sha256'],shot_id=frame['binding']['shot_id']),
                            source_motion=row['temporal_evidence'].get('actual_material_adjacency_evidence'))
                    if original_proof is not None:
                        row['recovered_original_temporal_proof_without_reconstruction']=True
                        row['recovered_readout_coordinate_domain']=original_proof['source_readout_coordinate_domain']
                        if original_proof['source_readout_coordinate_domain']=='actual_gradient_q50_float64':
                            current_sides=frame['quantiles'][idx,:,1,:]
                            row.update(uv=current_sides.mean(1).tolist(),source_sides_uv=current_sides.tolist(),candidate_sides=current_sides.tolist())
                    if 'quantiles' in frame:
                        row['source_side_quantiles_uv']=frame['quantiles'][idx].tolist()
                        m=dict(candidate_side_quantiles_uv=frame['quantiles'],source_readout_quantization_step_px=frame['readout_quantization_step_px'],
                            source_interpolation_grid_px=frame['source_interpolation_grid_px'],source_measurement_method=frame['measurement_method'])
                        uncertainty=_current_readout_uncertainty(m,idx,frame['uv'][idx],frame['normal'][idx])
                        if uncertainty is not None:
                            row['source_readout_uncertainty']=uncertainty
                            row['sigma_px']=np.max(np.abs(uncertainty['normal_interval_px']),axis=1).tolist()
                            row['sigma_semantics']='actual_source_readout_support_half_extent_not_calibrated_Gaussian_covariance'
                    clear=frame.get('candidate_local_source_clearance',{})
                    approved={int(i):clear[int(i)] for i in frame['ids'][idx] if int(i) in clear}
                    if approved:
                        row['candidate_local_source_clearance']=approved
                        row['original_source_exclusion_flag_was_cleared_only_by_explicit_local_material_proof']=True
                    if self.source_surface == 'board':
                        row['scope']='board'
                    yield row
        rows=recovered_rows()
        return dict(rows=rows if stream else list(rows),pending=pending,confirmed_saved_nodes=confirmed,
            source_proof_reachable_nodes=reachable,recoverable_source_seed_nodes=len(source_origins),
            preserved_original_temporal_proof_nodes=preserved_count,
            source_graph_reference=source_graph_reference,source_RGB_decoded_again=False,
            source_material_matching_repeated=False,source_quantiles_remeasured=False,visual_review_complete=False)

    def apply_confirmed_source_rows(self,rows,*,source_scope_review=None):
        """Apply genuinely reviewed anchor rows to saved actual adjacency.

        A new trace/navigation hint is insufficient: rows need exact existing
        candidate locators, current RGB hash and admitted source identity.
        No video decoding, predicted UV or new material observations occur.
        """
        from calibration.program_tracking.strict_absolute_source import _temporal_identity
        from calibration.program_tracking.source_identity_navigation import admissible_source_identity_context
        from calibration.program_tracking.strict_model_scope import (ExactSourceScopeReview,
            authenticate_source_reference,approved_source_pixel_indices)
        if not isinstance(source_scope_review,ExactSourceScopeReview) or not source_scope_review.reference:
            raise ValueError('An offline navigation hint needs a saved exact-source scope review before becoming an anchor')
        if (source_scope_review.source.get('video_sha256')!=self.source_sha256
                or str(source_scope_review.source.get('time_base'))!=self.time_base):
            raise ValueError('Offline anchor scope review belongs to another original source clock')
        authenticate_source_reference(source_scope_review.reference);source_scope_review.verify_evidence_unchanged()
        lookup={f['pts']:fi for fi,f in enumerate(self.frames)};queue=deque();validated=[]
        for row in rows:
            fi=lookup.get(int(row['actual_pts']))
            if fi is None:raise ValueError('Source anchor frame not present in saved graph')
            frame=self.frames[fi]
            if (row.get('decoded_native_RGB_sha256')!=frame['RGB_sha256'] or row.get('actual_source_RGB_measured') is not True
                    or not admissible_source_identity_context(row)
                    or row.get('admission_status') not in ('admitted_source_material_center_temporal_identity','admitted_source_physical_boundary_temporal_identity')):
                raise ValueError('An offline navigation hint cannot become a confirmed source identity')
            binding=frame['binding']
            if (row.get('native_pts')!=binding['native_pts'] or row.get('shot_id')!=binding['shot_id']
                    or row.get('source_sha256')!=self.source_sha256 or str(row.get('time_base'))!=self.time_base
                    or row.get('decoded_native_BGR_sha256')!=binding['decoded_BGR_sha256']
                    or row.get('static_material_confirmed') is not True or row.get('pixel_location_confirmed') is not True):
                raise ValueError('Offline anchor differs from the exact saved original source binding')
            if not _temporal_identity(row):raise ValueError('Offline anchor lacks strict actual-source temporal identity evidence')
            authority=row['source_seed_authority'];authenticate_source_reference(authority)
            seed=row['source_seed_reference']
            if self.source_surface == 'board' and seed.get('source_material_class') != self.source_material_class:
                raise ValueError('A board source anchor requires an explicit static board material seed')
            if (seed.get('curve_id')!=row['curve_id'] or seed.get('surface_and_physical_identity_confirmed') is not True
                    or seed.get('source_sha256')!=self.source_sha256 or str(seed.get('time_base'))!=self.time_base
                    or seed.get('authority',{}).get('path')!=authority.get('path')
                    or seed.get('authority',{}).get('sha256')!=authority.get('sha256')):
                raise ValueError('Offline anchor requires the authenticated original physical identity seed')
            temporal=row['temporal_evidence']
            if temporal['kind'] in ('same_shot_actual_RGB_temporal_identity','actual_RGB_material_field_order_and_continuity_with_model_navigation',
                    'actual_RGB_material_field_order_and_continuity_with_native_source_guides'):
                previous=lookup.get(int(temporal['previous_actual_pts']))
                if (previous is None or self.frames[previous]['binding']['shot_id']!=binding['shot_id']
                        or self.frames[previous]['binding']['decoded_BGR_sha256']!=temporal['previous_decoded_BGR_sha256']):
                    raise ValueError('Offline temporal anchor references an unauthenticated previous source frame')
            by_id={int(v):i for i,v in enumerate(frame['ids'])}
            candidates=np.asarray(row['original_candidate_indices'],np.int64)
            if candidates.ndim!=1 or len(np.unique(candidates))!=len(candidates):raise ValueError('Unique saved source candidate locators required')
            if any(int(v) not in by_id for v in candidates):raise ValueError('Source anchor locator not an actual saved material node')
            nodes=np.asarray([by_id[int(v)] for v in candidates],int)
            ci=self.inventory.index(next(v for v in self.inventory if v['curve_id']==row['curve_id']))
            if row.get('physical_material_id')!=row['curve_id'] or not np.all(frame['curve'][nodes]==ci):
                raise ValueError('Offline anchor cannot rename another saved source-material path')
            sides=np.asarray(row.get('source_sides_uv',row.get('candidate_sides',[])),np.float64)
            measured_uv=np.asarray(row.get('uv',[]),np.float64)
            raw_domain=np.array_equal(sides,frame['sides'][nodes]) and np.array_equal(measured_uv,frame['uv'][nodes])
            q50_domain=False
            if 'quantiles' in frame:
                q50=frame['quantiles'][nodes,:,1,:]
                if np.isfinite(q50).all():
                    q50_domain=np.array_equal(sides,q50) and np.array_equal(measured_uv,q50.mean(1))
            if not (raw_domain or q50_domain):
                raise ValueError('Offline anchor UV and actual material sides differ from saved readouts')
            approved=approved_source_pixel_indices(frame['pts'],measured_uv,source_scope_review,
                physical_material_id=row['physical_material_id'])
            if not np.array_equal(approved,np.arange(len(nodes))):
                raise ValueError('Every offline source-anchor pixel requires its own exact static-material approval')
            for uv in measured_uv:
                endpoint=source_scope_review.endpoint(frame['pts'],uv)
                if (endpoint.get('scope')!=self.source_surface or endpoint.get('native_pts')!=binding['native_pts']
                        or endpoint.get('decoded_BGR_sha256')!=binding['decoded_BGR_sha256']):
                    raise ValueError('Offline exact-pixel scope approval differs from the saved native source binding')
            if 'source_query_indices' in row and not np.array_equal(np.asarray(row['source_query_indices']),frame['query_ids'][nodes]):
                raise ValueError('Offline anchor query locators differ from saved source profiles')
            proof=dict(physical_material_id=row['physical_material_id'],authority=row['source_seed_authority'],
                anchor_actual_pts=int(row['source_seed_reference']['actual_pts']),source_seed_reference=row['source_seed_reference'],
                anchor_source_readout_coordinate_domain='actual_gradient_q50_float64' if q50_domain else 'raw_saved_candidate_readout')
            if self.inventory[ci]['operator']=='unresolved':raise ValueError('Unresolved saved material role cannot become an offline anchor')
            validated.append((fi,nodes,proof))
        # Validate all rows before any node is labelled. One forged readout or
        # changed authority must not partially promote a source graph.
        for fi,nodes,proof in validated:
            frame=self.frames[fi]
            for node in nodes:
                self._label(fi,int(node),proof,None,'anchor',queue)
                frame['emitted'].add(node)
        reached=set()
        while queue:
            fi,node=queue.popleft();current=self.frames[fi];proof=current['label'][node];reached.add(fi)
            parent=current['parents'].get(node)
            if parent is not None:
                pfi,pnode,evidence=parent;self._label(pfi,pnode,proof,(fi,node,evidence),'backward',queue)
            for nfi,nnode,evidence in current['children'].get(node,()):
                self._label(nfi,nnode,proof,(fi,node,evidence),'forward',queue)
        return dict(rows=[row for fi in sorted(reached) for row in self._new_rows(self.frames[fi])],
            source_RGB_decoded_again=False,projected_UV_used_as_measurements=False,visual_review_complete=False)

    def _label(self,fi,node,proof,neighbor,direction,queue):
        frame=self._editable_frame(fi)
        if node in frame['label']:
            if frame['label'][node]['physical_material_id']!=proof['physical_material_id']:
                raise ValueError('One original source material node reached competing physical identities')
            return
        curve=self.inventory[int(frame['curve'][node])]
        if curve['operator']=='unresolved':
            raise ValueError('Pending source band hypothesis needs confirmed effective material operator before identity admission')
        if curve['curve_id']!=proof['physical_material_id']:
            raise ValueError('Measured candidate search path cannot rename physical identity')
        frame['label'][node]={**proof,'neighbor':neighbor,'direction':direction}
        queue.append((fi,node))

    def _new_rows(self,frame):
        if getattr(self, '_source_frame_checkpoint', None) is not None:
            frame = self._editable_frame(frame['ordinal'])
        groups={}
        for node,proof in frame['label'].items():
            if node in frame['emitted'] or proof['direction']=='anchor':continue
            neighbor=proof['neighbor']
            groups.setdefault((int(frame['curve'][node]),proof['direction'],neighbor[0]),[]).append(node)
        rows=[]
        for (ci,direction,neighbor_frame),nodes in groups.items():
            if len(nodes)<2:continue
            curve=self.inventory[ci];proof=frame['label'][nodes[0]];previous=self.frames[neighbor_frame]
            selected=np.asarray(nodes,int)
            source_pixel_checks=[]
            if curve['operator']=='paint_center' and curve['feature_id'] in ('C','B1','B2','goal_line_left','goal_line_right'):
                from .vision_ice_model_context_identity import actual_paint_path_coherence
                readout=dict(candidate_sides=frame['sides'],source_readout_quantization_step_px=frame['readout_quantization_step_px'])
                if 'quantiles' in frame:readout['candidate_side_quantiles_uv']=frame['quantiles']
                coherent,source_pixel_checks=actual_paint_path_coherence(readout,selected)
                held=selected[~coherent]
                frame.setdefault('source_pixel_accuracy_pending_nodes',set()).update(int(i) for i in held)
                selected=selected[coherent];nodes=selected.tolist()
                if len(selected)<2:continue
            evidence=dict(kind='same_shot_actual_RGB_temporal_identity',direction=direction,
                physical_material_id=curve['curve_id'],shot_id=frame['binding']['shot_id'],current_actual_pts=frame['pts'],
                previous_shot_id=previous['binding']['shot_id'],previous_actual_pts=previous['pts'],
                previous_decoded_BGR_sha256=previous['binding']['decoded_BGR_sha256'],
                same_physical_entity_confirmed=True,current_frame_RGB_measured=True,previous_frame_RGB_measured=True,
                source_RGB_tracking_confirmed=True,current_feature_semantics_remeasured=True,
                model_projection_or_fit_residual_used_for_identity=False,
                source_identity_authority=proof['authority'],source_identity_anchor_actual_pts=proof['anchor_actual_pts'],
                actual_material_adjacency_evidence=[frame['label'][node]['neighbor'][2] for node in nodes])
            evidence.update(_tracking_method_evidence(evidence['actual_material_adjacency_evidence']))
            row=dict(actual_pts=frame['pts'],native_pts=frame['binding']['native_pts'],
                source_binding=frame['binding'],source_sha256=self.source_sha256,time_base=self.time_base,
                decoded_native_RGB_sha256=frame['RGB_sha256'],decoded_native_BGR_sha256=frame['binding']['decoded_BGR_sha256'],
                shot_id=frame['binding']['shot_id'],curve_id=curve['curve_id'],feature_id=curve['feature_id'],
                world_curve_role=curve['world_curve_role'],physical_material_id=curve['curve_id'],
                static_material_confirmed=True,actual_source_RGB_measured=True,pixel_location_confirmed=True,
                model_projection_or_fit_residual_used_for_identity=False,
                original_candidate_indices=frame['ids'][selected].tolist(),source_candidate_indices=frame['ids'][selected].tolist(),
                uv=frame['uv'][selected].tolist(),source_sides_uv=frame['sides'][selected].tolist(),
                candidate_sides=frame['sides'][selected].tolist(),raw_candidate_uv=frame['raw_uv'][selected].tolist(),
                center_definition='midpoint_of_two_current_actual_RGB_material_interfaces' if curve['operator']=='paint_center' else None,
                source_seed_authority=proof['authority'],source_seed_reference=proof['source_seed_reference'],
                temporal_evidence=evidence,
                admission_status='admitted_source_material_center_temporal_identity' if curve['operator']=='paint_center' else 'admitted_source_physical_boundary_temporal_identity',
                model_projection_used_as_measurement=False,source_center_smoothed_or_interpolated=False)
            if proof['source_seed_reference'].get('source_identity_navigation_evidence') is not None:
                from .vision_ice_model_context_identity import attach_grid_context_evidence
                attach_grid_context_evidence(row,proof['source_seed_reference'],source_binding=dict(source_sha256=self.source_sha256,
                    time_base=self.time_base,actual_pts=frame['pts'],native_pts=frame['binding']['native_pts'],
                    decoded_native_BGR_sha256=frame['binding']['decoded_BGR_sha256'],shot_id=frame['binding']['shot_id']),
                    source_motion=dict(actual_material_adjacency_evidence=evidence['actual_material_adjacency_evidence'],
                        current_source_pixel_accuracy_checks=source_pixel_checks))
            if 'quantiles' in frame:
                measurement=dict(candidate_side_quantiles_uv=frame['quantiles'],
                    source_readout_quantization_step_px=frame['readout_quantization_step_px'],
                    source_interpolation_grid_px=frame['source_interpolation_grid_px'],source_measurement_method=frame['measurement_method'])
                uncertainty=_current_readout_uncertainty(measurement,selected,frame['uv'][selected],frame['normal'][selected])
                if uncertainty is not None:
                    row['source_readout_uncertainty']=uncertainty
                    row['sigma_px']=np.max(np.abs(uncertainty['normal_interval_px']),axis=1).tolist()
                    row['sigma_semantics']='actual_source_readout_support_half_extent_not_calibrated_Gaussian_covariance'
            clearance=frame.get('candidate_local_source_clearance',{})
            proofs={int(i):clearance[int(i)] for i in frame['ids'][selected] if int(i) in clearance}
            if proofs:
                row['candidate_local_source_clearance']=proofs
                row['original_source_exclusion_flag_was_cleared_only_by_explicit_local_material_proof']=True
            if self.source_surface == 'board':
                row['scope']='board'
            frame['emitted'].update(nodes);rows.append(row)
        return rows
