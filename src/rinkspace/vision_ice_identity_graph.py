"""Single-decode, same-shot material links for source-seeded ice identities.

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


def _point_source_flow(previous, current, old, previous_mask, current_mask):
    """Bidirectional LK on measured material supports only; no background H."""
    centers, sides, normals = old['uv'], old['sides'], old['normal']
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
        source = support.reshape(-1, 3, 2)[i, accepted]
        if not len(np.unique(source, axis=0)):
            continue
        movement = np.median(delta[i, accepted], axis=0)
        predictions[i] = centers[i]+normals[i]*float(movement@normals[i])
        identified[i] = True
        errors[i] = float(np.median(error[i, accepted]))
    return predictions, identified, errors


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


def _actual_material_links(previous, current, old, now, previous_mask, current_mask,*,device='cpu'):
    predicted, flow_good, errors = _point_source_flow(previous, current, old, previous_mask, current_mask)
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
    def __init__(self, inventory, *, source_sha256, time_base, source_path=None,device='cpu'):
        self.inventory = [dict(row) for row in inventory]
        self.source_sha256, self.time_base, self.source_path = source_sha256, str(time_base),None if source_path is None else str(source_path)
        if device!='cpu' and not str(device).startswith('cuda'):raise ValueError('Source material matching device must be cpu or cuda')
        self.device=device
        self.frames = []
        self.previous_gray = self.previous_mask = None
        self.shot_id = None
        self.catalog = []

    def source_support_candidate_indices(self, measurement, exclusion_mask, *, source_ice_scope_mask):
        """Candidate indices needing real blur support before RGB release.

        These are pending source material nodes, not admitted observations.
        Future anchors can reach any of them, so reading only today's admitted
        rows would leave earlier exposures without real gradient blur evidence.
        """
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

    def consume(self, rgb, measurement, exclusion_mask, *, source_ice_scope_mask,
                selected_rows=(), source_binding=None):
        """Return newly reached rows; each row carries its own exact binding.

        The caller releases RGB after this call, normalizes delayed rows using
        the cached measurement at their actual PTS, and writes no PNG library.
        Pending graphs are discarded at a cut, never propagated between shots.
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
        mask = np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
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
        self.frames.append(frame)
        queue=deque()
        if ordinal:
            before=self.frames[-2]
            edges=_actual_material_links(self.previous_gray,gray,before,frame,self.previous_mask,mask,device=self.device)
            for a,b,error,corr in edges:
                evidence=dict(source_forward_backward_LK_error_px=error,current_material_context_correlation=corr,
                    source_RGB_support_only=True,full_2D_homography_estimated=False,
                    model_projection_or_fit_residual_used_for_identity=False)
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

    def save_compact_state(self,path):
        """Persist this shot's actual material graph before a cut resets it."""
        path=Path(path)
        if path.exists():raise FileExistsError('Do not overwrite a preserved source material graph')
        path.parent.mkdir(parents=True,exist_ok=True)
        offsets=np.r_[0,np.cumsum([len(f['ids']) for f in self.frames])].astype(np.int64)
        def cat(key,shape,dtype):
            arrays=[np.asarray(f[key],dtype=dtype) for f in self.frames]
            return np.concatenate(arrays) if arrays else np.empty(shape,dtype)
        edges=[];edge_lookup={}
        for fi,frame in enumerate(self.frames):
            for node,(pfi,pnode,evidence) in frame['parents'].items():
                edge_lookup[(pfi,pnode,fi,node)]=len(edges)
                edges.append([pfi,pnode,fi,node,float(evidence['source_forward_backward_LK_error_px']),
                    float(evidence['current_material_context_correlation'])])
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
        metadata=dict(schema='rinkspace.compact-actual-source-material-adjacency.v1',source_sha256=self.source_sha256,
            time_base=self.time_base,source_path=self.source_path,shot_id=self.shot_id,inventory=self.inventory,
            frames=frames,authority_groups=groups,source_material_matching_device=self.device,
            source_RGB_images_saved=False,model_projection_used_as_source_pixel_truth=False,
            model_navigation_context_used_for_identity=any(g['source_seed_reference'].get('source_identity_navigation_evidence') is not None for g in groups),
            visual_review_complete=False)
        preserved=[dict(frame_ordinal=fi,proof=proof) for fi,f in enumerate(self.frames)
            for proof in f.get('preserved_source_row_proofs',())]
        preserved_utf8=np.frombuffer(json.dumps(preserved,ensure_ascii=False,separators=(',',':')).encode('utf8'),np.uint8)
        np.savez_compressed(path,node_offsets=offsets,original_candidate_indices=cat('ids',(0,),np.int32),
            original_query_indices=cat('query_ids',(0,),np.int32),curve_indices=cat('curve',(0,),np.int16),
            uv=cat('uv',(0,2),np.float64),sides_uv=cat('sides',(0,2,2),np.float64),normals=cat('normal',(0,2),np.float64),
            widths=cat('width',(0,),np.float64),descriptors=cat('descriptor',(0,123),np.float16),raw_candidate_uv=cat('raw_uv',(0,2),np.float64),
            side_quantiles_uv=np.concatenate(quantiles) if quantiles else np.empty((0,2,3,2)),
            actual_source_edges=np.asarray(edges,np.float64).reshape(-1,6),confirmed_node_labels=np.asarray(labels,np.int64).reshape(-1,6),
            metadata_json=json.dumps(metadata,ensure_ascii=False,separators=(',',':')),
            preserved_source_row_proofs_utf8=preserved_utf8)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        return dict(path=str(path.resolve()),sha256=digest.hexdigest(),
            source_frames=len(frames),actual_material_nodes=int(offsets[-1]),actual_source_adjacency_edges=len(edges),
            confirmed_nodes=len(labels),visual_review_complete=False)

    @classmethod
    def load_compact_state(cls,path,*,device='cpu'):
        """Restore exact readouts/links for offline later source-anchor replay."""
        with np.load(path,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata_json'].item()))
            if metadata['schema']!='rinkspace.compact-actual-source-material-adjacency.v1':raise ValueError('Unknown source graph schema')
            graph=cls(metadata['inventory'],source_sha256=metadata['source_sha256'],time_base=metadata['time_base'],
                source_path=metadata['source_path'],device=device)
            graph.shot_id=metadata['shot_id']
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
            def evidence(row):return dict(source_forward_backward_LK_error_px=float(row[4]),
                current_material_context_correlation=float(row[5]),source_RGB_support_only=True,
                full_2D_homography_estimated=False,model_projection_or_fit_residual_used_for_identity=False)
            for row in edges:
                pfi,pnode,fi,node=map(int,row[:4]);proof=evidence(row)
                graph.frames[fi]['parents'][node]=(pfi,pnode,proof)
                graph.frames[pfi]['children'].setdefault(pnode,[]).append((fi,node,proof))
            for row in labels:
                fi,node,group,direction,edge,emitted=map(int,row)
                neighbor=None
                if edge>=0:
                    data=edges[edge];pfi,pnode,cfi,cnode=map(int,data[:4])
                    neighbor=(cfi,cnode,evidence(data)) if direction==-1 else (pfi,pnode,evidence(data))
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
                    or seed.get('source_material_class')!='static_painted_ice'
                    or seed.get('authority',{}).get('path')!=authority.get('path') or seed.get('authority',{}).get('sha256')!=authority.get('sha256')):
                raise ValueError('Preserved selector proof requires the authenticated original named paint seed')
            ev=row['temporal_evidence']
            if ev['kind'] in ('same_shot_actual_RGB_temporal_identity','actual_RGB_material_field_order_and_continuity_with_model_navigation'):
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
                    and seed.get('source_material_class')=='static_painted_ice'
                    and seed.get('authority',{}).get('path')==authority.get('path')
                    and seed.get('authority',{}).get('sha256')==authority.get('sha256'))
                validated[key]=valid
            return validated[key]
        for fi,frame in enumerate(self.frames):
            for node,proof in frame['label'].items():
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
                            source_RGB_tracking_method='preserved_bidirectional_actual_material_support_LK_and_RGB_context',
                            actual_material_adjacency_evidence=links)
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
            if (seed.get('curve_id')!=row['curve_id'] or seed.get('surface_and_physical_identity_confirmed') is not True
                    or seed.get('source_sha256')!=self.source_sha256 or str(seed.get('time_base'))!=self.time_base
                    or seed.get('authority',{}).get('path')!=authority.get('path')
                    or seed.get('authority',{}).get('sha256')!=authority.get('sha256')):
                raise ValueError('Offline anchor requires the authenticated original physical identity seed')
            temporal=row['temporal_evidence']
            if temporal['kind'] in ('same_shot_actual_RGB_temporal_identity','actual_RGB_material_field_order_and_continuity_with_model_navigation'):
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
                if (endpoint.get('scope')!='ice' or endpoint.get('native_pts')!=binding['native_pts']
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
        frame=self.frames[fi]
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
                actual_material_adjacency_evidence=[frame['label'][node]['neighbor'][2] for node in nodes],
                source_RGB_tracking_method='bidirectional_actual_material_support_LK_and_RGB_context')
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
            frame['emitted'].update(nodes);rows.append(row)
        return rows
