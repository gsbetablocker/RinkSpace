"""Explicit graph variant that checks final q50 source readouts before writing.

Frozen graph code is unchanged. The caller must pass the mutable selected-row
dict to consume, and use that same dict for both proof preservation and writer
normalization. Missing source support stays pending; no missing node is forged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .vision_ice_identity_graph import IceSourceMaterialIdentityGraph
from .vision_ice_model_context_identity import source_paint_pixel_accuracy_check
from .vision_programme_ice_review import _candidate_context_excluded


def _subset_source_row(original,keep):
    row=deepcopy(original);count=len(original['original_candidate_indices'])
    for key in ('uv','source_sides_uv','candidate_sides','raw_candidate_uv','source_candidate_indices',
            'original_candidate_indices','source_query_indices','original_point_indices','sigma_px',
            'actual_source_normal_uv','normal','source_side_quantiles_uv'):
        if key in row:
            value=np.asarray(row[key])
            if value.ndim and len(value)==count:row[key]=value[keep].tolist()
    continuity=row.get('continuity_evidence',{})
    for key in ('current_actual_paint_widths_px','current_source_RGB_material_context_correlations'):
        if key in continuity:
            value=np.asarray(continuity[key])
            if value.ndim and len(value)==count:continuity[key]=value[keep].tolist()
    uncertainty=row.get('source_readout_uncertainty',{})
    for key in ('actual_readout_blur_interval_px','normal_interval_px','quantization_margin_px',
            'actual_side_gradient_quantiles_uv'):
        if key in uncertainty:
            value=np.asarray(uncertainty[key])
            if value.ndim and len(value)==count:uncertainty[key]=value[keep].tolist()
    clearance=row.get('candidate_local_source_clearance')
    if isinstance(clearance,dict):
        retained=set(map(int,np.asarray(original['original_candidate_indices'])[keep]))
        row['candidate_local_source_clearance']={key:value for key,value in clearance.items() if int(key) in retained}
    return row


def _subset_nodes(frame,keep):
    """Remove unsupported nodes before they can be a source-flow parent."""
    for key in ('ids','query_ids','uv','sides','normal','width','curve','descriptor','raw_uv','quantiles'):
        if key in frame:frame[key]=frame[key][keep]


def _final_context_clear(frame,mask):
    """Read the same complete context as the source profile exclusion gate."""
    result=np.ones(len(frame['ids']),bool)
    for first in range(0,len(result),256):
        last=min(first+256,len(result));normal=frame['normal'][first:last]
        tangent=np.c_[-normal[:,1],normal[:,0]]
        points=np.concatenate([frame['uv'][first:last,None],frame['sides'][first:last]],axis=1)
        context=(points[:,:,None,None]+normal[:,None,None,None]*np.arange(-8,9)[None,None,:,None,None]
            +tangent[:,None,None,None]*np.arange(-3,4)[None,None,None,:,None])
        pixels=np.rint(context).astype(int).reshape(last-first,-1,2)
        inside=(pixels>=0).all(2)&(pixels<[mask.shape[1],mask.shape[0]]).all(2)
        safe=np.clip(pixels,[0,0],[mask.shape[1]-1,mask.shape[0]-1])
        result[first:last]=inside.all(1)&~mask[safe[:,:,1],safe[:,:,0]].any(1)
    return result


class PreciseSourceIceMaterialIdentityGraph(IceSourceMaterialIdentityGraph):
    """Use measured final quantile sides in both graph support and writer gate."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.final_source_precision_receipts=[]
        self.latest_source_final_precision_review=None

    def _nodes(self,rgb,measurement,mask):
        prepared=getattr(self,'_prepared_final_nodes',None)
        if prepared is not None:return prepared
        # These are stored measurements of this same source profile, not
        # projected positions or fitted coordinates. Raw f32 readouts remain
        # available as raw_uv and in the immutable source cache/overlay.
        final=dict(measurement)
        quantiles=measurement.get('candidate_side_quantiles_uv')
        if quantiles is not None:
            quantiles=np.asarray(quantiles,float)
            sides=np.asarray(measurement['candidate_sides'],float).copy()
            uv=np.asarray(measurement['candidate_uv'],float).copy()
            query=np.asarray(measurement['candidate_query'])
            curve=np.asarray(measurement['query_curve'])[query]
            supported=np.isfinite(quantiles[:,:,1]).all((1,2))
            roles=np.asarray([self.inventory[int(i)]['operator'] for i in curve],dtype=str)
            use=supported&np.isin(roles,['paint_center','material_interface'])
            sides[use]=quantiles[use,:,1]
            uv[use]=sides[use].mean(1)
            final['candidate_sides']=sides;final['candidate_uv']=uv
        frame=super()._nodes(rgb,final,mask)
        rejections={int(index):'final_q50_actual_paint_RGB_support_missing'
            for index in frame.get('source_pixel_accuracy_pending_candidate_ids',())}
        pixels=np.rint(np.asarray(final['candidate_uv'])).astype(int)
        inside=(pixels>=0).all(1)&(pixels<[mask.shape[1],mask.shape[0]]).all(1)
        clear=(np.asarray(final['candidate_flags'])&3)==0
        clear&=inside
        clear[inside]&=~mask[pixels[inside,1],pixels[inside,0]]
        missing=np.flatnonzero(clear&~np.isin(np.arange(len(clear)),frame['ids']))
        rejections.update({int(index):'actual_source_graph_descriptor_support_missing'
            for index in missing if int(index) not in rejections})
        frame['raw_uv']=np.asarray(measurement['candidate_uv'],float)[frame['ids']]
        frame['node_readout_coordinate_domain']='actual_gradient_q50_float64_when_available'
        # Quantile precision can move a transition across an exclusion pixel.
        # Gate every node now, including pending nodes reached by later anchors.
        clear=_final_context_clear(frame,mask)
        rejections.update({int(index):'final_q50_source_context_is_excluded_or_surface_unknown'
            for index in frame['ids'][~clear]})
        _subset_nodes(frame,clear)
        # Source-path precision applies to all possible delayed paint nodes,
        # not only nodes already named by the selector in the current frame.
        keep=np.ones(len(frame['ids']),bool)
        from .vision_ice_model_context_identity import actual_paint_path_coherence
        for ci,definition in enumerate(self.inventory):
            if definition['operator']!='paint_center' or definition['feature_id'] not in (
                    'C','B1','B2','goal_line_left','goal_line_right'):continue
            selected=np.flatnonzero(frame['curve']==ci)
            if not len(selected):continue
            readout=dict(candidate_sides=frame['sides'],source_readout_quantization_step_px=
                measurement.get('source_readout_quantization_step_px',1.))
            if 'quantiles' in frame:readout['candidate_side_quantiles_uv']=frame['quantiles']
            coherent,held=actual_paint_path_coherence(readout,selected)
            rejections.update({int(frame['ids'][value['original_candidate_index']]):value['status'] for value in held})
            keep[selected[~coherent]]=False
        _subset_nodes(frame,keep)
        self._latest_final_node_rejections=rejections
        return frame

    def consume(self,rgb,measurement,exclusion_mask,*,source_ice_scope_mask,
            selected_rows=(),source_binding=None):
        incoming=source_binding or measurement['source_binding']
        if self.frames and incoming['shot_id']!=self.shot_id:self.final_source_precision_receipts.clear()
        selected=selected_rows.get('rows',()) if isinstance(selected_rows,dict) else selected_rows
        if selected and not isinstance(selected_rows,(dict,list)):
            raise ValueError('Final precision needs the same mutable selected rows later used by the writer')
        initial_selected_count=len(selected)
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr=hashlib.sha256(np.ascontiguousarray(np.asarray(rgb)[:,:,::-1]).tobytes()).hexdigest()
        if (digest!=measurement['decoded_native_RGB_sha256']
                or incoming.get('decoded_native_BGR_sha256',incoming.get('decoded_BGR_sha256',bgr))!=bgr):
            raise ValueError('Final precision requires its own unchanged original RGB/BGR')
        mask=np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
        binding=dict(incoming,actual_pts=int(incoming.get('actual_pts',incoming.get('source_pts'))),
            native_pts=int(incoming.get('native_pts',incoming.get('actual_pts',incoming.get('source_pts')))),
            decoded_native_BGR_sha256=bgr,decoded_BGR_sha256=bgr)
        frame=self._nodes(rgb,measurement,mask)
        node_rejections=self._latest_final_node_rejections
        frame.update(binding=binding,pts=binding['actual_pts'],RGB_sha256=digest)
        by_id={int(value):i for i,value in enumerate(frame['ids'])}
        raw_uv=np.asarray(measurement['candidate_uv'],float)
        raw_sides=np.asarray(measurement['candidate_sides'],float)
        query=np.asarray(measurement['candidate_query']);curve=np.asarray(measurement['query_curve'])[query]
        quantiles=measurement.get('candidate_side_quantiles_uv')
        quantiles=None if quantiles is None else np.asarray(quantiles,float)
        output=[];pending=[];seen=set();rejected_nodes=set();RGB_field=np.asarray(rgb,np.float32)
        for original in selected:
            ids=np.asarray(original['original_candidate_indices'],np.int64)
            if (ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(raw_uv))):
                raise ValueError('Final source rows need unique exact current candidate indices')
            ci=next(i for i,value in enumerate(self.inventory) if value['curve_id']==original['curve_id'])
            if not np.all(curve[ids]==ci):raise ValueError('Final source row cannot rename another material curve')
            if any(int(i) in seen for i in ids):raise ValueError('Two source rows claim the same exact current candidate')
            seen.update(map(int,ids))
            definition=self.inventory[ci];paint=definition['operator']=='paint_center'
            expected_uv=raw_sides[ids].mean(1) if paint else raw_uv[ids]
            actual_sides=raw_sides[ids].copy();actual_uv=expected_uv.copy()
            if quantiles is not None:
                fine=np.isfinite(quantiles[ids,:,1]).all((1,2))
                actual_sides[fine]=quantiles[ids[fine],:,1]
                actual_uv[fine]=actual_sides[fine].mean(1)
            raw=(np.array_equal(np.asarray(original['uv'],float),expected_uv)
                and np.array_equal(np.asarray(original['source_sides_uv'],float),raw_sides[ids]))
            refined=(np.array_equal(np.asarray(original['uv'],float),actual_uv)
                and np.array_equal(np.asarray(original['source_sides_uv'],float),actual_sides))
            if not (raw or refined):raise ValueError('Final source proof UV/sides are not an exact saved raw or q50 readout')
            held={};normals=np.asarray(measurement['query_normal'],float)[query[ids]].copy()
            width=np.linalg.norm(actual_sides[:,1]-actual_sides[:,0],axis=1)
            paired=width>1e-6
            normals[paired]=(actual_sides[paired,1]-actual_sides[paired,0])/width[paired,None]
            for local,index in enumerate(ids):
                if np.asarray(measurement['candidate_flags'])[index]&3:
                    held[int(index)]='current_exact_source_candidate_is_unresolved_truncated_or_excluded'
                elif any(_candidate_context_excluded(mask,p,normals[local]) for p in
                        [actual_uv[local],*actual_sides[local]]):
                    held[int(index)]='final_q50_source_context_is_excluded_or_surface_unknown'
                elif int(index) not in by_id:
                    held[int(index)]=node_rejections.get(int(index),'actual_source_graph_descriptor_support_missing')
            if paint:
                checked=source_paint_pixel_accuracy_check(rgb,measurement,ids,definition,native_rgb_field=RGB_field)
                for item in checked['pending']:
                    held[int(item['original_candidate_index'])]=item['status']
            for local,index in enumerate(ids):
                if int(index) not in held:continue
                record=dict(original_candidate_index=int(index),source_query_index=int(query[index]),
                    curve_id=original['curve_id'],reason=held[int(index)],source_binding=dict(binding),
                    decoded_native_RGB_sha256=frame['RGB_sha256'],raw_candidate_uv=raw_uv[index].tolist(),
                    raw_candidate_sides_uv=raw_sides[index].tolist(),actual_final_q50_uv=actual_uv[local].tolist(),
                    actual_final_q50_sides_uv=actual_sides[local].tolist(),
                    actual_source_RGB_final_precision_checked=True,model_UV_used_as_pixel_truth=False,
                    preserved_graph_node_exists=int(index) in by_id)
                if quantiles is not None:record['actual_source_side_quantiles_uv']=quantiles[index].tolist()
                pending.append(record)
                if int(index) in by_id:
                    rejected_nodes.add(int(index))
            keep=np.asarray([int(i) not in held for i in ids],bool)
            if not keep.any():continue
            row=_subset_source_row(original,keep)
            row['uv']=actual_uv[keep].tolist();row['source_sides_uv']=actual_sides[keep].tolist()
            row['candidate_sides']=actual_sides[keep].tolist()
            row['raw_candidate_uv']=raw_uv[ids[keep]].tolist()
            row['final_actual_source_readout_domain']='actual_gradient_q50_float64_when_available'
            row['final_current_RGB_pixel_precision_checked']=True
            output.append(row)
        if isinstance(selected_rows,dict):
            selected_rows['rows']=output
            selected_rows.setdefault('decisions',[]).extend(dict(curve_id=value['curve_id'],actual_pts=frame['pts'],
                status='final_current_source_pixel_precision_pending',original_candidate_index=value['original_candidate_index'],
                source_precision_reason=value['reason']) for value in pending)
        elif isinstance(selected_rows,list):selected_rows[:]=output
        # Base consume must see only these final rows and final nodes. Filtering
        # after it returns is too late: a rejected selector anchor may already
        # have labelled earlier frames and generated delayed source rows.
        if rejected_nodes:
            _subset_nodes(frame,~np.isin(frame['ids'],list(rejected_nodes)))
            for record in pending:record['preserved_graph_node_exists']=False
        self._prepared_final_nodes=frame
        try:
            result=super().consume(rgb,measurement,exclusion_mask,
                source_ice_scope_mask=source_ice_scope_mask,selected_rows=selected_rows,source_binding=source_binding)
        finally:
            self._prepared_final_nodes=None
        frame=self.frames[-1];binding=frame['binding']
        receipt=dict(actual_pts=frame['pts'],source_binding=dict(binding),selected_rows_before=initial_selected_count,
            selected_rows_after=len(output),pending_actual_source_candidates=pending,
            final_source_node_rejections={str(index):reason for index,reason in node_rejections.items()},
            final_actual_source_RGB_readout_checked=True,missing_graph_nodes_forged=False,
            old_raw_readouts_preserved=True,unique_exact_node_requirement_unchanged=True)
        self.latest_source_final_precision_review=receipt
        if pending or node_rejections:self.final_source_precision_receipts.append(receipt)
        result['source_current_final_pixel_accuracy']=receipt
        return result

    def save_compact_state(self,path):
        path=Path(path)
        pending_path=path.with_name(path.stem+'_source_precision_pending.npz')
        if pending_path.exists():raise FileExistsError('Do not overwrite a pending actual source precision receipt')
        receipt=super().save_compact_state(path)
        payload=json.dumps(dict(schema='rinkspace.final-q50-source-precision-pending.v1',
            source_graph_reference=dict(path=receipt['path'],sha256=receipt['sha256']),
            source_precision_receipts=self.final_source_precision_receipts,
            missing_nodes_forged=False,source_points_not_admitted=True),ensure_ascii=False,separators=(',',':'))
        np.savez_compressed(pending_path,pending_actual_source_receipts_utf8=np.frombuffer(payload.encode('utf-8'),np.uint8))
        receipt['source_precision_pending_reference']=dict(path=str(pending_path.resolve()),
            sha256=hashlib.sha256(pending_path.read_bytes()).hexdigest())
        return receipt
