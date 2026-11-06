"""Read-only comparison of recorded native source metrics, never an accuracy gate.

Each source exposure has equal aggregate mass; distances within an exposure use
native polyline arc length with linearly interpolated recorded sample distances.
Clicks, features and model samples are correlated numerical samples, not trials.
No interpolation here proves image accuracy between actual recorded samples.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np


def _read(value):
    return json.loads(Path(value).read_text(encoding='utf-8-sig'))if isinstance(value,(str,Path))else value


def _key(row):return int(row['pts']),int(row['foot'])


def _stats(values):
    a=np.asarray(values,float)
    if not len(a):return dict(count=0,median=None,p95=None,max=None)
    if not np.isfinite(a).all():raise ValueError('Nonfinite reported metric')
    return dict(count=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))


def _arc(uv,distance,radius,mask=None):
    """Return (distance0,distance1,length,radius) segments, not independent points."""
    uv=np.asarray(uv,float);d=np.asarray(distance,float)
    if uv.shape!=d.shape+(2,)or d.ndim<1:
        raise ValueError('Native UV/distance shapes required')
    m=np.ones(d.shape,bool)if mask is None else np.asarray(mask)
    if m.shape!=d.shape or m.dtype.kind!='b':raise ValueError('Boolean model_visible_mask shape mismatch')
    if not np.isfinite(uv[m]).all()or not np.isfinite(d[m]).all()or(d[m]<0).any():
        raise ValueError('Finite nonnegative visible distances required')
    if not np.isfinite(radius)or radius<0:raise ValueError('Finite original source radius required')
    lengths=np.linalg.norm(np.diff(uv,axis=-2),axis=-1)
    keep=(lengths>0)&m[...,:-1]&m[...,1:]
    return [(float(a),float(b),float(w),float(radius))for a,b,w in zip(d[...,:-1][keep],d[...,1:][keep],lengths[keep])]


def _fraction(a,b,x):
    low,high=min(a,b),max(a,b)
    return float(x>=high)if high==low else float(min(max((x-low)/(high-low),0.),1.))


def _summary(segments):
    if not segments:return None
    total=sum(w for _,_,w,_ in segments)
    def quantile(q,outside=False):
        hi=max(max(a,b)-(r if outside else 0)for a,b,_,r in segments);hi=max(hi,0.)
        def cdf(x):return sum(w*_fraction(a,b,x+(r if outside else 0))for a,b,w,r in segments)/total
        if cdf(0)>=q:return 0.
        lo=0.
        for _ in range(60):
            mid=(lo+hi)/2
            if cdf(mid)>=q:hi=mid
            else:lo=mid
        return float(hi)
    return dict(median=quantile(.5),p95=quantile(.95),max=max(max(a,b)for a,b,_,_ in segments),
        outside_band_px=dict(median=quantile(.5,True),p95=quantile(.95,True),max=max(max(max(a,b)-r,0.)for a,b,_,r in segments)),
        arc_coverage_fraction=sum(w*_fraction(a,b,r)for a,b,w,r in segments)/total,
        arc_length_px=total)


def _aggregate(records,segment_key):
    # Equal exposure mass, even when one exposure has many clicks/components.
    exposures=defaultdict(list)
    for r in records:exposures[(r['pts'],r['foot'])].extend(r.get(segment_key,[]))
    normalized=[]
    for segments in exposures.values():
        total=sum(w for _,_,w,_ in segments)
        if total:normalized.extend((a,b,w/total,r)for a,b,w,r in segments)
    value=_summary(normalized)
    if value is not None:
        value.pop('arc_length_px')
        value['exposure_count']=sum(bool(s)for s in exposures.values())
        value['measure']='equal source exposure mass; within exposure native polyline arc length'
    return value


def _feature(row):
    # Metrics from early versions omit source_structure_feature. The component
    # suffix identifies a partition for counting only, never visibility admission.
    rid=str(row.get('id',''))
    component=bool(re.search(r'_component[0-9]+_',rid))or row.get('source_component_budget_share')is not None
    parent=re.sub(r'_component[0-9]+_.*$','',rid)
    feature=row.get('source_structure_feature')
    if feature is None:
        feature='blade_assembly_outline'if component else row['structure_feature']
    return feature,component,parent


def _terminal(rows,uv,radius,meaning):
    uv=np.asarray(uv,float);matches=[]
    for row in rows:
        points=np.asarray(row['source_polyline_uv'],float)
        ids=np.flatnonzero(np.all(np.isclose(points,uv,atol=1e-8,rtol=0),axis=1))
        model=np.asarray(row.get('model_closest_uv',[]),float)
        if model.shape!=points.shape:continue
        matches.extend(float(np.linalg.norm(model[i]-uv))for i in ids)
    if not matches:return dict(available=False,reason='no exact recorded source sample at finite terminal region')
    distance=max(matches)
    return dict(available=True,native_uv=uv.tolist(),radius_px=float(radius),distance_px=distance,
                outside_band_px=max(distance-float(radius),0.),meaning=meaning,
                recorded_matches=len(matches),multiple_match_policy='maximum, never choose best component')


def _forward(document,contracts):
    grouped=defaultdict(list)
    for row in document.get('visible_runner_contours',[]):
        feature,component,parent=_feature(row)
        grouped[(_key(row),parent,feature)].append(row)
    result=[]
    for ((pts,foot),parent,feature),rows in sorted(grouped.items()):
        contract=contracts.get((pts,foot,feature),{})
        segments=[]
        for row in rows:segments.extend(_arc(row['source_polyline_uv'],row['distance_px'],row['pixel_uncertainty_radius_px']))
        region=contract.get('source_range',{})
        original=region.get('points_uv_original')
        radius=region.get('pixel_uncertainty_radius_px_original')
        if not original and len(rows)==1:
            original=rows[0]['source_polyline_uv'];radius=rows[0]['pixel_uncertainty_radius_px']
        endpoints={}
        if original and radius is not None:
            endpoints['polyline_start']=_terminal(rows,original[0],radius,'first original source range, not a material point')
            endpoints['polyline_end']=_terminal(rows,original[-1],radius,'last original source range, not a material point')
        terminals=contract.get('source_regions',{}).get('silhouette_terminals')or{}
        for name,terminal in terminals.items():
            endpoints[name]=_terminal(rows,terminal['native_uv'],terminal['uncertainty_radius_px'],'declared finite appearance terminal, not fixed material')
        order=contract.get('source_order','unknown')
        tail=endpoints.get('rear_region')
        if tail is None and order in('front_to_rear','rear_to_front'):
            tail=endpoints.get('polyline_end'if order=='front_to_rear'else 'polyline_start')
        summary=_summary(segments)
        full=contract.get('visibility')=='visible'and contract.get('completeness')=='complete'
        result.append(dict(pts=pts,foot=foot,source_feature=feature,source_id=parent,
            metric_row_ids=[r.get('id')for r in rows],feature_row_count=len(rows),
            has_assembly_components=any(_feature(r)[1]for r in rows),
            source_to_model_native_px=summary,endpoints=endpoints,appearance_tail=tail,
            complete_visible_source_contract=full,
            full_length_source_arc_coverage_fraction=summary['arc_coverage_fraction']if full and summary else None,
            source_visibility=contract.get('visibility','unknown'),source_completeness=contract.get('completeness','unknown'),
            role=rows[0].get('role'),_segments=segments,
            _source_signature=sorted(tuple(np.r_[u,v,r['pixel_uncertainty_radius_px']].tolist())for r in rows
                for u,v in zip(r['source_polyline_uv'],r['source_polyline_uv'][1:]))))
    return result


def _reverse(document):
    result=[]
    for row in document.get('complete_runner_extents',[]):
        radius=row['pixel_uncertainty_radius_px'];mask=row.get('model_visible_mask')
        segments=_arc(row['actual_model_runner_uv'],row['model_to_source_distance_px'],radius,mask)
        result.append(dict(pts=int(row['pts']),foot=int(row['foot']),id=row.get('id'),
            visible_model_to_source_native_px=_summary(segments)if mask is not None else None,
            legacy_unmasked_two_side_diagnostic=_summary(segments)if mask is None else None,
            visible_mask_available=mask is not None,visible_sample_count=int(np.sum(mask))if mask is not None else None,
            diagnostic_sample_count=int(np.asarray(row['model_to_source_distance_px']).size),
            _visible_segments=segments if mask is not None else [],_legacy_segments=segments if mask is None else []))
    return result


def _xy(document):
    rows=[];seen=set()
    for r in document.get('source_support_locations',[]):
        key=_key(r)
        if key in seen:raise ValueError('Duplicate XY source exposure')
        seen.add(key)
        vector=np.asarray(r['outside_source_range_xy_conditional_m'],float)
        if vector.shape!=(2,)or not np.isfinite(vector).all():raise ValueError('Finite conditional XY outside vector required')
        frame=r.get('source_frame',{})
        rows.append(dict(pts=key[0],foot=key[1],role=r.get('role'),outside_distance_conditional_m=float(np.linalg.norm(vector)),
            outside_vector_conditional_m=vector.tolist(),source_pixel_uncertainty_radius_px=deepcopy(r.get('source_pixel_uncertainty_radius_px')),
            source_annotation=deepcopy(r.get('source_annotation')),source_annotation_provenance=deepcopy(frame.get('annotation_provenance')),
            source_crop_sha256=frame.get('source_crop_sha256'),source_range_model=r.get('source_range_model'),
            native_camera_row_time_seconds=r.get('native_camera_row_time_seconds'),
            witness_feature=r.get('witness_feature'),actual_contact_material_confirmed=r.get('actual_contact_material_confirmed',False)))
    return dict(source_count=len(rows),unit='conditional_metres_same_frozen_camera_and_ice',
        outside_distance_conditional_m=_stats([r['outside_distance_conditional_m']for r in rows]),per_source=rows,
        independent_3d_measurements=False,threshold_applied=False)


def _snapshot(document,old,contracts):
    forward=_forward(document,contracts);reverse=_reverse(document)
    known_keys={(pts,foot)for pts,foot,_ in contracts}
    groups={}
    for name,expected in [('old22',old),('new_source',known_keys-old)]:
        f=[r for r in forward if(r['pts'],r['foot'])in expected]
        canonical=[r for r in f if r['source_feature']=='runner_lower_edge'and not r['has_assembly_components']]
        rev=[r for r in reverse if(r['pts'],r['foot'])in expected]
        groups[name]=dict(expected_source_exposure_count=len(expected),
            source_exposure_count=len({(r['pts'],r['foot'])for r in f}),original_contour_count=len(f),
            feature_row_count=sum(r['feature_row_count']for r in f),canonical_lower_count=len(canonical),
            missing_canonical_lower_keys=[list(k)for k in sorted(expected-{(r['pts'],r['foot'])for r in canonical})],
            native_source_to_model_px=_aggregate(f,'_segments'),canonical_lower_source_to_model_px=_aggregate(canonical,'_segments'),
            visible_model_to_source_native_px=_aggregate(rev,'_visible_segments'),
            legacy_unmasked_two_side_diagnostic=_aggregate(rev,'_legacy_segments'),
            visible_reverse_source_count=sum(r['visible_mask_available']for r in rev),
            legacy_reverse_source_count=sum(not r['visible_mask_available']for r in rev),
            canonical_lower_full_length_source_coverage_fraction=_stats([r['full_length_source_arc_coverage_fraction']for r in canonical if r['full_length_source_arc_coverage_fraction']is not None]),
            canonical_lower_endpoint_outside_px={label:_stats([endpoint['outside_band_px']for r in canonical
                if(endpoint:=(r['appearance_tail']if label=='appearance_tail'else r['endpoints'].get(label)))and endpoint.get('available')])
                for label in('polyline_start','polyline_end','front_region','rear_region','appearance_tail')})
    return dict(cost=document.get('cost'),image_cost=document.get('image_cost'),groups=groups,
        feature_row_count=len(document.get('visible_runner_contours',[])),original_contour_count=len(forward),
        source_exposure_count=len({_key(r)for r in document.get('visible_runner_contours',[])}),
        body_pixel_distance_native_px=deepcopy(document.get('body_pixel_distance')),
        physical_diagnostics=deepcopy(document.get('physical')),xy=_xy(document),contours=forward,reverse=reverse,
        unclassified_source_keys=[list(k)for k in sorted({(r['pts'],r['foot'])for r in forward}-known_keys-old)])


def compare_metrics(before,after,old22_keys,contracts):
    """Return JSON-compatible, descriptive diagnostics; never fit or write files.

    before/after/contract inputs accept loaded objects or local JSON paths.
    old22_keys is an iterable of (PTS, physical foot), or the old annotation doc
    or path. The complete source contracts, not point counts, define membership.
    Missing visible masks cannot be promoted from legacy dual-side diagnostics.
    """
    before,after,contracts=_read(before),_read(after),_read(contracts)
    old22_keys=_read(old22_keys)
    if isinstance(old22_keys,Mapping):old22_keys=[_key(r)for r in old22_keys['frames']]
    old={tuple(map(int,k))for k in old22_keys}
    contract_rows=contracts.get('rows',[])if isinstance(contracts,Mapping)else contracts
    index={}
    for c in contract_rows:
        key=(*_key(c),c['feature'])
        if key in index:raise ValueError('Duplicate source contract')
        index[key]=c
    a,b=_snapshot(before,old,index),_snapshot(after,old,index)
    old_contours={(r['pts'],r['foot'],r['source_id']):r for r in a['contours']}
    new_contours={(r['pts'],r['foot'],r['source_id']):r for r in b['contours']}
    exposures=defaultdict(list)
    changes=[]
    for key in sorted(old_contours.keys()|new_contours.keys()):
        r0,r1=old_contours.get(key),new_contours.get(key)
        entry=dict(pts=key[0],foot=key[1],source_id=key[2],source_feature=(r1 or r0)['source_feature'],
            before_available=r0 is not None,after_available=r1 is not None,deltas={},regression_flags=[],
            source_range_comparable=bool(r0 and r1 and r0['_source_signature']==r1['_source_signature']))
        if r0 and not r1:entry['regression_flags'].append('source_metric_unavailable_after')
        if r0 and r1 and not entry['source_range_comparable']:
            entry['comparison_limitation']='Source native polyline/radius changed; no direct delta or regression judgement'
        if entry['source_range_comparable']and r0['source_to_model_native_px']and r1['source_to_model_native_px']:
            for field in('median','p95','max'):
                entry['deltas'][field+'_px']=r1['source_to_model_native_px'][field]-r0['source_to_model_native_px'][field]
            outside=r1['source_to_model_native_px']['outside_band_px']['p95']-r0['source_to_model_native_px']['outside_band_px']['p95']
            entry['deltas']['outside_band_p95_px']=outside
            if outside>0:entry['regression_flags'].append('outside_source_band_p95_increased')
            coverage=r1['source_to_model_native_px']['arc_coverage_fraction']-r0['source_to_model_native_px']['arc_coverage_fraction']
            entry['deltas']['source_arc_coverage_fraction']=coverage
            if coverage<0:entry['regression_flags'].append('source_arc_coverage_decreased')
            for label in('polyline_start','polyline_end','front_region','rear_region'):
                x,y=r0['endpoints'].get(label),r1['endpoints'].get(label)
                if x and y and x.get('available')and y.get('available'):
                    delta=y['outside_band_px']-x['outside_band_px'];entry['deltas'][label+'_outside_px']=delta
                    if delta>0:entry['regression_flags'].append(label+'_outside_source_band_increased')
            x,y=r0['appearance_tail'],r1['appearance_tail']
            if x and y and x.get('available')and y.get('available'):
                delta=y['outside_band_px']-x['outside_band_px'];entry['deltas']['appearance_tail_outside_px']=delta
                if delta>0:entry['regression_flags'].append('appearance_tail_outside_source_band_increased')
        changes.append(entry)
        if entry['regression_flags']:exposures[key[:2]].append(entry)
    xy0={_key(r):r for r in a['xy']['per_source']};xy1={_key(r):r for r in b['xy']['per_source']}
    xy_changes=[]
    for key in sorted(xy0.keys()|xy1.keys()):
        x,y=xy0.get(key),xy1.get(key)
        comparable=bool(x and y and all(x.get(k)==y.get(k)for k in('source_annotation','source_pixel_uncertainty_radius_px','source_crop_sha256','source_range_model')))
        delta=y['outside_distance_conditional_m']-x['outside_distance_conditional_m']if comparable else None
        record=dict(pts=key[0],foot=key[1],before_outside_conditional_m=None if x is None else x['outside_distance_conditional_m'],
                    after_outside_conditional_m=None if y is None else y['outside_distance_conditional_m'],delta_conditional_m=delta,
                    source_range_comparable=comparable)
        xy_changes.append(record)
        if delta is not None and delta>0:exposures[key].append(dict(source_feature='source_support_location_xy',delta_conditional_m=delta,regression_flags=['conditional_xy_outside_increased']))
    def public(value):
        if isinstance(value,dict):return {k:public(v)for k,v in value.items()if not k.startswith('_')}
        if isinstance(value,list):return [public(v)for v in value]
        return value
    return public(dict(schema='rinkspace.source_guided_metrics_comparison.v1',before=a,after=b,
        independent_heldout=dict(source_exposure_count=0,available=False,reason='No frozen independent heldout observations or role ledger supplied; all recorded source metrics remain development evidence'),
        contour_changes=changes,xy_changes=xy_changes,
        exposure_regressions=[dict(pts=k[0],foot=k[1],source_changes=v)for k,v in sorted(exposures.items())],
        accuracy_pass_declared=False,limitations=[
            'Descriptive changes relative to each original finite source band; no universal pixel/metre threshold and no accuracy acceptance.',
            'Native source/model arc statistics interpolate recorded distances; between-sample geometry and image truth are not verified.',
            'These metric files alone do not establish source-video/camera/model identity; caller must bind the same native clock and frozen calibration.',
            'All source exposures belong to a correlated fitted sequence; clicks/features/samples are not independent observations.',
            'Legacy unmasked reverse samples cover both model sides and are not comparable to the new visible-branch measurement.',
            'Appearance terminals are finite regions, not fixed material endpoints, blade contact or forces.',
            'Body detector pixel diagnostics and physical cavity/collar diagnostics are separate from source agreement.',
            'Costs are copied verbatim; stage objective weights may differ, so cost reduction is not a matched-objective accuracy verdict.']))
