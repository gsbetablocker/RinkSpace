"""Target-only physical-band reread after a bilinear source-domain audit.

Existing proposals are replayed in source order.  This never changes circles,
thin lines, or any unrelated wide feature, and does not publish observations.
"""
from __future__ import annotations
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import cv2
import numpy as np
from .repair_physical_wide_bands import (
    RD, MODEL, PhysicalBandProfiles, install_user_guide, Extractor, read, write,
    sha, current_broadcast_exclusion, current_ice_visible_domain)
from .projection import ground_xy, project_fast

OUT=RD/'wide_quantitative_diagnostics'
B1_OWNER=set(range(426,450))|set(range(615,629))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def context_mask(source,index):
    raw=cv2.imread(source['image_path'])
    broadcast,_=current_broadcast_exclusion(raw)
    outside,_=current_ice_visible_domain(index)
    mask=(cv2.imread(source['mask_path'],0)>0)|broadcast|outside
    return raw,mask


def source_tangent(feature):
    """The original straight query stations determine their sampling tangent."""
    rows=feature['source_line_fit']['direct_readouts']
    query=np.asarray([r['source_evidence']['query_uv']for r in rows])
    if len(query)<2 or np.linalg.norm(np.ptp(query,axis=0))<1:
        raise ValueError('Cannot reconstruct a straight source query tangent')
    center=query.mean(0);tangent=np.linalg.svd(query-center,full_matrices=False)[2][0]
    residual=np.max(abs((query-center)@np.array([-tangent[1],tangent[0]])))
    if residual>.01:
        raise ValueError(f'Queries are not a straight source path: {residual}')
    return tangent,float(residual)


def readout_context(camera,ev,tangent,mask):
    """Recreate source samples and require all four bilinear source pixels."""
    query=np.asarray(ev['query_uv']);origin,valid=ground_xy(camera,query[None])
    if not valid[0]:raise ValueError('Readout query does not invert onto ice')
    axis=np.asarray(ev['physical_width_axis_xy'])
    probe,_=project_fast(camera,np.c_[origin[0]+np.array([-.01,.01])[:,None]*axis,np.zeros(2)])
    scale=np.linalg.norm(probe[1]-probe[0])/.02
    lo,hi=ev['complete_band_native_interval_px']
    scalar_step=ev.get('physical_width_profile_normal_equivalent_sample_step',1.)
    # Both production readers start from an integer normal-equivalent radius.
    scalar=np.arange(np.ceil(lo/scalar_step-1e-6),np.floor(hi/scalar_step+1e-6)+1)*scalar_step
    step=ev.get('physical_width_profile_native_sampling_step_px')
    if step is None:
        step=1/ev['transverse_profile_image_normal_coverage_scale']
    world=origin[0]+(scalar*step)[:,None]/scale*axis
    uv,depth=project_fast(camera,np.c_[world,np.zeros(len(world))])
    # values() casts the central section to float32 before tangent offsets.
    single_section=ev.get('measurement_origin')=='current_native_source_verified_B1_physical_transverse_luminance_two_ice_side_flanks'
    tangent_offsets=np.array([0])if single_section else np.arange(-3,4)
    samples=uv.astype(np.float32)[None]+tangent_offsets[:,None,None]*tangent
    floor=np.floor(samples).astype(int);good=np.ones(samples.shape[:2],bool)
    outside=np.zeros_like(good);excluded=np.zeros_like(good)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=floor[...,0]+dx;y=floor[...,1]+dy
        inside=(x>=0)&(x<1920)&(y>=0)&(y<1080)
        masked=np.zeros_like(inside);masked[inside]=mask[y[inside],x[inside]]
        outside|=~inside;excluded|=masked;good&=inside&~masked
    counts=good.sum(0)
    return dict(outside_contributions=int(outside.sum()),
        masked_contributions=int(excluded.sum()),total_contributions=int(good.size),
        invalid_contributions=int((~good).sum()),
        actually_sampled_tangent_offsets_px=tangent_offsets.tolist(),
        valid_count_range=[int(counts.min()),int(counts.max())],
        zero_valid_profile_positions=int((counts==0).sum()),
        span_normal_equivalent=[float(lo),float(hi)])


def audit():
    extractor=Extractor(RD,MODEL)
    issues=[];used_issues=[];rows_checked=0;frames=[]
    for index in range(201,1249):
        packet=read(RD/f'physical_wide_proposals/{index:06d}.json')
        _,mask=context_mask(packet['source'],index)
        frame_issues=0
        for feature in packet['features']:
            fallback,residual=source_tangent(feature)
            for row in feature['source_line_fit']['direct_readouts']:
                rows_checked+=1;ev=row['source_evidence']
                tangent=np.asarray(ev.get('tangent_sampling_unit_direction_uv',fallback))
                result=readout_context(extractor.cameras[index],ev,tangent,mask)
                record=dict(frame=index,target_id=feature['target_id'],query=ev['query_index'],
                    explicit_sampling_tangent='tangent_sampling_unit_direction_uv'in ev,
                    reconstructed_query_line_residual_px=residual,**result)
                if result['invalid_contributions']:
                    issues.append(record);frame_issues+=1
                if ev.get('measurement_origin')=='current_native_source_verified_B1_physical_transverse_luminance_two_ice_side_flanks':
                    if result['invalid_contributions']:
                        used_issues.append(dict(**record,reason='single physical section used an excluded bilinear source contributor'))
                elif ev.get('tangent_median_sampling')=='only_visible_current_ice_bilinear_contributors':
                    if result['zero_valid_profile_positions']:
                        used_issues.append(dict(**record,reason='accepted span has no valid source median contributor'))
                    elif ev.get('tangent_source_valid_sample_count_range')!=result['valid_count_range']:
                        used_issues.append(dict(**record,reason='recorded valid contributor count does not match source reconstruction',recorded=ev.get('tangent_source_valid_sample_count_range')))
                elif result['invalid_contributions']:
                    used_issues.append(dict(**record,reason='legacy unmasked median included an excluded source contributor'))
        frames.append(dict(frame=index,packet_sha256=sha(RD/f'physical_wide_proposals/{index:06d}.json'),available_context_issues=frame_issues))
        if index%100==0:print('bilinear context',index,len(issues),len(used_issues),flush=True)
    result=dict(schema='rinkspace.independent-physical-wide-bilinear-source-context.v1',
        checked_frames=1048,native_material_readouts=rows_checked,
        available_invalid_context_readout_count=len(issues),issues=issues,
        used_source_context_issue_count=len(used_issues),used_source_context_issues=used_issues,
        frames=frames,no_candidate_or_model_modified=True,
        completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'physical_wide_bilinear_context_audit.json',result)
    print('audit complete',rows_checked,len(issues),len(used_issues),flush=True)


def apply():
    audit_path=OUT/'physical_wide_bilinear_context_audit.json';report=read(audit_path)
    receipt_path=OUT/'physical_wide_context_reread_receipt.json'
    prior=read(receipt_path)if receipt_path.exists()else{}
    audit_digest=sha(audit_path)
    write(OUT/f'physical_wide_bilinear_context_audit_{audit_digest[:12]}.json',report)
    affected=defaultdict(set)
    skipped=[]
    for issue in report['used_source_context_issues']:
        i=issue['frame'];target=issue['target_id']
        if target=='ice.B1:segment0'and i in B1_OWNER:
            skipped.append(dict(frame=i,target_id=target,reason='separate physical B1 entry/exit producer owns this target'))
        else:affected[i].add(target)
    extractor=Extractor(RD,MODEL);install_user_guide(extractor)
    changes=[];archive=OUT/'bilinear_context_before_reread';archive.mkdir(exist_ok=True,parents=True)
    for i in range(201,1249):
        path=RD/f'physical_wide_proposals/{i:06d}.json';packet=read(path)
        if i in affected:
            before=deepcopy(packet);before_sha=sha(path)
            source=packet['source'];raw,mask=context_mask(source,i)
            native=PhysicalBandProfiles(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),source,mask,extractor.cameras[i])
            replacements={};coverage_replacements={}
            for p in extractor.paths:
                target='ice.'+p['curve_id']
                if target not in affected[i]:continue
                guide,origin=extractor.guide(p,i)
                if len(guide)<2:raise ValueError(f'Previously measured target has no guide: {i}/{target}')
                guide,origin,policy=extractor.line_identity.select_guide(p,i,guide,origin)
                f,_,_=extractor.line(dict(p,_line_identity_policy=policy),guide,native)
                f['navigation_origin']=origin;replacements[target]=f
                coverage_replacements[target]=dict(target_id=target,points=f['point_count'],navigation_origin=origin,native_material_diagnostic=f.get('extraction_diagnostic'))
            if set(replacements)!=affected[i]:raise ValueError('Could not find every affected model target')
            packet['features']=[replacements.get(f['target_id'],f)for f in packet['features']]
            packet['features']=[f for f in packet['features']if f['point_count']]
            old_targets={f['target_id']for f in before['features']}
            for target,f in replacements.items():
                if target not in old_targets and f['point_count']:packet['features'].append(f)
            packet['per_feature_coverage']=[coverage_replacements.get(c['target_id'],c)for c in packet['per_feature_coverage']]
            present={c['target_id']for c in packet['per_feature_coverage']}
            packet['per_feature_coverage'] += [c for t,c in coverage_replacements.items()if t not in present]
            packet['point_count']=sum(f['point_count']for f in packet['features'])
            untouched_before=[f for f in before['features']if f['target_id']not in affected[i]]
            untouched_after=[f for f in packet['features']if f['target_id']not in affected[i]]
            coverage_before=[c for c in before['per_feature_coverage']if c['target_id']not in affected[i]]
            coverage_after=[c for c in packet['per_feature_coverage']if c['target_id']not in affected[i]]
            assert untouched_before==untouched_after and coverage_before==coverage_after
            assert packet['source']==before['source']
            if not (archive/f'{i:06d}.json').exists():write(archive/f'{i:06d}.json',before)
            write(path,packet)
            changes.append(dict(frame=i,targets=sorted(affected[i]),before_sha256=before_sha,after_sha256=sha(path),
                before_point_count=before['point_count'],after_point_count=packet['point_count'],
                untouched_features_sha256=digest(untouched_before),untouched_coverage_sha256=digest(coverage_before),
                unchanged_non_affected_features=True,unchanged_non_affected_coverage=True,
                target_counts=[dict(target_id=t,before=next((f['point_count']for f in before['features']if f['target_id']==t),0),after=replacements[t]['point_count'])for t in sorted(affected[i])]))
            print('reread',i,changes[-1]['target_counts'],flush=True)
        # Replay every accepted frame, including unchanged proposals, so a local
        # re-read never resets physical identity to an unrelated model branch.
        extractor.line_identity.record(i,packet['features'])
    all_changes=prior.get('changes',[])+changes
    write(receipt_path,dict(
        schema='rinkspace.target-only-physical-wide-context-reread.v1',
        audit_sha256=audit_digest,sequential_replay=[201,1248],changed_frames=len({c['frame']for c in all_changes}),
        changes=all_changes,skipped_separate_owner_targets=skipped,
        replay_batches=prior.get('replay_batches',[])+[dict(audit_sha256=audit_digest,changed_targets=len(changes))],
        only_physical_wide_proposals_modified=True,circles_thin_final_packets_and_models_not_modified=True,
        completed_utc=datetime.now(timezone.utc).isoformat()))


def render_changes():
    out=OUT/'context_reread_source_pages';out.mkdir(exist_ok=True,parents=True)
    receipt=read(OUT/'physical_wide_context_reread_receipt.json');cells=[];records=[]
    for change in receipt['changes']:
        i=change['frame'];packet=read(RD/f'physical_wide_proposals/{i:06d}.json')
        raw=cv2.imread(packet['source']['image_path'])
        for target in change['targets']:
            f=next(f for f in packet['features']if f['target_id']==target)
            points=np.vstack([s['uv']for s in f['segments']]);center=points.mean(0)
            tangent=np.linalg.svd(points-center,full_matrices=False)[2][0]
            if tangent[0]<0:tangent=-tangent
            normal=np.array([-tangent[1],tangent[0]])
            along=(points-center)@tangent;lo,hi=along.min()-25,along.max()+25
            rows=f['source_line_fit']['direct_readouts']
            sides=np.vstack([r['source_evidence']['actual_source_sides_uv']for r in rows])
            margin=max(45.,min(95.,float(np.max(abs((sides-center)@normal)))+18))
            length=int(np.ceil(hi-lo));height=int(np.ceil(margin*2))
            grid=center+np.linspace(lo,hi,length)[None,:,None]*tangent+np.linspace(-margin,margin,height)[:,None,None]*normal
            strip=cv2.remap(raw,grid[...,0].astype(np.float32),grid[...,1].astype(np.float32),cv2.INTER_LINEAR)
            marked=strip.copy()
            def xy(uv):
                p=np.asarray(uv)-center
                return np.rint(np.c_[((p@tangent)-lo)*(length-1)/(hi-lo),((p@normal)+margin)*(height-1)/(2*margin)]).astype(int)
            for row in rows:
                ends=xy(row['source_evidence']['actual_source_sides_uv'])
                for point in ends:cv2.circle(marked,tuple(point),2,(0,220,220),-1)
                cv2.circle(marked,tuple(xy([row['uv']])[0]),2,(0,0,255),-1)
            for seg in f['segments']:cv2.polylines(marked,[xy(seg['uv'])],False,(255,0,255),1)
            cell=np.full((290,960,3),30,np.uint8)
            cv2.putText(cell,f'{i} {target} {f["point_count"]} rep / {len(rows)} source',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            for pic,y in ((strip,28),(marked,158)):
                ratio=min(944/pic.shape[1],124/pic.shape[0]);small=cv2.resize(pic,None,fx=ratio,fy=ratio)
                cell[y:y+small.shape[0],8:8+small.shape[1]]=small
            if i in (452,1217,1218):
                native=raw.copy()
                for row in rows:
                    ends=np.rint(row['source_evidence']['actual_source_sides_uv']).astype(int)
                    cv2.line(native,tuple(ends[0]),tuple(ends[1]),(0,220,220),1)
                    for point in ends:cv2.circle(native,tuple(point),2,(0,220,220),-1)
                    cv2.circle(native,tuple(np.rint(row['uv']).astype(int)),2,(0,0,255),-1)
                for s in f['segments']:cv2.polylines(native,[np.rint(s['uv']).astype(int)],False,(255,0,255),1)
                source_uv=np.vstack([points,sides]);start=np.maximum(0,np.floor(source_uv.min(0)-50).astype(int));end=np.minimum([1920,1080],np.ceil(source_uv.max(0)+50).astype(int))
                x0,y0=start;x1,y1=end
                cv2.imwrite(str(out/f'{i:06d}_native.png'),np.hstack([raw[y0:y1,x0:x1],native[y0:y1,x0:x1]]))
            cells.append(cell);records.append(dict(frame=i,target_id=target,feature_sha256=digest(f),packet_sha256=sha(RD/f'physical_wide_proposals/{i:06d}.json')))
    pages=[]
    for pos in range(0,len(cells),6):
        canvas=np.full((290*3,1920,3),20,np.uint8)
        for k,cell in enumerate(cells[pos:pos+6]):canvas[(k//2)*290:(k//2+1)*290,(k%2)*960:(k%2+1)*960]=cell
        dest=out/f'page_{pos//6+1:02d}.jpg';cv2.imwrite(str(dest),canvas,[cv2.IMWRITE_JPEG_QUALITY,97])
        pages.append(dict(path=str(dest),sha256=sha(dest),records=records[pos:pos+6]))
    write(out/'render_manifest.json',dict(pages=pages,source_only_rows_current_native=True,not_visual_acceptance=True))
    print('rendered source change pages',len(pages),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');parser.add_argument('--apply',action='store_true');parser.add_argument('--render',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.audit:audit()
    if args.apply:apply()
    if args.render:render_changes()
