"""Source-only batch boundary identities; no fitted projections or row removal."""
from collections import Counter, defaultdict
from pathlib import Path
import json,re
import av,cv2,numpy as np
from .compact_identity_revision import _array
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _ref
from .source_seed_correspondence_v1 import distances,correlation
from .source_worker import _channels,_sample
from .io_utils import write_json


def signed_candidates(channels,paths,H,uv,normals,profiles):
    """Original-image paths within 3px, with actual measured channel/polarity."""
    H=np.asarray(H,float);inverse=np.linalg.inv(H);uv=np.asarray(uv,float)
    normals=np.array(normals,float,copy=True);normals/=np.linalg.norm(normals,axis=1)[:,None]
    lo=cv2.perspectiveTransform((uv-normals*.5)[None],inverse)[0]
    hi=cv2.perspectiveTransform((uv+normals*.5)[None],inverse)[0]
    direction=hi-lo;direction/=np.linalg.norm(direction,axis=1)[:,None]
    result=[[]for _ in uv]
    for j,path in enumerate(paths):
        poly=np.asarray(path['source_native_uv'],float)
        current=cv2.perspectiveTransform(poly[None],H)[0]
        dd,ss=distances(uv,current);indices=np.flatnonzero(dd<=3.)
        if not len(indices):continue
        z=ss[indices]*(len(current)-1);seg=np.minimum(z.astype(int),len(current)-2)
        feet=current[seg]+(z-seg)[:,None]*(current[seg+1]-current[seg])
        original=cv2.perspectiveTransform(feet[None],inverse)[0]
        minus=_sample(channels,original-direction[indices]*.5)
        plus=_sample(channels,original+direction[indices]*.5)
        slopes=(plus-minus)[np.arange(len(indices)),profiles[indices,3].astype(int)]
        _,stations=distances(original,poly)
        valid=(np.abs(slopes)>=.75)&(np.sign(slopes)==profiles[indices,4])
        for k in np.flatnonzero(valid):
            i=int(indices[k]);result[i].append(dict(canonical=j,label=path['label'],path=path['path'],distance_px=float(dd[i]),source_station=float(stations[k]),source_gradient=float(slopes[k]),source_foot=original[k]))
    return [sorted(x,key=lambda z:z['distance_px'])for x in result]


def review(*,packet,model,source_config,source_evidence,atlas,registrations,seed_label_pattern,row_start,output):
    """Require each seed's source-supported identity to agree across exposures.

    Selection is by original source-label structure, never cost or model error.
    Every emitted pixel independently satisfies unchanged distance, polarity,
    margin, registration and local RGB thresholds. Failed rows stay untouched.
    """
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    if Path(output).exists():raise FileExistsError('Immutable review exists')
    pk=read(packet);m=read(model);cfg=read(source_config);ev=read(source_evidence);a=read(atlas)
    regs={int(x['native_pts']):x['candidates'][0]for x in read(registrations)if x['candidates']}
    arrays={k:_array(pk['arrays'][k])for k in ['uv','native_pts','material_index','path','admission']}
    seed=_array(ev['columns']['seed_index']);normal=_array(ev['columns']['normal']);profile=_array(ev['columns']['profile'])
    chosen=[i for i,x in enumerate(cfg['seeds'])if re.fullmatch(seed_label_pattern,x['label'])]
    local=np.flatnonzero(np.isin(seed,chosen));rows=local+int(row_start)
    identities,_=_source_identities(Path(packet),rows.tolist());indices={x['label']:i for i,x in enumerate(m['materials'])}
    ref=cv2.imread(a['source_image']['file']);channels=_channels(ref[:,:,::-1]);gray_ref=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
    by_pts=defaultdict(list);unresolved=[];seen=set();tentative=[]
    for li,i in zip(local,rows):
        si=int(seed[li]);pt=int(arrays['native_pts'][i]);sid=identities[int(i)]
        if not sid.startswith(ev['source_dataset_id']+'/')or not sid.endswith(f'/{pt}/seed{si}'):raise ValueError('Actual immutable row/seed identity differs')
        if arrays['admission'][i]!=1:continue
        by_pts[pt].append((int(li),int(i),si))
    if by_pts:
      with av.open(cfg['source_video'])as con:
        stream=con.streams.video[0]
        if str(stream.time_base)!='1/12800':raise ValueError('Native source clock differs')
        con.seek(min(by_pts),stream=stream,backward=True)
        for frame in con.decode(stream):
            pt=int(frame.pts)
            if pt>max(by_pts):break
            if pt not in by_pts:continue
            seen.add(pt);entries=by_pts[pt];reg=regs.get(pt);image=frame.to_ndarray(format='gray')
            if reg is None or reg['correlation']<.65 or reg['registration_p95_px']>3.:
                unresolved.extend(dict(raw_row=i,source_row_id=identities[i],reason='source_registration_unconfirmed')for li,i,si in entries);continue
            ii=np.array([i for li,i,si in entries]);ll=np.array([li for li,i,si in entries]);candidates=signed_candidates(channels,a['paths'],reg['H'],arrays['uv'][ii],normal[ll],profile[ll]);corrs={}
            for (li,i,si),cc in zip(entries,candidates):
                reason=None
                if not cc:reason='no_same_polarity_original_interface'
                elif len(cc)>1 and cc[1]['distance_px']-cc[0]['distance_px']<3.:reason='ambiguous_same_polarity_original_interface'
                else:
                    c=cc[0];q=c['source_foot'];box=tuple(np.r_[np.floor(q-15),np.ceil(q+16)].astype(int));corr=corrs.get(box)
                    if corr is None:corr=correlation(gray_ref,image,np.asarray(reg['H']),q);corrs[box]=corr
                    if not np.isfinite(corr)or corr<.65:reason='current_local_RGB_unconfirmed'
                if reason:unresolved.append(dict(raw_row=i,source_row_id=identities[i],reason=reason));continue
                old=int(arrays['material_index'][i]);prefix=m['materials'][old]['label'].split('.')[0];label=prefix+'.'+c['label'].split('.',1)[1]
                if label not in indices:unresolved.append(dict(raw_row=i,source_row_id=identities[i],reason='canonical_same_panel_material_absent'));continue
                tentative.append(dict(raw_row=i,source_row_id=identities[i],native_pts=pt,uv=arrays['uv'][i].tolist(),old_label=m['materials'][old]['label'],old_material_index=old,old_path=int(arrays['path'][i]),new_label=label,new_material_index=indices[label],new_path=c['path'],new_station=c['source_station'],source_station_is_numeric_initializer=True,seed_index=si,source_gradient_channel=int(profile[li,3]),measured_interface_polarity=float(profile[li,4]),canonical_source_gradient=c['source_gradient'],nearest_px=c['distance_px'],next_same_polarity_boundary_px=cc[1]['distance_px']if len(cc)>1 else None,local_source_pixel_correlation=corr,registration_correlation=reg['correlation'],identity_basis='consistent_source_seed_and_per_exposure_signed_original_material_interface',whole_panel_registration_or_3D_projection_used=False))
    if seen!=set(by_pts):raise ValueError('Native source image readout incomplete')
    identities_by_seed=defaultdict(set)
    for x in tentative:identities_by_seed[x['seed_index']].add((x['new_label'],x['new_path']))
    conflicting={si for si,v in identities_by_seed.items()if len(v)>1};accepted=[];already_correct=0
    for x in tentative:
        if x['seed_index']in conflicting:unresolved.append(dict(raw_row=x['raw_row'],source_row_id=x['source_row_id'],reason='source_seed_identity_conflict'));continue
        if (x['old_material_index'],x['old_path'])==(x['new_material_index'],x['new_path']):already_correct+=1;continue
        accepted.append(x)
    for x in unresolved:x['original_row_retained']=True
    value=dict(schema='rinkfit.native_print_identity_rebind_review.v1',source_model_projection_used=False,source_pixels_changed=False,source_video_sha256=cfg['source_video_sha256'],native_time_base='1/12800',evidence=[_ref(x)for x in [__file__,packet,source_config,source_evidence,atlas,registrations,a['source_image']['file']]],source_seed_selection_pattern=seed_label_pattern,selected_source_seeds=len(chosen),selected_actual_source_rows=len(rows),active_actual_source_rows=sum(map(len,by_pts.values())),already_correct_rows=already_correct,conflicting_seed_identities=sorted(conflicting),rows=accepted,unresolved=unresolved,unconfirmed_source_rows_retained=True,source_rows_deleted=0,qualification='Consistent source-only canonical seed identity with per-current-pixel signed RGB interface; unchanged 3px distance/margin, .75 gradient, .65 correlation and 3px registration bounds.')
    write_json(output,value)
    return dict(review=_ref(output),rebound_rows=len(accepted),already_correct_rows=already_correct,unresolved_rows=len(unresolved),selected_rows=len(rows),conflicting_seeds=len(conflicting),unresolved_reasons=dict(Counter(x['reason']for x in unresolved)))
