"""Extend source-only identities with independently verified extra exposures."""
from collections import defaultdict,Counter
from pathlib import Path
import json
import av,cv2,numpy as np
from .compact_identity_revision import _array
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _ref
from .source_signed_interface_batch_v1 import signed_candidates
from .source_seed_correspondence_v1 import correlation
from .source_worker import _channels
from .io_utils import write_json


def extend(*,packet,model,source_config,source_evidence,atlas,old_registrations,new_registrations,prior_review,row_start,output):
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    if Path(output).exists():raise FileExistsError('Immutable extension exists')
    pk=read(packet);m=read(model);cfg=read(source_config);ev=read(source_evidence);a=read(atlas);prior=read(prior_review)
    old={x['native_pts']for x in read(old_registrations)if x['candidates']};regs={int(x['native_pts']):x['candidates'][0]for x in read(new_registrations)if x['candidates']and x['native_pts']not in old}
    arrays={k:_array(pk['arrays'][k])for k in ['uv','native_pts','material_index','path','admission']};seed=_array(ev['columns']['seed_index']);normal=_array(ev['columns']['normal']);profile=_array(ev['columns']['profile'])
    ids=[x['raw_row']for x in prior['unresolved']if x['reason']=='source_registration_unconfirmed'and int(arrays['native_pts'][x['raw_row']])in regs and arrays['admission'][x['raw_row']]];identities,_=_source_identities(Path(packet),ids);by_pts=defaultdict(list)
    for i in ids:by_pts[int(arrays['native_pts'][i])].append(i)
    expected=defaultdict(set)
    for x in prior['rows']:expected[x['seed_index']].add((x['new_label'],x['new_path']))
    blocked=set(prior['conflicting_seed_identities']);indices={x['label']:i for i,x in enumerate(m['materials'])};ref=cv2.imread(a['source_image']['file']);channels=_channels(ref[:,:,::-1]);reference=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY);tentative=[];unresolved=[];new_conflicts=set();seen=set()
    if by_pts:
      with av.open(cfg['source_video'])as con:
        stream=con.streams.video[0];con.seek(min(by_pts),stream=stream,backward=True)
        for frame in con.decode(stream):
            pt=int(frame.pts)
            if pt>max(by_pts):break
            if pt not in by_pts:continue
            seen.add(pt);rr=by_pts[pt];ll=np.asarray(rr)-row_start;reg=regs[pt];cc=signed_candidates(channels,a['paths'],reg['H'],arrays['uv'][rr],normal[ll],profile[ll]);gray=frame.to_ndarray(format='gray')
            for i,li,candidates in zip(rr,ll,cc):
                si=int(seed[li]);reason=None
                if si in blocked:reason='prior_seed_identity_conflict'
                elif not candidates:reason='no_same_polarity_original_interface'
                elif len(candidates)>1 and candidates[1]['distance_px']-candidates[0]['distance_px']<3.:reason='ambiguous_same_polarity_original_interface'
                else:
                    c=candidates[0];corr=correlation(reference,gray,np.asarray(reg['H']),c['source_foot'])
                    if not np.isfinite(corr)or corr<.65:reason='current_local_RGB_unconfirmed'
                    elif expected[si]and (c['label'],c['path'])not in expected[si]:reason='contradicts_prior_seed_identity';new_conflicts.add(si)
                if reason:unresolved.append(dict(raw_row=i,source_row_id=identities[i],reason=reason,original_row_retained=True));continue
                mi=int(arrays['material_index'][i]);tentative.append(dict(raw_row=i,source_row_id=identities[i],native_pts=pt,uv=arrays['uv'][i].tolist(),old_label=m['materials'][mi]['label'],old_material_index=mi,old_path=int(arrays['path'][i]),new_label=c['label'],new_material_index=indices[c['label']],new_path=c['path'],new_station=c['source_station'],source_station_is_numeric_initializer=True,seed_index=si,source_gradient_channel=int(profile[li,3]),measured_interface_polarity=float(profile[li,4]),canonical_source_gradient=c['source_gradient'],nearest_px=c['distance_px'],next_same_polarity_boundary_px=candidates[1]['distance_px']if len(candidates)>1 else None,local_source_pixel_correlation=corr,registration_correlation=reg['correlation'],identity_basis='actual_original_image_bridge_and_consistent_signed_material_interface',source_3D_projection_used=False))
    if seen!=set(by_pts):raise ValueError('Native source readout incomplete')
    choices=defaultdict(set)
    for x in tentative:choices[x['seed_index']].add((x['new_label'],x['new_path']))
    conflicts=new_conflicts|{si for si,v in choices.items()if len(v)>1};rows=[]
    for x in tentative:
        if x['seed_index']in conflicts:unresolved.append(dict(raw_row=x['raw_row'],source_row_id=x['source_row_id'],reason='source_seed_identity_conflict',original_row_retained=True));continue
        if (x['old_material_index'],x['old_path'])!=(x['new_material_index'],x['new_path']):rows.append(x)
    value=dict(schema='rinkfit.native_print_identity_rebind_review.v1',source_model_projection_used=False,source_pixels_changed=False,source_video_sha256=cfg['source_video_sha256'],native_time_base='1/12800',evidence=[_ref(x)for x in [__file__,packet,source_config,source_evidence,atlas,old_registrations,new_registrations,prior_review,a['source_image']['file']]],selected_actual_source_rows=len(ids),new_source_registered_exposures=len(regs),source_seed_conflicts_with_prior=sorted(new_conflicts),rows=rows,unresolved=unresolved,unconfirmed_source_rows_retained=True,source_rows_deleted=0)
    write_json(output,value);return dict(review=_ref(output),selected_rows=len(ids),rebound_rows=len(rows),unresolved_rows=len(unresolved),conflicts_with_prior=sorted(new_conflicts),reasons=dict(Counter(x['reason']for x in unresolved)))
