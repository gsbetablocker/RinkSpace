"""Original-image boundary identity using the recorded channel and polarity.

This consumes an explicitly reviewed seed identity. It cannot infer a material
from a fitted camera, optimize pixels, relax the existing image checks, or
change membership. Opposite interfaces near a narrow stroke remain distinct.
"""
from pathlib import Path
import json
import av, cv2, numpy as np
from .compact_identity_revision import _array
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _ref
from .source_seed_correspondence_v1 import distances, foot, correlation
from .source_worker import _channels, _sample
from .io_utils import write_json


def compatible_boundaries(reference_channels, paths, H, uv, normal, channel, polarity):
    """Return actual source interfaces passing the unchanged 3px/.75 rules."""
    H=np.asarray(H,float);inverse=np.linalg.inv(H);uv=np.asarray(uv,float)
    normal=np.array(normal,float,copy=True);normal/=np.linalg.norm(normal)
    probe=cv2.perspectiveTransform(np.asarray([uv-normal*.5,uv+normal*.5])[None],inverse)[0]
    direction=probe[1]-probe[0];direction/=np.linalg.norm(direction)
    result=[]
    for j,path in enumerate(paths):
        poly=np.asarray(path['source_native_uv'],float)
        current=cv2.perspectiveTransform(poly[None],H)[0]
        dd,ss=distances(uv[None],current)
        if dd[0]>3.:continue
        current_foot=foot(current,ss[0])
        source_foot=cv2.perspectiveTransform(current_foot[None,None],inverse)[0,0]
        values=_sample(reference_channels,np.asarray([source_foot-direction*.5,source_foot+direction*.5]))
        slope=float(values[1,channel]-values[0,channel])
        if abs(slope)<.75 or np.sign(slope)!=polarity:continue
        _,station=distances(source_foot[None],poly)
        result.append(dict(canonical=j,label=path['label'],path=path['path'],
            distance_px=float(dd[0]),source_station=float(station[0]),
            source_gradient=slope,source_foot=source_foot.tolist()))
    return sorted(result,key=lambda x:x['distance_px'])


def build_review(*,packet,model,source_config,source_evidence,atlas,registrations,
                 seed_identities,row_start,output):
    """Review existing tracked rows against signed original-image interfaces.

    seed_identities maps seed index to the explicitly source-reviewed canonical
    label. Every emitted current pixel independently passes source registration,
    local RGB appearance, measured polarity, and the unchanged distance/margin.
    The full input raw denominator and every rejected row remain untouched.
    """
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    out=Path(output)
    if out.exists():raise FileExistsError('Immutable source identity review exists')
    pk=read(packet);m=read(model);cfg=read(source_config);ev=read(source_evidence);a=read(atlas)
    regs={int(x['native_pts']):x['candidates'][0]for x in read(registrations)if x['candidates']}
    arrays={k:_array(pk['arrays'][k])for k in ['uv','native_pts','material_index','path','admission']}
    seed_index=_array(ev['columns']['seed_index']);normal=_array(ev['columns']['normal']);profile=_array(ev['columns']['profile'])
    chosen={int(k):v for k,v in seed_identities.items()}
    local=np.flatnonzero(np.isin(seed_index,list(chosen)));rows=local+int(row_start)
    identities,_=_source_identities(Path(packet),rows.tolist())
    indices={x['label']:i for i,x in enumerate(m['materials'])}
    ref=cv2.imread(a['source_image']['file']);channels=_channels(ref[:,:,::-1]);gray_ref=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
    by_pts={}
    for li,i in zip(local,rows):
        si=int(seed_index[li]);pt=int(arrays['native_pts'][i])
        if identities[int(i)]!=f'native_named_print_1022/s02/{pt}/seed{si}':raise ValueError('Actual immutable row/seed identity differs')
        if arrays['admission'][i]!=1:continue
        by_pts.setdefault(pt,[]).append((int(li),int(i),si))
    accepted=[];unresolved=[]
    with av.open(cfg['source_video'])as con:
        stream=con.streams.video[0]
        if str(stream.time_base)!='1/12800':raise ValueError('Actual native source clock differs')
        con.seek(min(by_pts),stream=stream,backward=True)
        for frame in con.decode(stream):
            pt=int(frame.pts)
            if pt>max(by_pts):break
            if pt not in by_pts:continue
            image=frame.to_ndarray(format='gray');reg=regs.get(pt)
            for li,i,si in by_pts[pt]:
                uv=arrays['uv'][i];old=int(arrays['material_index'][i]);target=chosen[si]
                reason=None
                if reg is None or reg['correlation']<.65 or reg['registration_p95_px']>3.:reason='source_registration_unconfirmed'
                else:
                    candidates=compatible_boundaries(channels,a['paths'],reg['H'],uv,normal[li],int(profile[li,3]),float(profile[li,4]))
                    candidates=[x for x in candidates if x['label']==target or x['distance_px']<=3.]
                    if not candidates or candidates[0]['label']!=target:reason='reviewed_seed_does_not_match_current_signed_interface'
                    elif len(candidates)>1 and candidates[1]['distance_px']-candidates[0]['distance_px']<3.:reason='ambiguous_same_polarity_source_interface'
                    else:
                        candidate=candidates[0];corr=correlation(gray_ref,image,np.asarray(reg['H']),np.asarray(candidate['source_foot']))
                        if not np.isfinite(corr)or corr<.65:reason='current_local_RGB_unconfirmed'
                if reason:
                    unresolved.append(dict(raw_row=i,source_row_id=identities[i],reason=reason,original_row_retained=True));continue
                new=indices[target]
                if m['materials'][old]['label'].split('.')[0]!=target.split('.')[0]:raise ValueError('Cross-panel identity is forbidden')
                accepted.append(dict(raw_row=i,source_row_id=identities[i],native_pts=pt,uv=uv.tolist(),
                    old_label=m['materials'][old]['label'],old_material_index=old,old_path=int(arrays['path'][i]),
                    new_label=target,new_material_index=new,new_path=candidate['path'],new_station=candidate['source_station'],
                    source_station_is_numeric_initializer=True,seed_index=si,source_gradient_channel=int(profile[li,3]),
                    measured_interface_polarity=float(profile[li,4]),canonical_source_gradient=candidate['source_gradient'],
                    nearest_px=candidate['distance_px'],next_same_polarity_boundary_px=candidates[1]['distance_px']if len(candidates)>1 else None,
                    local_source_pixel_correlation=corr,registration_correlation=reg['correlation'],
                    identity_basis='reviewed_original_seed_and_current_RGB_signed_material_interface',
                    whole_panel_registration_or_3D_projection_used=False))
    if set(by_pts)!={int(arrays['native_pts'][e['raw_row']])for e in accepted+unresolved}:raise ValueError('Native image readout census incomplete')
    value=dict(schema='rinkfit.native_print_identity_rebind_review.v1',source_model_projection_used=False,
        source_pixels_changed=False,source_video_sha256=cfg['source_video_sha256'],native_time_base='1/12800',
        evidence=[_ref(x)for x in [__file__,packet,source_config,source_evidence,atlas,registrations,a['source_image']['file']]],
        reviewed_seed_identities=chosen,rows=accepted,unresolved=unresolved,
        unconfirmed_source_rows_retained=True,source_rows_deleted=0,
        qualification='Explicit original seed identity plus per-current-pixel signed original-image boundary and RGB checks. No fitted camera, relaxed thresholds, UV changes, or membership changes.')
    write_json(out,value)
    return dict(review=_ref(out),rebound_rows=len(accepted),unresolved_rows=len(unresolved))
