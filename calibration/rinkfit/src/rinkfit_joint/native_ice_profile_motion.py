"""Actual adjacent source-profile correspondence, never observed UV.

The complete signed material context chooses a unique normal translation. A
successful local peak alone cannot identify an adjacent parallel target layer.
"""
import numpy as np
from scipy.signal import find_peaks


def normal_material_profile_correspondence(previous_result,current_result,previous_uv,normal):
    old = previous_result['source_material_receipt']; new = current_result['source_material_receipt']
    def profile(r):
        values = np.asarray(r.get('actual_five_profile_channels',r.get('actual_five_profiles')),float)
        offsets = np.asarray(r.get('actual_five_profile_offsets',r.get('normal_offsets')),float)
        if values.ndim==2:values=values[:,:,None]
        if values.shape[:2]!=(5,len(offsets))or values.ndim!=3 or not np.isfinite(values).all():
            raise ValueError('Actual complete signed material profiles unavailable')
        return offsets,np.median(values,axis=0)
    xo,yo=profile(old); xn,yn=profile(new); n=np.asarray(normal,float);n/=np.linalg.norm(n)
    previous_uv=np.asarray(previous_uv,float); os=np.asarray(old['original_navigation_seed_uv']); ns=np.asarray(new['original_navigation_seed_uv'])
    xo=xo+float((os-previous_uv)@n);xn=xn+float((ns-previous_uv)@n)
    step=float(np.median(np.diff(xn))); radius=min(float(np.ptp(xo)),float(np.ptp(xn)))*.5
    shifts=np.arange(-radius,radius+step*.1,step);scores=[]
    for d in shifts:
        mask=(xn-d>=xo.min())&(xn-d<=xo.max())
        if mask.sum()<max(24,int(.6*len(xn))):scores.append(-1.);continue
        a=np.stack([np.interp(xn[mask]-d,xo,yo[:,j])for j in range(yo.shape[1])],axis=1);b=yn[mask]
        a=a-a.mean(0);b=b-b.mean(0);den=np.linalg.norm(a)*np.linalg.norm(b)
        scores.append(float((a*b).sum()/den)if den>1e-8 else -1.)
    scores=np.asarray(scores);best=int(np.argmax(scores));peaks,_=find_peaks(scores,prominence=.05)
    alternatives=[int(i)for i in peaks if abs(shifts[i]-shifts[best])>2.]
    competitor=max((float(scores[i])for i in alternatives),default=-1.)
    measured=float((np.asarray(current_result['current_uv'])-previous_uv)@n)
    unique=bool(scores[best]>=.9 and competitor<.9 and abs(measured-shifts[best])<=2.)
    return dict(schema='rinkfit.adjacent_signed_material_profile_motion.v1',
        unique_source_material_profile_match=unique,best_normal_translation_px=float(shifts[best]),
        best_full_context_correlation=float(scores[best]),competing_distinct_material_peak_correlation=competitor,
        current_actual_reader_normal_translation_px=measured,
        readout_to_profile_translation_disagreement_px=abs(measured-float(shifts[best])),
        current_material_context_is_actual_RGB=True,profile_coordinates_are_observations=False,
        physical_identity_and_layer_review_still_required=True)
