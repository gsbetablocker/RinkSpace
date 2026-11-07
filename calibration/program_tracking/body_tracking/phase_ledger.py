"""Source-phase masks; no model static probabilities or hard-coded shot times."""
import numpy as np

def phase_weights(reference,pts,*,default=1.,stable=1.,transition=.1):
    pts=np.asarray(pts);out=np.full(len(pts),float(default))
    supported=np.zeros(len(pts),bool);transition_mask=np.zeros(len(pts),bool)
    ledger=getattr(reference,'source_document',{}).get('source_episode_review',{})
    for episode in ledger.get('episodes',[]):
        mask=(pts>=episode['start_pts'])&(pts<=episode['end_pts_inclusive']);kind=episode['kind']
        if kind=='support_candidate':supported|=mask
        if kind=='flight_visible_candidate'or'transition'in kind or'release'in episode.get('episode_id',''):transition_mask|=mask
    out[supported]=stable
    out[transition_mask]=transition
    return out

def flight_mask(reference,pts):
    pts=np.asarray(pts);out=np.zeros(len(pts),bool)
    for e in getattr(reference,'source_document',{}).get('source_episode_review',{}).get('episodes',[]):
        if e['kind']=='flight_visible_candidate':out|=(pts>=e['start_pts'])&(pts<=e['end_pts_inclusive'])
    return out
