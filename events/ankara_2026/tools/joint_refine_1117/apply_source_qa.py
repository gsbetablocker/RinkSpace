"""Apply documented source admission and measured paint-centre corrections."""
from collections import Counter
from argparse import ArgumentParser
from pathlib import Path
import shutil
import numpy as np
from shared import DATA,FIELD,OUT,arrays,read,save,write,source_corrections

EXCLUDE={
 'nayeon_s04':[57354,57355,57356,57357],
 'nayeon_s02':[192426,192427,192428],
 'nayeon_s06':[262711,262712],
 'nayeon_s10':[149753,149754],
 'jin_s12':[672,673,674,675,676,677],
 'jin_s14':[3334],
 'okada_s02':[1274,1275],
}


def apply(name,a):
    a=source_corrections(name,a)
    before=a['fold'].copy();old_uv=a['uv'].copy()
    reason=np.full(len(before),'',dtype='<U100')
    paired=np.full((len(before),2,2),np.nan);strength=np.full((len(before),2),np.nan)
    p=OUT/'red_paint_source'/(name+'.npz')
    if p.exists():
        qa=arrays(p);ix=qa['selected_row']
        np.testing.assert_array_equal(a['origin_row'][ix],qa['origin_row'])
        np.testing.assert_array_equal(a['pts'][ix],qa['pts'])
        good=qa['source_valid'];confirmed=qa['old_flanks_confirmed']
        if name.startswith(('jin_','okada_')):
            a['uv'][ix[good]]=qa['proposed_uv'][good]
            paired[ix[good]]=qa['source_bracket'][good]
            strength[ix[good]]=qa['source_side_contrast'][good]
            reason[ix[good]]='source_two_flanks_replace_legacy_unverified_edge_or_midpoint'
            bad=~good
        else:
            # Existing complete flank pairs remain original measurements when
            # the independent reread confirms them. Do not chase every noisy
            # alternative half-contrast crossing in a blurred stroke.
            bad=~(good|confirmed)
        a['fold'][ix[bad]]=5
        reason[ix[bad]]='raw_source_cannot_resolve_two_red_paint_flanks'
    # Reject ice labels falling above the measured far-board foot. This uses
    # source rail pixels at the same PTS, never the fitted rink projection.
    for pt in np.unique(a['pts']):
        foot=a['uv'][(a['pts']==pt)&(a['family']=='board.bottom')&(a['fold']<5)]
        if len(foot)<4:continue
        foot=foot[np.argsort(foot[:,0])]
        use=(a['pts']==pt)&~np.char.startswith(a['family'],'Ad')&~np.char.startswith(a['family'],'board.')&(a['fold']<5)
        ix=np.flatnonzero(use)
        if not len(ix):continue
        y=np.interp(a['uv'][ix,0],foot[:,0],foot[:,1])
        inside=(a['uv'][ix,0]>=foot[0,0])&(a['uv'][ix,0]<=foot[-1,0])
        bad=inside&(a['uv'][ix,1]<y-1.)
        a['fold'][ix[bad]]=5;reason[ix[bad]]='ice_label_above_same_exposure_measured_board_foot'
    for origin in EXCLUDE.get(name,[]):
        use=a['origin_row']==origin
        a['fold'][use]=5;reason[use]='manual_original_frame_identity_or_width_is_ambiguous'
    use=(a['fold']!=before)|np.any(a['uv']!=old_uv,axis=1)
    change=dict(origin_row=a['origin_row'][use],pts=a['pts'][use],path=a['path'][use],
        old_uv=old_uv[use],uv=a['uv'][use],old_fold=before[use],fold=a['fold'][use],
        reason=reason[use],source_bracket=paired[use],source_side_contrast=strength[use])
    return a,change


def run(audit_only=False):
    backup=OUT/'prepared_before_source_qa';backup.mkdir(exist_ok=True)
    folder=OUT/'source_admission';folder.mkdir(exist_ok=True);report={}
    setup=read(OUT/'prepared/setup.json');base=arrays(OUT/'prepared/baseline.npz')
    audit_root=DATA/'events/ankara_2026/scene/trapezoid_two_ends_1116/full_audit'
    audit=read(audit_root/'trapezoid_full_audit.json')
    path_by_branch={p['generated_code']-420:p['id'] for p in read(FIELD)['paths']
                    if p.get('generated_code') in range(420,424)}
    extra_manifest={}
    for name,s in setup['shots'].items():
        src=OUT/'prepared'/(name+'.npz');prior=backup/src.name
        if not prior.exists():shutil.copy2(src,prior)
        a,changes=apply(name,arrays(prior))
        if s['program'] in ('jin','okada','chen'):
            origin=audit_root/(s['program']+'_source_trials.npz');tr=arrays(origin)
            assert tr['source_sha256'].item()==s['source_sha256']
            use=(tr['shot']==s['shot'])&np.isfinite(tr['source_bracket']).all(axis=(1,2))
            ambiguous=audit['programmes'][s['program']]['ambiguous_proposals']
            for item in ambiguous:
                if item['shot']==s['shot']:
                    branch=audit['branch_axis_order'].index(item['branch'])
                    use&=~((tr['branch']==branch)&np.isin(tr['pts'],item['pts']))
            ix=np.flatnonzero(use)
            if len(ix):
                # Existing independently read width pairs replace earlier
                # sparse trapezoid labels; all rejected historical rows remain.
                retired=np.char.startswith(a['family'],'trapezoid')&(a['fold']!=5)
                count=int(retired.sum())
                superseded=dict(origin_row=a['origin_row'][retired],pts=a['pts'][retired],path=a['path'][retired],
                    old_uv=a['uv'][retired],uv=a['uv'][retired],old_fold=a['fold'][retired],
                    fold=np.full(count,5,dtype=a['fold'].dtype),
                    reason=np.repeat('superseded_by_source_reviewed_two_flank_trapezoid',count),
                    source_bracket=np.full((count,2,2),np.nan),source_side_contrast=np.full((count,2),np.nan))
                changes={k:np.concatenate((changes[k],superseded[k])) for k in changes}
                a['fold'][retired]=5
                pts=tr['pts'][ix];frame=np.searchsorted(base[name+'_pts'],pts)
                np.testing.assert_array_equal(base[name+'_pts'][frame],pts)
                old=arrays(s['observations']);fold=np.zeros(len(ix),np.int8)
                times,first=np.unique(old['pts'],return_index=True)
                exact=np.searchsorted(times,pts);known=(exact<len(times))
                known&=times[np.minimum(exact,len(times)-1)]==pts
                fold[known]=old['fold'][first[exact[known]]]
                values=dict(pts=pts,idx=frame,uv=tr['uv'][ix],
                    path=np.array([path_by_branch[int(b)] for b in tr['branch'][ix]]),
                    station=tr['station'][ix],fold=fold,
                    family=np.array(['trapezoid_'+('left' if b<2 else 'right') for b in tr['branch'][ix]]),
                    material=np.repeat('source_reviewed_two_flank_trapezoid',len(ix)),origin_row=-1-ix)
                # Keep original UVs and thin spatially and in time, as for the
                # existing dense source geometry grid.
                take=np.flatnonzero((frame%10==0)|np.isin(frame,[frame.min(),frame.max()]))
                key=np.rec.fromarrays((pts[take],values['path'][take],
                     np.floor(values['uv'][take,0]/8).astype(int),np.floor(values['uv'][take,1]/8).astype(int)))
                _,unique=np.unique(key,return_index=True);take=take[np.sort(unique)]
                a={k:np.concatenate((a[k],values[k][take])) for k in a}
                extra_manifest[name]=dict(source=str(origin),raw_rows=ix[take].tolist(),
                    selected_rows=len(take),identity='negative origin_row is -1 minus source_trials row',
                    source_sha256=s['source_sha256'],source_review=str(audit_root/'trapezoid_full_audit.json'))
        if audit_only:
            current=arrays(src)
            for key,value in a.items():np.testing.assert_array_equal(value,current[key],err_msg=name+'/'+key)
        else:save(src,**a)
        save(folder/(name+'.npz'),**changes)
        report[name]=dict(changed_rows=len(changes['pts']),fit_rows=int((a['fold']==0).sum()),
            reasons=dict(Counter(changes['reason'].tolist())))
    write(folder/'audit.json',dict(shots=report,
        source_identity_review=str(OUT/'circle_source/manifest.json'),
        source_profile_review=str(OUT/'red_paint_source/audit.json'),
        user_visual_acceptance=False,
        prior_packets=str(backup),original_packets_unchanged=True))
    write(folder/'extra_sources.json',extra_manifest)
    print({p:sum(v['changed_rows'] for k,v in report.items() if k.startswith(p+'_'))
           for p in ('jin','okada','chen','nayeon')})


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--audit-only',action='store_true')
    run(ap.parse_args().audit_only)
