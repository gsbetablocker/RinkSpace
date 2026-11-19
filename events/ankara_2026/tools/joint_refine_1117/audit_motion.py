"""Dense actual-source displacement checks on original and selected cameras."""
from argparse import ArgumentParser
import numpy as np
from shared import OUT, PREPARED, PROGRAMMES, arrays, read, save, sha, write
from review import camera_state, native_for
from motion_tracks import load
from solve import stats


def one(candidate,programme,source_visibility=False):
    setup=read(PREPARED/'setup.json');base=camera_state(candidate)
    native=native_for(candidate,programme);spans={};slots={};camera_checks={}
    for name,s in setup['shots'].items():
        if s['program']!=programme:continue
        r=base[name+'_rigs'];c=base[name+'_controls'];times=base[name+'_times']
        unique,inv=np.unique(r,axis=0,return_inverse=True)
        spans[name]=(0,len(c));slots[name]=dict(start=0,stop=len(unique),inverse=inv)
        statuses=[];focal=[]
        for i,rig in enumerate(unique):
            z=native.core.evaluate_motion_camera(rig,c,s['first'],s['dt'],times[inv==i])
            statuses.extend(z['status']);focal.extend(z['fx'])
        camera_checks[name]=dict(exposures=len(times),invalid=int(np.count_nonzero(statuses)),
            focal_px=[float(min(focal)),float(max(focal))])
        if camera_checks[name]['invalid']:raise ValueError((candidate,name,'invalid source exposure camera'))
    tracks,manifest=load(setup,base,spans,slots,native=native,current_depths=True,all_development=True,source_visibility=source_visibility)
    error=np.full(len(tracks['world']),np.inf);frames=np.zeros(len(error),np.int64)
    for name in spans:
        s=setup['shots'][name];ix=np.flatnonzero(tracks['shot']==name)
        index=np.searchsorted(base[name+'_pts'],tracks['pts'][ix,1]);frames[ix]=index
        np.testing.assert_array_equal(base[name+'_pts'][index],tracks['pts'][ix,1])
        unique,inv=np.unique(base[name+'_rigs'][index],axis=0,return_inverse=True)
        for i,rig in enumerate(unique):
            take=ix[inv==i];ind=frames[take]
            z=native.core.project_motion(rig,base[name+'_controls'],s['first'],s['dt'],1080,
                base[name+'_times'][ind],tracks['world'][take])
            valid=z['status']==0
            error[take[valid]]=np.linalg.norm(z['uv'][valid]-tracks['uv'][take[valid],1],axis=1)
    return dict(shot=tracks['shot'],source_row=tracks['source_row'],pts=tracks['pts'][:,1],
        error=error),dict(cameras=camera_checks,conditional_depths=manifest)


def run(candidate,source_visibility=False):
    result={};records={}
    for programme in PROGRAMMES:
        old,before=one('original',programme,source_visibility);new,after=one(candidate,programme,source_visibility)
        key=lambda a:np.rec.fromarrays((a['shot'],a['source_row']))
        _,i,j=np.intersect1d(key(old),key(new),return_indices=True)
        shots={}
        for name in np.unique(new['shot'][j]):
            use=new['shot'][j]==name
            shots[name]=dict(original=stats(old['error'][i[use]]),candidate=stats(new['error'][j[use]]),
                covered_right_exposures=len(np.unique(new['pts'][j[use]])))
        result[programme]=dict(shots=shots,original=stats(old['error'][i]),candidate=stats(new['error'][j]),
            original_valid_depth_pairs=len(old['error']),candidate_valid_depth_pairs=len(new['error']),
            common_source_pairs=len(i),original_geometry_camera=before,candidate_geometry_camera=after)
        for state,a in [('original',old),('candidate',new)]:
            for k,v in a.items():records[programme+'_'+state+'_'+k]=v
        print(programme,result[programme]['original'],result[programme]['candidate'],flush=True)
    write(OUT/candidate/'source_motion_audit.json',dict(programmes=result,
        state_sha256=sha(OUT/candidate/'state.npz'),
        method='unchanged actual left/right source pixels; native inverse left pixel onto each current board/ice, then native projection at right exposure',
        role='same source flow used during development; conditional depths, not independent 3D or statistical accuracy evidence'))
    save(OUT/candidate/'source_motion_audit.npz',**records)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True)
    ap.add_argument('--source-visibility',action='store_true');args=ap.parse_args()
    run(args.candidate,args.source_visibility)
