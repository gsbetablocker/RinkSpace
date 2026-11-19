"""Every eligible source track: displacement and curvature on native exposures.

Depths are conditional inverse projections, not measured landmarks. Source UV,
frame identity, and quality/foreground exclusions are preserved; no time thinning.
"""
from argparse import ArgumentParser
import numpy as np
from shared import OUT, PREPARED, arrays, read, save, sha, write
from review import camera_state, native_for
from motion_tracks import packet, off_graphics
from flow_depths import bind


def frame_median(index,value,select,count,minimum):
    result=np.full(count,np.nan)
    ix=np.flatnonzero(select&np.isfinite(value))
    ix=ix[np.argsort(index[ix],kind='stable')]
    for group in np.split(ix,np.flatnonzero(np.diff(index[ix]))+1):
        if len(group)>=minimum:result[index[group[0]]]=np.median(value[group])
    return result


def stat(value):
    value=value[np.isfinite(value)]
    return dict(frames=len(value),**dict(zip(('p50','p90','p95','p99','max'),
        map(float,np.quantile(value,[.5,.9,.95,.99,1.]))))) if len(value) else dict(frames=0)


def project(native,s,base,name,index,world):
    value=np.full((len(index),2),np.nan)
    rigs,inv=np.unique(base[name+'_rigs'][index],axis=0,return_inverse=True)
    for slot,rig in enumerate(rigs):
        ids=np.flatnonzero(inv==slot)
        for start in range(0,len(ids),20000):
            ix=ids[start:start+20000]
            pred=native.core.project_motion(rig,base[name+'_controls'],s['first'],s['dt'],1080,
                base[name+'_times'][index[ix]],world[ix])
            good=pred['status']==0
            value[ix[good]]=pred['uv'][good]
    return value


def run(candidate,programmes,shots=None):
    setup=read(PREPARED/'setup.json');base=camera_state(candidate)
    folder=OUT/candidate/'repair_motion';folder.mkdir(parents=True,exist_ok=True)
    natives={};report=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
    for name,s in setup['shots'].items():
        p=s['program']
        if p not in programmes:continue
        if shots and name not in shots:continue
        if p not in natives:natives[p]=native_for(candidate,p)
        native=natives[p];path=packet(name,s);a=arrays(path)
        pts=base[name+'_pts'];np.testing.assert_array_equal(pts,a['pts'])
        keep=(a.get('role',a.get('fold'))<4)&(a['quality']<16.)
        keep&=np.isfinite(a['uv']).all(1)&np.isfinite(a['left_uv']).all(1)
        if not bool(a.get('graphics_mask_per_exposure',False)):
            keep&=off_graphics(a['uv'])&off_graphics(a['left_uv'])&off_graphics(a['anchor_uv'])
        if p=='okada':
            visibility=arrays(OUT/'source_flow_visibility'/(name+'.npz'))
            assert visibility['source_packet_sha256'].item()==sha(path)
            keep&=visibility['keep']
        rows=np.flatnonzero(keep);index=a['index'][rows];left=index-a['lag'][rows]
        world,roundtrip=bind(native,s,base[name+'_rigs'],base[name+'_controls'],base[name+'_times'],
            left,a['left_uv'][rows],a['family'][rows])
        valid=np.isfinite(world).all(1)
        rows=rows[valid];index=index[valid];left=left[valid];world=world[valid]
        prediction=project(native,s,base,name,index,world)
        previous=project(native,s,base,name,left,world)
        displacement=(prediction-previous)-(a['uv'][rows]-a['left_uv'][rows])
        triple=(a['lag'][rows]==1)&(index>=2)&np.isfinite(a['prev2_uv'][rows]).all(1)
        curvature=np.full_like(displacement,np.nan)
        earlier=project(native,s,base,name,index[triple]-2,world[triple])
        curvature[triple]=(prediction[triple]-2*previous[triple]+earlier)-(
            a['uv'][rows[triple]]-2*a['left_uv'][rows[triple]]+a['prev2_uv'][rows[triple]])
        frames={};summary={}
        for material,family,minimum in [('board',0,4),('ice',1,3)]:
            for term,lag,err in [('adjacent',1,displacement),('lag8',8,displacement),('jitter',1,curvature)]:
                key=material+'_'+term
                select=(a['family'][rows]==family)&(a['lag'][rows]==lag)
                value=frame_median(index,np.linalg.norm(err,axis=1),select,len(pts),minimum)
                frames[key]=value;summary[key]=stat(value)
                finite=np.flatnonzero(np.isfinite(value))
                worst=finite[np.argsort(value[finite])[-12:][::-1]]
                summary[key]['worst']=[dict(pts=int(pts[i]),seconds=float(pts[i]/12800),px=float(value[i])) for i in worst]
        report[name]=dict(metrics=summary,source_rows=len(a['index']),eligible_rows=int(keep.sum()),
            valid_depth_rows=len(rows),source_path=str(path),source_sha256=sha(path),
            depth_invalid_rows=int((~valid).sum()))
        save(folder/(name+'.npz'),pts=pts,source_row=rows,index=index,lag=a['lag'][rows],
            family=a['family'][rows],displacement=displacement,curvature=curvature,**frames)
        write(folder/'summary.json',report)
        print(name,{k:round(v['p95'],4) for k,v in summary.items() if 'p95' in v},flush=True)
    write(folder/'binding.json',dict(candidate=candidate,prepared_directory=str(PREPARED),
        state_sha256=sha(OUT/candidate/'state.npz') if candidate!='original' else None,
        method='unchanged source track UV; each candidate native inverse left pixel gives conditional board/ice depth; exact native forward at next and previous exposures; per-frame median',
        limitations='development source flow; conditional depth and optical-flow uncertainty; not independent accuracy certification'))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--programmes',nargs='+',default=['jin','okada','chen','nayeon'])
    parser.add_argument('--shots',nargs='+')
    args=parser.parse_args();run(args.candidate,args.programmes,args.shots)
