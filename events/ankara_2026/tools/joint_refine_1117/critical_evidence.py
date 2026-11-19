"""Check dense actual source readings at the critical Nayeon advertisements/blue line."""
from argparse import ArgumentParser
import numpy as np
from shared import prepared_packet_path
from shared import OUT, PREPARED, arrays, read, save, sha, write
from review import camera_state, ji, localize, native_for
from solve import stats


def measure(candidate,name,family):
    setup=read(PREPARED/'setup.json');s=setup['shots'][name]
    a=arrays(prepared_packet_path(name));base=camera_state(candidate)
    eligible=a.get('eligible',np.ones(len(a['pts']),bool))
    use=eligible&(a['fold']<5)&(a['family']==family)
    a={k:a[k][use] for k in ('pts','idx','uv','path')}
    native=native_for(candidate,s['program']);table=ji.native_paths(native,np.unique(a['path']),count=513)
    groups,inv=np.unique(np.rec.fromarrays((a['idx'],a['path'])),return_inverse=True)
    order=np.argsort(inv,kind='stable');pieces=np.split(order,np.flatnonzero(np.diff(inv[order]))+1)
    error=np.full(len(a['pts']),np.inf)
    jobs=[(int(i),str(k),float(base[name+'_times'][i])) for i,k in groups]
    for lo in range(0,len(jobs),300):
        slab=jobs[lo:lo+300]
        projections=ji.project_groups(native,base[name+'_rigs'],base[name+'_controls'],s['first'],s['dt'],slab,{k:v[0] for k,v in table.items()})
        for j,((uv,status),(_,key,_)) in enumerate(zip(projections,slab),lo):
            uv=uv.copy();uv[status!=0]=np.nan;ix=pieces[j]
            _,error[ix]=localize(uv,table[key][1],a['uv'][ix])
    return a,error


def run(candidate):
    report={};all_arrays={}
    for name,family in [('nayeon_s06','Ad16'),('nayeon_s04','Ad3'),('nayeon_s08','B2')]:
        a,e0=measure('original',name,family);b,e1=measure(candidate,name,family)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        key=name+'_'+family
        all_arrays.update({key+'_'+k:v for k,v in a.items()});all_arrays[key+'_original_error']=e0;all_arrays[key+'_candidate_error']=e1
        report[key]=dict(original=stats(e0),candidate=stats(e1),source_frames=len(np.unique(a['pts'])),
            source_seconds=[float(a['pts'].min()/12800.),float(a['pts'].max()/12800.)],
            paths={str(p):dict(original=stats(e0[a['path']==p]),candidate=stats(e1[a['path']==p])) for p in np.unique(a['path'])})
    write(OUT/candidate/'critical_source_evidence.json',dict(cases=report,
        state_sha256=sha(OUT/candidate/'state.npz'),source_readings='all inherited eligible raw source rows of these families, before 0.2-second thinning',
        evidence_role='additional source temporal coverage of existing features; correlated development evidence, not independent geometry truth'))
    save(OUT/candidate/'critical_source_evidence.npz',**all_arrays)
    for k,v in report.items():print(k,v['original'],v['candidate'])


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True)
    run(ap.parse_args().candidate)
