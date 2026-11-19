"""Compare selected physical families on every retained source observation."""
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write
from review import camera_state, native_for, ji, localize
from solve import stats


def run(candidates, families, output):
    setup=read(PREPARED/'setup.json');chunks=defaultdict(list);results={}
    for candidate in candidates:
        base=camera_state(candidate);natives={};errors=[]
        for name,s in setup['shots'].items():
            a=arrays(prepared_packet_path(name));selected=np.flatnonzero((a['fold']<5)&np.isin(a['family'],families))
            if not len(selected):continue
            if s['program'] not in natives:natives[s['program']]=native_for(candidate,s['program'])
            n=natives[s['program']];table=ji.native_paths(n,np.unique(a['path'][selected]),count=513)
            grouped=np.rec.fromarrays((a['idx'][selected],a['path'][selected]))
            unique,inverse=np.unique(grouped,return_inverse=True);order=np.argsort(inverse,kind='stable')
            pieces=np.split(order,np.flatnonzero(np.diff(inverse[order]))+1)
            jobs=[(int(i),str(k),float(base[name+'_times'][i])) for i,k in unique]
            e=np.full(len(selected),np.inf)
            for first in range(0,len(jobs),300):
                projected=ji.project_groups(n,base[name+'_rigs'],base[name+'_controls'],s['first'],s['dt'],
                    jobs[first:first+300],{k:v[0] for k,v in table.items()})
                for j,((uv,status),(_,key,_)) in enumerate(zip(projected,jobs[first:first+300]),first):
                    loc=pieces[j];rows=selected[loc];uv=uv.copy();uv[status!=0]=np.nan
                    if n.paths[key]['kind']==0:
                        e[loc]=np.linalg.norm(a['uv'][rows]-uv[0],axis=1)
                    else:
                        if a.get('station_fixed',np.zeros(len(a['pts']),bool))[rows].any():
                            raise ValueError('Use full exact-station metrics for fixed curve material rows: '+key)
                        _,e[loc]=localize(uv,table[key][1],a['uv'][rows])
            errors.append(e)
            if candidate==candidates[0]:
                for key in ['path','family','pts','uv']:chunks[key].append(a[key][selected])
                chunks['row'].append(selected);chunks['shot'].append(np.repeat(name,len(selected)))
        results[candidate]=np.concatenate(errors)
    source={k:np.concatenate(v) for k,v in chunks.items()};report={}
    for candidate,e in results.items():
        report[candidate]={p:{f:stats(e[np.char.startswith(source['shot'],p+'_')&(source['family']==f)])
            for f in families} for p in ['jin','okada','chen','nayeon']}
    folder=OUT/output;folder.mkdir(parents=True,exist_ok=False)
    save(folder/'errors.npz',**source,**{'candidate_'+str(i):results[c] for i,c in enumerate(candidates)})
    write(folder/'report.json',dict(candidates=candidates,families=families,metrics=report,
        prepared=str(PREPARED),states={c:sha(OUT/c/'state.npz') for c in candidates},
        evidence='all active source rows, nearest finite projected material curve; source UV unchanged; reused development evidence'))
    for c,programmes in report.items():
        print(c,{p:{f:round(v['p90'],3) for f,v in items.items() if 'p90' in v} for p,items in programmes.items()},flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--candidates',nargs='+',required=True)
    p.add_argument('--families',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.candidates,a.families,a.out)
