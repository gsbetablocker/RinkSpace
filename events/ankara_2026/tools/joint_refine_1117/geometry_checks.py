"""Compare each common-geometry adjustment with identical cameras and source pixels."""
from argparse import ArgumentParser
from collections import defaultdict
import time
import numpy as np
from lane_end_evidence import BASE, ENDS, WORK
from lane_review import pixel_errors
from review import camera_state, native_for
from shared import PREPARED, arrays, prepared_packet_path, read, save, sha, write
from solve import stats
from source_bundle import relative


def changed_paths(left,right):
    paths=[]
    for k,p in left.paths.items():
        if p['layer']!='ice':continue
        q=right.paths[k]
        if any(not np.allclose(p[f],q[f],rtol=0,atol=1e-13) for f in ('parameters','vertices')):
            paths.append(k)
    return paths


def run(branch):
    dest=WORK/'geometry_checks'/branch;candidate=WORK/('bundle_'+branch)
    while not (candidate/'report.json').exists():time.sleep(3)
    parent=relative(WORK/'camera_assembly');b=camera_state(parent)
    n0=native_for(parent,'jin');n1=native_for(relative(candidate),'jin')
    old=native_for(BASE,'jin') if branch=='ends' else None
    paths=changed_paths(n0,n1)
    # Also include unchanged finite paths in an opened family for accountability.
    if branch=='ends':paths=sorted(set(paths)|set(ENDS))
    if not paths:raise ValueError(('empty block',branch))
    rows=defaultdict(lambda:defaultdict(list));by_path=defaultdict(lambda:defaultdict(list))
    setup=read(PREPARED/'setup.json')['shots'];candidates=['parent','adjusted']+(['inherited_geometry'] if old else [])
    for name,s in setup.items():
        a=arrays(prepared_packet_path(name));use=(a['fold']<7)&np.isin(a['path'],paths)
        if not use.any():continue
        file=dest/(name+'.npz')
        if file.exists():res=arrays(file)
        else:
            ids=np.flatnonzero(use);res={'row':ids}
            for label,native in zip(candidates,[n0,n1,old]):
                e,_=pixel_errors(native,b,s,name,a,use);res[label]=e[ids]
            save(file,**res)
        ids=res['row'];family=a['family'][ids];material=a['material'][ids];fold=a['fold'][ids];p=s['program']
        extra=np.isin(material,['paired_lane_end_source','real_paired_annulus_edges'])
        parts={'inherited':(fold<5)&~extra,'new':(fold<5)&extra,'held':(fold==6)&extra,'all':fold<5}
        for label in candidates:
            for k,mask in parts.items():
                if mask.any():rows[p+'/'+k][label].append(res[label][mask])
            for k in np.unique(family):
                for role,mask in parts.items():
                    take=mask&(family==k)
                    if take.any():by_path[k+'/'+role][label].append(res[label][take])
    def digest(values):
        return {k:{lab:stats(np.concatenate(v)) for lab,v in part.items()} for k,part in values.items()}
    write(dest/'comparison.json',dict(candidate=relative(candidate),parent=parent,
        cameras='identical jointly-refined shot cameras for each comparison',paths=paths,
        prepared=str(PREPARED),source_setup_sha256=sha(PREPARED/'setup.json'),
        programmes=digest(rows),families=digest(by_path)))
    print(branch,'complete',len(paths),flush=True)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('branches',nargs='+');a=ap.parse_args()
    for branch in a.branches:run(branch)
