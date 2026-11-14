"""Compare tighter motion and shared print geometry after independent source repair."""
from argparse import ArgumentParser
import solve_joint as s
import numpy as np
from scipy.sparse import block_diag, csr_matrix, kron
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos

ROOT=s.ROOT
s.INPUT=ROOT/'prepared_guarded'
s.MOTION_UNIT=np.array([.0012,.0012,.0027])
FOCUS={'jin_s03','jin_s04','jin_s06','jin_s10','jin_s12','jin_s13','jin_s14','jin_s15','jin_s16',
       'jin_s18','jin_s19','jin_s20','jin_s21','jin_s22','jin_s24','okada_s04','okada_s05'}
s.AD_NAMES=(9,13,14,17,18,19,26)
s.ICE_KEYS={}


def local_motion(b,series,shots):
    blocks=[];controls=[];spans={};cursor=0
    for name,meta in shots.items():
        c=b[name+'_controls'];t=b[name+'_times']
        n=len(c);first=meta['first'];dt=meta['dt']
        if name not in FOCUS:
            blocks.append(csr_matrix((3*n,0)))
        else:
            stride=31 if name in ('jin_s14','jin_s22') else 38 if meta['program']=='jin' else 51
            chart=UniformMotionChart.create(n,first,dt,stride,t[0])
            train=np.unique(series[name]['idx'][series[name]['fold']==0])
            inds,ws=cubic_samples(n,first,dt,t[train])
            response=np.einsum('ni,nik->nk',ws,chart.refinement[inds])
            active=np.flatnonzero(np.count_nonzero(abs(response)>.14,axis=0)>=4)
            matrix=chart.refinement[:,active]
            grid=(t[[0,-1]]-first)/dt
            if np.max(abs(grid-np.rint(grid)))<1.e-8:
                matrix=continue_exposure_halos(np.repeat(matrix[:,None],3,axis=1),first,dt,t[[0,-1]])[:,0,:]
            blocks.append(kron(csr_matrix(matrix),csr_matrix(np.diag(s.MOTION_UNIT)),format='csr'))
        spans[name]=(cursor,cursor+n);cursor+=n
        controls.append(c)
    return block_diag(blocks,format='csr'),np.vstack(controls),spans


def guarded_optics(shots):
    groups=s.optics_spec_original(shots)
    for name in groups:
        # Test the previously reported wide optical centre on S14/S22 only.
        # One coefficient per branch across the entire shot, including guards.
        groups[name]=({26:(2.,4.0),27:(2.,4.0),29:(3.,5.),30:(3.,5.)}
                      if name in ('jin_s14','jin_s22') else {})
    return groups


def report_progress(log):
    return log


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--out',default='guarded')
    p.add_argument('--variants',nargs='+',default=['motion','joint'],
                   choices=('motion','camera','joint'))
    v=p.parse_args()
    target=ROOT/v.out;target.mkdir(exist_ok=True)
    log=s.Journal(target/'solve.jsonl')
    log('start',variants=v.variants,device='native CUDA and cuDSS',prepared=str(s.INPUT))
    s.build_motion=local_motion
    s.optics_spec_original=s.optics_spec
    s.optics_spec=guarded_optics
    try:
        setup,core,b,series=s.load()
        for name in v.variants:
            s.solve_variant(name,setup,core,b,series,target,log,target/'stop')
            if (target/'stop').exists():break
    except BaseException as e:
        log('failure',reason=str(e))
        raise
if __name__=='__main__':main()
