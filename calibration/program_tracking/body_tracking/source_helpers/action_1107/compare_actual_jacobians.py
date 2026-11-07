import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

def cache(path):
    q=path/'linear_cache'
    registry=json.loads((path/'weighted_registry.json').read_text())
    names=('data','indices','indptr','residual','x')
    arrays={n:np.load(q/(n+'.npy'),mmap_mode='r') for n in names}
    return registry,arrays

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('A',type=Path);ap.add_argument('B',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args()
    a,X=cache(args.A);b,Y=cache(args.B)
    assert (a['rows'],a['columns'])==(b['rows'],b['columns'])
    np.testing.assert_array_equal(X['x'],Y['x'])
    bounds=lambda r:[(v['name'],v['start'],v['stop']) for v in r['blocks']]
    assert bounds(a)==bounds(b)
    n=a['columns'];m=a['rows']
    gen=np.random.default_rng(516839)
    body=np.arange(n-17-2244)
    local=body%69
    lower=np.isin(1+(local-6)//3,[1,2,4,5,7,8,10,11])&(local>=6)
    domains=dict(root=17+body[local<6],legs=17+body[lower],
        shoes=np.arange(n-2244,n),mixed=np.arange(17,n))
    D=np.zeros((n,len(domains)))
    for i,indices in enumerate(domains.values()):
        D[indices,i]=gen.standard_normal(len(indices))
        D[:,i]/=np.linalg.norm(D[:,i])
    report=[]
    grad_error=np.zeros(n);hessian_error=np.zeros((n,len(domains)))
    max_rows=8192
    def block(z,first,last):
        ptr=z['indptr'][first:last+1]
        start,stop=int(ptr[0]),int(ptr[-1])
        return csr_matrix((z['data'][start:stop],z['indices'][start:stop],ptr-start),shape=(last-first,n))
    for name,lo,hi in bounds(a):
        summary=dict(name=name,rows=hi-lo,raw_residual_max=0.,raw_J_max=0.,
            max_Jv=np.zeros(len(domains)),max_Jv_reference=np.zeros(len(domains)),
            max_raw_J_reference=0.,changed_residual_rows=0)
        jt=np.zeros(n);jr=np.zeros(n)
        for first in range(lo,hi,max_rows):
            last=min(first+max_rows,hi)
            P,Q=(block(z,first,last) for z in (X,Y))
            e=np.asarray(Y['residual'][first:last])-X['residual'][first:last]
            E=Q-P
            u=gen.standard_normal(last-first)
            summary['changed_residual_rows']+=int(np.count_nonzero(e))
            summary['raw_residual_max']=max(summary['raw_residual_max'],float(np.max(abs(e),initial=0.)))
            summary['raw_J_max']=max(summary['raw_J_max'],float(np.max(abs(E.data),initial=0.)))
            summary['max_raw_J_reference']=max(summary['max_raw_J_reference'],float(np.max(abs(P.data),initial=0.)))
            summary['max_Jv']=np.maximum(summary['max_Jv'],np.max(abs(E@D),axis=0,initial=0.))
            V=P@D;W=Q@D
            summary['max_Jv_reference']=np.maximum(summary['max_Jv_reference'],np.max(abs(V),axis=0,initial=0.))
            jt+=E.T@u
            jr+=P.T@u
            grad_error+=Q.T@np.asarray(Y['residual'][first:last])-P.T@np.asarray(X['residual'][first:last])
            hessian_error+=Q.T@W-P.T@V
        for k in ('max_Jv','max_Jv_reference'):
            summary[k]=dict(zip(domains,summary[k].tolist()))
        summary['max_JTu']=float(np.max(abs(jt)))
        summary['max_JTu_reference']=float(np.max(abs(jr)))
        report.append(summary)
    outcome=dict(full_rows=m,full_columns=n,same_exact_initial_x=True,block_order_equal=True,
        max_Gradient_change=float(np.max(abs(grad_error))),
        max_Hessian_v_change=dict(zip(domains,np.max(abs(hessian_error),axis=0).tolist())),
        max_abs_residual_change=max(v['raw_residual_max'] for v in report),
        max_abs_J_change=max(v['raw_J_max'] for v in report),blocks=report)
    with args.output.open('x',encoding='utf8') as f: json.dump(outcome,f,ensure_ascii=False,indent=2)
    print({k:v for k,v in outcome.items() if k!='blocks'})

if __name__=='__main__': main()
