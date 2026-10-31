"""A small independent QP oracle for the production boundary dual kernel."""
from pathlib import Path
import itertools,os,shutil,subprocess
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def dual_program(tmp_path_factory):
    compiler=shutil.which('g++') or shutil.which('cl')
    if compiler is None:pytest.skip('existing C++ compiler required')
    directory=tmp_path_factory.mktemp('projection_precision')
    cpp=directory/'dual.cpp'
    cpp.write_text('#include "rinkspace/coupled_dual.hpp"\n#include <vector>\n#include <iostream>\n#include <iomanip>\nusing namespace rinkspace;\n'+r'''
int main(){int n;std::cin>>n;std::vector<double> g(n*n),b(n),d(n),e(n);
 for(auto& x:g)std::cin>>x;for(auto& x:b)std::cin>>x;for(auto& x:e)std::cin>>x;
 double kkt;int sweeps=coupled_dual_minimize(n,n,g.data(),b.data(),d.data(),e.data(),&kkt);
 if(kkt!=coupled_dual_kkt(n,n,g.data(),b.data(),d.data(),e.data()))return 2;
 const int stride=n+3;std::vector<double> padded(n*stride,std::numeric_limits<double>::quiet_NaN()),other(n);
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)padded[i*stride+j]=g[i*n+j];
 double other_kkt;int other_sweeps=coupled_dual_minimize(n,stride,padded.data(),b.data(),other.data(),e.data(),&other_kkt);
 if(sweeps!=other_sweeps||kkt!=other_kkt||d!=other)return 3;
 double scale=1.;for(double value:b)scale=fmax(scale,fabs(value));
 if(coupled_dual_unresolved(n,b.data(),kkt)!=(kkt>8*std::numeric_limits<double>::epsilon()*scale))return 4;
 std::cout<<sweeps<<' '<<std::setprecision(17)<<kkt;
 for(auto x:d)std::cout<<' '<<x;return 0;}
''',encoding='utf-8')
    executable=directory/('dual.exe' if os.name=='nt' else 'dual')
    command=([compiler,'/nologo','/std:c++17','/EHsc','/O1','/I'+str(ROOT/'native/include'),str(cpp),'/Fe:'+str(executable)]
             if Path(compiler).stem.lower()=='cl' else [compiler,'-std=c++17','-O1','-I'+str(ROOT/'native/include'),str(cpp),'-o',str(executable)])
    subprocess.run(command,cwd=directory,capture_output=True,check=True)
    return executable

def oracle(a,b,equality):
    """Enumerate active faces in the primal Euclidean projection, no Hildreth."""
    fixed=list(np.flatnonzero(equality));free=list(np.flatnonzero(~equality));best=None
    for mask in itertools.product([False,True],repeat=len(free)):
        ids=fixed+[k for k,on in zip(free,mask) if on]
        x=np.linalg.lstsq(a[ids],b[ids],rcond=None)[0] if ids else np.zeros(a.shape[1])
        r=a@x-b
        if np.any(r[~equality]<-1e-9) or np.any(abs(r[equality])>1e-9):continue
        if best is None or x@x<best@best:best=x
    assert best is not None
    return best

@pytest.mark.parametrize('angle,multiplier,extra,equality',[
    (.07,400.,False,[False,False]),
    (.007,400.,False,[False,False]),
    (.07,400.,False,[True,False]),
    (.07,400.,False,[True,True]),
    (.07,400.,True,[False,False,False,False]),
    (.007,400.,True,[False,False,False,False]),
])
def test_near_parallel_large_duals_do_not_loosen_primal_accuracy(dual_program,angle,multiplier,extra,equality):
    a=np.array([[1.,0.],[np.cos(angle),np.sin(angle)]])
    expected=a.T@np.full(2,multiplier);b=a@expected
    if extra:
        # Opposite box faces are discovered and then released, as in the real
        # seven-cut joint fit. They cannot both remain active.
        a=np.vstack([a,-a]);b=np.r_[b,-b-1.]
    eq=np.asarray(equality,bool);gram=a@a.T
    data=' '.join(map(str,[len(a),*gram.ravel(),*b,*eq.astype(int)]))
    result=subprocess.run([str(dual_program)],input=data,text=True,capture_output=True,check=True,timeout=10)
    values=np.fromstring(result.stdout,sep=' ');dual=values[2:];point=a.T@dual
    np.testing.assert_allclose(point,oracle(a,b,eq),rtol=0.,atol=2e-9)
    assert np.all(dual[~eq]>=0)
    assert values[1]<2e-12,result.stdout
    assert np.min(a@point-b)>-2e-12


def test_exhausted_dual_reports_the_last_computed_kkt(dual_program):
    # Inconsistent equality faces exhaust the unchanged 4096-sweep attempt.
    data='2 1 1 1 1 1 2 1 1'
    result=subprocess.run([str(dual_program)],input=data,text=True,capture_output=True,check=True,timeout=10)
    values=np.fromstring(result.stdout,sep=' ')
    assert values[0]==4096
    assert values[1]==1.
