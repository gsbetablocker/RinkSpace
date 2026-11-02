#include "rinkspace/coupled_dual.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace rinkspace;
static int checks=0;
static void check(bool ok,const char* name){++checks;if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}}
int main(){
    for(int cached=0;cached<2;++cached){
        auto solve=[&](int n,int stride,const double* g,const double* t,double* d,const double* e,double& k){
            CoupledDualCache c;return cached?coupled_dual_minimize_cached(n,stride,g,t,d,e,c,&k):coupled_dual_minimize(n,stride,g,t,d,e,&k);};
        { double g[]={1.,.5,.5,1.},t[]={1e308,-1e308},d[]={0.,0.},e[]={1.,1.},k=-1.;
          solve(2,2,g,t,d,e,k);check(coupled_dual_unresolved(2,t,k),"finite arithmetic overflow is unresolved");
          check(std::isinf(k)&&k>0.,"overflow never produces a zero KKT certificate"); }
        for(double invalid:{NAN,INFINITY,-INFINITY}){
          double g[]={1.},t[]={invalid},d[]={0.},e[]={0.},k=-1.;solve(1,1,g,t,d,e,k);
          check(coupled_dual_unresolved(1,t,k),"nonfinite target rejected");
          t[0]=1.;g[0]=invalid;k=-1.;solve(1,1,g,t,d,e,k);
          check(coupled_dual_unresolved(1,t,k),"nonfinite Gram rejected");
          g[0]=1.;d[0]=invalid;k=-1.;solve(1,1,g,t,d,e,k);
          check(coupled_dual_unresolved(1,t,k),"nonfinite initial multiplier rejected");
        }
        { double g[]={1.,0.,NAN,0.,1.,NAN},t[]={2.,-3.},d[]={0.,0.},e[]={0.,0.},k=-1.;
          solve(2,3,g,t,d,e,k);check(d[0]==2.&&d[1]==0.&&k==0.,"valid inequalities and unused stride padding");
          d[0]=d[1]=0.;e[1]=1.;solve(2,3,g,t,d,e,k);
          check(d[0]==2.&&d[1]==-3.&&k==0.,"mixed equality and inequality"); }
        { double g[]={1.},t[]={-1.},d[]={0.},e[]={0.},k=-1.;
          solve(1,1,g,t,d,e,k);check(d[0]==0.&&k==0.,"inactive nonnegative multiplier"); }
    }
    {double g[]={1.},t[]={NAN},d[]={0.},e[]={0.};
     check(std::isinf(coupled_dual_kkt(1,1,g,t,d,e)),"direct KKT does not swallow NaN");
     CoupledDualCache c;c.reset(1,1,g,t,d,e);check(std::isinf(c.kkt()),"cached KKT does not swallow NaN");}
    {double t[]={1.};for(double k:{double(NAN),double(INFINITY),-double(INFINITY),-1.})check(coupled_dual_unresolved(1,t,k),"invalid certificate remains unresolved");}
    std::printf("PASS %d native assertions; compiled production coupled_dual.hpp\n",checks);
}
