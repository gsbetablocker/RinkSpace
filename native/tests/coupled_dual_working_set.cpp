#include "rinkspace/coupled_dual_working_set.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <fstream>
using namespace rinkspace;
void check(bool ok){if(!ok)std::abort();}
int main(int argc,char** argv){
    size_t cases=0,fast=0,fallback=0;
    auto compare=[&](int n,const std::vector<double>& g,const std::vector<double>& t,const std::vector<double>& seed,const std::vector<double>& e){
        auto candidate=seed,reference=seed;double kkt=-1.,ref_kkt=-1.;DualWorkingStats stats;
        auto result=coupled_dual_working_set(n,n,g.data(),t.data(),candidate.data(),e.data(),kkt,stats,[]{return false;});
        if(result==DualWorkingResult::solved){++fast;check(!coupled_dual_unresolved(n,t.data(),kkt));
            check(kkt==coupled_dual_kkt(n,n,g.data(),t.data(),candidate.data(),e.data()));
            for(int i=0;i<n;++i)check(finite_scalar(candidate[i])&&(e[i]||candidate[i]>=0.));
        }else{++fallback;check(result==DualWorkingResult::fallback);check(std::memcmp(seed.data(),candidate.data(),n*sizeof(double))==0);}
        // A failed fast solve must keep its public seed transactional while
        // optionally exporting a strictly better feasible original-KKT seed.
        auto warm_candidate=seed;std::vector<double> warm;double warm_kkt=-1.;DualWorkingStats warm_stats;
        auto warm_result=coupled_dual_working_set(n,n,g.data(),t.data(),warm_candidate.data(),e.data(),warm_kkt,warm_stats,[]{return false;},&warm);
        check(warm_result==result);
        if(warm_result==DualWorkingResult::fallback){
            check(warm_candidate==seed);
            if(!warm.empty()){
                for(int i=0;i<n;++i)check(finite_scalar(warm[i])&&(e[i]||warm[i]>=0.));
                check(coupled_dual_kkt(n,n,g.data(),t.data(),warm.data(),e.data())<coupled_dual_kkt(n,n,g.data(),t.data(),seed.data(),e.data()));
                CoupledDualCache refinement;double certificate;
                coupled_dual_minimize_cached(n,n,g.data(),t.data(),warm.data(),e.data(),refinement,&certificate);
                check(certificate==coupled_dual_kkt(n,n,g.data(),t.data(),warm.data(),e.data()));
            }
        }
        CoupledDualCache cache;coupled_dual_minimize_cached(n,n,g.data(),t.data(),reference.data(),e.data(),cache,&ref_kkt);
        if(result==DualWorkingResult::solved&&!coupled_dual_unresolved(n,t.data(),ref_kkt)){
            // Same convex QP: duals need not be unique. Compare the objective
            // and Gram action (the induced primal correction in this metric).
            double av=0.,bv=0.;for(int i=0;i<n;++i){double a=0.,b=0.;for(int j=0;j<n;++j){a=fma(g[size_t(i)*n+j],candidate[j],a);b=fma(g[size_t(i)*n+j],reference[j],b);}
                check(fabs(a-b)<=2e-10*(1+fabs(t[i])));av+=candidate[i]*(.5*a-t[i]);bv+=reference[i]*(.5*b-t[i]);}
            check(fabs(av-bv)<=2e-10*(1+fabs(bv)));
        }
        ++cases;
    };
    if(argc==2){std::ifstream input(argv[1],std::ios::binary);check(bool(input));int32_t n;
        while(input.read(reinterpret_cast<char*>(&n),sizeof(n))){check(n>0&&n<10000);std::vector<double> g(size_t(n)*n),t(n),seed(n),e(n);
            for(auto* v:{&g,&t,&seed,&e}){input.read(reinterpret_cast<char*>(v->data()),v->size()*sizeof(double));check(bool(input));}compare(n,g,t,seed,e);}
        check(input.eof()&&cases>0);
    }else{
        compare(2,{1.,.5,.5,1.},{1.,0.},{0.,0.},{0.,0.});
        compare(2,{1.,1.,1.,1.},{1.,1.},{0.,0.},{0.,0.});
        compare(2,{1.,-1.,-1.,1.},{1.,1.},{0.,0.},{0.,0.});
        compare(2,{1.,.99999999999999,.99999999999999,1.},{1.,-.7},{0.,0.},{1.,1.});
        // Nearly coincident inequalities: the entering cut replaces a weaker
        // active cut. A rank-truncated absolute solve used to add/remove the
        // entering cut at zero step and fall back despite this bounded QP.
        for(double epsilon:{0.,1e-8,1e-7}){
            const double c=std::sqrt(1.-epsilon*epsilon);
            const std::vector<double> g{1.,c,0.,c,1.,epsilon,0.,epsilon,1.};
            const std::vector<double> t{2*c-1e-5,2+epsilon,1+2*epsilon},seed{1.,0.,1.},e(3,0.);
            auto value=seed;double kkt=-1.;DualWorkingStats stats;int successful_checks=0;
            check(coupled_dual_working_set(3,3,g.data(),t.data(),value.data(),e.data(),kkt,stats,[&]{++successful_checks;return false;})==DualWorkingResult::solved);
            check(!coupled_dual_unresolved(3,t.data(),kkt));
            check(value[0]==0.&&fabs(value[1]-2.)<1e-12&&fabs(value[2]-1.)<1e-12);
            compare(3,g,t,seed,e);
            // Cancellation during the extra rank-handling work remains
            // transactional: neither the seed nor certificate is published.
            for(int stop_at=1;stop_at<=successful_checks;++stop_at){
                value=seed;kkt=-1.;int checks=0;DualWorkingStats cancelled;
                check(coupled_dual_working_set(3,3,g.data(),t.data(),value.data(),e.data(),kkt,cancelled,[&]{return ++checks==stop_at;})==DualWorkingResult::stopped);
                check(value==seed&&kkt==-1.);
            }
        }
        std::mt19937_64 random(77129);std::uniform_real_distribution<double> u(-1.,1.);
        for(int n:{2,7,32,83})for(int sample=0;sample<8;++sample){
            std::vector<double> a(size_t(n)*n),g(size_t(n)*n),scale(n),t(n),seed(n),e(n),solution(n);
            for(double& v:a)v=u(random);
            for(int i=0;i<n;++i)for(int j=0;j<n;++j){double x=i==j?.5:0.;for(int k=0;k<n;++k)x=fma(a[size_t(k)*n+i],a[size_t(k)*n+j],x);g[size_t(i)*n+j]=x;}
            for(int i=0;i<n;++i)scale[i]=sqrt(g[size_t(i)*n+i]);
            for(int i=0;i<n;++i)for(int j=0;j<n;++j)g[size_t(i)*n+j]/=scale[i]*scale[j];
            for(int i=0;i<n;++i){e[i]=i%7==0;solution[i]=e[i]?u(random):(i%3?fabs(u(random)):0.);seed[i]=e[i]?u(random):fabs(u(random));}
            for(int i=0;i<n;++i){t[i]=e[i]||solution[i]>0.?0.:-.2;for(int j=0;j<n;++j)t[i]=fma(g[size_t(i)*n+j],solution[j],t[i]);}
            compare(n,g,t,seed,e);
        }
        double g[]{1.},t[]{1.},seed[]{.25},e[]{0.},kkt=-1.;DualWorkingStats stats;
        check(coupled_dual_working_set(1,1,g,t,seed,e,kkt,stats,[]{return true;})==DualWorkingResult::stopped);
        check(seed[0]==.25&&kkt==-1.);
        int checks=0;
        check(coupled_dual_working_set(1,1,g,t,seed,e,kkt,stats,[&]{return ++checks==2;})==DualWorkingResult::stopped);
        check(seed[0]==.25&&kkt==-1.);
        for(double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
            double invalid[]{bad};
            check(coupled_dual_working_set(1,1,g,invalid,seed,e,kkt,stats,[]{return false;})==DualWorkingResult::fallback);
            check(seed[0]==.25&&kkt==-1.);
        }
        check(fast>0);
        // The forward-error routing does not turn an unresolved KKT into a
        // success certificate, and does not cover a materially violated cut.
        {const int n=64;std::vector<double> gram(size_t(n)*n,0.),target(n,1.),dual(n,1.),eq(n,0.);
         for(int i=0;i<n;++i)gram[size_t(i)*n+i]=1.;
         target[0]+=32*std::numeric_limits<double>::epsilon();
         const double failed=coupled_dual_kkt(n,n,gram.data(),target.data(),dual.data(),eq.data());
         check(coupled_dual_unresolved(n,target.data(),failed));
         check(coupled_dual_roundoff_limited(n,n,gram.data(),target.data(),dual.data(),eq.data()));
         target[0]+=1e-8;check(!coupled_dual_roundoff_limited(n,n,gram.data(),target.data(),dual.data(),eq.data()));}
        {double gram[]{1.,-1.,-1.,1.},target[]{1.,1.},dual[]{0.,0.},eq[]{0.,0.};
         check(!coupled_dual_roundoff_limited(2,2,gram,target,dual,eq));}
    }
    std::printf("PASS working-set cases=%zu solved=%zu original-seed fallbacks=%zu; original KKT/objective/Gram action and stop protected\n",cases,fast,fallback);
}
