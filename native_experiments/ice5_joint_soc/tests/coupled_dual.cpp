#include "rinkspace/coupled_dual.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <random>
int main(int argc,char** argv){
    using namespace rinkspace;CoupledDualCache cache;uint64_t hits=0,evaluations=0,skipped=0;
    double reference_seconds=0.,cached_seconds=0.;
    auto compare=[&](int n,const std::vector<double>& gram,const std::vector<double>& target,
            const std::vector<double>& equality,const std::vector<double>& seed){
        auto ref=seed,fast=seed;double rk,fk;auto start=std::chrono::steady_clock::now();
        int rs=coupled_dual_minimize(n,n,gram.data(),target.data(),ref.data(),equality.data(),&rk);
        reference_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        start=std::chrono::steady_clock::now();
        int fs=coupled_dual_minimize_cached(n,n,gram.data(),target.data(),fast.data(),equality.data(),cache,&fk);
        cached_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        assert(rs==fs);assert(std::memcmp(ref.data(),fast.data(),n*sizeof(double))==0);assert(std::memcmp(&rk,&fk,sizeof(double))==0);
        hits+=cache.hits;evaluations+=cache.evaluations;skipped+=cache.skipped_sweeps;return fast;
    };
    if(argc==2){
        std::ifstream input(argv[1],std::ios::binary);assert(input);int cases=0;int32_t n;
        while(input.read(reinterpret_cast<char*>(&n),sizeof(n))){
            assert(n>0&&n<10000);std::vector<double> gram(size_t(n)*n),target(n),seed(n),equality(n);
            for(auto* v:{&gram,&target,&seed,&equality}){input.read(reinterpret_cast<char*>(v->data()),v->size()*sizeof(double));assert(input);}
            compare(n,gram,target,equality,seed);++cases;
        }
        assert(input.eof()&&cases>0);
        std::cout<<"real_cases="<<cases<<" bitwise_identical=true reference_s="<<reference_seconds<<" cached_s="<<cached_seconds
            <<" skipped_sweeps="<<skipped<<" evaluations="<<evaluations<<" cache_hits="<<hits<<"\n";return 0;
    }
    auto d=compare(2,{1.,.5,.5,1.},{1.,0.},{0.,0.},{0.,3.});assert(d[0]==1.&&d[1]==0.);
    d=compare(2,{1.,.25,.25,1.},{-1.,2.},{1.,0.},{0.,0.});assert(d[0]<0.&&d[1]>0.);
    // Singular, opposing and nearly parallel faces preserve the unresolved
    // dual and its 4096-sweep fallback, including every binary64 result.
    compare(2,{1.,1.,1.,1.},{1.,1.},{0.,0.},{.2,.3});
    compare(2,{1.,-1.,-1.,1.},{1.,1.},{0.,0.},{0.,0.});
    compare(2,{1.,1.-1e-13,1.-1e-13,1.},{1.,1.1},{0.,0.},{0.,0.});
    for(double cross:{.999,.999999,.99999999999,.99999999999999})
        compare(2,{1.,cross,cross,1.},{1.,-.7},{1.,1.},{0.,0.});
    std::mt19937_64 rng(42967);std::uniform_real_distribution<double> u(-1.,1.);
    for(int n:{2,7,32,83})for(int sample=0;sample<4;++sample){
        std::vector<double> a(size_t(n)*n),g(size_t(n)*n),scale(n),solution(n),slack(n),t(n),eq(n),seed(n);
        for(double& v:a)v=u(rng);
        for(int i=0;i<n;++i)for(int j=0;j<n;++j){double value=i==j?.5:0.;for(int k=0;k<n;++k)value+=a[size_t(k)*n+i]*a[size_t(k)*n+j];g[size_t(i)*n+j]=value;}
        for(int i=0;i<n;++i)scale[i]=std::sqrt(g[size_t(i)*n+i]);
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)g[size_t(i)*n+j]/=scale[i]*scale[j];
        for(int i=0;i<n;++i){eq[i]=(i%11==0);solution[i]=eq[i]?u(rng):(i%3?std::abs(u(rng)):0.);slack[i]=(eq[i]||solution[i]>0.)?0.:-.2-std::abs(u(rng));seed[i]=eq[i]?u(rng):std::abs(u(rng));}
        for(int i=0;i<n;++i){t[i]=slack[i];for(int j=0;j<n;++j)t[i]+=g[size_t(i)*n+j]*solution[j];}
        d=compare(n,g,t,eq,seed);
        // Independently constructed KKT solution of a strictly convex problem.
        for(int i=0;i<n;++i)assert(std::abs(d[i]-solution[i])<2e-10);
    }
    assert(hits>0);assert(skipped>0);
    std::cout<<"bitwise identical duals/KKT/sweeps; evaluations="<<evaluations<<" hits="<<hits
        <<" skipped_sweeps="<<skipped<<" reference_s="<<reference_seconds<<" cached_s="<<cached_seconds<<"\n";
}
