#pragma once
#include "core_types.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rinkspace {
RS_HD inline double coupled_dual_residual(int i,int count,int stride,const double* gram,const double* target,const double* dual){
    double value=target[i];for(int j=0;j<count;++j)value=fma(-gram[size_t(i)*stride+j],dual[j],value);return value;
}
RS_HD inline double coupled_dual_kkt(int count,int stride,const double* gram,const double* target,const double* dual,const double* equality){
    double result=0.;for(int i=0;i<count;++i){double r=coupled_dual_residual(i,count,stride,gram,target,dual);
        result=fmax(result,(equality[i]||dual[i]>0.)?fabs(r):fmax(0.,r));}return result;
}
RS_HD inline int coupled_dual_minimize(int count,int stride,const double* gram,const double* target,double* dual,const double* equality,double* final_kkt=nullptr){
    double scale=1.;for(int i=0;i<count;++i)scale=fmax(scale,fabs(target[i]));
    const double tolerance=8*std::numeric_limits<double>::epsilon()*scale;
    double kkt=0.;
    for(int sweep=0;sweep<4096;++sweep){
        for(int i=0;i<count;++i){double r=coupled_dual_residual(i,count,stride,gram,target,dual);
            dual[i]=equality[i]?dual[i]+r:fmax(0.,dual[i]+r);}
        // Most restored cuts already satisfy the same KKT tolerance after
        // coordinate updates; do not scan every pair once the solve is done.
        kkt=coupled_dual_kkt(count,stride,gram,target,dual,equality);
        if(kkt<=tolerance){if(final_kkt)*final_kkt=kkt;return sweep+1;}
        // Exact two-coordinate minimization accelerates nearly parallel faces
        // without changing their Schur metric or nonnegative multiplier domain.
        for(int i=0;i<count;++i)for(int j=i+1;j<count;++j){double cross=gram[size_t(i)*stride+j];
            if(fabs(cross)<=std::numeric_limits<double>::epsilon())continue;
            double ri=coupled_dual_residual(i,count,stride,gram,target,dual),rj=coupled_dual_residual(j,count,stride,gram,target,dual);
            double best=0.,bi=0.,bj=0.;
            auto offer=[&](double di,double dj){
                if(!finite_scalar(di)||!finite_scalar(dj)||(!equality[i]&&dual[i]+di<0.)||(!equality[j]&&dual[j]+dj<0.))return;
                double value=-(ri*di+rj*dj)+.5*(di*di+dj*dj+2*cross*di*dj);
                if(value<best){best=value;bi=di;bj=dj;}
            };
            const double det=(1-cross)*(1+cross);
            if(det>64*std::numeric_limits<double>::epsilon())offer((ri-cross*rj)/det,(rj-cross*ri)/det);
            if(!equality[i]){double di=-dual[i],dj=rj-cross*di;if(!equality[j])dj=fmax(-dual[j],dj);offer(di,dj);}
            if(!equality[j]){double dj=-dual[j],di=ri-cross*dj;if(!equality[i])di=fmax(-dual[i],di);offer(di,dj);}
            dual[i]+=bi;dual[j]+=bj;
        }
        // Large cancelling multipliers must not loosen primal feasibility.
        kkt=coupled_dual_kkt(count,stride,gram,target,dual,equality);
        if(kkt<=tolerance){if(final_kkt)*final_kkt=kkt;return sweep+1;}
    }
    if(final_kkt)*final_kkt=kkt;
    return 4096;
}
RS_HD inline bool coupled_dual_unresolved(int count,const double* target,double kkt){
    double scale=1.;for(int i=0;i<count;++i)scale=fmax(scale,fabs(target[i]));
    return kkt>8*std::numeric_limits<double>::epsilon()*scale;
}
// Exact memoization within one fixed multiplier state. A coordinate or pair
// update invalidates cached residuals before the next use. Summation order,
// pair order, update arithmetic, stopping tolerance and fallback are unchanged.
struct CoupledDualCache {
    int count=0,stride=0;
    const double *gram=nullptr,*target=nullptr,*dual=nullptr,*equality=nullptr;
    std::vector<double> values,cycle_anchor;
    std::vector<uint64_t> stamps;
    uint64_t generation=1,hits=0,evaluations=0,skipped_sweeps=0;
    int cycle_power=1,cycle_distance=0;
    void reset(int n,int ld,const double* g,const double* t,const double* d,const double* e){
        count=n;stride=ld;gram=g;target=t;dual=d;equality=e;
        values.resize(n);stamps.assign(n,0);generation=1;hits=evaluations=0;
        cycle_anchor.assign(d,d+n);cycle_power=1;cycle_distance=0;skipped_sweeps=0;
    }
    void changed(double before,double after){
        if(before!=after||std::signbit(before)!=std::signbit(after)){
            if(++generation==0){std::fill(stamps.begin(),stamps.end(),0);generation=1;}
        }
    }
    double residual(int i){
        if(stamps[i]==generation){++hits;return values[i];}
        ++evaluations;stamps[i]=generation;
        return values[i]=coupled_dual_residual(i,count,stride,gram,target,dual);
    }
    void pair(int i,int j,double& ri,double& rj){
        if(stamps[i]==generation||stamps[j]==generation){ri=residual(i);rj=residual(j);return;}
        // These two reductions are independent. Interleave them so the CPU
        // can overlap FMA dependency chains, preserving each sum's order.
        ri=target[i];rj=target[j];const double* a=gram+size_t(i)*stride;const double* b=gram+size_t(j)*stride;
        for(int k=0;k<count;++k){ri=fma(-a[k],dual[k],ri);rj=fma(-b[k],dual[k],rj);}
        evaluations+=2;stamps[i]=stamps[j]=generation;values[i]=ri;values[j]=rj;
    }
    double kkt(){double result=0.;for(int i=0;i<count;++i){double r=residual(i);
        result=fmax(result,(equality[i]||dual[i]>0.)?fabs(r):fmax(0.,r));}return result;}
    int repeating_sweeps(int remaining){
        ++cycle_distance;
        // Exact Brent cycle detection, not an approximate stagnation test.
        // Deterministic repeated binary64 states repeat the same future
        // sweeps. Skip only whole periods so the original 4096th state,
        // unresolved status and KKT value are unchanged.
        if(std::memcmp(cycle_anchor.data(),dual,size_t(count)*sizeof(double))==0){
            int skip=(remaining/cycle_distance)*cycle_distance;skipped_sweeps+=skip;return skip;
        }
        if(cycle_distance==cycle_power){cycle_anchor.assign(dual,dual+count);cycle_power*=2;cycle_distance=0;}
        return 0;
    }
};
inline int coupled_dual_minimize_cached(int count,int stride,const double* gram,const double* target,double* dual,const double* equality,CoupledDualCache& cache,double* final_kkt=nullptr){
    cache.reset(count,stride,gram,target,dual,equality);
    double scale=1.;for(int i=0;i<count;++i)scale=fmax(scale,fabs(target[i]));
    const double tolerance=8*std::numeric_limits<double>::epsilon()*scale;
    double kkt=0.;
    for(int sweep=0;sweep<4096;++sweep){
        for(int i=0;i<count;++i){double r=cache.residual(i);
            const double old=dual[i];dual[i]=equality[i]?dual[i]+r:fmax(0.,dual[i]+r);cache.changed(old,dual[i]);}
        // Most restored cuts already satisfy the same KKT tolerance after
        // coordinate updates; do not scan every pair once the solve is done.
        kkt=cache.kkt();
        if(kkt<=tolerance){if(final_kkt)*final_kkt=kkt;return sweep+1;}
        // Exact two-coordinate minimization accelerates nearly parallel faces
        // without changing their Schur metric or nonnegative multiplier domain.
        for(int i=0;i<count;++i)for(int j=i+1;j<count;++j){double cross=gram[size_t(i)*stride+j];
            if(fabs(cross)<=std::numeric_limits<double>::epsilon())continue;
            double ri,rj;cache.pair(i,j,ri,rj);
            double best=0.,bi=0.,bj=0.;
            auto offer=[&](double di,double dj){
                if(!finite_scalar(di)||!finite_scalar(dj)||(!equality[i]&&dual[i]+di<0.)||(!equality[j]&&dual[j]+dj<0.))return;
                double value=-(ri*di+rj*dj)+.5*(di*di+dj*dj+2*cross*di*dj);
                if(value<best){best=value;bi=di;bj=dj;}
            };
            const double det=(1-cross)*(1+cross);
            if(det>64*std::numeric_limits<double>::epsilon())offer((ri-cross*rj)/det,(rj-cross*ri)/det);
            if(!equality[i]){double di=-dual[i],dj=rj-cross*di;if(!equality[j])dj=fmax(-dual[j],dj);offer(di,dj);}
            if(!equality[j]){double dj=-dual[j],di=ri-cross*dj;if(!equality[i])di=fmax(-dual[i],di);offer(di,dj);}
            const double old_i=dual[i],old_j=dual[j];dual[i]+=bi;dual[j]+=bj;cache.changed(old_i,dual[i]);cache.changed(old_j,dual[j]);
        }
        // Large cancelling multipliers must not loosen primal feasibility.
        kkt=cache.kkt();
        if(kkt<=tolerance){if(final_kkt)*final_kkt=kkt;return sweep+1;}
        sweep+=cache.repeating_sweeps(4095-sweep);
    }
    if(final_kkt)*final_kkt=kkt;
    return 4096;
}
}
