#pragma once
#include "coupled_dual.hpp"
#include <set>

namespace rinkspace {
enum class DualWorkingResult { solved, fallback, stopped };
struct DualWorkingStats {size_t iterations=0,factorizations=0;};

// Host-only active-set fast path for the SAME normalized convex Schur Gram.
// Rank decisions affect this attempt only. The original full-Gram KKT test
// certifies success; failure/cancellation never writes the caller's seed.
template<class Stop> inline DualWorkingResult coupled_dual_working_set(
        int n,int stride,const double* gram,const double* target,double* dual,const double* equality,
        double& final_kkt,DualWorkingStats& stats,Stop stop){
    if(stop())return DualWorkingResult::stopped;
    if(!coupled_dual_finite_input(n,stride,gram,target,dual,equality))return DualWorkingResult::fallback;
    double scale=1.;for(int i=0;i<n;++i){
        if((!equality[i]&&dual[i]<0.)||gram[size_t(i)*stride+i]<=0.)return DualWorkingResult::fallback;
        scale=std::max(scale,fabs(target[i]));
    }
    const double tolerance=8*std::numeric_limits<double>::epsilon()*scale;
    std::vector<double> current(dual,dual+n),z(n),rhs(n),residual(n);
    std::vector<unsigned char> active(n);for(int i=0;i<n;++i)active[i]=equality[i]||current[i]>0.;
    std::set<std::vector<uint64_t>> seen;
    // This is an attempted fast path, not an outer iteration/stop policy.
    // The reference 4096-sweep path remains the fallback from the ORIGINAL seed.
    for(int iteration=0;iteration<4096;++iteration){
        if(stop())return DualWorkingResult::stopped;++stats.iterations;
        std::vector<uint64_t> identity(2*size_t(n));
        if(n)std::memcpy(identity.data(),current.data(),n*sizeof(double));
        for(int i=0;i<n;++i)identity[size_t(n)+i]=active[i];
        if(!seen.insert(std::move(identity)).second)return DualWorkingResult::fallback;
        std::vector<int> order;for(int i=0;i<n;++i)if(active[i])order.push_back(i);
        const int m=int(order.size());std::vector<double> l(size_t(m)*m,0.),diagonal(m);
        double largest=0.;for(int i=0;i<m;++i){diagonal[i]=gram[size_t(order[i])*stride+order[i]];largest=std::max(largest,diagonal[i]);}
        const double rank_tolerance=64*std::numeric_limits<double>::epsilon()*std::max(1,m)*largest;
        if(stop())return DualWorkingResult::stopped;++stats.factorizations;int rank=0;
        for(int k=0;k<m;++k){
            int pivot=k;for(int i=k+1;i<m;++i)if(diagonal[i]>diagonal[pivot])pivot=i;
            if(!finite_scalar(diagonal[pivot]))return DualWorkingResult::fallback;
            if(diagonal[pivot]<=rank_tolerance)break;
            std::swap(order[k],order[pivot]);std::swap(diagonal[k],diagonal[pivot]);
            for(int j=0;j<k;++j)std::swap(l[size_t(k)*m+j],l[size_t(pivot)*m+j]);
            l[size_t(k)*m+k]=std::sqrt(diagonal[k]);
            for(int i=k+1;i<m;++i){double x=gram[size_t(order[i])*stride+order[k]];
                for(int j=0;j<k;++j)x=fma(-l[size_t(i)*m+j],l[size_t(k)*m+j],x);
                const double value=x/l[size_t(k)*m+k];if(!finite_scalar(value))return DualWorkingResult::fallback;
                l[size_t(i)*m+k]=value;diagonal[i]=fma(-value,value,diagonal[i]);}
            ++rank;
        }
        // Independent cuDSS responses can round the two triangles differently.
        // The factor is an approximate solve only; refinement and the final
        // certificate always use EVERY entry of the original unsymmetrized G.
        auto solve=[&](const double* b,std::vector<double>& out){
            std::fill(out.begin(),out.end(),0.);std::vector<double> y(rank);
            for(int i=0;i<rank;++i){double x=b[order[i]];
                for(int j=0;j<i;++j)x=fma(-l[size_t(i)*m+j],y[j],x);
                y[i]=x/l[size_t(i)*m+i];if(!finite_scalar(y[i]))return false;}
            for(int i=rank-1;i>=0;--i){double x=y[i];
                for(int j=i+1;j<rank;++j)x=fma(-l[size_t(j)*m+i],out[order[j]],x);
                out[order[i]]=x/l[size_t(i)*m+i];if(!finite_scalar(out[order[i]]))return false;}
            return true;
        };
        // A violated dependent cut can require a feasible basis exchange.
        // An absolute rank-truncated solve can instead add and immediately
        // remove the same zero multiplier forever. Use the dependent column
        // to find a near-null descent direction and move to a multiplier bound.
        // Its curvature and decrease use the ORIGINAL Gram, including the
        // small nonzero curvature discarded by the tentative factorization.
        if(rank<m){
            bool exchanged=false;
            std::vector<double> column(n),direction(n),rr(n),next(n);
            for(int i=0;i<n;++i)rr[i]=coupled_dual_residual(i,n,stride,gram,target,current.data());
            for(int free_index=rank;free_index<m&&!exchanged;++free_index){
                if(stop())return DualWorkingResult::stopped;
                const int free=order[free_index];
                for(int i=0;i<n;++i)column[i]=gram[size_t(i)*stride+free];
                if(!solve(column.data(),direction))continue;
                for(double& value:direction)value=-value;direction[free]=1.;
                double slope=0.,norm1=0.,curvature=0.;
                for(int i=0;i<n;++i){slope=fma(direction[i],rr[i],slope);norm1+=fabs(direction[i]);}
                if(!finite_scalar(slope)||fabs(slope)<=tolerance*std::max(1.,norm1))continue;
                for(int i=0;i<n;++i){double value=0.;
                    for(int j=0;j<n;++j)value=fma(gram[size_t(i)*stride+j],direction[j],value);
                    curvature=fma(direction[i],value,curvature);}
                if(!finite_scalar(curvature)||fabs(curvature)>rank_tolerance*std::max(1.,norm1*norm1))continue;
                if(slope<0.){for(double& value:direction)value=-value;slope=-slope;}
                double alpha=INFINITY;int leaving=-1;
                for(int i=0;i<n;++i)if(!equality[i]&&direction[i]<0.){
                    double limit=-current[i]/direction[i];if(limit<alpha){alpha=limit;leaving=i;}}
                if(leaving<0||!finite_scalar(alpha)||!(alpha>0.))continue;
                const double gain=alpha*(slope-.5*alpha*curvature);
                if(!finite_scalar(gain)||!(gain>0.))continue;
                bool feasible=true;
                for(int i=0;i<n;++i){next[i]=i==leaving?0.:fma(alpha,direction[i],current[i]);
                    feasible&=finite_scalar(next[i])&&(equality[i]||next[i]>=0.);}
                if(feasible){current=next;active[leaving]=0;exchanged=true;}
            }
            if(exchanged)continue;
        }
        if(!solve(target,z))return DualWorkingResult::fallback;
        // Binary64 iterative refinement uses the original ordered FMA residual,
        // not a recursively updated approximation as the final certificate.
        double previous=INFINITY;
        for(;;){
            if(stop())return DualWorkingResult::stopped;double norm=0.;
            for(int i:order){residual[i]=coupled_dual_residual(i,n,stride,gram,target,z.data());
                if(!finite_scalar(residual[i]))return DualWorkingResult::fallback;norm=std::max(norm,fabs(residual[i]));}
            if(norm<=tolerance||!(norm<previous))break;previous=norm;
            if(!solve(residual.data(),rhs))return DualWorkingResult::fallback;
            bool changed=false;for(int i:order){double next=z[i]+rhs[i];if(!finite_scalar(next))return DualWorkingResult::fallback;changed|=next!=z[i];z[i]=next;}
            if(!changed)break;
        }
        int blocking=-1;double fraction=1.;
        for(int i:order)if(!equality[i]&&z[i]<0.){
            double denominator=current[i]-z[i];if(!finite_scalar(denominator)||!(denominator>0.))return DualWorkingResult::fallback;
            double value=current[i]/denominator;if(value<fraction){fraction=value;blocking=i;}}
        if(blocking>=0){
            for(int i=0;i<n;++i){double next=fma(fraction,z[i]-current[i],current[i]);
                if(!finite_scalar(next)||(!equality[i]&&next<0.&&i!=blocking))return DualWorkingResult::fallback;current[i]=next;}
            current[blocking]=0.;active[blocking]=0;continue;
        }
        current=z;
        const double kkt=coupled_dual_kkt(n,stride,gram,target,current.data(),equality);
        if(!coupled_dual_unresolved(n,target,kkt)){
            if(stop())return DualWorkingResult::stopped;
            std::copy(current.begin(),current.end(),dual);final_kkt=kkt;return DualWorkingResult::solved;
        }
        int entering=-1;double worst=tolerance;
        for(int i=0;i<n;++i)if(!active[i]){
            double r=coupled_dual_residual(i,n,stride,gram,target,current.data());
            if(!finite_scalar(r))return DualWorkingResult::fallback;
            if(r>worst){worst=r;entering=i;}}
        if(entering<0){
            // Refine the already computed solution instead of immediately
            // discarding it and restarting an all-pairs solve from the seed.
            // Coordinate corrections use original ordered FP64 residuals;
            // only the unchanged full-Gram KKT certificate accepts a result.
            auto polished=current;
            std::set<std::vector<uint64_t>> rounding_states;
            for(int sweep=0;sweep<4096;++sweep){
                if(stop())return DualWorkingResult::stopped;
                std::vector<uint64_t> identity(size_t(n),0);
                if(n)std::memcpy(identity.data(),polished.data(),n*sizeof(double));
                if(!rounding_states.insert(std::move(identity)).second)break;
                for(int i=0;i<n;++i){
                    if(stop())return DualWorkingResult::stopped;
                    double r=coupled_dual_residual(i,n,stride,gram,target,polished.data());
                    double value=polished[i]+r/gram[size_t(i)*stride+i];
                    if(!finite_scalar(value))return DualWorkingResult::fallback;
                    polished[i]=equality[i]?value:std::max(0.,value);
                }
                const double certificate=coupled_dual_kkt(n,stride,gram,target,polished.data(),equality);
                if(!coupled_dual_unresolved(n,target,certificate)){
                    if(stop())return DualWorkingResult::stopped;
                    std::copy(polished.begin(),polished.end(),dual);final_kkt=certificate;return DualWorkingResult::solved;
                }
            }
            // A rank-truncated coordinate with exactly zero multiplier need
            // not remain in the active factorization. Retry the independent
            // basis before falling back with the caller's seed unchanged.
            bool changed=false;
            for(int i=0;i<n;++i)if(active[i]&&!equality[i]&&current[i]==0.){active[i]=0;changed=true;}
            if(changed)continue;
            return DualWorkingResult::fallback;
        }
        active[entering]=1;
    }
    return DualWorkingResult::fallback;
}
}
