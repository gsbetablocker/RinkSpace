#pragma once
#include "ice_tolerance.hpp"

namespace rinkspace {

// Compatibility witness only: these six independent groups do not check rink
// topology, nominal priors, ratio bounds or cross-group ordering. The caller
// must retain those checks. Radii and their deviations are house-major [8][4].
struct IceActualWitness {
    double c[2];
    double radius[4];
    double x_deviation[8];
    double radius_deviation[32];
};

namespace ice_actual_detail {
constexpr double epsilon=ice_relative_tolerance;

RS_HD inline double midpoint(double low,double high){
    // Avoid overflow, including an interval that straddles zero. This also
    // keeps equal subnormal endpoints intact (unlike low/2 + high/2).
    return low<0. && high>0. ? (low+high)*.5 : low+(high-low)*.5;
}

RS_HD inline double next_toward(double value,double target){
#if defined(__CUDA_ARCH__)
    return ::nextafter(value,target);
#else
    return std::nextafter(value,target);
#endif
}

RS_HD inline double deviation(double actual,double nominal,double pitch,bool relative){
    // (r-R)/R is algebraically r/R-1, without cancellation after division.
    // In particular, r={99,101}, R=100 exports exactly {-0.01,+0.01}.
    return (actual-nominal)/(relative?nominal:pitch);
}

RS_HD inline bool same_bits(double a,double b){
    const auto* pa=reinterpret_cast<const unsigned char*>(&a);
    const auto* pb=reinterpret_cast<const unsigned char*>(&b);
    for(std::size_t i=0;i<sizeof(double);++i)if(pa[i]!=pb[i])return false;
    return true;
}

RS_HD inline double forward(double nominal,double d,double pitch,bool relative){
    if(relative)return nominal*(1.+d);
    const double offset=pitch*d;
    return nominal+offset;
}

RS_HD inline bool export_deviation(double actual,double nominal,double pitch,
                                   bool relative,double& d){
    d=deviation(actual,nominal,pitch,relative);
    if(!finite_scalar(d))return false;
    if(d>=-epsilon && d<=epsilon)return true;
    const double boundary=d < -epsilon ? -epsilon : epsilon;
    // A rounded native forward state can have a slightly out-of-box inverse.
    // Accept its boundary witness only with exact object-representation
    // equality, including the sign of zero. This selects a proven old forward
    // representation of unchanged actual geometry; it is not geometric clip.
    if(!same_bits(forward(nominal,boundary,pitch,relative),actual))return false;
    d=boundary;
    return true;
}

RS_HD inline bool choose_nominal(double minimum,double maximum,double pitch,
                                 bool relative,double low,double high,double& nominal){
    if(!(low<=high) || !finite_scalar(low) || !finite_scalar(high))return false;
    bool checked_endpoints=false;
    while(low<=high){
        const double candidate=midpoint(low,high);
        const double lower=deviation(minimum,candidate,pitch,relative);
        const double upper=deviation(maximum,candidate,pitch,relative);
        if(!finite_scalar(lower) || !finite_scalar(upper))return false;
        double lower_export,upper_export;
        const bool lower_ok=export_deviation(minimum,candidate,pitch,relative,lower_export);
        const bool upper_ok=export_deviation(maximum,candidate,pitch,relative,upper_export);
        const bool too_high=lower < -epsilon && !lower_ok;
        const bool too_low=upper > epsilon && !upper_ok;
        if(!too_high && !too_low){nominal=candidate;return true;}
        // An exact rounded-forward boundary witness can sit just beyond the
        // stable-inverse admissible values. Check both interval endpoints
        // before pruning by monotonicity of the stable inverse.
        if(!checked_endpoints){
            for(int endpoint=0;endpoint<2;++endpoint){
                const double value=endpoint?high:low;
                if(export_deviation(minimum,value,pitch,relative,lower_export) &&
                   export_deviation(maximum,value,pitch,relative,upper_export)){
                    nominal=value;return true;
                }
            }
            checked_endpoints=true;
        }
        // Both deviations decrease monotonically with nominal. For radii,
        // subtraction is exact by Sterbenz on this near-unity interval.
        if(too_high && too_low)return false;
        if(too_low){
            if(!(candidate<high))return false;
            low=next_toward(candidate,high);
        }else{
            if(!(candidate>low))return false;
            high=next_toward(candidate,low);
        }
        // Each iteration excludes the rejected representable value. No
        // tolerance, iteration cap or geometric clipping widens the box.
    }
    return false;
}

RS_HD inline double magnitude(double x){return x<0.?-x:x;}

RS_HD inline bool reconstruction_close(double actual,double reconstructed,double scale,
                                       double relative_error){
    if(!finite_scalar(reconstructed))return false;
    const double difference=magnitude(reconstructed-actual);
    if(!finite_scalar(difference))return false;
    // Absolute term covers gradual-underflow rounding. Division avoids
    // overflow while testing the ordinary FP64 forward-error bound.
    return difference<=4.*std::numeric_limits<double>::denorm_min() ||
        (scale>0. && difference/scale<=relative_error);
}
} // namespace ice_actual_detail

// Prefer the midpoint of each mathematical witness interval, then search its
// representable values if rounding puts a deviation outside inclusive +/-1%.
// If the inverse itself rounds outside the box, a legal boundary deviation
// is accepted only when the native forward map reproduces actual bit for bit.
// Unproved/unrepresentable exports return false, as do invalid inputs. Neither
// geometric clipping nor a relaxed comparison can produce success, and
// out is updated only after every group and its old forward map is verified.
// Assumes ordinary IEEE FP64 arithmetic with gradual underflow, not fast-math.
RS_HD inline bool ice_actual_witness(const double* actual_x,const double* actual_radius,
                                     double pitch,IceActualWitness& out){
    using namespace ice_actual_detail;
    if(!finite_array(actual_x,8) || !finite_array(actual_radius,32) ||
       !finite_scalar(pitch) || !(pitch>0.))return false;
    for(int i=0;i<32;++i)if(!(actual_radius[i]>0.))return false;
    IceActualWitness result{};
    const double largest=std::numeric_limits<double>::max();
    const double machine_epsilon=std::numeric_limits<double>::epsilon();
    for(int group=0;group<2;++group){
        double minimum=actual_x[group*4],maximum=minimum;
        for(int j=1;j<4;++j){
            const double value=actual_x[group*4+j];
            if(value<minimum)minimum=value;
            if(value>maximum)maximum=value;
        }
        const double range=maximum-minimum;
        if(!finite_scalar(range) || range>2*epsilon*pitch)return false;
        const double delta=epsilon*pitch;
        double low=maximum-delta,high=minimum+delta;
        if(low < -largest)low=-largest;
        if(high > largest)high=largest;
        if(!choose_nominal(minimum,maximum,pitch,false,low,high,result.c[group]))return false;
        for(int j=0;j<4;++j){
            const int i=group*4+j;
            double d;
            if(!export_deviation(actual_x[i],result.c[group],pitch,false,d) ||
               d < -epsilon || d > epsilon)return false;
            const double offset=pitch*d;
            const double rebuilt=result.c[group]+offset;
            double scale=magnitude(actual_x[i]);
            if(magnitude(result.c[group])>scale)scale=magnitude(result.c[group]);
            if(magnitude(offset)>scale)scale=magnitude(offset);
            // Three rounded operations: a conservative 12 eps * max scale
            // (plus four subnormal units), including cancellation in X.
            if(!reconstruction_close(actual_x[i],rebuilt,scale,12.*machine_epsilon))return false;
            result.x_deviation[i]=d;
        }
    }
    for(int layer=0;layer<4;++layer){
        double minimum=actual_radius[layer],maximum=minimum;
        for(int house=1;house<8;++house){
            const double value=actual_radius[house*4+layer];
            if(value<minimum)minimum=value;
            if(value>maximum)maximum=value;
        }
        // Positive division preserves extrema: these are max(r/1.01) and
        // min(r/.99). Intersect an overflowing upper endpoint with finite R.
        const double low=maximum/(1+epsilon);
        double high=minimum/(1-epsilon);
        if(high>largest)high=largest;
        if(!(low>0.) || !choose_nominal(minimum,maximum,pitch,true,low,high,result.radius[layer]))return false;
        for(int house=0;house<8;++house){
            const int i=house*4+layer;
            double d;
            if(!export_deviation(actual_radius[i],result.radius[layer],pitch,true,d) ||
               d < -epsilon || d > epsilon)return false;
            const double rebuilt=result.radius[layer]*(1.+d);
            const double scale=actual_radius[i]>result.radius[layer]?actual_radius[i]:result.radius[layer];
            // Stable inverse plus old multiply/add forward map: 4 eps *
            // max(r,R), plus four subnormal units, bounds FP64 roundoff.
            if(!reconstruction_close(actual_radius[i],rebuilt,scale,4.*machine_epsilon))return false;
            result.radius_deviation[i]=d;
        }
    }
    out=result;
    return true;
}
} // namespace rinkspace
