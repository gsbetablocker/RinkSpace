#include "rinkspace/ice_actual_witness.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace rinkspace;
using XValues=std::array<double,8>;
using Radii=std::array<double,32>;
static void require(bool ok,const char* message){
    if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}
}
static XValues seed_x(){return {{-.3,-.3,-.3,-.3,.2,.2,.2,.2}};}
static Radii seed_radii(){
    Radii result{};
    for(int house=0;house<8;++house)for(int layer=0;layer<4;++layer)
        result[house*4+layer]=.003*double(layer+1);
    return result;
}
static void roundoff_close(double actual,double rebuilt,long double scale,long double eps_count,
                           const char* message){
    // Independent forward-error check in wider host arithmetic where available.
    // The additive term covers gradual underflow, not a geometric tolerance.
    const long double error=std::abs(static_cast<long double>(rebuilt)-actual);
    const long double bound=eps_count*std::numeric_limits<double>::epsilon()*scale+
        4.L*std::numeric_limits<double>::denorm_min();
    require(std::isfinite(rebuilt)&&error<=bound,message);
}
static void inverse_or_exact_boundary(double actual,double rebuilt,double d,double inverse){
    if(d==inverse)return;
    require(d==-.01 || d==.01,"alternative deviation must be an inclusive boundary");
    require(std::memcmp(&actual,&rebuilt,sizeof(double))==0,"boundary deviation requires exact forward geometry");
}
static void verify(const XValues& x,const Radii& r,double pitch,const IceActualWitness& witness){
    for(int i=0;i<8;++i){
        const double d=witness.x_deviation[i],c=witness.c[i/4];
        require(std::isfinite(c)&&std::isfinite(d)&&d>=-.01&&d<=.01,"inclusive X deviation");
        inverse_or_exact_boundary(x[i],c+pitch*d,d,(x[i]-c)/pitch);
        long double scale=std::abs(static_cast<long double>(x[i]));
        if(std::abs(static_cast<long double>(c))>scale)scale=std::abs(static_cast<long double>(c));
        if(std::abs(static_cast<long double>(pitch*d))>scale)scale=std::abs(static_cast<long double>(pitch*d));
        roundoff_close(x[i],c+pitch*d,scale,12.L,"legacy X roundoff bound");
    }
    for(int i=0;i<32;++i){
        const double d=witness.radius_deviation[i],nominal=witness.radius[i%4];
        require(std::isfinite(nominal)&&nominal>0.&&std::isfinite(d)&&d>=-.01&&d<=.01,
                "inclusive radius deviation");
        inverse_or_exact_boundary(r[i],nominal*(1.+d),d,(r[i]-nominal)/nominal);
        const long double scale=r[i]>nominal?r[i]:nominal;
        roundoff_close(r[i],nominal*(1.+d),scale,4.L,"legacy radius roundoff bound");
    }
}
static IceActualWitness succeeds(const XValues& x,const Radii& r,double pitch){
    const auto old_x=x;const auto old_r=r;IceActualWitness out{};
    require(ice_actual_witness(x.data(),r.data(),pitch,out),"witness exists");
    require(x==old_x&&r==old_r,"actual geometry remains unchanged");
    verify(x,r,pitch,out);
    return out;
}
static void fails(const double* x,const double* r,double pitch,const char* message){
    IceActualWitness out{};
    for(double& value:out.c)value=42.;
    for(double& value:out.radius)value=43.;
    for(double& value:out.x_deviation)value=44.;
    for(double& value:out.radius_deviation)value=45.;
    const auto before=out;
    require(!ice_actual_witness(x,r,pitch,out),message);
    require(std::memcmp(&out,&before,sizeof(out))==0,"failure leaves output unchanged");
}
int main(){
    auto x=seed_x();auto r=seed_radii();
    const auto ordinary=succeeds(x,r,.1);
    require(ordinary.c[0]==-.3&&ordinary.c[1]==.2,"equal X values retain their midpoint");

    // The audit's counterexample is feasible at R=100, while a mean gauge is
    // infeasible. Dividing first loses its inclusive boundary; stable inversion
    // preserves it and its old forward geometry exactly.
    require(99./100.-1. < -.01&&101./100.-1. > .01,"division-first cancellation regression");
    require((99.-100.5)/100.5 < -.01,"mean gauge excludes legal geometry");
    for(int house=0;house<8;++house)for(int layer=0;layer<4;++layer)
        r[house*4+layer]=std::ldexp(house?101.:99.,layer);
    const auto counterexample=succeeds(x,r,.1);
    for(int layer=0;layer<4;++layer){
        require(counterexample.radius[layer]==std::ldexp(100.,layer),"singleton nominal interval");
        for(int house=0;house<8;++house)
            require(counterexample.radius[layer]*(1.+counterexample.radius_deviation[house*4+layer])==
                    r[house*4+layer],"counterexample reconstructs exactly");
    }

    // Multiple min/max ties are all retained and validated.
    const double tied[]={99.,99.,101.,101.,99.,101.,100.,100.};
    for(int house=0;house<8;++house)for(int layer=0;layer<4;++layer)
        r[house*4+layer]=std::ldexp(tied[house],layer);
    x={{-1.,1.,-1.,1.,9.,11.,9.,11.}};
    const auto ties=succeeds(x,r,100.);
    require(ties.c[0]==0.&&ties.c[1]==10.,"X boundary midpoint");
    require(ties.x_deviation[0]==-.01&&ties.x_deviation[1]==.01,"inclusive X endpoints");

    x=seed_x();r=seed_radii();r[0]=99.;r[4]=102.;
    fails(x.data(),r.data(),.1,"empty radius witness interval");
    r=seed_radii();x[0]=0.;x[1]=.0200001;
    fails(x.data(),r.data(),1.,"X range exceeds .02 pitch");

    // Exactly two adjacent representable nominal values: the rounded midpoint
    // is the wrong endpoint; nextafter search must recover the legal neighbor.
    x=seed_x();r=seed_radii();
    const double narrow_min=0x1.fd8a4971d3bbep-251;
    const double narrow_max=0x1.03eabe07a39a0p-250;
    const double narrow_low=narrow_max/1.01,narrow_high=narrow_min/.99;
    require(std::nextafter(narrow_low,narrow_high)==narrow_high,"adjacent radius interval");
    const double narrow_mid=narrow_low+(narrow_high-narrow_low)*.5;
    require((narrow_min-narrow_mid)/narrow_mid < -.01,"rounded radius midpoint fails");
    require(narrow_mid*.99!=narrow_min,"failed radius midpoint has no exact boundary witness");
    for(int house=0;house<8;++house)r[house*4+2]=house%2?narrow_max:narrow_min;
    require(succeeds(x,r,.1).radius[2]==narrow_low,"radius neighbor recovers inclusive export");

    const double x_min=0x1.356ca63b454ebp+175,x_max=0x1.df209134f9ca4p+176;
    const double large_pitch=0x1.fae6010478269p+181;
    const double x_low=x_max-.01*large_pitch,x_high=x_min+.01*large_pitch;
    require(std::nextafter(x_low,x_high)==x_high,"adjacent X interval");
    require((x_max-x_low)/large_pitch > .01,"rounded X midpoint fails");
    x={{x_min,x_max,x_min,x_max,0.,0.,0.,0.}};
    const auto adjacent_x=succeeds(x,seed_radii(),large_pitch);
    require(adjacent_x.c[0]==x_low&&adjacent_x.x_deviation[1]==.01,
            "X midpoint retains exact rounded-forward boundary witness");

    // Rounded old forward geometry may have slightly out-of-box inverse
    // quotients. Exact boundary-forward equality proves its legal old witness.
    x=seed_x();r=seed_radii();
    const double recovered_min=0x1.50f4be8d121bep+380;
    const double recovered_max=0x1.57c362b94b5d2p+380;
    const double recovered_nominal=recovered_max/1.01;
    require(recovered_nominal==recovered_min/.99,"rounded-forward singleton");
    require((recovered_min-recovered_nominal)/recovered_nominal < -.01&&
            (recovered_max-recovered_nominal)/recovered_nominal > .01,"inverse quotients need exact proof");
    require(recovered_nominal*.99==recovered_min&&recovered_nominal*1.01==recovered_max,
            "exact forward proves unchanged radius geometry");
    for(int house=0;house<8;++house)r[house*4]=house%2?recovered_max:recovered_min;
    const auto recovered=succeeds(x,r,.1);
    require(recovered.radius_deviation[0]==-.01&&recovered.radius_deviation[4]==.01,
            "recover proven radius boundary deviations");

    const double recovered_x_min=0x1.fcbb464a5be0fp+46;
    const double recovered_x_max=0x1.13a08d68b5c32p+47;
    const double recovered_pitch=0x1.09c4704c21c96p+49;
    x={{recovered_x_min,recovered_x_max,recovered_x_min,recovered_x_max,0.,0.,0.,0.}};
    const auto recovered_x=succeeds(x,seed_radii(),recovered_pitch);
    require((x[0]-recovered_x.c[0])/recovered_pitch < -.01&&recovered_x.x_deviation[0]==-.01,
            "recover proven X boundary deviation");

    // Nonempty singleton with no inclusive stable inverse and no exact old
    // boundary forward representation: unlike the recovered state above, the
    // minimum is one representable radius outside the boundary-forward value.
    x=seed_x();r=seed_radii();
    const double impossible_min=0x1.ffeab62351882p-236;
    const double impossible_max=0x1.052119ad2ae1fp-235;
    const double impossible_nominal=impossible_max/1.01;
    require(impossible_nominal==impossible_min/.99,"nonempty unrepresentable singleton");
    require((impossible_min-impossible_nominal)/impossible_nominal < -.01&&
            impossible_nominal*.99!=impossible_min,"inverse failure has no exact forward proof");
    for(int house=0;house<8;++house)r[house*4]=house%2?impossible_max:impossible_min;
    fails(x.data(),r.data(),.1,"explicitly reject unproved out-of-box inverse");

    // Preserve forward geometry throughout the FP64 exponent range, with
    // nonzero interior deviations, varying centers and every radius layer.
    for(int exponent:{-1000,-500,0,500,1000})for(int sample=0;sample<32;++sample){
        const double scale=std::ldexp(1.,exponent),pitch=.1*scale;
        for(int i=0;i<8;++i)x[i]=(i<4?-.3:.2)*scale+pitch*.009*std::sin(double(i*3+sample));
        for(int house=0;house<8;++house)for(int layer=0;layer<4;++layer)
            r[house*4+layer]=.003*(layer+1)*scale*(1.+.009*std::sin(double(house*7+layer+sample)));
        succeeds(x,r,pitch);
    }
    x.fill(0.);r.fill(std::numeric_limits<double>::denorm_min());
    succeeds(x,r,std::numeric_limits<double>::denorm_min());

    x=seed_x();r=seed_radii();
    const double infinity=std::numeric_limits<double>::infinity();
    const double nan=std::numeric_limits<double>::quiet_NaN();
    fails(nullptr,r.data(),.1,"null X");fails(x.data(),nullptr,.1,"null radius");
    for(double invalid:{0.,-1.,infinity,-infinity,nan})
        fails(x.data(),r.data(),invalid,"invalid pitch");
    for(int i=0;i<8;++i)for(double invalid:{infinity,-infinity,nan}){
        auto invalid_x=x;invalid_x[i]=invalid;
        fails(invalid_x.data(),r.data(),.1,"nonfinite X");
    }
    for(int i=0;i<32;++i)for(double invalid:{0.,-1.,infinity,-infinity,nan}){
        auto invalid_r=r;invalid_r[i]=invalid;
        fails(x.data(),invalid_r.data(),.1,"invalid radius");
    }
    std::puts("ice actual witness: inclusive boxes, audit counterexample, ties, narrow/singleton intervals, representability and FP64 forward geometry passed; no optimizer called");
}
