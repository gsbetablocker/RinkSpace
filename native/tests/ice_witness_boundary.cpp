#include "rinkspace/ice_actual_chart.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace rinkspace;
int failures=0,checks=0;
void require(bool ok,const char* label){++checks;if(!ok){++failures;std::fprintf(stderr,"FAIL: %s\n",label);}}
bool same(double a,double b){return std::memcmp(&a,&b,sizeof(a))==0;}
int main(){
    // This is an unchanged legal X boundary while a radius is updated. The
    // production zero-step shortcut is deliberately NOT used as the repro.
    std::array<double,152> values{};
    values[6]=.3;values[7]=-.2;values[9]=.1;values[10]=.03;
    values[11]=.25;values[12]=.5;values[13]=.75;
    values[49]=-.01;values[50]=.01;values[51]=0.;values[52]=.005;
    const double xlo=values[6]+values[9]*values[49],xhi=values[6]+values[9]*values[50];
    require(xhi-xlo>.02*values[9],"counterexample exceeds naive range only by FP rounding");
    auto delta=[](size_t i){return i==61?.001:0.;};
    for(bool trial:{false,true}){
        IceActualWitness w{};
        bool ok=ice_actual_chart_retract(values.data(),delta,w,trial);
        require(ok,trial?"constraint-trial radius step":"ordinary radius step");
        if(!ok)continue;
        for(int i=0;i<8;++i)require(w.x_deviation[i]>=-.01&&w.x_deviation[i]<=.01,"unchanged X exports inclusive original bounds");
        require(same(w.c[0]+values[9]*w.x_deviation[0],xlo),"min X preserved bit-for-bit");
        require(same(w.c[0]+values[9]*w.x_deviation[1],xhi),"max X preserved bit-for-bit");
        double old_radius=values[10]*values[11];
        double rebuilt=w.radius[0]*(1+w.radius_deviation[0]);
        require(std::abs(rebuilt-old_radius*std::exp(.001))<=8*std::numeric_limits<double>::epsilon()*old_radius,"radius update preserved");
        for(int i=0;i<32;++i)require(w.radius_deviation[i]>=-.01&&w.radius_deviation[i]<=.01,"radius deviation stays in original box");
    }
    // Scale invariance of the same representable counterexample over powers of two.
    for(int exponent:{-500,-100,0,100,500}){
        double x[8],r[32],pitch=std::ldexp(.1,exponent);
        for(int i=0;i<8;++i)x[i]=std::ldexp(i%2?xhi:xlo,exponent);
        for(int i=0;i<32;++i)r[i]=std::ldexp(.003*(i%4+1),exponent);
        IceActualWitness out{};bool ok=ice_actual_witness(x,r,pitch,out);
        require(ok,"scaled rounded boundary is valid");
        if(ok)for(int i=0;i<8;++i){
            require(out.x_deviation[i]>=-.01&&out.x_deviation[i]<=.01,"scaled deviations stay bounded");
            require(same(out.c[i/4]+pitch*out.x_deviation[i],x[i]),"scaled X remains exact");
        }
        // A real 0.1% additional spread is not roundoff and must remain rejected.
        x[1]=std::ldexp(.301001,exponent);const auto before=out;
        require(!ice_actual_witness(x,r,pitch,out),"real out-of-box X rejected");
        require(std::memcmp(&out,&before,sizeof(out))==0,"failure keeps output intact");
    }
    std::printf("checks=%d failures=%d\n",checks,failures);return failures?1:0;
}
