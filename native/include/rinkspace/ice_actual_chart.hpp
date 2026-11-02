#pragma once
#include "ice_actual_witness.hpp"

namespace rinkspace {
// Finite layout146 chart. Storage remains the legacy board/layout: six
// nominal witnesses have no independent tangent, forty actual quantities do.
constexpr int ice_chart_actual_x=10,ice_chart_actual_log_radius=11;
constexpr int ice_chart_witness_x=12,ice_chart_witness_radius=13;
RS_HD inline bool ice_actual_chart_enabled(const int* chart){return chart&&chart[6]==ice_chart_witness_x;}
RS_HD inline bool ice_actual_component(size_t i){return i==6||i==7||(i>=10&&i<=13)||(i>=49&&i<57)||(i>=61&&i<93);}
RS_HD inline bool ice_actual_chart_pattern(const int* chart,size_t n){
    if(!chart||n<152||chart[6]!=12||chart[7]!=12||chart[9]!=1)return false;
    for(int i=10;i<=13;++i)if(chart[i]!=13)return false;
    for(int i=49;i<57;++i)if(chart[i]!=10)return false;
    for(int i=61;i<93;++i)if(chart[i]!=11)return false;
    for(size_t i=0;i<n;++i)if(chart[i]>=10&&!ice_actual_component(i))return false;
    return true;
}
template<class Delta> RS_HD inline bool ice_actual_chart_retract(const double* values,Delta delta_at,IceActualWitness& witness,bool constraint_trial=false){
    const double* q=values+6;double x[8],r[32];
    const double pitch=q[3]*std::exp(delta_at(9));
    for(int i=0;i<8;++i)x[i]=q[i<4?0:1]+q[3]*q[39+4+i]+delta_at(49+i);
    const double radius[]={q[4]*q[5],q[4]*q[6],q[4]*q[7],q[4]};
    for(int i=0;i<32;++i)r[i]=radius[i%4]*(1.+q[39+16+i])*std::exp(delta_at(61+i));
    if(ice_actual_witness(x,r,pitch,witness))return true;
    if(!constraint_trial||!finite_scalar(pitch)||!(pitch>0.)||!finite_array(x,8)||!finite_array(r,32))return false;
    // A QP separation point can violate a group range. Keep its ACTUAL
    // geometry representable so the group face and its normal can repair it.
    // These temporary deviations are not bounded witnesses. Every exported
    // accepted state still passes the unchanged old boxes and all relations.
    IceActualWitness temporary{};
    temporary.c[0]=q[0];temporary.c[1]=q[1];
    for(int layer=0;layer<4;++layer){if(!(radius[layer]>0.)||!finite_scalar(radius[layer]))return false;temporary.radius[layer]=radius[layer];}
    for(int i=0;i<8;++i)temporary.x_deviation[i]=(x[i]-temporary.c[i/4])/pitch;
    for(int i=0;i<32;++i){if(!(r[i]>0.))return false;temporary.radius_deviation[i]=(r[i]-radius[i%4])/radius[i%4];}
    if(!finite_array(temporary.x_deviation,8)||!finite_array(temporary.radius_deviation,32))return false;
    witness=temporary;return true;
}
RS_HD inline double ice_actual_export_component(const IceActualWitness& w,size_t i){
    if(i==6||i==7)return w.c[i-6];
    if(i==10)return w.radius[3];
    if(i>=11&&i<=13)return w.radius[i-11]/w.radius[3];
    if(i>=49&&i<57)return w.x_deviation[i-49];
    if(i>=61&&i<93)return w.radius_deviation[i-61];
    return std::numeric_limits<double>::quiet_NaN();
}
} // namespace rinkspace
