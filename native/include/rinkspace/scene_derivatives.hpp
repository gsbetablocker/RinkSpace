#pragma once
#include "local_jet.hpp"
#include "entities.hpp"
namespace rinkspace {
/* Finite coordinate charts: additive, positive log, log(exponent-1), and
 * three ordered (0,1) radii expressed by four positive interval lengths. */
RS_HD inline double scene_chart_component(const double* values,const int* chart,
    const double* directions,int q,int column,size_t i){
    double d=directions[i*q+column];int type=chart?chart[i]:0;
    if(type==1)d*=values[i];else if(type==2)d*=values[i]-1.;
    else if(type>=3&&type<=5){size_t start=i-(type-3);double ratio=values[i],previous=0.;d=0.;
        for(int k=0;k<3;++k){double gap=values[start+k]-previous;previous=values[start+k];d+=gap*((start+k<=i?1.:0.)-ratio)*directions[(start+k)*q+column];}}
    return d;
}
RS_HD inline void scene_chart_direction(const double* values,size_t n,const int* chart,
    const double* directions,int q,int column,double* out){
    for(size_t i=0;i<n;++i)out[i*q+column]=scene_chart_component(values,chart,directions,q,column,i);
}
template<class Delta> RS_HD inline double scene_chart_retract_delta(const double* values,const int* chart,size_t index,Delta delta_at){
    int type=chart?chart[index]:0;double delta=delta_at(index);
    if(type==0)return values[index]+delta;
    if(type==1)return delta==0.?values[index]:values[index]*std::exp(delta);
    if(type==2)return delta==0.?values[index]:1.+(values[index]-1.)*std::exp(delta);
    size_t start=index-(type-3);double logits[4],previous=0.;bool moved=false;
    for(int k=0;k<3;++k){double d=delta_at(start+k);
        moved|=d!=0.;logits[k]=std::log(values[start+k]-previous)+d;previous=values[start+k];}
    if(!moved)return values[index];logits[3]=std::log(1.-values[start+2]);
    double maximum=std::max(std::max(logits[0],logits[1]),std::max(logits[2],logits[3])),sum=0.,prefix=0.;
    for(int k=0;k<4;++k){double w=std::exp(logits[k]-maximum);sum+=w;if(start+k<=index)prefix+=w;}
    return prefix/sum;
}
RS_HD inline double scene_chart_retract(const double* values,const int* chart,size_t index,const double* directions,int q,const double* step,double alpha){
    return scene_chart_retract_delta(values,chart,index,[&](size_t row){double d=0.;for(int j=0;j<q;++j)d+=directions[row*q+j]*(alpha*step[j]);return d;});
}
/* Immutable identity references, all values are in the common scene state.
 * family: 0 unchanged material seed, 1 monotone shape10, 2 unfolded square,
 * 3 rectangle side. type: 0 explicit primitive, 1 shared ice, 2 print.
 * For prints, code=1 declares containment in the physical unit-panel chart;
 * code=0 has no such relation. It is not inferred from a bounding box. */
struct ScenePath {
    int type,code,size,placement,shape,shape_count,family,side,reverse,vertices,count,bounds;
};
constexpr int scene_path_width=12;
static_assert(sizeof(ScenePath)==scene_path_width*sizeof(int),"packed scene path ABI");
RS_HD inline Status scene_print_chart(const ScenePath& path,const double* data,const double* state,
    const double* directions,int q,int column,LocalJet station,Vec2Value<LocalJet>& chart,
    Vec2Value<LocalJet>* material=nullptr){
    auto value=[&](int i){return LocalJet(state[i],directions?directions[size_t(i)*q+column]:0.);};
    LocalJet install[5]={value(path.placement),value(path.placement+1),value(path.placement+2),
        exp(value(path.size)),exp(value(path.size+1))};
    if(!(install[3]>0&&install[4]>0)||!finite_array(install,5))return Status::nonfinite;
    if(station<0||station>1)return Status::outside_support;
    LocalJet u{},v{};
    if(path.family==2||path.family==3){
        LocalJet values[4];for(int k=0;k<path.shape_count;++k)values[k]=value(path.shape+k);
        LocalJet vertices[10];auto st=print_analytic_vertices(install,values,path.family==2?0:1,vertices);if(st!=Status::ok)return st;
        LocalJet coordinate=path.reverse?1-station:station;
        LocalJet s=path.family==2?4*coordinate:coordinate;
        int edge=path.family==2?std::min(int(std::floor(primal(s))),3):path.side;
        LocalJet t=path.family==2?s-edge:s;
        u=(1-t)*vertices[2*edge]+t*vertices[2*edge+2];v=(1-t)*vertices[2*edge+1]+t*vertices[2*edge+3];
    }else{
        if(path.count<2)return Status::invalid_argument;
        LocalJet t=station*(path.count-1);int j=std::min(int(std::floor(primal(t))),path.count-2);t-=j;
        const double* seed=data+path.vertices;
        u=(1-t)*seed[2*j]+t*seed[2*j+2];v=(1-t)*seed[2*j+1]+t*seed[2*j+3];
        // Keep the current interpolation's proven endpoint roundoff behavior.
        u.a=print_convex_interpolate(seed[2*j],seed[2*j+2],t.a);
        v.a=print_convex_interpolate(seed[2*j+1],seed[2*j+3],t.a);
        if(path.family==1){
            double shape_values[10];for(int k=0;k<10;++k)shape_values[k]=state[path.shape+k];
            PrintShape10 shape;auto st=shape.initialize(data+path.bounds,4,shape_values,10);if(st!=Status::ok)return st;
            Vec2 point{};double jac[20],ds[2];st=shape.point(u.a,v.a,point,jac,ds);if(st!=Status::ok)return st;
            double du=ds[0]*u.v,dv=ds[1]*v.v;
            if(directions)for(int k=0;k<10;++k){double d=directions[size_t(path.shape+k)*q+column];du+=jac[k]*d;dv+=jac[10+k]*d;}
            u={point.x,du};v={point.y,dv};
        }
    }
    if(material)*material={u,v};
    chart=installed_print_chart(install,u,v);
    return finite_scalar(chart.x)&&finite_scalar(chart.y)?Status::ok:Status::nonfinite;
}
RS_HD inline Status shared_scene_point(const double* board,const double* layout,size_t nl,unsigned options,int code,
    double station,const double* directions,int q,int column,Vec3& xyz,Vec3& derivative){
    if(!shared_layout_size_valid(nl))return Status::invalid_argument;
    LocalJet b[6],params[8],values[ice_tolerance_layout_size];for(int i=0;i<6;++i)b[i]={board[i],directions?directions[size_t(i)*q+column]:0.};
    for(size_t i=0;i<nl;++i)values[i]={layout[i],directions?directions[(6+i)*q+column]:0.};
    int kind;auto st=shared_entity_definition(b,6,values,nl,code,kind,params,(nl>=35&&(code==410||code==411))?8:6,options,false);
    if(st!=Status::ok)return st;
    auto x=sample_planar_primitive(kind,params,LocalJet(station));
    xyz={x.x.a,x.y.a,x.z.a};derivative={x.x.v,x.y.v,x.z.v};
    return finite(x)?Status::ok:Status::nonfinite;
}
} // namespace rinkspace
