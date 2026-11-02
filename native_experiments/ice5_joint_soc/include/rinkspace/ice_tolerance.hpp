#pragma once
#include "geometry.hpp"
#include "ice_relation_meta.hpp"

namespace rinkspace {
// Option64: layout39 nominal geometry followed by 107 dimensionless deviations.
// Every deviation is relative to the SAME current nominal parameter, never the
// previous trial. The old layouts and options retain their original meanings.
constexpr int ice_nominal_size=39,ice_deviation_count=107;
constexpr int ice_tolerance_layout_size=ice_nominal_size+ice_deviation_count;
constexpr double ice_relative_tolerance=.05;
constexpr int ice_tolerance_margin_count=206;
RS_HD inline bool shared_layout_size_valid(size_t n){return n==33||n==35||n==39||n==ice_tolerance_layout_size;}
RS_HD inline bool shared_options_valid(size_t n,unsigned options){
    if(options&64u)return n==ice_tolerance_layout_size&&options==127u;
    return (n==33||n==35||n==39)&&!(options&~63u)&&
        (!((options&4u)||(options&8u))||(options&2u))&&
        (!(options&16u)||((options&2u)&&n>=35))&&(!(options&32u)||n==39);
}

template<class T> struct IceToleranceGeometry {
    const T* q;const T* d;BoardShapeValue<T> board;
    RS_HD IceToleranceGeometry(const T* b,const T* p):q(p),d(p+39),board{b[0],b[1],b[2],b[3],b[4],b[5]}{}
    RS_HD T row_y(int row)const{
        T adjustment=0,total=0;for(int j=0;j<3;++j){total+=d[j];if(j<row)adjustment+=d[j];}
        const T margin=board.half_width-1.5*q[3];
        return (-1.5+row)*q[3]+q[3]*(adjustment-total/2)+margin*d[3];
    }
    RS_HD Vec3Value<T> house(int index)const{
        const int row=index%4;return {q[index<4?0:1]+q[3]*d[4+index],
            row_y(row)+(index<4?T(0):q[3]*d[12+row]),0};
    }
    RS_HD T radius(int house_index,int ring)const{
        const T ratios[]={q[5],q[6],q[7],T(1)};
        return q[4]*ratios[ring]*(1+d[16+4*house_index+ring]);
    }
    RS_HD T centre_slope(int row)const{const auto f=house(row),r=house(row+4);return (r.y-f.y)/(r.x-f.x);}
    RS_HD T centre_y(int row,T x)const{const auto f=house(row);return f.y+(x-f.x)*centre_slope(row);}
    RS_HD T width(int row,T x)const{
        const auto f=house(row),r=house(row+4);
        return q[9]*(1+d[48+row]+(x-f.x)*(d[52+row]-d[48+row])/(r.x-f.x));
    }
    RS_HD T side_y(int row,int sign,T x)const{return centre_y(row,x)+sign*width(row,x)/2;}
    RS_HD T side_slope(int row,int sign)const{
        const auto f=house(row),r=house(row+4);
        return centre_slope(row)+sign*q[9]*(d[52+row]-d[48+row])/(2*(r.x-f.x));
    }
    // A transverse carrier is x=x0+k*(y-row_y). Intersecting it with each
    // longitudinal side gives the SAME endpoints to both incident primitives.
    RS_HD Vec3Value<T> lane_node(int row,int sign,bool transverse)const{
        const T x0=q[transverse?35:8]+q[9]*d[(transverse?64:56)+row];
        const T k=d[(transverse?68:60)+row],y0=row_y(row),a=side_slope(row,sign);
        const T stable_x=x0+k*(side_y(row,sign,x0)-y0)/(1-k*a);
        return {stable_x,side_y(row,sign,stable_x),0};
    }
    RS_HD T carrier_x(int index)const{
        const T nominal[]={T(0),-q[19],q[19],-q[24],q[24]};
        const T scale=index==0?q[20]:(index<3?q[19]:q[24]);
        return nominal[index]+scale*d[72+index];
    }
    RS_HD T carrier_slope(int index)const{return d[77+index];}
    RS_HD T level(T x,T y)const{
        using std::abs;using std::pow;
        const T u=std::max(T(0),(abs(x)-(board.half_length-board.rx))/board.rx);
        const T v=std::max(T(0),(abs(y)-(board.half_width-board.ry))/board.ry);
        return pow(u,board.exponent)+pow(v,board.exponent)-1;
    }
    RS_HD Status carrier(int index,SegmentValue<T>& out)const{
        using std::pow;using std::abs;
        const T x0=carrier_x(index),k=carrier_slope(index);
        if(level(x0,T(0))>=0)return Status::outside_support;
        SegmentValue<T> initial{};if(board.vertical_section(x0,initial)!=Status::ok)return Status::outside_support;
        Vec3Value<T> points[2];
        for(int branch=0;branch<2;++branch){
            const double sign=branch?-1.:1.;T y=branch?initial.b.y:initial.a.y;
            for(int n=0;n<16;++n){
                const T x=x0+k*y;
                const T u=std::max(T(0),(abs(x)-(board.half_length-board.rx))/board.rx);
                const T v=std::max(T(0),(abs(y)-(board.half_width-board.ry))/board.ry);
                const T gradient=board.exponent*(pow(u,board.exponent-1)*(x<0?-1.:1.)*k/board.rx+
                    pow(v,board.exponent-1)*sign/board.ry);
                if(abs(gradient)<1e-12)return Status::degenerate;
                const T residual=level(x,y);y-=residual/gradient;
                // Always take one implicit correction, including at zero tilt,
                // so its directional derivative is present at the exact seed.
                if(abs(primal(residual))<2e-15)break;
            }
            const T x=x0+k*y;if(abs(level(x,y))>1e-11)return Status::no_intersection;
            points[branch]={x,y,0};
        }
        out={points[0],points[1]};return Status::ok;
    }
    RS_HD Vec3Value<T> midpoint(int index)const{SegmentValue<T> line{};carrier(index,line);return mul(T(.5),add(line.a,line.b));}
    RS_HD Vec3Value<T> faceoff(int i)const{
        const double sx[]={1,1,-1,-1},sy[]={-1,1,1,-1};
        return {sx[i]*q[25]+q[20]*d[82+i],sy[i]*q[27]+q[20]*d[86+i],0};
    }
    RS_HD T hockey_radius(int i)const{return q[20]*(1+d[90+i]);}
    RS_HD T crease_axis(int side,int axis)const{return q[29+axis]*(1+d[95+2*side+axis]);}
    RS_HD Vec3Value<T> trapezoid_goal(int i)const{
        using std::sqrt;const int side=i/2;const double sign=i%2?1.:-1.;
        const T span=q[33]*(1+d[99+i]),k=carrier_slope(3+side),norm=sqrt(1+k*k);
        return add(midpoint(3+side),Vec3Value<T>{sign*span*k/norm,sign*span/norm,0});
    }
    RS_HD Vec3Value<T> trapezoid_board(int i)const{
        using std::abs;using std::pow;const double sx=i<2?-1.:1.,sy=i%2?1.:-1.;
        const T y=sy*q[34]*(1+d[103+i]),ay=abs(y);
        T x=board.half_length;
        if(ay>board.half_width-board.ry){const T v=(ay-board.half_width+board.ry)/board.ry;
            x=board.half_length-board.rx+board.rx*pow(1-pow(v,board.exponent),1/board.exponent);}
        return {sx*x,y,0};
    }
};

// Stable, finite relation indices are also used by the LM physical faces and
// stopping cone. Sampling points are not parameters. All lengths are normalized
// by a declared LOCAL nominal size; directions use transverse drift / run.
// Both signs of a relative relation are separate inequalities margin>=0.
template<class T,class Visit> RS_HD inline void ice_tolerance_margins(const IceToleranceGeometry<T>& g,Visit visit,int requested=-1){
    const T* q=g.q;int id=0;
    auto positive=[&](auto compute){if(requested<0||requested==id)visit(id,compute());++id;};
    auto strict=[&](auto compute){positive([&](){return compute()-64*std::numeric_limits<double>::epsilon();});};
    auto relative=[&](auto actual,T nominal,T scale){
        if(requested<0||requested==id||requested==id+1){const T e=actual()-nominal,t=ice_relative_tolerance*scale;
            positive([&](){return t-e;});positive([&](){return t+e;});}else id+=2;};
    const T pitch=q[3],width=q[9],margin=g.board.half_width-1.5*pitch;
    for(int column=0;column<2;++column){
        if(requested>=0&&(requested<id||requested>=id+10)){id+=10;continue;}
        for(int row=0;row<3;++row)relative([&](){return g.house(column*4+row+1).y-g.house(column*4+row).y;},pitch,pitch);
        relative([&](){return g.house(column*4).y;},-1.5*pitch,margin);
        relative([&](){return g.house(column*4+3).y;},1.5*pitch,margin);
    }
    for(int i=0;i<8;++i)for(int ring=0;ring<3;++ring)strict([&](){return g.radius(i,ring+1)-g.radius(i,ring);});
    for(int row=0;row<4;++row){
        const int count=row<3?36:34;if(requested>=0&&(requested<id||requested>=id+count)){id+=count;continue;}
        const auto f=g.house(row),r=g.house(row+4);
        strict([&](){return r.x-f.x;});relative([&](){return g.centre_slope(row);},T(0),T(1));
        const auto rt=g.lane_node(row,1,false),rb=g.lane_node(row,-1,false);
        const auto ct=g.lane_node(row,1,true),cb=g.lane_node(row,-1,true);
        const T left=q[10+row],end=q[14+row];
        strict([&](){return rt.x-left;});strict([&](){return rb.x-left;});positive([&](){return ct.x-left;});positive([&](){return cb.x-left;});
        strict([&](){return rt.x-ct.x;});strict([&](){return rb.x-cb.x;});strict([&](){return end-r.x;});
        // Centreline end is its own unmeasured support, strictly before N.
        const T k=g.d[60+row],x0=q[8]+width*g.d[56+row];
        strict([&](){return x0+k*(g.centre_y(row,end)-g.row_y(row))-end;});
        for(int sign=-1;sign<=1;sign+=2){
            const auto right=sign>0?rt:rb,cross=sign>0?ct:cb;
            relative([&](){return right.x;},q[8],width);relative([&](){return cross.x;},q[35],width);
            relative([&](){return g.side_slope(row,sign);},T(0),T(1));
            relative([&](){return g.width(row,left);},width,width);relative([&](){return g.width(row,right.x);},width,width);
            // Left-open support cannot cross the actual tilted left goal.
            positive([&](){const T y=g.side_y(row,sign,left);return left-(g.carrier_x(3)+g.carrier_slope(3)*y);});
        }
        positive([&](){return f.x-(g.carrier_x(3)+g.carrier_slope(3)*f.y);});
        if(row<3){const T lo=std::min(left,q[11+row]),hi=std::max(rt.x,g.lane_node(row+1,1,false).x);
            strict([&](){return g.side_y(row+1,-1,lo)-g.side_y(row,1,lo);});
            strict([&](){return g.side_y(row+1,-1,hi)-g.side_y(row,1,hi);});}
    }
    const int order[]={3,1,0,2,4};
    for(int i=0;i<4;++i)for(int sign=-1;sign<=1;sign+=2){const T y=sign*g.board.half_width;
        strict([&](){return g.carrier_x(order[i+1])+g.carrier_slope(order[i+1])*y-g.carrier_x(order[i])-g.carrier_slope(order[i])*y;});}
    for(int i=0;i<4;++i){if(requested>=0&&(requested<id||requested>=id+3)){id+=3;continue;}const auto p=g.faceoff(i);const int side=i<2?4:3;
        strict([&](){return i<2?g.carrier_x(side)+g.carrier_slope(side)*p.y-p.x:p.x-g.carrier_x(side)-g.carrier_slope(side)*p.y;});
        const T goal_span=q[33]*(1+g.d[99+i]),board_span=q[34]*(1+g.d[103+i]);
        strict([&](){return g.board.half_width-board_span;});
        SegmentValue<T> goal{};g.carrier(3+i/2,goal);
        using std::sqrt;const auto diff=sub(goal.a,goal.b);
        strict([&](){return sqrt(diff.x*diff.x+diff.y*diff.y)/2-goal_span;});
    }
}
template<class T> RS_HD inline T ice_tolerance_margin(const IceToleranceGeometry<T>& g,int relation){
    T result=std::numeric_limits<double>::quiet_NaN();
    ice_tolerance_margins(g,[&](int,T value){result=value;},relation);return result;
}

// Ambient board/layout support of a finite relation. This follows the typed
// construction, never the derivative's value at one fitted state. Some
// cancelling intermediates deliberately retain conservative support.
template<class Include> RS_HD inline bool ice_relation_source_support(int relation,Include include){
    const auto meta=ice_relation_meta(relation);if(!meta.valid())return false;
    auto q=[&](int i){include(6+i);};auto d=[&](int i){q(ice_nominal_size+i);};
    auto row_y=[&](){include(1);q(3);for(int i=0;i<4;++i)d(i);};
    auto house_x=[&](int i){q(i<4?0:1);q(3);d(4+i);};
    auto house=[&](int i){house_x(i);row_y();if(i>=4)d(12+i%4);};
    auto centre=[&](int row){house(row);house(row+4);};
    auto width=[&](int row){house_x(row);house_x(row+4);q(9);d(48+row);d(52+row);};
    auto side=[&](int row){centre(row);width(row);};
    auto node=[&](int row,bool transverse){side(row);q(transverse?35:8);d((transverse?64:56)+row);d((transverse?68:60)+row);};
    auto carrier=[&](int index,bool endpoints){q(index==0?20:(index<3?19:24));d(72+index);d(77+index);if(endpoints)for(int i=0;i<5;++i)include(i);};
    const int row=meta.row;
    switch(meta.family){
        case IceRelationFamily::row_gap:
            q(3);d(row);if(meta.column){d(12+row);d(13+row);}break;
        case IceRelationFamily::row_clearance:
            row_y();if(meta.column)d(12+row);break;
        case IceRelationFamily::ring_order:{int house_id=4*meta.column+row;
            q(4);q(5+meta.index);if(meta.index+1<3)q(6+meta.index);d(16+4*house_id+meta.index);d(17+4*house_id+meta.index);break;}
        case IceRelationFamily::lane_house_order:house_x(row);house_x(row+4);break;
        case IceRelationFamily::lane_centre_slope:centre(row);break;
        case IceRelationFamily::lane_right_after_left:node(row,false);q(10+row);break;
        case IceRelationFamily::lane_transverse_after_left:node(row,true);q(10+row);break;
        case IceRelationFamily::lane_right_after_transverse:node(row,false);node(row,true);break;
        case IceRelationFamily::lane_centre_end_after_house:house_x(row+4);q(14+row);break;
        case IceRelationFamily::lane_centre_end_before_right:node(row,false);q(14+row);break;
        case IceRelationFamily::lane_right_x:node(row,false);break;
        case IceRelationFamily::lane_transverse_x:node(row,true);break;
        case IceRelationFamily::lane_side_slope:side(row);break;
        case IceRelationFamily::lane_width_left:width(row);q(10+row);break;
        case IceRelationFamily::lane_width_right:width(row);node(row,false);break;
        case IceRelationFamily::lane_left_goal_support:side(row);q(10+row);carrier(3,false);break;
        case IceRelationFamily::lane_house_goal_support:house(row);carrier(3,false);break;
        case IceRelationFamily::lane_separation:side(row);side(row+1);q(10+row);q(11+row);node(row,false);node(row+1,false);break;
        case IceRelationFamily::carrier_order:{const int order[]={3,1,0,2,4};include(1);carrier(order[meta.index],false);carrier(order[meta.index+1],false);break;}
        case IceRelationFamily::faceoff_goal_order:q(20);q(25);q(27);d(82+meta.index);d(86+meta.index);carrier(meta.index<2?4:3,false);break;
        case IceRelationFamily::trapezoid_board_support:include(1);q(34);d(103+meta.index);break;
        case IceRelationFamily::trapezoid_goal_support:carrier(3+meta.index/2,true);q(33);d(99+meta.index);break;
        default:return false;
    }
    return true;
}

// Structural support of the spatial primitive, valid throughout layout146.
// Global validity/relationship constraints keep their complete scene support.
// Board coordinates are included independently by the scene planner.
// This follows the same finite geometry generators below; no numeric-zero
// derivative at one fitted state is used to infer independence.
template<class Include> RS_HD inline bool ice_tolerance_layout_support(int code,Include include){
    auto deviation=[&](int index){include(ice_nominal_size+index);};
    auto row_y=[&](){include(3);for(int j=0;j<4;++j)deviation(j);};
    auto house=[&](int index){include(index<4?0:1);row_y();deviation(4+index);if(index>=4)deviation(12+index%4);};
    auto centre=[&](int row){house(row);house(row+4);};
    auto lane=[&](int row,bool transverse){centre(row);include(9);deviation(48+row);deviation(52+row);
        include(transverse?35:8);deviation((transverse?64:56)+row);deviation((transverse?68:60)+row);};
    auto carrier=[&](int index){include(index==0?20:(index<3?19:24));deviation(72+index);deviation(77+index);};
    if(code>=100&&code<132){int index=(code-100)/4,ring=(code-100)%4;house(index);include(4);if(ring<3)include(5+ring);deviation(16+4*index+ring);return true;}
    if(code>=200&&code<204){int row=code-200;centre(row);include(14+row);return true;}
    if(code>=300&&code<312){int row=(code-300)/3;lane(row,false);include(10+row);return true;}
    if(code>=320&&code<324){lane(code-320,true);return true;}
    if(code>=400&&code<=404){carrier(code-400);return true;}
    if(code==405){carrier(0);include(20);deviation(90);return true;}
    if(code>=406&&code<=409){int index=code-406;include(20);include(25);include(27);deviation(82+index);deviation(86+index);deviation(91+index);return true;}
    if(code==410||code==411){int side=code-410;carrier(3+side);include(29);include(30);deviation(95+2*side);deviation(96+2*side);return true;}
    if(code>=420&&code<=423){int index=code-420;carrier(3+index/2);include(33);include(34);deviation(99+index);deviation(103+index);return true;}
    return false;
}
template<class T> RS_HD inline Status tolerant_shared_entity_definition(const T* bp,size_t nb,const T* q,size_t nq,
        int code,int& kind,T* out,size_t no,unsigned options,bool check_relations=true){
    using std::abs;using std::atan;
    if(!bp||nb!=6||!q||nq!=ice_tolerance_layout_size||options!=127u||!out||!finite_array(q,nq))return Status::invalid_argument;
    const bool ellipse=code==410||code==411;if(no!=(ellipse?8u:6u))return Status::invalid_argument;
    IceToleranceGeometry<T> g(bp,q);
    if(!g.board.valid()||!(q[0]<q[1]&&q[3]>0&&q[4]>0&&0<q[5]&&q[5]<q[6]&&q[6]<q[7]&&q[7]<1&&q[9]>0&&
        q[19]>0&&q[20]>0&&q[24]>0&&q[25]>0&&q[27]>0&&q[29]>0&&q[30]>0&&q[33]>0&&q[34]>0&&g.board.half_width>1.5*q[3]))return Status::invalid_argument;
    if(check_relations)for(int i=0;i<ice_deviation_count;++i)if(abs(g.d[i])>ice_relative_tolerance)return Status::outside_support;
    if(check_relations)for(int index=0;index<5;++index){SegmentValue<T> line{};if(g.carrier(index,line)!=Status::ok)return Status::outside_support;}
    bool valid=true;if(check_relations)ice_tolerance_margins(g,[&](int,T value){if(!finite_scalar(value)||value < -1e-14)valid=false;});
    if(!valid)return Status::outside_support;
    auto line=[&](Vec3Value<T> a,Vec3Value<T> b){kind=0;store3(out,a);store3(out+3,b);return Status::ok;};
    auto circle=[&](Vec3Value<T> c,T radius){kind=1;store3(out,c);out[3]=radius;out[4]=0;out[5]=2*pi;return Status::ok;};
    if(code>=100&&code<132){const int i=(code-100)/4,ring=(code-100)%4;return circle(g.house(i),g.radius(i,ring));}
    if(code>=200&&code<204){const int row=code-200;return line(g.house(row),{q[14+row],g.centre_y(row,q[14+row]),0});}
    if(code>=300&&code<312){const int row=(code-300)/3,edge=(code-300)%3;
        const auto top=g.lane_node(row,1,false),bottom=g.lane_node(row,-1,false);
        if(edge==0)return line({q[10+row],g.side_y(row,1,q[10+row]),0},top);
        if(edge==1)return line(top,bottom);
        return line(bottom,{q[10+row],g.side_y(row,-1,q[10+row]),0});}
    if(code>=320&&code<324){const int row=code-320;return line(g.lane_node(row,1,true),g.lane_node(row,-1,true));}
    if(code>=400&&code<=404){SegmentValue<T> carrier{};auto status=g.carrier(code-400,carrier);if(status!=Status::ok)return status;return line(carrier.a,carrier.b);}
    if(code==405)return circle(g.midpoint(0),g.hockey_radius(0));
    if(code>=406&&code<=409)return circle(g.faceoff(code-406),g.hockey_radius(code-405));
    if(ellipse){const int side=code-410;kind=6;store3(out,g.midpoint(3+side));
        out[3]=g.crease_axis(side,0);out[4]=g.crease_axis(side,1);out[5]=-atan(g.carrier_slope(3+side));
        out[6]=side?pi/2:-pi/2;out[7]=out[6]+pi;return Status::ok;}
    if(code>=420&&code<=423)return line(g.trapezoid_goal(code-420),g.trapezoid_board(code-420));
    return Status::invalid_argument;
}
} // namespace rinkspace
