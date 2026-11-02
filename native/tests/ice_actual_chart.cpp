#include "rinkspace/scene_derivatives.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace rinkspace;
constexpr std::size_t ambient_size=6+ice_tolerance_layout_size;
using Ambient=std::array<double,ambient_size>;
using Charts=std::array<int,ambient_size>;
using Direction=std::array<double,ambient_size>;
static int failures=0;
static double worst_fd_error=0.;
static bool check(bool ok,const char* message,const char* fixture,int direction=-1,int code=-1,int component=-1){
    if(!ok){++failures;std::fprintf(stderr,"FAIL %s fixture=%s direction=%d code=%d component=%d\n",
        message,fixture,direction,code,component);}
    return ok;
}
static std::vector<int> primitive_codes(){
    std::vector<int> result;
    for(int i=100;i<132;++i)result.push_back(i);
    for(int i=200;i<204;++i)result.push_back(i);
    for(int i=300;i<312;++i)result.push_back(i);
    for(int i=320;i<324;++i)result.push_back(i);
    for(int i=400;i<412;++i)result.push_back(i);
    for(int i=420;i<424;++i)result.push_back(i);
    return result;
}
static Charts actual_charts(){
    Charts chart{};
    for(int i:{1,2,3,5})chart[i]=1;
    chart[4]=2;
    // Existing positive and ordered layout charts, before the six nominal
    // coordinates are replaced by non-independent witness storage.
    for(int i:{3,4,9,19,20,24,25,27,29,30,33,34})chart[6+i]=1;
    chart[11]=3;chart[12]=4;chart[13]=5;
    chart[6]=chart[7]=ice_chart_witness_x;
    for(int i=10;i<=13;++i)chart[i]=ice_chart_witness_radius;
    for(int i=49;i<57;++i)chart[i]=ice_chart_actual_x;
    for(int i=61;i<93;++i)chart[i]=ice_chart_actual_log_radius;
    return chart;
}
static Ambient interior_fixture(){
    Ambient a{};
    const double board[]={.5,.25,.1,.12,1.85,.02};
    const double nominal[]={-.3,.2,0.,.1,.03,.1,.35,.65,.31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
        0.,.12,.06,-.24,.24,0.,.45,.3,0.,.11,0.,.035,.025,0.,0.,.05,.07,-.23,0.,0.,0.};
    for(int i=0;i<6;++i)a[i]=board[i];
    for(int i=0;i<39;++i)a[6+i]=nominal[i];
    for(int i=0;i<107;++i)a[45+i]=.0004*std::sin(.73*i+.2);
    // Retain the established geometry fixture, with genuinely unequal width
    // deviations at F and R and nonzero right/transverse/carrier tilts.
    for(int row=0;row<4;++row){
        a[45+52+row]=a[45+48+row]+(row%2?-.00012:.00012);
        a[45+60+row]=.0002*(row-1.5);
        a[45+68+row]=.00015*(1.5-row);
    }
    for(int i=0;i<5;++i)a[45+77+i]=.0003*std::sin(i+.7);
    return a;
}
static Ambient boundary_fixture(){
    auto a=interior_fixture();
    for(int i=0;i<107;++i)a[45+i]=0.;
    a[6]=-.3125;a[7]=.25;a[9]=.125;
    a[10]=.03125;a[11]=.125;a[12]=.375;a[13]=.625;
    for(int i=0;i<8;++i)a[49+i]=i%2?.01:-.01;
    for(int house=0;house<8;++house)for(int layer=0;layer<4;++layer)
        a[61+house*4+layer]=house%2?.01:-.01;
    for(int row=0;row<4;++row){a[45+48+row]=-.0002;a[45+52+row]=.0002;a[45+60+row]=.0001;}
    return a;
}
static Ambient latest_fixture(){
    // Board/layout only, copied bit-exactly from the current complete seed:
    // D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/solver_speed_validation/
    // pro_20261101/constrained_progress_1101/seed_latest.npz
    // SHA256 aaaa9abec30b92746d4127766ea87b71f12d7ea0a5df28116376bb32ca7819c5.
    // This is distinct from the public native_joint9_guarded_lines snapshot.
    return Ambient{{
        0x1.0000000000000p-1,0x1.00b2db73feb44p-2,0x1.1ebdb329ca231p-3,0x1.23c1703e6f93cp-3,
        0x1.f86a4f8b49a24p+0,0x1.179a3370dfe28p-6,-0x1.58deef7568638p-2,0x1.c7943e1deef4ap-3,
        -0x1.57ed149423b46p-3,0x1.db4289f9d644bp-4,0x1.01ff7c897f591p-5,0x1.f3d1ad8295938p-4,
        0x1.6453ffb858154p-2,0x1.523c858e99022p-1,0x1.428a7e857d6a6p-2,0x1.4aa3325590956p-4,
        -0x1.b7742868df27fp-2,-0x1.a472423cd5378p-2,-0x1.b70eea2193184p-2,-0x1.b6bd02815812cp-2,
        0x1.3b948995f5ffap-2,0x1.2b4bcb0b05a01p-2,0x1.2f5082869464ep-2,0x1.27fcbb57b1b56p-2,
        0x0.0p+0,0x1.e086e63dbde54p-4,0x1.31057811ee4bbp-4,-0x1.d045c94fdcf93p-3,
        0x1.f44c705131bdcp-3,0x0.0p+0,0x1.ba430ce809df8p-2,0x1.540b25a9415a0p-2,
        0x0.0p+0,0x1.db67b830c321fp-4,0x0.0p+0,0x1.0ca636b886545p-5,
        0x1.e11598e28ffa0p-6,0x0.0p+0,0x0.0p+0,0x1.b67ba73a753c0p-5,
        0x1.1ec6cafa8b2cep-4,-0x1.d9ef077a0c149p-3,-0x1.d6a6c4757080cp-3,0x0.0p+0,
        0x0.0p+0,0x1.373e283c93f47p-9,-0x1.479bc176931d1p-7,0x1.47ae147ad147bp-7,
        -0x1.0cb8dc2b1473cp-7,0x1.47ae147ad147bp-7,-0x1.47ae147ad147bp-7,-0x1.2490b8f730d16p-7,
        0x1.98fb8f17dbf8ep-9,-0x1.47ae147ad1462p-7,0x1.47ae147ad147bp-7,0x1.6bb9acaa4710fp-8,
        0x1.daf593b34a024p-11,0x1.373e283aac573p-9,-0x1.47ae147ad147bp-7,0x1.cdeea09c24f13p-8,
        0x1.cdeea09b2fc46p-8,0x1.476748c07727cp-7,0x1.47ae147ad147bp-7,0x1.47ae147ad147ap-7,
        0x1.47ae147ad147bp-7,0x1.477c254c125f8p-7,0x1.47ac1991054ebp-7,-0x1.87138acc0abfep-8,
        -0x1.47ae147ad147bp-7,-0x1.47ae147ad147bp-7,-0x1.47ade0753a4c4p-7,-0x1.d36f0f6e6e5d9p-9,
        -0x1.f4f7658f8164ep-9,0x1.47ae147ad147bp-7,0x1.47ae147ad147bp-7,0x1.47a6a899d7dd3p-7,
        0x1.47a16a9bbf471p-7,0x0.0p+0,0x0.0p+0,0x0.0p+0,
        0x0.0p+0,-0x1.47ae147ad147fp-7,-0x1.47ac6334b163bp-7,0x1.5ea6b898f0605p-9,
        -0x1.2829c9ee47252p-9,-0x1.47ae147ad147ep-7,-0x1.47ae147ad147bp-7,-0x1.47ae147ad147bp-7,
        -0x1.0123af2ba1fa9p-7,-0x1.47ae147ad147dp-7,-0x1.1735730106551p-7,-0x1.1300e22be39c3p-8,
        -0x1.417fd31dd0a09p-7,-0x1.ea57e9728104cp-8,-0x1.06248e167b0cfp-7,-0x1.47a1dbb0c2c79p-7,
        -0x1.5d10813014895p-8,0x1.eb9756f256b1ap-8,0x1.e8f59edc203e9p-8,-0x1.47a30de8d0e29p-7,
        -0x1.715b6d90fddf8p-10,-0x1.47ae1477fd9b2p-7,0x1.f4eefcb8b9951p-8,0x1.529f86aff6394p-8,
        -0x1.2447116d42956p-9,-0x1.09f37b04ce000p-37,0x1.3c467bd749843p-8,-0x1.47ae147ad147bp-7,
        -0x1.47a9f2fbb2c5dp-7,-0x1.48cea27c68b66p-8,0x1.3cd2c8136cc25p-8,0x1.ab8a05719b347p-11,
        0x1.479d2d61f47bcp-7,-0x1.47ad75c3c228dp-7,-0x1.47a93822c4eeap-7,0x1.47ae147ad147bp-7,
        -0x1.10bd74bbc9872p-18,-0x1.47ae147ad147bp-7,-0x1.470e297ea6cf4p-7,0x1.c819be66e7fe0p-11,
        0x1.cb63a56cf5ec9p-9,0x1.d1804a65bdf32p-10,0x1.131082f81dda7p-8,0x1.85052c005125dp-9,
        -0x1.1b10305788aadp-8,0x1.30aca5d7af2acp-10,-0x1.a650a1aa5fa61p-8,0x1.47ae147ad147bp-7,
        -0x1.09a95ef107bcep-7,0x1.47ae147ad147bp-7,0x1.3ca46a4e6409dp-9,0x1.2257f5c88794bp-7,
        0x1.47ae147ad147bp-7,-0x1.47ae147ad147bp-7,-0x1.47accfd3da875p-7,-0x1.47682a0608170p-7,
        0x1.47ad3387ee53dp-7,-0x1.47ae147ad147bp-7,0x1.47ae147ad147bp-7,0x1.47ae147ad147bp-7,
        0x0.0p+0,0x0.0p+0,-0x1.5a5f8a0781293p-8,-0x1.0c3c224b18f3dp-7,
        0x0.0p+0,0x0.0p+0,0x1.31ddc753f8eabp-8,0x0.0p+0,
        0x0.0p+0,0x0.0p+0,-0x1.fa1aa246034f7p-8,0x0.0p+0
    }};
}
static bool legacy_domain(const Ambient& a,const char* label,int direction=-1){
    if(!check(finite_array(a.data(),a.size()),"finite ambient",label,direction))return false;
    bool valid=true;
    for(int i=0;i<107;++i)
        valid&=check(a[45+i]>=-.01&&a[45+i]<=.01,"strict inclusive legacy box",label,direction,-1,i);
    int count=0;
    IceToleranceGeometry<double> g(a.data(),a.data()+6);
    ice_tolerance_margins(g,[&](int id,double value){
        valid&=check(id==count++,"stable 206 relation IDs",label,direction,-1,id);
        // Use the existing native acceptance oracle's exact comparison.
        valid&=check(std::isfinite(value)&&value>=-1e-14,"native 206 relation domain",label,direction,-1,id);
    });
    valid&=check(count==206,"all original relations enumerated",label,direction);
    for(int code:primitive_codes()){
        int kind;double out[8];const int n=code==410||code==411?8:6;
        valid&=check(shared_entity_definition(a.data(),6,a.data()+6,146,code,kind,out,n,127,true)==Status::ok,
                     "full original primitive oracle",label,direction,code);
    }
    return valid;
}
static bool retract(const Ambient& a,const Charts& chart,const Direction& direction,double step,
                    Ambient& out,const char* label,int column){
    for(std::size_t i=0;i<a.size();++i){
        out[i]=scene_chart_retract_delta(a.data(),chart.data(),i,[&](std::size_t row){return step*direction[row];});
        if(!check(std::isfinite(out[i]),"finite chart retraction",label,column,-1,int(i)))return false;
    }
    return true;
}
static void close_target(double actual,double target,const char* label,int direction,int component){
    const double scale=std::max(std::abs(actual),std::abs(target));
    const double bound=16.*std::numeric_limits<double>::epsilon()*scale+4.*std::numeric_limits<double>::denorm_min();
    check(std::isfinite(actual)&&std::isfinite(target)&&std::abs(actual-target)<=bound,
          "actual target equals exported legacy geometry within FP64 roundoff",label,direction,-1,component);
}
static void targets(const Ambient& a,const Ambient& retracted,const Direction& direction,double step,
                    const char* label,int column){
    IceToleranceGeometry<double> before(a.data(),a.data()+6),after(retracted.data(),retracted.data()+6);
    close_target(retracted[9],a[9]*std::exp(step*direction[9]),label,column,9);
    for(int house=0;house<8;++house){
        close_target(after.house(house).x,before.house(house).x+step*direction[49+house],label,column,49+house);
        for(int layer=0;layer<4;++layer){
            close_target(after.radius(house,layer),before.radius(house,layer)*std::exp(step*direction[61+house*4+layer]),
                         label,column,61+house*4+layer);
            int kind;double primitive[6];const int code=100+4*house+layer;
            const auto status=shared_entity_definition(retracted.data(),6,retracted.data()+6,146,code,kind,primitive,6,127,false);
            check(status==Status::ok&&kind==1,"exported circle primitive",label,column,code);
            close_target(primitive[0],after.house(house).x,label,column,49+house);
            close_target(primitive[3],after.radius(house,layer),label,column,61+house*4+layer);
        }
    }
}
static void zero_step(const Ambient& a,const Charts& chart,const char* label){
    Direction zero{};Ambient out{};
    if(retract(a,chart,zero,0.,out,label,-1)){
        for(std::size_t i=0;i<a.size();++i)
            check(std::memcmp(&a[i],&out[i],sizeof(double))==0,"zero tangent preserves ambient bits",label,-1,-1,int(i));
        legacy_domain(out,label);
    }
    for(std::size_t i=0;i<a.size();++i)
        check(scene_chart_component(a.data(),chart.data(),zero.data(),1,0,i)==0.,"zero pullback direction",label,-1,-1,int(i));
}
static void separation_point(const Ambient& accepted){
    const char* label="out_of_group_separation_point";
    Direction delta{};delta[49]=.006;delta[61]=std::log(1.05);
    const auto delta_at=[&](std::size_t row){return delta[row];};
    IceActualWitness bounded{},temporary{};
    check(!ice_actual_chart_retract(accepted.data(),delta_at,bounded),
          "default retraction requires a bounded witness",label);
    if(!check(ice_actual_chart_retract(accepted.data(),delta_at,temporary,true),
              "constraint trial represents an unbounded separation point",label))return;
    IceToleranceGeometry<double> before(accepted.data(),accepted.data()+6);
    std::array<double,8> actual_x{};std::array<double,32> actual_radius{};
    bool x_box_violation=false,radius_box_violation=false;
    for(int house=0;house<8;++house){
        actual_x[house]=before.house(house).x+delta[49+house];
        close_target(temporary.c[house/4]+accepted[9]*temporary.x_deviation[house],actual_x[house],label,0,49+house);
        x_box_violation|=std::abs(temporary.x_deviation[house])>.01;
        for(int layer=0;layer<4;++layer){
            const int index=4*house+layer;
            actual_radius[index]=before.radius(house,layer)*std::exp(delta[61+index]);
            close_target(temporary.radius[layer]*(1.+temporary.radius_deviation[index]),actual_radius[index],label,0,61+index);
            radius_box_violation|=std::abs(temporary.radius_deviation[index])>.01;
        }
    }
    check(x_box_violation&&radius_box_violation,"temporary X and radius deviations explicitly exceed legacy boxes",label);
    check(!ice_actual_witness(actual_x.data(),actual_radius.data(),accepted[9],bounded),
          "separation geometry is not a legal bounded witness",label);
    Ambient exported=accepted;
    for(std::size_t i=0;i<exported.size();++i)
        if(ice_actual_component(i))exported[i]=ice_actual_export_component(temporary,i);
    for(int code:{100,104}){
        int kind;double primitive[6];const int house=(code-100)/4;
        check(shared_entity_definition(exported.data(),6,exported.data()+6,146,code,kind,primitive,6,127,false)==Status::ok,
              "unchecked separation primitive remains evaluable",label,0,code);
        close_target(primitive[0],actual_x[house],label,0,49+house);
        close_target(primitive[3],actual_radius[4*house],label,0,61+4*house);
        check(shared_entity_definition(exported.data(),6,exported.data()+6,146,code,kind,primitive,6,127,true)==Status::outside_support,
              "unchanged native acceptance rejects temporary out-of-box state",label,0,code);
    }
}
static void finite_differences(const Ambient& a,const Charts& chart,const std::vector<Direction>& directions,
                               const char* label,double h=2e-6){
    const int q=int(directions.size());std::vector<double> matrix(a.size()*q);
    for(std::size_t row=0;row<a.size();++row)for(int col=0;col<q;++col)matrix[row*q+col]=directions[col][row];
    for(int column=0;column<q;++column){
        Ambient plus{},minus{};
        if(!retract(a,chart,directions[column],h,plus,label,column)||
           !retract(a,chart,directions[column],-h,minus,label,column))continue;
        legacy_domain(plus,label,column);legacy_domain(minus,label,column);
        targets(a,plus,directions[column],h,label,column);targets(a,minus,directions[column],-h,label,column);
        std::array<LocalJet,ambient_size> jet{};
        for(std::size_t row=0;row<a.size();++row)
            jet[row]={a[row],scene_chart_component(a.data(),chart.data(),matrix.data(),q,column,row)};
        for(int code:primitive_codes()){
            double lo[8],hi[8];LocalJet analytic[8];int k0,k1,kj;const int n=code==410||code==411?8:6;
            const auto sj=shared_entity_definition(jet.data(),6,jet.data()+6,146,code,kj,analytic,n,127,false);
            const auto sl=shared_entity_definition(minus.data(),6,minus.data()+6,146,code,k0,lo,n,127,false);
            const auto sh=shared_entity_definition(plus.data(),6,plus.data()+6,146,code,k1,hi,n,127,false);
            if(!check(sj==Status::ok&&sl==Status::ok&&sh==Status::ok&&k0==k1&&k0==kj,
                      "matching primitive branches for AD/FD",label,column,code))continue;
            for(int component=0;component<n;++component){
                const double fd=(hi[component]-lo[component])/(2*h),ad=analytic[component].v;
                const double error=std::abs(fd-ad),scale=1.+std::abs(fd)+std::abs(ad);
                if(error/scale>worst_fd_error)worst_fd_error=error/scale;
                if(!check(std::isfinite(ad)&&std::isfinite(fd)&&error<=5e-9+5e-7*std::max(std::abs(fd),std::abs(ad)),
                          "primitive central FD matches chart pullback LocalJet",label,column,code,component))
                    std::fprintf(stderr,"  AD=%.17g FD=%.17g error=%.17g h=%.17g\n",ad,fd,error,h);
            }
        }
    }
}
int main(){
    const auto chart=actual_charts();const auto interior=interior_fixture();
    check(primitive_codes().size()==68,"all 68 primitive identities","setup");
    int outputs=0;for(int code:primitive_codes())outputs+=code==410||code==411?8:6;
    check(outputs==412,"all 412 primitive outputs","setup");
    check(ice_actual_chart_pattern(chart.data(),chart.size()),"finite actual chart pattern","setup");
    legacy_domain(interior,"nonzero_taper_tilt");zero_step(interior,chart,"nonzero_taper_tilt");
    std::vector<Direction> directions;
    for(int i=49;i<57;++i){Direction d{};d[i]=.02;directions.push_back(d);}
    for(int i=61;i<93;++i){Direction d{};d[i]=.2;directions.push_back(d);}
    Direction pitch{};pitch[9]=.1;directions.push_back(pitch);
    for(int sample=0;sample<8;++sample){
        Direction d{};
        for(std::size_t i=0;i<d.size();++i){
            d[i]=.0003*std::sin(.7*i+sample);
            if(chart[i]==1||chart[i]==2)d[i]=.003*std::cos(.4*i+sample);
            if(chart[i]==ice_chart_actual_x)d[i]=.01*std::sin(.8*i+sample);
            if(chart[i]==ice_chart_actual_log_radius)d[i]=.1*std::cos(.6*i+sample);
            if(chart[i]==ice_chart_witness_x||chart[i]==ice_chart_witness_radius)d[i]=0.;
        }
        d[9]=.03*std::cos(double(sample));directions.push_back(d);
    }
    finite_differences(interior,chart,directions,"nonzero_taper_tilt");
    separation_point(interior);

    const auto boundary=boundary_fixture();legacy_domain(boundary,"inclusive_boundaries");
    zero_step(boundary,chart,"inclusive_boundaries");
    Direction inward{};
    for(int i=0;i<8;++i)inward[49+i]=-boundary[9]*boundary[49+i];
    for(int i=0;i<32;++i)inward[61+i]=boundary[61+i]>0.?-.005:.005;
    inward[9]=.02;Ambient boundary_inside{};
    if(retract(boundary,chart,inward,1e-6,boundary_inside,"inclusive_boundaries",0)){
        legacy_domain(boundary_inside,"inclusive_boundaries");targets(boundary,boundary_inside,inward,1e-6,"inclusive_boundaries",0);
    }

    const auto latest=latest_fixture();legacy_domain(latest,"latest_complete_seed");
    zero_step(latest,chart,"latest_complete_seed");
    IceToleranceGeometry<double> real(latest.data(),latest.data()+6);
    for(int layer=0;layer<4;++layer){
        double low=0.,high=std::numeric_limits<double>::max();
        for(int house=0;house<8;++house){low=std::max(low,real.radius(house,layer)/1.01);high=std::min(high,real.radius(house,layer)/.99);}
        check(low<=high&&high-low<1e-12,"latest seed retains narrow nonempty witness intervals","latest_complete_seed",-1,-1,layer);
    }
    std::vector<Direction> common_radius;
    for(int layer=0;layer<4;++layer){Direction d{};for(int house=0;house<8;++house)d[61+4*house+layer]=.02;common_radius.push_back(d);}
    finite_differences(latest,chart,common_radius,"latest_complete_seed",2e-5);
    std::printf("ice actual chart: %d failures; 68 primitives / 412 outputs, %zu interior and %zu narrow-seed directions; worst normalized FD error %.17g; no optimizer called\n",
                failures,directions.size(),common_radius.size(),worst_fd_error);
    return failures?1:0;
}
