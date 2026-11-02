#include "rinkspace/local_jet.hpp"
#include "rinkspace/ice_tolerance.hpp"
#include "rinkspace/ice_relation_meta.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <utility>
#include <set>
#include <vector>
using namespace rinkspace;
using F=IceRelationFamily;
using B=IceRelationBound;
using R=IceRelationReferenceScale;
static void require(bool ok,const char* message){
    if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}
}
static void close(double a,double b,double tolerance,const char* message){
    require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance*(1+std::abs(a)+std::abs(b)),message);
}
static constexpr int duplicate_pairs[][2]={{61,72},{62,73},{97,108},{98,109},
    {133,144},{134,145},{169,180},{170,181}};
static bool expected_equivalent(int i,int j){
    if(i==j)return true;
    for(const auto& p:duplicate_pairs)if((i==p[0]&&j==p[1])||(i==p[1]&&j==p[0]))return true;
    return false;
}
static void check_legacy_order(){
    int id=0;
    auto next=[&](F family,int row,int column,int side,int index,B bound,bool strict,R scale){
        const auto m=ice_relation_meta(id++);
        require(m.family==family&&m.row==row&&m.column==column&&m.side==side&&m.index==index&&
            m.bound==bound&&m.strict==strict&&m.reference_scale==scale,"complete legacy metadata order");
    };
    auto relative=[&](F family,int row,int column,int side,int index,R scale){
        next(family,row,column,side,index,B::upper,false,scale);
        next(family,row,column,side,index,B::lower,false,scale);
    };
    // Independent enumeration of the public generator's stable visit order.
    for(int column=0;column<2;++column){
        for(int row=0;row<3;++row)relative(F::row_gap,row,column,0,row,R::nominal_pitch);
        relative(F::row_clearance,0,column,-1,-1,R::nominal_row_clearance);
        relative(F::row_clearance,3,column,1,-1,R::nominal_row_clearance);
    }
    for(int house=0;house<8;++house)for(int ring=0;ring<3;++ring)
        next(F::ring_order,house%4,house/4,0,ring,B::none,true,R::nominal_ring_outer_radius);
    for(int row=0;row<4;++row){
        next(F::lane_house_order,row,-1,0,-1,B::none,true,R::nominal_house_run);
        relative(F::lane_centre_slope,row,-1,0,-1,R::direction_unit);
        for(int side:{1,-1})next(F::lane_right_after_left,row,-1,side,-1,B::none,true,R::nominal_house_run);
        for(int side:{1,-1})next(F::lane_transverse_after_left,row,-1,side,-1,B::none,false,R::nominal_house_run);
        for(int side:{1,-1})next(F::lane_right_after_transverse,row,-1,side,-1,B::none,true,R::nominal_house_run);
        next(F::lane_centre_end_after_house,row,-1,0,-1,B::none,true,R::nominal_house_run);
        next(F::lane_centre_end_before_right,row,-1,0,-1,B::none,true,R::nominal_house_run);
        for(int side:{-1,1}){
            relative(F::lane_right_x,row,-1,side,-1,R::nominal_lane_width);
            relative(F::lane_transverse_x,row,-1,side,-1,R::nominal_lane_width);
            relative(F::lane_side_slope,row,-1,side,-1,R::direction_unit);
            relative(F::lane_width_left,row,-1,side,-1,R::nominal_lane_width);
            relative(F::lane_width_right,row,-1,side,-1,R::nominal_lane_width);
            next(F::lane_left_goal_support,row,-1,side,-1,B::none,false,R::nominal_house_run);
        }
        next(F::lane_house_goal_support,row,-1,0,-1,B::none,false,R::nominal_house_run);
        if(row<3)for(int sample=0;sample<2;++sample)
            next(F::lane_separation,row,-1,0,sample,B::none,true,R::nominal_pitch);
    }
    for(int pair=0;pair<4;++pair)for(int side:{-1,1})
        next(F::carrier_order,-1,-1,side,pair,B::none,true,R::board_half_length);
    for(int i=0;i<4;++i){
        next(F::faceoff_goal_order,-1,i<2?1:0,i==0||i==3?-1:1,i,B::none,true,R::nominal_goal_offset);
        next(F::trapezoid_board_support,-1,i/2,i%2?1:-1,i,B::none,true,R::board_half_width);
        next(F::trapezoid_goal_support,-1,i/2,i%2?1:-1,i,B::none,true,R::nominal_trapezoid_goal_span);
    }
    require(id==ice_relation_legacy_count,"metadata enumerates all 206 legacy IDs");
}
struct SourceSupportStats {
    int states=0;
    long long omitted_directions=0;
    double max_noise=0,max_noise_fraction=0;
    std::set<std::pair<int,int>> missing;
};
static void check_source_support(const double* board,
        const std::array<double,ice_tolerance_layout_size>& values,int sample,SourceSupportStats& stats){
    constexpr int sources=6+ice_tolerance_layout_size;
    std::array<std::array<bool,sources>,ice_relation_legacy_count> support{};
    std::vector<std::array<double,sources>> jacobian(ice_relation_legacy_count);
    std::array<double,ice_relation_legacy_count> margins{};
    for(int relation=0;relation<ice_relation_legacy_count;++relation)
        require(ice_relation_source_support(relation,[&](int source){
            require(source>=0&&source<sources,"source support index in board/layout range");
            support[relation][source]=true;
        }),"every legacy relation has structural source support");
    for(int source=0;source<sources;++source){
        LocalJet b[6],p[ice_tolerance_layout_size];
        for(int i=0;i<6;++i)b[i]={board[i],source==i?1.:0.};
        for(int i=0;i<ice_tolerance_layout_size;++i)p[i]={values[i],source==6+i?1.:0.};
        int visited=0;
        // Use the full original generator, independently of requested relation
        // or its declared support, so support omissions cannot suppress AD.
        ice_tolerance_margins(IceToleranceGeometry<LocalJet>(b,p),[&](int relation,LocalJet margin){
            require(relation==visited++&&std::isfinite(margin.a)&&std::isfinite(margin.v),
                "full relation source Jacobian finite and complete");
            jacobian[relation][source]=margin.v;margins[relation]=margin.a;
        });
        require(visited==ice_relation_legacy_count,"source Jacobian contains all 206 relations");
    }
    for(int relation=0;relation<ice_relation_legacy_count;++relation){
        double max_derivative=0;
        for(double derivative:jacobian[relation])max_derivative=std::max(max_derivative,std::abs(derivative));
        // All ambient sources are the native normalized arena/deviation
        // scalars. Scale roundoff by this row's value and largest UNIT-source
        // derivative. This allows cancelling row_y intermediates at epsilon
        // scale, while a missing true term must be below ~1e-14 of the row's
        // actual derivative scale to escape detection. There is no 1e-6 floor.
        const double tolerance=128*std::numeric_limits<double>::epsilon()*
            (std::abs(margins[relation])+max_derivative);
        for(int source=0;source<sources;++source){
            if(support[relation][source])continue;
            ++stats.omitted_directions;
            const double noise=std::abs(jacobian[relation][source]);
            stats.max_noise=std::max(stats.max_noise,noise);
            const double fraction=tolerance>0?noise/tolerance:(noise==0?0:std::numeric_limits<double>::infinity());
            stats.max_noise_fraction=std::max(stats.max_noise_fraction,fraction);
            if(noise>tolerance&&stats.missing.insert({relation,source}).second)
                std::fprintf(stderr,
                    "MISSING support sample=%d relation=%d source=%d (%s[%d]) derivative=%.17g margin=%.17g row_max=%.17g tolerance=%.17g\n",
                    sample,relation,source,source<6?"board":"layout",source<6?source:source-6,
                    jacobian[relation][source],margins[relation],max_derivative,tolerance);
        }
    }
    ++stats.states;
}
int main(){
    static_assert(ice_relation_legacy_count==ice_tolerance_margin_count,"legacy relation count changed");
    static_assert(ice_relation_deviation_count==ice_deviation_count,"legacy deviation count changed");
    static_assert(ice_relation_meta(0).canonical_id==206,"gap upper identity");
    static_assert(ice_relation_meta(1).canonical_id==207,"gap lower identity");
    static_assert(ice_relation_meta(72).canonical_id==61,"first width alias");
    check_legacy_order();
    for(int id:{-1,206,std::numeric_limits<int>::min(),std::numeric_limits<int>::max()}){
        require(!ice_relation_meta(id).valid()&&ice_relation_canonical_id(id)==-1,"invalid IDs fail closed");
        require(ice_relation_legacy_aliases(id,[](int){})==0,"invalid alias list empty");
    }
    require(ice_deviation_box_canonical_id(-1,B::upper)==-1&&
        ice_deviation_box_canonical_id(107,B::lower)==-1&&
        ice_deviation_box_canonical_id(0,B::none)==-1&&
        ice_deviation_box_canonical_id(0,static_cast<B>(99))==-1,"invalid box identities");
    std::set<int> canonical;int independent=0,box_aliases=0,width_aliases=0,strict=0;
    for(int d=0;d<107;++d)for(auto bound:{B::upper,B::lower})
        canonical.insert(ice_deviation_box_canonical_id(d,bound));
    for(int id=0;id<206;++id){
        const auto m=ice_relation_meta(id);
        require(m.valid()&&m.legacy_id==id,"each legacy ID retained");
        require(m.shape==ConstraintShape::nonlinear,"raw chart classification conservative");
        require(m.reference_scale!=R::invalid&&m.canonical_reference_scale!=R::invalid,"declared physical scale");
        canonical.insert(m.canonical_id);strict+=m.strict;
        if(id<6){
            ++box_aliases;
            require(m.canonical_kind==IceCanonicalKind::deviation_box&&m.canonical_index==id/2&&
                m.bound==(id%2?B::lower:B::upper)&&m.canonical_shape==ConstraintShape::affine&&
                m.canonical_reference_scale==R::deviation_unit,"gap maps to correct affine box face");
        }else{
            require(m.canonical_kind==IceCanonicalKind::relation&&m.canonical_shape==ConstraintShape::nonlinear,
                "remaining relation shapes conservative");
            if(m.canonical_id!=id)++width_aliases;
        }
        independent+=ice_relation_is_independent(id);
        std::vector<int> aliases;
        const int count=ice_relation_legacy_aliases(id,[&](int alias){
            require(expected_equivalent(id,alias),"alias is an exact specified legacy pair");
            aliases.push_back(alias);
        });
        require(count==int(aliases.size())&&count==(m.family==F::lane_width_left?2:1),"complete raw alias list");
        for(int other=0;other<206;++other)
            require((m.canonical_id==ice_relation_canonical_id(other))==expected_equivalent(id,other),
                "no approximate normal merging");
    }
    require(independent==192&&box_aliases==6&&width_aliases==8&&canonical.size()==406,
        "exactly fourteen internal duplicates removed from 206+214 faces");
    require(strict==78,"strict topology remains present");
    require(ice_relation_canonical_id(136)!=ice_relation_canonical_id(147),"different right endpoint faces retained");

    double board[]={.5,.25,.1,.12,1.85,.02};
    const double nominal[]={-.3,.2,0.,.1,.03,.1,.35,.65,.31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
        0.,.12,.06,-.24,.24,0.,.45,.3,0.,.11,0.,.035,.025,0.,0.,.05,.07,-.23,0.,0.,0.};
    std::array<double,ice_tolerance_layout_size> q{};
    std::copy(nominal,nominal+39,q.begin());
    SourceSupportStats support_stats;
    for(int sample=0;sample<4;++sample){
        auto values=q;
        for(int i=0;i<107;++i)values[39+i]=.002*std::sin(double(3*i+sample));
        values[3]*=1.+.03*sample;values[9]*=1.-.01*sample;
        values[39+50]=.002;values[39+54]=-.003;values[39+62]=.004;
        IceToleranceGeometry<double> g(board,values.data());
        check_source_support(board,values,sample,support_stats);
        std::array<double,206> margins{};int visited=0;
        ice_tolerance_margins(g,[&](int id,double value){
            require(id==visited++&&std::isfinite(value),"untouched original margins stay finite");
            margins[id]=value;
        });
        require(visited==206,"all raw margins evaluated");
        for(int id=0;id<206;++id){
            require(ice_tolerance_margin(g,id)==margins[id],"single relation preserves original legacy value");
            ice_relation_legacy_aliases(id,[&](int alias){
                require(ice_tolerance_margin(g,alias)==margins[alias],"aliases expose original legacy values");
                if(alias!=id)require(margins[alias]==margins[id],"exact width alias values");
            });
        }
        for(int id=0;id<6;++id){
            const auto m=ice_relation_meta(id);const double d=values[39+m.canonical_index];
            const double box=.01+(m.bound==B::upper?-d:d);
            close(margins[id],values[3]*box,2e-16,"gap upper/lower formula");
        }
        require(std::abs(margins[136]-margins[147])>1e-12,"tilted tapered right widths are genuinely different");
        // Exact duplicate identities must preserve directional derivatives for
        // every board/layout source, including nonzero taper and tilt.
        for(int column=0;column<6+ice_tolerance_layout_size;++column){
            LocalJet b[6],p[ice_tolerance_layout_size];
            for(int i=0;i<6;++i)b[i]={board[i],column==i?1.:0.};
            for(int i=0;i<ice_tolerance_layout_size;++i)p[i]={values[i],column==6+i?1.:0.};
            const IceToleranceGeometry<LocalJet> jg(b,p);
            for(const auto& pair:duplicate_pairs){
                const auto a=ice_tolerance_margin(jg,pair[0]),z=ice_tolerance_margin(jg,pair[1]);
                require(a.a==z.a&&a.v==z.v&&std::isfinite(a.v),"exact width alias derivatives");
            }
        }
    }
    // Deterministic random LEGAL states move every nominal dimension, all
    // board sources, and all deviations. Every seed passes the original full
    // native domain oracle before its support is examined.
    std::mt19937_64 random(0x6e32a9d417b580c1ULL);
    std::uniform_real_distribution<double> symmetric(-1.,1.);
    for(int sample=0;sample<12;++sample){
        double random_board[6];auto values=q;
        for(int i=0;i<6;++i)random_board[i]=board[i]*(1.+.004*symmetric(random));
        for(int i=0;i<39;++i)values[i]=q[i]*(1.+.004*symmetric(random));
        for(int i=0;i<107;++i)values[39+i]=.0015*symmetric(random);
        int kind=-1;double primitive[6];
        require(tolerant_shared_entity_definition(random_board,6,values.data(),values.size(),
            100,kind,primitive,6,127,true)==Status::ok,"random support seed passes full native domain oracle");
        check_source_support(random_board,values,100+sample,support_stats);
    }
    std::printf("ice source support: states=%d omitted_directions=%lld max_abs_noise=%.17g max_noise_fraction=%.17g missing_pairs=%zu\n",
        support_stats.states,support_stats.omitted_directions,support_stats.max_noise,
        support_stats.max_noise_fraction,support_stats.missing.size());
    require(support_stats.missing.empty(),"no structural source support omissions");
    for(int deviation=0;deviation<3;++deviation)for(int sign:{-1,1}){
        auto values=q;values[39+deviation]=sign*.01;
        IceToleranceGeometry<double> g(board,values.data());
        const int active=2*deviation+(sign>0?0:1);
        close(ice_tolerance_margin(g,active),0.,2e-16,"correct box face becomes active at signed boundary");
        require(ice_tolerance_margin(g,active^1)>0,"opposite box face remains interior");
        LocalJet b[6],p[ice_tolerance_layout_size];
        for(int i=0;i<6;++i)b[i]=board[i];
        for(int i=0;i<ice_tolerance_layout_size;++i)p[i]={values[i],i==39+deviation?1.:0.};
        const auto raw=ice_tolerance_margin(IceToleranceGeometry<LocalJet>(b,p),active);
        close(raw.v/q[3],-sign,2e-15,"boundary tangent has correct upper/lower orientation");
    }
    std::puts("ice relation metadata: 206 legacy IDs, exact 14 canonical removals, original aliases, signs and nonlinear classification passed; CPU only");
}
