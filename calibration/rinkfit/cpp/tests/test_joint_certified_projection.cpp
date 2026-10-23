#include "rinkfit/joint_model.hpp"
#include <iostream>
#include <iomanip>

using namespace rinkfit;
EventParameterBlock block(std::string name, std::vector<double> values) {
    EventParameterBlock b; b.name=name; b.values=values;
    b.lower.assign(values.size(),-100); b.upper.assign(values.size(),100);
    b.scales.assign(values.size(),1); return b;
}
JointEventModel fixture(int kind) {
    JointEventModel m; m.scene_block=0;
    m.blocks.push_back(block("scene/rink5",{.5,.1,.1,2.,.02}));
    std::vector<double> head(15,0); head[6]=0;head[7]=-1;head[8]=.04;
    m.blocks.push_back(block("rig/main/head15",head));
    std::vector<double> optics(12,0); optics[0]=959.5;optics[1]=539.5;
    m.blocks.push_back(block("rig/main/optics12",optics));
    m.blocks.push_back(block("shot/s01/reference_angles",{0,0}));
    m.blocks.push_back(block("print_component/test", kind==6?
        std::vector<double>{.35,.015,.45,.015}:
        std::vector<double>{.4,.012,.04,0,.005}));
    JointRig r; r.id="main";r.head_block=1;r.optics_block=2;
    r.math.R0_seed={1,0,0,0,0,-1,0,1,0};
    r.math.n_seed={0,0,1};r.math.b_seed={1,0,0};
    r.math.n_tangent1={1,0,0};r.math.n_tangent2={0,1,0};
    r.math.f_ref=1000;r.math.radius_max=2;r.math.tau_bound=.02;
    m.rigs.push_back(r);m.rig_map["main"]=0;
    JointShot s; s.id="s01";s.rig_id="main";s.first_pts=0;s.last_pts=2560;s.time_origin_pts=0;
    s.reference_angles_block=3;s.math=r.math;s.math.dt=.1;s.math.first_control_time=-.2;
    s.math.n_controls=8;s.math.dependent_control=2;
    s.math.reference_indices={1,2,3,4};s.math.reference_weights={1./6,4./6,1./6,0};
    for(int i=0;i<8;++i){int j=m.blocks.size();m.blocks.push_back(block("control/"+std::to_string(i),{0,0,std::log(1000.)}));s.control_blocks.push_back(j);s.control_sources.push_back(j);s.control_offsets.push_back(0);}
    m.shots.push_back(s);m.shot_map["s01"]=0;
    EventPath p;p.id="print";p.variant_id="print";p.kind=kind;p.geometry_block=4;
    EventMaterial material;material.label="print";material.paths.push_back(p);m.materials.push_back(material);
    m.path_geometry_offsets[{0,0}]=0;m.path_geometry_indices[{0,0}]={};m.path_station_intervals[{0,0}]={{0,1}};
    return m;
}

namespace {
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
double distance(const Vec2<double>&a,const Vec2<double>&b){return std::hypot(a[0]-b[0],a[1]-b[1]);}
void configure(JointEventModel&m,int kind,double height,bool closed,bool dynamic,double tau){
  auto&p=m.materials[0].paths[0];p.kind=kind;p.closed=closed;
  auto&q=m.blocks[4].values;
  if(kind==6){q={.35,height,.45,1.1*height,.5,.8*height};if(closed)m.path_geometry_indices[{0,0}]={2,3,0,1,4,5};}
  else if(kind==7)q={.4,height,.04,.006,.2*height};
  else if(kind==0)q={-.08,0,.08,0};
  else if(kind==1)q={0,0,.08};
  else if(kind==4||kind==5)q={0,0,.08,.3};
  else if(kind==2){q={1,0,0,0,1,0};p.advertisement_block=int(m.blocks.size());m.blocks.push_back(block("ad/test",{.35,.1,.002,.018}));p.points={{0,0},{1,0},{1,1},{0,1},{0,0}};}
  else if(kind==3){q={.65};p.fixed_height_fraction=.65;p.boundary_segment=0;}
  m.blocks[2].values[11]=tau;
  if(dynamic){m.blocks[3].values={.025,-.015};m.blocks[1].values[0]=.01;
    for(int i=0;i<8;++i)m.blocks[m.shots[0].control_blocks[i]].values={.003*(i-2),-.002*(i-2),std::log(1000.)+.005*(i-2)};}
}
}
int main(int argc,char**argv){try{
  double worst=0,api_worst=0;std::size_t cases=0,width_cases=0,dynamic_cases=0;
  eventio::Json python_probes=eventio::Json::array();
  for(int kind:{0,1,2,3,4,5,6,7})for(double height:{.002,.0075,.015})for(bool closed:{false,true})for(bool dynamic:{false,true})for(double tau:{0.,.018,-.018}){
    if(closed&&kind!=6)continue;
    auto m=fixture(kind==7?7:6);configure(m,kind,height,closed,dynamic,tau);
    if(kind<6&&height!=.002)continue;
    for(int width_sign:{0,-1,1}){
      if(width_sign&&(kind!=0&&kind!=1&&kind!=4&&kind!=5))continue;
      int width=-1;if(width_sign){width=int(m.blocks.size());m.blocks.push_back(block("ice/width",{.003}));}
      for(auto pts:{std::int64_t(0),std::int64_t(1280),std::int64_t(2560)})for(double station:{0.,.07,.25,.5,.83,1.}){
        auto l=m.layout(0,0,0,pts,width,width_sign);auto ptr=m.pointers(l);auto global=m.global(l,ptr.data());
        std::vector<const double*>local;std::vector<double>geometry;auto ll=m.legacy_layout(l,local,global,ptr.data(),geometry);
        const auto world=m.world_point(l,station,local.data(),ll,m.shots[0].math);
        auto legacy=m.legacy_snapshot(0);const auto old_blocks=legacy.blocks.size(),old_materials=legacy.materials.size();
        const auto certified=m.certified_projection(legacy,l,station,539.5);
        require(certified.valid&&certified.unique,"valid fixture did not obtain a whole-interval RS certificate");
        const auto explicit_uv=m.predict_explicit(l,station,certified.row,ptr.data());
        require(explicit_uv.valid(),"same-row explicit fixture invalid");
        const double error=distance(explicit_uv.uv,certified.uv);worst=std::max(worst,error);
        require(error<1e-7,"certified projection changed the world point or camera");
        require(certified.equality_px<1e-7,"RS row is not its own projection");
        require(legacy.blocks.size()==old_blocks&&legacy.materials.size()==old_materials,"read-only certification added fabricated geometry");
        legacy.materials.clear();const auto direct=legacy.project_world_rs(world,l.frame_time,539.5);
        require(direct.valid&&direct.unique,"world-point API depends on a material placeholder");
        api_worst=std::max(api_worst,distance(direct.uv,certified.uv));
        if((kind==6||kind==7)&&!closed&&pts==1280){
          eventio::Json b=eventio::Json::array();for(const auto&x:legacy.blocks)b.push_back({{"name",x.name},{"values",x.values}});
          eventio::Json camera={{"R0_seed",legacy.R0_seed},{"n_seed",legacy.n_seed},{"b_seed",legacy.b_seed},{"n_tangent1",legacy.n_tangent1},{"n_tangent2",legacy.n_tangent2},{"f_ref",legacy.f_ref},{"radius_max",legacy.radius_max},{"dt",legacy.dt},{"first_control_time",legacy.first_control_time},{"n_controls",legacy.n_controls},{"control_block_begin",legacy.control_block_begin},{"dependent_control",legacy.dependent_control},{"reference_indices",legacy.reference_indices},{"reference_weights",legacy.reference_weights}};
          eventio::Json model={{"blocks",b},{"camera",camera},{"materials",eventio::Json::array({{{"label","print"},{"paths",eventio::Json::array({{{"kind",kind},{"geometry_block",4}}})}}})}};
          python_probes.push_back({{"event_model",model},{"dynamic",dynamic},{"tau",tau},{"station",station},{"frame_time",l.frame_time},{"world",world},{"uv",certified.uv}});
        }
        ++cases;if(width_sign)++width_cases;if(dynamic&&tau!=0)++dynamic_cases;
      }
    }
  }
  std::size_t real_cases=0,real_valid=0,real_unknown=0;double real_worst=0;
  if(argc>1){auto m=JointEventModel::load(argv[1]);for(std::size_t sh=0;sh<m.shots.size();++sh){auto legacy=m.legacy_snapshot(int(sh));const auto&shot=m.shots[sh];
    for(const auto&label:{"Ad2.left_top","ice.C0:outer","board.top","board.contact"}){
      auto mat=std::find_if(m.materials.begin(),m.materials.end(),[&](const auto&x){return x.label==label;});if(mat==m.materials.end())continue;
      int mi=int(mat-m.materials.begin());for(std::size_t pi=0;pi<mat->paths.size();++pi)for(auto pts:{shot.first_pts,shot.first_pts+((shot.last_pts-shot.first_pts)/512)*256,shot.last_pts})for(double station:{0.,.31,1.}){
        if(!m.station_allowed(mi,int(pi),station))continue;
        auto l=m.layout(int(sh),mi,int(pi),pts);auto old=legacy.project_rs(mi,int(pi),station,l.frame_time);auto current=m.certified_projection(legacy,l,station);
        require(old.valid==current.valid&&old.unique==current.unique&&old.roots==current.roots&&old.reason==current.reason,"existing native projection status changed");
        if(finite_array(old.uv)&&finite_array(current.uv)){const double e=distance(old.uv,current.uv);real_worst=std::max(real_worst,e);require(e<1e-7,"existing native material projection changed");}
        ++real_cases;(current.valid&&current.unique?real_valid:real_unknown)++;
      }
    }
  }}
  const auto invalid=fixture(6).legacy_snapshot(0).project_world_rs({NAN,0,0},0.);require(!invalid.valid&&!invalid.unique&&invalid.reason=="nonfinite_world_point","nonfinite XYZ was certified");
  eventio::Json result={{"passed",true},{"fixture_cases",cases},{"nonzero_RS_dynamic_cases",dynamic_cases},{"width_interface_cases",width_cases},{"max_explicit_same_row_difference_px",worst},{"max_direct_XYZ_difference_px",api_worst},{"actual_model_probes",real_cases},{"actual_valid_unique",real_valid},{"actual_unknown_or_invalid_preserved",real_unknown},{"actual_model_old_vs_new_max_difference_px",real_worst},{"independent_root_and_full_interval_rules_unchanged",true},{"model_packet_CP_modified",false},{"fit_started",false},{"python_probes",python_probes}};
  if(argc>2)eventio::atomic_json(argv[2],result);std::cout<<"fixture_cases="<<cases<<" dynamic_nonzero_RS="<<dynamic_cases<<" width_cases="<<width_cases<<" max_difference_px="<<std::setprecision(15)<<worst<<" actual_cases="<<real_cases<<" actual_max_difference_px="<<real_worst<<" passed\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
