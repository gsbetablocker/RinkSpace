#include "rinkfit/event_model.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>

namespace rinkfit {
EventModel EventModel::load(const std::string& path) {
  std::ifstream f(path);if(!f)throw std::runtime_error("cannot open event model: "+path);
  nlohmann::json j;f>>j;EventModel m;
  if(j.at("schema")!="rinkfit.event_model.v1.1")throw std::runtime_error("event model schema mismatch");
  m.model_id=j.at("model_id");m.source_identity=j.at("source_identity");
  for(const auto& b:j.at("blocks")){EventParameterBlock p;p.name=b.at("name");p.values=b.at("values").get<std::vector<double>>();
    p.lower=b.at("lower").get<std::vector<double>>();p.upper=b.at("upper").get<std::vector<double>>();p.scales=b.at("scales").get<std::vector<double>>();p.constant=b.value("constant",false);
    p.fixed_indices=b.value("fixed_indices",std::vector<int>{});
    if(p.values.empty()||p.lower.size()!=p.values.size()||p.upper.size()!=p.values.size()||p.scales.size()!=p.values.size())throw std::runtime_error("invalid event block shape");
    for(std::size_t i=0;i<p.values.size();++i)if(!std::isfinite(p.values[i])||!(p.scales[i]>0)||p.values[i]<p.lower[i]||p.values[i]>p.upper[i])throw std::runtime_error("invalid event parameter domain "+p.name);
    m.blocks.push_back(std::move(p));}
  if(m.blocks.empty()||m.blocks[0].values.size()!=kEventGlobalSize)throw std::runtime_error("global32 block absent");
  const auto& c=j.at("camera");m.R0_seed=c.at("R0_seed").get<Mat3<double>>();m.n_seed=c.at("n_seed").get<Vec3<double>>();m.b_seed=c.at("b_seed").get<Vec3<double>>();
  m.n_tangent1=c.at("n_tangent1").get<Vec3<double>>();m.n_tangent2=c.at("n_tangent2").get<Vec3<double>>();m.f_ref=c.at("f_ref");m.radius_max=c.at("radius_max");
  m.dt=c.at("dt");m.first_control_time=c.at("first_control_time");m.n_controls=c.at("n_controls");m.control_block_begin=c.at("control_block_begin");m.dependent_control=c.at("dependent_control");
  m.reference_indices=c.at("reference_indices").get<std::array<int,4>>();m.reference_weights=c.at("reference_weights").get<std::array<double,4>>();m.tau_bound=c.at("tau_bound");
  if(!valid_rotation(m.R0_seed)||std::abs(dot(m.n_seed,m.n_seed)-1)>1e-9||!(m.f_ref>0)||!(m.radius_max>0)||!(m.dt>0)||m.n_controls<4)throw std::runtime_error("invalid event camera seed");
  for(const auto& a:j.at("materials")){EventMaterial mat;mat.label=a.at("label");
    for(const auto& p:a.at("paths")){EventPath path;path.id=p.at("id");path.variant_id=p.at("variant_id");path.kind=p.at("kind");path.geometry_block=p.value("geometry_block",-1);path.advertisement_block=p.value("advertisement_block",-1);
      path.physical_point=p.value("physical_point",false);path.closed=p.value("closed",false);path.provenance=p.value("provenance","");path.points=p.value("points",std::vector<Vec2<double>>{});
      path.start_angle=p.value("start_angle",0.);path.end_angle=p.value("end_angle",6.283185307179586);path.fixed_height_fraction=p.value("fixed_height_fraction",0.);
      path.radius_index=p.value("radius_index",2);
      path.boundary_segment=p.value("boundary_segment",-1);
      path.shared_station_block=p.value("shared_station_block",-1);
      for(int id:{path.geometry_block,path.advertisement_block})if(id>=int(m.blocks.size()))throw std::runtime_error("path references missing block");
      if(path.shared_station_block>=0){
        if(path.shared_station_block>=int(m.blocks.size())||!path.physical_point||path.kind!=0)throw std::runtime_error("shared station requires a physical point on an ice segment");
        const auto& b=m.blocks[path.shared_station_block];
        if(b.values.size()!=1||b.lower[0]<0||b.upper[0]>1)throw std::runtime_error("shared station requires a scalar finite [0,1] block");
      }
      if(path.kind==2&&(path.advertisement_block<0||path.points.empty()))throw std::runtime_error("board path lacks shared installation");mat.paths.push_back(std::move(path));}
    if(mat.paths.empty())throw std::runtime_error("material has no finite path");m.materials.push_back(std::move(mat));}
  return m;
}

EventRowLayout EventModel::layout(int material,int path,double time) const {
  const auto& p=materials.at(material).paths.at(path);EventRowLayout l;l.material=material;l.path=path;l.frame_time=time;l.block_ids={0};
  if(p.geometry_block>=0){l.geometry_slot=int(l.block_ids.size());l.block_ids.push_back(p.geometry_block);}
  if(p.advertisement_block>=0){l.advertisement_slot=int(l.block_ids.size());l.block_ids.push_back(p.advertisement_block);}
  if(p.shared_station_block>=0){l.shared_station_slot=int(l.block_ids.size());l.block_ids.push_back(p.shared_station_block);}
  const int lo=int(std::floor((time-tau_bound/2-first_control_time)/dt))-1;
  const int hi=int(std::floor((time+tau_bound/2-first_control_time)/dt))+2;
  std::set<int> ids;for(int i=lo;i<=hi;++i)if(i>=0&&i<n_controls)ids.insert(i);
  if(ids.count(dependent_control))for(int k=0;k<4;++k)if(reference_weights[k]!=0&&reference_indices[k]!=dependent_control)ids.insert(reference_indices[k]);
  for(int i:ids){l.control_ids.push_back(i);l.control_slots.push_back(int(l.block_ids.size()));l.block_ids.push_back(control_block_begin+i);}
  return l;
}
std::vector<const double*> EventModel::pointers(const EventRowLayout& l) const {
  std::vector<const double*> out;for(int id:l.block_ids)out.push_back(blocks.at(id).values.data());return out;
}
double EventModel::corner_length() const {return quarter_arclength(1.5707963267948966,blocks[0].values.data());}

EventRootResult EventModel::project_rs(int material,int path,double station,double time,double initial) const {
  EventRootResult out;const auto l=layout(material,path,time);const auto ptrs=pointers(l);
  Vec3<double> fixed_world;try{fixed_world=world_point(materials[material].paths[path],station,ptrs.data(),l);}catch(...){out.reason="finite_material_domain_failure";return out;}
  const auto fixed_head=head(blocks[0].values.data());const auto fixed_optics=optics(blocks[0].values.data());
  auto eval=[&](double v)->Projection<double>{try{const auto tr=trajectory(row_time(time,blocks[0].values[31],v),ptrs.data(),l);const auto oo=zoom_optics(fixed_optics,tr[2]);
      return project(fixed_world,camera_pose(fixed_head,tr[0],tr[1],oo.z),oo.intr,oo.brown);}catch(...){return {};}};
  if(blocks[0].values[31]==0){const auto p=eval(539.5);out.uv=p.uv;out.depth=p.depth;out.row=p.uv[1];out.equality_px=0;out.implicit_denominator=1;
    out.valid=p.valid()&&p.uv[1]>=0&&p.uv[1]<=1079;out.unique=out.valid;out.roots=out.valid?1:0;out.sampled_monotonic=out.valid;
    out.reason=out.valid?"unique_exact_global_shutter_root":"global_shutter_domain_or_row_failure";return out;}
  // A complete monotonic interval certificate is provided by the independent
  // native replay. This solver reports sampled root multiplicity honestly.
  constexpr int cells=32;double prevx=0;auto prev=eval(0);double prevf=prev.valid()?-prev.uv[1]:NAN;
  std::vector<double> roots;bool allvalid=prev.valid(),mono=true;double lastf=prevf;
  for(int i=1;i<=cells;++i){const double x=1079.*i/cells;const auto now=eval(x);const double f=now.valid()?x-now.uv[1]:NAN;
    allvalid=allvalid&&now.valid();if(std::isfinite(f)&&std::isfinite(lastf)&&f<=lastf)mono=false;
    if(std::isfinite(f)&&std::isfinite(prevf)){
      if(std::abs(prevf)<1e-10)roots.push_back(prevx);
      if(prevf*f<0){double a=prevx,b=x,fa=prevf;for(int it=0;it<48;++it){const double mid=(a+b)/2;const auto p=eval(mid);if(!p.valid()){allvalid=false;break;}
          const double fm=mid-p.uv[1];if(std::abs(fm)<1e-10){a=b=mid;break;}if(fa*fm<=0)b=mid;else {a=mid;fa=fm;}}
        roots.push_back((a+b)/2);}}
    prevx=x;prevf=f;lastf=f;
  }
  if(std::isfinite(prevf)&&std::abs(prevf)<1e-10)roots.push_back(1079);
  roots.erase(std::unique(roots.begin(),roots.end(),[](double a,double b){return std::abs(a-b)<1e-7;}),roots.end());
  out.roots=int(roots.size());out.sampled_monotonic=mono&&allvalid;
  if(roots.empty()){out.reason=allvalid?"no_valid_row_root":"arithmetic_domain_failure_during_RS_search";return out;}
  out.row=*std::min_element(roots.begin(),roots.end(),[&](double a,double b){return std::abs(a-initial)<std::abs(b-initial);});
  const auto p=eval(out.row);out.uv=p.uv;out.depth=p.depth;out.equality_px=std::abs(out.row-p.uv[1]);
  const double h=1e-3;const auto pp=eval(std::min(1079.,out.row+h)),pm=eval(std::max(0.,out.row-h));
  if(pp.valid()&&pm.valid())out.implicit_denominator=1-(pp.uv[1]-pm.uv[1])/(std::min(1079.,out.row+h)-std::max(0.,out.row-h));
  out.valid=p.valid()&&out.equality_px<1e-7&&std::abs(out.implicit_denominator)>1e-6;
  // Sufficient whole-readout certificate, independent of root samples:
  // derivative B-spline convex hull -> rigid Y velocity norm -> depth/ray box
  // -> polynomial Brown Jacobian -> |tau*pi_y,t/1079| < 1.
  const auto& g=blocks[0].values;const double tau=g[31],half=std::abs(tau)/2;
  auto fullcontrol=[&](int i,int channel){if(channel<2&&i==dependent_control){double s=0,pivot=0;
      for(int k=0;k<4;++k){if(reference_indices[k]==i)pivot=reference_weights[k];else s+=reference_weights[k]*blocks[control_block_begin+reference_indices[k]].values[channel];}return -s/pivot;}
    return blocks[control_block_begin+i].values[channel];};
  std::array<double,3> rates{};const int begin=int(std::floor((time-half-first_control_time)/dt))-2,end=int(std::floor((time+half-first_control_time)/dt))+2;
  for(int i=std::max(0,begin);i<=std::min(n_controls-2,end);++i)for(int c=0;c<3;++c)rates[c]=std::max(rates[c],std::abs(fullcontrol(i+1,c)-fullcontrol(i,c))/dt);
  for(auto& r:rates)r=std::nextafter(r*(1+1e-12)+1e-12,INFINITY);
  bool certified=false;
  try {
    const auto& world=fixed_world;
    const auto tr=trajectory(time,ptrs.data(),l);const auto hh=head(g.data());const auto oo=zoom_optics(optics(g.data()),tr[2]);const auto pose=camera_pose(hh,tr[0],tr[1],oo.z);
    const auto Y=matvec(pose.R,sub(world,pose.C));const double omega=rates[0]+rates[1],zd=rates[2],zabs=std::abs(oo.z)+zd*half;
    const double dy=omega*(std::sqrt(dot(sub(world,hh.P),sub(world,hh.P)))+std::abs(hh.a))+(std::abs(hh.pupil_linear)+2*std::abs(hh.pupil_quadratic)*zabs)*zd;
    const double span=dy*half,zmin=Y[2]-span;
    if(zmin>0){const double Xabs=std::abs(Y[0])+span,Yabs=std::abs(Y[1])+span,x=Xabs/zmin,y=Yabs/zmin,r2=x*x+y*y,r=std::sqrt(r2);
      const double xp=dy*(zmin+Xabs)/(zmin*zmin),yp=dy*(zmin+Yabs)/(zmin*zmin);
      const double k1a=std::abs(g[25])+std::abs(g[26])*zabs,k2a=std::abs(g[27])+std::abs(g[28])*zabs,p1=std::abs(g[29]),p2=std::abs(g[30]);
      const double L=1+k1a*r2+k2a*r2*r2,Jyx=2*x*y*(k1a+2*k2a*r2)+2*p1*x+2*p2*y,Jyy=L+2*y*y*(k1a+2*k2a*r2)+6*p1*y+2*p2*x;
      const double ybrown=y*L+p1*(r2+2*y*y)+2*p2*x*y,direct=y*(std::abs(g[26])*r2+std::abs(g[28])*r2*r2)*zd;
      const double fymax=std::exp(tr[2]+zd*half-g[22]/2),ppprime=std::abs(g[24])*zd;
      const double speed=std::nextafter((fymax*(ybrown*zd+Jyx*xp+Jyy*yp+direct)+ppprime)*(1+1e-10)+1e-8,INFINITY);
      const double k1lo=oo.brown.k1-std::abs(g[26])*zd*half,k2lo=oo.brown.k2-std::abs(g[28])*zd*half;
      const double fold=std::min(quadratic_minimum(1.,k1lo,k2lo,r2),quadratic_minimum(1.,3*k1lo,5*k2lo,r2))-6*std::hypot(p1,p2)*r;
      certified=r<=radius_max&&fold>1e-10&&std::abs(tau)*speed/1079.<1.-1e-9;
    }
  }catch(...){certified=false;}
  out.unique=out.valid&&roots.size()==1&&certified;
  out.reason=out.unique?"unique_nonsingular_full_interval_contraction_certified":out.valid?(roots.size()==1?"root_requires_full_interval_certificate":"multiple_row_roots"):"singular_or_invalid_RS_root";return out;
}

std::vector<double> EventModel::physical_constraints() const {
  std::vector<double> v;for(const auto& spec:physical_specs()){std::vector<const double*> p;for(int id:spec.block_ids)p.push_back(blocks[id].values.data());v.push_back(evaluate_physical(spec,p.data()));}
  return v;
}
EventRayResult EventModel::inverse_ray(double u,double v,double frame_time) const {
  EventRayResult out;if(!std::isfinite(u)||!std::isfinite(v)||v<0||v>1079){out.reason="outside_native_row_domain";return out;}
  try {const auto& g=blocks[0].values;const auto l=layout(0,0,frame_time);const auto ptrs=pointers(l);out.native_time=row_time(frame_time,g[31],v);
    const auto tr=trajectory(out.native_time,ptrs.data(),l);const auto oo=zoom_optics(optics(g.data()),tr[2]);const auto pose=camera_pose(head(g.data()),tr[0],tr[1],oo.z);
    if(!pose.valid()||brown_no_fold_lower_bound(oo.brown)<=0){out.reason="uncertified_camera_or_Brown_domain";return out;}
    const Vec2<double> target={(u-oo.intr.cx)/oo.intr.fx,(v-oo.intr.cy)/oo.intr.fy};Vec2<double> xy=target;
    if(dot(Vec3<double>{xy[0],xy[1],0},Vec3<double>{xy[0],xy[1],0})>radius_max*radius_max)xy={0.,0.};
    double error=INFINITY;
    for(int iter=0;iter<60;++iter){const auto pred=brown_distort(xy,oo.brown);const double ex=pred[0]-target[0],ey=pred[1]-target[1];error=std::hypot(ex,ey);if(error<1e-13)break;
      const double x=xy[0],y=xy[1],r2=x*x+y*y,k1=oo.brown.k1,k2=oo.brown.k2,p1=oo.brown.p1,p2=oo.brown.p2,L=1+k1*r2+k2*r2*r2,A=k1+2*k2*r2;
      const double Jxx=L+2*x*x*A+2*p1*y+6*p2*x,Jxy=2*x*y*A+2*p1*x+2*p2*y,Jyy=L+2*y*y*A+6*p1*y+2*p2*x,det=Jxx*Jyy-Jxy*Jxy;
      if(!(det>1e-12)){out.reason="singular_inverse_distortion";return out;}
      const double dx=(Jyy*ex-Jxy*ey)/det,dy=(-Jxy*ex+Jxx*ey)/det;bool accepted=false;
      for(double step=1.;step>=1e-8;step*=.5){const Vec2<double> next={x-step*dx,y-step*dy};if(next[0]*next[0]+next[1]*next[1]>radius_max*radius_max)continue;
        const auto test=brown_distort(next,oo.brown);if(std::hypot(test[0]-target[0],test[1]-target[1])<error){xy=next;accepted=true;break;}}
      if(!accepted){out.reason="inverse_distortion_no_domain_step";return out;}}
    out.inverse_distortion_error=error*std::max(oo.intr.fx,oo.intr.fy);if(out.inverse_distortion_error>1e-6){out.reason="inverse_distortion_not_converged";return out;}
    out.origin=pose.C;out.direction=event_normalized(matvec(transpose(pose.R),Vec3<double>{xy[0],xy[1],1.}));out.valid=true;
    if(out.direction[2]<-1e-12){const double lambda=-pose.C[2]/out.direction[2];if(lambda>0){out.forward_ice_intersection=true;out.ice=add(pose.C,scale(out.direction,lambda));out.inside_rink=rink_membership(out.ice[0],out.ice[1],g.data())<=1e-10;}}
    out.reason=out.forward_ice_intersection?(out.inside_rink?"geometric_ice_ray_requires_visibility_evidence":"ray_hits_ice_plane_outside_rink"):"ray_does_not_hit_forward_ice_plane";return out;
  }catch(...){out.reason="inverse_ray_arithmetic_domain_failure";return out;}
}
std::vector<EventPhysicalConstraint> EventModel::physical_specs() const {
  std::vector<EventPhysicalConstraint> out;
  auto add=[&](std::string id,int type,int block,int index=0,int axis=0,int direction=1,double scale=1.){
    EventPhysicalConstraint c;c.id=std::move(id);c.type=type;c.block_ids={0};if(block!=0)c.block_ids.push_back(block);c.index=index;c.axis=axis;c.direction=direction;c.scale=scale;out.push_back(std::move(c));};
  add("rink/nonzero_short_straight",0,0,0,0,1,.5);
  for(std::size_t i=1;i<blocks.size();++i){const auto& b=blocks[i];
    if(b.name.rfind("ad/",0)==0){add(b.name+"/bottom_on_board",1,int(i),0,0,1,.02);add(b.name+"/nonzero_height",2,int(i),0,0,1,.02);add(b.name+"/top_on_board",3,int(i),0,0,1,.02);add(b.name+"/span_not_larger_than_perimeter",4,int(i));}
    if(b.name.rfind("ice/circle/",0)==0){for(std::size_t j=2;j<b.values.size();++j)add(b.name+"/positive_radius_"+std::to_string(j),5,int(i),int(j),0,1,.05);
      for(std::size_t j=2;j+1<b.values.size();++j)add(b.name+"/nested_radius_"+std::to_string(j),6,int(i),int(j),0,1,.05);
      // Explicit sufficient convex containment: the outer circle is contained
      // by its circumscribed square; checking all four vertices is exact for
      // that square and therefore sufficient for the full unsampled circle.
      for(int sx:{-1,1})for(int sy:{-1,1})add(b.name+"/conservative_outer_square_"+std::to_string(sx)+"_"+std::to_string(sy),9,int(i),int(b.values.size()-1),sx,sy);
    }
    if(b.name.rfind("ice/line/",0)==0){add(b.name+"/endpoint0_in_rink",7,int(i),0);add(b.name+"/endpoint1_in_rink",7,int(i),2);add(b.name+"/nonzero_length",8,int(i),0,0,1,.25);}
    if(b.name.rfind("print/",0)==0){std::vector<Vec2<double>> vertices;int ad=-1;for(const auto& m:materials)for(const auto& p:m.paths)if(p.geometry_block==int(i)){vertices.insert(vertices.end(),p.points.begin(),p.points.end());ad=p.advertisement_block;}
      // u/v may describe an INTERNAL graphic atlas, not the physical outer ad
      // canvas. Only sourced board height bounds apply to all finite paths.
      for(int direction:{-1,1}){add(b.name+"/board_height_"+std::to_string(direction),12,int(i),0,0,direction,.02);out.back().vertices=vertices;out.back().block_ids.push_back(ad);}
      add(b.name+"/positive_affine_orientation",11,int(i));}
  }
  return out;
}
} // namespace rinkfit
