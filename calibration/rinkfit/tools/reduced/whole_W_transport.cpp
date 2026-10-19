#define main unused_frozen_qbar_main
#include "transport_qbar_base.cpp"
#undef main
#include <iomanip>
#ifdef RINKFIT_GPU_ROWS
#include "row_uv_cache.hpp"
#endif
int main(int argc,char**argv){try{
 const auto start=std::chrono::steady_clock::now();if(argc!=3)throw std::runtime_error("usage CP OUT");
 auto cp=std::filesystem::u8path(argv[1]),out=std::filesystem::u8path(argv[2]);const auto hash=eventio::file_sha256(cp);const auto doc=eventio::read_json(cp);if(doc.at("schema")!="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1")throw std::runtime_error("complete CP required");
 std::filesystem::create_directories(out);if(std::filesystem::exists(out/"manifest.json"))throw std::runtime_error("immutable output exists");
 auto ref=[&](const Json&r){auto p=std::filesystem::u8path(r.at("path").get<std::string>());eventio::verify(p,r.at("sha256"));return p;};
 const auto mp=ref(doc.at("model"));const auto lm=eventio::read_json(ref(doc.at("linearization")));auto pp=ref(lm.at("packet")),sp=ref(lm.at("semantic"));auto model=EventModel::load(mp.u8string());EventPacket packet(pp);packet.load_semantics(sp);packet.load_query(std::filesystem::u8path(packet.manifest.at("query_packet").at("manifest").get<std::string>()),packet.root.parent_path()/"query_initialization.json");
 const auto n=packet.rows.size();const auto& a=doc.at("row_state_files");const auto s=eventio::read_array<double>(cp.parent_path(),a.at("station"),"<f8"),v=eventio::read_array<double>(cp.parent_path(),a.at("row"),"<f8"),alpha=eventio::read_array<double>(cp.parent_path(),a.at("alpha"),"<f8");const auto branch=eventio::read_array<std::uint32_t>(cp.parent_path(),a.at("branch"),"<u4");if(n!=182017||s.size()!=n||v.size()!=n||alpha.size()!=n||branch.size()!=n||packet.active_rows.size()!=182016)throw std::runtime_error("source coverage mismatch");
 CSR uvj;std::vector<double> error(2*n,NAN),uv(2*n,NAN),row(n,NAN),rs_h(n,NAN),query_h(2*n,NAN),ds_right(2*n,0),ds_left(2*n,0),slo(n,0),shi(n,0),slo_left(n,0),shi_left(n,0),rcond(n,NAN);std::vector<std::uint8_t> stat(n,255),vertex(n,0);Json failures=Json::array();std::size_t profiles=0,entities=0;double max_rs=0,max_query=0;
#ifdef RINKFIT_GPU_ROWS
 // Only the raw UV partials use CUDA. Every true RS root is freshly certified
 // by the original CPU implementation at this model and all four auxiliaries.
 // No station refinement, optimizer or checkpoint update occurs in this tool.
 const auto gpu_root_begin=std::chrono::steady_clock::now();
 std::vector<double> gpu_root_rows(n,NAN);
 for(std::size_t i=0;i<n;++i)if(packet.is_active(i)){
  const int mi=packet.model_material(i),pi=branch[i];
  const auto& path=model.materials.at(mi).paths.at(pi);
  if(path.variant_id!=packet.model_variant_id(i))throw std::runtime_error("GPU finite variant mismatch");
  const auto layout=model.layout(mi,pi,(packet.pts[i]-256256)/12800.);
  const auto root=model.project_rs(mi,pi,s[i],layout.frame_time,v[i]);
  if(!root.valid||!root.unique)throw std::runtime_error("GPU prepass strict RS root failure");
  gpu_root_rows[i]=root.row;
 }
 const double gpu_root_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-gpu_root_begin).count();
 const auto gpu_begin=std::chrono::steady_clock::now();
 rinkfit::gpu_rows::RowUVCache gpu_rows(model,packet,branch);
 gpu_rows.evaluate(model,packet,branch,s,gpu_root_rows,alpha,doc.at("b").get<double>(),hash);
 const double gpu_partial_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-gpu_begin).count();
#endif
 for(std::size_t i=0;i<n;++i){if(!packet.is_active(i)){uvj.end();uvj.end();continue;}try{
  int mi=packet.model_material(i),pi=branch[i];const auto& path=model.materials.at(mi).paths.at(pi);if(path.variant_id!=packet.model_variant_id(i))throw std::runtime_error("finite variant mismatch");const auto layout=model.layout(mi,pi,(packet.pts[i]-256256)/12800.);
#ifdef RINKFIT_GPU_ROWS
  // Retain the original projection's CPU authoritative primal. The root value
  // is from the exact CPU prepass above; no saved numerical result is reused.
  struct {double row;} root{gpu_root_rows[i]};
  const auto z=gpu_rows.projection(model,i,layout,s[i],root.row);
#else
  const auto root=model.project_rs(mi,pi,s[i],layout.frame_time,v[i]);if(!root.valid||!root.unique)throw std::runtime_error("strict RS root failure");const auto z=projection_ad(model,layout,s[i],root.row);
#endif
  const auto& sv=z.jac.back();double us=sv[0],ur=sv[1],vs=sv[2],vr=sv[3],den=1-vr;if(!(den>0))throw std::runtime_error("implicit RS denominator");row[i]=root.row;rs_h[i]=v[i]-z.uv[1];max_rs=std::max(max_rs,std::abs(rs_h[i]));uv[2*i]=z.uv[0];uv[2*i+1]=z.uv[1];error[2*i]=z.uv[0]-packet.uv[2*i];error[2*i+1]=z.uv[1]-packet.uv[2*i+1];
  Eigen::Matrix3d inv;bool profile=packet.kind[i]==2;
  if(profile){++profiles;const auto& q=packet.queries.at(packet.query_index[i]);if(alpha[i]<=q.lower||alpha[i]>=q.upper||s[i]<=0||s[i]>=1)throw std::runtime_error("profile active finite boundary requires dedicated chart");const auto U=rational(q.numerator_u,alpha[i]),V=rational(q.numerator_v,alpha[i]),D=rational(q.denominator,alpha[i]);if(std::abs(D[0])<1e-14)throw std::runtime_error("profile rational denominator");const double qp0=(U[1]*D[0]-U[0]*D[1])/(D[0]*D[0]),qp1=(V[1]*D[0]-V[0]*D[1])/(D[0]*D[0]);query_h[2*i]=z.uv[0]-U[0]/D[0];query_h[2*i+1]=z.uv[1]-V[0]/D[0];max_query=std::max({max_query,std::abs(query_h[2*i]),std::abs(query_h[2*i+1])});Eigen::Matrix3d A;A<<-vs,den,0,us,ur,-qp0,vs,vr,-qp1;Eigen::FullPivLU<Eigen::Matrix3d> lu(A);rcond[i]=lu.rcond();if(!lu.isInvertible()||rcond[i]<128*std::numeric_limits<double>::epsilon())throw std::runtime_error("singular profile chart");inv=lu.inverse();if(path.kind==2&&std::abs(s[i]*(path.points.size()-1)-std::round(s[i]*(path.points.size()-1)))<1e-7)throw std::runtime_error("profile at nonsmooth vertex");
  }
  if(std::abs(rs_h[i])>1e-6||(profile&&std::max(std::abs(query_h[2*i]),std::abs(query_h[2*i+1]))>1e-6))throw std::runtime_error("saved exact root residual invalid");
  std::vector<std::tuple<int,int,double,double>> deriv;
  for(std::size_t b=0;b<z.ids.size();++b){int id=z.ids[b];const auto& spec=model.blocks[id];const auto size=spec.values.size();for(std::size_t c=0;c<size;++c){if(spec.constant||std::find(spec.fixed_indices.begin(),spec.fixed_indices.end(),int(c))!=spec.fixed_indices.end())continue;const double ut=z.jac[b][c],vt=z.jac[b][size+c];double du=ut+ur*vt/den,dv=vt/den;if(profile){const Eigen::Vector3d dz=-inv*Eigen::Vector3d(-vt,ut,vt);du=ut+us*dz[0]+ur*dz[1];dv=vt+vs*dz[0]+vr*dz[1];}if(!std::isfinite(du)||!std::isfinite(dv))throw std::runtime_error("nonfinite physical derivative");deriv.emplace_back(id,int(c),du,dv);}}
  for(auto[b,c,du,dv]:deriv)uvj.add(b,c,du);uvj.end();for(auto[b,c,du,dv]:deriv)uvj.add(b,c,dv);uvj.end();
  if(!profile&&!path.physical_point){ds_right[2*i]=ds_left[2*i]=us+ur*vs/den;ds_right[2*i+1]=ds_left[2*i+1]=vs/den;double lower=0,upper=1,lowerL=0,upperL=1;if(path.kind==2){const double edges=path.points.size()-1,raw=s[i]*edges;const auto cell=std::min(path.points.size()-2,std::size_t(std::floor(raw)));lower=double(cell)/edges;upper=double(cell+1)/edges;lowerL=lower;upperL=upper;bool at_vertex=std::abs(raw-std::round(raw))<1e-9;if(at_vertex){vertex[i]=1;double k=std::round(raw);lower=k/edges;upper=std::min(1.,(k+1)/edges);lowerL=std::max(0.,(k-1)/edges);upperL=k/edges;if(k==edges){lower=lowerL;upper=upperL;}if(k>0){double sl=std::nextafter(k/edges,-std::numeric_limits<double>::infinity());auto zz=projection_ad(model,layout,sl,root.row);auto jj=zz.jac.back();double dl=1-jj[3];if(!(dl>0))throw std::runtime_error("left RS derivative singular");ds_left[2*i]=jj[0]+jj[1]*jj[2]/dl;ds_left[2*i+1]=jj[2]/dl;}}}
   // Station bounds remain finite; periodic seams can use another legal chart.
   slo[i]=lower-s[i];shi[i]=upper-s[i];slo_left[i]=lowerL-s[i];shi_left[i]=upperL-s[i];
  }else if(path.physical_point)++entities;stat[i]=0;
 }catch(const std::exception&e){if(uvj.offsets.size()!=2*i+1)throw;uvj.end();uvj.end();stat[i]=1;failures.push_back({{"source_row",i},{"reason",e.what()}});}if((i+1)%32768==0)std::cout<<Json{{"kind","whole_W_transport_progress"},{"raw_accounted",i+1}}.dump()<<std::endl;}
 if(!failures.empty())throw std::runtime_error("transport failed "+failures.dump());
 std::vector<int> membership(n,0);std::vector<double> wres,gradient(2*n,0);Json group_desc=Json::array();std::vector<std::uint64_t> group_rows;std::vector<double> group_W;double W=0;
 for(const auto& g:packet.groups){const auto ro=group_rows.size(),wo=group_W.size(),rank=g.rank,width=2*g.rows.size();for(auto i:g.rows){++membership[i];group_rows.push_back(i);}group_W.insert(group_W.end(),g.W.begin(),g.W.end());std::vector<double> r(rank,0);for(std::size_t k=0;k<rank;++k){for(std::size_t j=0;j<g.rows.size();++j)for(int c=0;c<2;++c)r[k]+=g.W[k*width+2*j+c]*error[2*g.rows[j]+c];wres.push_back(r[k]);W+=.5*r[k]*r[k];}for(std::size_t j=0;j<g.rows.size();++j)for(int c=0;c<2;++c)for(std::size_t k=0;k<rank;++k)gradient[2*g.rows[j]+c]+=g.W[k*width+2*j+c]*r[k];group_desc.push_back({{"row_offset",ro},{"members",g.rows.size()},{"weight_offset",wo},{"rank",rank}});}
 for(std::size_t i=0;i<n;++i)if(membership[i]!=(packet.is_active(i)?1:0))throw std::runtime_error("W partition mismatch");if(wres.size()!=214112||profiles!=961)throw std::runtime_error("W/profile rank mismatch");
 auto write=[&](std::string fn,const auto& data,std::string dt){using T=typename std::decay_t<decltype(data)>::value_type;std::ofstream f(out/fn,std::ios::binary);if(!data.empty())f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(T)));f.close();if(!f)throw std::runtime_error("write failure");return Json{{"file",fn},{"dtype",dt},{"shape",Json::array({data.size()})},{"sha256",eventio::file_sha256(out/fn)}};};
 Json arrays{{"error_uv",write("error_uv.f64",error,"<f8")},{"uv",write("uv.f64",uv,"<f8")},{"true_rs_row",write("true_rs_row.f64",row,"<f8")},{"saved_RS_h",write("saved_RS_h.f64",rs_h,"<f8")},{"query_h",write("query_h.f64",query_h,"<f8")},{"status",write("status.u8",stat,"|u1")},{"station_UV_right",write("station_UV_right.f64",ds_right,"<f8")},{"station_UV_left",write("station_UV_left.f64",ds_left,"<f8")},{"station_delta_lower_right",write("station_lower_right.f64",slo,"<f8")},{"station_delta_upper_right",write("station_upper_right.f64",shi,"<f8")},{"station_delta_lower_left",write("station_lower_left.f64",slo_left,"<f8")},{"station_delta_upper_left",write("station_upper_left.f64",shi_left,"<f8")},{"vertex",write("vertex.u8",vertex,"|u1")},{"profile_implicit_rcond",write("profile_implicit_rcond.f64",rcond,"<f8")},{"W_residual",write("W_residual.f64",wres,"<f8")},{"W_error_gradient",write("W_error_gradient.f64",gradient,"<f8")},{"group_rows",write("group_rows.u64",group_rows,"<u8")},{"group_W",write("group_W.f64",group_W,"<f8")}};
 Json csr{{"row_offsets",write("UV_row_offsets.u64",uvj.offsets,"<u8")},{"block_id",write("UV_block_id.u32",uvj.blocks,"<u4")},{"component",write("UV_component.u32",uvj.components,"<u4")},{"values",write("UV_values.f64",uvj.values,"<f8")}};
 Json report{{"schema","rinkfit.whole_original_W_native_UV_transport.v1"},{"no_Solve",true},{"checkpoint",{{"path",cp.u8string()},{"sha256",hash}}},{"model",doc.at("model")},{"packet",lm.at("packet")},{"semantic",lm.at("semantic")},{"raw_rows",n},{"active_rows",packet.active_rows.size()},{"raw_arrays_preserved",true},{"physical_theta_changed",false},{"ordinary_station_refined",false},{"ordinary_fixed_station_true_RS_UV_derivative",true},{"ordinary_station_tangent_is_not_group_W_elimination",true},{"profile_exact_finite_Q_implicit_S_V_alpha",true},{"profile_count",profiles},{"entity_count",entities},{"full_W_rank",wres.size()},{"original_W_half_cost",W},{"original_groups_including_independent",group_desc.size()},{"statistical_partition_exactly_once",true},{"arrays",arrays},{"physical_UV_CSR",csr},{"groups",group_desc},{"max_saved_RS_h",max_rs},{"max_query_h",max_query},{"source_order_is_raw_array_order",true},{"excluded_rows",packet.excluded_rows},{"final_PhaseII_or_KKT_success",false},{"elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
#ifdef RINKFIT_GPU_ROWS
 report["compute_backend"]="CUDA_FP64_raw_UV_partials_CPU_primal_RS_profile_W";
 report["gpu_rows"]=gpu_rows.report();report["cpu_true_RS_root_seconds"]=gpu_root_seconds;
 report["gpu_setup_transfer_partial_seconds"]=gpu_partial_seconds;
#else
 report["compute_backend"]="CPU_AD64";
#endif
 if(eventio::file_sha256(cp)!=hash)throw std::runtime_error("CP changed");eventio::atomic_json(out/"manifest.json",report);std::cout<<Json{{"W",W},{"active_rows",packet.active_rows.size()},{"UV_nnz",uvj.values.size()},{"profiles",profiles},{"elapsed_seconds",report.at("elapsed_seconds")}}.dump()<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
