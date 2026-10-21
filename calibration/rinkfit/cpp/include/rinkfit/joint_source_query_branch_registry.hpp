#pragma once
// Source association policy only. The raw packet ledger and original shared
// Query objects are never rewritten. Load before documentary metadata trim;
// retain this object with the frozen problem for exact idempotent reuse.
#include "rinkfit/joint_packet.hpp"
#include <cstring>
#include <memory>
namespace rinkfit {
class JointSourceQueryBranchRegistry {
 public:
 using Json=eventio::Json;
 struct Seed {std::uint32_t path=0;double station=0,row=0,alpha=0;};
 struct Entry {std::size_t index=0;std::string id;double lower=0,upper=0,measured=0;Seed seed;Json definition;std::shared_ptr<const JointQuery>original,clipped;};
 static constexpr std::size_t first=910012,stop=916340,count=6328;
 static std::shared_ptr<JointSourceQueryBranchRegistry> load(const Json&binding,const JointEventModel&model,const JointPacket&packet){
  auto out=std::shared_ptr<JointSourceQueryBranchRegistry>(new JointSourceQueryBranchRegistry);out->binding_=binding;require(binding.contains("consumer_module_ref"),"canonical consumer implementation pin is required");out->verify_tree(binding);
  const auto file=out->verified(binding.at("registry"));out->registry_=eventio::read_json(file);auto&reg=out->registry_;require(reg.at("schema")=="rinkfit.C_source_branch_registry.v2","unsupported source branch registry");out->verify_tree(reg);
  auto gate=eventio::read_json(out->verified(binding.at("source_gate")));require(gate.at("schema")=="rinkfit.C_source_branch_overlay_gate.v2"&&gate.at("registry").at("sha256")==binding.at("registry").at("sha256")&&gate.value("accepted_for_source_identity_domain",false)&&gate.at("full_source_rows_retained")==count&&!gate.value("physical_capture_complete",true),"source gate incomplete or falsely closes capture");out->verify_tree(gate);
  require(gate.at("negative_cases").at("opposite_flank_measured_alpha_rejected")==count&&gate.at("negative_cases").at("original_ROI_lower_rejected")==count&&gate.at("negative_cases").at("original_ROI_upper_rejected")==count&&gate.at("negative_cases").at("wrong_source_identity_rejected").get<bool>(),"source branch negative evidence incomplete");
  const auto policy=eventio::read_json(out->verified(reg.at("source_branch_policy")));out->verify_tree(policy);require(policy.at("raw_rows")==count&&policy.at("source_q50_replay_max_abs_diff").get<double>()<1e-9&&policy.at("frozen_source_rows").at("sha256")==reg.at("frozen_source").at("sha256")&&!policy.value("model_or_screen_geometry_used_in_policy",true)&&!policy.value("source_UV_ID_or_packet_modified",true),"source policy provenance mismatch");
  const auto generation=eventio::read_json(out->verified(policy.at("source_generation")));out->verify_tree(generation);require(generation.at("source_video").at("sha256")=="e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d"&&generation.at("source_video").at("time_base")=="1/12800"&&generation.at("candidates").at("sha256")==policy.at("frozen_source_profiles").at("sha256"),"source RGB/native profile generation changed");
  const auto qualification=eventio::read_json(out->verified(policy.at("parent_qualification")));out->verify_tree(qualification);require(qualification.at("source_rows_qualified")==3164&&qualification.at("source_generation").at("sha256")==policy.at("source_generation").at("sha256")&&!qualification.value("source_UV_modified",true),"parent two-boundary material qualification differs");
  require(packet.rows.size()>=stop&&reg.at("alpha_domains").size()==count,"C branch denominator truncated");
  if(packet.source_hash!=reg.at("base_packet").at("sha256").get<std::string>()){
   require(packet.source_json.contains("source_query_branch_registry"),"successor packet has not bound new mathematical source policy");const auto&b=packet.source_json.at("source_query_branch_registry");const auto&r=b.contains("registry")?b.at("registry"):b;require(r.at("sha256")==binding.at("registry").at("sha256"),"successor packet source policy generation differs");
  }
  std::map<std::size_t,Json>raw_queries;for(const auto&q:packet.source_json.at("queries"))if(q.at("row_index").get<std::size_t>()>=first&&q.at("row_index").get<std::size_t>()<stop)require(raw_queries.emplace(q.at("row_index"),q).second,"duplicate raw source Query");
  require(raw_queries.size()==count,"original full Query ledger missing");out->base_model_=eventio::read_json(out->verified(reg.at("base_model")));
  std::map<std::string,Json>physical_pairs;for(const auto&b:model.source_json.at("stroke_pair_model_bindings"))physical_pairs.emplace(b.at("pair_id"),b);
  std::vector<double>ss=out->seed_array<double>(reg.at("numerical_seed_overlay").at("arrays").at("station"),"<f8"),rr=out->seed_array<double>(reg.at("numerical_seed_overlay").at("arrays").at("row"),"<f8"),aa=out->seed_array<double>(reg.at("numerical_seed_overlay").at("arrays").at("alpha"),"<f8");auto paths=out->seed_array<std::uint32_t>(reg.at("numerical_seed_overlay").at("arrays").at("path"),"<u4");
  for(std::size_t j=0;j<count;++j){const auto&d=reg.at("alpha_domains").at(j);Entry e;e.index=d.at("row_index");e.id=d.at("source_row_id");e.lower=d.at("allowed_alpha").at(0);e.upper=d.at("allowed_alpha").at(1);e.measured=d.at("source_measured_alpha");e.seed={paths[j],ss[j],rr[j],aa[j]};e.definition=d;
   require(e.index==first+j&&d.at("status")=="source_identity_branch_qualified"&&e.lower<e.upper&&e.measured>=e.lower&&e.measured<=e.upper&&e.seed.alpha==e.measured,"pending/incorrect own-flank source branch");
   const auto&q=raw_queries.at(e.index);require(q==d.at("original_full_Query"),"full source Query/cubic changed");require(q.at("source_acquisition_window_only").get<bool>()&&q.at("source_profile_ref").at("sha256")==policy.at("frozen_source_profiles").at("sha256"),"full ROI is not the frozen source acquisition profile");e.original=query_from_json(q);e.clipped=clip(*e.original,e.lower,e.upper);
   const auto&r=packet.rows[e.index];out->validate_row(e,r);require(packet.source_json.at("source_family_ids").at(e.index)==d.at("source_family_id"),"source family changed");require(packet.source_json.at("stroke_pairs").at(r.stroke_pair_index)==d.at("original_packet_pair"),"source two-flank pair membership changed");
   require(physical_pairs.at(d.at("pair_id").get<std::string>())==d.at("original_model_pair_binding"),"physical width pair binding changed");require(model.materials.at(r.material).label==d.at("material_label")&&model.source_json.at("materials").at(r.material)==d.at("material_projection_definition"),"material projection definition changed");require(model.blocks.at(r.width_block).name==d.at("width_block_name")&&model.blocks.at(r.width_block).values.size()==1&&model.blocks.at(r.width_block).values[0]>0,"local width block identity/domain changed");
   require(r.query&&(equal(*r.query,*e.original)||equal(*r.query,*e.clipped)),"typed Query differs from original and clipped source function");require(r.station_allowed(model,int(e.seed.path),e.seed.station,1e-12)&&std::isfinite(e.seed.row)&&e.seed.row>=0&&e.seed.row<=1079,"invalid physical curve RS numerical seed domain");
   Vec2<double>measured;require(e.original->point(e.measured,&measured)&&std::hypot(measured[0]-r.uv[0],measured[1]-r.uv[1])<1e-9,"measured alpha no longer reconstructs immutable source UV");out->entries_.push_back(std::move(e));
  }
  for(std::size_t j=0;j<count;j+=2)require(out->entries_[j].upper<=out->entries_[j+1].lower,"opposite source flank branch overlap");
  out->aux_root_=out->verified(reg.at("base_authoritative_aux")).parent_path();out->aux_manifest_=eventio::read_json(out->aux_root_/"manifest.json");for(const auto&key:{"path.u32","station.f64","row.f64","alpha.f64"})eventio::verify(out->aux_root_/key,out->aux_manifest_.at("files").at(key).at("sha256"));
  out->identity_=binding.at("registry").at("sha256");out->report_={{"schema","rinkfit.C_source_query_branch_registry_runtime_report.v2"},{"identity_kind","registry_file_sha256"},{"identity",out->identity_},{"binding",binding},{"affected_source_rows",count},{"full_Query_kept_in_raw_ledger",true},{"typed_Query_clones_only",true},{"source_gate",gate},{"source_policy",policy},{"verified_unique_dependency_pins",out->pins_.size()},{"large_immutable_dependency_bytes_verified_at_load",true},{"boundary_large_file_pin_check","size and last-write-time; changed pin always rejects, no repeated source-video read"}};
  return out;
 }
 const Entry* find(std::size_t index)const{return index>=first&&index<stop?&entries_.at(index-first):nullptr;}
 std::pair<double,double> alpha_bounds(std::size_t index)const{const auto*e=find(index);require(e,"row outside C source policy");return {e->lower,e->upper};}
 const Json& binding()const{return binding_;}
 const std::string& identity()const{return identity_;}
 const Json& report()const{return report_;}
 const Json& registry_document()const{return registry_;}
 // Invoke at frozen/batch boundaries, never once per factor. Source producers
 // and JSON identity registries are re-hashed. Large immutable video/arrays
 // were hashed at load and must retain their exact filesystem pin.
 Json assert_dependencies()const{
  std::size_t byte_checks=0,metadata_checks=0;for(const auto&item:pins_){const auto&p=item.second;require(std::filesystem::exists(p.path)&&std::filesystem::file_size(p.path)==p.size&&std::filesystem::last_write_time(p.path)==p.modified,"source/producer dependency filesystem pin changed: "+p.path.u8string());++metadata_checks;if(p.strong){eventio::verify(p.path,p.sha);++byte_checks;}}
  return {{"identity",identity_},{"unique_dependency_pins",pins_.size()},{"byte_hashes_rechecked",byte_checks},{"filesystem_pins_rechecked",metadata_checks},{"large_source_video_re_read",false},{"passed",true}};
 }
 static std::shared_ptr<const JointQuery> clone_source_branch(const JointQuery&q,double lo,double hi){return clip(q,lo,hi);}
 Json apply_typed_queries(JointPacket&packet)const{return apply_typed_queries(packet.rows);}
 Json apply_typed_queries(std::vector<JointObservation>&rows)const{
  require(rows.size()>=stop,"source packet truncated during source Query attach");std::size_t attached=0,reused=0;double maxerr=0;
  for(const auto&e:entries_){auto&r=rows[e.index];validate_row(e,r);require(bool(r.query),"missing source Query before branch attach");if(equal(*r.query,*e.clipped)){++reused;continue;}require(equal(*r.query,*e.original),"attempt to apply branch to changed typed Query");
   for(double a:{e.lower,e.measured,.5*(e.lower+e.upper),e.upper}){Vec2<double>x,y;require(r.query->point(a,&x)&&e.clipped->point(a,&y),"clipped source Query point unavailable");maxerr=std::max(maxerr,std::hypot(x[0]-y[0],x[1]-y[1]));}
   r.query=e.clipped;++attached;
  }
  require(maxerr<1e-9,"branch clipping changed source Query function");return {{"schema","rinkfit.C_typed_source_Query_branch_attachment.v1"},{"C_rows",count},{"new_typed_copies_attached",attached},{"existing_identical_typed_copies_reused",reused},{"original_full_Query_or_source_UV_rewritten",false},{"same_source_Query_function_max_abs_px",maxerr},{"registry",binding_.at("registry")}};
 }
 Json rebind_C_seeds_after_complete_CP(const JointEventModel&model,JointPacket&packet,std::vector<double>&station,std::vector<double>&row,std::vector<double>&alpha)const{
  const auto n=packet.rows.size();require(n>=stop&&station.size()==n&&row.size()==n&&alpha.size()==n,"numerical seed overlay builder shape");
  // The overlay was generated from this complete physical CP, never from a
  // model-only pin. Newly appended physical blocks are permitted after it.
  const auto&base=base_model_.at("blocks");require(model.blocks.size()>=base.size(),"base complete physical CP truncated");for(std::size_t i=0;i<base.size();++i)require(model.blocks[i].name==base[i].at("name")&&model.blocks[i].values==base[i].at("values").get<std::vector<double>>(),"C seed overlay needs exact complete parent physical CP values");
  bool already=true;for(const auto&e:entries_)already&=packet.rows[e.index].path==int(e.seed.path)&&same(station[e.index],e.seed.station)&&same(row[e.index],e.seed.row)&&same(alpha[e.index],e.seed.alpha);
  for(const auto&key:{"station","row","alpha","path"}){
   const auto bytes=incoming_bytes(packet,station,row,alpha,key,0,already?first:stop);const auto filename=std::string(key)+(std::string(key)=="path"?".u32":".f64");auto expected=eventio::bytes(aux_root_/filename);require(expected.size()>=bytes.size()&&std::equal(bytes.begin(),bytes.end(),expected.begin()),"complete parent four-aux import must precede C-only rebind: "+std::string(key));
  }
  std::map<std::string,std::string>before;for(const auto&key:{"station","row","alpha","path"})before[key]=unaffected_digest(packet,station,row,alpha,key);std::size_t changed=0;
  for(const auto&e:entries_){auto&r=packet.rows[e.index];validate_row(e,r);changed+=!same(station[e.index],e.seed.station)||!same(row[e.index],e.seed.row)||!same(alpha[e.index],e.seed.alpha)||r.path!=int(e.seed.path);station[e.index]=r.station=e.seed.station;row[e.index]=r.row=e.seed.row;alpha[e.index]=r.alpha=e.seed.alpha;r.path=e.seed.path;}
  for(const auto&key:{"station","row","alpha","path"})require(before[key]==unaffected_digest(packet,station,row,alpha,key),"C source seed overlay changed an unrelated source auxiliary");
  return {{"schema","rinkfit.C_source_branch_numerical_seed_overlay_receipt.v1"},{"affected_source_rows",count},{"actually_changed_C_rows",changed},{"idempotent_existing_overlay",already},{"original_complete_parent_physical_and_four_aux_import_verified",true},{"all_other_four_aux_bitwise_equal",true},{"all6328_alpha_equal_original_source_measured_alpha",true},{"source_UV_or_full_Query_changed",false},{"strict_source_branch_capture_pending_rows",5017},{"initial_Query_closure_claimed",false},{"old_AL_duals_may_be_resumed",false},{"new_mathematical_identity_registry",binding_.at("registry")}};
 }
 private:
 struct DependencyPin {std::filesystem::path path;std::string sha;std::uintmax_t size=0;std::filesystem::file_time_type modified;bool strong=false;};
 Json registry_,binding_,base_model_,aux_manifest_,report_;std::string identity_;std::map<std::string,DependencyPin>pins_;std::vector<Entry>entries_;std::filesystem::path aux_root_;std::map<std::string,std::string>verified_;
 static void require(bool x,const std::string&why){if(!x)throw std::runtime_error("source Query branch registry: "+why);}
 std::filesystem::path verified(const Json&ref){auto file=std::filesystem::u8path(ref.contains("file")?ref.at("file").get<std::string>():ref.at("path").get<std::string>());require(file.is_absolute(),"all source policy references must be absolute");auto expected=ref.at("sha256").get<std::string>();auto key=file.lexically_normal().u8string();auto it=verified_.find(key);if(it!=verified_.end()){require(it->second==expected,"one source path has conflicting generation hashes");return file;}eventio::verify(file,expected);verified_.emplace(key,expected);const auto ext=file.extension().u8string();const auto size=std::filesystem::file_size(file);const bool strong=size<=8*1024*1024||ext==".json"||ext==".py"||ext==".hpp"||ext==".cpp"||ext==".ptx";pins_.emplace(key,DependencyPin{file,expected,size,std::filesystem::last_write_time(file),strong});return file;}
 void verify_tree(const Json&v){if(v.is_object()){if((v.contains("file")||v.contains("path"))&&v.contains("sha256")){verified(v);return;}for(auto it=v.begin();it!=v.end();++it)verify_tree(it.value());}else if(v.is_array())for(const auto&x:v)verify_tree(x);}
 template<class T>std::vector<T>seed_array(const Json&d,const char*dtype){require(d.at("dtype")==dtype&&d.at("shape")==Json::array({count}),"source seed dtype/shape changed");auto b=eventio::bytes(verified(d));require(b.size()==count*sizeof(T),"source seed binary size changed");std::vector<T>x(count);std::memcpy(x.data(),b.data(),b.size());return x;}
 static std::shared_ptr<const JointQuery>query_from_json(const Json&q){auto p=std::make_shared<JointQuery>();p->alpha=q.at("alpha").get<std::vector<double>>();p->uv=q.at("native_uv").get<std::vector<Vec2<double>>>();p->nx=q.value("numerator_x",std::vector<double>{});p->ny=q.value("numerator_y",std::vector<double>{});p->den=q.value("denominator",std::vector<double>{});p->origin=q.value("alpha_origin",0.);p->scale=q.value("alpha_scale",1.);return p;}
 static bool equal(const JointQuery&a,const JointQuery&b){return a.alpha==b.alpha&&a.uv==b.uv&&a.nx==b.nx&&a.ny==b.ny&&a.den==b.den&&a.origin==b.origin&&a.scale==b.scale;}
 static std::shared_ptr<const JointQuery>clip(const JointQuery&q,double lo,double hi){require(q.alpha.size()>=2&&lo>=q.alpha.front()&&hi<=q.alpha.back()&&lo<hi,"branch outside frozen acquisition domain");auto r=std::make_shared<JointQuery>(q);r->alpha.clear();r->uv.clear();auto add=[&](double a){Vec2<double>v;require(q.point(a,&v),"original source Query point undefined");r->alpha.push_back(a);r->uv.push_back(v);};add(lo);for(double a:q.alpha)if(a>lo&&a<hi)add(a);add(hi);return r;}
 static void validate_row(const Entry&e,const JointObservation&r){const auto&d=e.definition;require(r.active&&r.kind==d.at("kind")&&r.material==d.at("material_index")&&r.shot_id=="s09"&&r.source_row_id==e.id&&r.source_dataset_id==d.at("source_dataset_id")&&r.pts==d.at("native_pts")&&r.uv==d.at("source_uv").get<Vec2<double>>()&&r.normal==d.at("source_normal").get<Vec2<double>>()&&r.E==d.at("original_E")&&r.stroke_pair_index==d.at("pair_index")&&r.stroke_source_side==d.at("source_side_index")&&r.width_block==d.at("width_block")&&r.width_sign==d.at("world_side_sign"),"C typed source ID/UV/PTS/material/width/pair changed at "+std::to_string(e.index));}
 static bool same(double x,double y){return std::memcmp(&x,&y,sizeof(double))==0;}
 static std::vector<std::uint8_t>incoming_bytes(const JointPacket&p,const std::vector<double>&s,const std::vector<double>&r,const std::vector<double>&a,const std::string&key,std::size_t lo,std::size_t hi){const std::size_t width=key=="path"?4:8;std::vector<std::uint8_t>b((hi-lo)*width);for(std::size_t i=lo;i<hi;++i)if(key=="path"){const std::uint32_t v=p.rows[i].path;std::memcpy(b.data()+(i-lo)*width,&v,width);}else{const double v=key=="station"?s[i]:key=="row"?r[i]:a[i];std::memcpy(b.data()+(i-lo)*width,&v,width);}return b;}
 static std::string unaffected_digest(const JointPacket&p,const std::vector<double>&s,const std::vector<double>&r,const std::vector<double>&a,const std::string&key){auto b=incoming_bytes(p,s,r,a,key,0,first);auto c=incoming_bytes(p,s,r,a,key,stop,p.rows.size());eventio::SHA256Stream h;h.update(b.data(),b.size());h.update(c.data(),c.size());return h.finish();}
};
} // namespace rinkfit
