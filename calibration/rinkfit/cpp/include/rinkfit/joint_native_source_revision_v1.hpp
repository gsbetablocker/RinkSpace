#pragma once
#include "rinkfit/joint_packet.hpp"
#include <cmath>
#include <map>
#include <set>
namespace rinkfit {
template<class Packet, class Model, class Pin>
eventio::Json validate_joint_native_source_revision_v1(const Packet& p, const Packet& parent,
 const Model& model, const eventio::Json& qualification, const eventio::Json& revision,
 const std::filesystem::path& revision_file, Pin pin) {
 auto require=[](bool ok,const std::string& reason){if(!ok)throw std::runtime_error("native source revision: "+reason);};
 const auto base=revision_file.parent_path();
 require(revision.at("schema")=="rinkfit.native_source_measurement_revision.v1"&&
  revision.at("parent_packet")==qualification.at("parent_packet")&&
  revision.at("parent_qualification")==qualification.at("parent_qualification")&&
  revision.at("source_rows_deleted")==0&&revision.at("original_source_UV_preserved_in_parent")==true&&
  revision.at("actual_native_RGB_replayed")==true&&revision.at("source_model_projection_used")==false&&
  revision.at("source_images_changed")==false&&revision.at("measured_uv_changed")==true&&
  revision.at("source_noise_W_or_sigma_issued")==false,"independent actual RGB source-generation contract");
 pin(revision.at("issuer"),base);
 const auto review_file=pin(revision.at("review"),base);const auto review=eventio::read_json(review_file);
 require(review.at("schema")=="rinkfit.native_source_measurement_revision_review.v1"&&
  review.at("parent_packet")==qualification.at("parent_packet")&&review.at("source_model_projection_used")==false&&
  review.at("source_images_changed")==false&&review.at("measured_uv_changed")==true&&!review.at("evidence").empty(),"exact reviewed source ancestry");
 pin(review.at("source_config"),review_file.parent_path());pin(review.at("source_coverage"),review_file.parent_path());
 for(const auto& ref:review.at("evidence"))pin(ref,review_file.parent_path());
 require(!revision.at("actual_dependencies").empty(),"missing actual source dependencies");
 for(const auto& ref:revision.at("actual_dependencies"))pin(ref,base);
 const auto n=p.rows.size();require(n==parent.rows.size()&&n==revision.at("raw_rows")&&n==qualification.at("raw_rows"),"complete original source denominator");
 std::map<std::size_t,eventio::Json> rows, reviewed;
 for(const auto& item:review.at("rows")){const auto i=item.at("raw_row").template get<std::size_t>();require(i<n&&reviewed.emplace(i,item).second,"duplicate reviewed source row");}
 std::map<std::int64_t,eventio::Json> frames;std::map<std::int64_t,std::size_t> counts;
 for(const auto& frame:revision.at("frames")){const auto pts=frame.at("native_pts").template get<std::int64_t>();require(frames.emplace(pts,frame).second,"duplicate source exposure");pin(frame.at("person_mask"),base);if(!frame.at("unknown_mask").is_null())pin(frame.at("unknown_mask"),base);}
 for(const auto& item:revision.at("rows")) {
  const auto i=item.at("raw_row").template get<std::size_t>();require(i<n&&rows.emplace(i,item).second&&reviewed.count(i),"distinct reviewed revised source rows");
  const auto& original=reviewed.at(i);const auto& a=parent.rows.at(i);const auto& c=p.rows.at(i);
  const auto old_uv=item.at("old_uv").template get<Vec2<double>>(),uv=item.at("current_uv").template get<Vec2<double>>();
  const auto normal=item.at("normal").template get<Vec2<double>>(),nav=item.at("navigation_uv").template get<Vec2<double>>();
  const auto profile=item.at("profile").template get<std::vector<double>>();
  require(a.active&&c.active&&a.kind==0&&c.kind==0&&!a.query&&!c.query&&a.stroke_pair_index<0&&c.stroke_pair_index<0&&
   a.alpha==0&&c.alpha==0&&a.source_row_id==item.at("source_row_id")&&a.pts==item.at("native_pts")&&
   a.uv==old_uv&&c.uv==uv&&a.material==item.at("old_material_index")&&a.path==item.at("old_path")&&
   c.material==item.at("new_material_index")&&c.path==item.at("new_path"),"exact original and current source ID, UV, clock and finite path");
  require(original.at("source_row_id")==item.at("source_row_id")&&original.at("native_pts")==item.at("native_pts")&&
   original.at("old_uv")==item.at("old_uv")&&original.at("navigation_uv")==item.at("navigation_uv")&&
   original.at("physical_identity_finding")==item.at("physical_identity_finding")&&
   original.at("physical_identity_evidence")==item.at("physical_identity_evidence"),"source reviewer and actual replay identity differ");
  const auto requested_uv=original.at("current_uv").template get<Vec2<double>>();
  const auto requested_normal=original.at("normal").template get<Vec2<double>>();
  const auto requested_profile=original.at("profile").template get<std::vector<double>>();
  require(profile.size()==5&&requested_profile.size()==5,"finite material profile width");
  for(int k=0;k<2;++k)require(std::isfinite(uv[k])&&std::isfinite(normal[k])&&std::isfinite(nav[k])&&std::abs(uv[k]-requested_uv[k])<=1e-6&&std::abs(normal[k]-requested_normal[k])<=1e-12,"current RGB mean differs from reviewed reading");
  for(int k=0;k<5;++k)require(std::isfinite(profile[k])&&std::abs(profile[k]-requested_profile[k])<=1e-6,"actual profile differs from reviewed reading");
  require(uv[0]>=4&&uv[0]<1916&&uv[1]>=4&&uv[1]<1076&&std::abs(normal[0]*normal[0]+normal[1]*normal[1]-1.)<1e-10&&
   profile[0]>=.75&&profile[1]>=4&&std::abs(profile[2])<3&&profile[3]>=0&&profile[3]<=2&&
   std::floor(profile[3])==profile[3]&&std::abs(profile[4])==1&&
   std::abs(nav[0]+profile[2]*normal[0]-uv[0])<=1e-9&&std::abs(nav[1]+profile[2]*normal[1]-uv[1])<=1e-9&&
   item.at("measurement_operator")=="material_interface"&&item.at("original_source_row_retained")==true&&
   item.at("actual_RGB_profile_replay_max_abs_px").template get<double>()<=1e-6,"current finite signed profile failed RGB replay");
  require(!item.at("physical_identity_finding").template get<std::string>().empty()&&!item.at("physical_identity_evidence").empty(),"missing physical material identity evidence");
  for(const auto& ref:item.at("physical_identity_evidence"))pin(ref,review_file.parent_path());
  const auto old_label=model.materials.at(a.material).label,label=model.materials.at(c.material).label;
  require(label==item.at("material_label")&&c.path>=0&&std::size_t(c.path)<model.materials.at(c.material).paths.size()&&
   !model.materials.at(c.material).paths.at(c.path).physical_point&&
   (a.material==c.material||(old_label.rfind("Ad",0)==0&&old_label.substr(0,old_label.find('.'))==label.substr(0,label.find('.')))),"current source changed physical surface or introduced a fixed point");
  require(frames.count(c.pts)&&frames.at(c.pts).at("source_BGR_sha256")==item.at("native_BGR_sha256")&&
   original.at("native_BGR_sha256")==item.at("native_BGR_sha256"),"exact current native RGB exposure binding");++counts[c.pts];
 }
 std::vector<std::size_t> indices;for(const auto& entry:rows)indices.push_back(entry.first);
 require(!indices.empty()&&indices==revision.at("revised_rows").template get<std::vector<std::size_t>>()&&
  indices.size()==reviewed.size()&&indices.size()==qualification.at("revised_rows"),"reviewed replacement denominator");
 for(const auto& frame:frames)require(counts[frame.first]==frame.second.at("reviewed_rows"),"complete revised per-exposure denominator");
 for(const auto& g:parent.groups)for(auto i:g.rows)if(rows.count(i))require(g.rank==0&&g.W.empty()&&g.statistic_kind=="constraint_only_unknown_noise","revised source has calibrated statistical information; separate covariance migration required");
 for(auto i:parent.source_json.at("independent_rows"))require(!rows.count(i.template get<std::size_t>()),"independent statistical row cannot be remeasured here");
 const std::set<std::string> permitted={"uv","query_uv","profile_normal","row","material_index","path"};
 for(const auto& entry:p.source_json.at("arrays").items()) {
  require(qualification.at("membership_array_sha256").at(entry.key())==entry.value().at("sha256"),"source-generation array SHA binding");pin(entry.value(),base);
  if(!permitted.count(entry.key()))require(entry.value()==parent.source_json.at("arrays").at(entry.key()),"revised source changed qualification, covariance or another operator");
 }
 for(const char* key:{"queries","stroke_pairs","independent_rows","source_station_domains","groups","dense_source_binding","native_ad_source_binding","native_ad_source_bindings","allowed_paths"})
  require(p.source_json.value(key,eventio::Json())==parent.source_json.value(key,eventio::Json()),"revised source changed inherited operators or finite path authority");
 for(std::size_t i=0;i<n;++i) {
  const auto& a=parent.rows[i];const auto& c=p.rows[i];const auto changed=rows.find(i);
  require(a.source_row_id==c.source_row_id&&a.source_dataset_id==c.source_dataset_id&&a.shot_id==c.shot_id&&a.pts==c.pts&&
   a.E==c.E&&a.kind==c.kind&&a.station==c.station&&a.alpha==c.alpha&&a.active==c.active&&a.fit_role==c.fit_role&&
   c.uv==(changed==rows.end()?a.uv:changed->second.at("current_uv").template get<Vec2<double>>())&&
   c.row==(changed==rows.end()?a.row:c.uv[1])&&c.material==(changed==rows.end()?a.material:changed->second.at("new_material_index").template get<int>())&&
   c.path==(changed==rows.end()?a.path:changed->second.at("new_path").template get<int>()),"unreviewed source or immutable measurement operator changed");
 }
 for(const char* key:{"query_uv","profile_normal"}) {
  auto before=eventio::read_array<double>(base,parent.source_json.at("arrays").at(key),"<f8");
  auto after=eventio::read_array<double>(base,p.source_json.at("arrays").at(key),"<f8");require(before.size()==2*n&&after.size()==2*n,"complete source-direction array denominator");
  for(std::size_t i=0;i<n;++i)for(int k=0;k<2;++k) {
   const auto changed=rows.find(i);const double expected=changed==rows.end()?before[2*i+k]:changed->second.at(std::string(key)=="query_uv"?"current_uv":"normal").at(k).template get<double>();
   require(after[2*i+k]==expected,"unreviewed profile direction/query mean changed");
  }
 }
 return {{"schema","rinkfit.actual_native_source_measurement_revision_consumption.v1"},{"passed",true},
  {"raw_rows",n},{"revised_rows",rows.size()},{"native_exposures",frames.size()},{"original_source_UV_preserved_in_parent",true},
  {"source_rows_deleted",0},{"source_images_changed",false},{"measured_uv_changed",true},{"source_model_projection_used",false},{"quality_accepted",false}};
}
}
