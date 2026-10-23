#pragma once
#include "rinkfit/joint_al.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace rinkfit {
// Source definitions remain two-dimensional latent coordinates. Only the
// runtime optimization tangent is restricted, after unchanged source proofs.
// This is a new-stage atlas choice, never a recovered identical Ceres graph.
inline eventio::Json install_joint_atlas_gauge_v1(
    JointALBuilder& b, ceres::Problem& problem, const eventio::Json& cfg) {
  eventio::Json receipt={{"schema","rinkfit.actual_runtime_atlas_gauge.v1"},
                        {"enabled",cfg.contains("runtime_atlas_gauge_v1")}};
  if(!cfg.contains("runtime_atlas_gauge_v1")) return receipt;
  const auto& policy=cfg.at("runtime_atlas_gauge_v1");
  const auto require=[](bool condition,const char* message){
    if(!condition) throw std::runtime_error(message);
  };
  require(policy.at("schema")=="rinkfit.runtime_atlas_gauge.v1" &&
          policy.value("source_model_and_source_identity_unchanged",false) &&
          policy.value("new_stage_warm_start_only",false),"untyped atlas gauge policy");
  require(cfg.contains("audit_expected_free_physical_blocks"),
          "atlas gauge requires actual scalar stage allowlist");
  // Introducing this policy requires the explicitly typed new-stage warm start.
  // A later resume of the SAME policy/graph is legitimate; its checkpoint
  // fingerprint is verified by the recovery entry, not disabled here.
  const int chart=policy.at("chart_block").get<int>();
  auto& ad=b.core.model.blocks.at(chart);
  require(ad.name==policy.at("chart_name").get<std::string>() &&
          ad.values.size()==4 && problem.HasParameterBlock(ad.values.data()) &&
          !problem.IsParameterBlockConstant(ad.values.data()) &&
          problem.ParameterBlockTangentSize(ad.values.data())==4,
          "atlas gauge must retain the variable four-coordinate installation");
  std::set<int> members;
  std::size_t rows=0;
  for(const auto& row:b.core.packet.rows) {
    if(!row.active) continue;
    const auto& path=b.core.model.materials.at(row.material).paths.at(row.path);
    if(path.advertisement_block!=chart) continue;
    const auto& definition=b.core.model.source_json.at("materials").at(row.material).at("paths").at(row.path);
    require(path.kind==2 && path.physical_point && path.geometry_block>=0 &&
            definition.at("points")==eventio::Json::array({eventio::Json::array({0,0})}),
            "atlas chart has active fixed-local or non-point evidence");
    auto& point=b.core.model.blocks.at(path.geometry_block);
    require(point.name.rfind("board_texture_shared_uv/",0)==0 &&
            point.values.size()==6 && !point.constant &&
            point.fixed_indices==std::vector<int>({0,1,3,4}),
            "source two-coordinate point definition changed");
    members.insert(path.geometry_block); ++rows;
  }
  require(rows==policy.at("active_chart_rows").get<std::size_t>() &&
          members.size()==policy.at("active_latent_blocks").get<std::size_t>() && members.size()>=2,
          "atlas active dependency denominator changed");
  std::map<int,std::pair<double,double>> extent;
  for(int coordinate:{2,5}) {
    double lo=INFINITY,hi=-INFINITY;
    for(int block:members) {
      const double value=b.core.model.blocks.at(block).values.at(coordinate);
      require(std::isfinite(value) && value>=0 && value<=1,"atlas point outside original source bounds");
      lo=std::min(lo,value);hi=std::max(hi,value);
    }
    require(hi-lo>1e-8,"atlas span cannot define an affine coordinate gauge");
    extent.emplace(coordinate,std::make_pair(lo,hi));
  }
  const auto& selected=policy.at("selected_coordinates");
  require(selected.size()==4,"atlas gauge must remove exactly four scalar freedoms");
  std::map<int,std::set<int>> additions;
  std::set<std::pair<int,int>> unique;
  std::set<std::pair<int,std::string>> extrema;
  for(const auto& item:selected) {
    const int block=item.at("block_index"),coordinate=item.at("index");
    require(members.count(block) && (coordinate==2 || coordinate==5) &&
            unique.emplace(block,coordinate).second,"invalid or duplicate atlas gauge scalar");
    auto& point=b.core.model.blocks.at(block);
    require(point.name==item.at("block_name").get<std::string>() &&
            point.values.at(coordinate)==item.at("fixed_chart_value").get<double>() &&
            !item.value("world_coordinate_measured",true),"atlas fixed value or latent meaning changed");
    const auto selection=item.at("selection").get<std::string>();
    const bool minimum=selection=="maximum_spanning_pair_minimum";
    require(minimum || selection=="maximum_spanning_pair_maximum","unknown atlas span selection");
    require(extrema.emplace(coordinate,selection).second,
            "atlas gauge has duplicate coordinate-span endpoint semantics");
    additions[block].insert(coordinate);
  }
  // The generator proves maximum span at the transition checkpoint. Other
  // latent coordinates may later cross the initial extrema without changing
  // the coordinate convention; rebuilding a graph must not freeze them.
  for(int coordinate:{2,5}) {
    double minimum=NAN,maximum=NAN;
    for(const auto& item:selected) if(item.at("index").get<int>()==coordinate) {
      const double value=item.at("fixed_chart_value");
      if(item.at("selection")=="maximum_spanning_pair_minimum") minimum=value;
      else maximum=value;
    }
    require(std::isfinite(minimum) && std::isfinite(maximum) && maximum-minimum>1e-8,
            "atlas selected coordinate span is degenerate");
  }
  int removed=0;
  for(const auto& item:additions) {
    auto& point=b.core.model.blocks.at(item.first);
    double* value=point.values.data();
    require(problem.HasParameterBlock(value) && !problem.IsParameterBlockConstant(value) &&
            problem.ParameterBlockSize(value)==6 && problem.ParameterBlockTangentSize(value)==2,
            "atlas point actual graph is not the original two-coordinate manifold");
    const auto* old=problem.GetManifold(value);
    std::vector<double> jacobian(12);
    require(old && old->PlusJacobian(value,jacobian.data()),"atlas original tangent unavailable");
    for(int row=0;row<6;++row)for(int column=0;column<2;++column)
      require(jacobian[row*2+column]==((row==2&&column==0)||(row==5&&column==1)?1.:0.),
              "atlas gauge would replace a noncanonical optimization manifold");
    std::vector<int> fixed={0,1,3,4};
    fixed.insert(fixed.end(),item.second.begin(),item.second.end());
    std::sort(fixed.begin(),fixed.end());
    problem.SetManifold(value,new ceres::SubsetManifold(6,fixed));
    if(fixed.size()==6) problem.SetParameterBlockConstant(value);
    removed+=int(item.second.size());
  }
  require(removed==4,"atlas tangent count mismatch");
  receipt["active_chart_rows"]=rows;
  receipt["active_latent_blocks"]=members.size();
  receipt["selected_coordinates"]=selected;
  receipt["physical_tangent_dimensions_removed"]=removed;
  receipt["source_model_values_flags_paths_and_source_identity_unchanged"]=true;
  receipt["ad_chart_remains_variable"]=true;
  receipt["same_problem_resume_claimed"]=false;
  receipt["actual_stage_guard_required"]=true;
  receipt["initial_maximum_span_proved_by_generator_receipt"]=true;
  return receipt;
}
}  // namespace rinkfit
