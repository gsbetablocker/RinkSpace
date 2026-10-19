// Coordinate-only original-AL builder. No full-source Ceres graph is constructed.
// Requires this NON-VIRTUAL declaration in a temporary copy of joint_al.hpp:
// BuildInventory build_advertisement_coordinate_problem(ceres::Problem&,
//     const ObjectiveContext&, int ad_block,
//     std::vector<std::size_t>* affected_rows);
// No data member, virtual entry, production header, or objective is changed.
// Link against the original joint_al.cpp / PHR / model objects.
// This subset inventory MUST NOT be passed to the full-schema validate_problem.
// Owner must accept/reject the candidate with complete original AL before/after.
#include "rinkfit/joint_al.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace rinkfit {
BuildInventory JointALBuilder::build_advertisement_coordinate_problem(
    ceres::Problem& problem, const ObjectiveContext& objective, int ad_block,
    std::vector<std::size_t>* affected_rows) {
  if (problem.NumParameterBlocks() || problem.NumResidualBlocks())
    throw std::invalid_argument("advertisement coordinate builder needs an empty problem");
  if (allow_initialization_surrogate_ || phase_one_geometry_guidance_)
    throw std::invalid_argument("advertisement coordinate step requires original canonical AL");
  // The original factor factories attach this cache to their rows. A reduced
  // CPU problem must not consume stale full-graph GPU callback data.
  if (gpu_uv_)
    throw std::invalid_argument("advertisement coordinate builder requires a builder without GPU UV cache");
  if (!(objective.rho > 0.) || !std::isfinite(objective.rho)
      || objective.lambda.size() != schema_.inequality_ids.size()
      || objective.nu.size() != schema_.equality_ids.size())
    throw std::invalid_argument("advertisement coordinate original AL context mismatch");
  if (objective.phase != Phase::kFeasibility && objective.phase != Phase::kPrecision)
    throw std::invalid_argument("unsupported advertisement coordinate AL phase");
  const std::size_t m = core.model.blocks.size(), n = core.packet.rows.size();
  if (ad_block < 0 || std::size_t(ad_block) >= m)
    throw std::invalid_argument("advertisement coordinate block outside model");
  const auto target = std::size_t(ad_block);
  const auto& installation = core.model.blocks.at(target);
  if (installation.name.rfind("ad/", 0) != 0 || installation.values.size() != 4
      || installation.lower.size() != 4 || installation.upper.size() != 4
      || installation.scales.size() != 4 || installation.constant
      || !installation.fixed_indices.empty())
    throw std::invalid_argument("coordinate step needs one free standard four-value advertisement installation");
  for (int j = 0; j < 4; ++j)
    if (!std::isfinite(installation.values[j]) || !std::isfinite(installation.lower[j])
        || !std::isfinite(installation.upper[j]) || installation.lower[j] > installation.upper[j]
        || installation.values[j] < installation.lower[j] || installation.values[j] > installation.upper[j]
        || !(installation.scales[j] > 0.) || !std::isfinite(installation.scales[j]))
      throw std::invalid_argument("invalid original advertisement coordinate box/scale");
  if (target == std::size_t(core.model.scene_block))
    throw std::invalid_argument("advertisement block aliases scene");
  for (const auto& rig : core.model.rigs)
    if (ad_block == rig.head_block || ad_block == rig.optics_block)
      throw std::invalid_argument("advertisement block aliases rig");
  for (const auto& shot : core.model.shots)
    if (ad_block == shot.reference_angles_block
        || std::find(shot.control_sources.begin(), shot.control_sources.end(), ad_block) != shot.control_sources.end()
        || std::find(shot.control_blocks.begin(), shot.control_blocks.end(), ad_block) != shot.control_blocks.end())
      throw std::invalid_argument("advertisement block aliases camera control");
  for (const auto& manifold : core.model.source_json.value("linear_control_manifolds", eventio::Json::array()))
    if (manifold.at("block").get<std::size_t>() == target)
      throw std::invalid_argument("advertisement installation has a control manifold");

  std::vector<unsigned char> touched(n, 0);
  std::vector<std::size_t> selected_rows;
  std::set<std::size_t> free_ids{target};
  // No new_source_start, source-dataset, shot, or corner-only filter: all old
  // and new active observations whose ORIGINAL layout uses this installation.
  for (auto i : active_) {
    const auto& observation = core.packet.rows.at(i);
    const auto layout = observation.layout(core.model);
    if (std::find(layout.block_ids.begin(), layout.block_ids.end(), ad_block) == layout.block_ids.end())
      continue;
    touched.at(i) = 1;
    selected_rows.push_back(i);
    free_ids.insert(row_id(i));
    if (observation.kind != 1) free_ids.insert(station_id(i));
    if (observation.query) free_ids.insert(alpha_id(i));
  }
  if (selected_rows.empty())
    throw std::invalid_argument("advertisement installation has no active source-dependent rows");

  BuildInventory inventory;
  // This is a coordinate graph, not an attestation of a complete formal graph.
  inventory.exact_final_physical_graph = false;
  std::set<std::size_t> registered;
  auto depends = [&](const auto& ids) {
    for (auto id : ids) if (free_ids.count(std::size_t(id))) return true;
    return false;
  };
  auto register_factor = [&](const Factor& factor) {
    if (!depends(factor.ids))
      throw std::logic_error("coordinate builder selected a constant original factor");
    const auto factor_sizes = sizes(factor.ids);
    for (std::size_t j = 0; j < factor.ids.size(); ++j)
      if (registered.insert(factor.ids[j]).second)
        problem.AddParameterBlock(pointer(factor.ids[j]), factor_sizes[j]);
  };
  auto add_inequality = [&](Factor factor, std::size_t global_index) {
    register_factor(factor);
    auto parameters = pointers(factor);
    inventory.inequality_blocks.push_back(add_phr_inequality(problem,
        std::move(factor.cost), parameters, objective.lambda.at(global_index),
        objective.rho, objective.inequality_kink_active));
  };
  auto add_equality = [&](Factor factor, std::size_t global_index) {
    register_factor(factor);
    auto parameters = pointers(factor);
    inventory.equality_blocks.push_back(add_phr_equality(problem,
        std::move(factor.cost), parameters, objective.nu.at(global_index), objective.rho));
  };
  std::size_t inequality_index = 0;
  for (auto i : active_) {
    if (touched.at(i)) add_inequality(pixel(i), inequality_index);
    ++inequality_index;
  }
  for (std::size_t i = 0; i < physical_.size(); ++i) {
    if (depends(physical_[i].block_ids)) add_inequality(physical(i), inequality_index);
    ++inequality_index;
  }
  for (std::size_t i = 0; i < native_domains_.size(); ++i) {
    if (touched.at(native_domains_[i].row)) add_inequality(native_domain(i), inequality_index);
    ++inequality_index;
  }
  // Target is explicitly barred from scene/rig/control/zoom aliases above;
  // all zoom factors are constant in this exact coordinate restriction.
  inequality_index += zoom_bounds_.size() + 2 * zoom_enclosures_.size();
  if (inequality_index != schema_.inequality_ids.size())
    throw std::logic_error("coordinate inequality original index inventory mismatch");
  for (std::size_t i = 0; i < equalities_.size(); ++i)
    if (touched.at(equalities_[i].row)) add_equality(equality(i), i);
  for (std::size_t i = 0; i < hard_equalities_.size(); ++i)
    if (depends(hard_equalities_[i].ids)) add_equality(equality(equalities_.size() + i), equalities_.size() + i);

  // Canonical Phase I has no W. Phase II retains every column/row of each
  // original touched correlated group; unrelated group members stay constant.
  if (objective.phase == Phase::kPrecision)
    for (const auto& group : core.packet.groups) {
      if (!group.rank) continue;
      bool hit = false;
      for (auto i : group.rows) hit = hit || touched.at(i);
      if (!hit) continue;
      auto factor = statistic(group);
      register_factor(factor);
      auto parameters = pointers(factor);
      inventory.statistical_blocks.push_back(problem.AddResidualBlock(factor.cost.release(), nullptr, parameters));
    }

  for (auto id : free_ids)
    if (!registered.count(id))
      throw std::logic_error("coordinate free parameter has no original factor");
  for (auto id : registered) {
    double* value = pointer(id);
    if (!free_ids.count(id)) {
      problem.SetParameterBlockConstant(value);
      continue;
    }
    if (id == target) {
      for (int j = 0; j < 4; ++j) {
        problem.SetParameterLowerBound(value, j, installation.lower[j]);
        problem.SetParameterUpperBound(value, j, installation.upper[j]);
      }
    } else if (id >= m && id < m + n) {
      problem.SetParameterLowerBound(value, 0, 0.);
      problem.SetParameterUpperBound(value, 0, 1.);
    } else if (id >= m + n && id < m + 2 * n) {
      problem.SetParameterLowerBound(value, 0, 0.);
      problem.SetParameterUpperBound(value, 0, 1079.);
    } else if (id >= m + 2 * n && id < m + 3 * n) {
      const auto i = id - m - 2 * n;
      const auto& query = core.packet.rows.at(i).query;
      if (!query) throw std::logic_error("coordinate alpha without original frozen Query");
      problem.SetParameterLowerBound(value, 0, query->alpha.front());
      problem.SetParameterUpperBound(value, 0, query->alpha.back());
    } else {
      throw std::logic_error("unexpected free advertisement coordinate parameter");
    }
  }
  // Budget is constant: its Phase-I objective and every omitted original
  // factor contribute only an additive constant to this coordinate problem.
  if (affected_rows) *affected_rows = std::move(selected_rows);
  return inventory;
}
}  // namespace rinkfit
