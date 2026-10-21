#pragma once
#include "rinkfit/joint_model.hpp"
#include <set>
#include <tuple>

namespace rinkfit {
// Add containment only for endpoint coordinates consumed by an actual line path.
// Existing specifications retain their original order and IDs.
inline std::vector<EventPhysicalConstraint> joint_consumed_line_domain_specs_v1(
    const JointEventModel& model, const std::vector<EventPhysicalConstraint>& registered) {
  std::set<std::tuple<int,int,int>> covered;
  for (const auto& c : registered)
    if (c.type == 7 && c.block_ids.size() == 2 && c.block_ids[0] == model.scene_block)
      covered.emplace(c.block_ids[1], c.index, c.index + 1);
  std::vector<EventPhysicalConstraint> extra;
  for (std::size_t mi = 0; mi < model.materials.size(); ++mi) {
    const auto& material = model.materials[mi];
    for (std::size_t pi = 0; pi < material.paths.size(); ++pi) {
      const auto& path = material.paths[pi];
      if (path.kind != 0 || path.geometry_block < 0) continue;
      const int bi = path.geometry_block;
      const auto& block = model.blocks.at(bi);
      if (block.name.rfind("ice/line/", 0) != 0) continue;
      const auto key = std::make_pair(int(mi), int(pi));
      std::vector<int> indices;
      const auto found = model.path_geometry_indices.find(key);
      if (found != model.path_geometry_indices.end() && !found->second.empty()) indices = found->second;
      else {
        const auto offset = model.path_geometry_offsets.find(key);
        const int start = offset == model.path_geometry_offsets.end() ? 0 : offset->second;
        indices = {start, start + 1, start + 2, start + 3};
      }
      if (indices.size() != 4) throw std::runtime_error("finite ice line requires four consumed geometry indices");
      for (int k : {0, 2}) {
        const int ix = indices[k], iy = indices[k + 1];
        if (ix < 0 || iy != ix + 1 || iy >= int(block.values.size()))
          throw std::runtime_error("finite ice line endpoint requires an in-range consecutive XY pair");
        if (!covered.emplace(bi, ix, iy).second) continue;
        EventPhysicalConstraint c;
        c.id = block.name + "/endpoint_index" + std::to_string(ix) + "_in_rink";
        c.type = 7;
        c.index = ix;
        c.block_ids = {model.scene_block, bi};
        extra.push_back(std::move(c));
      }
    }
  }
  return extra;
}
} // namespace rinkfit
