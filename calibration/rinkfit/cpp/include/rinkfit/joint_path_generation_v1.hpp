#pragma once
#include "rinkfit/joint_packet.hpp"
#include <limits>
#include <stdexcept>
namespace rinkfit {
// Runtime layout ownership only, deliberately absent from mathematical state.
// Actual path/cell transactions advance before their first write, after all
// consumers of the previous Ceres/GPU generation have been destroyed.
inline void advance_joint_path_generation_v1(JointPacket&p){if(p.numerical_path_generation_v1==std::numeric_limits<std::uint64_t>::max())throw std::runtime_error("path graph generation exhausted");++p.numerical_path_generation_v1;}
}
