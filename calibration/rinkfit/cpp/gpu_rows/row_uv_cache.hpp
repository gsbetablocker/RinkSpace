#pragma once
#include "abi.hpp"
#include "rinkfit/event_model.hpp"
#include "rinkfit/event_packet.hpp"
#include <memory>
namespace rinkfit::gpu_rows {
struct Projection {Vec2<double> uv;std::vector<std::vector<double>> jac;std::vector<int> ids;std::array<double,2> nuisance;};
// A persistent CUDA topology/allocation. It is never a persistent theta cache.
// Every evaluate uploads every physical value and all four auxiliary primals.
// Provide CPU-authoritative strict RS roots in V. Root certification, finite
// charts, implicit RS/profile, W, domain constraints, and acceptance stay CPU.
class RowUVCache {
 public:
  RowUVCache(const EventModel&,const EventPacket&,const std::vector<std::uint32_t>& branch);
  ~RowUVCache();
  RowUVCache(const RowUVCache&)=delete;RowUVCache&operator=(const RowUVCache&)=delete;
  void evaluate(const EventModel&,const EventPacket&,const std::vector<std::uint32_t>& branch,const std::vector<double>& S,const std::vector<double>& V,const std::vector<double>& alpha,double b,const std::string& state_identity);
  Projection projection(const EventModel&,std::size_t raw_row,const EventRowLayout&,double s,double v)const;
  const UVPartial& partial(std::size_t raw_row,std::size_t local_column)const;
  std::size_t local_columns(std::size_t raw_row)const;
  std::size_t total_columns()const;
  const eventio::Json& report()const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace rinkfit::gpu_rows
