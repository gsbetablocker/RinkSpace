#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace rinkfit::joint_gpu {
using DevicePtr = std::uint64_t;
struct GPUMeta { double R0[9], n[3], b[3], t1[3], t2[3], fref, Rmax, wide_boundary, wide_coefficients[4], dt, first, refw[4]; int nc, cb, dep, ref[4]; };
struct GPUPath { int kind, geom, ad, ri, bs, begin, count; double a0, a1, h; };
struct UVRow { double t, s, v, alpha, budget; };
struct UVLayout { int global, geom, ad, ref, path, block_begin, block_count, control_begin, control_count, geometry_begin, geometry_count, geometry_offset; };
struct UVJob { int row, column; };
// Shared host storage/view type; CUDA writes this unchanged binary layout.
struct UVPartial { double u=0, v=0; int arithmetic=0, status=0; };
static_assert(std::is_standard_layout_v<UVPartial> && std::is_trivially_copyable_v<UVPartial>, "raw UV partial must remain a plain transfer type");
static_assert(offsetof(UVPartial,u)==0 && offsetof(UVPartial,v)==8 && offsetof(UVPartial,arithmetic)==16 && offsetof(UVPartial,status)==20, "raw UV partial member ABI differs");
static_assert(sizeof(GPUMeta)==304 && sizeof(GPUPath)==56 && sizeof(UVRow)==40 && sizeof(UVLayout)==48 && sizeof(UVJob)==8 && sizeof(UVPartial)==24, "Joint raw CUDA ABI differs");
struct RawArrays {
 DevicePtr offsets=0, values=0, dims=0, paths=0, points=0, rows=0, block_ids=0, controls=0, slots=0, control_offsets=0, geometry_indices=0, layouts=0, jobs=0, output=0;
};
// Loads only the installed system driver and a reviewed PTX. No CUDA SDK/link dependency.
// Context owns allocations until free() or destruction. Operations and construction/destruction
// restore the caller context; set_current() is the only explicit persistent selection.
// Transfer wall times describe the synchronous copy API call itself. Pageable
// H2D may stage bytes before DMA ends; the following raw-kernel completion sync
// covers that dependency. CUDA event time measures completed device kernel work.
struct DriverTelemetry {
 std::uint64_t h2d_bytes=0,d2h_bytes=0,h2d_calls=0,d2h_calls=0,kernel_calls=0;
 std::size_t allocated_bytes=0,peak_allocated_bytes=0;
 double h2d_completed_seconds=0,d2h_completed_seconds=0,kernel_completed_seconds=0,kernel_device_seconds=0,kernel_sync_wait_seconds=0;
};
class DriverContext {
 public:
 explicit DriverContext(const std::string& ptx_file, int device_ordinal=0);
 DriverContext(const std::string& ptx_file, int device_ordinal, const char* kernel_symbol);
 ~DriverContext();
 DriverContext(const DriverContext&)=delete;
 DriverContext& operator=(const DriverContext&)=delete;
 DevicePtr allocate(std::size_t bytes);
 void free(DevicePtr pointer);
 void upload(DevicePtr pointer, const void* source, std::size_t bytes, std::size_t offset=0);
 void download(void* destination, DevicePtr pointer, std::size_t bytes, std::size_t offset=0);
 void set_current();
 void synchronize();
 // Synchronous and blocking-sync (no busy CPU wait). Returns launch+sync seconds.
 double launch_raw(const GPUMeta&, const RawArrays&, int begin, int end, unsigned block_size=128);
 std::size_t allocated_bytes() const;
 DriverTelemetry telemetry() const;
 private:
 struct Impl;
 std::unique_ptr<Impl> impl_;
};
} // namespace rinkfit::joint_gpu
