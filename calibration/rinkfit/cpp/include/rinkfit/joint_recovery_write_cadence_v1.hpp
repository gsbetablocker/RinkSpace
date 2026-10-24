#pragma once
#include <cmath>
#include <stdexcept>
namespace rinkfit {
// I/O scheduling only. This never changes or drops an accepted primal state.
struct JointRecoveryWriteCadenceV1 {
  double minimum_seconds=0., last_seconds=0.;
  bool written=false;
  void configure(double seconds) {
    if(!std::isfinite(seconds)||seconds<0.)
      throw std::runtime_error("checkpoint/preview interval must be finite and nonnegative");
    minimum_seconds=seconds;
  }
  bool due(double now,bool force=false) const {
    return force||!written||minimum_seconds==0.||now-last_seconds>=minimum_seconds;
  }
  void completed(double now) {last_seconds=now;written=true;}
};
}
