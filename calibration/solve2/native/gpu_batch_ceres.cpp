// GPU policy variant of the current common bridge; no objective or model changes.
// New block-sparse Ceres bridge. No scene, projection, noise or legacy solver code.
#include <ceres/ceres.h>
#include <ceres/evaluation_callback.h>
#include <ceres/ordered_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define RS_EXPORT extern "C" __declspec(dllexport)
using EvalFn = int (*)(void*, const double*, int64_t, int, int, double*, int64_t, double*, int64_t);
using IterFn = int (*)(void*, int, int, int, double, double, double, double, double, const double*, int64_t);
struct RsProblem {
  int32_t abi_version, nblocks, nfactors, reserved;
  int64_t nparameters, nedges;
  double* parameters;
  const int32_t* block_sizes;
  const int32_t* residual_sizes;
  const int64_t* factor_edge_offsets;
  const int32_t* edge_blocks;
  const uint8_t* constant_blocks;
  const double* lower_bounds;
  const double* upper_bounds;
  const int32_t* ordering_groups;
};
struct RsOptions {
  int32_t abi_version, max_iterations, num_threads, linear_solver;
  int32_t minimizer_progress, reserved0, reserved1, reserved2;
  double max_time_seconds, function_tolerance, gradient_tolerance, parameter_tolerance;
  double initial_trust_region_radius, max_trust_region_radius;
};
struct RsResult {
  int32_t abi_version, termination_type, solution_usable, linear_solver_used;
  int32_t iterations, successful_steps, unsuccessful_steps, residual_evaluations;
  int32_t jacobian_evaluations, batch_evaluations, batch_jacobian_evaluations, callback_failures;
  double initial_cost, final_cost, total_seconds, residual_seconds, jacobian_seconds, linear_solver_seconds;
};
static void message(char* target, int capacity, const std::string& s) {
  if (target && capacity > 0) {
    const auto n = std::min<size_t>(s.size(), static_cast<size_t>(capacity-1));
    std::memcpy(target,s.data(),n); target[n]=0;
  }
}
static int64_t checked_add(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max()-b) throw std::invalid_argument("layout overflow");
  return a+b;
}
class Batch final : public ceres::EvaluationCallback {
 public:
  const RsProblem& p;
  EvalFn fn;
  void* context;
  std::vector<int64_t> param_offsets, residual_offsets, jac_offsets;
  std::vector<double> residual, jacobian;
  bool valid=false, has_jacobian=false;
  int calls=0, jac_calls=0, failures=0;
  Batch(const RsProblem& problem, EvalFn f, void* c):p(problem),fn(f),context(c) {
    param_offsets.push_back(0); residual_offsets.push_back(0); jac_offsets.push_back(0);
    if (!fn || p.abi_version!=1 || p.nblocks<=0 || p.nfactors<=0 || p.nparameters<=0 || p.nedges<=0 ||
        !p.parameters || !p.block_sizes || !p.residual_sizes || !p.factor_edge_offsets || !p.edge_blocks)
      throw std::invalid_argument("missing buffers or unsupported ABI");
    for (int b=0;b<p.nblocks;++b) {
      if(p.block_sizes[b]<=0) throw std::invalid_argument("nonpositive parameter block");
      param_offsets.push_back(checked_add(param_offsets.back(),p.block_sizes[b]));
    }
    if(param_offsets.back()!=p.nparameters) throw std::invalid_argument("parameter buffer size mismatch");
    if(p.factor_edge_offsets[0]!=0 || p.factor_edge_offsets[p.nfactors]!=p.nedges)
      throw std::invalid_argument("factor edge offsets do not span edge array");
    for (int fidx=0;fidx<p.nfactors;++fidx) {
      const int r=p.residual_sizes[fidx];
      if(r<=0 || p.factor_edge_offsets[fidx]>=p.factor_edge_offsets[fidx+1]) throw std::invalid_argument("empty residual factor");
      if(p.factor_edge_offsets[fidx]<0 || p.factor_edge_offsets[fidx+1]>p.nedges) throw std::invalid_argument("factor edge range");
      residual_offsets.push_back(checked_add(residual_offsets.back(),r));
      std::vector<int> seen;
      for(int64_t e=p.factor_edge_offsets[fidx];e<p.factor_edge_offsets[fidx+1];++e) {
        const int b=p.edge_blocks[e];
        if(b<0 || b>=p.nblocks) throw std::invalid_argument("edge block index out of range");
        if(std::find(seen.begin(),seen.end(),b)!=seen.end()) throw std::invalid_argument("duplicate parameter block in one factor");
        seen.push_back(b);
        jac_offsets.push_back(checked_add(jac_offsets.back(),int64_t(r)*p.block_sizes[b]));
      }
    }
    residual.resize(static_cast<size_t>(residual_offsets.back()));
    jacobian.resize(static_cast<size_t>(jac_offsets.back()));
    for(int64_t i=0;i<p.nparameters;++i) {
      const double x=p.parameters[i];
      const double lo=p.lower_bounds ? p.lower_bounds[i] : -std::numeric_limits<double>::infinity();
      const double hi=p.upper_bounds ? p.upper_bounds[i] : std::numeric_limits<double>::infinity();
      if(!std::isfinite(x)||std::isnan(lo)||std::isnan(hi)||lo>hi||x<lo||x>hi)
        throw std::invalid_argument("invalid parameter or initial bound violation");
    }
    for(int b=0;b<p.nblocks;++b) {
      if(p.constant_blocks && p.constant_blocks[b]>1) throw std::invalid_argument("invalid constant flag");
      if(p.ordering_groups && p.ordering_groups[b]<0) throw std::invalid_argument("negative ordering group");
    }
  }
  void PrepareForEvaluation(bool want_jac, bool new_point) override {
    ++calls; if(want_jac) ++jac_calls;
    std::fill(residual.begin(),residual.end(),std::numeric_limits<double>::quiet_NaN());
    if(want_jac) std::fill(jacobian.begin(),jacobian.end(),std::numeric_limits<double>::quiet_NaN());
    int status=0;
    try { status=fn(context,p.parameters,p.nparameters,want_jac?1:0,new_point?1:0,
          residual.data(),residual.size(),want_jac?jacobian.data():nullptr,want_jac?jacobian.size():0); }
    catch(...) { status=0; }
    valid=status!=0 && std::all_of(residual.begin(),residual.end(),[](double v){return std::isfinite(v);});
    has_jacobian=want_jac && valid;
    if(has_jacobian) valid=std::all_of(jacobian.begin(),jacobian.end(),[](double v){return std::isfinite(v);});
    if(!valid) { has_jacobian=false; ++failures; }
  }
};
class Factor final : public ceres::CostFunction {
  Batch* batch; int factor;
 public:
  Factor(Batch* b,int f):batch(b),factor(f) {
    set_num_residuals(b->p.residual_sizes[f]);
    for(int64_t e=b->p.factor_edge_offsets[f];e<b->p.factor_edge_offsets[f+1];++e)
      mutable_parameter_block_sizes()->push_back(b->p.block_sizes[b->p.edge_blocks[e]]);
  }
  bool Evaluate(double const* const*,double* r,double** j) const override {
    if(!batch->valid || (j && !batch->has_jacobian)) return false;
    std::memcpy(r,batch->residual.data()+batch->residual_offsets[factor],sizeof(double)*num_residuals());
    if(j) for(int64_t e=batch->p.factor_edge_offsets[factor];e<batch->p.factor_edge_offsets[factor+1];++e) {
      const int local=static_cast<int>(e-batch->p.factor_edge_offsets[factor]);
      if(j[local]) std::memcpy(j[local],batch->jacobian.data()+batch->jac_offsets[e],
                              sizeof(double)*(batch->jac_offsets[e+1]-batch->jac_offsets[e]));
    }
    return true;
  }
};
class Iteration final : public ceres::IterationCallback {
  const RsProblem& p; IterFn fn; void* context;
 public:
  Iteration(const RsProblem& problem,IterFn f,void* c):p(problem),fn(f),context(c){}
  ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
    if(!fn) return ceres::SOLVER_CONTINUE;
    const int rc=fn(context,s.iteration,s.step_is_valid?1:0,s.step_is_successful?1:0,
                    s.cost,s.cost_change,s.gradient_max_norm,s.step_norm,s.trust_region_radius,
                    p.parameters,p.nparameters);
    return rc>0 ? ceres::SOLVER_TERMINATE_SUCCESSFULLY : rc<0 ? ceres::SOLVER_ABORT : ceres::SOLVER_CONTINUE;
  }
};
// A full-rank numerical chart: ambient values and bounds retain caller units.
// Identity LM damping is in these declared numerical tangent units, never an
// extra observation, a physical prior, or a reduction of parameter freedom.
class UnitChart final : public ceres::Manifold {
  std::vector<double> scales;
 public:
  UnitChart(const double* s,int n):scales(s,s+n) {
    if(n<=0 || !std::all_of(scales.begin(),scales.end(),[](double v){return std::isfinite(v)&&v>0;}))
      throw std::invalid_argument("unit chart scales must be finite and strictly positive");
  }
  int AmbientSize() const override {return static_cast<int>(scales.size());}
  int TangentSize() const override {return AmbientSize();}
  bool Plus(const double* x,const double* delta,double* y) const override {
    for(int i=0;i<AmbientSize();++i)y[i]=x[i]+scales[i]*delta[i];
    return std::all_of(y,y+AmbientSize(),[](double v){return std::isfinite(v);});
  }
  bool PlusJacobian(const double*,double* out) const override {
    const int n=AmbientSize();std::fill(out,out+int64_t(n)*n,0.0);
    for(int i=0;i<n;++i)out[int64_t(i)*n+i]=scales[i];return true;
  }
  bool RightMultiplyByPlusJacobian(const double*,int rows,const double* ambient,double* tangent) const override {
    const int n=AmbientSize();for(int r=0;r<rows;++r)for(int c=0;c<n;++c)tangent[int64_t(r)*n+c]=ambient[int64_t(r)*n+c]*scales[c];return true;
  }
  bool Minus(const double* y,const double* x,double* delta) const override {
    for(int i=0;i<AmbientSize();++i)delta[i]=(y[i]-x[i])/scales[i];return true;
  }
  bool MinusJacobian(const double*,double* out) const override {
    const int n=AmbientSize();std::fill(out,out+int64_t(n)*n,0.0);
    for(int i=0;i<n;++i)out[int64_t(i)*n+i]=1.0/scales[i];return true;
  }
};
RS_EXPORT int rs_unit_chart_abi_version() {return 1;}
RS_EXPORT int rs_batch_abi_version() { return 1; }
RS_EXPORT const char* rs_ceres_version() { return CERES_VERSION_STRING; }
static int batch_solve_impl(const RsProblem* p,const RsOptions* o,const double* scales,int64_t nscales,EvalFn eval,IterFn iteration,
                            void* context,RsResult* out,char* report,int report_capacity) {
  try {
    if(!p||!o||!out||o->abi_version!=1) throw std::invalid_argument("missing problem/options/result or options ABI");
    *out={}; out->abi_version=1;
    Batch batch(*p,eval,context);
    if((scales && nscales!=p->nparameters)||(!scales && nscales!=0))throw std::invalid_argument("unit chart parameter scale count mismatch");
    if(scales && !std::all_of(scales,scales+nscales,[](double v){return std::isfinite(v)&&v>0;}))throw std::invalid_argument("unit chart scales must be finite and strictly positive");
    ceres::Problem::Options po; po.evaluation_callback=&batch;
    ceres::Problem problem(po);
    for(int b=0;b<p->nblocks;++b) {
      double* values=p->parameters+batch.param_offsets[b];
      problem.AddParameterBlock(values,p->block_sizes[b]);
      if(scales)problem.SetManifold(values,new UnitChart(scales+batch.param_offsets[b],p->block_sizes[b]));
      if(p->constant_blocks && p->constant_blocks[b]) problem.SetParameterBlockConstant(values);
      for(int j=0;j<p->block_sizes[b];++j) {
        const int64_t index=batch.param_offsets[b]+j;
        if(p->lower_bounds && std::isfinite(p->lower_bounds[index])) problem.SetParameterLowerBound(values,j,p->lower_bounds[index]);
        if(p->upper_bounds && std::isfinite(p->upper_bounds[index])) problem.SetParameterUpperBound(values,j,p->upper_bounds[index]);
      }
    }
    for(int f=0;f<p->nfactors;++f) {
      std::vector<double*> blocks;
      for(int64_t e=p->factor_edge_offsets[f];e<p->factor_edge_offsets[f+1];++e)
        blocks.push_back(p->parameters+batch.param_offsets[p->edge_blocks[e]]);
      problem.AddResidualBlock(new Factor(&batch,f),nullptr,blocks);
    }
    ceres::Solver::Options options;
    if(o->max_iterations<=0 || o->num_threads<=0 || o->max_time_seconds<=0 ||
       o->function_tolerance<0 || o->gradient_tolerance<0 || o->parameter_tolerance<0 ||
       o->initial_trust_region_radius<=0 || o->max_trust_region_radius<o->initial_trust_region_radius)
      throw std::invalid_argument("invalid solver option");
    options.max_num_iterations=o->max_iterations; options.num_threads=o->num_threads;
    options.max_solver_time_in_seconds=o->max_time_seconds;
    options.function_tolerance=o->function_tolerance; options.gradient_tolerance=o->gradient_tolerance;
    options.parameter_tolerance=o->parameter_tolerance;
    options.initial_trust_region_radius=o->initial_trust_region_radius;
    options.max_trust_region_radius=o->max_trust_region_radius;
    if(scales) {options.jacobi_scaling=false;options.min_lm_diagonal=1.0;options.max_lm_diagonal=1.0;}
    switch(o->linear_solver) {
      case 0: options.linear_solver_type=ceres::DENSE_QR; break;
      case 1: options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY; break;
      case 2: options.linear_solver_type=ceres::SPARSE_SCHUR; break;
      case 3: options.linear_solver_type=ceres::ITERATIVE_SCHUR; options.preconditioner_type=ceres::SCHUR_JACOBI; break;
      case 4: options.linear_solver_type=ceres::CGNR; options.preconditioner_type=ceres::JACOBI; break;
      case 5: options.linear_solver_type=ceres::DENSE_SCHUR; options.dense_linear_algebra_library_type=ceres::CUDA; break;
      default: throw std::invalid_argument("unknown linear solver");
    }
    options.sparse_linear_algebra_library_type=(o->linear_solver==4 ? ceres::CUDA_SPARSE : ceres::EIGEN_SPARSE);
    options.minimizer_progress_to_stdout=o->minimizer_progress!=0;
    options.logging_type=ceres::SILENT;
    if(p->ordering_groups) {
      options.linear_solver_ordering=std::make_shared<ceres::ParameterBlockOrdering>();
      for(int b=0;b<p->nblocks;++b) options.linear_solver_ordering->AddElementToGroup(p->parameters+batch.param_offsets[b],p->ordering_groups[b]);
    }
    options.update_state_every_iteration=true;
    Iteration hook(*p,iteration,context);
    if(iteration) options.callbacks.push_back(&hook);
    std::string invalid;
    if(!options.IsValid(&invalid)) throw std::invalid_argument(invalid);
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    out->termination_type=summary.termination_type;
    out->solution_usable=summary.IsSolutionUsable()?1:0;
    out->linear_solver_used=summary.linear_solver_type_used;
    out->iterations=static_cast<int>(summary.iterations.size());
    out->successful_steps=summary.num_successful_steps; out->unsuccessful_steps=summary.num_unsuccessful_steps;
    out->residual_evaluations=summary.num_residual_evaluations;out->jacobian_evaluations=summary.num_jacobian_evaluations;
    out->batch_evaluations=batch.calls;out->batch_jacobian_evaluations=batch.jac_calls;out->callback_failures=batch.failures;
    out->initial_cost=summary.initial_cost;out->final_cost=summary.final_cost;out->total_seconds=summary.total_time_in_seconds;
    out->residual_seconds=summary.residual_evaluation_time_in_seconds;
    out->jacobian_seconds=summary.jacobian_evaluation_time_in_seconds;
    out->linear_solver_seconds=summary.linear_solver_time_in_seconds;
    message(report,report_capacity,summary.FullReport()+(scales ? "\nRinkSolve numerical metric: full-rank ambient-preserving unit tangent chart; Jacobi scaling false; min_lm_diagonal=max_lm_diagonal=1; ambient bounds unchanged.\n" : "\nRinkSolve numerical metric: public Ceres default (no unit chart).\n"));
    return 0;
  } catch(const std::exception& e) {message(report,report_capacity,e.what());return -1;}
    catch(...) {message(report,report_capacity,"unknown native bridge exception");return -2;}
}


RS_EXPORT int rs_batch_solve(const RsProblem* p,const RsOptions* o,EvalFn eval,IterFn iteration,
                            void* context,RsResult* out,char* report,int report_capacity) {
  return batch_solve_impl(p,o,nullptr,0,eval,iteration,context,out,report,report_capacity);
}
RS_EXPORT int rs_batch_solve_scaled(const RsProblem* p,const RsOptions* o,const double* scales,int64_t nscales,EvalFn eval,IterFn iteration,
                            void* context,RsResult* out,char* report,int report_capacity) {
  if(!scales){message(report,report_capacity,"explicit nonnull unit chart scales required");return -1;}
  return batch_solve_impl(p,o,scales,nscales,eval,iteration,context,out,report,report_capacity);
}

// Exact sparse J.T v for diagnostics; no observation-by-parameter dense matrix.
RS_EXPORT int rs_layout_gradient(const RsProblem* p,const double* jac,int64_t nj,
                                 const double* weights,int64_t nr,double* out,int64_t np,
                                 char* report,int capacity) {
  try {
    if(!p||p->abi_version!=1||p->nblocks<=0||p->nfactors<=0||!p->block_sizes||!p->residual_sizes||
       !p->factor_edge_offsets||!p->edge_blocks||!jac||!weights||!out||np!=p->nparameters)
      throw std::invalid_argument("gradient layout arguments");
    std::vector<int64_t> offsets(1,0);
    for(int b=0;b<p->nblocks;++b) {
      if(p->block_sizes[b]<=0)throw std::invalid_argument("gradient block size");
      offsets.push_back(checked_add(offsets.back(),p->block_sizes[b]));
    }
    if(offsets.back()!=np||p->factor_edge_offsets[0]!=0||p->factor_edge_offsets[p->nfactors]!=p->nedges)
      throw std::invalid_argument("gradient parameter/edge span");
    int64_t expected_r=0,expected_j=0;
    for(int f=0;f<p->nfactors;++f) {
      int r=p->residual_sizes[f];if(r<=0)throw std::invalid_argument("gradient residual size");
      expected_r=checked_add(expected_r,r);
      if(p->factor_edge_offsets[f]<0||p->factor_edge_offsets[f]>=p->factor_edge_offsets[f+1]||p->factor_edge_offsets[f+1]>p->nedges)
        throw std::invalid_argument("gradient edge range");
      for(int64_t e=p->factor_edge_offsets[f];e<p->factor_edge_offsets[f+1];++e) {
        int b=p->edge_blocks[e];if(b<0||b>=p->nblocks)throw std::invalid_argument("gradient block index");
        expected_j=checked_add(expected_j,int64_t(r)*p->block_sizes[b]);
      }
    }
    if(expected_r!=nr||expected_j!=nj)throw std::invalid_argument("gradient buffer length");
    std::fill(out,out+np,0.0);
    int64_t ri=0,ji=0;
    for(int f=0;f<p->nfactors;++f) {
      const int r=p->residual_sizes[f];
      for(int64_t e=p->factor_edge_offsets[f];e<p->factor_edge_offsets[f+1];++e) {
        const int b=p->edge_blocks[e],d=p->block_sizes[b];
        double* target=out+offsets[b];
        for(int row=0;row<r;++row) {
          const double weight=weights[ri+row];
          for(int col=0;col<d;++col) target[col]+=jac[ji++]*weight;
        }
      }
      ri+=r;
    }
    if(!std::all_of(out,out+np,[](double v){return std::isfinite(v);}))throw std::invalid_argument("nonfinite gradient");
    return 0;
  } catch(const std::exception& e){message(report,capacity,e.what());return -1;}
}

// Generic combinations of 2-vector block Jacobian rows. All layouts explicit.
RS_EXPORT int rs_combine_rows(int64_t count,const int64_t* rows,const int64_t* raw_edge_offsets,
  const int64_t* raw_jac_offsets,const int32_t* edge_widths,const double* raw_jac,
  const double* coefficients,const int64_t* out_offsets,const int64_t* direct_positions,
  const double* direct_values,double* out,int64_t out_count,int threads) {
  if(count<0||!rows||!raw_edge_offsets||!raw_jac_offsets||!edge_widths||!raw_jac||!coefficients||!out_offsets||!out||threads<=0)return -1;
  #pragma omp parallel for num_threads(threads)
  for(int64_t i=0;i<count;++i) {
    const int64_t row=rows[i];int64_t target=out_offsets[i];
    const double u=coefficients[2*i],v=coefficients[2*i+1];
    for(int64_t e=raw_edge_offsets[row];e<raw_edge_offsets[row+1];++e) {
      const int width=edge_widths[e];const int64_t source=raw_jac_offsets[e];
      for(int j=0;j<width;++j)out[target++]=u*raw_jac[source+j]+v*raw_jac[source+width+j];
    }
    if(direct_positions && direct_positions[i]>=0)out[direct_positions[i]]+=direct_values[i];
  }
  for(int64_t i=0;i<out_count;++i)if(!std::isfinite(out[i]))return -2;
  return 0;
}

// Block-sparse J v for derivative audits and iterative diagnostics.
RS_EXPORT int rs_layout_product(int64_t nf,const int64_t* residual_offsets,const int64_t* edge_offsets,
 const int64_t* parameter_offsets,const int64_t* jacobian_offsets,const int32_t* edge_blocks,
 const int32_t* block_sizes,const double* jac,const double* direction,double* out,int64_t nr,int threads) {
 if(nf<=0||!residual_offsets||!edge_offsets||!parameter_offsets||!jacobian_offsets||!edge_blocks||!block_sizes||!jac||!direction||!out||threads<=0)return -1;
 std::fill(out,out+nr,0.0);
 #pragma omp parallel for num_threads(threads)
 for(int64_t f=0;f<nf;++f) {
  const int r=static_cast<int>(residual_offsets[f+1]-residual_offsets[f]);
  for(int64_t e=edge_offsets[f];e<edge_offsets[f+1];++e) {
   const int b=edge_blocks[e],d=block_sizes[b];const double* j=jac+jacobian_offsets[e];const double* v=direction+parameter_offsets[b];
   for(int row=0;row<r;++row) {double value=0;for(int col=0;col<d;++col)value+=j[row*d+col]*v[col];out[residual_offsets[f]+row]+=value;}
  }
 }
 for(int64_t i=0;i<nr;++i)if(!std::isfinite(out[i]))return -2;
 return 0;
}
