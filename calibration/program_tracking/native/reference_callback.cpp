/*
 * Isolated GTSAM CustomFactor factory: the factor and Values arguments are
 * borrowed pybind references, avoiding std::function's implicit Values copy.
 */
#include <gtsam/nonlinear/CustomFactor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <boost/make_shared.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

namespace py = pybind11;

namespace {
struct CallbackState {
    py::function callback;
    py::object jacobian_type;
    CallbackState(py::function f, py::object type)
        : callback(std::move(f)), jacobian_type(std::move(type)) {}
};
}

PYBIND11_MODULE(reference_callback, m) {
    m.doc() = "Reference-preserving GTSAM CustomFactor factory";

    m.def("create",
          [](const gtsam::SharedNoiseModel& noise,
             py::iterable keys,
             py::function callback) {
              gtsam::KeyVector native_keys;
              for (py::handle key : keys) {
                  native_keys.push_back(py::cast<gtsam::Key>(key));
              }

              // GTSAM's JacobianVector binding is module_local, so an external
              // extension cannot py::cast its C++ pointer to that Python class.
              // Use the actual gtsam.JacobianVector class for the callback and
              // copy its matrices back into the original C++ output pointer.
              // Values and factor remain borrowed references with no copies.
              py::object jacobian_type =
                  py::module_::import("gtsam").attr("JacobianVector");
              auto state = std::shared_ptr<CallbackState>(
                  new CallbackState(std::move(callback),
                                    std::move(jacobian_type)),
                  [](CallbackState* ptr) {
                      py::gil_scoped_acquire gil;
                      delete ptr;
                  });

              gtsam::CustomErrorFunction error =
                  [state](const gtsam::CustomFactor& factor,
                          const gtsam::Values& values,
                          const gtsam::JacobianVector* H)
                      -> gtsam::Vector {
                      py::gil_scoped_acquire gil;
                      py::object factor_ref =
                          py::cast(&factor, py::return_value_policy::reference);
                      py::object values_ref =
                          py::cast(&values, py::return_value_policy::reference);
                      py::object H_ref = py::none();
                      if (H) {
                          H_ref = state->jacobian_type();
                          for (const gtsam::Matrix& matrix : *H) {
                              H_ref.attr("append")(py::cast(matrix));
                          }
                      }
                      py::object result =
                          state->callback(factor_ref, values_ref, H_ref);
                      if (H) {
                          if (static_cast<std::size_t>(py::len(H_ref)) !=
                              H->size()) {
                              throw std::runtime_error(
                                  "Callback changed JacobianVector length");
                          }
                          auto* writable =
                              const_cast<gtsam::JacobianVector*>(H);
                          for (std::size_t i = 0; i < H->size(); ++i) {
                              py::object matrix = H_ref[py::int_(i)];
                              (*writable)[i] = matrix.cast<gtsam::Matrix>();
                          }
                      }
                      return result.cast<gtsam::Vector>();
                  };

              return boost::make_shared<gtsam::CustomFactor>(
                  noise, native_keys, error);
          },
          py::arg("noise"), py::arg("keys"), py::arg("callback"));
}
