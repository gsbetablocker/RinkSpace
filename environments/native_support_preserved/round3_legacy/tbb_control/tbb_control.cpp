#include <oneapi/tbb/global_control.h>
#include <cstddef>
#include <memory>
#include <mutex>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

namespace {
std::mutex control_mutex;
std::unique_ptr<tbb::global_control> control;
}

EXPORT int tbb_control_set_max_allowed_parallelism(std::size_t value) noexcept {
    try {
        std::lock_guard<std::mutex> lock(control_mutex);
        control.reset();
        if (value != 0) {
            control = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, value);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

EXPORT std::size_t tbb_control_active_max_allowed_parallelism() noexcept {
    try {
        std::lock_guard<std::mutex> lock(control_mutex);
        return tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism);
    } catch (...) {
        return 0;
    }
}
