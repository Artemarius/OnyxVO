#ifndef ONYX_VO_TIMER_H
#define ONYX_VO_TIMER_H

#include <chrono>

namespace onyx {

// RAII scoped timer — writes elapsed microseconds to the referenced variable on destruction.
// Usage:
//   double elapsed_us;
//   { ScopedTimer t(elapsed_us); /* work */ }
//   // elapsed_us now contains the duration
class ScopedTimer {
public:
    explicit ScopedTimer(double& result_us)
        : result_(result_us)
        , start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        result_ = std::chrono::duration<double, std::micro>(end - start_).count();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& result_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace onyx

#endif // ONYX_VO_TIMER_H
