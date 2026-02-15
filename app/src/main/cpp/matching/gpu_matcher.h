#ifndef ONYX_VO_GPU_MATCHER_H
#define ONYX_VO_GPU_MATCHER_H

#include <vector>
#include <memory>
#include <Eigen/Core>
#include "matching/cpu_matcher.h"  // for Match struct

namespace kp {
class Manager;
class Tensor;
class Algorithm;
class Sequence;
template<typename T> class TensorT;
}

namespace onyx {
namespace matching {

class GpuMatcher {
public:
    explicit GpuMatcher(int max_descriptors = 600);
    ~GpuMatcher();

    // Returns true if Vulkan/Kompute initialized successfully
    bool isAvailable() const;

    // Brute-force L2 matching on GPU with ratio test applied CPU-side.
    // desc1: [n1 x 64], desc2: [n2 x 64]
    std::vector<Match> match(
        const Eigen::MatrixXf& desc1, int n1,
        const Eigen::MatrixXf& desc2, int n2,
        float ratio_threshold = 0.8f,
        double* matching_us = nullptr);

private:
    bool available_ = false;
    int max_desc_;

    std::unique_ptr<kp::Manager> manager_;
    std::shared_ptr<kp::TensorT<float>> t_desc1_;
    std::shared_ptr<kp::TensorT<float>> t_desc2_;
    std::shared_ptr<kp::TensorT<int>>   t_match_indices_;
    std::shared_ptr<kp::TensorT<float>> t_match_distances_;
    std::shared_ptr<kp::TensorT<float>> t_second_distances_;
    std::shared_ptr<kp::Algorithm>      algorithm_;
    std::shared_ptr<kp::Sequence>       seq_;

    // Pre-allocated buffers to avoid per-frame allocation
    std::vector<Match> matches_buf_;
    std::vector<std::shared_ptr<kp::Tensor>> input_tensors_;
    std::vector<std::shared_ptr<kp::Tensor>> output_tensors_;
};

} // namespace matching
} // namespace onyx

#endif // ONYX_VO_GPU_MATCHER_H
