#ifndef ONYX_VO_CPU_MATCHER_H
#define ONYX_VO_CPU_MATCHER_H

#include <vector>
#include <Eigen/Core>

namespace onyx {
namespace matching {

struct Match {
    int idx1;
    int idx2;
    float distance;       // L2 squared distance of best match
    float ratio_quality;  // 1.0 - sqrt(best/second), higher = more distinctive
};

class CpuMatcher {
public:
    // Brute-force L2 matching with Lowe's ratio test.
    // desc1: [n1 x 64], desc2: [n2 x 64] (L2-normalized descriptors)
    // ratio_threshold: Lowe's ratio test threshold (best/second < threshold)
    // matching_us: optional output for timing in microseconds
    std::vector<Match> match(
        const Eigen::MatrixXf& desc1, int n1,
        const Eigen::MatrixXf& desc2, int n2,
        float ratio_threshold = 0.8f,
        double* matching_us = nullptr);
};

} // namespace matching
} // namespace onyx

#endif // ONYX_VO_CPU_MATCHER_H
