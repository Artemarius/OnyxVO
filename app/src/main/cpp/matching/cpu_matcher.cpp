#include "matching/cpu_matcher.h"
#include "utils/timer.h"
#include <cmath>
#include <limits>

namespace onyx {
namespace matching {

std::vector<Match> CpuMatcher::match(
    const Eigen::MatrixXf& desc1, int n1,
    const Eigen::MatrixXf& desc2, int n2,
    float ratio_threshold,
    double* matching_us) {

    double elapsed = 0.0;
    std::vector<Match> matches;

    {
        ScopedTimer timer(elapsed);

        if (n1 <= 0 || n2 <= 1) {
            // Need at least 2 descriptors in set 2 for ratio test
            if (matching_us) *matching_us = 0.0;
            return matches;
        }

        matches.reserve(n1);
        const float ratio_sq = ratio_threshold * ratio_threshold;

        for (int i = 0; i < n1; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            float second_dist = std::numeric_limits<float>::max();
            int best_idx = -1;

            for (int j = 0; j < n2; ++j) {
                float dist = (desc1.row(i) - desc2.row(j)).squaredNorm();

                if (dist < best_dist) {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_idx = j;
                } else if (dist < second_dist) {
                    second_dist = dist;
                }
            }

            // Lowe's ratio test (on squared distances)
            if (best_idx >= 0 && second_dist > 0.0f &&
                best_dist < ratio_sq * second_dist) {
                float rq = 1.0f - std::sqrt(best_dist / second_dist);
                matches.push_back({i, best_idx, best_dist, rq});
            }
        }
    }

    if (matching_us) *matching_us = elapsed;
    return matches;
}

} // namespace matching
} // namespace onyx
