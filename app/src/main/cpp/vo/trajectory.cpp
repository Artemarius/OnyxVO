#include "vo/trajectory.h"
#include <cmath>

namespace onyx {
namespace vo {

Trajectory::Trajectory(int max_positions)
    : R_world_(Eigen::Matrix3d::Identity()),
      t_world_(Eigen::Vector3d::Zero()),
      max_positions_(max_positions),
      keyframe_count_(0) {
    points_.reserve(std::min(max_positions, 512));
    points_.push_back({t_world_, 0.0f, 0.0f, false});
}

void Trajectory::update(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                         float inlier_ratio, bool is_keyframe) {
    // Compose: T_world = T_world * T_incremental
    // t_world += R_world * t
    // R_world = R_world * R
    t_world_ += R_world_ * t;
    R_world_ = R_world_ * R;

    // Ring-buffer eviction when exceeding max
    if (static_cast<int>(points_.size()) >= max_positions_) {
        points_.erase(points_.begin());
    }

    TrajectoryPoint pt;
    pt.position = t_world_;
    pt.heading_rad = computeHeading();
    pt.inlier_ratio = inlier_ratio;
    pt.is_keyframe = is_keyframe;
    points_.push_back(pt);
}

float Trajectory::computeHeading() const {
    // Forward direction is Z column of R_world (camera looks along +Z)
    Eigen::Vector3d forward = R_world_.col(2);
    return static_cast<float>(std::atan2(forward.x(), forward.z()));
}

std::vector<Eigen::Vector3d> Trajectory::positions() const {
    std::vector<Eigen::Vector3d> pos;
    pos.reserve(points_.size());
    for (const auto& pt : points_) {
        pos.push_back(pt.position);
    }
    return pos;
}

void Trajectory::reset() {
    R_world_ = Eigen::Matrix3d::Identity();
    t_world_ = Eigen::Vector3d::Zero();
    points_.clear();
    points_.push_back({t_world_, 0.0f, 0.0f, false});
    keyframe_count_ = 0;
}

} // namespace vo
} // namespace onyx
