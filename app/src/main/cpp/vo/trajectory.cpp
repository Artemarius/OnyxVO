#include "vo/trajectory.h"

namespace onyx {
namespace vo {

Trajectory::Trajectory(int max_positions)
    : R_world_(Eigen::Matrix3d::Identity()),
      t_world_(Eigen::Vector3d::Zero()),
      max_positions_(max_positions),
      keyframe_count_(0) {
    positions_.reserve(std::min(max_positions, 512));
    positions_.push_back(t_world_);
}

void Trajectory::update(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    // Compose: T_world = T_world * T_incremental
    // t_world += R_world * t
    // R_world = R_world * R
    t_world_ += R_world_ * t;
    R_world_ = R_world_ * R;

    // Ring-buffer eviction when exceeding max
    if (static_cast<int>(positions_.size()) >= max_positions_) {
        positions_.erase(positions_.begin());
    }
    positions_.push_back(t_world_);
}

void Trajectory::reset() {
    R_world_ = Eigen::Matrix3d::Identity();
    t_world_ = Eigen::Vector3d::Zero();
    positions_.clear();
    positions_.push_back(t_world_);
    keyframe_count_ = 0;
}

} // namespace vo
} // namespace onyx
