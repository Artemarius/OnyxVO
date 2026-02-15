#include "vo/trajectory.h"
#include <cmath>

namespace onyx {
namespace vo {

Trajectory::Trajectory(int max_positions)
    : R_world_(Eigen::Matrix3d::Identity()),
      t_world_(Eigen::Vector3d::Zero()),
      head_(0),
      size_(1),
      max_positions_(max_positions),
      keyframe_count_(0) {
    ring_buf_.resize(max_positions);
    ring_buf_[0] = {t_world_, 0.0f, 0.0f, false};
}

void Trajectory::update(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                         float inlier_ratio, bool is_keyframe) {
    // Compose: T_world = T_world * T_incremental
    // t_world += R_world * t
    // R_world = R_world * R
    t_world_ += R_world_ * t;
    R_world_ = R_world_ * R;

    TrajectoryPoint pt;
    pt.position = t_world_;
    pt.heading_rad = computeHeading();
    pt.inlier_ratio = inlier_ratio;
    pt.is_keyframe = is_keyframe;

    if (size_ < max_positions_) {
        // Buffer not yet full: write at next slot after head
        int idx = (head_ + size_) % max_positions_;
        ring_buf_[idx] = pt;
        ++size_;
    } else {
        // Buffer full: overwrite oldest, advance head
        ring_buf_[head_] = pt;
        head_ = (head_ + 1) % max_positions_;
    }
}

float Trajectory::computeHeading() const {
    // Forward direction is Z column of R_world (camera looks along +Z)
    Eigen::Vector3d forward = R_world_.col(2);
    return static_cast<float>(std::atan2(forward.x(), forward.z()));
}

std::vector<TrajectoryPoint> Trajectory::points() const {
    std::vector<TrajectoryPoint> out;
    out.reserve(size_);
    for (int i = 0; i < size_; ++i) {
        out.push_back(ring_buf_[(head_ + i) % max_positions_]);
    }
    return out;
}

std::vector<Eigen::Vector3d> Trajectory::positions() const {
    std::vector<Eigen::Vector3d> pos;
    pos.reserve(size_);
    for (int i = 0; i < size_; ++i) {
        pos.push_back(ring_buf_[(head_ + i) % max_positions_].position);
    }
    return pos;
}

void Trajectory::reset() {
    R_world_ = Eigen::Matrix3d::Identity();
    t_world_ = Eigen::Vector3d::Zero();
    head_ = 0;
    size_ = 1;
    ring_buf_[0] = {t_world_, 0.0f, 0.0f, false};
    keyframe_count_ = 0;
}

} // namespace vo
} // namespace onyx
