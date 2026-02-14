#ifndef ONYX_VO_TRAJECTORY_H
#define ONYX_VO_TRAJECTORY_H

#include <vector>
#include <Eigen/Core>

namespace onyx {
namespace vo {

class Trajectory {
public:
    explicit Trajectory(int max_positions = 2000);

    // Compose incremental (R, t) onto the world pose and record position.
    void update(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    void reset();

    const std::vector<Eigen::Vector3d>& positions() const { return positions_; }
    const Eigen::Matrix3d& currentRotation() const { return R_world_; }
    const Eigen::Vector3d& currentPosition() const { return t_world_; }

    int keyframeCount() const { return keyframe_count_; }
    void incrementKeyframeCount() { ++keyframe_count_; }

private:
    Eigen::Matrix3d R_world_;
    Eigen::Vector3d t_world_;
    std::vector<Eigen::Vector3d> positions_;
    int max_positions_;
    int keyframe_count_;
};

} // namespace vo
} // namespace onyx

#endif // ONYX_VO_TRAJECTORY_H
