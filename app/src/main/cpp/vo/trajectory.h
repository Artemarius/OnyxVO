#ifndef ONYX_VO_TRAJECTORY_H
#define ONYX_VO_TRAJECTORY_H

#include <vector>
#include <Eigen/Core>

namespace onyx {
namespace vo {

struct TrajectoryPoint {
    Eigen::Vector3d position;
    float heading_rad = 0.0f;
    float inlier_ratio = 0.0f;
    bool is_keyframe = false;
};

class Trajectory {
public:
    explicit Trajectory(int max_positions = 2000);

    // Compose incremental (R, t) onto the world pose and record position.
    void update(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                float inlier_ratio = 0.0f, bool is_keyframe = false);
    void reset();

    const std::vector<TrajectoryPoint>& points() const { return points_; }

    // Backward-compat: extract just positions from points
    std::vector<Eigen::Vector3d> positions() const;

    const Eigen::Matrix3d& currentRotation() const { return R_world_; }
    const Eigen::Vector3d& currentPosition() const { return t_world_; }

    int keyframeCount() const { return keyframe_count_; }
    void incrementKeyframeCount() { ++keyframe_count_; }

private:
    Eigen::Matrix3d R_world_;
    Eigen::Vector3d t_world_;
    std::vector<TrajectoryPoint> points_;
    int max_positions_;
    int keyframe_count_;

    float computeHeading() const;
};

} // namespace vo
} // namespace onyx

#endif // ONYX_VO_TRAJECTORY_H
