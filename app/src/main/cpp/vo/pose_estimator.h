#ifndef ONYX_VO_POSE_ESTIMATOR_H
#define ONYX_VO_POSE_ESTIMATOR_H

#include <vector>
#include <random>
#include <Eigen/Core>

namespace onyx {
namespace vo {

struct CameraIntrinsics {
    double fx, fy, cx, cy;
    Eigen::Matrix3d K() const;
    Eigen::Matrix3d K_inv() const;
};

struct PoseResult {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    int inlier_count = 0;
    int total_matches = 0;
    std::vector<bool> inlier_mask;
    double estimation_us = 0.0;
    bool valid = false;
};

class PoseEstimator {
public:
    struct Config {
        int ransac_iterations = 200;
        double inlier_threshold_px = 1.5;  // symmetric epipolar distance
        int min_inliers = 15;
    };

    explicit PoseEstimator(const CameraIntrinsics& intrinsics);
    PoseEstimator(const CameraIntrinsics& intrinsics, const Config& config);

    PoseResult estimatePose(
        const std::vector<Eigen::Vector2f>& pts1,
        const std::vector<Eigen::Vector2f>& pts2);

private:
    // Hartley isotropic normalization: shift centroid to origin,
    // scale so mean distance from origin = sqrt(2).
    // Returns the 3x3 normalization matrix T.
    static Eigen::Matrix3d normalizePoints(
        const std::vector<Eigen::Vector2d>& pts,
        std::vector<Eigen::Vector2d>& pts_norm);

    // 8-point fundamental matrix from exactly 8 correspondences
    static Eigen::Matrix3d computeFundamental8pt(
        const std::vector<Eigen::Vector2d>& pts1,
        const std::vector<Eigen::Vector2d>& pts2);

    // Enforce rank-2 constraint on F (zero smallest singular value)
    static Eigen::Matrix3d enforceRank2(const Eigen::Matrix3d& F);

    // Symmetric epipolar distance: measures how well a pair satisfies x2^T F x1 = 0
    static double symmetricEpipolarDist(
        const Eigen::Matrix3d& F,
        const Eigen::Vector2d& p1,
        const Eigen::Vector2d& p2);

    // Decompose E into 4 candidate (R, t) solutions, pick via cheirality check
    bool decomposeEssential(
        const Eigen::Matrix3d& E,
        const std::vector<Eigen::Vector2d>& pts1,
        const std::vector<Eigen::Vector2d>& pts2,
        const std::vector<bool>& inlier_mask,
        Eigen::Matrix3d& R_out,
        Eigen::Vector3d& t_out);

    // DLT linear triangulation (4x4 SVD)
    static Eigen::Vector3d triangulatePoint(
        const Eigen::Matrix<double, 3, 4>& P1,
        const Eigen::Matrix<double, 3, 4>& P2,
        const Eigen::Vector2d& x1,
        const Eigen::Vector2d& x2);

    CameraIntrinsics K_;
    Config config_;
    std::mt19937 rng_;
};

} // namespace vo
} // namespace onyx

#endif // ONYX_VO_POSE_ESTIMATOR_H
