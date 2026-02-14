#include "vo/pose_estimator.h"
#include "utils/android_log.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

namespace onyx {
namespace vo {

// ---- CameraIntrinsics ----

Eigen::Matrix3d CameraIntrinsics::K() const {
    Eigen::Matrix3d m;
    m << fx, 0, cx,
         0, fy, cy,
         0,  0,  1;
    return m;
}

Eigen::Matrix3d CameraIntrinsics::K_inv() const {
    Eigen::Matrix3d m;
    m << 1.0/fx,      0, -cx/fx,
              0, 1.0/fy, -cy/fy,
              0,      0,      1;
    return m;
}

// ---- PoseEstimator ----

PoseEstimator::PoseEstimator(const CameraIntrinsics& intrinsics)
    : K_(intrinsics), config_(),
      rng_(std::random_device{}()) {}

PoseEstimator::PoseEstimator(const CameraIntrinsics& intrinsics,
                             const Config& config)
    : K_(intrinsics), config_(config),
      rng_(std::random_device{}()) {}

Eigen::Matrix3d PoseEstimator::normalizePoints(
    const std::vector<Eigen::Vector2d>& pts,
    std::vector<Eigen::Vector2d>& pts_norm) {

    const int n = static_cast<int>(pts.size());
    pts_norm.resize(n);

    // Compute centroid
    double cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x(); cy += p.y(); }
    cx /= n;
    cy /= n;

    // Compute mean distance from centroid
    double mean_dist = 0;
    for (const auto& p : pts) {
        mean_dist += std::sqrt((p.x() - cx) * (p.x() - cx) +
                               (p.y() - cy) * (p.y() - cy));
    }
    mean_dist /= n;

    // Scale so mean distance = sqrt(2)
    double scale = (mean_dist > 1e-10) ? std::sqrt(2.0) / mean_dist : 1.0;

    for (int i = 0; i < n; ++i) {
        pts_norm[i].x() = (pts[i].x() - cx) * scale;
        pts_norm[i].y() = (pts[i].y() - cy) * scale;
    }

    Eigen::Matrix3d T;
    T << scale,     0, -scale * cx,
             0, scale, -scale * cy,
             0,     0,           1;
    return T;
}

Eigen::Matrix3d PoseEstimator::computeFundamental8pt(
    const std::vector<Eigen::Vector2d>& pts1,
    const std::vector<Eigen::Vector2d>& pts2) {

    const int n = static_cast<int>(pts1.size());

    // Build Nx9 measurement matrix A
    // Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    Eigen::MatrixXd A(n, 9);
    for (int i = 0; i < n; ++i) {
        double x1 = pts1[i].x(), y1 = pts1[i].y();
        double x2 = pts2[i].x(), y2 = pts2[i].y();
        A(i, 0) = x2 * x1;
        A(i, 1) = x2 * y1;
        A(i, 2) = x2;
        A(i, 3) = y2 * x1;
        A(i, 4) = y2 * y1;
        A(i, 5) = y2;
        A(i, 6) = x1;
        A(i, 7) = y1;
        A(i, 8) = 1.0;
    }

    // SVD of A, F = last column of V reshaped to 3x3
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svd.matrixV().col(8);

    Eigen::Matrix3d F;
    F << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    return F;
}

Eigen::Matrix3d PoseEstimator::enforceRank2(const Eigen::Matrix3d& F) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sigma = svd.singularValues();
    sigma(2) = 0.0;  // Zero smallest singular value
    return svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();
}

double PoseEstimator::symmetricEpipolarDist(
    const Eigen::Matrix3d& F,
    const Eigen::Vector2d& p1,
    const Eigen::Vector2d& p2) {

    Eigen::Vector3d x1(p1.x(), p1.y(), 1.0);
    Eigen::Vector3d x2(p2.x(), p2.y(), 1.0);

    // Epipolar lines
    Eigen::Vector3d l2 = F * x1;       // line in image 2
    Eigen::Vector3d l1 = F.transpose() * x2;  // line in image 1

    double x2tFx1 = x2.dot(l2);
    double val = x2tFx1 * x2tFx1;

    double denom1 = l1(0) * l1(0) + l1(1) * l1(1);
    double denom2 = l2(0) * l2(0) + l2(1) * l2(1);

    if (denom1 < 1e-20 || denom2 < 1e-20) return 1e10;

    return val * (1.0 / denom1 + 1.0 / denom2);
}

Eigen::Vector3d PoseEstimator::triangulatePoint(
    const Eigen::Matrix<double, 3, 4>& P1,
    const Eigen::Matrix<double, 3, 4>& P2,
    const Eigen::Vector2d& x1,
    const Eigen::Vector2d& x2) {

    // DLT: build 4x4 system from two projection equations
    Eigen::Matrix4d A;
    A.row(0) = x1.x() * P1.row(2) - P1.row(0);
    A.row(1) = x1.y() * P1.row(2) - P1.row(1);
    A.row(2) = x2.x() * P2.row(2) - P2.row(0);
    A.row(3) = x2.y() * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);

    if (std::abs(X(3)) < 1e-15) return Eigen::Vector3d(0, 0, -1);  // at infinity
    return X.head<3>() / X(3);
}

bool PoseEstimator::decomposeEssential(
    const Eigen::Matrix3d& E,
    const std::vector<Eigen::Vector2d>& pts1,
    const std::vector<Eigen::Vector2d>& pts2,
    const std::vector<bool>& inlier_mask,
    Eigen::Matrix3d& R_out,
    Eigen::Vector3d& t_out) {

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation (det = +1)
    if (U.determinant() < 0) U.col(2) *= -1;
    if (V.determinant() < 0) V.col(2) *= -1;

    // W rotation matrix (90-degree rotation)
    Eigen::Matrix3d W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    // 4 candidate solutions
    Eigen::Matrix3d R_candidates[4];
    Eigen::Vector3d t_candidates[4];

    R_candidates[0] = U * W * V.transpose();
    t_candidates[0] = U.col(2);

    R_candidates[1] = U * W * V.transpose();
    t_candidates[1] = -U.col(2);

    R_candidates[2] = U * W.transpose() * V.transpose();
    t_candidates[2] = U.col(2);

    R_candidates[3] = U * W.transpose() * V.transpose();
    t_candidates[3] = -U.col(2);

    // Convert inlier points to normalized coordinates for triangulation
    Eigen::Matrix3d K_inv = K_.K_inv();

    // Projection matrix for camera 1 (identity)
    Eigen::Matrix<double, 3, 4> P1;
    P1.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    P1.col(3) = Eigen::Vector3d::Zero();

    int best_count = 0;
    int best_idx = -1;

    // Collect inlier indices (sample up to 50 for cheirality to keep it fast)
    std::vector<int> inlier_indices;
    for (int i = 0; i < static_cast<int>(inlier_mask.size()); ++i) {
        if (inlier_mask[i]) inlier_indices.push_back(i);
    }
    int check_count = std::min(static_cast<int>(inlier_indices.size()), 50);

    for (int c = 0; c < 4; ++c) {
        // Ensure valid rotation
        if (std::abs(R_candidates[c].determinant() - 1.0) > 0.1) continue;

        Eigen::Matrix<double, 3, 4> P2;
        P2.block<3, 3>(0, 0) = R_candidates[c];
        P2.col(3) = t_candidates[c];

        int positive_depth = 0;
        for (int j = 0; j < check_count; ++j) {
            int idx = inlier_indices[j];
            Eigen::Vector3d x1h(pts1[idx].x(), pts1[idx].y(), 1.0);
            Eigen::Vector3d x2h(pts2[idx].x(), pts2[idx].y(), 1.0);
            Eigen::Vector3d x1n = K_inv * x1h;
            Eigen::Vector3d x2n = K_inv * x2h;

            Eigen::Vector2d x1_norm(x1n.x() / x1n.z(), x1n.y() / x1n.z());
            Eigen::Vector2d x2_norm(x2n.x() / x2n.z(), x2n.y() / x2n.z());

            Eigen::Vector3d X = triangulatePoint(P1, P2, x1_norm, x2_norm);

            // Depth in camera 1
            double z1 = X.z();
            // Depth in camera 2
            Eigen::Vector3d X2 = R_candidates[c] * X + t_candidates[c];
            double z2 = X2.z();

            if (z1 > 0 && z2 > 0) positive_depth++;
        }

        if (positive_depth > best_count) {
            best_count = positive_depth;
            best_idx = c;
        }
    }

    if (best_idx < 0 || best_count < check_count / 2) {
        return false;
    }

    R_out = R_candidates[best_idx];
    t_out = t_candidates[best_idx];
    // Normalize translation to unit length (monocular: unknown scale)
    double t_norm = t_out.norm();
    if (t_norm > 1e-10) t_out /= t_norm;

    return true;
}

PoseResult PoseEstimator::estimatePose(
    const std::vector<Eigen::Vector2f>& pts1_f,
    const std::vector<Eigen::Vector2f>& pts2_f) {

    auto t_start = std::chrono::high_resolution_clock::now();

    PoseResult result;
    const int n = static_cast<int>(pts1_f.size());
    result.total_matches = n;

    // Need at least 8 correspondences for 8-point algorithm
    if (n < 8) {
        auto t_end = std::chrono::high_resolution_clock::now();
        result.estimation_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        return result;
    }

    // Convert float to double for numerical precision
    std::vector<Eigen::Vector2d> pts1(n), pts2(n);
    for (int i = 0; i < n; ++i) {
        pts1[i] = pts1_f[i].cast<double>();
        pts2[i] = pts2_f[i].cast<double>();
    }

    // Hartley normalization
    std::vector<Eigen::Vector2d> pts1_norm, pts2_norm;
    Eigen::Matrix3d T1 = normalizePoints(pts1, pts1_norm);
    Eigen::Matrix3d T2 = normalizePoints(pts2, pts2_norm);

    // Convert threshold from pixels to normalized space for RANSAC
    // The threshold is in terms of symmetric epipolar distance (squared pixel error).
    double thresh_sq = config_.inlier_threshold_px * config_.inlier_threshold_px;

    Eigen::Matrix3d best_F = Eigen::Matrix3d::Zero();
    std::vector<bool> best_inlier_mask(n, false);
    int best_inliers = 0;

    // Index array for sampling
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // RANSAC loop
    for (int iter = 0; iter < config_.ransac_iterations; ++iter) {
        // Fisher-Yates shuffle for first 8 elements
        for (int i = 0; i < 8; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            std::swap(indices[i], indices[dist(rng_)]);
        }

        // Build 8-point sample
        std::vector<Eigen::Vector2d> sample1(8), sample2(8);
        for (int i = 0; i < 8; ++i) {
            sample1[i] = pts1_norm[indices[i]];
            sample2[i] = pts2_norm[indices[i]];
        }

        // Compute fundamental matrix from normalized points
        Eigen::Matrix3d F_norm = computeFundamental8pt(sample1, sample2);
        F_norm = enforceRank2(F_norm);

        // Denormalize: F = T2^T * F_norm * T1
        Eigen::Matrix3d F = T2.transpose() * F_norm * T1;

        // Count inliers
        int inlier_count = 0;
        std::vector<bool> inlier_mask(n, false);
        for (int i = 0; i < n; ++i) {
            double d = symmetricEpipolarDist(F, pts1[i], pts2[i]);
            if (d < thresh_sq) {
                inlier_mask[i] = true;
                inlier_count++;
            }
        }

        if (inlier_count > best_inliers) {
            best_inliers = inlier_count;
            best_F = F;
            best_inlier_mask = inlier_mask;
        }
    }

    // Check if we have enough inliers
    if (best_inliers < config_.min_inliers) {
        LOGD("Pose estimation: only %d inliers (need %d)", best_inliers, config_.min_inliers);
        auto t_end = std::chrono::high_resolution_clock::now();
        result.estimation_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        result.inlier_count = best_inliers;
        result.inlier_mask = best_inlier_mask;
        return result;
    }

    // Extract essential matrix: E = K^T * F * K
    Eigen::Matrix3d K_mat = K_.K();
    Eigen::Matrix3d E = K_mat.transpose() * best_F * K_mat;

    // Decompose E into R, t via cheirality check
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    bool decomposed = decomposeEssential(E, pts1, pts2, best_inlier_mask, R, t);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.estimation_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    result.inlier_count = best_inliers;
    result.inlier_mask = best_inlier_mask;

    if (decomposed) {
        result.R = R;
        result.t = t;
        result.valid = true;
    }

    return result;
}

} // namespace vo
} // namespace onyx
