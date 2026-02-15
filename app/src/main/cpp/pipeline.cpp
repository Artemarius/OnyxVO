#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include "utils/android_log.h"
#include "utils/timer.h"
#include "preprocessing/neon_ops.h"
#include "feature/xfeat_extractor.h"
#include "matching/cpu_matcher.h"
#include "matching/gpu_matcher.h"
#include "vo/pose_estimator.h"
#include "vo/trajectory.h"

namespace onyx {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

Pipeline::Pipeline()
    : cpu_matcher_(std::make_unique<matching::CpuMatcher>())
    , trajectory_(std::make_unique<vo::Trajectory>())
{}

Pipeline::~Pipeline() = default;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void Pipeline::ensurePreprocessBuffers() {
    if (preprocess_initialized_) return;

    const int pixels = config_.target_width * config_.target_height;
    resize_buf_.reset(new uint8_t[pixels]);
    normalize_buf_.reset(new float[pixels]);
    preprocess_initialized_ = true;

    LOGI("Pipeline: preprocess buffers allocated (%dx%d)",
         config_.target_width, config_.target_height);
}

double Pipeline::elapsedUs(std::chrono::high_resolution_clock::time_point start) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now - start).count();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool Pipeline::initModel(AAssetManager* mgr, bool use_int8) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    auto model_type = use_int8
        ? feature::XFeatExtractor::ModelType::INT8
        : feature::XFeatExtractor::ModelType::FP32;

    try {
        extractor_ = std::make_unique<feature::XFeatExtractor>(
            mgr, model_type, config_.max_keypoints);
        model_loaded_ = true;
        has_prev_frame_ = false;
        LOGI("Pipeline: model initialized (%s)", extractor_->modelName());
        return true;
    } catch (const std::exception& e) {
        LOGE("Pipeline::initModel failed: %s", e.what());
        extractor_.reset();
        model_loaded_ = false;
        return false;
    }
}

bool Pipeline::initMatcher() {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    try {
        gpu_matcher_ = std::make_unique<matching::GpuMatcher>(config_.max_descriptors);
        gpu_available_ = gpu_matcher_->isAvailable();
        use_gpu_ = gpu_available_;
        matcher_ready_ = true;
        has_prev_frame_ = false;

        LOGI("Pipeline: matcher initialized, GPU=%s",
             gpu_available_ ? "yes" : "no (CPU fallback)");
        return gpu_available_;
    } catch (const std::exception& e) {
        LOGE("Pipeline::initMatcher GPU init failed: %s", e.what());
        gpu_available_ = false;
        use_gpu_ = false;
        matcher_ready_ = true;  // CPU matcher always available
        has_prev_frame_ = false;
        return false;
    }
}

void Pipeline::initVO(const Config& config) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    config_ = config;

    vo::CameraIntrinsics K;
    K.fx = config_.fx;
    K.fy = config_.fy;
    K.cx = config_.cx;
    K.cy = config_.cy;

    vo::PoseEstimator::Config pose_cfg;
    pose_cfg.ransac_iterations = config_.ransac_iterations;
    pose_cfg.inlier_threshold_px = config_.inlier_threshold_px;
    pose_cfg.min_inliers = config_.min_inliers;

    estimator_ = std::make_unique<vo::PoseEstimator>(K, pose_cfg);
    trajectory_ = std::make_unique<vo::Trajectory>();
    vo_initialized_ = true;

    LOGI("Pipeline: VO initialized (fx=%.0f fy=%.0f cx=%.0f cy=%.0f, "
         "RANSAC iters=%d thresh=%.1f min_inliers=%d)",
         K.fx, K.fy, K.cx, K.cy,
         pose_cfg.ransac_iterations, pose_cfg.inlier_threshold_px,
         pose_cfg.min_inliers);
}

// ---------------------------------------------------------------------------
// Per-frame processing
// ---------------------------------------------------------------------------

FrameResult Pipeline::processFrame(const uint8_t* y_plane, int width, int height,
                                    int row_stride, bool use_neon) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    FrameResult result;
    auto frame_start = std::chrono::high_resolution_clock::now();

    // Helper: populate trajectory data + keyframe counter into result
    auto copyTrajectoryToResult = [&]() {
        if (trajectory_) {
            result.trajectory_points = trajectory_->points();
            result.stats.trajectory_count = static_cast<int>(result.trajectory_points.size());
            result.stats.keyframe_count = trajectory_->keyframeCount();
        }
        result.stats.frames_since_keyframe = frames_since_keyframe_;
    };

    // -- Stage 1: Preprocessing (resize + normalize) --------------------------

    ensurePreprocessBuffers();

    auto pp_timing = preprocessing::preprocess_frame(
        y_plane, width, height, row_stride,
        resize_buf_.get(), normalize_buf_.get(),
        config_.target_width, config_.target_height,
        config_.norm_mean, config_.norm_std,
        use_neon);

    result.stats.preprocess_us = pp_timing.total_us;

    // Budget check after preprocessing
    if (elapsedUs(frame_start) > config_.frame_budget_us) {
        result.stats.budget_exceeded = true;
        result.stats.total_us = elapsedUs(frame_start);
        copyTrajectoryToResult();
        return result;
    }

    // -- Stage 2: Feature extraction ------------------------------------------

    if (!model_loaded_ || !extractor_) {
        result.stats.total_us = elapsedUs(frame_start);
        return result;
    }

    auto features = extractor_->extract(
        normalize_buf_.get(), config_.target_width, config_.target_height);

    result.stats.inference_us = features.inference_us;
    result.stats.keypoint_count = features.count;

    // Copy keypoints + scores for visualization
    result.keypoints = features.keypoints;
    result.keypoint_scores = features.scores;
    result.keypoint_match_info.assign(features.count, 0.0f);  // all unmatched initially

    // Budget check after extraction
    if (elapsedUs(frame_start) > config_.frame_budget_us) {
        result.stats.budget_exceeded = true;
        result.stats.total_us = elapsedUs(frame_start);
        // Still cache features for next frame
        prev_features_ = std::make_unique<feature::FeatureResult>(std::move(features));
        has_prev_frame_ = true;
        copyTrajectoryToResult();
        return result;
    }

    // -- Stage 3: Matching (frame-to-frame) -----------------------------------

    std::vector<matching::Match> matches;

    if (matcher_ready_ && has_prev_frame_ && prev_features_ &&
        features.count > 0 && prev_features_->count > 0) {

        double match_us = 0.0;

        if (use_gpu_ && gpu_matcher_ && gpu_matcher_->isAvailable()) {
            matches = gpu_matcher_->match(
                prev_features_->descriptors, prev_features_->count,
                features.descriptors, features.count,
                config_.ratio_threshold, &match_us);
        } else {
            matches = cpu_matcher_->match(
                prev_features_->descriptors, prev_features_->count,
                features.descriptors, features.count,
                config_.ratio_threshold, &match_us);
        }

        result.stats.matching_us = match_us;
        result.stats.match_count = static_cast<int>(matches.size());
    }

    // Budget check after matching
    if (elapsedUs(frame_start) > config_.frame_budget_us) {
        result.stats.budget_exceeded = true;
        result.stats.total_us = elapsedUs(frame_start);
        prev_features_ = std::make_unique<feature::FeatureResult>(std::move(features));
        has_prev_frame_ = true;
        copyTrajectoryToResult();
        return result;
    }

    // -- Stage 4: Pose estimation + trajectory --------------------------------

    vo::PoseResult pose_result;
    bool is_keyframe = false;
    float inlier_ratio = 0.0f;

    if (vo_initialized_ && estimator_ && !matches.empty()) {
        // Build matched point vectors
        std::vector<Eigen::Vector2f> matched_pts1(matches.size());
        std::vector<Eigen::Vector2f> matched_pts2(matches.size());
        for (size_t i = 0; i < matches.size(); ++i) {
            matched_pts1[i] = prev_features_->keypoints[matches[i].idx1];
            matched_pts2[i] = features.keypoints[matches[i].idx2];
        }

        pose_result = estimator_->estimatePose(matched_pts1, matched_pts2);
        result.stats.pose_us = pose_result.estimation_us;
        result.stats.inlier_count = pose_result.inlier_count;
        result.pose_valid = pose_result.valid;

        if (pose_result.valid) {
            inlier_ratio = static_cast<float>(pose_result.inlier_count)
                         / static_cast<float>(matches.size());

            // Keyframe management
            if (shouldUpdateKeyframe(
                    pose_result.inlier_count,
                    static_cast<int>(matches.size()),
                    matched_pts1, matched_pts2,
                    pose_result.inlier_mask)) {
                is_keyframe = true;
                frames_since_keyframe_ = 0;
                LOGI("Pipeline: keyframe updated (inliers=%d/%d)",
                     pose_result.inlier_count, static_cast<int>(matches.size()));
            } else {
                ++frames_since_keyframe_;
            }

            trajectory_->update(pose_result.R, pose_result.t,
                                inlier_ratio, is_keyframe);
            trajectory_->incrementKeyframeCount();
        }
    }

    // -- Build enriched match lines (after pose, so we have inlier info) ------

    if (!matches.empty() && prev_features_) {
        result.match_lines.reserve(matches.size());
        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& m = matches[i];
            bool is_inlier = !pose_result.inlier_mask.empty()
                           && i < pose_result.inlier_mask.size()
                           && pose_result.inlier_mask[i];

            result.match_lines.push_back({
                prev_features_->keypoints[m.idx1].x(),
                prev_features_->keypoints[m.idx1].y(),
                features.keypoints[m.idx2].x(),
                features.keypoints[m.idx2].y(),
                m.ratio_quality,
                is_inlier ? 1.0f : 0.0f
            });

            // Set keypoint_match_info for the current-frame keypoint
            if (m.idx2 >= 0 && m.idx2 < static_cast<int>(result.keypoint_match_info.size())) {
                result.keypoint_match_info[m.idx2] = is_inlier
                    ? m.ratio_quality
                    : -m.ratio_quality;
            }
        }
    }

    // -- Finalize -------------------------------------------------------------

    // Compute frame quality score: 60% inlier_ratio + 40% kp_count fraction
    float kp_frac = static_cast<float>(features.count)
                  / static_cast<float>(std::max(config_.max_keypoints, 1));
    kp_frac = std::min(kp_frac, 1.0f);
    result.stats.frame_quality_score = inlier_ratio * 0.6f + kp_frac * 0.4f;

    // Copy trajectory data for visualization
    copyTrajectoryToResult();

    // Cache current features for next-frame matching
    prev_features_ = std::make_unique<feature::FeatureResult>(std::move(features));
    has_prev_frame_ = true;

    result.stats.total_us = elapsedUs(frame_start);

    return result;
}

// ---------------------------------------------------------------------------
// Keyframe management
// ---------------------------------------------------------------------------

bool Pipeline::shouldUpdateKeyframe(
        int inlier_count, int match_count,
        const std::vector<Eigen::Vector2f>& pts1,
        const std::vector<Eigen::Vector2f>& pts2,
        const std::vector<bool>& inlier_mask) {

    if (match_count <= 0) return true;

    // Condition 1: inlier ratio too low
    double inlier_ratio = static_cast<double>(inlier_count) / match_count;
    if (inlier_ratio < config_.min_inlier_ratio) {
        LOGI("Pipeline: keyframe trigger — inlier ratio %.2f < %.2f",
             inlier_ratio, config_.min_inlier_ratio);
        return true;
    }

    // Condition 2: median feature displacement too large
    double median_disp = computeMedianDisplacement(pts1, pts2, inlier_mask);
    if (median_disp > config_.max_median_displacement) {
        LOGI("Pipeline: keyframe trigger — median displacement %.1f > %.1f px",
             median_disp, config_.max_median_displacement);
        return true;
    }

    return false;
}

double Pipeline::computeMedianDisplacement(
        const std::vector<Eigen::Vector2f>& pts1,
        const std::vector<Eigen::Vector2f>& pts2,
        const std::vector<bool>& inlier_mask) {

    std::vector<float> displacements;
    displacements.reserve(pts1.size());

    for (size_t i = 0; i < pts1.size(); ++i) {
        // Only consider inliers if mask is available
        if (!inlier_mask.empty() && i < inlier_mask.size() && !inlier_mask[i]) {
            continue;
        }
        float dx = pts2[i].x() - pts1[i].x();
        float dy = pts2[i].y() - pts1[i].y();
        displacements.push_back(std::sqrt(dx * dx + dy * dy));
    }

    if (displacements.empty()) return 0.0;

    size_t mid = displacements.size() / 2;
    std::nth_element(displacements.begin(),
                     displacements.begin() + static_cast<ptrdiff_t>(mid),
                     displacements.end());
    return static_cast<double>(displacements[mid]);
}

// ---------------------------------------------------------------------------
// Runtime controls
// ---------------------------------------------------------------------------

void Pipeline::releaseComputeResources() {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    extractor_.reset();
    gpu_matcher_.reset();
    prev_features_.reset();
    estimator_.reset();

    has_prev_frame_ = false;
    model_loaded_ = false;
    gpu_available_ = false;
    use_gpu_ = false;
    matcher_ready_ = false;
    vo_initialized_ = false;
    frames_since_keyframe_ = 0;

    if (trajectory_) {
        trajectory_->reset();
    }

    LOGI("Pipeline: compute resources released (pause)");
}

void Pipeline::resetTrajectory() {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    if (trajectory_) {
        trajectory_->reset();
    }
    has_prev_frame_ = false;
    frames_since_keyframe_ = 0;

    LOGI("Pipeline: trajectory reset");
}

bool Pipeline::switchModel(AAssetManager* mgr, bool use_int8) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    if (!extractor_) {
        LOGE("Pipeline::switchModel: model not initialized");
        return false;
    }

    auto model_type = use_int8
        ? feature::XFeatExtractor::ModelType::INT8
        : feature::XFeatExtractor::ModelType::FP32;

    try {
        extractor_->switchModel(mgr, model_type);
        // Invalidate previous frame cache — descriptor space may differ
        has_prev_frame_ = false;
        prev_features_.reset();
        LOGI("Pipeline: switched to model %s", extractor_->modelName());
        return true;
    } catch (const std::exception& e) {
        LOGE("Pipeline::switchModel failed: %s", e.what());
        return false;
    }
}

void Pipeline::setUseGpu(bool use_gpu) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    bool can_use_gpu = gpu_matcher_ && gpu_matcher_->isAvailable();
    use_gpu_ = use_gpu && can_use_gpu;

    LOGI("Pipeline: matcher mode = %s", use_gpu_ ? "GPU" : "CPU");
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

bool Pipeline::isGpuAvailable() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    return gpu_available_;
}

bool Pipeline::isModelLoaded() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    return model_loaded_;
}

bool Pipeline::isMatcherReady() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    return matcher_ready_;
}

} // namespace onyx
