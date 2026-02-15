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
        has_prev_features_valid_ = false;
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
        has_prev_features_valid_ = false;

        LOGI("Pipeline: matcher initialized, GPU=%s",
             gpu_available_ ? "yes" : "no (CPU fallback)");
        return gpu_available_;
    } catch (const std::exception& e) {
        LOGE("Pipeline::initMatcher GPU init failed: %s", e.what());
        gpu_available_ = false;
        use_gpu_ = false;
        matcher_ready_ = true;  // CPU matcher always available
        has_prev_features_valid_ = false;
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
// Adaptive frame skipping
// ---------------------------------------------------------------------------

void Pipeline::updateAdaptiveSkip(double frame_total_us) {
    if (!frame_skip_enabled_) return;

    // Update EMA of processing time
    if (!ema_initialized_) {
        processing_time_ema_us_ = frame_total_us;
        ema_initialized_ = true;
    } else {
        processing_time_ema_us_ = config_.frame_skip_ema_alpha * frame_total_us
                                + (1.0 - config_.frame_skip_ema_alpha) * processing_time_ema_us_;
    }

    const double threshold_us = config_.frame_budget_us * config_.frame_skip_budget_ratio;

    if (processing_time_ema_us_ > threshold_us) {
        // Over budget
        consecutive_under_budget_ = 0;
        ++consecutive_over_budget_;

        if (consecutive_over_budget_ >= config_.frame_skip_up_threshold
            && skip_interval_ < config_.frame_skip_max_interval) {
            ++skip_interval_;
            consecutive_over_budget_ = 0;  // reset after adjustment
            LOGI("Pipeline: skip interval increased to %d (EMA=%.0f us, threshold=%.0f us)",
                 skip_interval_, processing_time_ema_us_, threshold_us);
        }
    } else {
        // Under budget
        consecutive_over_budget_ = 0;
        ++consecutive_under_budget_;

        if (consecutive_under_budget_ >= config_.frame_skip_down_threshold
            && skip_interval_ > 1) {
            --skip_interval_;
            consecutive_under_budget_ = 0;  // reset after adjustment
            LOGI("Pipeline: skip interval decreased to %d (EMA=%.0f us, threshold=%.0f us)",
                 skip_interval_, processing_time_ema_us_, threshold_us);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame processing
// ---------------------------------------------------------------------------

FrameResult Pipeline::processFrame(const uint8_t* y_plane, int width, int height,
                                    int row_stride, bool use_neon) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    // --- Adaptive frame skipping: check if this frame should be skipped ---
    ++frame_counter_;
    if (skip_interval_ > 1 && (frame_counter_ % skip_interval_) != 0 && has_cached_result_) {
        // Return cached result with skip metadata updated
        cached_result_.stats.frame_skipped = true;
        cached_result_.stats.skip_interval = skip_interval_;
        return cached_result_;
    }

    // Clear reusable result — vectors keep their heap capacity from prior frames
    frame_result_.keypoints.clear();
    frame_result_.keypoint_scores.clear();
    frame_result_.keypoint_match_info.clear();
    frame_result_.match_lines.clear();
    frame_result_.trajectory_points.clear();
    frame_result_.pose_valid = false;
    frame_result_.stats = FrameStats{};

    auto frame_start = std::chrono::high_resolution_clock::now();

    // Helper: populate trajectory data + keyframe counter + skip info into result
    auto copyTrajectoryToResult = [&]() {
        if (trajectory_) {
            frame_result_.trajectory_points = trajectory_->points();
            frame_result_.stats.trajectory_count = static_cast<int>(frame_result_.trajectory_points.size());
            frame_result_.stats.keyframe_count = trajectory_->keyframeCount();
        }
        frame_result_.stats.frames_since_keyframe = frames_since_keyframe_;
        frame_result_.stats.skip_interval = skip_interval_;
        frame_result_.stats.frame_skipped = false;
    };

    // -- Stage 1: Preprocessing (resize + normalize) --------------------------

    ensurePreprocessBuffers();

    auto pp_timing = preprocessing::preprocess_frame(
        y_plane, width, height, row_stride,
        resize_buf_.get(), normalize_buf_.get(),
        config_.target_width, config_.target_height,
        config_.norm_mean, config_.norm_std,
        use_neon);

    frame_result_.stats.preprocess_us = pp_timing.total_us;

    // Budget check after preprocessing
    if (elapsedUs(frame_start) > config_.frame_budget_us) {
        frame_result_.stats.budget_exceeded = true;
        frame_result_.stats.total_us = elapsedUs(frame_start);
        copyTrajectoryToResult();
        updateAdaptiveSkip(frame_result_.stats.total_us);
        cached_result_ = frame_result_;
        has_cached_result_ = true;
        return frame_result_;
    }

    // -- Stage 2: Feature extraction ------------------------------------------

    if (!model_loaded_ || !extractor_) {
        frame_result_.stats.total_us = elapsedUs(frame_start);
        frame_result_.stats.skip_interval = skip_interval_;
        updateAdaptiveSkip(frame_result_.stats.total_us);
        cached_result_ = frame_result_;
        has_cached_result_ = true;
        return frame_result_;
    }

    auto features = extractor_->extract(
        normalize_buf_.get(), config_.target_width, config_.target_height);

    frame_result_.stats.inference_us = features.inference_us;
    frame_result_.stats.keypoint_count = features.count;

    // Move keypoints + scores for visualization (avoids copy; features still owns descriptors)
    frame_result_.keypoints = features.keypoints;      // copy — needed later via features.keypoints
    frame_result_.keypoint_scores = features.scores;    // copy — needed later via features.scores
    frame_result_.keypoint_match_info.assign(features.count, 0.0f);  // all unmatched initially

    // Budget check after extraction
    if (elapsedUs(frame_start) > config_.frame_budget_us) {
        frame_result_.stats.budget_exceeded = true;
        frame_result_.stats.total_us = elapsedUs(frame_start);
        // Still cache features for next frame
        prev_features_storage_ = std::move(features);
        has_prev_features_valid_ = true;
        copyTrajectoryToResult();
        updateAdaptiveSkip(frame_result_.stats.total_us);
        cached_result_ = frame_result_;
        has_cached_result_ = true;
        return frame_result_;
    }

    // -- Stage 3: Matching (frame-to-frame) -----------------------------------

    matches_buf_.clear();

    if (matcher_ready_ && has_prev_features_valid_ &&
        features.count > 0 && prev_features_storage_.count > 0) {

        double match_us = 0.0;

        if (use_gpu_ && gpu_matcher_ && gpu_matcher_->isAvailable()) {
            matches_buf_ = gpu_matcher_->match(
                prev_features_storage_.descriptors, prev_features_storage_.count,
                features.descriptors, features.count,
                config_.ratio_threshold, &match_us);
        } else {
            matches_buf_ = cpu_matcher_->match(
                prev_features_storage_.descriptors, prev_features_storage_.count,
                features.descriptors, features.count,
                config_.ratio_threshold, &match_us);
        }

        frame_result_.stats.matching_us = match_us;
        frame_result_.stats.match_count = static_cast<int>(matches_buf_.size());
    }

    // Budget check after matching
    if (elapsedUs(frame_start) > config_.frame_budget_us) {
        frame_result_.stats.budget_exceeded = true;
        frame_result_.stats.total_us = elapsedUs(frame_start);
        prev_features_storage_ = std::move(features);
        has_prev_features_valid_ = true;
        copyTrajectoryToResult();
        updateAdaptiveSkip(frame_result_.stats.total_us);
        cached_result_ = frame_result_;
        has_cached_result_ = true;
        return frame_result_;
    }

    // -- Stage 4: Pose estimation + trajectory --------------------------------

    vo::PoseResult pose_result;
    bool is_keyframe = false;
    float inlier_ratio = 0.0f;

    if (vo_initialized_ && estimator_ && !matches_buf_.empty()) {
        // Build matched point vectors (reuse pre-allocated buffers)
        matched_pts1_buf_.resize(matches_buf_.size());
        matched_pts2_buf_.resize(matches_buf_.size());
        for (size_t i = 0; i < matches_buf_.size(); ++i) {
            matched_pts1_buf_[i] = prev_features_storage_.keypoints[matches_buf_[i].idx1];
            matched_pts2_buf_[i] = features.keypoints[matches_buf_[i].idx2];
        }

        pose_result = estimator_->estimatePose(matched_pts1_buf_, matched_pts2_buf_);
        frame_result_.stats.pose_us = pose_result.estimation_us;
        frame_result_.stats.inlier_count = pose_result.inlier_count;
        frame_result_.pose_valid = pose_result.valid;

        if (pose_result.valid) {
            inlier_ratio = static_cast<float>(pose_result.inlier_count)
                         / static_cast<float>(matches_buf_.size());

            // Keyframe management
            if (shouldUpdateKeyframe(
                    pose_result.inlier_count,
                    static_cast<int>(matches_buf_.size()),
                    matched_pts1_buf_, matched_pts2_buf_,
                    pose_result.inlier_mask)) {
                is_keyframe = true;
                frames_since_keyframe_ = 0;
                LOGI("Pipeline: keyframe updated (inliers=%d/%d)",
                     pose_result.inlier_count, static_cast<int>(matches_buf_.size()));
            } else {
                ++frames_since_keyframe_;
            }

            trajectory_->update(pose_result.R, pose_result.t,
                                inlier_ratio, is_keyframe);
            trajectory_->incrementKeyframeCount();
        }
    }

    // -- Build enriched match lines (after pose, so we have inlier info) ------

    if (!matches_buf_.empty() && has_prev_features_valid_) {
        frame_result_.match_lines.reserve(matches_buf_.size());
        for (size_t i = 0; i < matches_buf_.size(); ++i) {
            const auto& m = matches_buf_[i];
            bool is_inlier = !pose_result.inlier_mask.empty()
                           && i < pose_result.inlier_mask.size()
                           && pose_result.inlier_mask[i];

            frame_result_.match_lines.push_back({
                prev_features_storage_.keypoints[m.idx1].x(),
                prev_features_storage_.keypoints[m.idx1].y(),
                features.keypoints[m.idx2].x(),
                features.keypoints[m.idx2].y(),
                m.ratio_quality,
                is_inlier ? 1.0f : 0.0f
            });

            // Set keypoint_match_info for the current-frame keypoint
            if (m.idx2 >= 0 && m.idx2 < static_cast<int>(frame_result_.keypoint_match_info.size())) {
                frame_result_.keypoint_match_info[m.idx2] = is_inlier
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
    frame_result_.stats.frame_quality_score = inlier_ratio * 0.6f + kp_frac * 0.4f;

    // Copy trajectory data for visualization
    copyTrajectoryToResult();

    // Cache current features for next-frame matching (move avoids copy)
    prev_features_storage_ = std::move(features);
    has_prev_features_valid_ = true;

    frame_result_.stats.total_us = elapsedUs(frame_start);

    // Update adaptive skip interval based on this frame's processing time
    updateAdaptiveSkip(frame_result_.stats.total_us);

    // Cache the result for returning during skipped frames
    cached_result_ = frame_result_;
    has_cached_result_ = true;

    return frame_result_;
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

    displacements_buf_.clear();
    displacements_buf_.reserve(pts1.size());

    for (size_t i = 0; i < pts1.size(); ++i) {
        // Only consider inliers if mask is available
        if (!inlier_mask.empty() && i < inlier_mask.size() && !inlier_mask[i]) {
            continue;
        }
        float dx = pts2[i].x() - pts1[i].x();
        float dy = pts2[i].y() - pts1[i].y();
        displacements_buf_.push_back(std::sqrt(dx * dx + dy * dy));
    }

    if (displacements_buf_.empty()) return 0.0;

    size_t mid = displacements_buf_.size() / 2;
    std::nth_element(displacements_buf_.begin(),
                     displacements_buf_.begin() + static_cast<ptrdiff_t>(mid),
                     displacements_buf_.end());
    return static_cast<double>(displacements_buf_[mid]);
}

// ---------------------------------------------------------------------------
// Runtime controls
// ---------------------------------------------------------------------------

void Pipeline::releaseComputeResources() {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    extractor_.reset();
    gpu_matcher_.reset();
    prev_features_storage_ = feature::FeatureResult{};
    estimator_.reset();

    has_prev_features_valid_ = false;
    model_loaded_ = false;
    gpu_available_ = false;
    use_gpu_ = false;
    matcher_ready_ = false;
    vo_initialized_ = false;
    frames_since_keyframe_ = 0;

    // Reset adaptive frame skipping state
    skip_interval_ = 1;
    frame_counter_ = 0;
    processing_time_ema_us_ = 0.0;
    ema_initialized_ = false;
    consecutive_over_budget_ = 0;
    consecutive_under_budget_ = 0;
    has_cached_result_ = false;

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
    has_prev_features_valid_ = false;
    frames_since_keyframe_ = 0;

    // Reset adaptive frame skipping — fresh start
    skip_interval_ = 1;
    frame_counter_ = 0;
    processing_time_ema_us_ = 0.0;
    ema_initialized_ = false;
    consecutive_over_budget_ = 0;
    consecutive_under_budget_ = 0;
    has_cached_result_ = false;

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
        has_prev_features_valid_ = false;
        prev_features_storage_ = feature::FeatureResult{};
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

void Pipeline::setFrameSkipEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    frame_skip_enabled_ = enabled;
    if (!enabled) {
        // Reset skip state so every frame is processed
        skip_interval_ = 1;
        frame_counter_ = 0;
        processing_time_ema_us_ = 0.0;
        ema_initialized_ = false;
        consecutive_over_budget_ = 0;
        consecutive_under_budget_ = 0;
    }

    LOGI("Pipeline: adaptive frame skip %s", enabled ? "enabled" : "disabled");
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
