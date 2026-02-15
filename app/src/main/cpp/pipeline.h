#ifndef ONYX_VO_PIPELINE_H
#define ONYX_VO_PIPELINE_H

#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <Eigen/Core>
#include <android/asset_manager.h>
#include "feature/xfeat_extractor.h"
#include "matching/cpu_matcher.h"
#include "vo/trajectory.h"

// Forward declarations — avoid pulling heavyweight headers into every TU
namespace onyx {
namespace feature {
class XFeatExtractor;
}  // FeatureResult: full definition needed (included via xfeat_extractor.h)
namespace matching {
class GpuMatcher;
class CpuMatcher;
struct Match;
}
namespace vo {
class PoseEstimator;
struct CameraIntrinsics;
}
}

namespace onyx {

// Per-frame timing and statistics returned to the JNI layer.
struct FrameStats {
    double preprocess_us = 0.0;
    double inference_us = 0.0;
    double matching_us = 0.0;
    double pose_us = 0.0;
    double total_us = 0.0;
    int keypoint_count = 0;
    int match_count = 0;
    int inlier_count = 0;
    int keyframe_count = 0;
    int trajectory_count = 0;
    bool budget_exceeded = false;
    int frames_since_keyframe = 0;
    float frame_quality_score = 0.0f;
    int skip_interval = 1;       // current adaptive skip interval (1 = process every frame)
    bool frame_skipped = false;  // true if this frame was skipped (returned cached result)
};

// Complete per-frame output: stats + visualization data.
struct FrameResult {
    FrameStats stats;
    std::vector<Eigen::Vector2f> keypoints;              // current frame keypoints
    std::vector<float> keypoint_scores;                  // detection scores from XFeat
    std::vector<float> keypoint_match_info;              // 0=unmatched, +q=inlier, -q=outlier
    std::vector<std::array<float, 6>> match_lines;       // [prev_x, prev_y, curr_x, curr_y, ratio_quality, is_inlier]
    std::vector<vo::TrajectoryPoint> trajectory_points;
    bool pose_valid = false;
};

// Unified frame pipeline: preprocess -> extract -> match -> pose -> trajectory.
//
// Owns all compute state (buffers, sessions, matchers, pose estimator,
// trajectory).  The JNI layer holds a single Pipeline instance and delegates
// frame processing to it.
class Pipeline {
public:
    struct Config {
        // Preprocessing
        int target_width = 640;
        int target_height = 480;
        float norm_mean = 0.0f;
        float norm_std = 255.0f;

        // Feature extraction
        int max_keypoints = 500;
        int max_descriptors = 600;

        // Matching
        float ratio_threshold = 0.8f;

        // Frame budget (microseconds). Stages are skipped once exceeded.
        double frame_budget_us = 500000.0;  // 500 ms — generous default; tighten after optimization

        // Camera intrinsics (defaults for 640x480 model resolution)
        double fx = 525.0, fy = 525.0;
        double cx = 320.0, cy = 240.0;

        // Pose estimation (RANSAC)
        int ransac_iterations = 200;
        double inlier_threshold_px = 1.5;
        int min_inliers = 15;

        // Keyframe management
        double min_inlier_ratio = 0.5;
        double max_median_displacement = 50.0;  // pixels

        // Frame skipping — adaptive skip interval based on processing time EMA
        double frame_skip_budget_ratio = 0.8;   // trigger skip increase when EMA > budget * ratio
        double frame_skip_ema_alpha = 0.15;     // EMA smoothing factor (0..1, higher = more responsive)
        int frame_skip_max_interval = 3;        // maximum skip interval (1 = no skip, 3 = process 1 in 3)
        int frame_skip_up_threshold = 5;        // consecutive over-budget frames before increasing skip
        int frame_skip_down_threshold = 8;      // consecutive under-budget frames before decreasing skip
    };

    Pipeline();
    ~Pipeline();

    // --- Initialization (call in order) ---

    // Load XFeat ONNX model. Returns true on success.
    // ep_type: 0=CPU, 1=XNNPACK, 2=NNAPI
    bool initModel(AAssetManager* mgr, bool use_int8, int ep_type = 1);

    // Create GPU matcher (with CPU fallback). Returns true if GPU available.
    bool initMatcher();

    // Create pose estimator + trajectory from current config.
    void initVO(const Config& config);

    // --- Per-frame processing ---

    FrameResult processFrame(const uint8_t* y_plane, int width, int height,
                              int row_stride, bool use_neon);

    // --- Runtime controls ---

    // Release heavyweight compute resources (ORT session, Vulkan/Kompute).
    // Called on app pause. Lightweight buffers and CPU matcher are kept.
    // Re-init via initModel() + initMatcher() + initVO() on resume.
    void releaseComputeResources();

    void resetTrajectory();
    // ep_type: 0=CPU, 1=XNNPACK, 2=NNAPI
    // Returns the actual EP in use after switch (may differ from requested due to fallback),
    // or -1 on failure.
    int switchModel(AAssetManager* mgr, bool use_int8, int ep_type = 1);
    void setUseGpu(bool use_gpu);
    void setFrameSkipEnabled(bool enabled);

    // --- Queries ---

    bool isGpuAvailable() const;
    bool isModelLoaded() const;
    bool isMatcherReady() const;

    const Config& config() const { return config_; }

private:
    Config config_;

    // Preprocessing buffers (pre-allocated at first use)
    std::unique_ptr<uint8_t[]> resize_buf_;
    std::unique_ptr<float[]> normalize_buf_;
    bool preprocess_initialized_ = false;
    void ensurePreprocessBuffers();

    // Feature extraction
    std::unique_ptr<feature::XFeatExtractor> extractor_;
    bool model_loaded_ = false;

    // Matching
    std::unique_ptr<matching::GpuMatcher> gpu_matcher_;
    std::unique_ptr<matching::CpuMatcher> cpu_matcher_;
    bool use_gpu_ = false;
    bool matcher_ready_ = false;
    bool gpu_available_ = false;

    // Previous frame cache (keyframe) — stored inline, no per-frame allocation
    feature::FeatureResult prev_features_storage_;
    bool has_prev_features_valid_ = false;

    // Pre-allocated frame result (reused each frame, returned by copy)
    FrameResult frame_result_;

    // Pre-allocated intermediate buffers (cleared and reused each frame)
    std::vector<matching::Match> matches_buf_;
    std::vector<Eigen::Vector2f> matched_pts1_buf_;
    std::vector<Eigen::Vector2f> matched_pts2_buf_;
    std::vector<float> displacements_buf_;

    // Pose estimation & trajectory
    std::unique_ptr<vo::PoseEstimator> estimator_;
    std::unique_ptr<vo::Trajectory> trajectory_;
    bool vo_initialized_ = false;
    int frames_since_keyframe_ = 0;

    // Keyframe management helpers
    bool shouldUpdateKeyframe(int inlier_count, int match_count,
                               const std::vector<Eigen::Vector2f>& pts1,
                               const std::vector<Eigen::Vector2f>& pts2,
                               const std::vector<bool>& inlier_mask);
    double computeMedianDisplacement(const std::vector<Eigen::Vector2f>& pts1,
                                      const std::vector<Eigen::Vector2f>& pts2,
                                      const std::vector<bool>& inlier_mask);

    // --- Adaptive frame skipping ---
    //
    // Tracks an EMA of frame processing time. When consistently over budget,
    // increases skip_interval_ (process 1 in N frames). Hysteresis counters
    // prevent rapid oscillation.
    int skip_interval_ = 1;                  // current skip interval (1 = process every frame)
    int frame_counter_ = 0;                  // monotonic counter, wraps are fine
    double processing_time_ema_us_ = 0.0;    // exponential moving average of total_us
    bool ema_initialized_ = false;           // first frame seeds the EMA
    int consecutive_over_budget_ = 0;        // frames where EMA > budget threshold
    int consecutive_under_budget_ = 0;       // frames where EMA <= budget threshold

    // Cached last valid result for returning during skipped frames
    FrameResult cached_result_;
    bool has_cached_result_ = false;
    bool frame_skip_enabled_ = true;

    // Update EMA and adapt skip interval after a fully processed frame.
    void updateAdaptiveSkip(double frame_total_us);

    // Returns elapsed microseconds since the given start time.
    static double elapsedUs(std::chrono::high_resolution_clock::time_point start);

    // Thread safety — guards all mutable state
    mutable std::mutex pipeline_mutex_;
};

} // namespace onyx

#endif // ONYX_VO_PIPELINE_H
