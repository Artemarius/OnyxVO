#ifndef ONYX_VO_PIPELINE_H
#define ONYX_VO_PIPELINE_H

#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <Eigen/Core>
#include <android/asset_manager.h>

// Forward declarations — avoid pulling heavyweight headers into every TU
namespace onyx {
namespace feature {
class XFeatExtractor;
struct FeatureResult;
}
namespace matching {
class GpuMatcher;
class CpuMatcher;
struct Match;
}
namespace vo {
class PoseEstimator;
class Trajectory;
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
};

// Complete per-frame output: stats + visualization data.
struct FrameResult {
    FrameStats stats;
    std::vector<Eigen::Vector2f> keypoints;            // current frame keypoints
    std::vector<std::array<float, 4>> match_lines;     // [prev_x, prev_y, curr_x, curr_y]
    std::vector<Eigen::Vector3d> trajectory_positions;
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
    };

    Pipeline();
    ~Pipeline();

    // --- Initialization (call in order) ---

    // Load XFeat ONNX model. Returns true on success.
    bool initModel(AAssetManager* mgr, bool use_int8);

    // Create GPU matcher (with CPU fallback). Returns true if GPU available.
    bool initMatcher();

    // Create pose estimator + trajectory from current config.
    void initVO(const Config& config);

    // --- Per-frame processing ---

    FrameResult processFrame(const uint8_t* y_plane, int width, int height,
                              int row_stride, bool use_neon);

    // --- Runtime controls ---

    void resetTrajectory();
    bool switchModel(AAssetManager* mgr, bool use_int8);
    void setUseGpu(bool use_gpu);

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

    // Previous frame cache (keyframe)
    std::unique_ptr<feature::FeatureResult> prev_features_;
    bool has_prev_frame_ = false;

    // Pose estimation & trajectory
    std::unique_ptr<vo::PoseEstimator> estimator_;
    std::unique_ptr<vo::Trajectory> trajectory_;
    bool vo_initialized_ = false;

    // Keyframe management helpers
    bool shouldUpdateKeyframe(int inlier_count, int match_count,
                               const std::vector<Eigen::Vector2f>& pts1,
                               const std::vector<Eigen::Vector2f>& pts2,
                               const std::vector<bool>& inlier_mask);
    double computeMedianDisplacement(const std::vector<Eigen::Vector2f>& pts1,
                                      const std::vector<Eigen::Vector2f>& pts2,
                                      const std::vector<bool>& inlier_mask);

    // Returns elapsed microseconds since the given start time.
    static double elapsedUs(std::chrono::high_resolution_clock::time_point start);

    // Thread safety — guards all mutable state
    mutable std::mutex pipeline_mutex_;
};

} // namespace onyx

#endif // ONYX_VO_PIPELINE_H
