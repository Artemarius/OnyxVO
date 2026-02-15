#ifndef ONYX_VO_XFEAT_EXTRACTOR_H
#define ONYX_VO_XFEAT_EXTRACTOR_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <onnxruntime_cxx_api.h>
#include <android/asset_manager.h>

namespace onyx {
namespace feature {

struct FeatureResult {
    std::vector<Eigen::Vector2f> keypoints;   // [N] pixel coords in model space
    Eigen::MatrixXf descriptors;              // [N x 64] L2-normalized
    std::vector<float> scores;                // [N] confidence
    int count = 0;                            // actual feature count
    double inference_us = 0.0;                // inference timing
};

class XFeatExtractor {
public:
    enum class ModelType { FP32, INT8 };

    // max_keypoints: upper bound on returned keypoints (clamped from model output)
    XFeatExtractor(AAssetManager* asset_mgr, ModelType type, int max_keypoints = 500);
    ~XFeatExtractor();

    // Run inference on preprocessed grayscale image.
    // image: float32 buffer of size [h * w], normalized to [0,1]
    // w, h: image dimensions (must match model input: 640x480)
    FeatureResult extract(const float* image, int w, int h);

    // Switch between FP32 and INT8 models at runtime
    void switchModel(AAssetManager* asset_mgr, ModelType type);

    ModelType currentModel() const { return model_type_; }
    const char* modelName() const { return model_type_ == ModelType::FP32 ? "FP32" : "INT8"; }

private:
    void loadModel(AAssetManager* asset_mgr, ModelType type);

    // Softmax + reshape keypoint logits [1,65,feat_h,feat_w] to full-res heatmap
    void buildFullResHeatmap(const float* kp_data, int feat_h, int feat_w);

    // Generic bilinear interpolation on a [map_h x map_w] float map
    static float bilinearSample(const float* map, int map_w, int map_h, float fx, float fy);

    std::mutex session_mutex_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Model binary kept alive for session lifetime
    std::vector<uint8_t> model_data_;

    // Pre-allocated buffer for 3-channel expansion (XFeat needs RGB)
    std::vector<float> rgb_buffer_;

    // Full-resolution detection heatmap (480*640), pre-allocated
    std::vector<float> heatmap_full_;

    ModelType model_type_;
    int max_keypoints_;

    // Cached input/output info
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    std::vector<int64_t> input_shape_;

    // Cached ORT memory info (avoid per-frame creation)
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault)};

    // Pre-allocated input shape arrays (avoid per-frame vector allocation)
    std::vector<int64_t> input_shape_1ch_ = {1, 1, 0, 0};
    std::vector<int64_t> input_shape_3ch_ = {1, 3, 0, 0};

    // Pre-allocated backbone decode buffers
    struct Candidate { int px, py; float detection; };
    struct ScoredCandidate { int px, py; float score; };
    std::vector<Candidate> candidates_buf_;
    std::vector<ScoredCandidate> scored_buf_;
};

} // namespace feature
} // namespace onyx

#endif // ONYX_VO_XFEAT_EXTRACTOR_H
