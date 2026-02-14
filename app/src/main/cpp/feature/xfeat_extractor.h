#ifndef ONYX_VO_XFEAT_EXTRACTOR_H
#define ONYX_VO_XFEAT_EXTRACTOR_H

#include <vector>
#include <string>
#include <memory>
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

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Model binary kept alive for session lifetime
    std::vector<uint8_t> model_data_;

    // Pre-allocated buffer for 3-channel expansion (XFeat needs RGB)
    std::vector<float> rgb_buffer_;

    ModelType model_type_;
    int max_keypoints_;

    // Cached input/output info
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    std::vector<int64_t> input_shape_;
};

} // namespace feature
} // namespace onyx

#endif // ONYX_VO_XFEAT_EXTRACTOR_H
