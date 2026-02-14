#include "feature/xfeat_extractor.h"
#include "utils/android_log.h"
#include "utils/timer.h"
#include <algorithm>
#include <cstring>

namespace onyx {
namespace feature {

XFeatExtractor::XFeatExtractor(AAssetManager* asset_mgr, ModelType type,
                               int max_keypoints)
    : max_keypoints_(max_keypoints)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnyxVO");

    // Session options: optimize graph, single thread (mobile CPU)
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);

    loadModel(asset_mgr, type);
}

XFeatExtractor::~XFeatExtractor() = default;

void XFeatExtractor::loadModel(AAssetManager* asset_mgr, ModelType type) {
    model_type_ = type;

    const char* filename = (type == ModelType::FP32)
        ? "xfeat_fp32.onnx" : "xfeat_int8.onnx";

    LOGI("Loading ONNX model: %s", filename);

    // Read asset into memory
    AAsset* asset = AAssetManager_open(asset_mgr, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        LOGE("Failed to open asset: %s", filename);
        return;
    }

    off_t asset_size = AAsset_getLength(asset);
    model_data_.resize(static_cast<size_t>(asset_size));
    AAsset_read(asset, model_data_.data(), model_data_.size());
    AAsset_close(asset);

    LOGI("Model loaded: %s (%.1f MB)", filename,
         static_cast<float>(model_data_.size()) / (1024.0f * 1024.0f));

    // Create session from memory buffer
    session_ = std::make_unique<Ort::Session>(
        *env_, model_data_.data(), model_data_.size(), session_options_);

    // Cache input info
    size_t num_inputs = session_->GetInputCount();
    input_names_.clear();
    input_name_ptrs_.clear();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.emplace_back(name.get());

        auto shape = session_->GetInputTypeInfo(i)
            .GetTensorTypeAndShapeInfo().GetShape();
        LOGI("  Input[%zu]: %s [%s]", i, input_names_.back().c_str(),
             [&shape]() {
                 std::string s;
                 for (size_t j = 0; j < shape.size(); ++j) {
                     if (j > 0) s += ",";
                     s += (shape[j] < 0) ? "?" : std::to_string(shape[j]);
                 }
                 return s;
             }().c_str());
    }
    for (auto& n : input_names_) input_name_ptrs_.push_back(n.c_str());

    // Cache output info
    size_t num_outputs = session_->GetOutputCount();
    output_names_.clear();
    output_name_ptrs_.clear();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(name.get());

        auto shape = session_->GetOutputTypeInfo(i)
            .GetTensorTypeAndShapeInfo().GetShape();
        LOGI("  Output[%zu]: %s [%s]", i, output_names_.back().c_str(),
             [&shape]() {
                 std::string s;
                 for (size_t j = 0; j < shape.size(); ++j) {
                     if (j > 0) s += ",";
                     s += (shape[j] < 0) ? "?" : std::to_string(shape[j]);
                 }
                 return s;
             }().c_str());
    }
    for (auto& n : output_names_) output_name_ptrs_.push_back(n.c_str());

    // Determine input shape from model (expect [1, C, H, W])
    input_shape_ = session_->GetInputTypeInfo(0)
        .GetTensorTypeAndShapeInfo().GetShape();

    // Pre-allocate RGB buffer if model expects 3 channels
    if (input_shape_.size() == 4 && input_shape_[1] == 3) {
        int64_t h = input_shape_[2];
        int64_t w = input_shape_[3];
        rgb_buffer_.resize(static_cast<size_t>(3 * h * w));
        LOGI("  Model expects 3-channel input, allocated RGB buffer (%lldx%lld)",
             static_cast<long long>(w), static_cast<long long>(h));
    } else if (input_shape_.size() == 4 && input_shape_[1] == 1) {
        LOGI("  Model expects 1-channel input (grayscale)");
    }

    LOGI("Model ready: %s (%zu inputs, %zu outputs)", filename,
         num_inputs, num_outputs);
}

FeatureResult XFeatExtractor::extract(const float* image, int w, int h) {
    FeatureResult result;

    if (!session_) {
        LOGE("extract: no model loaded");
        return result;
    }

    double inference_us = 0.0;
    {
        ScopedTimer timer(inference_us);

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor{nullptr};
        const int pixel_count = w * h;

        if (input_shape_.size() == 4 && input_shape_[1] == 3) {
            // Model expects [1, 3, H, W]: replicate grayscale to RGB
            const size_t channel_size = static_cast<size_t>(pixel_count);
            // All three channels point to the same data (copy for contiguous layout)
            std::memcpy(rgb_buffer_.data(), image, channel_size * sizeof(float));
            std::memcpy(rgb_buffer_.data() + channel_size, image, channel_size * sizeof(float));
            std::memcpy(rgb_buffer_.data() + 2 * channel_size, image, channel_size * sizeof(float));

            std::vector<int64_t> shape = {1, 3, h, w};
            input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, rgb_buffer_.data(),
                rgb_buffer_.size(), shape.data(), shape.size());
        } else {
            // Model expects [1, 1, H, W]: zero-copy wrap of preprocessed buffer
            std::vector<int64_t> shape = {1, 1, h, w};
            input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(image),
                static_cast<size_t>(pixel_count), shape.data(), shape.size());
        }

        // Run inference
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs_.data(), &input_tensor, 1,
            output_name_ptrs_.data(), output_name_ptrs_.size());

        // Parse outputs — handle both end-to-end and backbone-only models
        // End-to-end: keypoints [N, 2], descriptors [N, 64], scores [N]
        if (output_names_.size() >= 3 && output_names_[0] == "keypoints") {
            // End-to-end model
            auto kp_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int n = static_cast<int>(kp_shape[0]);
            n = std::min(n, max_keypoints_);

            const float* kp_data = outputs[0].GetTensorData<float>();
            const float* desc_data = outputs[1].GetTensorData<float>();
            const float* score_data = outputs[2].GetTensorData<float>();

            result.keypoints.resize(n);
            result.descriptors.resize(n, 64);
            result.scores.resize(n);
            result.count = n;

            for (int i = 0; i < n; ++i) {
                result.keypoints[i] = Eigen::Vector2f(kp_data[i * 2], kp_data[i * 2 + 1]);
                result.scores[i] = score_data[i];
            }
            // Copy descriptors as a block
            if (n > 0) {
                std::memcpy(result.descriptors.data(), desc_data,
                            n * 64 * sizeof(float));
            }
        } else {
            // Backbone-only model: feats [1,64,H/8,W/8], heatmap [1,65,H/8,W/8], scores [1,1,H/8,W/8]
            // Extract keypoints from heatmap using simple argmax over 8x8 blocks
            auto feat_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int feat_h = static_cast<int>(feat_shape[2]);
            int feat_w = static_cast<int>(feat_shape[3]);

            const float* heatmap_data = outputs[1].GetTensorData<float>();
            const float* score_data = outputs[2].GetTensorData<float>();
            const float* feat_data = outputs[0].GetTensorData<float>();

            // Simple top-k from reliability scores
            struct CellScore {
                int cx, cy;
                float score;
            };
            std::vector<CellScore> cells;
            cells.reserve(feat_h * feat_w);

            for (int cy = 0; cy < feat_h; ++cy) {
                for (int cx = 0; cx < feat_w; ++cx) {
                    float s = score_data[cy * feat_w + cx];
                    cells.push_back({cx, cy, s});
                }
            }

            // Sort by score descending
            std::partial_sort(cells.begin(),
                cells.begin() + std::min(static_cast<int>(cells.size()), max_keypoints_),
                cells.end(),
                [](const CellScore& a, const CellScore& b) { return a.score > b.score; });

            int n = std::min(static_cast<int>(cells.size()), max_keypoints_);
            result.keypoints.resize(n);
            result.descriptors.resize(n, 64);
            result.scores.resize(n);
            result.count = n;

            for (int i = 0; i < n; ++i) {
                // Map cell center back to pixel coordinates
                float px = (cells[i].cx + 0.5f) * 8.0f;
                float py = (cells[i].cy + 0.5f) * 8.0f;
                result.keypoints[i] = Eigen::Vector2f(px, py);
                result.scores[i] = cells[i].score;

                // Extract 64-dim descriptor for this cell
                int idx = cells[i].cy * feat_w + cells[i].cx;
                for (int d = 0; d < 64; ++d) {
                    result.descriptors(i, d) = feat_data[d * feat_h * feat_w + idx];
                }
            }
        }
    }

    result.inference_us = inference_us;
    return result;
}

void XFeatExtractor::switchModel(AAssetManager* asset_mgr, ModelType type) {
    if (type == model_type_ && session_) {
        LOGI("Model already loaded: %s", modelName());
        return;
    }
    session_.reset();
    model_data_.clear();
    loadModel(asset_mgr, type);
}

} // namespace feature
} // namespace onyx
