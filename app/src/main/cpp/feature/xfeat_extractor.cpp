#include "feature/xfeat_extractor.h"
#include "utils/android_log.h"
#include "utils/timer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace onyx {
namespace feature {

XFeatExtractor::XFeatExtractor(AAssetManager* asset_mgr, ModelType type,
                               int max_keypoints)
    : max_keypoints_(max_keypoints)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnyxVO");

    // Session options: optimize graph, single thread (mobile CPU)
    // Note: XNNPACK EP was tested but doesn't support INT8 quantized models
    // (FusedNodeAndGraph compile failure). Default CPU EP with full graph
    // optimization is well-tuned for ARM via ONNX Runtime's internal NEON kernels.
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);

    // Pre-allocate full-res heatmap (480*640)
    heatmap_full_.resize(480 * 640, 0.0f);

    // Pre-allocate backbone decode buffers
    candidates_buf_.reserve(4096);
    scored_buf_.reserve(4096);

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
    std::lock_guard<std::mutex> lock(session_mutex_);

    if (!session_) {
        LOGE("extract: no model loaded");
        return result;
    }

    double inference_us = 0.0;
    {
        ScopedTimer timer(inference_us);

        Ort::Value input_tensor{nullptr};
        const int pixel_count = w * h;

        if (input_shape_.size() == 4 && input_shape_[1] == 3) {
            // Model expects [1, 3, H, W]: replicate grayscale to RGB
            const size_t channel_size = static_cast<size_t>(pixel_count);
            // All three channels point to the same data (copy for contiguous layout)
            std::memcpy(rgb_buffer_.data(), image, channel_size * sizeof(float));
            std::memcpy(rgb_buffer_.data() + channel_size, image, channel_size * sizeof(float));
            std::memcpy(rgb_buffer_.data() + 2 * channel_size, image, channel_size * sizeof(float));

            input_shape_3ch_[2] = h;
            input_shape_3ch_[3] = w;
            input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, rgb_buffer_.data(),
                rgb_buffer_.size(), input_shape_3ch_.data(), input_shape_3ch_.size());
        } else {
            // Model expects [1, 1, H, W]: zero-copy wrap of preprocessed buffer
            input_shape_1ch_[2] = h;
            input_shape_1ch_[3] = w;
            input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(image),
                static_cast<size_t>(pixel_count), input_shape_1ch_.data(), input_shape_1ch_.size());
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
            // Backbone-only model: feats [1,64,H/8,W/8], keypoints [1,65,H/8,W/8], reliability [1,1,H/8,W/8]
            // Proper SuperPoint-style decoding: softmax -> full-res heatmap -> NMS -> top-K
            auto feat_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int feat_h = static_cast<int>(feat_shape[2]);  // 60
            int feat_w = static_cast<int>(feat_shape[3]);  // 80
            int full_h = feat_h * 8;  // 480
            int full_w = feat_w * 8;  // 640

            const float* feat_data = outputs[0].GetTensorData<float>();       // [1,64,60,80]
            const float* kp_data   = outputs[1].GetTensorData<float>();       // [1,65,60,80]
            const float* rel_data  = outputs[2].GetTensorData<float>();       // [1,1,60,80]

            // Step 1: Softmax + reshape to full-resolution heatmap
            buildFullResHeatmap(kp_data, feat_h, feat_w);

            // Step 2: NMS (5x5 local max) + candidate collection
            constexpr int kNmsRadius = 2;  // 5x5 kernel
            constexpr float kDetThreshold = 0.05f;

            candidates_buf_.clear();

            for (int py = kNmsRadius; py < full_h - kNmsRadius; ++py) {
                for (int px = kNmsRadius; px < full_w - kNmsRadius; ++px) {
                    float val = heatmap_full_[py * full_w + px];
                    if (val <= kDetThreshold) continue;

                    // Check if local max in 5x5 neighborhood
                    bool is_max = true;
                    for (int dy = -kNmsRadius; dy <= kNmsRadius && is_max; ++dy) {
                        for (int dx = -kNmsRadius; dx <= kNmsRadius && is_max; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            if (heatmap_full_[(py + dy) * full_w + (px + dx)] >= val) {
                                is_max = false;
                            }
                        }
                    }
                    if (is_max) {
                        candidates_buf_.push_back({px, py, val});
                    }
                }
            }

            // Step 3: Score = detection * reliability (bilinear-sampled)
            scored_buf_.clear();

            for (auto& c : candidates_buf_) {
                float fx = static_cast<float>(c.px) / 8.0f;
                float fy = static_cast<float>(c.py) / 8.0f;
                float reliability = bilinearSample(rel_data, feat_w, feat_h, fx, fy);
                scored_buf_.push_back({c.px, c.py, c.detection * reliability});
            }

            // Step 4: Top-K by score
            int n = std::min(static_cast<int>(scored_buf_.size()), max_keypoints_);
            if (n > 0) {
                std::partial_sort(scored_buf_.begin(), scored_buf_.begin() + n, scored_buf_.end(),
                    [](const ScoredCandidate& a, const ScoredCandidate& b) {
                        return a.score > b.score;
                    });
            }

            result.keypoints.resize(n);
            result.descriptors.resize(n, 64);
            result.scores.resize(n);
            result.count = n;

            // Step 5: Descriptor interpolation + L2 normalize
            for (int i = 0; i < n; ++i) {
                result.keypoints[i] = Eigen::Vector2f(
                    static_cast<float>(scored_buf_[i].px),
                    static_cast<float>(scored_buf_[i].py));
                result.scores[i] = scored_buf_[i].score;

                // Bilinear-sample 64-dim descriptor from feats[0,:,feat_h,feat_w]
                float fx = static_cast<float>(scored_buf_[i].px) / 8.0f;
                float fy = static_cast<float>(scored_buf_[i].py) / 8.0f;

                float norm_sq = 0.0f;
                for (int d = 0; d < 64; ++d) {
                    const float* channel = feat_data + d * feat_h * feat_w;
                    float val = bilinearSample(channel, feat_w, feat_h, fx, fy);
                    result.descriptors(i, d) = val;
                    norm_sq += val * val;
                }

                // L2 normalize
                float inv_norm = (norm_sq > 1e-12f) ? (1.0f / std::sqrt(norm_sq)) : 0.0f;
                for (int d = 0; d < 64; ++d) {
                    result.descriptors(i, d) *= inv_norm;
                }
            }
        }
    }

    result.inference_us = inference_us;
    return result;
}

void XFeatExtractor::switchModel(AAssetManager* asset_mgr, ModelType type) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    if (type == model_type_ && session_) {
        LOGI("Model already loaded: %s", modelName());
        return;
    }
    session_.reset();
    model_data_.clear();
    loadModel(asset_mgr, type);
}

void XFeatExtractor::buildFullResHeatmap(const float* kp_data, int feat_h, int feat_w) {
    int full_w = feat_w * 8;
    int full_h = feat_h * 8;

    // Ensure buffer is large enough and zeroed
    if (static_cast<int>(heatmap_full_.size()) < full_h * full_w) {
        heatmap_full_.resize(full_h * full_w);
    }
    std::memset(heatmap_full_.data(), 0, full_h * full_w * sizeof(float));

    // For each cell in the grid: softmax across 65 channels, scatter to full-res
    for (int cy = 0; cy < feat_h; ++cy) {
        for (int cx = 0; cx < feat_w; ++cx) {
            int cell_idx = cy * feat_w + cx;

            // Gather 65 logits and find max for numerical stability
            float max_val = -1e30f;
            for (int ch = 0; ch < 65; ++ch) {
                float v = kp_data[ch * feat_h * feat_w + cell_idx];
                if (v > max_val) max_val = v;
            }

            // Compute exp and sum (all 65 channels including dustbin)
            float exp_vals[65];
            float sum_exp = 0.0f;
            for (int ch = 0; ch < 65; ++ch) {
                exp_vals[ch] = std::exp(kp_data[ch * feat_h * feat_w + cell_idx] - max_val);
                sum_exp += exp_vals[ch];
            }

            float inv_sum = 1.0f / sum_exp;

            // Scatter first 64 channels (skip dustbin ch=64) to full-res pixels
            for (int ch = 0; ch < 64; ++ch) {
                float prob = exp_vals[ch] * inv_sum;
                int dy = ch / 8;
                int dx = ch % 8;
                int py = cy * 8 + dy;
                int px = cx * 8 + dx;
                heatmap_full_[py * full_w + px] = prob;
            }
        }
    }
}

float XFeatExtractor::bilinearSample(const float* map, int map_w, int map_h,
                                      float fx, float fy) {
    // Clamp to valid range
    fx = std::max(0.0f, std::min(fx, static_cast<float>(map_w - 1)));
    fy = std::max(0.0f, std::min(fy, static_cast<float>(map_h - 1)));

    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    int x1 = std::min(x0 + 1, map_w - 1);
    int y1 = std::min(y0 + 1, map_h - 1);

    float dx = fx - static_cast<float>(x0);
    float dy = fy - static_cast<float>(y0);

    float v00 = map[y0 * map_w + x0];
    float v10 = map[y0 * map_w + x1];
    float v01 = map[y1 * map_w + x0];
    float v11 = map[y1 * map_w + x1];

    return v00 * (1.0f - dx) * (1.0f - dy)
         + v10 * dx * (1.0f - dy)
         + v01 * (1.0f - dx) * dy
         + v11 * dx * dy;
}

} // namespace feature
} // namespace onyx
