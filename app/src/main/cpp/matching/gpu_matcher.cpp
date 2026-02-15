#include "matching/gpu_matcher.h"
#include "utils/android_log.h"
#include "utils/timer.h"
#include "match_descriptors_comp_spv.h"
#include <cmath>

#include <kompute/Kompute.hpp>

// NDK Vulkan wrapper — must call InitVulkan() to dlopen libvulkan.so
// and load function pointers before any Kompute/Vulkan API calls.
extern int InitVulkan(void);

namespace onyx {
namespace matching {

static bool isValidWorkgroupSize(uint32_t size) {
    return size == 64 || size == 128 || size == 256;
}

GpuMatcher::GpuMatcher(int max_descriptors, uint32_t workgroup_size)
    : max_desc_(max_descriptors)
    , workgroup_size_(isValidWorkgroupSize(workgroup_size) ? workgroup_size : 256) {

    if (!isValidWorkgroupSize(workgroup_size)) {
        LOGW("GpuMatcher: invalid workgroup size %u, falling back to 256. "
             "Valid values: 64, 128, 256", workgroup_size);
    }

    try {
        // Load Vulkan function pointers via NDK wrapper before any Vulkan calls.
        // Without this, vkCreateInstance is a null pointer -> SIGSEGV.
        if (!InitVulkan()) {
            LOGW("GpuMatcher: Vulkan not available (InitVulkan failed)");
            available_ = false;
            return;
        }

        manager_ = std::make_unique<kp::Manager>();

        auto props = manager_->getDeviceProperties();
        LOGI("GpuMatcher: Vulkan device: %s", props.deviceName);

        // Pre-allocate tensors at max size
        const int desc_buf_size = max_desc_ * 64;

        std::vector<float> zeros_f(desc_buf_size, 0.0f);
        std::vector<int>   zeros_i(max_desc_, -1);
        std::vector<float> zeros_d(max_desc_, 0.0f);

        t_desc1_            = manager_->tensorT<float>(zeros_f);
        t_desc2_            = manager_->tensorT<float>(zeros_f);
        t_match_indices_    = manager_->tensorT<int>(zeros_i);
        t_match_distances_  = manager_->tensorT<float>(zeros_d);
        t_second_distances_ = manager_->tensorT<float>(zeros_d);

        // Create algorithm with push constants [n1, n2] as uint32_t
        // and specialization constant [workgroup_size] as uint32_t
        std::vector<std::shared_ptr<kp::Tensor>> tensors = {
            t_desc1_, t_desc2_,
            t_match_indices_, t_match_distances_, t_second_distances_
        };

        std::vector<uint32_t> push_consts = {0, 0};  // placeholder
        kp::Workgroup workgroup = {1, 1, 1};  // will be set per-dispatch

        // Specialization constant id=0: workgroup local_size_x
        std::vector<uint32_t> spec_consts = { workgroup_size_ };

        algorithm_ = manager_->algorithm<uint32_t, uint32_t>(
            tensors,
            match_descriptors_comp_spv,
            workgroup,
            spec_consts,
            push_consts
        );

        seq_ = manager_->sequence();

        // Pre-allocate tensor lists for sequence recording (avoids per-frame vector alloc)
        input_tensors_ = { t_desc1_, t_desc2_ };
        output_tensors_ = { t_match_indices_, t_match_distances_, t_second_distances_ };

        available_ = true;
        LOGI("GpuMatcher: initialized (max_desc=%d, workgroup_size=%u)",
             max_desc_, workgroup_size_);
    } catch (const std::exception& e) {
        LOGW("GpuMatcher: Vulkan init failed: %s — will use CPU fallback", e.what());
        available_ = false;
    }
}

GpuMatcher::~GpuMatcher() {
    if (manager_) {
        manager_->destroy();
    }
}

bool GpuMatcher::isAvailable() const {
    return available_;
}

uint32_t GpuMatcher::workgroupSize() const {
    return workgroup_size_;
}

std::vector<Match> GpuMatcher::match(
    const Eigen::MatrixXf& desc1, int n1,
    const Eigen::MatrixXf& desc2, int n2,
    float ratio_threshold,
    double* matching_us) {

    double elapsed = 0.0;
    matches_buf_.clear();

    if (!available_ || n1 <= 0 || n2 <= 1) {
        if (matching_us) *matching_us = 0.0;
        return matches_buf_;
    }

    if (n1 > max_desc_ || n2 > max_desc_) {
        LOGW("GpuMatcher: descriptor count exceeds max (%d/%d > %d), clamping",
             n1, n2, max_desc_);
        n1 = std::min(n1, max_desc_);
        n2 = std::min(n2, max_desc_);
    }

    {
        ScopedTimer timer(elapsed);

        // Copy descriptor data to tensor host memory.
        // Eigen MatrixXf is [N x 64] row-major after .row() access,
        // but storage is column-major by default. We need contiguous rows.
        // desc1.data() is column-major, so we copy row by row.
        float* d1_ptr = t_desc1_->data();
        float* d2_ptr = t_desc2_->data();
        for (int i = 0; i < n1; ++i) {
            Eigen::Map<Eigen::RowVectorXf>(d1_ptr + i * 64, 64) = desc1.row(i);
        }
        for (int i = 0; i < n2; ++i) {
            Eigen::Map<Eigen::RowVectorXf>(d2_ptr + i * 64, 64) = desc2.row(i);
        }

        // Set push constants
        uint32_t push_consts[2] = {
            static_cast<uint32_t>(n1),
            static_cast<uint32_t>(n2)
        };

        // Update workgroup for this dispatch: ceil(n1 / workgroup_size)
        uint32_t wg_x = (static_cast<uint32_t>(n1) + workgroup_size_ - 1u) / workgroup_size_;
        algorithm_->setWorkgroup(kp::Workgroup{wg_x, 1, 1});

        // Record and execute: sync to device -> dispatch -> sync results back
        // (uses pre-allocated input_tensors_ / output_tensors_ members)

        // Clear previous recorded ops before re-recording
        seq_->clear();
        seq_->record<kp::OpTensorSyncDevice>(input_tensors_);
        seq_->record<kp::OpAlgoDispatch>(
            algorithm_,
            std::vector<uint32_t>{push_consts[0], push_consts[1]}
        );
        seq_->record<kp::OpTensorSyncLocal>(output_tensors_);
        seq_->eval();

        // Read results and apply ratio test CPU-side
        const int*   indices   = t_match_indices_->data();
        const float* distances = t_match_distances_->data();
        const float* second    = t_second_distances_->data();

        const float ratio_sq = ratio_threshold * ratio_threshold;
        matches_buf_.reserve(n1);

        for (int i = 0; i < n1; ++i) {
            if (indices[i] >= 0 && second[i] > 0.0f &&
                distances[i] < ratio_sq * second[i]) {
                float rq = 1.0f - std::sqrt(distances[i] / second[i]);
                matches_buf_.push_back({i, indices[i], distances[i], rq});
            }
        }
    }

    if (matching_us) *matching_us = elapsed;
    return matches_buf_;
}

} // namespace matching
} // namespace onyx
