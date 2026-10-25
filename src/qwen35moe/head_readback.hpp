#pragma once
#include "strata/qwen35moe/head_pick.hpp"
#include "strata/kernels/radix_sampler.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::qwen35moe {

// Runtime owns this stream and drains its existing head/epoch boundary before
// copy_to(), enter/leave_decode() and destruction. The snapshot receives no work.
class HeadReadback {
public:
    static bool enabled(bool domain) {
        const char* raw = std::getenv("STRATA_QWEN35_HEAD_READBACK");
        if (!raw) return domain;
        const std::string value(raw);
        if (value != "0" && value != "1")
            throw std::invalid_argument("head readback requires0 or1");
        if (value == "1" && !domain)
            throw std::invalid_argument("head readback requires SM120 NVFP4 GPU head");
        return value == "1";
    }

    explicit HeadReadback(cudaStream_t stream) : stream_(stream) {
        raw_policy_ = policy_ = read_policy(stream_, "read initial head stream policy");
        check(cudaGetDeviceFlags(&context_flags_), "observe head context flags");
        try {
            // Created on the same current context as Runtime's owned stream.
            check(cudaStreamCreateWithFlags(&snapshot_, cudaStreamNonBlocking), "create head policy snapshot");
            check(cudaHostAlloc(&data_, bytes(), cudaHostAllocDefault), "allocate head readback");
        } catch (...) {
            if (data_) cudaFreeHost(data_);
            if (snapshot_) cudaStreamDestroy(snapshot_);
            throw;
        }
    }
    ~HeadReadback() {
        // Runtime has already drained; failed cleanup cannot escape a destructor.
        try { leave_decode(); } catch (...) {}
        if (snapshot_) cudaStreamDestroy(snapshot_);
        if (data_) cudaFreeHost(data_);
    }
    void enter_decode() {
        if (decoding_) return;
        if (restore_required_) throw std::logic_error("head policy restoration is incomplete");
        // Refresh at each epoch so legitimate PP-side attribute changes survive.
        check(cudaStreamCopyAttributes(snapshot_, stream_), "snapshot current head stream attributes");
        snapshot_policy_ = read_policy(snapshot_, "read saved head stream policy");
        raw_policy_ = snapshot_policy_;
        snapshot_valid_ = true;
        restore_required_ = true; // Preserve the snapshot after any partial Set/Get.
        cudaStreamAttrValue desired{};
        desired.syncPolicy = cudaSyncPolicyBlockingSync;
        check(cudaStreamSetAttribute(stream_, cudaStreamAttributeSynchronizationPolicy, &desired),
              "enter head decode blocking policy");
        policy_ = read_policy(stream_, "verify head decode blocking policy");
        if (policy_ != int(cudaSyncPolicyBlockingSync))
            throw std::runtime_error("head decode blocking policy actual=" + std::to_string(policy_));
        decoding_ = true;
    }
    void leave_decode() {
        if (!restore_required_) return;
        check(cudaStreamCopyAttributes(stream_, snapshot_), "restore original head stream attributes");
        policy_ = read_policy(stream_, "verify restored head stream policy");
        if (policy_ != raw_policy_)
            throw std::runtime_error("head stream restore expected=" + std::to_string(raw_policy_) +
                                     " actual=" + std::to_string(policy_));
        restore_required_ = decoding_ = false;
    }
    int original_policy() const { return raw_policy_; }
    int raw_policy() const { return raw_policy_; }
    unsigned context_flags() const { return context_flags_; } // Observation only; never normalization.
    int snapshot_policy() const { return snapshot_policy_; }
    bool snapshot_valid() const { return snapshot_valid_; }
    bool decoding() const { return decoding_; }
    HeadReadback(const HeadReadback&) = delete;
    HeadReadback& operator=(const HeadReadback&) = delete;

    static constexpr size_t bytes() {
        return sizeof(HeadPick) > sizeof(kernels::SamplePick) ? sizeof(HeadPick) : sizeof(kernels::SamplePick);
    }
    int policy() const { return policy_; }
    void* data(size_t count) {
        bounds(count);
        return data_;
    }
    void copy_to(void* target, size_t count) const {
        bounds(count);
        if (!target) throw std::invalid_argument("null head readback target");
        std::memcpy(target, data_, count);
    }
private:
    static void bounds(size_t count) {
        if (!count || count > bytes()) throw std::out_of_range("head readback byte range");
    }
    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
    static int read_policy(cudaStream_t stream, const char* what) {
        cudaStreamAttrValue actual{};
        check(cudaStreamGetAttribute(stream, cudaStreamAttributeSynchronizationPolicy, &actual), what);
        return int(actual.syncPolicy);
    }
    cudaStream_t stream_ = nullptr, snapshot_ = nullptr;
    void* data_ = nullptr;
    int policy_ = 0, raw_policy_ = 0, snapshot_policy_ = 0;
    unsigned context_flags_ = 0;
    bool decoding_ = false, restore_required_ = false, snapshot_valid_ = false;
};

} // namespace strata::qwen35moe
