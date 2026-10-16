#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <memory>
#include <string>
namespace strata::deepseek_v4 {
// Owns one immutable MTP source; embedding/head are borrowed from the target.
class MtpSidecar {
public:
    MtpSidecar(const std::string& path,ggml_backend_t backend);
    ~MtpSidecar();
    MtpSidecar(const MtpSidecar&)=delete;
    MtpSidecar& operator=(const MtpSidecar&)=delete;
    ggml_tensor* tensor(const std::string& name) const;
    size_t dense_device_bytes() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
