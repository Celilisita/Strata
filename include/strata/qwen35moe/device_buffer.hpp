#pragma once
#include <cstddef>
namespace strata::qwen35moe {
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    void reset(size_t bytes = 0);
    void view(const DeviceBuffer& owner, size_t offset, size_t bytes);
    void* data() const { return data_; }
    template<class T> T* as() const { return static_cast<T*>(data_); }
    size_t bytes() const { return bytes_; }
    size_t owned_bytes() const { return owned_ ? bytes_ : 0; }
private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
    bool owned_ = true;
};

} // namespace strata::qwen35moe
