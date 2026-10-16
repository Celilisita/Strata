#include "strata/models/deepseek_v4/hyper_connections.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<class T> T read_scalar(std::ifstream& file) {
    T value{};
    if (!file.read(reinterpret_cast<char*>(&value), sizeof value)) throw std::runtime_error("truncated fixture");
    return value;
}

std::vector<float> read_values(std::ifstream& file, std::size_t count) {
    std::vector<float> values(count);
    if (!file.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))) {
        throw std::runtime_error("truncated fixture array");
    }
    for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("nonfinite fixture value");
    return values;
}

struct Error {
    double maximum = 0, rms = 0;
    std::size_t bit_mismatches = 0;
};

Error difference(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison shape mismatch");
    Error error;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite candidate result");
        const double delta = static_cast<double>(actual[i]) - expected[i];
        error.maximum = std::max(error.maximum, std::abs(delta));
        error.rms += delta * delta;
        error.bit_mismatches += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]);
    }
    error.rms = std::sqrt(error.rms / actual.size());
    return error;
}

void print_error(const char* name, const Error& error, bool comma) {
    std::cout << '"' << name << "\":{\"max_abs\":" << error.maximum
              << ",\"rms\":" << error.rms << ",\"bit_mismatches\":" << error.bit_mismatches << '}';
    if (comma) std::cout << ',';
}

} // namespace

int main(int argc, char** argv) {
    using namespace strata::deepseek_v4;
    try {
        if (argc != 2) throw std::runtime_error("usage: strata-deepseek-v4-residual-bench <checkpoint.bin>");
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error("fixture currently requires little-endian host");
        }
        std::ifstream file(argv[1], std::ios::binary);
        std::array<char, 8> magic{};
        if (!file.read(magic.data(), magic.size()) || std::string(magic.data(), magic.size()) != "DSV4HC01") {
            throw std::runtime_error("invalid residual checkpoint signature");
        }
        const auto dim = read_scalar<std::uint32_t>(file);
        const auto iterations = read_scalar<std::uint32_t>(file);
        const auto norm_epsilon = read_scalar<float>(file);
        const auto hc_epsilon = read_scalar<float>(file);
        if (dim == 0 || dim > 16384 || iterations < 1 || iterations > 100) {
            throw std::runtime_error("fixture geometry outside supported range");
        }
        const std::size_t width = dim * hc_streams;
        const auto residual = read_values(file, width);
        const auto projection = read_values(file, width * hc_mix_rows);
        const auto scales = read_values(file, 3);
        const auto bases = read_values(file, hc_mix_rows);
        const auto sublayer = read_values(file, dim);
        const auto expected_pre = read_values(file, hc_streams);
        const auto expected_post = read_values(file, hc_streams);
        const auto expected_combination = read_values(file, hc_streams * hc_streams);
        const auto expected_reduced = read_values(file, dim);
        const auto expected_output = read_values(file, width);
        if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("unexpected fixture trailer");
        HcWeights weights{static_cast<int>(dim), projection, scales, bases,
                          static_cast<int>(iterations), norm_epsilon, hc_epsilon};
        HcMix mix;
        std::vector<float> reduced(dim), output(width);
        const auto start = std::chrono::steady_clock::now();
        hc_pre(weights, residual, mix, reduced);
        hc_post(mix, residual, sublayer, output);
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout.precision(9);
        std::cout << "{\"schema\":\"strata.deepseek_v4.residual.v1\",\"dim\":" << dim
                  << ",\"cpu_residual_ms\":" << elapsed
                  << ",\"full_model_correctness_claim\":false,\"equivalence_threshold_applied\":false,\"errors\":{";
        print_error("pre", difference(mix.pre, expected_pre), true);
        print_error("post", difference(mix.post, expected_post), true);
        print_error("combination", difference(mix.combination, expected_combination), true);
        print_error("reduced_bf16", difference(reduced, expected_reduced), true);
        print_error("output_bf16", difference(output, expected_output), false);
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
