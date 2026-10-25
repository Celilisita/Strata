#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr uint32_t magic = 0x31564453; // SDV1, little endian
constexpr uint32_t text_record = 0, image_record = 1;
using Chunks = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
using TextModel = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Encoder = std::unique_ptr<mtmd_context, decltype(&mtmd_free)>;

void quiet_log(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

void startup_stage(const char *stage) {
    const char *enabled = std::getenv("STRATA_DENSE_VISION_DIAGNOSTICS");
    if (enabled && std::string(enabled) == "1") {
        std::fprintf(stderr, "dense vision startup: %s\n", stage);
        std::fflush(stderr);
    }
}

struct Image {
    mtmd_helper_bitmap_wrapper bitmap{};
    Chunks chunks{nullptr, mtmd_input_chunks_free};
    Image() = default;
    Image(const Image &) = delete;
    Image &operator=(const Image &) = delete;
    ~Image() {
        if (bitmap.bitmap) mtmd_bitmap_free(bitmap.bitmap);
        if (bitmap.video_ctx) mtmd_helper_video_free(bitmap.video_ctx);
    }
};

struct Options {
    std::string model, projector;
    fs::path output_root;
    bool gpu = false;
    int threads = 0, max_tokens = 0;
    uint32_t request_tokens = 131072;
};

int positive(const std::string &value) {
    size_t consumed = 0;
    const long number = std::stol(value, &consumed);
    if (consumed != value.size() || number < 1 || number > 1000000)
        throw std::invalid_argument("numeric option must be an integer in 1..1000000");
    return int(number);
}

Options parse_options(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto next = [&]() {
            if (++i >= argc) throw std::invalid_argument("missing argument for " + option);
            return std::string(argv[i]);
        };
        if (option == "--model") o.model = next();
        else if (option == "--mmproj") o.projector = next();
        else if (option == "--output-dir") o.output_root = fs::u8path(next());
        else if (option == "--threads") o.threads = positive(next());
        else if (option == "--max-tokens") o.max_tokens = positive(next());
        else if (option == "--max-request-tokens") o.request_tokens = uint32_t(positive(next()));
        else if (option == "--gpu") o.gpu = true;
        else throw std::invalid_argument("unknown option " + option);
    }
    if (o.model.empty() || o.projector.empty() || o.output_root.empty())
        throw std::invalid_argument("usage: strata-dense-vision --model text.gguf --mmproj projector.gguf "
                                    "--output-dir task-directory [--gpu] [--threads N] "
                                    "[--max-tokens N] [--max-request-tokens N]");
    o.output_root = fs::canonical(o.output_root);
    if (!fs::is_directory(o.output_root)) throw std::invalid_argument("output-dir is not a directory");
    return o;
}

void validate_models(const Options &o) {
    auto open = [](const std::string &path) {
        gguf_init_params params{true, nullptr};
        auto *result = gguf_init_from_file(path.c_str(), params);
        if (!result) throw std::runtime_error("cannot read GGUF metadata: " + path);
        return std::unique_ptr<gguf_context, decltype(&gguf_free)>(result, gguf_free);
    };
    auto text = open(o.model), projector = open(o.projector);
    const int64_t arch = gguf_find_key(text.get(), "general.architecture");
    const int64_t hidden = gguf_find_key(text.get(), "qwen35.embedding_length");
    const int64_t projected = gguf_find_key(projector.get(), "clip.vision.projection_dim");
    if (arch < 0 || gguf_get_kv_type(text.get(), arch) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(text.get(), arch)) != "qwen35" || hidden < 0 || projected < 0 ||
        gguf_get_kv_type(text.get(), hidden) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(projector.get(), projected) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(text.get(), hidden) != 1024 ||
        gguf_get_val_u32(projector.get(), projected) != 1024)
        throw std::runtime_error("dense vision requires matching qwen35 and projector width 1024");
}

std::vector<std::string> fields(const std::string &line) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (true) {
        const size_t end = line.find('\t', begin);
        result.push_back(line.substr(begin, end == std::string::npos ? end : end-begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

fs::path partial_path(fs::path output) {
    output += ".partial";
    return output;
}

fs::path artifact_path(const Options &o, const std::string &name) {
    if (name.empty()) throw std::invalid_argument("empty artifact path");
    auto requested = fs::u8path(name);
    if (requested.is_relative()) requested = o.output_root/requested;
    const auto parent = fs::canonical(requested.parent_path());
    const auto relative = parent.lexically_relative(o.output_root);
    if (relative.empty() && parent != o.output_root)
        throw std::invalid_argument("artifact is outside output-dir");
    for (const auto &part : relative)
        if (part == "..") throw std::invalid_argument("artifact is outside output-dir");
    const auto output = parent/requested.filename();
    if (fs::exists(output) || fs::exists(partial_path(output)))
        throw std::invalid_argument("artifact or partial file already exists");
    return output;
}

void write(std::ofstream &out, const void *data, size_t bytes) {
    out.write(static_cast<const char *>(data), std::streamsize(bytes));
    if (!out) throw std::runtime_error("artifact write failed");
}
template<size_t N> void write(std::ofstream &out, const uint32_t (&values)[N]) {
    write(out, values, sizeof(values));
}

void encode(mtmd_context *ctx, const Options &o, const std::vector<std::string> &request) {
    if (request.size() < 3 || request.size() > 34 || request[0] != "ENC")
        throw std::invalid_argument("expected ENC<TAB>artifact<TAB>image [<TAB>image...], maximum 32 images");
    const auto started = std::chrono::steady_clock::now();
    const auto output = artifact_path(o, request[1]);
    std::vector<std::unique_ptr<Image>> images;
    uint64_t records = 0, text_tokens = 0, image_tokens = 0;
    const std::string marker = mtmd_get_marker(ctx);
    for (size_t i = 2; i < request.size(); ++i) {
        if (request[i].empty()) throw std::invalid_argument("empty image path");
        auto image = std::make_unique<Image>();
#if defined(STRATA_MTMD_FILE_OPTIONS)
        image->bitmap = mtmd_helper_bitmap_init_from_file(ctx, request[i].c_str(), false,
                                                          mtmd_helper_init_opt_default());
#else
        image->bitmap = mtmd_helper_bitmap_init_from_file(ctx, request[i].c_str(), false);
#endif
        auto *bitmap = image->bitmap.bitmap;
        if (!bitmap || image->bitmap.video_ctx || mtmd_bitmap_is_audio(bitmap))
            throw std::runtime_error("input is not a readable still image: " + request[i]);
        image->chunks.reset(mtmd_input_chunks_init());
        const mtmd_input_text prompt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap *bitmaps[]{bitmap};
        if (!image->chunks || mtmd_tokenize(ctx, image->chunks.get(), &prompt, bitmaps, 1))
            throw std::runtime_error("image preprocessing failed: " + request[i]);
        uint64_t image_chunks = 0;
        for (size_t j = 0; j < mtmd_input_chunks_size(image->chunks.get()); ++j) {
            const auto *chunk = mtmd_input_chunks_get(image->chunks.get(), j);
            const auto kind = mtmd_input_chunk_get_type(chunk);
            const auto count = mtmd_input_chunk_get_n_tokens(chunk);
            if (kind == MTMD_INPUT_CHUNK_TYPE_TEXT) text_tokens += count;
            else if (kind == MTMD_INPUT_CHUNK_TYPE_IMAGE) { image_tokens += count; ++image_chunks; }
            else throw std::runtime_error("unexpected non-image media chunk");
            ++records;
        }
        if (!image_chunks) throw std::runtime_error("preprocessor returned no IMAGE chunks");
        if (text_tokens+image_tokens > o.request_tokens)
            throw std::runtime_error("image request exceeds max-request-tokens");
        images.push_back(std::move(image));
    }
    const auto partial = partial_path(output);
    try {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open artifact in output-dir");
        const uint32_t header[]{magic, 1, 1024, uint32_t(images.size()), uint32_t(records),
                                uint32_t(text_tokens), uint32_t(image_tokens), uint32_t(o.gpu)};
        write(out, header);
        for (size_t i = 0; i < images.size(); ++i) {
            const auto &image = *images[i];
            const auto *bitmap = image.bitmap.bitmap;
            for (size_t j = 0; j < mtmd_input_chunks_size(image.chunks.get()); ++j) {
                const auto *chunk = mtmd_input_chunks_get(image.chunks.get(), j);
                const bool is_image = mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
                const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
                const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk);
                if (!count || n_pos < 1 || count > o.request_tokens)
                    throw std::runtime_error("invalid chunk token/position count");
                const uint32_t flags = is_image ? uint32_t(mtmd_decode_use_mrope(ctx)) |
                    (uint32_t(mtmd_decode_use_non_causal(ctx, chunk)) << 1) : 0;
                const uint32_t record[]{is_image ? image_record : text_record, uint32_t(i), uint32_t(j),
                    uint32_t(count), uint32_t(n_pos), mtmd_bitmap_get_nx(bitmap), mtmd_bitmap_get_ny(bitmap), flags};
                write(out, record);
                if (!is_image) {
                    size_t size = 0;
                    const auto *tokens = mtmd_input_chunk_get_tokens_text(chunk, &size);
                    if (!tokens || size != count) throw std::runtime_error("invalid text chunk payload");
                    write(out, tokens, size*sizeof(llama_token));
                    continue;
                }
                const auto *tokens = mtmd_input_chunk_get_tokens_image(chunk);
                if (!tokens || mtmd_encode_chunk(ctx, chunk))
                    throw std::runtime_error("image encoder/projector failed");
                const float *embedding = mtmd_get_output_embd(ctx);
                if (!embedding) throw std::runtime_error("encoder returned no embeddings");
                for (size_t k = 0; k < count*1024; ++k)
                    if (!std::isfinite(embedding[k])) throw std::runtime_error("non-finite visual embedding");
                std::vector<uint32_t> positions(count*4);
                for (size_t k = 0; k < count; ++k) {
                    const auto position = mtmd_image_tokens_get_decoder_pos(tokens, 0, k);
                    positions[4*k] = position.t; positions[4*k+1] = position.x;
                    positions[4*k+2] = position.y; positions[4*k+3] = position.z;
                }
                write(out, positions.data(), positions.size()*sizeof(uint32_t));
                write(out, embedding, count*1024*sizeof(float));
            }
        }
        out.close();
        if (!out) throw std::runtime_error("artifact close failed");
        fs::rename(partial, output);
    } catch (...) { std::error_code ignored; fs::remove(partial, ignored); throw; }
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-started).count();
    std::printf("OK %zu %llu %llu %llu %.3f\n", images.size(),
                 static_cast<unsigned long long>(records), static_cast<unsigned long long>(image_tokens),
                 static_cast<unsigned long long>(text_tokens), elapsed);
}
} // namespace

int main(int argc, char **argv) {
    try {
        const uint32_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1 || sizeof(llama_token) != 4)
            throw std::runtime_error("SDV1 requires little-endian storage and 32-bit tokens");
        const auto options = parse_options(argc, argv);
        startup_stage("options parsed");
        validate_models(options);
        startup_stage("metadata validated");
        llama_log_set(quiet_log, nullptr); mtmd_helper_log_set(quiet_log, nullptr);
        ggml_backend_load_all();
        startup_stage("backends loaded");
        llama_backend_init();
        startup_stage("llama backend initialized");
        auto model_params = llama_model_default_params();
        startup_stage("text parameters created");
        model_params.vocab_only = true;
        TextModel text(llama_model_load_from_file(options.model.c_str(), model_params), llama_model_free);
        startup_stage("text vocabulary loaded");
        if (!text) throw std::runtime_error("cannot load text vocabulary metadata");
        auto encoder_params = mtmd_context_params_default();
        startup_stage("vision parameters created");
        encoder_params.use_gpu = options.gpu;
        encoder_params.print_timings = false; encoder_params.warmup = false;
        if (options.threads) encoder_params.n_threads = options.threads;
        if (options.max_tokens) encoder_params.image_max_tokens = options.max_tokens;
        Encoder encoder(mtmd_init_from_file(options.projector.c_str(), text.get(), encoder_params), mtmd_free);
        startup_stage("vision projector loaded");
        if (!encoder || !mtmd_support_vision(encoder.get()) || !mtmd_decode_use_mrope(encoder.get()))
            throw std::runtime_error("projector does not provide the required Qwen M-RoPE vision pipeline");
        std::printf("READY 1024 SDV1\n"); std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "QUIT") break;
            try { encode(encoder.get(), options, fields(line)); }
            catch (const std::exception &e) { std::printf("ERR %s\n", e.what()); }
            std::fflush(stdout);
        }
        encoder.reset(); text.reset(); llama_backend_free();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "strata-dense-vision: %s\n", e.what());
        return 1;
    }
}
