#include "strata/dense/spec.hpp"
#include <cstdio>
#include <exception>
#include <map>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: strata-dense-inspect MODEL.gguf\n");
        return 2;
    }
    try {
        strata::GgufFile file(argv[1]);
        strata::dense::Spec spec;
        std::string error;
        if (!strata::dense::load_spec(file, spec, error)) {
            std::fprintf(stderr, "dense: %s\n", error.c_str());
            return 1;
        }
        unsigned recurrent = 0;
        for (const auto &layer : spec.layers) recurrent += layer.recurrent;
        std::printf("dense architecture=%s layers=%llu recurrent=%u attention=%llu "
                    "hidden=%llu ff=%llu vocab=%llu context=%llu\n",
                    spec.architecture.c_str(), (unsigned long long)spec.n_layers, recurrent,
                    (unsigned long long)(spec.n_layers - recurrent), (unsigned long long)spec.n_embd,
                    (unsigned long long)spec.n_ff, (unsigned long long)spec.n_vocab,
                    (unsigned long long)spec.context_length);
        std::printf("dense attention heads=%llu kv_heads=%llu head_dim=%llu query_gate=%d "
                    "rope_dim=%llu rope_base=%.9g rms_epsilon=%.9g tied_head=%d\n",
                    (unsigned long long)spec.n_head, (unsigned long long)spec.n_head_kv,
                    (unsigned long long)spec.head_dim, int(spec.query_gated),
                    (unsigned long long)spec.rope_dim, spec.rope_base, spec.rms_epsilon,
                    int(spec.output == spec.embedding));
        if (spec.query_gated)
            std::printf("dense gdn state=%llu group=%llu rank=%llu inner=%llu conv=%llu "
                        "rope_sections=%llu,%llu,%llu,%llu\n",
                        (unsigned long long)spec.ssm_state, (unsigned long long)spec.ssm_group,
                        (unsigned long long)spec.ssm_rank, (unsigned long long)spec.ssm_inner,
                        (unsigned long long)spec.ssm_conv, (unsigned long long)spec.rope_sections[0],
                        (unsigned long long)spec.rope_sections[1], (unsigned long long)spec.rope_sections[2],
                        (unsigned long long)spec.rope_sections[3]);
        uint64_t total = 0;
        std::map<uint32_t, std::pair<size_t, uint64_t>> formats;
        for (const auto &tensor : file.tensors()) {
            const auto bytes = strata::dense::tensor_bytes(file, tensor);
            auto &format = formats[tensor.type];
            ++format.first;
            format.second += bytes;
            total += bytes;
        }
        std::printf("dense inventory tensors=%zu weight_bytes=%llu embedding_type=%u output_type=%u\n",
                    file.tensors().size(), (unsigned long long)total,
                    spec.embedding->type, spec.output->type);
        for (const auto &[type, value] : formats)
            std::printf("dense format type=%u name=%s tensors=%zu bytes=%llu\n", type,
                        strata::ggml_type_name(type), value.first, (unsigned long long)value.second);
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "dense: %s\n", exception.what());
        return 1;
    }
}
