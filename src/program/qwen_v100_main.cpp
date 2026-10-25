#include <cstring>
#include <iostream>

int strata_qwen_v100_completed_serve(int argc, char** argv);

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::cout
            << "Strata Dense native service\n"
            << "  --family qwen35_dense --variant qwen35_27b|qwen36_27b|qwen38_27b\n"
            << "  --model MODEL.gguf --context 262144 --slots 1 --mtp 0|1|2\n"
            << "  --kv-policy full|ram8k-reuse\n"
            << "The process accepts the native line protocol on stdin.\n"
            << "Model and CUDA initialization begin only after argument parsing.\n";
        return 0;
    }
    return strata_qwen_v100_completed_serve(argc, argv);
}
