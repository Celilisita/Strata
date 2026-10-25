#include <cstring>
#include <iostream>

int strata_gemma4_native_serve(int argc, char** argv);

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::cout
            << "Strata Gemma4 native service\n"
            << "  --model MODEL.gguf --assistant ASSISTANT.gguf\n"
            << "  --context 262144 --batch 8192 --mtp 1\n"
            << "The process accepts the native line protocol on stdin.\n"
            << "Consult docs/gemma-integration.md for optional candidate policies.\n";
        return 0;
    }
    return strata_gemma4_native_serve(argc, argv);
}
