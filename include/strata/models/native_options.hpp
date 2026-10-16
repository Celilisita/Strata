#pragma once
#include <map>
#include <string>
#include <stdexcept>

namespace strata::models {
struct NativeOptions {
    std::string family, variant, model, aux, kv = "ram8k-reuse";
    int context = 131072, slots = 1, mtp = 0;
    bool dspark = false;
    static NativeOptions parse(int argc, char** argv) {
        std::map<std::string, std::string> values;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc || !values.emplace(argv[i], argv[i + 1]).second)
                throw std::invalid_argument("native options require unique --key value pairs");
        }
        auto take = [&](const char* key, const std::string& fallback = "") {
            auto found = values.find(key);
            if (found == values.end()) return fallback;
            std::string value = found->second; values.erase(found); return value;
        };
        auto number = [&](const char* key, int fallback) {
            std::string value = take(key, std::to_string(fallback));
            size_t used = 0; int out = std::stoi(value, &used);
            if (used != value.size()) throw std::invalid_argument(key);
            return out;
        };
        NativeOptions o;
        o.family = take("--family"); o.variant = take("--variant");
        o.model = take("--model"); o.aux = take("--aux");
        o.context = number("--context", o.family == "qwen35_dense" ? 262144 : 131072); o.slots = number("--slots", 1);
        o.mtp = number("--mtp", 0); int dspark = number("--dspark", 0);
        o.kv = take("--kv-policy", o.kv);
        if (!values.empty() || o.model.empty() || o.context < 4 ||
            o.context > 262144 || o.slots < 1 || o.slots > 2 ||
            o.mtp < 0 || o.mtp > 2 || dspark < 0 || dspark > 1)
            throw std::invalid_argument("invalid native family configuration");
        o.dspark = dspark != 0;
        if (o.family == "qwen35_moe") {
            if ((o.variant != "minicpm_v47" && o.variant != "qwen36_35b") || o.aux.empty() || o.mtp ||
                (o.dspark && o.variant != "minicpm_v47") ||
                (o.dspark && o.slots != 1) || o.context > 196608)
                throw std::invalid_argument("Qwen35 MoE requires its explicit variant; DSpark is MiniCPM single slot");
        } else if (o.family == "qwen35_dense") {
            if (o.variant != "qwen38_27b" || o.slots != 1 || o.dspark ||
                (o.kv != "ram8k-reuse" && o.kv != "full"))
                throw std::invalid_argument("invalid Dense variant, slot or KV policy");
        } else throw std::invalid_argument("unsupported native family");
        return o;
    }
};
} // namespace strata::models
