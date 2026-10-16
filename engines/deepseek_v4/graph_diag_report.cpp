#include "graph_diag.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
void fields(std::ostream& out, uint32_t value) {
    constexpr const char* names[] = {"data_pointer", "shape", "stride", "operation_or_type",
        "operation_parameters", "CPU_source_pointers", "device_source_pointers", "source_shape",
        "source_stride", "view_pointer_or_offset", "backend_buffer", "extra", "flags", "other_struct_bytes"};
    out << '['; bool comma = false;
    for (unsigned i = 0; i < std::size(names); ++i) {
        if (!(value & (1u << i))) continue;
        if (comma) out << ',';
        out << std::quoted(names[i]); comma = true;
    }
    out << ']';
}
}
void write_graph_diagnostic(const std::filesystem::path& directory) {
    const auto state = graph_diag_snapshot();
    if (!state.enabled) return;
    std::ofstream out(directory / "graph-diagnostic.json");
    out << "{\"scope\":\"last measured request only; canonical UID shortcut/property checks unchanged; bounded host comparison, no GPU math changes; diagnostic timings excluded from performance claims\",\"dropped\":"
        << state.dropped << ",\"records\":[";
    for (uint32_t i = 0; i < state.count; ++i) {
        if (i) out << ',';
        const auto& r = state.records[i];
        out << "{\"tokens\":" << r.tokens << ",\"old_uid\":" << r.old_uid << ",\"new_uid\":" << r.new_uid
            << ",\"old_nodes\":" << r.old_nodes << ",\"new_nodes\":" << r.new_nodes
            << ",\"changed_nodes\":" << r.changed_nodes << ",\"first_node\":" << r.first_node
            << ",\"operation\":" << r.operation << ",\"first_name\":" << std::quoted(r.first_name)
            << ",\"first_byte\":" << r.first_byte << ",\"last_byte\":" << r.last_byte << ",\"different_bytes\":" << r.different_bytes
            << ",\"tensor_bytes\":" << r.tensor_bytes << ",\"name_offset\":" << r.name_offset << ",\"reserved_offset\":" << r.reserved_offset
            << ",\"first_fields\":";
        fields(out, r.first_fields); out << ",\"all_fields\":"; fields(out, r.all_fields); out << '}';
    }
    out << "]}\n";
    if (!out) throw std::runtime_error("bounded graph diagnostic write failed");
}
}
