#include "strata/dense/visual.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace strata::dense {
namespace {
template<class T> void read(std::ifstream &file, T *out, size_t count) {
    if (count > size_t((std::numeric_limits<std::streamsize>::max)())/sizeof(T))
        throw std::runtime_error("DVI1 array size overflow");
    file.read(reinterpret_cast<char *>(out), std::streamsize(count*sizeof(T)));
    if (!file) throw std::runtime_error("truncated DVI1 payload");
}
}
VisualInput read_visual_input(const std::string &path, const Spec &s, int context, int max_new) {
    if (s.architecture != "qwen35" || s.n_embd != 1024)
        throw std::invalid_argument("DVI1 requires the bound Qwen3.5-0.8B model");
    const uint32_t endian=1;
    if (*reinterpret_cast<const unsigned char *>(&endian)!=1 || sizeof(int)!=4)
        throw std::runtime_error("DVI1 requires little-endian 32-bit input IDs");
    std::ifstream file(std::filesystem::u8path(path),std::ios::binary|std::ios::ate);
    if (!file) throw std::runtime_error("cannot open prepared visual input");
    const auto file_size=file.tellg();file.seekg(0);
    uint32_t h[8];read(file,h,8);
    if (h[0]!=0x31495644 || h[1]!=1 || h[2]!=1024 || h[5]!=4 || h[7]!=0)
        throw std::runtime_error("unsupported DVI1 header");
    const uint64_t n=h[3],m=h[4];
    if (!n || !m || m>n || max_new<1 || context<1 || n+uint64_t(max_new)>uint64_t(context) ||
        !h[6] || uint64_t(h[6])+max_new>s.context_length)
        throw std::runtime_error("DVI1 physical or logical context exceeds budget");
    const uint64_t expected=32+20*n+8*m+uint64_t(4)*1024*m;
    if (file_size<0 || uint64_t(file_size)!=expected)
        throw std::runtime_error("DVI1 file size differs from exact declared payload");
    VisualInput out;out.width=1024;out.next_logical=int(h[6]);
    out.ids.resize(size_t(n));out.positions.resize(size_t(4*n));
    out.image_rows.resize(size_t(m));out.embeddings.resize(size_t(1024*m));
    read(file,out.ids.data(),out.ids.size());
    read(file,out.positions.data(),out.positions.size());
    read(file,out.image_rows.data(),out.image_rows.size());
    read(file,out.embeddings.data(),out.embeddings.size());
    for (int id:out.ids) if (id<0 || uint64_t(id)>=s.n_vocab)
        throw std::runtime_error("DVI1 token ID is outside the model vocabulary");
    int maximum=0;
    for (int axis=0;axis<4;++axis) for (size_t row=0;row<size_t(n);++row) {
        const int pos=out.positions[size_t(axis)*n+row];
        if (pos<0 || uint64_t(pos)>=s.context_length || (axis==3 && pos!=0))
            throw std::runtime_error("DVI1 invalid Qwen still-image position");
        if (axis<3) maximum=std::max(maximum,pos);
    }
    if (maximum+1!=out.next_logical)
        throw std::runtime_error("DVI1 final logical cursor differs from position extent");
    int64_t previous=-1;
    for (int64_t row:out.image_rows) {
        if (row<=previous || uint64_t(row)>=n)
            throw std::runtime_error("DVI1 image row index is duplicated or out of bounds");
        previous=row;
    }
    for (float value:out.embeddings) if (!std::isfinite(value))
        throw std::runtime_error("DVI1 nonfinite visual embedding");
    return out;
}

InputBatch VisualInput::slice(int begin,int end) const {
    if (begin<0 || end<=begin || size_t(end)>ids.size() || width!=1024)
        throw std::invalid_argument("invalid prepared visual batch span");
    InputBatch out;
    out.ids.assign(ids.begin()+begin,ids.begin()+end);
    out.positions.reserve(size_t(end-begin)*4);
    for (int axis=0;axis<4;++axis)
        out.positions.insert(out.positions.end(),positions.begin()+size_t(axis)*ids.size()+begin,
                              positions.begin()+size_t(axis)*ids.size()+end);
    auto first=std::lower_bound(image_rows.begin(),image_rows.end(),int64_t(begin));
    auto last=std::lower_bound(first,image_rows.end(),int64_t(end));
    for (auto p=first;p!=last;++p) out.image_rows.push_back(*p-begin);
    const auto image_begin=size_t(first-image_rows.begin()),image_end=size_t(last-image_rows.begin());
    out.embeddings.assign(embeddings.begin()+image_begin*width,embeddings.begin()+image_end*width);
    if (size_t(end)==ids.size()) out.next_logical=next_logical;
    else {
        int maximum=0;
        for (int axis=0;axis<3;++axis)
            maximum=std::max(maximum,*std::max_element(positions.begin()+size_t(axis)*ids.size(),
                                                       positions.begin()+size_t(axis)*ids.size()+end));
        out.next_logical=maximum+1;
    }
    return out;
}
} // namespace strata::dense
