#pragma once
#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <string>
namespace strata::gemma4::native {
struct StageCost {double milliseconds=0;unsigned calls=0;};
// Optional fixed-capacity benchmark hook. Never enabled in a captured Graph.
// It records no payload/logits and adds no GPU copies or synchronization points.
class Profile {
    struct Point {cudaEvent_t event=nullptr;const char* name=nullptr;int layer=-1,tokens=0;};
    std::vector<Point> points_;size_t used_=0;
public:
    explicit Profile(size_t maximum=1536);
    ~Profile();
    Profile(const Profile&)=delete;
    Profile& operator=(const Profile&)=delete;
    void mark(const char*,int,int,cudaStream_t);
    std::map<std::string,StageCost> summary();
};
}
