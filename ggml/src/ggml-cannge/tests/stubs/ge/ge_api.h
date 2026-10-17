#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

// Fault injection for the production GE wrapper; no device execution is simulated.
namespace cannge_test {
inline bool fail_add = false;
inline bool fail_compile = false;
inline bool fail_remove = false;
inline int remove_calls = 0;
inline std::set<uint32_t> graphs;
inline std::vector<std::string> events;
}

namespace gert { class Tensor {}; }
namespace ge {
using AscendString = std::string;
using Status = int;
constexpr Status SUCCESS = 0;
class Graph {};
inline Status GEInitialize(const std::map<AscendString, AscendString> &) { return SUCCESS; }
inline void GEFinalize() { cannge_test::events.push_back("finalize"); }
inline std::string GEGetErrorMsg() { return "injected failure"; }
class Session {
public:
    explicit Session(const std::map<AscendString, AscendString> &) {}
    ~Session() { cannge_test::graphs.clear(); cannge_test::events.push_back("session"); }
    Status AddGraphWithCopy(uint32_t id, Graph &) {
        if (cannge_test::fail_add) { return 1; }
        cannge_test::graphs.insert(id);
        return SUCCESS;
    }
    Status CompileGraph(uint32_t) { return cannge_test::fail_compile ? 1 : SUCCESS; }
    Status RemoveGraph(uint32_t id) {
        cannge_test::remove_calls++;
        cannge_test::events.push_back("remove");
        if (cannge_test::fail_remove) { return 1; }
        return cannge_test::graphs.erase(id) == 1 ? SUCCESS : 1;
    }
    Status LoadGraph(uint32_t, const std::map<AscendString, AscendString> &, void *) { return SUCCESS; }
    Status ExecuteGraphWithStreamAsync(uint32_t, void *, const std::vector<gert::Tensor> &,
                                      std::vector<gert::Tensor> &) { return SUCCESS; }
};
}
