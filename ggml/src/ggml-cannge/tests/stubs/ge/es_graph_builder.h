#pragma once
#include <graph/types.h>
#include <graph/graph.h>
#include <memory>
#include <cstdint>
#include <vector>
namespace ge { namespace es {
struct EsTensorHolder { int id = 0; };
inline int next_holder = 1;
inline int64_t fail_output_index = -1;
inline int output_calls = 0;
inline int build_calls = 0;
inline EsTensorHolder make_holder() { return {next_holder++}; }
class EsGraphBuilder {
public:
    explicit EsGraphBuilder(const char *) {}
    static int64_t SetOutput(const EsTensorHolder &, int64_t index) {
        ++output_calls;
        return index == fail_output_index ? 17 : 0;
    }
    std::unique_ptr<Graph> BuildAndReset() { ++build_calls; return std::make_unique<Graph>(); }
    EsTensorHolder CreateInput(int64_t, const char *, DataType, Format, const std::vector<int64_t> &) { return make_holder(); }
    template<class T> EsTensorHolder CreateScalar(T) { return make_holder(); }
    template<class T> EsTensorHolder CreateConst(const std::vector<T> &, const std::vector<int64_t> &) { return make_holder(); }
    template<class T> EsTensorHolder CreateVector(const std::vector<T> &) { return make_holder(); }
};
} }
