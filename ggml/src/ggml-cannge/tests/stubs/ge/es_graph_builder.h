#pragma once
#include <graph/types.h>
#include <cstdint>
#include <vector>
namespace ge { namespace es {
struct EsTensorHolder { int id = 0; };
inline int next_holder = 1;
inline EsTensorHolder make_holder() { return {next_holder++}; }
class EsGraphBuilder {
public:
    explicit EsGraphBuilder(const char *) {}
    EsTensorHolder CreateInput(int64_t, const char *, DataType, Format, const std::vector<int64_t> &) { return make_holder(); }
    template<class T> EsTensorHolder CreateScalar(T) { return make_holder(); }
    template<class T> EsTensorHolder CreateConst(const std::vector<T> &, const std::vector<int64_t> &) { return make_holder(); }
};
} }
