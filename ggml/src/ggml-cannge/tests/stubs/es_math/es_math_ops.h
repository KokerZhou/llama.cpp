#pragma once
#include <ge/es_graph_builder.h>
// These stubs test control flow and holder publication, not ES signatures or numerical results.
namespace ge { namespace es {
inline EsTensorHolder Reshape(EsTensorHolder, const std::vector<int64_t> &) { return make_holder(); }
inline EsTensorHolder Permute(EsTensorHolder, const std::vector<int64_t> &) { return make_holder(); }
inline EsTensorHolder Slice(EsTensorHolder, const std::vector<int64_t> &, const std::vector<int64_t> &) { return make_holder(); }
inline EsTensorHolder Cast(EsTensorHolder, int64_t) { return make_holder(); }
inline EsTensorHolder Abs(EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Relu(EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Add(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Sub(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Mul(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder GatherV2(EsTensorHolder, EsTensorHolder, int64_t) { return make_holder(); }
inline EsTensorHolder Concat(EsTensorHolder, const std::vector<EsTensorHolder> &, int64_t) { return make_holder(); }
inline EsTensorHolder TensorScatterUpdate(EsTensorHolder, EsTensorHolder, EsTensorHolder) { return make_holder(); }
} }
