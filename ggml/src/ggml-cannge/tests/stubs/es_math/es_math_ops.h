#pragma once
#include <ge/es_graph_builder.h>
// These stubs test control flow and holder publication, not ES signatures or numerical results.
namespace ge { namespace es {
inline std::vector<int64_t> last_reshape_shape;
inline int concat_calls = 0;
inline int slice_calls = 0;
inline std::vector<int64_t> last_slice_size;
inline EsTensorHolder Reshape(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder ReduceSum(EsTensorHolder, EsTensorHolder, bool) { return make_holder(); }
inline EsTensorHolder BatchMatMulV3(EsTensorHolder, EsTensorHolder, const void *, const void *, bool, bool) { return make_holder(); }
inline EsTensorHolder Reshape(EsTensorHolder, const std::vector<int64_t> & shape) {
    last_reshape_shape = shape;
    return make_holder();
}
inline EsTensorHolder Permute(EsTensorHolder, const std::vector<int64_t> &) { return make_holder(); }
inline EsTensorHolder Slice(EsTensorHolder, const std::vector<int64_t> &, const std::vector<int64_t> & size) {
    ++slice_calls;
    last_slice_size = size;
    return make_holder();
}
inline EsTensorHolder Cast(EsTensorHolder, int64_t) { return make_holder(); }
inline EsTensorHolder Abs(EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Relu(EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Add(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Sub(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder Mul(EsTensorHolder, EsTensorHolder) { return make_holder(); }
inline EsTensorHolder GatherV2(EsTensorHolder, EsTensorHolder, int64_t) { return make_holder(); }
inline EsTensorHolder Concat(EsTensorHolder, const std::vector<EsTensorHolder> &, int64_t) { ++concat_calls; return make_holder(); }
inline EsTensorHolder TensorScatterUpdate(EsTensorHolder, EsTensorHolder, EsTensorHolder) { return make_holder(); }
} }
