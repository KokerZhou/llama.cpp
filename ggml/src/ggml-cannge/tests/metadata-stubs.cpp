#include "ggml.h"
#include "ggml-backend.h"

// Metadata-only ggml functions keep this test independent of CANN and GE.
size_t ggml_element_size(const ggml_tensor * t) { return t->type == GGML_TYPE_F16 ? 2 : 4; }
bool ggml_is_view(const ggml_tensor * t) { return t->view_src != nullptr; }
int ggml_n_dims(const ggml_tensor * t) {
    for (int d = GGML_MAX_DIMS - 1; d > 0; d--) {
        if (t->ne[d] != 1) { return d + 1; }
    }
    return 1;
}
size_t ggml_nbytes(const ggml_tensor * t) {
    size_t bytes = ggml_element_size(t);
    for (int d = 0; d < GGML_MAX_DIMS; d++) { bytes += (t->ne[d] - 1) * t->nb[d]; }
    return bytes;
}
int64_t ggml_nelements(const ggml_tensor * t) {
    int64_t n = 1;
    for (int d = 0; d < GGML_MAX_DIMS; d++) { n *= t->ne[d]; }
    return n;
}
bool ggml_is_empty(const ggml_tensor * t) { return ggml_nelements(t) == 0; }
bool ggml_is_contiguous(const ggml_tensor * t) {
    size_t stride = ggml_element_size(t);
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (t->ne[d] > 1 && t->nb[d] != stride) { return false; }
        stride *= t->ne[d];
    }
    return true;
}
bool ggml_cannge_float_dtype(ggml_type type) { return type == GGML_TYPE_F32 || type == GGML_TYPE_F16; }
bool ggml_backend_buffer_is_host(ggml_backend_buffer_t) { return false; }
const char * ggml_op_name(ggml_op) { return "test-op"; }
