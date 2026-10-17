#include "ggml-backend-impl.h"
#include "ggml-impl.h"

bool ggml_backend_cannge_graph_env_enabled() { return true; }
void ggml_log_internal(enum ggml_log_level, const char *, ...) {}
const char * ggml_backend_cannge_buffer_type_name(ggml_backend_buffer_type_t) { return "test-cannge"; }
ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type() { return nullptr; }
size_t ggml_type_size(ggml_type type) { return type == GGML_TYPE_F16 ? 2 : 4; }
bool ggml_are_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int d = 0; d < GGML_MAX_DIMS; d++) { if (a->ne[d] != b->ne[d]) { return false; } }
    return true;
}
bool ggml_is_contiguous_rows(const ggml_tensor * t) { return t->nb[0] == ggml_type_size(t->type); }
enum ggml_unary_op ggml_get_unary_op(const ggml_tensor * t) { return (ggml_unary_op) t->op_params[0]; }
enum ggml_glu_op ggml_get_glu_op(const ggml_tensor * t) { return (ggml_glu_op) t->op_params[0]; }
