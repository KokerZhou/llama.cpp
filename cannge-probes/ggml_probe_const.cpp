// probe: CreateConst(iota) fidelity - the set_rows mask depends on it
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <unistd.h>
int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    const int64_t S = 512;
    struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);
    ggml_tensor * ar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S);
    ggml_backend_alloc_ctx_tensors(ctx, cannge);
    std::vector<float> want(S), got(S);
    for (int i = 0; i < S; i++) want[i] = (float) i;
    ggml_backend_tensor_set(ar, want.data(), 0, S * 4);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ar);
    ggml_backend_graph_compute(cannge, gf);
    ggml_backend_synchronize(cannge);
    ggml_backend_tensor_get(ar, got.data(), 0, S * 4);
    int bad = 0;
    for (int i = 0; i < S; i++) if (got[i] != want[i]) bad++;
    printf("const-iota bad=%d/%d first_got=%f\n", bad, (int) S, got[0]);
    _exit(bad != 0);
}
