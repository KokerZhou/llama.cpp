// probe: the exact set_rows select chain via ggml ops (arange as input)
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
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    const int64_t S = 512, R = 1024;
    struct ggml_init_params ip = { 64*1024*1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);
    ggml_tensor * arange = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S);
    ggml_tensor * idx    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * ones   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S);
    ggml_tensor * row    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, R);
    ggml_tensor * cache  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, S);
    ggml_tensor * d    = ggml_sub(ctx, arange, idx);
    ggml_tensor * mask = ggml_relu(ctx, ggml_sub(ctx, ones, ggml_abs(ctx, d)));
    ggml_tensor * m2   = ggml_reshape_2d(ctx, mask, 1, S);
    ggml_tensor * r2   = ggml_reshape_2d(ctx, row, R, 1);
    ggml_tensor * out  = ggml_add(ctx, ggml_mul(ctx, cache, m2), ggml_mul(ctx, r2, m2));
    std::vector<float> ar(S), on(S, 1.0f), rw(R), ca(R*S), want(R*S);
    for (int i = 0; i < S; i++) ar[i] = (float) i;
    for (int i = 0; i < R; i++) rw[i] = 0.5f * sinf(0.03f*i) + 3.0f;
    for (int i = 0; i < R*S; i++) ca[i] = 0.02f * (float)((int)(i*2654435761u)%89-44)/44.0f;
    int slot = 7;
    for (int i = 0; i < R*S; i++) want[i] = ca[i];
    for (int e = 0; e < R; e++) want[slot*R + e] = rw[e];
    for (int side = 0; side < 2; side++) {
        ggml_backend_t be = side==0?cannge:cpu; const char* sn = side==0?"cannge":"cpu";
        ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(arange, ar.data(), 0, S*4);
        ggml_backend_tensor_set(idx, &slot, 0, 4); // placeholder, set below properly
        float fslot = (float) slot;
        ggml_backend_tensor_set(idx, &fslot, 0, 4);
        ggml_backend_tensor_set(ones, on.data(), 0, S*4);
        ggml_backend_tensor_set(row, rw.data(), 0, R*4);
        ggml_backend_tensor_set(cache, ca.data(), 0, R*S*4);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("mask FAILED on %s\n", sn); _exit(1); }
        ggml_backend_synchronize(be);
        std::vector<float> got(R*S);
        ggml_backend_tensor_get(out, got.data(), 0, R*S*4);
        int bad = 0; double mr = 0;
        for (int i = 0; i < R*S; i++) { double dd = fabs(got[i]-want[i]); double dn = std::max(1.0, (double)fabs(want[i])); if (dd/dn > mr) mr = dd/dn; if (dd/dn > 1e-2 && dd > 1e-3) bad++; }
        printf("%s-mask %s max_rel=%.2e bad=%d\n", sn, bad?"FAIL":"PASS", mr, bad);
    }
    _exit(0);
}
