// probe: GE execute fixed-overhead measurement. Repeatedly executes small
// CANNGE graphs to get per-execution latency (the per-split floor that
// dominates decode when the graph is fragmented).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>

using clk = std::chrono::steady_clock;

// mode 0: enqueue-only pipelined (host cost); mode 1: per-iter synchronize
// (true end-to-end latency incl. device execution, like the sched does)
static double bench(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * gf, int iters, int mode) {
    for (int i = 0; i < 3; i++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return -1;
    }
    ggml_backend_synchronize(backend);
    auto t0 = clk::now();
    for (int i = 0; i < iters; i++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return -1;
        if (mode == 1) ggml_backend_synchronize(backend);
    }
    ggml_backend_synchronize(backend);
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    if (!cannge) { printf("FATAL\n"); return 1; }
    struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };

    const int64_t D = 2560, NT = 1;
    const int iters = 200;

    // A: single-node MUL (norm weight mul, [2560,1])
    {
        struct ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(D * NT, 0.5f), wv(D, 0.25f);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
        ggml_tensor * out = ggml_mul(ctx, x, w);
        ggml_cgraph * gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        printf("A mul-1node      : pipe %.3f | e2e %.3f ms/exec\n", bench(cannge, ctx, gf, iters, 0), bench(cannge, ctx, gf, iters, 1));
        ggml_free(ctx);
    }
    // B: 2-node RMS+MUL
    {
        struct ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(D * NT, 0.5f), wv(D, 0.25f);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
        ggml_tensor * nm  = ggml_rms_norm(ctx, x, 1e-6f);
        ggml_tensor * out = ggml_mul(ctx, nm, w);
        ggml_cgraph * gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        printf("B rms+mul-2node  : pipe %.3f | e2e %.3f ms/exec\n", bench(cannge, ctx, gf, iters, 0), bench(cannge, ctx, gf, iters, 1));
        ggml_free(ctx);
    }
    // C: gemv [2560,2560]x[2560,1] (2-op Mul+ReduceSum + casts)
    {
        struct ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, D);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, NT);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<ggml_fp16_t> w16(D * D, ggml_fp32_to_fp16(0.01f)), x16(D * NT, ggml_fp32_to_fp16(0.5f));
        ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
        ggml_backend_tensor_set(x, x16.data(), 0, x16.size() * 2);
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        printf("C gemv-2560      : pipe %.3f | e2e %.3f ms/exec\n", bench(cannge, ctx, gf, 50, 0), bench(cannge, ctx, gf, 50, 1));
        ggml_free(ctx);
    }
    // D: whole-layer-ish chain: rms -> gemv -> rms -> gemv (4 nodes)
    {
        struct ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT);
        ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, D);
        ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, D);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(D * NT, 0.5f);
        std::vector<ggml_fp16_t> w1v(D * D, ggml_fp32_to_fp16(0.01f)), w2v(1024 * D, ggml_fp32_to_fp16(0.01f));
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        ggml_backend_tensor_set(w1, w1v.data(), 0, w1v.size() * 2);
        ggml_backend_tensor_set(w2, w2v.data(), 0, w2v.size() * 2);
        ggml_tensor * nm1 = ggml_rms_norm(ctx, x, 1e-6f);
        ggml_tensor * h1  = ggml_mul_mat(ctx, w1, nm1); // f32 [2560,1]... dtype mismatch f16 w x f32 in -> gate rejects mixed
        ggml_cgraph * gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, h1);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        double t = bench(cannge, ctx, gf, 50, 0);
        double u = bench(cannge, ctx, gf, 50, 1);
        printf("D rms+gemv mixed : pipe %.3f | e2e %.3f (%s)\n", t, u, t < 0 ? "REJECTED" : "ok");
        ggml_free(ctx);
    }
    fflush(stdout);
    _exit(0);
}
