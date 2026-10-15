// probe: BatchMatMulV3 with large weights (note-10 256K silent-zero cap recheck)
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>

int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    if (!cannge) { printf("FATAL\n"); return 1; }

    const int64_t K = 4096, N = 2560, M = 32; // attn_output-like, padded batch
    struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_backend_alloc_ctx_tensors(ctx, cannge);
    std::vector<ggml_fp16_t> w16(K * N);
    std::vector<float>       xf(K * M);
    for (int64_t i = 0; i < K * N; i++) w16[i] = ggml_fp32_to_fp16(0.01f * (float) (((int) (i * 2654435761u) % 97) - 48) / 48.0f);
    for (int64_t i = 0; i < K * M; i++) xf[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
    ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_alloc_ctx_tensors(ctx, cannge);
    if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) { printf("COMPUTE FAILED\n"); _exit(1); }
    ggml_backend_synchronize(cannge);
    std::vector<float> got(N * M);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
    double max_rel = 0; int bad = 0;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            double want = 0;
            for (int64_t k = 0; k < K; k++) want += ggml_fp16_to_fp32(w16[k + K * n]) * xf[k + K * m];
            const double d = std::fabs(got[n + N * m] - want);
            if (d / std::max(1.0, std::fabs(want)) > 1e-2 && d > 1e-2) bad++;
            if (d / std::max(1.0, std::fabs(want)) > max_rel) max_rel = d / std::max(1.0, std::fabs(want));
        }
    printf("big-mm K=%lld N=%lld M=%lld: max_rel=%.3e bad=%d/%lld -> %s\n", (long long) K, (long long) N,
           (long long) M, max_rel, bad, (long long) (N * M), bad == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    _exit(bad != 0);
}
