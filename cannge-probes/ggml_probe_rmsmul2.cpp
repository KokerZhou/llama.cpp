// probe: MUL([128,32,2], [128,1]) - Qwen3 per-head norm weight broadcast,
// the [.,1] degenerate operand class (same family as the get_rows #19 bug).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>

static int n_pass = 0;
static int n_fail = 0;

static void report(const char * name, const float * got, const float * want, int n) {
    double max_abs = 0.0;
    int    bad     = 0;
    for (int i = 0; i < n; i++) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        if (d > max_abs) max_abs = d;
        if (d > 2e-2) bad++;
    }
    printf("%-24s %s  max_abs=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_abs, bad, n);
    if (bad) {
        for (int i = 0; i < n && i < 6; i++) printf("    [%d] got=%f want=%f\n", i, got[i], want[i]);
        n_fail++;
    } else {
        n_pass++;
    }
}

int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cannge || !cpu) { printf("FATAL: backend init failed\n"); return 1; }

    const int64_t D = 2560, H = 1, NT = 15;

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        const char * sn = side == 0 ? "cannge" : "cpu";
        printf("== backend %s ==\n", sn);

        // rms_norm(x) then mul by [128,1] weight (the exact llama q_norm chain)
        for (int variant = 0; variant < 2; variant++) {
            struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx  = ggml_init(ip);
            struct ggml_context * hctx = ggml_init(ip);
            ggml_tensor * x = variant == 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT)
                                           : ggml_new_tensor_2d(hctx, GGML_TYPE_F32, D, NT);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_alloc_ctx_tensors(hctx, cpu);
            std::vector<float> xv(D * H * NT), wv(D);
            for (int64_t t = 0; t < H * NT; t++)
                for (int64_t d = 0; d < D; d++)
                    xv[t * D + d] = 0.3f * sinf(0.01f * (d + 13 * t)) + 0.01f * (t + 1);
            for (int64_t d = 0; d < D; d++) wv[d] = 0.5f + 0.001f * (d % 251);
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
            ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);

            ggml_tensor * nm  = ggml_rms_norm(ctx, x, 1e-6f);
            ggml_tensor * out = ggml_mul(ctx, nm, w);
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("rms+mul v%d FAILED on %s\n", variant, sn);
                n_fail++;
                ggml_free(hctx); ggml_free(ctx);
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(D * H * NT);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            std::vector<float> want(D * H * NT);
            for (int64_t t = 0; t < H * NT; t++) {
                double ss = 0.0;
                for (int64_t d = 0; d < D; d++) ss += (double) xv[t * D + d] * xv[t * D + d];
                const float scale = 1.0f / sqrtf((float) (ss / D) + 1e-6f);
                for (int64_t d = 0; d < D; d++) want[t * D + d] = xv[t * D + d] * scale * wv[d];
            }
            char nm2[80];
            snprintf(nm2, sizeof nm2, "%s-rmsmul-%d", sn, variant);
            report(nm2, got.data(), want.data(), (int) (D * H * NT));
            ggml_free(hctx);
            ggml_free(ctx);
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
