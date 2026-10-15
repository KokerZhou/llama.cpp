// probe: RMS_NORM shape matrix - the [128,32]/[128,8] per-head q/k norm shapes
// used by Qwen3 are the h1-corruption suspects. CANNGE vs CPU, device+host inputs.
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
    printf("%-22s %s  max_abs=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_abs, bad, n);
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

    // {D, extra dims as a single multiplier} - 2D cases keep H*NT flat,
    // 3D cases listed separately below
    const int64_t shapes[][2] = { { 2560, 1 }, { 128, 32 }, { 128, 8 }, { 512, 1 }, { 2048, 1 }, { 128, 1 } };
    const int64_t NTs[]       = { 1, 15 };
    // 3D per-head norm shapes from Qwen3 attention: [head_dim, n_head, n_tok]
    const int64_t shapes3[][3] = { { 128, 32, 2 }, { 128, 8, 2 }, { 128, 32, 15 }, { 128, 8, 15 }, { 128, 32, 1 }, { 128, 8, 1 } };

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        printf("== backend %s ==\n", side == 0 ? "CANNGE" : "CPU");
        for (auto & sh3 : shapes3) {
            const int64_t D = sh3[0], H = sh3[1], NT3 = sh3[2];
            for (int variant = 0; variant < 2; variant++) {
                struct ggml_init_params iparams = { 16 * 1024 * 1024, nullptr, true };
                struct ggml_context * ctx  = ggml_init(iparams);
                struct ggml_context * hctx = ggml_init(iparams);
                ggml_tensor * x = variant == 0 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, NT3)
                                               : ggml_new_tensor_3d(hctx, GGML_TYPE_F32, D, H, NT3);
                ggml_backend_alloc_ctx_tensors(ctx, backend);
                ggml_backend_alloc_ctx_tensors(hctx, cpu);
                const int64_t R = H * NT3;
                std::vector<float> xv(D * R);
                for (int64_t t = 0; t < R; t++)
                    for (int64_t d = 0; d < D; d++)
                        xv[t * D + d] = 0.3f * sinf(0.01f * (d + 13 * t)) + 0.01f * (t + 1);
                ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
                ggml_tensor * out = ggml_rms_norm(ctx, x, 1e-6f);
                ggml_cgraph * gf  = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                    printf("rms3 [%lld,%lld,%lld] v%d FAILED on %s\n", (long long) D, (long long) H, (long long) NT3,
                           variant, side == 0 ? "CANNGE" : "CPU");
                    n_fail++;
                    ggml_free(hctx); ggml_free(ctx);
                    continue;
                }
                ggml_backend_synchronize(backend);
                std::vector<float> got(D * R);
                ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
                std::vector<float> want(D * R);
                for (int64_t t = 0; t < R; t++) {
                    double ss = 0.0;
                    for (int64_t d = 0; d < D; d++) ss += (double) xv[t * D + d] * xv[t * D + d];
                    const float scale = 1.0f / sqrtf((float) (ss / D) + 1e-6f);
                    for (int64_t d = 0; d < D; d++) want[t * D + d] = xv[t * D + d] * scale;
                }
                char nm[80];
                snprintf(nm, sizeof nm, "%s-rms3-%lldx%lldx%lld-v%d", side == 0 ? "cannge" : "cpu",
                         (long long) D, (long long) H, (long long) NT3, variant);
                report(nm, got.data(), want.data(), (int) (D * R));
                ggml_free(hctx);
                ggml_free(ctx);
            }
        }
        for (auto & sh : shapes) {
            const int64_t D = sh[0], H = sh[1];
            for (int64_t NT : NTs) {
                for (int variant = 0; variant < 2; variant++) {
                    struct ggml_init_params iparams = { 16 * 1024 * 1024, nullptr, true };
                    struct ggml_context * ctx  = ggml_init(iparams);
                    struct ggml_context * hctx = ggml_init(iparams);
                    ggml_tensor * x = variant == 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, H * NT)
                                                   : ggml_new_tensor_2d(hctx, GGML_TYPE_F32, D, H * NT);
                    ggml_backend_alloc_ctx_tensors(ctx, backend);
                    ggml_backend_alloc_ctx_tensors(hctx, cpu);
                    std::vector<float> xv(D * H * NT);
                    for (int64_t t = 0; t < H * NT; t++)
                        for (int64_t d = 0; d < D; d++)
                            xv[t * D + d] = 0.3f * sinf(0.01f * (d + 13 * t)) + 0.01f * (t + 1);
                    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);

                    ggml_tensor * out = ggml_rms_norm(ctx, x, 1e-6f);
                    ggml_cgraph * gf  = ggml_new_graph(ctx);
                    ggml_build_forward_expand(gf, out);
                    ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                        printf("rms [%lld,%lld] v%d FAILED on %s\n", (long long) D, (long long) (H * NT), variant,
                               side == 0 ? "CANNGE" : "CPU");
                        n_fail++;
                        ggml_free(hctx);
                        ggml_free(ctx);
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
                        for (int64_t d = 0; d < D; d++) want[t * D + d] = xv[t * D + d] * scale;
                    }
                    char nm[80];
                    snprintf(nm, sizeof nm, "%s-rms-%lldx%lld-v%d", side == 0 ? "cannge" : "cpu", (long long) D,
                             (long long) (H * NT), variant);
                    report(nm, got.data(), want.data(), (int) (D * H * NT));
                    ggml_free(hctx);
                    ggml_free(ctx);
                }
            }
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
