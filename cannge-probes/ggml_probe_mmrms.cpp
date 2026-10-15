// probe #20: mm(f16 512x512) -> rms chain vs separated graphs, CANNGE vs CPU.
// small weights pass the 256K gate; tolerance matches f16-accumulate reality.
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

// rel-or-abs tolerance: f16 matmul accumulate needs ~2% rel
static void report(const char * name, const float * got, const float * want, int n) {
    double max_rel = 0.0;
    int    bad     = 0;
    for (int i = 0; i < n; i++) {
        const double d   = std::fabs((double) got[i] - (double) want[i]);
        const double den = std::max(1.0, std::fabs((double) want[i]));
        const double r   = d / den;
        if (r > max_rel) max_rel = r;
        if (r > 2e-2 && d > 2e-2) bad++;
    }
    printf("%-20s %s  max_rel=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_rel, bad, n);
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

    const int64_t K = 512, D = 512, NT = 15;

    std::vector<ggml_fp16_t> w16(K * D), x16(K * NT);
    for (size_t i = 0; i < w16.size(); i++) w16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f);
    for (int i = 0; i < K * NT; i++) x16[i] = ggml_fp32_to_fp16(0.5f * sinf(0.03f * i) + 0.01f * (i / K));

    // fp32 reference of the matmul (CPU side truth, same data)
    std::vector<float> mmv(D * NT, 0.0f);
    for (int n = 0; n < D; n++)
        for (int t = 0; t < NT; t++)
            for (int k = 0; k < K; k++)
                mmv[n * NT + t] += ggml_fp16_to_fp32(w16[(size_t) k * D + n]) * ggml_fp16_to_fp32(x16[(size_t) k * NT + t]);
    std::vector<float> want_rms(D * NT);
    {
        for (int t = 0; t < NT; t++) {
            double ss = 0.0;
            for (int d = 0; d < D; d++) ss += (double) mmv[d * NT + t] * mmv[d * NT + t];
            const float scale = 1.0f / sqrtf((float) (ss / D) + 1e-6f);
            for (int d = 0; d < D; d++) want_rms[d * NT + t] = mmv[d * NT + t] * scale;
        }
    }

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        printf("== backend %s ==\n", side == 0 ? "CANNGE" : "CPU");
        const char * sn = side == 0 ? "cannge" : "cpu";

        // A: fused graph mm -> rms
        {
            struct ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, D);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, NT);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
            ggml_backend_tensor_set(x, x16.data(), 0, x16.size() * 2);
            ggml_tensor * mm  = ggml_mul_mat(ctx, w, x);
            ggml_tensor * out = ggml_rms_norm(ctx, mm, 1e-6f);
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("fused mm-rms FAILED on %s\n", sn); n_fail++; ggml_free(ctx); continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(D);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            char nm[64]; snprintf(nm, sizeof nm, "%s-fused-mm-rms", sn);
            report(nm, got.data(), want_rms.data(), D);
            ggml_free(ctx);
        }

        // B: mm alone (control)
        {
            struct ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, D);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, NT);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
            ggml_backend_tensor_set(x, x16.data(), 0, x16.size() * 2);
            ggml_tensor * out = ggml_mul_mat(ctx, w, x);
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("mm FAILED on %s\n", sn); n_fail++; ggml_free(ctx); continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(D);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            char nm[64]; snprintf(nm, sizeof nm, "%s-mm", sn);
            report(nm, got.data(), mmv.data(), D);
            ggml_free(ctx);
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
