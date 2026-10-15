// probe: ROPE exact model shape/params — [128,32,T] f32 NEOX n_dims=128
// freq_base=1e6 (Qwen3-Instruct-2507), pos 0..T-1. CANNGE vs CPU.
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
        if (d > 1e-3) bad++;
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
    if (!cannge || !cpu) { printf("FATAL\n"); return 1; }

    const int64_t D = 128, H = 32;
    const int64_t Ts[] = { 1, 15 };
    const float freq_base = 1000000.0f;

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        const char * sn = side == 0 ? "cannge" : "cpu";
        printf("== backend %s ==\n", sn);
        for (int64_t T : Ts) {
            struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(ip);
            ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
            ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<float> xv(D * H * T);
            for (int64_t t = 0; t < T; t++)
                for (int64_t h = 0; h < H; h++)
                    for (int64_t d = 0; d < D; d++)
                        xv[(t * H + h) * D + d] = 0.3f * sinf(0.011f * (d + 7 * h + 3 * t)) + 0.02f * (t + 1);
            std::vector<int32_t> pv(T);
            for (int64_t t = 0; t < T; t++) pv[t] = (int32_t) t;
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
            ggml_backend_tensor_set(pos, pv.data(), 0, pv.size() * 4);

            ggml_tensor * out = ggml_rope_ext(ctx, x, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, 0,
                                              freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("rope T=%lld FAILED on %s\n", (long long) T, sn);
                n_fail++;
                ggml_free(ctx);
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(D * H * T);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            // CPU reference (ggml semantics: NEOX = rotate_half: pairs (d, d+half))
            std::vector<float> want(D * H * T);
            for (int64_t t = 0; t < T; t++) {
                const float p = (float) pv[t];
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t k = 0; k < D / 2; k++) {
                        const float fr = powf(freq_base, -2.0f * (float) k / (float) D);
                        const float th = p * fr;
                        const float c = cosf(th), s = sinf(th);
                        const float x1 = xv[(t * H + h) * D + k];
                        const float x2 = xv[(t * H + h) * D + k + D / 2];
                        want[(t * H + h) * D + k]           = x1 * c - x2 * s;
                        want[(t * H + h) * D + k + D / 2]   = x2 * c + x1 * s;
                    }
                }
            }
            char nm[64];
            snprintf(nm, sizeof nm, "%s-rope-T%lld", sn, (long long) T);
            report(nm, got.data(), want.data(), (int) (D * H * T));
            ggml_free(ctx);
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
