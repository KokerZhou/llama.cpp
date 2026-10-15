// probe: ADD -> RMS_NORM chain (the m2 corrupting composition), staged inputs.
// (A) add(a,b) -> rms ; (B) add -> rms -> mul(w) ; shapes [2560,15]/[2560,1].
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

    const int64_t D = 2560;
    const int64_t NTs[] = { 15, 1 };

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        const char * sn = side == 0 ? "cannge" : "cpu";
        printf("== backend %s ==\n", sn);
        for (int64_t NT : NTs) {
            // (A) add -> rms, staged host inputs
            {
                struct ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
                struct ggml_context * ctx  = ggml_init(ip);
                struct ggml_context * hctx = ggml_init(ip);
                ggml_tensor * a = ggml_new_tensor_2d(hctx, GGML_TYPE_F32, D, NT);
                ggml_tensor * b = ggml_new_tensor_2d(hctx, GGML_TYPE_F32, D, NT);
                ggml_backend_alloc_ctx_tensors(ctx, backend);
                ggml_backend_alloc_ctx_tensors(hctx, cpu);
                std::vector<float> av(D * NT), bv(D * NT);
                for (int64_t i = 0; i < D * NT; i++) {
                    av[i] = 0.4f * sinf(0.013f * i) + 0.02f * (i / D);
                    bv[i] = 0.7f * cosf(0.007f * i) - 0.01f * (i / D);
                }
                std::vector<float> wv(D);
                for (int64_t d = 0; d < D; d++) wv[d] = 0.5f + 0.001f * (d % 251);
                ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
                ggml_backend_tensor_set(a, av.data(), 0, av.size() * 4);
                ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * 4);
                ggml_tensor * add = ggml_add(ctx, a, b);
                ggml_tensor * nmz = ggml_rms_norm(ctx, add, 1e-6f);
                ggml_tensor * out = ggml_mul(ctx, nmz, w);
                ggml_cgraph * gf  = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                ggml_backend_alloc_ctx_tensors(ctx, backend);
                ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                    printf("add-rms-mul NT=%lld FAILED on %s\n", (long long) NT, sn);
                    n_fail++;
                    ggml_free(hctx);
                    ggml_free(ctx);
                    continue;
                }
                ggml_backend_synchronize(backend);
                std::vector<float> got(D * NT);
                ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
                std::vector<float> want(D * NT);
                for (int64_t t = 0; t < NT; t++) {
                    double ss = 0.0;
                    for (int64_t d = 0; d < D; d++) {
                        const float s = av[t * D + d] + bv[t * D + d];
                        ss += (double) s * s;
                    }
                    const float scale = 1.0f / sqrtf((float) (ss / D) + 1e-6f);
                    for (int64_t d = 0; d < D; d++) want[t * D + d] = (av[t * D + d] + bv[t * D + d]) * scale * wv[d];
                }
                char nm[80];
                snprintf(nm, sizeof nm, "%s-add-rms-mul-%d", sn, (int) NT);
                report(nm, got.data(), want.data(), (int) (D * NT));
                ggml_free(hctx);
                ggml_free(ctx);
            }
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
