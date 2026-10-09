// minimal end-to-end probe for the CANNGE backend: builds small cgraphs,
// runs them through the compiled GE path and compares against CPU references
#include "ggml.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

static void report(const char * name, const float * got, const float * want, int n) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        max_abs = std::max(max_abs, d);
        if (std::fabs((double) want[i]) > 1e-6) {
            max_rel = std::max(max_rel, d / std::fabs((double) want[i]));
        }
    }
    const bool pass = max_abs < 1e-2 || max_rel < 1e-2;
    printf("%-12s %s  max_abs=%.3e max_rel=%.3e n=%d\n", name, pass ? "PASS" : "FAIL", max_abs, max_rel, n);
    if (!pass) {
        for (int i = 0; i < n && i < 8; i++) {
            printf("    [%d] got=%f want=%f\n", i, got[i], want[i]);
        }
        n_fail++;
    } else {
        n_pass++;
    }
}

static ggml_backend_t g_backend = nullptr;
static struct ggml_context * g_ctx = nullptr;

static void fill(ggml_tensor * t, const std::vector<float> & v) {
    ggml_backend_alloc_ctx_tensors(g_ctx, g_backend); // no-op once allocated
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

int main() {
    ggml_backend_t backend = ggml_backend_cannge_init(0);
    g_backend = backend;
    if (backend == nullptr) {
        printf("FATAL: no CANNGE backend\n");
        return 1;
    }
    printf("backend: %s\n", "CANNGE");

    struct ggml_init_params iparams = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(iparams);
    g_ctx = ctx;

    // ---- RMS_NORM: [4,2] eps=1e-5 ----
    {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        fill(x, { 1, 2, 3, 4, -1, -2, -1, 0.5f });
        ggml_tensor * out = ggml_rms_norm(ctx, x, 1e-5f);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("RMS_NORM FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[8];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            float want[8];
            const float v[8] = { 1, 2, 3, 4, -1, -2, -1, 0.5f };
            for (int m = 0; m < 2; m++) {
                double ss = 0.0;
                for (int k = 0; k < 4; k++) {
                    ss += (double) v[k + 4 * m] * v[k + 4 * m];
                }
                const float rms = (float) std::sqrt(ss / 4.0 + 1e-5);
                for (int k = 0; k < 4; k++) {
                    want[k + 4 * m] = v[k + 4 * m] / rms;
                }
            }
            report("rms_norm", got, want, 8);
        }
    }


    printf("summary: %d pass, %d fail\n", n_pass, n_fail);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return n_fail == 0 ? 0 : 2;
}
