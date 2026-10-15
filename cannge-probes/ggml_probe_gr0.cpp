// probe: the exact t1 corruption-suspect sub-graph:
//   g1 = get_rows(node_1432_like [2560,1], zeros[1])
//   g2 = get_rows(l_out_like  [2560,1], zeros[1])
//   out = g1 + g2     (ffn_inp-35 shape)
// gathers run on single-row tables with idx=0 (identity), like the model.
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
        if (d > 1e-2) bad++;
    }
    printf("%-18s %s  max_abs=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_abs, bad, n);
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

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        printf("== backend %s ==\n", side == 0 ? "CANNGE" : "CPU");

        struct ggml_init_params iparams = { 16 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(iparams);

        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1); // node_1432-like
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1); // l_out-34-like
        ggml_tensor * z = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);    // leaf_473-like (value 0)
        ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> av(D), bv(D);
        for (int i = 0; i < D; i++) {
            av[i] = 0.5f + 0.001f * (i % 977);
            bv[i] = -1.0f + 0.002f * (i % 613);
        }
        int32_t zero = 0;
        ggml_backend_tensor_set(a, av.data(), 0, av.size() * 4);
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * 4);
        ggml_backend_tensor_set(z, &zero, 0, 4);

        ggml_tensor * g1  = ggml_get_rows(ctx, a, z); // [2560,1]
        ggml_tensor * g2  = ggml_get_rows(ctx, b, z);
        ggml_tensor * out = ggml_add(ctx, g1, g2);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend); // allocate op outputs

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("graph_compute FAILED on %s\n", side == 0 ? "CANNGE" : "CPU");
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(backend);

        std::vector<float> got(D);
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
        std::vector<float> want(D);
        for (int i = 0; i < D; i++) want[i] = av[i] + bv[i];

        char nm[64]; snprintf(nm, sizeof nm, "%s-gr0-add", side == 0 ? "cannge" : "cpu");
        report(nm, got.data(), want.data(), D);

        ggml_free(ctx);
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
