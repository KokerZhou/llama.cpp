// probe: GET_ROWS x2 + ADD merged sub-graph (the t1 corruption suspect)
// g1 = gather(table, idx1), g2 = gather(table, idx2), out = g1 + g2
// compare CANNGE vs CPU reference. also single-gather + ADD as control.
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

    const int64_t D = 2560, NV = 151936, NT = 1;

    // host-side table data (same for both backends)
    std::vector<ggml_fp16_t> table16(D * NV);
    for (size_t i = 0; i < table16.size(); i++) {
        table16[i] = ggml_fp32_to_fp16(-0.5f + 0.0001f * (float) (i % 9973) + 0.003f * (float) (i % 61));
    }
    std::vector<int32_t> idx1 = { 1234 }, idx2 = { 5678 };

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        printf("== backend %s ==\n", side == 0 ? "CANNGE" : "CPU");

        for (int variant = 0; variant < 2; variant++) {
        // variant 0: all tensors backend-resident (direct bind)
        // variant 1: idx + residual host-resident (exercises input_staging path,
        //             same as the model where token ids and CPU-side activations
        //             are graph inputs in host buffers)
        struct ggml_init_params iparams = { 64 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(iparams);
        struct ggml_context * hctx = ggml_init(iparams);

        ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, NV);
        ggml_tensor * tidx1 = variant == 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NT) : ggml_new_tensor_1d(hctx, GGML_TYPE_I32, NT);
        ggml_tensor * tidx2 = variant == 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NT) : ggml_new_tensor_1d(hctx, GGML_TYPE_I32, NT);
        ggml_tensor * res   = variant == 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT) : ggml_new_tensor_2d(hctx, GGML_TYPE_F32, D, NT);
        ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_alloc_ctx_tensors(hctx, cpu);
        ggml_backend_tensor_set(table, table16.data(), 0, table16.size() * 2);
        ggml_backend_tensor_set(tidx1, idx1.data(), 0, (size_t) NT * 4);
        ggml_backend_tensor_set(tidx2, idx2.data(), 0, (size_t) NT * 4);
        std::vector<float> resv(D * NT);
        for (int i = 0; i < D * NT; i++) resv[i] = 0.01f * (float) (i % 251);
        ggml_backend_tensor_set(res, resv.data(), 0, resv.size() * 4);

        ggml_tensor * g1  = ggml_get_rows(ctx, table, tidx1); // f32 out (ggml_get_rows forces F32)
        ggml_tensor * g2  = ggml_get_rows(ctx, table, tidx2);
        ggml_tensor * add = ggml_add(ctx, g1, g2);
        ggml_tensor * out = ggml_add(ctx, add, res);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("graph_compute FAILED on %s variant %d\n", side == 0 ? "CANNGE" : "CPU", variant);
            n_fail++;
            ggml_free(hctx);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(backend);

        std::vector<float> want(D * NT);
        for (int d = 0; d < D; d++) {
            want[d] = ggml_fp16_to_fp32(table16[(size_t) idx1[0] * D + d]) +
                      ggml_fp16_to_fp32(table16[(size_t) idx2[0] * D + d]) + resv[d];
        }

        // execute 3x with rotating idx: plan-cache hit path must rebind fresh
        // idx values (model-scale decode changes token ids every step)
        std::vector<std::pair<int32_t, int32_t>> rounds = { { 1234, 5678 }, { 42, 99999 }, { 151935, 7 } };
        for (size_t r = 0; r < rounds.size(); r++) {
            const int32_t i1 = rounds[r].first, i2 = rounds[r].second;
            ggml_backend_tensor_set(tidx1, &i1, 0, (size_t) NT * 4);
            ggml_backend_tensor_set(tidx2, &i2, 0, (size_t) NT * 4);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("round %zu compute FAILED on %s\n", r, side == 0 ? "CANNGE" : "CPU");
                n_fail++;
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(D * NT);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            for (int d = 0; d < D; d++) {
                want[d] = ggml_fp16_to_fp32(table16[(size_t) i1 * D + d]) +
                          ggml_fp16_to_fp32(table16[(size_t) i2 * D + d]) + resv[d];
            }
            char nm[64]; snprintf(nm, sizeof nm, "%s-g2a-v%d-r%zu", side == 0 ? "cannge" : "cpu", variant, r);
            report(nm, got.data(), want.data(), D * NT);
        }

        ggml_free(hctx);
        ggml_free(ctx);
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
