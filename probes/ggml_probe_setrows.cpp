// probe: in-graph SET_ROWS (GE ScatterUpdate) + post-write view read, vs CPU.
// Model pattern: cache {1024,512} f16, scatter new rows (f32 data -> f16 cast),
// then read the cache through a PERMUTE view in the same graph.
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

static void report(const char * name, const ggml_fp16_t * got, const ggml_fp16_t * want, int n) {
    double max_rel = 0.0;
    int    bad     = 0;
    for (int i = 0; i < n; i++) {
        const double d   = std::fabs((double) ggml_fp16_to_fp32(got[i]) - (double) ggml_fp16_to_fp32(want[i]));
        const double den = std::max(1.0, std::fabs((double) ggml_fp16_to_fp32(want[i])));
        if (d / den > max_rel) max_rel = d / den;
        if (d / den > 1e-2 && d > 1e-3) bad++;
    }
    printf("%-26s %s  max_rel=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_rel, bad, n);
    if (bad) {
        for (int i = 0; i < n && i < 6; i++)
            printf("    [%d] got=%.6f want=%.6f\n", i, ggml_fp16_to_fp32(got[i]), ggml_fp16_to_fp32(want[i]));
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

    const int64_t R = 1024, S = 512, D = 128, G = 8;

    for (int nrows = 1; nrows <= 2; nrows++) {
        struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, R, S);
        ggml_tensor * data  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, nrows);
        ggml_tensor * idx   = ggml_new_tensor_1d(ctx, getenv("IDX64") ? GGML_TYPE_I64 : GGML_TYPE_I32, nrows);
        // read side: {D,G,S} view -> permute {D,S,G} (the K cache pattern)
        ggml_tensor * v3  = ggml_view_3d(ctx, cache, D, G, S, cache->nb[1], (size_t) R * 2, 0);
        ggml_tensor * pv  = ggml_permute(ctx, v3, 0, 2, 1, 3);  // {D,S,G} strided view
        ggml_tensor * out = ggml_cont(ctx, pv);                 // dense own buffer

        std::vector<ggml_fp16_t> cache16(R * S);
        std::vector<float>       data32(R * nrows);
        std::vector<int64_t>     idx32(nrows);
        for (int64_t i = 0; i < R * S; i++) cache16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f);
        for (int64_t i = 0; i < R * nrows; i++) data32[i] = 0.5f * sinf(0.03f * (float) i) + 3.0f;
        idx32[0] = getenv("SLOT0") ? 0 : 7;
        if (nrows == 2) idx32[1] = 300;

        std::vector<ggml_fp16_t> want(R * S);
        // CPU reference: scatter rows then expose the permuted view (bytes of cache)
        for (int64_t i = 0; i < R * S; i++) want[i] = cache16[i];
        for (int64_t r = 0; r < nrows; r++) {
            ggml_fp16_t * row = &want[(size_t) idx32[r] * R];
            for (int64_t e = 0; e < R; e++) row[e] = ggml_fp32_to_fp16(data32[r * R + e]);
        }

        char nm[64];
        for (int side = 0; side < 2; side++) {
            ggml_backend_t backend = side == 0 ? cannge : cpu;
            const char *   sn      = side == 0 ? "cannge" : "cpu";
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(cache, cache16.data(), 0, cache16.size() * 2);
            ggml_backend_tensor_set(data, data32.data(), 0, data32.size() * 4);
            ggml_backend_tensor_set(idx, idx32.data(), 0, idx32.size() * (getenv("IDX64") ? 8 : 4));

            ggml_tensor * dst = ggml_view_2d(ctx, cache, R, S, cache->nb[1], 0);
            ggml_tensor * sr  = ggml_set_rows(ctx, dst, data, idx);
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, sr);
            ggml_build_forward_expand(gf, out);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("set_rows FAILED on %s\n", sn);
                n_fail++;
                continue;
            }
            ggml_backend_synchronize(backend);
            // the cont output holds the {D,S,G} dense view: element (d,s,g) at
            // d + D*s + D*S*g, sourced from cache (d,g,s) at d + D*g + D*G*s
            const bool in_graph = getenv("GGML_CANNGE_SCATTER") != nullptr;
            std::vector<ggml_fp16_t> got(R * S), want_perm(R * S);
            for (int64_t g = 0; g < G; g++)
                for (int64_t s = 0; s < S; s++)
                    for (int64_t d = 0; d < D; d++)
                        want_perm[d + D * s + D * S * g] = want[d + D * g + D * G * s];
            if (in_graph) { // only meaningful with the in-graph write
                ggml_backend_tensor_get(out, got.data(), 0, got.size() * 2);
                snprintf(nm, sizeof nm, "%s-setrows-%dr", sn, nrows);
                report(nm, got.data(), want_perm.data(), (int) got.size());
            }
            // the cache buffer itself must hold the fresh row (writeback)
            std::vector<ggml_fp16_t> got2(R * S);
            ggml_backend_tensor_get(cache, got2.data(), 0, got2.size() * 2);
            snprintf(nm, sizeof nm, "%s-setrows-cache-%dr", sn, nrows);
            report(nm, got2.data(), want.data(), (int) got2.size());
        }
        ggml_free(ctx);
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
