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

    // ---- ADD: [4] + [4], then re-run with new data (plan cache hit path) ----
    {
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        fill(a, { 1, 2, 3, 4 });
        fill(b, { 10, 20, 30, 40 });
        ggml_tensor * out = ggml_add(ctx, a, b);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("ADD FAIL: graph_compute returned error (first run)\n");
            n_fail++;
        } else {
            auto t1 = std::chrono::steady_clock::now();
            ggml_backend_synchronize(backend);
            float got[4];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            const float want[4] = { 11, 22, 33, 44 };
            report("add", got, want, 4);
            printf("add first-call time: %.1f ms\n",
                   std::chrono::duration<double, std::milli>(t1 - t0).count());

            // second execution with fresh data: same plan, rebinding
            fill(a, { 5, 6, 7, 8 });
            fill(b, { 1, 1, 1, 1 });
            auto t2 = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("ADD FAIL: graph_compute returned error (second run)\n");
                n_fail++;
            } else {
                auto t3 = std::chrono::steady_clock::now();
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(out, got, 0, sizeof(got));
                const float want2[4] = { 6, 7, 8, 9 };
                report("add(2nd)", got, want2, 4);
                printf("add second-call time: %.1f ms\n",
                       std::chrono::duration<double, std::milli>(t3 - t2).count());
            }
        }
    }

    // ---- MUL_MAT F32: w[16,16] x x[16,16] -> [16,16] (310P matmul needs 32B-aligned dims) ----
    {
        const int K = 16, N = 16, M = 16;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        std::vector<float> wv(K * N);
        std::vector<float> xv(K * M);
        for (int i = 0; i < K * N; i++) {
            wv[i] = 0.1f * ((i * 7) % 13 - 6);
        }
        for (int i = 0; i < K * M; i++) {
            xv[i] = 0.1f * ((i * 5) % 11 - 5);
        }
        fill(w, wv);
        fill(x, xv);
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("MUL_MAT FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            auto t1 = std::chrono::steady_clock::now();
            ggml_backend_synchronize(backend);
            std::vector<float> got(N * M);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
            std::vector<float> want(N * M, 0.0f);
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    for (int k = 0; k < K; k++) {
                        want[n + N * m] += wv[k + K * n] * xv[k + K * m];
                    }
                }
            }
            // fp32 inputs are computed in fp16 on 310P (allow_fp32_to_fp16)
            double max_abs = 0.0, max_rel = 0.0;
            for (size_t i = 0; i < got.size(); i++) {
                const double d = std::fabs((double) got[i] - (double) want[i]);
                max_abs = std::max(max_abs, d);
                if (std::fabs((double) want[i]) > 1e-6) {
                    max_rel = std::max(max_rel, d / std::fabs((double) want[i]));
                }
            }
            const bool pass = max_abs < 5e-2 || max_rel < 2e-2;
            printf("%-12s %s  max_abs=%.3e max_rel=%.3e n=%d (fp16 compute)\n", "mul_mat_f32",
                   pass ? "PASS" : "FAIL", max_abs, max_rel, (int) got.size());
            if (pass) {
                n_pass++;
            } else {
                n_fail++;
            }
            printf("mul_mat first-call time: %.1f ms\n",
                   std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

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

    // ---- SOFT_MAX: [4,2] ----
    {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        fill(x, { 1, 2, 3, 4, 0.5f, -0.5f, 2, 0 });
        ggml_tensor * out = ggml_soft_max(ctx, x);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("SOFT_MAX FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[8];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            float want[8];
            const float v[8] = { 1, 2, 3, 4, 0.5f, -0.5f, 2, 0 };
            for (int m = 0; m < 2; m++) {
                double mx = v[0 + 4 * m];
                for (int k = 1; k < 4; k++) {
                    mx = std::max(mx, (double) v[k + 4 * m]);
                }
                double s = 0.0;
                for (int k = 0; k < 4; k++) {
                    s += std::exp((double) v[k + 4 * m] - mx);
                }
                for (int k = 0; k < 4; k++) {
                    want[k + 4 * m] = (float) (std::exp((double) v[k + 4 * m] - mx) / s);
                }
            }
            report("soft_max", got, want, 8);
        }
    }

    // ---- SILU: [4] ----
    {
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        fill(x, { -1, 0, 1, 2 });
        ggml_tensor * out = ggml_silu(ctx, x);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("SILU FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[4];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            float want[4];
            const float v[4] = { -1, 0, 1, 2 };
            for (int i = 0; i < 4; i++) {
                want[i] = v[i] / (1.0f + std::exp(-v[i]));
            }
            report("silu", got, want, 4);
        }
    }

    // ---- GET_ROWS: table [4,5] f32, indices [2] i32 {0,3} -> [4,2] ----
    {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5);
        std::vector<float> tv(20);
        for (int i = 0; i < 20; i++) {
            tv[i] = (float) (i + 1);
        }
        fill(t, tv);
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        const int32_t iv[2] = { 0, 3 };
        ggml_backend_alloc_ctx_tensors(g_ctx, g_backend);
        ggml_backend_tensor_set(idx, iv, 0, sizeof(iv));
        ggml_tensor * out = ggml_get_rows(ctx, t, idx);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("GET_ROWS FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[8];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            float want[8];
            for (int k = 0; k < 4; k++) {
                want[k] = tv[k + 4 * 0];
                want[k + 4] = tv[k + 4 * 3];
            }
            report("get_rows", got, want, 8);
        }
    }

    // ---- PERMUTE 3D non-self-inverse consumed by ADD: x[2,3,4] permute(1,2,0,3) -> ne {4,2,3} ----
    {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4);
        std::vector<float> xv(24);
        for (int i = 0; i < 24; i++) {
            xv[i] = (float) i;
        }
        fill(x, xv);
        ggml_tensor * p  = ggml_permute(ctx, x, 1, 2, 0, 3); // ggml move-dim: ne {4,2,3}
        ggml_tensor * c  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 3);
        fill(c, std::vector<float>(24, 1.0f));
        ggml_tensor * out = ggml_add(ctx, p, c);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("PERMUTE FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            std::vector<float> got(24);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
            std::vector<float> want(24);
            // out.ne = {4,2,3}; out[i0,i1,i2] = x[i1,i2,i0] + 1
            for (int i0 = 0; i0 < 4; i0++) {
                for (int i1 = 0; i1 < 2; i1++) {
                    for (int i2 = 0; i2 < 3; i2++) {
                        want[i0 + 4 * i1 + 8 * i2] = xv[i1 + 2 * i2 + 6 * i0] + 1.0f;
                    }
                }
            }
            report("permute", got.data(), want.data(), 24);
        }
    }

    // ---- VIEW with offset: base[8], view offset 4 elems -> add const [4] ----
    {
        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        fill(base, { 0, 1, 2, 3, 10, 20, 30, 40 });
        ggml_tensor * v = ggml_view_1d(ctx, base, 4, 4 * sizeof(float));
        ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        fill(c, { 1, 1, 1, 1 });
        ggml_tensor * out = ggml_add(ctx, v, c);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("VIEW FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[4];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            const float want[4] = { 11, 21, 31, 41 };
            report("view_offset", got, want, 4);
        }
    }

    // ---- CPY cast f32 -> f16 and back: round trip [4] ----
    {
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        fill(x, { 0.5f, -1.25f, 3.75f, 100.0f });
        ggml_tensor * h = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 4);
        ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_tensor * c1 = ggml_cpy(ctx, x, h);     // c1 is a view-of-h node
        ggml_tensor * c2 = ggml_cpy(ctx, c1, out);  // c2 is a view-of-out node

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c2);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("CPY FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[4];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            float want[4];
            const float v[4] = { 0.5f, -1.25f, 3.75f, 100.0f };
            for (int i = 0; i < 4; i++) {
                want[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[i]));
            }
            report("cpy_cast", got, want, 4);
        }
    }

    // ---- CONCAT: [2,3] + [2,3] along dim 0 -> [4,3] ----
    {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        fill(a, { 1, 2, 3, 4, 5, 6 });       // a[i0 + 2*i1]
        fill(b, { 10, 20, 30, 40, 50, 60 });
        ggml_tensor * out = ggml_concat(ctx, a, b, 0);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("CONCAT FAIL: graph_compute returned error\n");
            n_fail++;
        } else {
            ggml_backend_synchronize(backend);
            float got[12];
            ggml_backend_tensor_get(out, got, 0, sizeof(got));
            float want[12];
            const float av[6] = { 1, 2, 3, 4, 5, 6 };
            const float bv[6] = { 10, 20, 30, 40, 50, 60 };
            for (int i1 = 0; i1 < 3; i1++) {
                for (int i0 = 0; i0 < 2; i0++) {
                    want[i0 + 4 * i1] = av[i0 + 2 * i1];
                    want[i0 + 2 + 4 * i1] = bv[i0 + 2 * i1];
                }
            }
            report("concat", got, want, 12);
        }
    }

    printf("summary: %d pass, %d fail\n", n_pass, n_fail);

    ggml_free(ctx);
    ggml_backend_free(backend);
    return n_fail == 0 ? 0 : 2;
}
