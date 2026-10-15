// probe: merged-attention pattern - in-fragment KV cache read + fresh-row
// splice. cache {D*G,S} f16, SET_ROWS writes slot n, K/V chains read in-graph,
// kq (GQA) + softmax + kqv. Reference: same graph on CPU (host scatter exact).
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
    double max_rel = 0.0;
    int    bad     = 0;
    for (int i = 0; i < n; i++) {
        const double d   = std::fabs((double) got[i] - (double) want[i]);
        const double den = std::max(1.0, std::fabs((double) want[i]));
        if (d / den > max_rel) max_rel = d / den;
        if (d / den > 1e-2 && d > 1e-3) bad++;
    }
    printf("%-26s %s  max_rel=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_rel, bad, n);
    if (bad) {
        for (int i = 0; i < n && i < 6; i++) printf("    [%d] got=%.6f want=%.6f\n", i, got[i], want[i]);
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

    const int64_t D = 128, G = 8, S = 16, H = 32;
    const int64_t ROW = D * G; // 1024
    const float   scale = 1.0f / sqrtf((float) D);

    struct ggml_init_params ip = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);

    ggml_tensor * ck    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ROW, S); // K cache
    ggml_tensor * cv    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ROW, S); // V cache
    ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, H);
    ggml_tensor * mask  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, 1, 1);
    ggml_tensor * kdata = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROW, 1); // Kcur row
    ggml_tensor * vdata = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROW, 1); // Vcur row
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    int32_t       slot  = 5;
    ggml_tensor * kdst  = ggml_view_2d(ctx, ck, ROW, S, ck->nb[1], 0);
    ggml_tensor * vdst  = ggml_view_2d(ctx, cv, ROW, S, cv->nb[1], 0);

    // K read chain: view3 {D,G,S} -> permute {D,S,G}
    ggml_tensor * k3  = ggml_view_3d(ctx, ck, D, G, S, ck->nb[1], (size_t) ROW * 2, 0);
    ggml_tensor * kp  = ggml_permute(ctx, k3, 0, 2, 1, 3);            // {D,S,G}
    ggml_tensor * kq  = ggml_mul_mat(ctx, kp, q);                      // {S,1,H}
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    ggml_set_output(kq);
    ggml_tensor * kqs = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    // V read chain: view3 -> permute {D,S,G} -> transpose {S,D,G} -> cont
    ggml_tensor * v3   = ggml_view_3d(ctx, cv, D, G, S, cv->nb[1], (size_t) ROW * 2, 0);
    ggml_tensor * vp   = ggml_permute(ctx, v3, 0, 2, 1, 3);           // {D,S,G}
    ggml_tensor * vt   = ggml_transpose(ctx, vp);                     // {S,D,G}
    ggml_tensor * vc   = ggml_cont(ctx, vt);                          // dense {S,D,G}
    ggml_tensor * kqv  = ggml_mul_mat(ctx, vc, kqs);                  // {D,1,H}

    // reference (CPU, doubles): scatter rows then attention over the full cache
    std::vector<ggml_fp16_t> ck16(ROW * S), cv16(ROW * S);
    std::vector<float>       q32(D * H), mask32(S), kd32(ROW), vd32(ROW);
    for (int64_t i = 0; i < ROW * S; i++) {
        ck16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f);
        cv16[i] = ggml_fp32_to_fp16(0.03f * (float) (((int) (i * 2246822519u) % 83) - 41) / 41.0f);
    }
    for (int64_t i = 0; i < D * H; i++) q32[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
    for (int64_t i = 0; i < S; i++) mask32[i] = i < 10 ? 0.0f : -INFINITY;
    for (int64_t i = 0; i < ROW; i++) {
        kd32[i] = 0.4f * sinf(0.05f * (float) i) + 1.0f;
        vd32[i] = 0.6f * cosf(0.04f * (float) i) - 0.5f;
    }
    std::vector<ggml_fp16_t> rk16 = ck16, rv16 = cv16;
    for (int64_t e = 0; e < ROW; e++) {
        rk16[(size_t) slot * ROW + e] = ggml_fp32_to_fp16(kd32[e]);
        rv16[(size_t) slot * ROW + e] = ggml_fp32_to_fp16(vd32[e]);
    }
    std::vector<float> want_kq(S * H), want_kqv(D * H);
    for (int64_t h = 0; h < H; h++) {
        const int64_t g = h / (H / G);
        for (int64_t s = 0; s < S; s++) {
            double acc = 0;
            for (int64_t d = 0; d < D; d++)
                acc += ggml_fp16_to_fp32(rk16[d + D * g + ROW * s]) * (double) q32[d + D * h];
            want_kq[s + S * h] = (float) acc;
        }
    }
    for (int64_t h = 0; h < H; h++) {
        const int64_t g = h / (H / G);
        float mx = -INFINITY;
        for (int64_t s = 0; s < S; s++) mx = std::max(mx, want_kq[s + S * h] * scale + mask32[s]);
        double sum = 0;
        for (int64_t s = 0; s < S; s++) sum += expf(want_kq[s + S * h] * scale + mask32[s] - mx);
        for (int64_t d = 0; d < D; d++) {
            double acc = 0;
            for (int64_t s = 0; s < S; s++) {
                const float p = expf(want_kq[s + S * h] * scale + mask32[s] - mx) / (float) sum;
                acc += ggml_fp16_to_fp32(rv16[d + D * g + ROW * s]) * p;
            }
            want_kqv[d + D * h] = (float) acc;
        }
    }

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        const char * sn = side == 0 ? "cannge" : "cpu";
        ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(ck, ck16.data(), 0, ck16.size() * 2);
        ggml_backend_tensor_set(cv, cv16.data(), 0, cv16.size() * 2);
        ggml_backend_tensor_set(q, q32.data(), 0, q32.size() * 4);
        ggml_backend_tensor_set(mask, mask32.data(), 0, mask32.size() * 4);
        ggml_backend_tensor_set(kdata, kd32.data(), 0, kd32.size() * 4);
        ggml_backend_tensor_set(vdata, vd32.data(), 0, vd32.size() * 4);
        ggml_backend_tensor_set(idx, &slot, 0, 4);

        ggml_tensor * srk = ggml_set_rows(ctx, kdst, kdata, idx);
        ggml_tensor * srv = ggml_set_rows(ctx, vdst, vdata, idx);
        ggml_cgraph * gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, srk);
        ggml_build_forward_expand(gf, srv);
        ggml_build_forward_expand(gf, kqv);
        ggml_build_forward_expand(gf, kq);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("kvattn FAILED on %s\n", sn);
            n_fail++;
            continue;
        }
        ggml_backend_synchronize(backend);
        std::vector<float> g_kq(S * H), g_kqv(D * H);
        ggml_backend_tensor_get(kq, g_kq.data(), 0, g_kq.size() * 4);
        ggml_backend_tensor_get(kqv, g_kqv.data(), 0, g_kqv.size() * 4);
        char nm[64];
        snprintf(nm, sizeof nm, "%s-kvattn-kq", sn);
        report(nm, g_kq.data(), want_kq.data(), (int) g_kq.size());
        snprintf(nm, sizeof nm, "%s-kvattn-kqv", sn);
        report(nm, g_kqv.data(), want_kqv.data(), (int) g_kqv.size());
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
