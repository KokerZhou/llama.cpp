// probe: merged-attention fresh-row splice, model-faithful.
// NR=1: decode (single token, single scattered row, batched GQA gemv path).
// NR=2: prefill-like (two tokens, two scattered rows at the current position,
// BatchMatMulV3 attention path; the V-side matrix splice is under test).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    const int NR = argc > 1 ? atoi(argv[1]) : 1;
    const int SS = argc > 2 ? atoi(argv[2]) : 32; // slots

    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cannge || !cpu) { printf("FATAL: backend init failed\n"); return 1; }

    const int64_t D = 128, G = 8, S = SS, H = 32;
    const int64_t ROW = D * G;
    const float   scale = 1.0f / sqrtf((float) D);
    const int     NV   = 10;                 // old valid slots
    int32_t slots[64];
    for (int i = 0; i < NR; i++) slots[i] = NR == 1 ? 5 : S - NR + i;

    struct ggml_init_params ip = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);

    ggml_tensor * ck    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ROW, S);
    ggml_tensor * cv    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ROW, S);
    ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NR, H);
    ggml_tensor * mask  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, NR, 1);
    ggml_tensor * kdata = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROW, NR);
    ggml_tensor * vdata = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROW, NR);
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NR);
    ggml_tensor * kdst  = ggml_view_2d(ctx, ck, ROW, S, ck->nb[1], 0);
    ggml_tensor * vdst  = ggml_view_2d(ctx, cv, ROW, S, cv->nb[1], 0);

    ggml_tensor * k3  = ggml_view_3d(ctx, ck, D, G, S, ck->nb[1], (size_t) ROW * 2, 0);
    ggml_tensor * kp  = ggml_permute(ctx, k3, 0, 2, 1, 3);
    ggml_tensor * kq  = ggml_mul_mat(ctx, kp, q);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    ggml_tensor * kqs = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor * v3  = ggml_view_3d(ctx, cv, D, G, S, cv->nb[1], (size_t) ROW * 2, 0);
    ggml_tensor * vp  = ggml_permute(ctx, v3, 0, 2, 1, 3);
    ggml_tensor * vt  = ggml_transpose(ctx, vp);
    ggml_tensor * vc  = ggml_cont(ctx, vt);
    ggml_tensor * kqv = ggml_mul_mat(ctx, vc, kqs);
    if (NR == 1) {
        ggml_set_output(kq);
    } else {
        // GQA M>1 has no GE attention path (batch-dim mismatch, by design);
        // the prefill cont crosses to CPU. Check the cont bytes directly.
        ggml_set_output(vc);
    }

    std::vector<ggml_fp16_t> ck16(ROW * S), cv16(ROW * S);
    std::vector<float>       q32(D * NR * H), mask32(S * NR), kd32(ROW * NR), vd32(ROW * NR);
    for (int64_t i = 0; i < ROW * S; i++) {
        ck16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f);
        cv16[i] = ggml_fp32_to_fp16(0.03f * (float) (((int) (i * 2246822519u) % 83) - 41) / 41.0f);
    }
    for (int64_t i = 0; i < D * NR * H; i++) q32[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
    for (int64_t i = 0; i < ROW * NR; i++) {
        kd32[i] = 0.4f * sinf(0.05f * (float) i) + 1.0f;
        vd32[i] = 0.6f * cosf(0.04f * (float) i) - 0.5f;
    }
    for (int t = 0; t < NR; t++) {
        for (int64_t s = 0; s < S; s++) {
            mask32[s + S * t] = (s < NV || s >= S - NR + t) && s != S - 1 ? 0.0f : -INFINITY;
        }
    }
    std::vector<ggml_fp16_t> rk16 = ck16, rv16 = cv16;
    for (int r = 0; r < NR; r++) {
        for (int64_t e = 0; e < ROW; e++) {
            rk16[(size_t) slots[r] * ROW + e] = ggml_fp32_to_fp16(kd32[r * ROW + e]);
            rv16[(size_t) slots[r] * ROW + e] = ggml_fp32_to_fp16(vd32[r * ROW + e]);
        }
    }
    // reference: kq {S,NR,H} then per-token softmax then kqv {D,NR,H}
    std::vector<float> want_kq(S * NR * H, 0), want_kqv(D * NR * H, 0);
    for (int t = 0; t < NR; t++) {
        for (int64_t h = 0; h < H; h++) {
            const int64_t g = h / (H / G);
            for (int64_t s = 0; s < S; s++) {
                double acc = 0;
                for (int64_t d = 0; d < D; d++)
                    acc += ggml_fp16_to_fp32(rk16[d + D * g + ROW * s]) * (double) q32[d + D * (t + NR * h)];
                want_kq[s + S * (t + NR * h)] = (float) acc;
            }
            float mx = -INFINITY;
            for (int64_t s = 0; s < S; s++) mx = std::max(mx, want_kq[s + S * (t + NR * h)] * scale + mask32[s + S * t]);
            double sum = 0;
            for (int64_t s = 0; s < S; s++) sum += expf(want_kq[s + S * (t + NR * h)] * scale + mask32[s + S * t] - mx);
            for (int64_t d = 0; d < D; d++) {
                double acc = 0;
                for (int64_t s = 0; s < S; s++) {
                    const float p = expf(want_kq[s + S * (t + NR * h)] * scale + mask32[s + S * t] - mx) / (float) sum;
                    acc += ggml_fp16_to_fp32(rv16[d + D * g + ROW * s]) * p;
                }
                want_kqv[d + D * (t + NR * h)] = (float) acc;
            }
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
        ggml_backend_tensor_set(idx, slots, 0, NR * 4);

        ggml_tensor * srk = ggml_set_rows(ctx, kdst, kdata, idx);
        ggml_tensor * srv = ggml_set_rows(ctx, vdst, vdata, idx);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, srk);
        ggml_build_forward_expand(gf, srv);
        if (NR == 1) {
            ggml_build_forward_expand(gf, kqv);
            ggml_build_forward_expand(gf, kq);
        } else {
            ggml_build_forward_expand(gf, vc);
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("kvattn FAILED on %s\n", sn);
            n_fail++;
            continue;
        }
        ggml_backend_synchronize(backend);
        char nm[64];
        if (NR == 1) {
            std::vector<float> g_kq(S * H);
            ggml_backend_tensor_get(kq, g_kq.data(), 0, g_kq.size() * 4);
            snprintf(nm, sizeof nm, "%s-kvattn-kq", sn);
            report(nm, g_kq.data(), want_kq.data(), (int) g_kq.size());
        }
        if (NR == 1) {
            std::vector<float> g_kqv(D * NR * H);
            ggml_backend_tensor_get(kqv, g_kqv.data(), 0, g_kqv.size() * 4);
            snprintf(nm, sizeof nm, "%s-kvattn-kqv", sn);
            report(nm, g_kqv.data(), want_kqv.data(), (int) g_kqv.size());
        } else {
            // vc dense {S,D,G}: byte (s,d,g) = fresh rv16[d + D*g + ROW*s]
            std::vector<ggml_fp16_t> g_vc(S * D * G);
            ggml_backend_tensor_get(vc, g_vc.data(), 0, g_vc.size() * 2);
            // vc ggml {S,D,G}: linear = s + S*d + S*D*g (ne[0]=S innermost)
            std::vector<ggml_fp16_t> w_vc(S * D * G);
            for (int64_t g = 0; g < G; g++)
                for (int64_t d = 0; d < D; d++)
                    for (int64_t s = 0; s < S; s++)
                        w_vc[s + S * d + S * D * g] = rv16[d + D * g + ROW * s];
            double mr = 0;
            int bad = 0;
            for (size_t i = 0; i < g_vc.size(); i++) {
                double dd = std::fabs((double) ggml_fp16_to_fp32(g_vc[i]) - (double) ggml_fp16_to_fp32(w_vc[i]));
                if (dd > 1e-3) bad++;
                if (dd > mr) mr = dd;
            }
            snprintf(nm, sizeof nm, "%s-kvattn-cont", sn);
            printf("%-26s %s  max_abs=%.3e bad=%d/%zu\n", nm, bad == 0 ? "PASS" : "FAIL", mr, bad, g_vc.size());
            for (int i = 0; i < 4; i++)
                printf("    [%d] got=%.4f want=%.4f\n", i, ggml_fp16_to_fp32(g_vc[i]), ggml_fp16_to_fp32(w_vc[i]));
            for (size_t i = 10; i < 14; i++)
                printf("    slot10[%zu] got=%.4f want=%.4f\n", i - 10 * D * G, ggml_fp16_to_fp32(g_vc[i]),
                       ggml_fp16_to_fp32(w_vc[i]));
            if (bad) n_fail++; else n_pass++;
        }
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
