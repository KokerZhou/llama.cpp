// probe: FLASH_ATTN_EXT model-shaped — q[128,15,32] k/v[128,15,8] f16,
// causal additive mask [15,15], prec f32 out, scale=1/sqrt(128).
// decode variant S_q=1 too. CANNGE vs CPU.
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
static int g_mode = 0; // 4 = model-shaped: q F32, k/v F16, f32 additive mask

static void report(const char * name, const float * got, const float * want, int n) {
    double max_rel = 0.0;
    int    bad     = 0;
    for (int i = 0; i < n; i++) {
        const double d   = std::fabs((double) got[i] - (double) want[i]);
        const double den = std::max(1.0, std::fabs((double) want[i]));
        if (d / den > max_rel) max_rel = d / den;
        if (d / den > 2e-2 && d > 2e-2) bad++;
    }
    printf("%-22s %s  max_rel=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_rel, bad, n);
    if (bad) {
        int shown = 0;
        for (int i = 0; i < n && shown < 6; i++) {
            const double d2 = std::fabs((double) got[i] - (double) want[i]);
            if (d2 > 2e-2 && d2 / std::max(1.0, std::fabs((double) want[i])) > 2e-2) {
                printf("    [d=%d s=%d h=%d] got=%f want=%f\n", i % 128, (i / 128) % 15, i / (128 * 15),
                       got[i], want[i]);
                shown++;
            }
        }
        n_fail++;
    } else {
        n_pass++;
    }
}

static std::vector<float> g_cpu_ref;
static void run_case(ggml_backend_t backend, const char * sn, int64_t S) {
    const int64_t D = g_mode == 6 ? 64 : 128,
                NH = (g_mode == 6 || g_mode == 9) ? 1 : (g_mode == 7 ? 8 : (g_mode == 3 ? 8 : 32)),
                NKV = (g_mode == 6 || g_mode == 9) ? 1 : 8,
                KV = g_mode == 12 ? (getenv("FA_KV") ? atoll(getenv("FA_KV")) : 8) : S;
    const int64_t NHx  = g_mode == 12 && getenv("FA_NH") ? atoll(getenv("FA_NH")) : NH;
    const int64_t NKVx = g_mode == 12 && getenv("FA_NKV") ? atoll(getenv("FA_NKV")) : NKV;
    struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);

    ggml_tensor * q0 = ggml_new_tensor_4d(ctx, g_mode == 4 ? GGML_TYPE_F32 : GGML_TYPE_F16, D, NHx, S, 1);
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKVx, KV, 1);
    ggml_tensor * v0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKVx, KV, 1);
    ggml_tensor * q  = ggml_permute(ctx, q0, 0, 2, 1, 3); // [D, S, NH, 1], (d,n,s) memory
    ggml_tensor * k  = ggml_permute(ctx, k0, 0, 2, 1, 3);
    ggml_tensor * v  = ggml_permute(ctx, v0, 0, 2, 1, 3);
    ggml_tensor * m = g_mode == 2 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, KV, S)
                                  : ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KV, S);
    (void) 0;
    ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<ggml_fp16_t> qv(g_mode == 4 ? 0 : D * S * NHx);
    std::vector<float>       qv32(g_mode == 4 ? D * S * NHx : 0);
    std::vector<ggml_fp16_t> kv(D * KV * NKVx), vv(D * KV * NKVx), mv(KV * S);
    std::vector<float>       mv32(KV * S);
    auto rnd = [](int64_t i) { return 0.5f * sinf(0.017f * (float) i) + 0.25f * cosf(0.031f * (float) (i * 7)); };
    // q0 is [D, NH, S]: element (d, h, s) at d + D*h + D*NH*s
    for (int64_t s = 0; s < S; s++)
        for (int64_t h = 0; h < NHx; h++)
            for (int64_t d = 0; d < D; d++) {
                const float val = rnd(d + D * s + D * S * h);
                if (g_mode == 4) qv32[(d + D * h + D * NH * s)] = val;
                else             qv[(d + D * h + D * NH * s)]   = ggml_fp32_to_fp16(val);
            }
    for (int64_t t = 0; t < KV; t++)
        for (int64_t g = 0; g < NKVx; g++)
            for (int64_t d = 0; d < D; d++) {
                kv[(d + D * g + D * NKV * t)] = ggml_fp32_to_fp16(rnd(d + D * t + D * KV * g + 12345));
                vv[(d + D * g + D * NKV * t)] = ggml_fp32_to_fp16(rnd(d + D * t + D * KV * g + 54321));
            }
    if (g_mode == 2) {
        for (int64_t r = 0; r < KV; r++)
            for (int64_t c = 0; c < S; c++) mv32[r * S + c] = (r <= c ? 0.0f : -INFINITY);
        ggml_backend_tensor_set(m, mv32.data(), 0, mv32.size() * 4);
    } else {
        for (int64_t r = 0; r < KV; r++)
            for (int64_t c = 0; c < S; c++) {
                const int64_t keepn = getenv("FA_KEEP") ? atoll(getenv("FA_KEEP")) : 3;
                const bool  keep    = g_mode == 12 ? (r < keepn) : (r <= c);
                mv[r * S + c] = ggml_fp32_to_fp16(keep ? 0.0f : -INFINITY);
            }
    }
    if (g_mode == 11) {
        // anatomy mode: k = 0 -> uniform attention; v identifies (d, t, g)
        std::vector<ggml_fp16_t> kz(kv.size(), ggml_fp32_to_fp16(0.0f));
        kv = kz;
        for (int64_t t = 0; t < KV; t++)
            for (int64_t g = 0; g < NKV; g++)
                for (int64_t d = 0; d < D; d++) {
                    vv[(d + D * g + D * NKV * t)] = ggml_fp32_to_fp16((float) d + 10.0f * (float) t + 100.0f * (float) g);
                }
    }
    if (g_mode == 4) ggml_backend_tensor_set(q, qv32.data(), 0, qv32.size() * 4);
    else             ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * 2);
    ggml_backend_tensor_set(k, kv.data(), 0, kv.size() * 2);
    ggml_backend_tensor_set(v, vv.data(), 0, vv.size() * 2);
    if (g_mode != 2) ggml_backend_tensor_set(m, mv.data(), 0, mv.size() * 2);

    const float scale = 1.0f / sqrtf((float) D);
    const bool fa_nomask = g_mode == 12 && getenv("FA_NOMASK") != nullptr;
    ggml_tensor * out = (g_mode == 1 || g_mode == 6 || g_mode == 7 || g_mode == 9 || g_mode == 11 || fa_nomask) ? ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f)
                                    : ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("fa S=%lld FAILED on %s\n", (long long) S, sn);
        n_fail++;
        ggml_free(ctx);
        return;
    }
    ggml_backend_synchronize(backend);
    std::vector<float> got(D * S * NHx);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);

    if (g_mode == 11) {
        // expected: out[d,s,h] = d + 10*mean_t + 100*(h/(NH/NKV)), uniform over KV
        const int64_t NHr = NHx, NKVR = NKVx;
        double maxe = 0;
        for (int64_t h = 0; h < NHr; h++) {
            const int64_t g = h / (NHr / NKVR);
            const float want_d0 = 10.0f * (float)(KV - 1) / 2.0f + 100.0f * (float) g;
            for (int64_t s = 0; s < S; s++) {
                for (int64_t d = 0; d < D; d += 17) {
                    const float gv = got[(size_t)(d + D * s + D * S * h)];
                    const float wv = (float) d + want_d0;
                    maxe = std::max(maxe, (double) std::fabs(gv - wv));
                }
            }
        }
        printf("anatomy-m11-S%lld %s max_abs=%.4f\n", (long long) S, maxe < 0.5 ? "PASS" : "FAIL", maxe);
        for (int64_t s = 0; s < S && s < 4; s++) {
            for (int64_t h = 0; h < NHr && h < 8; h++) {
                printf("  s%lld h%lld d0=%.1f d17=%.1f", (long long) s, (long long) h,
                       got[(size_t)(0 + D * s + D * S * h)], got[(size_t)(17 + D * s + D * S * h)]);
            }
            printf("\n");
        }
        printf("  (correct: d+10*mean(t)+100*(h/4); KV=%lld NH=%lld NKV=%lld)\n", (long long) KV, (long long) NHr, (long long) NKVR);
        if (maxe < 0.5) n_pass++; else n_fail++;
        ggml_free(ctx);
        return;
    }
    // CPU reference: per head h: softmax over kv of (q_h . k_g * scale + mask) . v_g
    // ggml layout [D, S, NH]: element (d, s, h) at d + D*s + D*S*h
    std::vector<float> want(D * S * NHx, 0.0f);
    const int64_t gqa = NHx / NKVx;
    for (int64_t h = 0; h < NHx; h++) {
        const int64_t g = h / gqa;
        for (int64_t s = 0; s < S; s++) {
            std::vector<float> scores(KV);
            float mx = -INFINITY;
            for (int64_t t = 0; t < KV; t++) {
                float dot = 0;
                for (int64_t d = 0; d < D; d++) {
                    const float qf = g_mode == 4 ? qv32[d + D * s + D * S * h]
                                                 : ggml_fp16_to_fp32(qv[d + D * s + D * S * h]);
                    const float kf = ggml_fp16_to_fp32(kv[d + D * t + D * KV * g]);
                    dot += qf * kf;
                }
                const bool no_mask = g_mode == 1 || g_mode == 6 || g_mode == 7 || g_mode == 9 || g_mode == 11;
                const float mval = no_mask ? 0.0f
                                 : g_mode == 2 ? mv32[t * S + s]
                                               : ggml_fp16_to_fp32(mv[t * S + s]);
                scores[t]     = dot * scale + mval;
                mx            = std::max(mx, scores[t]);
            }
            float sum = 0;
            for (int64_t t = 0; t < KV; t++) sum += expf(scores[t] - mx);
            for (int64_t d = 0; d < D; d++) {
                float acc = 0;
                for (int64_t t = 0; t < KV; t++) {
                    const float w = expf(scores[t] - mx) / sum;
                    acc += w * ggml_fp16_to_fp32(vv[d + D * t + D * KV * g]);
                }
                want[d + D * s + D * S * h] = acc;
            }
        }
    }
    if (strcmp(sn, "cannge") == 0 && getenv("FA_DUMP") && S <= 3) {
        char fn[128];
        snprintf(fn, sizeof fn, "/tmp/fa_dump_got_m%d_S%lld.bin", g_mode, (long long) S);
        FILE * f = fopen(fn, "wb");
        if (f) { fwrite(got.data(), 4, got.size(), f); fclose(f); }
        snprintf(fn, sizeof fn, "/tmp/fa_dump_want_m%d_S%lld.bin", g_mode, (long long) S);
        f = fopen(fn, "wb");
        if (f) { fwrite(want.data(), 4, want.size(), f); fclose(f); }
    }
    if (g_mode == 4 || g_mode == 6 || g_mode == 7 || g_mode == 8 || g_mode == 9 || g_mode == 11 || g_mode == 12) {
        // analytic reference: cpu backend may reject mixed q-f32/k-f16
        char nm[64];
        snprintf(nm, sizeof nm, "%s-vs-ref-m%d-S%lld", sn, g_mode, (long long) S);
        report(nm, got.data(), want.data(), (int) got.size());
    } else if (strcmp(sn, "cpu") == 0) {
        g_cpu_ref = got;
        printf("cpu-fa-m%d-S%lld stored\n", g_mode, (long long) S);
    } else {
        char nm[64];
        snprintf(nm, sizeof nm, "%s-vs-cpu-m%d-S%lld", sn, g_mode, (long long) S);
        report(nm, got.data(), g_cpu_ref.data(), (int) got.size());
    }
    ggml_free(ctx);
}

int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cannge || !cpu) { printf("FATAL\n"); return 1; }
    g_mode = 0;
    printf("-- S=2 dissection --\n");
    run_case(cpu, "cpu", 2);
    run_case(cannge, "cannge", 2);
    printf("-- S=3 --\n");
    run_case(cpu, "cpu", 3);
    run_case(cannge, "cannge", 3);
    printf("-- decode S=1 causal --\n");
    run_case(cpu, "cpu", 1);
    run_case(cannge, "cannge", 1);
    printf("pass=%d fail=%d\n", n_pass, n_fail);
    printf("-- mode 4: model-shaped q-F32 k/v-F16 f32-mask --\n");
    g_mode = 4;
    run_case(cannge, "cannge", 2);
    run_case(cannge, "cannge", 3);
    run_case(cannge, "cannge", 1);
    printf("pass=%d fail=%d\n", n_pass, n_fail);
    printf("-- mode 1: no mask, uniform f16 --\n");
    g_mode = 1;
    run_case(cpu, "cpu", 2);
    run_case(cannge, "cannge", 2);
    run_case(cpu, "cpu", 3);
    run_case(cannge, "cannge", 3);
    printf("pass=%d fail=%d\n", n_pass, n_fail);
    printf("-- mode 6: N=1 D=64 no mask --\n");
    g_mode = 6;
    run_case(cannge, "cannge", 2);
    run_case(cannge, "cannge", 3);
    printf("-- mode 7: N=8 NKV=8 D=128 no mask (MHA) --\n");
    g_mode = 7;
    run_case(cannge, "cannge", 2);
    run_case(cannge, "cannge", 3);
    printf("-- mode 8: N=32 NKV=8 D=128 no mask (GQA) --\n");
    g_mode = 8;
    run_case(cannge, "cannge", 2);
    run_case(cannge, "cannge", 3);
    printf("-- mode 9: N=1 D=128 no mask --\n");
    g_mode = 9;
    run_case(cannge, "cannge", 2);
    run_case(cannge, "cannge", 3);
    printf("-- mode 7 with longer S (8,16) --\n");
    g_mode = 7;
    run_case(cannge, "cannge", 8);
    run_case(cannge, "cannge", 16);
    printf("-- mode 12: IFA decode s_q=1 kv=8 mask first 3 --\n");
    g_mode = 12;
    run_case(cannge, "cannge", 1);
    printf("-- mode 11: anatomy k=0 --\n");
    g_mode = 11;
    run_case(cannge, "cannge", 2);
    run_case(cannge, "cannge", 8);
    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
