// probe: full decode attention chain KQ gemv -> scale+mask+softmax -> V gemv,
// all GE-internal handoffs, vs CPU reference. Model shapes: Qwen3-4B-2507.
// Outputs kq, kq_soft, kqv as separate graph outputs to bisect divergence.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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
    // variant: 1 = kq only (gqa-view replica, KV=16); 2 = kq+softmax out; 3 = full chain (default)
    const int variant = argc > 1 ? atoi(argv[1]) : 3;
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cannge || !cpu) { printf("FATAL: backend init failed\n"); return 1; }

    // decode shapes: KV visible slots, D head dim, G kv heads, H q heads
    const int64_t D = 128, KV = 16, G = 8, H = 32;
    const float   scale = 1.0f / sqrtf((float) D);

    struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx  = ggml_init(ip);

    // dense cache-layout tensors (probe-proven staging paths)
    ggml_tensor * k0  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, G, KV); // cache layout (d,g,s)
    ggml_tensor * v0  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, G, KV);
    ggml_tensor * q   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, H);
    ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KV, 1, 1);
    // v in contraction layout {KV, D, G} as v_cont materializes it
    ggml_tensor * vc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KV, D, G);

    std::vector<ggml_fp16_t> k16(D * G * KV), v16(D * G * KV), vc16(KV * D * G);
    std::vector<float>       q32(D * H), mask32(KV);
    for (int64_t i = 0; i < D * G * KV; i++) {
        int64_t d = i % D;
        k16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f + 1e-3f * (float) d);
        v16[i] = ggml_fp32_to_fp16(0.03f * (float) (((int) (i * 2246822519u) % 83) - 41) / 41.0f);
    }
    // vc[s,d,g] = v0[d,g,s] with correct dense strides
    for (int64_t s = 0; s < KV; s++)
        for (int64_t d = 0; d < D; d++)
            for (int64_t g = 0; g < G; g++)
                vc16[s + KV * d + KV * D * g] = v16[d + D * g + D * G * s];
    for (int64_t i = 0; i < D * H; i++) q32[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
    for (int64_t i = 0; i < KV; i++) mask32[i] = i < 10 ? 0.0f : -INFINITY; // 10 visible slots

    // graph: k = permute(k0) {D,KV,G}; kq = mul_mat(k, q); kq_s = soft_max_ext(kq, mask, scale);
    //        kqv = mul_mat(vc, kq_s)
    ggml_tensor * k    = ggml_permute(ctx, k0, 0, 2, 1, 3);                  // {D,KV,G} strided
    ggml_tensor * kq   = ggml_mul_mat(ctx, k, q);                            // {KV,1,H}
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    ggml_tensor * kq_s = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);      // {KV,1,H}
    ggml_tensor * kqv  = ggml_mul_mat(ctx, vc, kq_s);                        // {D,1,H}

    std::vector<float> want_kq(KV * H), want_ks(KV * H), want_kqv(D * H);
    // ggml dense strides: k0/v0 {D,G,KV}: idx = d + D*g + D*G*s
    //                       q {D,1,H}:   idx = d + D*h
    //                       kq/kq_s {KV,1,H}: idx = s + KV*h
    //                       vc {KV,D,G}: idx = s + KV*d + KV*D*g
    //                       kqv {D,1,H}: idx = d + D*h
    for (int64_t h = 0; h < H; h++) {
        const int64_t g = h / (H / G);
        for (int64_t s = 0; s < KV; s++) {
            double acc = 0;
            for (int64_t d = 0; d < D; d++)
                acc += ggml_fp16_to_fp32(k16[d + D * g + D * G * s]) * (double) q32[d + D * h];
            want_kq[s + KV * h] = (float) acc;
        }
    }
    for (int64_t h = 0; h < H; h++) {
        float mx = -INFINITY;
        for (int64_t s = 0; s < KV; s++) mx = std::max(mx, want_kq[s + KV * h] * scale + mask32[s]);
        double sum = 0;
        for (int64_t s = 0; s < KV; s++) sum += expf(want_kq[s + KV * h] * scale + mask32[s] - mx);
        for (int64_t s = 0; s < KV; s++)
            want_ks[s + KV * h] = expf(want_kq[s + KV * h] * scale + mask32[s] - mx) / (float) sum;
    }
    for (int64_t h = 0; h < H; h++) {
        const int64_t g = h / (H / G);
        for (int64_t d = 0; d < D; d++) {
            double acc = 0;
            for (int64_t s = 0; s < KV; s++)
                acc += ggml_fp16_to_fp32(vc16[s + KV * d + KV * D * g]) * (double) want_ks[s + KV * h];
            want_kqv[d + D * h] = (float) acc;
        }
    }

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        const char * sn = side == 0 ? "cannge" : "cpu";
        ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(k0, k16.data(), 0, k16.size() * 2);
        ggml_backend_tensor_set(v0, v16.data(), 0, v16.size() * 2);
        ggml_backend_tensor_set(vc, vc16.data(), 0, vc16.size() * 2);
        ggml_backend_tensor_set(q, q32.data(), 0, q32.size() * 4);
        ggml_backend_tensor_set(mask, mask32.data(), 0, mask32.size() * 4);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        if (variant == 1) {
            ggml_build_forward_expand(gf, kq);
        } else if (variant == 2) {
            ggml_build_forward_expand(gf, kq_s);
            ggml_build_forward_expand(gf, kq);
        } else {
            ggml_build_forward_expand(gf, kqv);
            ggml_build_forward_expand(gf, kq_s);
            ggml_build_forward_expand(gf, kq);
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("chain FAILED on %s\n", sn);
            n_fail++;
            continue;
        }
        ggml_backend_synchronize(backend);
        std::vector<float> g_kq(KV * H), g_ks(KV * H), g_kqv(D * H);
        ggml_backend_tensor_get(kq, g_kq.data(), 0, g_kq.size() * 4);
        ggml_backend_tensor_get(kq_s, g_ks.data(), 0, g_ks.size() * 4);
        ggml_backend_tensor_get(kqv, g_kqv.data(), 0, g_kqv.size() * 4);
        char nm[80];
        snprintf(nm, sizeof nm, "%s-chain-kq", sn);
        report(nm, g_kq.data(), want_kq.data(), (int) g_kq.size());
        snprintf(nm, sizeof nm, "%s-chain-soft", sn);
        report(nm, g_ks.data(), want_ks.data(), (int) g_ks.size());
        snprintf(nm, sizeof nm, "%s-chain-kqv", sn);
        report(nm, g_kqv.data(), want_kqv.data(), (int) g_kqv.size());
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
