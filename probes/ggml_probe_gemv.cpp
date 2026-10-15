// probe: M=1 GEMV via GE Mul(broadcast)+ReduceSum vs CPU mul_mat.
// decode matmul shapes: w[K,N] f16 x x[K,1] f16 -> [N,1] f32.
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
    printf("%-22s %s  max_rel=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_rel, bad, n);
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

    const int64_t K = 2560;
    const int64_t Ns[] = { 2560, 1024, 768, 3328 % 4096, 151936 };

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        const char * sn = side == 0 ? "cannge" : "cpu";
        printf("== backend %s ==\n", sn);
        for (int64_t N : Ns) {
            struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx  = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N); // [K, N]
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, 1); // [K, M=1]
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<ggml_fp16_t> w16(K * N), x16(K);
            for (int64_t i = 0; i < K * N; i++) { int64_t k = i % K, n = i / K; w16[k + n * K] = ggml_fp32_to_fp16(0.02f * (float) (((int) ((k + n * K) * 2654435761u) % 89) - 44) / 44.0f); }
            for (int64_t i = 0; i < K; i++) x16[i] = ggml_fp32_to_fp16(0.5f * sinf(0.03f * i) + 0.01f);
            ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
            ggml_backend_tensor_set(x, x16.data(), 0, x16.size() * 2);

            ggml_tensor * out = ggml_mul_mat(ctx, w, x); // [N, 1] f32
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("gemv [%lld,%lld] FAILED on %s\n", (long long) K, (long long) N, sn);
                n_fail++;
                ggml_free(ctx);
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(N);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            std::vector<float> want(N, 0.0f);
            for (int64_t n = 0; n < N; n++)
                for (int64_t k = 0; k < K; k++)
                    want[n] += ggml_fp16_to_fp32(w16[k + n * K]) * ggml_fp16_to_fp32(x16[k]);
            char nm[64];
            snprintf(nm, sizeof nm, "%s-gemv-%lld", sn, (long long) N);
            report(nm, got.data(), want.data(), (int) N);
            ggml_free(ctx);
        }

        // batched M==1 GEMV: w [K, N, B] f16 x x [K, 1, B] f32 -> [N, 1, B] f32
        // (classic-attention V side: contract kv, batch heads)
        {
            const int64_t K3 = 256, N3 = 128, B3 = 32;
            struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx  = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K3, N3, B3);
            ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K3, 1, B3);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<ggml_fp16_t> w16(K3 * N3 * B3);
            std::vector<float>       x32(K3 * B3);
            for (int64_t i = 0; i < K3 * N3 * B3; i++) { int64_t k = i % K3; w16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f + 1e-3f * (float) k); }
            for (int64_t i = 0; i < K3 * B3; i++) x32[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
            ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
            ggml_backend_tensor_set(x, x32.data(), 0, x32.size() * 4);
            ggml_tensor * out = ggml_mul_mat(ctx, w, x); // [N, 1, B] f32
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("batched gemv FAILED on %s\n", sn);
                n_fail++;
                ggml_free(ctx);
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(N3 * B3);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            std::vector<float> want(N3 * B3, 0.0f);
            for (int64_t b = 0; b < B3; b++)
                for (int64_t n = 0; n < N3; n++)
                    for (int64_t k = 0; k < K3; k++)
                        want[n + N3 * b] += ggml_fp16_to_fp32(w16[k + K3 * n + K3 * N3 * b]) * x32[k + K3 * b];
            char nm[64];
            snprintf(nm, sizeof nm, "%s-gemv-3d", sn);
            report(nm, got.data(), want.data(), (int) got.size());
            ggml_free(ctx);
        }
        // GQA batched gemv: v {K,N,G=8} f16 x probs {K,1,H=32} f32 -> {N,1,H}
        {
            const int64_t K3 = 256, N3 = 128, G3 = 8, H3 = 32;
            struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx  = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K3, N3, G3);
            ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K3, 1, H3);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<ggml_fp16_t> w16(K3 * N3 * G3);
            std::vector<float>       x32(K3 * H3);
            for (int64_t i = 0; i < K3 * N3 * G3; i++) { int64_t k = i % K3; w16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f + 1e-3f * (float) k); }
            for (int64_t i = 0; i < K3 * H3; i++) x32[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
            ggml_backend_tensor_set(w, w16.data(), 0, w16.size() * 2);
            ggml_backend_tensor_set(x, x32.data(), 0, x32.size() * 4);
            ggml_tensor * out = ggml_mul_mat(ctx, w, x);
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("gqa gemv FAILED on %s\n", sn);
                n_fail++;
                ggml_free(ctx);
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(N3 * H3);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            std::vector<float> want(N3 * H3, 0.0f);
            for (int64_t h = 0; h < H3; h++) {
                const int64_t g = h / (H3 / G3); // ggml-cpu: i02 = i12 / r2
                for (int64_t n = 0; n < N3; n++)
                    for (int64_t k = 0; k < K3; k++)
                        want[n + N3 * h] += ggml_fp16_to_fp32(w16[k + K3 * n + K3 * N3 * g]) * x32[k + K3 * h];
            }
            {
                const char * names[4] = { "h=g*R+r", "h=r*G+g", "g=h%G(b)", "h=g*R+r(swap)" };
                for (int variant = 0; variant < 4; variant++) {
                    int bad = 0;
                    for (int64_t h = 0; h < H3; h++) {
                        int64_t g;
                        if (variant == 0) g = h / (H3 / G3);
                        else if (variant == 1) g = h % G3;
                        else if (variant == 2) g = h % G3;
                        else g = (h % (H3 / G3)) * G3 / (H3 / G3);
                        for (int64_t n = 0; n < N3; n++) {
                            double want = 0;
                            for (int64_t k = 0; k < K3; k++)
                                want += ggml_fp16_to_fp32(w16[k + K3 * n + K3 * N3 * g]) * x32[k + K3 * h];
                            const double d = std::fabs(got[n + N3 * h] - want);
                            if (d > 2e-2 && d / std::max(1.0, std::fabs(want)) > 2e-2) bad++;
                        }
                    }
                    printf("  %s fit %-14s bad=%d\n", sn, names[variant], bad);
                }
            }
            char nm[64];
            snprintf(nm, sizeof nm, "%s-gemv-gqa", sn);
            report(nm, got.data(), want.data(), (int) got.size());
            ggml_free(ctx);
        }
        // GQA with a PERMUTED cache-view operand (model pattern): dense
        // cache {D,NKV,KV} memory (d,n,s); view = permute -> {D,KV,NKV}
        // strided; mul_mat(view, q {D,1,H}) -> {KV,1,H}
        {
            const int64_t D = 128, G3 = 8, KV = 256, H3 = 32;
            struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx  = ggml_init(ip);
            ggml_tensor * w0  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, G3, KV);   // cache layout
            ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, H3);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<ggml_fp16_t> w16(D * G3 * KV);
            std::vector<float>       x32(D * H3);
            for (int64_t i = 0; i < D * G3 * KV; i++) { int64_t k = i % D; w16[i] = ggml_fp32_to_fp16(0.02f * (float) (((int) (i * 2654435761u) % 89) - 44) / 44.0f + 1e-3f * (float) k); }
            for (int64_t i = 0; i < D * H3; i++) x32[i] = 0.5f * sinf(0.03f * (float) i) + 0.01f;
            ggml_backend_tensor_set(w0, w16.data(), 0, w16.size() * 2);
            ggml_backend_tensor_set(x, x32.data(), 0, x32.size() * 4);
            ggml_tensor * wv  = ggml_permute(ctx, w0, 0, 2, 1, 3);                   // {D, KV, NKV} strided
            ggml_tensor * out = ggml_mul_mat(ctx, wv, x);                            // {KV, 1, H}
            ggml_cgraph * gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("gqa-view gemv FAILED on %s\n", sn);
                n_fail++;
                ggml_free(ctx);
                continue;
            }
            ggml_backend_synchronize(backend);
            std::vector<float> got(KV * H3);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            std::vector<float> want(KV * H3, 0.0f);
            for (int64_t h = 0; h < H3; h++) {
                const int64_t g = h / (H3 / G3); // ggml-cpu: i02 = i12 / r2
                for (int64_t s = 0; s < KV; s++)
                    for (int64_t d = 0; d < D; d++)
                        want[s + KV * h] += ggml_fp16_to_fp32(w16[d + D * g + D * G3 * s]) * x32[d + D * h];
            }
            char nm[64];
            snprintf(nm, sizeof nm, "%s-gemv-gqa-view", sn);
            report(nm, got.data(), want.data(), (int) got.size());
            ggml_free(ctx);
        }


    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0);
}
