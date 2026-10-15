// probe: replicate the c0 smoke KV chain on CANNGE vs CPU:
// pre-allocated f16 cache [1024,512], SET_ROWS f32 data w/ i64 idx,
// then VIEW [128,8,256] -> PERMUTE(0,2,1,3) -> TRANSPOSE like llama attn prep.
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
    printf("%-16s %s  max_abs=%.3e bad=%d/%d\n", name, bad == 0 ? "PASS" : "FAIL", max_abs, bad, n);
    if (bad) {
        for (int i = 0; i < n && i < 8; i++) printf("    [%d] got=%f want=%f\n", i, got[i], want[i]);
        n_fail++;
    } else {
        n_pass++;
    }
}

int main() {
    setbuf(stdout, NULL); // GE teardown can hang in exit handlers; keep output unbuffered
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cannge || !cpu) { printf("FATAL: backend init failed\n"); return 1; }

    const int64_t D = 1024, L = 512, H = 8, HD = 128, NKV = 256, NT = 7;

    for (int side = 0; side < 2; side++) {
        ggml_backend_t backend = side == 0 ? cannge : cpu;
        printf("== backend %s ==\n", side == 0 ? "CANNGE" : "CPU");

        struct ggml_init_params iparams = { 64 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(iparams);

        // f16 KV cache, pre-allocated in the backend buffer (like llama -ngl all)
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, L);

        // SET_ROWS: dst[c[i], i2, i3] = data[i, i2, i3]; f32 data -> f16 cache
        ggml_tensor * data = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT);
        ggml_tensor * idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, NT);
        ggml_backend_alloc_ctx_tensors(ctx, backend);
        {
            std::vector<ggml_fp16_t> z(D * L, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(cache, z.data(), 0, z.size() * 2);
            // fill encodes position: value(t, h, dd) = 0.001*dd + 0.1*h + t so
            // any stride error decodes to wrong (t,h,dd)
            std::vector<float> dv(D * NT);
            for (int t = 0; t < NT; t++)
                for (int h = 0; h < H; h++)
                    for (int dd = 0; dd < HD; dd++) {
                        const int d = h * HD + dd;
                        dv[t * D + d] = 0.001f * dd + 0.1f * h + (float) t;
                    }
            std::vector<int64_t> iv = { 3, 100, 255, 0, 42, 511, 200 };
            ggml_backend_tensor_set(data, dv.data(), 0, dv.size() * 4);
            ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * 8);
        }
        ggml_tensor * sr = ggml_set_rows(ctx, cache, data, idx); // a=dst cache, b=data, c=idx (ggml.h:1748)

        // V view chain: [128,8,256] -> permute(0,2,1,3) [128,256,8] -> transpose [256,128,8]
        ggml_tensor * v3    = ggml_view_4d(ctx, cache, HD, H, NKV, 1, 2 * HD, 2 * D, 2 * D * L, 0);
        ggml_tensor * vperm = ggml_permute(ctx, v3, 0, 2, 1, 3);
        ggml_tensor * vt    = ggml_transpose(ctx, vperm); // [256,128,8], strided view
        // strided views cannot be read back flat: copy into a contiguous tensor
        // in the same graph (CPY with contiguous dst is supported by both backends)
        ggml_tensor * cvt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, NKV, HD, H);
        ggml_cpy(ctx, vt, cvt);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, sr);  // side-effecting scatter must be a graph node
        ggml_build_forward_expand(gf, cvt);
        ggml_backend_alloc_ctx_tensors(ctx, backend); // allocate cvt (views alias cache)

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("graph_compute FAILED on %s\n", side == 0 ? "CANNGE" : "CPU");
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(backend);

        // read the cache back (f16) and the contiguous copy of the transposed view
        std::vector<ggml_fp16_t> got_cache(D * L);
        ggml_backend_tensor_get(cache, got_cache.data(), 0, got_cache.size() * 2);
        std::vector<ggml_fp16_t> got_vt16((size_t) NKV * HD * H);
        ggml_backend_tensor_get(cvt, got_vt16.data(), 0, got_vt16.size() * 2);
        std::vector<float> got_vt(got_vt16.size());
        for (size_t i = 0; i < got_vt16.size(); i++) got_vt[i] = ggml_fp16_to_fp32(got_vt16[i]);

        // CPU reference in plain floats
        auto dv_at = [&](int t, int h, int dd) { return 0.001f * dd + 0.1f * h + (float) t; };
        std::vector<float> ref_cache(D * L, 0.0f);
        {
            std::vector<int64_t> iv = { 3, 100, 255, 0, 42, 511, 200 };
            for (int t = 0; t < NT; t++)
                for (int h = 0; h < H; h++)
                    for (int dd = 0; dd < HD; dd++)
                        ref_cache[iv[t] * D + h * HD + dd] = dv_at(t, h, dd);
        }
        {
            char nm[64]; snprintf(nm, sizeof nm, "%s-cache", side == 0 ? "cannge" : "cpu");
            std::vector<float> gc(D * L);
            for (size_t i = 0; i < got_cache.size(); i++) gc[i] = ggml_fp16_to_fp32(got_cache[i]);
            report(nm, gc.data(), ref_cache.data(), D * L);
        }
        {
            // v3 is contiguous ([2,256,2048] bytes): flat read is valid.
            // v3(dd,h,t) = cache(dd + 128*h, t) = 0.001*dd + 0.1*h + invslot(t)
            std::vector<ggml_fp16_t> got_v3_16((size_t) HD * H * NKV);
            ggml_backend_tensor_get(v3, got_v3_16.data(), 0, got_v3_16.size() * 2);
            std::vector<float> got_v3(got_v3_16.size()), ref_v3(got_v3_16.size(), 0.0f);
            std::vector<int64_t> iv = { 3, 100, 255, 0, 42, 511, 200 };
            for (int t = 0; t < NT; t++)
                for (int h = 0; h < H; h++)
                    for (int dd = 0; dd < HD; dd++) {
                        const size_t li = (size_t) dd + HD * ((size_t) h + H * (size_t) iv[t]);
                        if (iv[t] < NKV) ref_v3[li] = dv_at(t, h, dd);
                    }
            for (size_t i = 0; i < got_v3_16.size(); i++) got_v3[i] = ggml_fp16_to_fp32(got_v3_16[i]);
            char nm[64]; snprintf(nm, sizeof nm, "%s-v3", side == 0 ? "cannge" : "cpu");
            report(nm, got_v3.data(), ref_v3.data(), (int) ref_v3.size());
        }
        {
            // vt [256,128,8]: (t,d,h) -> v3(d,h,t) -> cache(d + 128*h, t)
            std::vector<float> ref_vt((size_t) NKV * HD * H);
            for (int64_t t = 0; t < NKV; t++)
                for (int64_t d = 0; d < HD; d++)
                    for (int64_t h = 0; h < H; h++)
                        ref_vt[(t * HD + d) * H + h] = ref_cache[(d + HD * h) + D * t];
            char nm[64]; snprintf(nm, sizeof nm, "%s-vt", side == 0 ? "cannge" : "cpu");
            report(nm, got_vt.data(), ref_vt.data(), (int) ref_vt.size());
        }

        ggml_free(ctx);
    }

    printf("pass=%d fail=%d\n", n_pass, n_fail);
    fflush(stdout);
    _exit(n_fail != 0); // skip GE exit-handler teardown (pthread_cond_destroy hang)
}
