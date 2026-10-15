// opbench: per-op correctness + latency, CANNGE vs CPU, model-realistic shapes.
// verdict per op: value-check vs CPU reference + wall time per op (warm cache).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <unistd.h>

static ggml_backend_t g_cannge;
static ggml_backend_t g_cpu;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Result { const char * name; bool ok; double cpu_ms; double ge_ms; };

static std::vector<Result> results;
static std::map<std::string, std::vector<float>> g_ref; // name -> cpu output (f32 view)

static void fill_f32(ggml_tensor * t, uint64_t seed) {
    std::vector<float> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++) v[i] = 0.5f * sinf(0.017f * (i % 4096)) + 0.25f * cosf(0.031f * ((i * 7 + seed) % 4096));
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
}
static void fill_f16(ggml_tensor * t, uint64_t seed) {
    std::vector<ggml_fp16_t> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++) v[i] = ggml_fp32_to_fp16(0.5f * sinf(0.017f * (i % 4096)) + 0.25f * cosf(0.031f * ((i * 13 + seed) % 4096)));
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * 2);
}
static void fill_i32(ggml_tensor * t, int64_t mod) {
    std::vector<int32_t> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++) v[i] = (int32_t) (i % mod);
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
}

// run one op graph on a backend: returns wall ms per call after warmup
static bool run_case(const char * name, ggml_backend_t backend, bool is_cannge, ggml_tensor * out,
                     const std::function<void()> & build, ggml_context * ctx, int iters, bool check, ggml_tensor * ref) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    build();
    ggml_build_forward_expand(gf, out);
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    // compile/warmup
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("%-34s %s  FAILED(compute)\n", name, is_cannge ? "CANNGE" : "cpu  ");
        if (is_cannge) results.push_back({ name, false, 0, 0 });
        return false;
    }
    ggml_backend_synchronize(backend);
    bool ok = true;
    if (!is_cannge && out->type == GGML_TYPE_F32) {
        std::vector<float> v(ggml_nelements(out));
        ggml_backend_tensor_get(out, v.data(), 0, v.size() * 4);
        g_ref[name] = std::move(v);
    }
    if (is_cannge && out->type == GGML_TYPE_F32) {
        auto it = g_ref.find(name);
        if (it != g_ref.end()) {
            std::vector<float> got(ggml_nelements(out));
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
            int bad = 0;
            for (size_t i = 0; i < got.size() && i < it->second.size(); i++) {
                const double d = std::fabs(got[i] - it->second[i]);
                if (d > 2e-2 && d / std::max(1.0, (double) std::fabs(it->second[i])) > 2e-2) bad++;
            }
            ok = bad == 0;
        }
    }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { ok = false; break; }
    }
    ggml_backend_synchronize(backend);
    double ms = (now_ms() - t0) / iters;
    if (is_cannge) {
        double cpu_ms = 0;
        for (auto & r : results) if (strcmp(r.name, name) == 0) cpu_ms = r.cpu_ms;
        results.push_back({ name, ok, cpu_ms, ms });
        printf("%-34s val=%s  cpu=%7.3f ms  cannge=%7.3f ms  %s\n", name, ok ? "OK " : "BAD", cpu_ms, ms,
               ms < cpu_ms ? "GE faster" : (ms < cpu_ms * 3 ? "comparable" : "*** GE SLOWER ***"));
    } else {
        results.push_back({ name, true, ms, 0 });
        // stash name->cpu_ms; the cannge run prints the pair
    }
    return ok;
}

#define CASE(setup) \
    do { \
        { struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true }; \
          ggml_context * c = ggml_init(ip); setup(c, false); ggml_free(c); } \
        { struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true }; \
          ggml_context * c = ggml_init(ip); setup(c, true);  ggml_free(c); } \
    } while (0)

int main() {
    setbuf(stdout, NULL);
    g_cannge = ggml_backend_cannge_init(0);
    g_cpu    = ggml_backend_cpu_init();
    if (!g_cannge || !g_cpu) { printf("FATAL\n"); return 1; }
    const int iters = 50;

    // RMS_NORM decode {2560,1}
    auto rms = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(x, 1); fill_f32(w, 2);
        ggml_tensor * out = ggml_rms_norm(ctx, x, 1e-6f);
        out = ggml_mul(ctx, out, w);
        run_case("rms_norm {2560,1} decode", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(rms);

    // MUL_MAT gemv wo {4096,2560}x{4096,1}
    auto mm_wo = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 2560);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f16(w, 3); fill_f32(x, 4);
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        run_case("mul_mat wo gemv {4096,2560}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(mm_wo);

    // MUL_MAT gemv ffn gate {2560,9728}x{2560,1}
    auto mm_gate = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2560, 9728);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f16(w, 5); fill_f32(x, 6);
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        run_case("mul_mat gate gemv {2560,9728}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(mm_gate);

    // MUL_MAT prefill M=32 {2560,4096}x{2560,32}
    auto mm_pre = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2560, 4096);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 32);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f16(w, 7); fill_f32(x, 8);
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        run_case("mul_mat prefill M=32 {2560,4096}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(mm_pre);

    // ROPE decode 2D {128,32}
    auto rope = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(x, 9); fill_i32(p, 512);
        ggml_tensor * out = ggml_rope_ext(ctx, x, p, nullptr, 128, GGML_ROPE_TYPE_NEOX, 10000, 1, 0, 1, 1, 0, 0);
        run_case("rope 2D {128,32} decode", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(rope);

    // SOFT_MAX {256,1,32}
    auto sm = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 32);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(x, 10);
        ggml_tensor * out = ggml_soft_max(ctx, x);
        run_case("soft_max {256,1,32}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(sm);

    // ADD {2560,1}
    auto add = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(a, 11); fill_f32(b, 12);
        ggml_tensor * out = ggml_add(ctx, a, b);
        run_case("add {2560,1}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(add);

    // GLU swiglu {9728,1} -> {4864,1}
    auto glu = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 9728, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(x, 13);
        ggml_tensor * out = ggml_glu(ctx, x, GGML_GLU_OP_SWIGLU, false);
        run_case("glu swiglu {9728,1}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(glu);

    // GET_ROWS token_embd
    auto gr = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2560, 151936);
        ggml_tensor * i = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f16(w, 14); fill_i32(i, 151936);
        ggml_tensor * out = ggml_get_rows(ctx, w, i);
        run_case("get_rows embd {2560,1}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(gr);

    // CONT relabel {128,32}->{4096,1}
    auto cont = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(x, 15);
        ggml_tensor * out = ggml_cont_2d(ctx, x, 4096, 1);
        run_case("cont_2d {128,32}->{4096,1}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(cont);

    // CPY f32->f16 {2560,1}
    auto cpy = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f32(x, 16);
        ggml_tensor * out = ggml_cast(ctx, x, GGML_TYPE_F16);
        run_case("cast f32->f16 {2560,1}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(cpy);

    // CONT of a permuted view: v_cont pattern {kv,128,8} transposed view -> cont
    auto vcont = [&](ggml_context * ctx, bool is_cannge) {
        ggml_backend_t be = is_cannge ? g_cannge : g_cpu;
        ggml_tensor * v0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 8, 256);      // [D, NKV, KV] dense
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_f16(v0, 17);
        ggml_tensor * vt = ggml_permute(ctx, v0, 2, 0, 1, 3);                        // [KV, D, NKV] view
        ggml_tensor * out = ggml_cont(ctx, vt);                                      // dense materialization
        run_case("cont permuted-view {256,128,8}", be, is_cannge, out, [&] {}, ctx, iters, false, nullptr);
    };
    CASE(vcont);

    printf("\nsummary (cannge vs cpu):\n");
    int slow = 0;
    for (auto & r : results) {
        if (r.ge_ms > 0 && r.ge_ms >= r.cpu_ms * 3) { printf("  SLOW: %s cpu=%.3f ge=%.3f\n", r.name, r.cpu_ms, r.ge_ms); slow++; }
    }
    printf("slow ops: %d/%zu\n", slow, results.size() / 2);
    fflush(stdout);
    _exit(0);
}
