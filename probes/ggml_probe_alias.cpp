// probe: GE output-port aliasing. Two shapes:
//  A) ancestor chain: y1 = x*2, y2 = y1+1, y3 = y2+1; all three flagged
//     outputs. If GE binds ancestor output ports to one buffer, all three
//     read y1's value.
//  B) independent siblings: y1 = x*2, y2 = x*3 (flagged outputs) - control.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <unistd.h>

static int n_fail = 0;

static void chk(const char * name, const float * got, const float * want, int n) {
    double mr = 0;
    int    bad = 0;
    for (int i = 0; i < n; i++) {
        double d = std::fabs((double) got[i] - (double) want[i]);
        double den = std::max(1.0, std::fabs((double) want[i]));
        if (d / den > mr) mr = d / den;
        if (d / den > 1e-3 && d > 1e-4) bad++;
    }
    printf("%-24s %s max_rel=%.2e bad=%d/%d", name, bad ? "FAIL" : "PASS", mr, bad, n);
    if (bad) {
        printf("  got[0]=%.4f want[0]=%.4f", got[0], want[0]);
        n_fail++;
    }
    printf("\n");
}

int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    if (!cannge) { printf("FATAL\n"); return 1; }
    const int64_t N = getenv("PSIZE") ? atoi(getenv("PSIZE")) : 256;

    // variant 6: MANY dual-use ports (Kcur pattern at scale) - y_i = x*(i+1)
    // consumed in-graph by z_i = y_i + c (terminal) AND y_i flagged as output.
    for (int variant = 6; variant < 7; variant++) {
        const int NP = getenv("NPORTS") ? atoi(getenv("NPORTS")) : 24;
        struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        std::vector<ggml_tensor *> ys(NP), zs(NP);
        const int PAD = getenv("CHAINPAD") ? atoi(getenv("CHAINPAD")) : 0;
        for (int i = 0; i < NP; i++) {
            ys[i] = ggml_scale(ctx, x, (float) (i + 1)); // dual-use
            ggml_tensor * z = ys[i];
            for (int k = 0; k < PAD; k++) {
                z = ggml_add(ctx, z, c);
            }
            zs[i] = z; // terminal consumer
            ggml_set_output(ys[i]);
            ggml_set_output(zs[i]);
        }
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(N, 1.5f), cv(N, 1.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, N * 4);
        ggml_backend_tensor_set(c, cv.data(), 0, N * 4);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64 * 1024, false);
        for (int i = 0; i < NP; i++) {
            ggml_build_forward_expand(gf, zs[i]);
        }
        if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) {
            printf("manydual FAILED\n");
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(cannge);
        int badp = 0;
        for (int i = 0; i < NP; i++) {
            std::vector<float> g(N);
            ggml_backend_tensor_get(ys[i], g.data(), 0, N * 4);
            const float want = 1.5f * (i + 1);
            if (std::fabs(g[0] - want) > 1e-3) {
                printf("  dual port %d got %.3f want %.3f\n", i, g[0], want);
                badp++;
            }
        }
        printf("manydual N=%d %s (bad %d)\n", NP, badp ? "FAIL" : "PASS", badp);
        if (badp) n_fail++;
        ggml_free(ctx);
    }
    // variant 5: DUAL-USE tensor - y = x*2 consumed in-graph by z = y+1
    // (terminal output) AND y itself flagged as output. Kcur pattern.
    for (int variant = 5; variant < 6; variant++) {
        struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * y = ggml_mul(ctx, x, c);   // 2x, dual-use
        ggml_tensor * z = ggml_add(ctx, y, c);   // 2x+1, terminal
        ggml_set_output(y);
        ggml_set_output(z);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(N, 1.5f), cv(N, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, N * 4);
        ggml_backend_tensor_set(c, cv.data(), 0, N * 4);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, z);
        ggml_build_forward_expand(gf, y);
        if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) {
            printf("dualuse FAILED\n");
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(cannge);
        std::vector<float> gy(N), gz(N);
        ggml_backend_tensor_get(y, gy.data(), 0, N * 4);
        ggml_backend_tensor_get(z, gz.data(), 0, N * 4);
        std::vector<float> wy(N, 3.0f), wz(N, 5.0f);
        chk("dual y=3", gy.data(), wy.data(), N);
        chk("dual z=5", gz.data(), wz.data(), N);
        printf("   y[0]=%.3f z[0]=%.3f\n", gy[0], gz[0]);
        ggml_free(ctx);
    }
    // variant 4: MANY sibling forced outputs (Kcur/Vcur port pattern).
    // N independent y_i = x * (i+1), all flagged; verify each readback.
    for (int variant = 4; variant < 5; variant++) {
        const int NP = getenv("NPORTS") ? atoi(getenv("NPORTS")) : 24;
        struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        std::vector<ggml_tensor *> ys(NP);
        for (int i = 0; i < NP; i++) {
            ys[i] = ggml_scale(ctx, x, (float) (i + 1));
            ggml_set_output(ys[i]);
        }
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(N, 1.5f);
        ggml_backend_tensor_set(x, xv.data(), 0, N * 4);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (int i = 0; i < NP; i++) {
            ggml_build_forward_expand(gf, ys[i]);
        }
        if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) {
            printf("manyports FAILED\n");
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(cannge);
        int badp = 0;
        for (int i = 0; i < NP; i++) {
            std::vector<float> g(N);
            ggml_backend_tensor_get(ys[i], g.data(), 0, N * 4);
            const float want = 1.5f * (i + 1);
            if (std::fabs(g[0] - want) > 1e-3) {
                printf("  port %d got %.3f want %.3f\n", i, g[0], want);
                badp++;
            }
        }
        printf("manyports N=%d %s (bad %d)\n", NP, badp ? "FAIL" : "PASS", badp);
        if (badp) n_fail++;
        ggml_free(ctx);
    }
    // variant 3: LONG chain (scale test) - x -> +c repeated L times; three
    // ancestors flagged. If GE aliases output ports under memory pressure,
    // the three reads come back identical.
    for (int variant = 3; variant < 4; variant++) {
        const int L = 1200;
        struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * y = x;
        ggml_tensor * yA = nullptr, * yB = nullptr, * yC = nullptr;
        for (int i = 1; i <= L; i++) {
            y = ggml_add(ctx, y, c);
            if (i == L / 4) yA = y;
            if (i == L / 2) yB = y;
            if (i == 3 * L / 4) yC = y;
        }
        ggml_set_output(yA);
        ggml_set_output(yB);
        ggml_set_output(yC);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(N, 0.0f), cv(N, 1.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, N * 4);
        ggml_backend_tensor_set(c, cv.data(), 0, N * 4);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) {
            printf("longchain FAILED\n");
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(cannge);
        std::vector<float> gA(N), gB(N), gC(N);
        ggml_backend_tensor_get(yA, gA.data(), 0, N * 4);
        ggml_backend_tensor_get(yB, gB.data(), 0, N * 4);
        ggml_backend_tensor_get(yC, gC.data(), 0, N * 4);
        std::vector<float> wA(N, (float) (L / 4)), wB(N, (float) (L / 2)), wC(N, (float) (3 * L / 4));
        chk("long yA", gA.data(), wA.data(), N);
        chk("long yB", gB.data(), wB.data(), N);
        chk("long yC", gC.data(), wC.data(), N);
        printf("   yA[0]=%.1f yB[0]=%.1f yC[0]=%.1f (want %.0f/%.0f/%.0f)\n", gA[0], gB[0], gC[0],
               wA[0], wB[0], wC[0]);
        ggml_free(ctx);
    }
    for (int variant = 0; variant < 3; variant++) {
        struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * y1 = variant != 1 ? ggml_mul(ctx, x, one) : ggml_scale(ctx, x, 2.0f);
        ggml_tensor * y2;
        ggml_tensor * y3 = nullptr;
        if (variant != 1) {
            // ancestor chain y1 -> y2 -> y3
            y2 = ggml_add(ctx, y1, one);
            y3 = ggml_add(ctx, y2, one);
            ggml_set_output(y1);
            ggml_set_output(y2);
            ggml_set_output(y3);
        } else {
            // independent siblings y1 = 2x, y2 = 3x
            y2 = ggml_scale(ctx, x, 3.0f);
            ggml_set_output(y1);
            ggml_set_output(y2);
        }
        ggml_backend_alloc_ctx_tensors(ctx, cannge);
        std::vector<float> xv(N, 1.5f), onev(N, 1.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, N * 4);
        ggml_backend_tensor_set(one, onev.data(), 0, N * 4);
        if (variant == 2) {
            // env-force only, ancestor chain like variant 0
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        if (variant != 1) {
            ggml_build_forward_expand(gf, y3);
            ggml_build_forward_expand(gf, y2);
            ggml_build_forward_expand(gf, y1);
        } else {
            ggml_build_forward_expand(gf, y2);
            ggml_build_forward_expand(gf, y1);
        }
        if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) {
            printf("alias FAILED variant %d\n", variant);
            n_fail++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(cannge);
        std::vector<float> g1(N), g2(N), g3(N);
        ggml_backend_tensor_get(y1, g1.data(), 0, N * 4);
        ggml_backend_tensor_get(y2, g2.data(), 0, N * 4);
        if (variant != 1) {
            ggml_backend_tensor_get(y3, g3.data(), 0, N * 4);
            std::vector<float> w1(N, 1.5f), w2(N, 2.5f), w3(N, 3.5f);
            chk(variant == 0 ? "A y1" : "C-env y1", g1.data(), w1.data(), N);
            chk(variant == 0 ? "A y2" : "C-env y2", g2.data(), w2.data(), N);
            chk(variant == 0 ? "A y3" : "C-env y3", g3.data(), w3.data(), N);
        } else {
            std::vector<float> w1(N, 3.0f), w2(N, 4.5f);
            chk("B y1=3", g1.data(), w1.data(), N);
            chk("B y2=4.5", g2.data(), w2.data(), N);
        }
        ggml_free(ctx);
    }
    printf("fail=%d\n", n_fail);
    _exit(n_fail != 0);
}
