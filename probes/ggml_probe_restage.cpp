// probe: staged-input refresh across plan reuse. A device-resident leaf with
// NON-DENSE (permuted) strides is the input of a 1-node CONT fragment; the
// plan is compiled on round 0 and reused on round 1 with fresh addresses and
// fresh data. The backend stages non-dense inputs into a per-plan staging
// buffer; if the staging is treated as immutable (stage-once), round 1 reads
// round 0's bytes. Model repro: CONT_NC pass-2 kqv_out fragments.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cannge.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    setbuf(stdout, NULL);
    ggml_backend_t cannge = ggml_backend_cannge_init(0);
    if (!cannge) { printf("FATAL init\n"); return 1; }

    const int64_t N = 128 * 32 * 9; // 36864 f32 = 147456 B (> 64 KB staging gate)
    float             results[2] = { -1, -1 };

    for (int round = 0; round < 2; round++) {
        struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx  = ggml_init(ip);
        // logical {128,9,32} carried on permuted strides of a {128,32,9} buffer:
        // nb = {4, 16384, 512} - non-dense, so the backend must stage it
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 9, 32);
        strcpy(x->name, "kqv-like-leaf");
        x->nb[0] = 4;
        x->nb[1] = 16384;
        x->nb[2] = 512;
        x->nb[3] = 147456;
        ggml_tensor * y = ggml_cont_2d(ctx, x, 4096, 9);
        strcpy(y->name, "kqv_out-x");
        ggml_set_output(y);
        ggml_backend_alloc_ctx_tensors(ctx, cannge);

        const float fill = round == 0 ? 3.0f : 7.0f;
        std::vector<float> xv(N, fill);
        ggml_backend_tensor_set(x, xv.data(), 0, N * 4); // uniform: stride-free

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        printf("round %d: x->data=%p y->data=%p\n", round, x->data, y->data);
        if (ggml_backend_graph_compute(cannge, gf) != GGML_STATUS_SUCCESS) {
            printf("round %d compute FAILED\n", round);
            return 1;
        }
        ggml_backend_synchronize(cannge);
        std::vector<float> g(N);
        ggml_backend_tensor_get(y, g.data(), 0, N * 4);
        printf("round %d: y[0]=%.3f y[mid]=%.3f y[N-1]=%.3f (want %.3f)\n", round, g[0], g[N / 2], g[N - 1],
               fill);
        results[round] = g[0];
        ggml_free(ctx);
    }
    const bool pass = results[0] == 3.0f && results[1] == 7.0f;
    printf("staged-rebind: %s\n", pass ? "PASS" : "FAIL(stale staging)");
    return pass ? 0 : 1;
}
