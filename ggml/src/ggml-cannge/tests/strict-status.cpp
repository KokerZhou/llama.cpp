#include "ggml-backend.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

// ACL's public ABI uses int for aclError and void * for aclrtStream.
// Catch asynchronous failures that surface after graph submission succeeds.
extern "C" int aclrtSynchronizeStream(void * stream) {
    using synchronize_fn = int (*)(void *);
    static auto synchronize = reinterpret_cast<synchronize_fn>(dlsym(RTLD_NEXT, "aclrtSynchronizeStream"));
    if (!synchronize) std::abort();
    const int status = synchronize(stream);
    if (status != 0) {
        std::fprintf(stderr, "STRICT_STATUS: aclrtSynchronizeStream status=%d\n", status);
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(90);
    }
    return status;
}

extern "C" enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    using compute_fn = enum ggml_status (*)(ggml_backend_t, ggml_cgraph *);
    static auto compute = reinterpret_cast<compute_fn>(dlsym(RTLD_NEXT, "ggml_backend_graph_compute"));
    if (!compute) std::abort();
    const auto status = compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "STRICT_STATUS: backend=%s status=%d nodes=%d\n", ggml_backend_name(backend), int(status), ggml_graph_n_nodes(graph));
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * t = ggml_graph_node(graph, i);
            std::fprintf(stderr, "node=%s op=%s ne=[%lld,%lld,%lld,%lld]\n", t->name, ggml_op_name(t->op), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        }
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(90);
    }
    return status;
}
