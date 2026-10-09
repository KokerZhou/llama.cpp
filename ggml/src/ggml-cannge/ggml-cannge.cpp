/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// CANNGE backend: Ascend Graph Engine (GE) whole-graph compilation backend.
// Independent from ggml-cann: owns its ACL device discovery, buffers,
// transfers, streams, GE sessions and graph plans.

#include "ggml-cannge.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ge.h"
#include "graph-build.h"
#include "plan.h"

#include <acl/acl.h>

// gert::Tensor: zero-copy execution binding (exe_graph/runtime/tensor.h)
#include <exe_graph/runtime/tensor.h>
#include <graph/graph.h>
#include <graph/types.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define GGML_CANNGE_NAME "CANNGE"

// plan cache bound; HBM budget based eviction per D6 comes later
#define GGML_CANNGE_MAX_PLANS 64

#define ACL_CHECK(expr)                                                                                                          \
    do {                                                                                                                         \
        aclError err = (expr);                                                                                                   \
        if (err != ACL_SUCCESS) {                                                                                                \
            GGML_LOG_ERROR("%s: ACL error %d (%s) at %s:%d\n", __func__, err, aclGetRecentErrMsg(), __FILE__, __LINE__);         \
        }                                                                                                                        \
    } while (0)

static void ggml_backend_cannge_set_device(int32_t device) {
    aclError err = aclrtSetDevice(device);
    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to set device %d: %s\n", __func__, device, aclGetRecentErrMsg());
    }
}

// device context, one per device
struct ggml_backend_cannge_device_context {
    int32_t     device = -1;
    std::string name;
    std::string description;
};

// backend context, one per backend instance
struct ggml_backend_cannge_context {
    int32_t     device = -1;
    std::string name;
    aclrtStream stream = nullptr;

    std::unique_ptr<ggml_cannge_ge_context> ge;           // lazy, created on first plan compile
    std::map<std::string, ggml_cannge_plan *> plans;      // signature -> plan, owned
    uint64_t                                next_plan_seq = 0;

    ggml_backend_cannge_context(int32_t device) : device(device), name(GGML_CANNGE_NAME + std::to_string(device)) {
        ggml_backend_cannge_set_device(device);
        ACL_CHECK(aclrtCreateStream(&stream));
    }

    ~ggml_backend_cannge_context() {
        ggml_backend_cannge_set_device(device);
        // GE teardown order: sync stream, drop session (via ge), then GEFinalize
        if (stream != nullptr) {
            ACL_CHECK(aclrtSynchronizeStream(stream));
        }
        ge.reset();
        for (auto & kv : plans) {
            delete kv.second;
        }
        if (stream != nullptr) {
            ACL_CHECK(aclrtDestroyStream(stream));
        }
    }

    // drop the least recently inserted plan; HBM budget based eviction per D6
    // keeps the oldest-first policy, the GE graph is removed from the session
    void evict_oldest_plan() {
        auto oldest = plans.end();
        for (auto it = plans.begin(); it != plans.end(); ++it) {
            if (oldest == plans.end() || it->second->seq < oldest->second->seq) {
                oldest = it;
            }
        }
        if (oldest != plans.end()) {
            // the evicted graph may still have async work on our stream, drain
            // it before RemoveGraph releases the device-side graph
            if (stream != nullptr) {
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
            if (ge != nullptr && oldest->second->graph_id != 0) {
                ge->remove(oldest->second->graph_id);
            }
            delete oldest->second;
            plans.erase(oldest);
        }
    }
};

// buffer

struct ggml_backend_cannge_buffer_context {
    int32_t device;
    void *  dev_ptr;
};

static const char * ggml_backend_cannge_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cannge_device_context * buft_ctx = (ggml_backend_cannge_device_context *) buft->context;
    return buft_ctx->name.c_str();
}

static void ggml_backend_cannge_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtFree(ctx->dev_ptr));
    delete ctx;
}

static void * ggml_backend_cannge_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static void ggml_backend_cannge_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                     size_t offset, size_t size) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtMemset((char *) tensor->data + offset, size, value, size));
}

static void ggml_backend_cannge_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                  size_t offset, size_t size) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtMemcpy((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

static void ggml_backend_cannge_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                  size_t offset, size_t size) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtMemcpy(data, size, (const char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

static void ggml_backend_cannge_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
}

static const ggml_backend_buffer_i ggml_backend_cannge_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cannge_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cannge_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_cannge_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cannge_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cannge_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_cannge_buffer_clear,
    /* .reset           = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_cannge_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t size) {
    ggml_backend_cannge_device_context * buft_ctx = (ggml_backend_cannge_device_context *) buft->context;

    ggml_backend_cannge_set_device(buft_ctx->device);

    void * dev_ptr = nullptr;
    aclError err   = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes on device %d: %s\n", __func__, size, buft_ctx->device,
                       aclGetRecentErrMsg());
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cannge_buffer_interface,
                                    new ggml_backend_cannge_buffer_context{ buft_ctx->device, dev_ptr }, size);
}

static size_t ggml_backend_cannge_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // 512 bytes is the alignment required by Ascend matmul kernels for weights
    GGML_UNUSED(buft);
    return 512;
}

static size_t ggml_backend_cannge_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool ggml_backend_cannge_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_backend_cannge_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cannge_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cannge_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cannge_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ ggml_backend_cannge_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cannge_buffer_type_is_host,
};

// backend

static const char * ggml_backend_cannge_get_name(ggml_backend_t backend) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_cannge_free(ggml_backend_t backend) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_cannge_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
                                                 size_t offset, size_t size) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtMemcpyAsync((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream));
    // synchronous: the caller reuses the host staging buffer right after; an
    // async copy would make the driver pin it until a later sync, ballooning
    // host memory during model load (seen: tens of GB pinned vs 64G cgroup)
    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
}

static void ggml_backend_cannge_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                                 size_t offset, size_t size) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtMemcpyAsync(data, size, (const char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST, ctx->stream));
    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
}

static bool ggml_backend_cannge_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                                 const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend_dst->context;
    ggml_backend_cannge_set_device(ctx->device);
    // cross-backend copies can involve host buffers (CPU fallback outputs),
    // ACL has no auto-direction memcpy: pick the kind from the buffer types
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        kind = ACL_MEMCPY_DEVICE_TO_HOST;
    }
    ACL_CHECK(aclrtMemcpyAsync(dst->data, ggml_nbytes(dst), src->data, ggml_nbytes(src), kind, ctx->stream));
    return true;
}

static void ggml_backend_cannge_synchronize(ggml_backend_t backend) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
}

static bool ggml_backend_cannge_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

// L0 gate of the fallback ladder: the whole-graph path is opt-in via env,
// read once; when off the backend reports no op support (default behavior)
static bool ggml_backend_cannge_graph_env_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CANNGE_GRAPH");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

// zero-copy binding of a ggml tensor to a gert::Tensor; the device address is
// the ggml tensor data pointer (views already include their offset), the
// storage shape follows the GE outermost-first axis order
static bool ggml_cannge_bind_tensor(ggml_tensor * t, gert::Tensor & tensor, void * dev_ptr) {
    ge::DataType dt;
    if (!ggml_cannge_ge_dtype(t->type, dt)) {
        GGML_LOG_ERROR("%s: unsupported dtype %d of tensor %s\n", __func__, (int) t->type, t->name);
        return false;
    }

    const std::vector<int64_t> shape = ggml_cannge_ge_shape(t);

    tensor.MutableOriginShape().SetDimNum((size_t) shape.size());
    tensor.MutableStorageShape().SetDimNum((size_t) shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.MutableOriginShape().SetDim((int) i, shape[i]);
        tensor.MutableStorageShape().SetDim((int) i, shape[i]);
    }
    tensor.SetOriginFormat(ge::FORMAT_ND);
    tensor.SetStorageFormat(ge::FORMAT_ND);
    tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
    tensor.SetDataType(dt);
    tensor.SetData(gert::TensorData((uint8_t *) dev_ptr, nullptr, ggml_nbytes(t), gert::TensorPlacement::kOnDeviceHbm));
    return true;
}

ggml_cannge_plan::~ggml_cannge_plan() {
    for (void * p : staging) {
        aclrtFree(p);
    }
    for (void * p : input_staging) {
        aclrtFree(p);
    }
}

// copy between a strided ggml view layout and a compact dense buffer, via
// ne1*ne2*ne3 element-granularity 2D copies (width = element size, height =
// ne0, pitch = nb[0]); nb[0] can exceed the element size on permuted views
static void ggml_cannge_strided_copy(const ggml_tensor * t, void * dense, bool dense_to_strided, aclrtStream stream) {
    const size_t es   = ggml_element_size(t);
    const size_t slab = (size_t) t->ne[0] * es; // one dense (i1,i2,i3) slab
    const int64_t n   = t->ne[1] * t->ne[2] * t->ne[3];
    // a staged input may live in host memory (CPU fallback output): ACL has no
    // auto-direction memcpy
    const bool    host = t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
    // SDMA strided copies need >=4B rows and 4B-aligned pitches (f16
    // transposed views have 2B strides): bounce the whole footprint through
    // host memory for those, one aligned H2D/D2H per direction
    bool aligned_ok = es >= 4;
    for (int i = 0; i < GGML_MAX_DIMS && aligned_ok; i++) {
        aligned_ok = t->nb[i] % 4 == 0;
    }
    if (!aligned_ok) {
        int64_t off_last = 0;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            off_last += (t->ne[i] - 1) * t->nb[i];
        }
        const size_t span = (size_t) off_last + es;
        const size_t slab = (size_t) t->ne[0] * es;
        std::vector<char> bounce(span);
        if (dense_to_strided) {
            std::vector<char> dense_host((size_t) n * slab);
            ACL_CHECK(aclrtMemcpy(dense_host.data(), (size_t) n * slab, dense, (size_t) n * slab,
                                  ACL_MEMCPY_DEVICE_TO_HOST));
            for (int64_t l = 0; l < n; l++) {
                const int64_t i1 = l % t->ne[1];
                const int64_t i2 = (l / t->ne[1]) % t->ne[2];
                const int64_t i3 = l / (t->ne[1] * t->ne[2]);
                char * dst = bounce.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                memcpy(dst, dense_host.data() + (size_t) l * slab, slab);
            }
            ACL_CHECK(aclrtMemcpy(t->data, span, bounce.data(), span, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            if (host) {
                for (int64_t l = 0; l < n; l++) {
                    const int64_t i1 = l % t->ne[1];
                    const int64_t i2 = (l / t->ne[1]) % t->ne[2];
                    const int64_t i3 = l / (t->ne[1] * t->ne[2]);
                    const char * src = (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    memcpy(bounce.data() + (size_t) l * slab, src, slab);
                }
            } else {
                ACL_CHECK(aclrtMemcpy(bounce.data(), span, t->data, span, ACL_MEMCPY_DEVICE_TO_HOST));
                std::vector<char> dense_host((size_t) n * slab);
                ACL_CHECK(aclrtMemcpy(dense_host.data(), (size_t) n * slab, dense, (size_t) n * slab,
                                      ACL_MEMCPY_DEVICE_TO_HOST));
                for (int64_t l = 0; l < n; l++) {
                    const int64_t i1 = l % t->ne[1];
                    const int64_t i2 = (l / t->ne[1]) % t->ne[2];
                    const int64_t i3 = l / (t->ne[1] * t->ne[2]);
                    char * dst = bounce.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    memcpy(dst, dense_host.data() + (size_t) l * slab, slab);
                }
            }
            ACL_CHECK(aclrtMemcpy(dense, (size_t) n * slab, bounce.data(), (size_t) n * slab,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
        }
        return;
    }
    const aclrtMemcpyKind kind =
        dense_to_strided ? (host ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE)
                         : (host ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COPY] %s t->data=%p host=%d dir=%s ne=%lld,%lld,%lld,%lld nb0=%lld\n",
                dense_to_strided ? "out" : "in", t->data, (int) host, dense_to_strided ? "D2H" : (host ? "H2D" : "D2D"),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                (long long) t->nb[0]);
    }
    for (int64_t l = 0; l < n; l++) {
        const int64_t i1 = l % t->ne[1];
        const int64_t i2 = (l / t->ne[1]) % t->ne[2];
        const int64_t i3 = l / (t->ne[1] * t->ne[2]);
        char *       view_row = (char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        char *       dense_row = (char *) dense + (size_t) l * slab;
        const void * src    = dense_to_strided ? (const void *) dense_row : (const void *) view_row;
        void *       dst    = dense_to_strided ? (void *) view_row : (void *) dense_row;
        const size_t spitch = dense_to_strided ? es : (size_t) t->nb[0];
        const size_t dpitch = dense_to_strided ? (size_t) t->nb[0] : es;
        ACL_CHECK(aclrtMemcpy2dAsync(dst, dpitch, src, spitch, es, (size_t) t->ne[0], kind, stream));
    }
}

static enum ggml_status ggml_backend_cannge_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;

    if (!ggml_backend_cannge_graph_env_enabled()) {
        GGML_LOG_ERROR("%s: compiled graph path disabled, set GGML_CANNGE_GRAPH=1 to enable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_cannge_set_device(ctx->device);

    // static capability pre-check (L0 gate of the fallback ladder)
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_cannge_reg(), ctx->device);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }
        if (!ggml_backend_cannge_device_supports_op(dev, node)) {
            GGML_LOG_ERROR("%s: op %d not supported by CANNGE\n", __func__, (int) node->op);
            return GGML_STATUS_FAILED;
        }
    }

    std::string err;

    // per-call IO analysis with fresh pointers: the compute context (and its
    // tensors) may be freed between graph_compute calls, so nothing that
    // survives in the plan cache may reference ggml tensors
    ggml_cannge_plan_io io;
    if (!ggml_cannge_plan_analyze(cgraph, io, err)) {
        GGML_LOG_ERROR("%s: plan analyze failed: %s\n", __func__, err.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::string signature = ggml_cannge_plan_signature(cgraph);

    ggml_cannge_plan * plan = nullptr;
    auto               it   = ctx->plans.find(signature);
    if (it != ctx->plans.end()) {
        plan = it->second;
    }

    // compile -> load once per signature, bind + execute on every call
    if (plan == nullptr) {
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COMPILE] start nodes=%d first=%s(%s) last=%s(%s)\n", cgraph->n_nodes,
                    cgraph->nodes[0]->name, ggml_op_name(cgraph->nodes[0]->op), cgraph->nodes[cgraph->n_nodes - 1]->name,
                    ggml_op_name(cgraph->nodes[cgraph->n_nodes - 1]->op));
        }
        // build and compile first so a build failure is not cached; a
        // compile/load failure caches a FAILED plan below
        std::unique_ptr<ggml_cannge_plan> fresh(new ggml_cannge_plan);
        fresh->signature = signature;

        void * graph = nullptr;
        if (!ggml_cannge_graph_build(*fresh, io, cgraph, graph, err)) {
            GGML_LOG_ERROR("%s: graph build failed: %s\n", __func__, err.c_str());
            return GGML_STATUS_FAILED;
        }
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COMPILE] ES build done nodes=%d, GE compile...\n", cgraph->n_nodes);
        }

        if (ctx->ge == nullptr) {
            ctx->ge.reset(ggml_cannge_ge_context::create(ctx->device, err));
            if (ctx->ge == nullptr) {
                GGML_LOG_ERROR("%s: GE init failed: %s\n", __func__, err.c_str());
                delete (ge::Graph *) graph;
                return GGML_STATUS_FAILED;
            }
        }

        // the session owns an IR copy now, release the caller-side graph
        const bool compiled = ctx->ge->add_and_compile(graph, fresh->graph_id, err);
        delete (ge::Graph *) graph;
        if (!compiled) {
            GGML_LOG_ERROR("%s: compile failed: %s\n", __func__, err.c_str());
            fresh->state = ggml_cannge_plan::FAILED;
            plan = fresh.release();
        } else {
            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-COMPILE] done nodes=%d\n", cgraph->n_nodes);
            }
            fresh->state = ggml_cannge_plan::COMPILED;
            plan = fresh.release();
        }

        if (ctx->plans.size() >= GGML_CANNGE_MAX_PLANS) {
            ctx->evict_oldest_plan();
        }
        plan->seq = ctx->next_plan_seq++;
        ctx->plans[signature] = plan;
    }

    if (plan->state == ggml_cannge_plan::FAILED) {
        return GGML_STATUS_FAILED;
    }

    if (plan->state == ggml_cannge_plan::COMPILED) {
        if (!ctx->ge->load(plan->graph_id, ctx->stream, err)) {
            GGML_LOG_ERROR("%s: load failed: %s\n", __func__, err.c_str());
            plan->state = ggml_cannge_plan::FAILED;
            return GGML_STATUS_FAILED;
        }
        plan->state = ggml_cannge_plan::LOADED;
    }

    if (plan->state == ggml_cannge_plan::LOADED) {
        // bind all graph IO to the ggml device buffers of this call
        std::vector<gert::Tensor> inputs;
        std::vector<gert::Tensor> outputs;
        inputs.reserve(io.inputs.size());
        outputs.reserve(io.output_tensors.size());
        if (plan->input_staging.size() < io.inputs.size()) {
            plan->input_staging.resize(io.inputs.size(), nullptr);
        }
        for (size_t i = 0; i < io.inputs.size(); i++) {
            ggml_tensor * t = io.inputs[i];
            void *        dev_ptr = t->data;
            if (io.input_staged[i]) {
                if (plan->input_staging[i] == nullptr &&
                    aclrtMalloc(&plan->input_staging[i], ggml_nbytes(t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                    GGML_LOG_ERROR("%s: failed to allocate staging buffer for input %s (%zu bytes)\n", __func__, t->name,
                                   ggml_nbytes(t));
                    return GGML_STATUS_FAILED;
                }
                ggml_cannge_strided_copy(t, plan->input_staging[i], false, ctx->stream);
                dev_ptr = plan->input_staging[i];
            }
            gert::Tensor tensor;
            if (!ggml_cannge_bind_tensor(t, tensor, dev_ptr)) {
                return GGML_STATUS_FAILED;
            }
            inputs.push_back(std::move(tensor));
        }
        // output ports whose device range overlaps an input range (aliased
        // views, e.g. PERMUTE of an input) must not be bound in place: GE
        // treats undeclared input/output overlap as unvalidated UB and may
        // silently skip the output write (ge 仓 runtime/v2/kernel/
        // common_kernel_impl/memory_copy.cc:574-579 TensorToOut, src==dst
        // case, @00ecb5c). The declared in-place channel
        // ge.exec.outputReuseInputMemIndexes requires exact address equality,
        // which offset views do not satisfy (ge 仓 docs/zh/design/features/
        // memory_management.md:139-167 @00ecb5c). Bind a backend-held staging
        // buffer instead and copy back on the same stream after execute.
        struct range {
            const uint8_t * lo;
            const uint8_t * hi;
        };
        std::vector<range> input_ranges;
        input_ranges.reserve(io.inputs.size());
        for (ggml_tensor * t : io.inputs) {
            input_ranges.push_back({ (const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t) });
        }
        if (plan->staging.size() < io.output_tensors.size()) {
            plan->staging.resize(io.output_tensors.size(), nullptr);
        }
        // staged output port + copy-back mode: strided when the boundary output
        // is a non-dense view (e.g. PERMUTE as graph result), dense memcpy only
        // for the aliased-address case above
        std::vector<std::pair<size_t, bool>> staged_ports;
        for (size_t i = 0; i < io.output_tensors.size(); i++) {
            ggml_tensor * t = io.output_tensors[i];
            const uint8_t * lo = (const uint8_t *) t->data;
            const uint8_t * hi = lo + ggml_nbytes(t);
            bool overlaps = false;
            for (const range & r : input_ranges) {
                overlaps = lo < r.hi && r.lo < hi;
                if (overlaps) {
                    break;
                }
            }
            // strided wins over the dense alias copy: a non-dense view output
            // that aliases its input still needs the scatter copy-back
            bool strided = false;
            for (size_t j = 0; j < io.outputs.size(); j++) {
                if (io.outputs[j] == t) {
                    strided = io.output_strided[j] != 0;
                    break;
                }
            }
            void * dev_ptr = t->data;
            if (overlaps || strided) {
                if (plan->staging[i] == nullptr && aclrtMalloc(&plan->staging[i], ggml_nbytes(t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                    GGML_LOG_ERROR("%s: failed to allocate staging buffer for output %s (%zu bytes)\n", __func__, t->name,
                                   ggml_nbytes(t));
                    return GGML_STATUS_FAILED;
                }
                dev_ptr = plan->staging[i];
                staged_ports.push_back({ i, strided });
            }
            gert::Tensor tensor;
            if (!ggml_cannge_bind_tensor(t, tensor, dev_ptr)) {
                return GGML_STATUS_FAILED;
            }
            outputs.push_back(std::move(tensor));
        }

        if (!ctx->ge->execute(plan->graph_id, ctx->stream, inputs, outputs, err)) {
            GGML_LOG_ERROR("%s: execute failed: %s\n", __func__, err.c_str());
            plan->state = ggml_cannge_plan::FAILED;
            return GGML_STATUS_FAILED;
        }

        // staged output ports: D2D copy from staging into the real ggml buffer
        // on the same stream, ordered after the graph; no internal sync, ggml
        // calls ggml_backend_synchronize
        for (const auto & sp : staged_ports) {
            ggml_tensor * t = io.output_tensors[sp.first];
            if (sp.second) {
                ggml_cannge_strided_copy(t, plan->staging[sp.first], true, ctx->stream);
            } else {
                ACL_CHECK(aclrtMemcpyAsync(t->data, ggml_nbytes(t), plan->staging[sp.first], ggml_nbytes(t),
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx->stream));
            }
        }

        // side-effect writebacks (future SET support): D2D copy back on the
        // same stream, ordered after the graph; no internal sync, ggml calls
        // ggml_backend_synchronize
        for (const ggml_cannge_writeback & wb : io.writebacks) {
            ACL_CHECK(aclrtMemcpyAsync(wb.dst->data, ggml_nbytes(wb.dst), wb.src->data, ggml_nbytes(wb.src),
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctx->stream));
        }

        // KV cache row scatter (SET_ROWS): dst[slot, i2, i3] = data[i, i2, i3]
        // with slot = idx[i, i2%ne12, i3%ne13], ggml-cpu ops.cpp set_rows_impl
        for (const ggml_cannge_set_rows_op & sr : io.set_rows) {
            const int64_t nidx    = sr.idx->ne[0];
            const size_t  n_slots = (size_t) nidx * sr.idx->ne[1] * sr.idx->ne[2];
            const bool    idx_host = sr.idx->buffer != nullptr && ggml_backend_buffer_is_host(sr.idx->buffer);
            // indices may have been produced on this stream just before: sync
            if (!idx_host) {
                ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
            }
            std::vector<int64_t> slots64;
            const int64_t *      slots = nullptr;
            if (sr.idx->type == GGML_TYPE_I64) {
                slots64.resize(n_slots);
                if (idx_host) {
                    memcpy(slots64.data(), sr.idx->data, n_slots * 8);
                } else {
                    ACL_CHECK(aclrtMemcpy(slots64.data(), n_slots * 8, sr.idx->data, ggml_nbytes(sr.idx),
                                          ACL_MEMCPY_DEVICE_TO_HOST));
                }
            } else {
                std::vector<int32_t> tmp(n_slots);
                if (idx_host) {
                    memcpy(tmp.data(), sr.idx->data, n_slots * 4);
                } else {
                    ACL_CHECK(aclrtMemcpy(tmp.data(), n_slots * 4, sr.idx->data, ggml_nbytes(sr.idx),
                                          ACL_MEMCPY_DEVICE_TO_HOST));
                }
                slots64.assign(tmp.begin(), tmp.end());
            }
            slots = slots64.data();
            const bool    data_host = sr.data->buffer != nullptr && ggml_backend_buffer_is_host(sr.data->buffer);
            const size_t  es_d      = ggml_element_size(sr.data);
            const size_t  es_t      = ggml_element_size(sr.dst);
            const aclrtMemcpyKind row_kind = data_host ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE;
            // host-side cast scratch for the f32 rope output -> f16 cache case;
            // device-resident data rows are staged to host (D2H) before the cast
            std::vector<char> cast_buf;
            std::vector<char> stage_row;
            if (es_d != es_t) {
                cast_buf.resize((size_t) sr.data->ne[0] * es_t);
                if (!data_host) {
                    stage_row.resize((size_t) sr.data->ne[0] * es_d);
                }
            }
            for (int64_t i3 = 0; i3 < sr.data->ne[3]; i3++) {
                for (int64_t i2 = 0; i2 < sr.data->ne[2]; i2++) {
                    const int64_t i11 = i2 % sr.idx->ne[1];
                    const int64_t i12 = i3 % sr.idx->ne[2];
                    for (int64_t i = 0; i < nidx; i++) {
                        const int64_t sl =
                            slots[(size_t) i + (size_t) i11 * nidx + (size_t) i12 * nidx * sr.idx->ne[1]];
                        const char * src = (const char *) sr.data->data + i * sr.data->nb[1] + i2 * sr.data->nb[2] +
                                           i3 * sr.data->nb[3];
                        char * dst_row = (char *) sr.dst->data + (size_t) sl * sr.dst->nb[1] + i2 * sr.dst->nb[2] +
                                         i3 * sr.dst->nb[3];
                        const size_t row_dst = (size_t) sr.data->ne[0] * es_t;
                        if (es_d == es_t) {
                            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                            fprintf(stderr, "[DBG-SR] row %s->%s src=%p dst=%p kind=%d host=%d\n", sr.data->name,
                                    sr.dst->name, src, dst_row, (int) row_kind, (int) data_host);
                        }
                        ACL_CHECK(aclrtMemcpyAsync(dst_row, row_dst, src, row_dst, row_kind, ctx->stream));
                        } else {
                            const char * srow = src;
                            if (!data_host) {
                                ACL_CHECK(aclrtMemcpy(stage_row.data(), stage_row.size(), src, stage_row.size(),
                                                      ACL_MEMCPY_DEVICE_TO_HOST));
                                srow = stage_row.data();
                            }
                            // cast in host memory, then a single H2D row copy
                            if (sr.data->type == GGML_TYPE_F32 && sr.dst->type == GGML_TYPE_F16) {
                                for (int64_t e = 0; e < sr.data->ne[0]; e++) {
                                    ((ggml_fp16_t *) cast_buf.data())[e] =
                                        ggml_fp32_to_fp16(((const float *) srow)[e]);
                                }
                            } else {
                                for (int64_t e = 0; e < sr.data->ne[0]; e++) {
                                    ((float *) cast_buf.data())[e] =
                                        ggml_fp16_to_fp32(((const ggml_fp16_t *) srow)[e]);
                                }
                            }
                            ACL_CHECK(aclrtMemcpyAsync(dst_row, row_dst, cast_buf.data(), row_dst,
                                                       ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream));
                        }
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    return GGML_STATUS_FAILED;
}

static const ggml_backend_i ggml_backend_cannge_interface = {
    /* .get_name                = */ ggml_backend_cannge_get_name,
    /* .free                    = */ ggml_backend_cannge_free,
    /* .set_tensor_async        = */ ggml_backend_cannge_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cannge_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_cannge_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cannge_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_cannge_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

// device

static const char * ggml_backend_cannge_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    return dev_ctx->name.c_str();
}

static const char * ggml_backend_cannge_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    return dev_ctx->description.c_str();
}

static void ggml_backend_cannge_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    ggml_backend_cannge_set_device(dev_ctx->device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

static enum ggml_backend_dev_type ggml_backend_cannge_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // GPU (not ACCEL): the device holds model weights and takes -ngl offload,
    // same classification as the native ggml-cann backend
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cannge_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cannge_device_get_name(dev);
    props->description = ggml_backend_cannge_device_get_description(dev);
    ggml_backend_cannge_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = ggml_backend_cannge_device_get_type(dev);

    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    ggml_backend_cannge_set_device(dev_ctx->device);
    props->device_id = nullptr;
    props->caps = { /* .async                  = */ true,
                    /* .host_buffer            = */ false,
                    /* .buffer_from_host_ptr   = */ false,
                    /* .events                 = */ false,
                    /* .mmap_support           = */ false };
}

static ggml_backend_t ggml_backend_cannge_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    return ggml_backend_cannge_init(dev_ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cannge_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    return ggml_backend_cannge_buffer_type(dev_ctx->device);
}

static bool ggml_cannge_float_dtype(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

// L0 static capability table (docs/cannge D3): doc-supported (op, dtype,
// shape) combos only, conservative rejects are by design; GE adds/Mul broadcast
// dims that are equal or 1, MatMul contraction is ne0 of both operands
static bool ggml_backend_cannge_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (!ggml_backend_cannge_graph_env_enabled()) {
        return false;
    }

    // leaf tensors carry no compute, every backend accepts them
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    // in-place variants write the result back into src[i]'s buffer; GE
    // drops aliased output writes (910B silent, note-10 F1). The *_inplace
    // builders also alias through view_src (out is a view of src0), same
    // dropped-write hazard (note-10 F8); metadata view ops are exempt.
    // SET_ROWS also views its target (src[2]), but the write is applied by
    // the backend explicitly after execute, so it is exempt as well
    if (op->op != GGML_OP_VIEW && op->op != GGML_OP_PERMUTE && op->op != GGML_OP_TRANSPOSE &&
        op->op != GGML_OP_SET_ROWS && op->view_src != nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] == op) {
            return false;
        }
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: {
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            if (!ggml_cannge_float_dtype(src0->type) || !ggml_cannge_float_dtype(src1->type)) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (src0->ne[d] != src1->ne[d] && src0->ne[d] != 1 && src1->ne[d] != 1) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_SCALE:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_SOFT_MAX: {
            if (src0 == nullptr || !ggml_cannge_float_dtype(src0->type) ||
                !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            float scale = 1.0f;
            float max_bias = 0.0f;
            std::memcpy(&scale, (const float *) op->op_params + 0, sizeof(scale));
            std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(max_bias));
            // attention sinks (src[2]) and ALiBi slopes have no GE mapping yet
            if (op->src[2] != nullptr || max_bias != 0.0f) {
                return false;
            }
            // optional mask, added after scaling; row width must match, other
            // dims broadcast like ADD (GE equal-or-1, docs/cannge note-7)
            if (src1 != nullptr) {
                if (!ggml_cannge_float_dtype(src1->type) || src1->ne[0] != src0->ne[0]) {
                    return false;
                }
                for (int d = 1; d < GGML_MAX_DIMS; d++) {
                    if (src1->ne[d] != 1 && src0->ne[d] != src1->ne[d]) {
                        return false;
                    }
                }
            }
            return true;
        }

        case GGML_OP_UNARY: {
            if (src0 == nullptr || !ggml_cannge_float_dtype(src0->type) || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    // STEP has no es_math counterpart; ELU and the rest unmapped
                    return false;
            }
        }

        case GGML_OP_NORM:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_RMS_NORM:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_MUL_MAT: {
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            // both operands must share the contraction dtype; FP32 computes in
            // FP16 on 310P internally (note-7), still accepted here
            if (src0->type != src1->type || !ggml_cannge_float_dtype(src0->type)) {
                return false;
            }
            if (src0->ne[0] != src1->ne[0]) {
                return false; // contraction dim mismatch
            }
            // 32-byte alignment of M/N/K: not in the op spec (the aclnn spec
            // has no alignment constraint and allows non-contiguous inputs),
            // but the GE graph-compile tiling path hard-fails unaligned
            // MatMul shapes (CheckDimsAligned310P, EZ9999) on 310P and 910B
            // alike; reject conservatively, product-independent. 未文档化，
            // 依据见 note-10 F3
            // spec: <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMatmul.md>
#ifdef ASCEND_310P
            // 310P TBE registers fp16 only, so GE down-converts F32 matmul inputs
            // to fp16 before the tiling check (F7, note-10): the 32B rule applies
            // to the fp16 element size, F32 needs 16 elements (note-16 known precision)
            const size_t align_elems = src0->type == GGML_TYPE_F32 ? 16 : 32 / ggml_type_size(src0->type);
#else
            const size_t align_elems = 32 / ggml_type_size(src0->type);
#endif
            if (src0->ne[0] % align_elems != 0 || src0->ne[1] % align_elems != 0 || src1->ne[1] % align_elems != 0) {
                return false;
            }
            // operands must be dense: permuted leaves are non-dense graph IO and
            // need staged copies (note-14); internal permutes stay graph-internal
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }
            // 2D and batched (3D/4D) matmul: leading batch dims must match,
            // GE BatchMatMulV3 batches on the outermost dims; contraction stays ne0
            if (ggml_n_dims(src0) != ggml_n_dims(src1)) {
                return false;
            }
            return ggml_n_dims(src0) >= 2 && ggml_n_dims(src0) <= 4 && src0->ne[2] == src1->ne[2] &&
                   src0->ne[3] == src1->ne[3];
        }

        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_VIEW:
            // mirror the build precondition: dense strides only, the Slice path
            // needs standard contiguous layout relative to the view
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type) &&
                   ggml_is_contiguous(op);

        case GGML_OP_CONT:
            // mirror the build precondition: only contiguous inputs can alias
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type) &&
                   ggml_is_contiguous(src0);

        case GGML_OP_CPY:
            // F16 <-> F32 via Cast; aliased same-dtype copies into views are
            // rejected until the writeback mechanism is enabled
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(src0->type) ||
                !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            if (!ggml_are_same_shape(src0, src1)) {
                return false;
            }
            return src1->view_src == nullptr && ggml_is_contiguous(src1);

        case GGML_OP_DUP:
            // same-dtype copy into the node's own buffer (in-place variants are
            // rejected by the generic view_src check above); the build aliases
            // src0 and analyze forces a GE output port for the copy
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_GET_ROWS:
            return src0 != nullptr && src1 != nullptr && ggml_cannge_float_dtype(src0->type) &&
                   src1->type == GGML_TYPE_I32 && ggml_cannge_float_dtype(op->type);

        case GGML_OP_SET_ROWS: {
            // KV cache write: a[src[1][i]] = src[0][i] per row, applied by the
            // backend as post-execute D2D row copies (no GE scatter needed);
            // ggml.c ggml_set_rows: src0 = row data, src1 = indices, src2 = cache
            const struct ggml_tensor * data = op->src[0];
            const struct ggml_tensor * idx  = op->src[1];
            const struct ggml_tensor * dst  = op->src[2];
            if (data == nullptr || idx == nullptr || dst == nullptr) {
                return false;
            }
            // dtype mismatch (f32 rope output -> f16 cache) is handled by a
            // host-side cast in the scatter when the data sits in host memory
            if (!ggml_cannge_float_dtype(data->type) || !ggml_cannge_float_dtype(dst->type) ||
                (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64)) {
                return false;
            }
            if (data->ne[0] != dst->ne[0] || data->ne[2] != dst->ne[2] || data->ne[3] != dst->ne[3]) {
                return false;
            }
            return ggml_is_contiguous_rows(data) && ggml_is_contiguous_rows(dst) && ggml_is_contiguous(dst);
        }

        case GGML_OP_CONCAT: {
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            if (src0->type != src1->type || !ggml_cannge_float_dtype(src0->type)) {
                return false;
            }
            const int axis = (int) op->op_params[0];
            if (axis < 0 || axis >= GGML_MAX_DIMS) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (d != axis && src0->ne[d] != src1->ne[d]) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_GLU: {
            // single-input glu only: y = act(first half of ne[0]) * second half
            // (ggml-cpu ops.cpp ggml_compute_forward_reglu_f32). The swapped
            // forms, two-input splits and the clamp/oai variants stay on CPU
            if (src0 == nullptr || src1 != nullptr || !ggml_cannge_float_dtype(src0->type) ||
                !ggml_cannge_float_dtype(op->type) || op->type != src0->type) {
                return false;
            }
            if (ggml_get_op_params_i32(op, 1) != 0 || src0->ne[0] % 2 != 0 || !ggml_is_contiguous(src0)) {
                return false;
            }
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return true;
                default:
                    return false;
            }
        }

        case GGML_OP_ROPE: {
            // minimal subset: full-head NORMAL/NEOX rotation, no YaRN scaling
            // (freq_scale/ext/attn/betas at defaults), no freq factors (src2),
            // no n_offs; cos/sin tables are built inside the graph from src1
            const struct ggml_tensor * src2 = op->src[2];
            if (src0 == nullptr || src1 == nullptr || src2 != nullptr || !ggml_cannge_float_dtype(src0->type) ||
                op->type != src0->type || src1->type != GGML_TYPE_I32) {
                return false;
            }
            const int n_dims = ggml_get_op_params_i32(op, 1);
            const int mode   = ggml_get_op_params_i32(op, 2);
            if ((mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) || n_dims != src0->ne[0] ||
                n_dims % 2 != 0 || ggml_get_op_params_i32(op, 15) != 0) {
                return false;
            }
            // op_params floats: [5]freq_base [6]freq_scale [7]ext_factor
            // [8]attn_factor [9]beta_fast [10]beta_slow (ggml.c ggml_rope_impl)
            if (ggml_get_op_params_f32(op, 6) != 1.0f || ggml_get_op_params_f32(op, 7) != 0.0f ||
                ggml_get_op_params_f32(op, 8) != 1.0f || ggml_get_op_params_f32(op, 9) != 0.0f ||
                ggml_get_op_params_f32(op, 10) != 0.0f) {
                return false;
            }
            const int nd = ggml_n_dims(src0);
            return nd >= 3 && nd <= 4 && ggml_is_contiguous(src0);
        }

        default:
            return false;
    }
}

static bool ggml_backend_cannge_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_cannge_buffer_type_name;
}

static const ggml_backend_device_i ggml_backend_cannge_device_interface = {
    /* .get_name             = */ ggml_backend_cannge_device_get_name,
    /* .get_description      = */ ggml_backend_cannge_device_get_description,
    /* .get_memory           = */ ggml_backend_cannge_device_get_memory,
    /* .get_type             = */ ggml_backend_cannge_device_get_type,
    /* .get_props            = */ ggml_backend_cannge_device_get_props,
    /* .init_backend         = */ ggml_backend_cannge_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cannge_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_cannge_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cannge_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// registry

struct ggml_backend_cannge_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cannge_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CANNGE_NAME;
}

static size_t ggml_backend_cannge_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cannge_reg_context * ctx = (ggml_backend_cannge_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cannge_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cannge_reg_context * ctx = (ggml_backend_cannge_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_cannge_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cannge_reg_interface = {
    /* .get_name          = */ ggml_backend_cannge_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cannge_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cannge_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cannge_reg_get_proc_address,
};

static ggml_guid_t ggml_backend_cannge_guid() {
    static ggml_guid guid = { 0x2e, 0x63, 0x51, 0x87, 0x0a, 0xf5, 0x4e, 0x1d,
                              0x9b, 0x38, 0x77, 0x2a, 0xc4, 0xd5, 0xe6 };
    return &guid;
}

ggml_backend_reg_t ggml_backend_cannge_reg() {
    static ggml_backend_reg reg;
    static bool             initialized = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ACL_CHECK(aclInit(nullptr));

            ggml_backend_cannge_reg_context * ctx = new ggml_backend_cannge_reg_context;

            uint32_t device_count = 0;
            ACL_CHECK(aclrtGetDeviceCount(&device_count));
            if (device_count > GGML_CANNGE_MAX_DEVICES) {
                GGML_LOG_WARN("%s: device count %u exceeds the supported maximum %d, extra devices are ignored\n",
                              __func__, device_count, (int) GGML_CANNGE_MAX_DEVICES);
                device_count = GGML_CANNGE_MAX_DEVICES;
            }

            for (uint32_t i = 0; i < device_count; i++) {
                ggml_backend_cannge_device_context * dev_ctx = new ggml_backend_cannge_device_context();
                dev_ctx->device                              = (int32_t) i;
                dev_ctx->name                                = GGML_CANNGE_NAME + std::to_string(i);
                ggml_backend_cannge_set_device(dev_ctx->device);
                const char * soc_name = aclrtGetSocName();
                dev_ctx->description = soc_name != nullptr ? soc_name : "unknown";

                ggml_backend_dev_t dev = new ggml_backend_device{ /* .iface   = */ ggml_backend_cannge_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_cannge_reg_interface,
                                    /* .context     = */ ctx };
        }
        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cannge_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_cannge_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cannge_context * ctx = new ggml_backend_cannge_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cannge_backend =
        new ggml_backend{ /* .guid      = */ ggml_backend_cannge_guid(),
                          /* .interface = */ ggml_backend_cannge_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cannge_reg(), device),
                          /* .context   = */ ctx };

    return cannge_backend;
}

bool ggml_backend_is_cannge(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_cannge_guid());
}

ggml_backend_buffer_type_t ggml_backend_cannge_buffer_type(int32_t device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device < 0 || device >= ggml_backend_cannge_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cannge_buffer_types[GGML_CANNGE_MAX_DEVICES];

    static bool ggml_backend_cannge_buffer_type_initialized = false;

    if (!ggml_backend_cannge_buffer_type_initialized) {
        ggml_backend_reg_t reg = ggml_backend_cannge_reg();
        for (int32_t i = 0; i < ggml_backend_cannge_get_device_count(); i++) {
            ggml_backend_cannge_device_context * dev_ctx =
                (ggml_backend_cannge_device_context *) ggml_backend_reg_dev_get(reg, i)->context;
            ggml_backend_cannge_buffer_types[i] = {
                /* .iface   = */ ggml_backend_cannge_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(reg, i),
                /* .context = */ dev_ctx,
            };
        }
        ggml_backend_cannge_buffer_type_initialized = true;
    }

    return &ggml_backend_cannge_buffer_types[device];
}

int32_t ggml_backend_cannge_get_device_count() {
    uint32_t device_count = 0;
    ACL_CHECK(aclrtGetDeviceCount(&device_count));
    return (int32_t) device_count;
}

void ggml_backend_cannge_get_device_description(int32_t device, char * description, size_t description_size) {
    ggml_backend_cannge_set_device(device);
    const char * soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name != nullptr ? soc_name : "unknown");
}

void ggml_backend_cannge_get_device_memory(int32_t device, size_t * free, size_t * total) {
    ggml_backend_cannge_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cannge_reg)
