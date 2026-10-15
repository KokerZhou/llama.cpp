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
#include "ggml-cannge-priv.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ge.h"
#include "graph-build.h"
#include "plan.h"

#include <acl/acl.h>

#include <chrono>
#include <fstream>
#include <algorithm>
#include <map>




// gert::Tensor: zero-copy execution binding (exe_graph/runtime/tensor.h)
#include <exe_graph/runtime/tensor.h>
#include <graph/graph.h>
#include <ge/ge_ir_build.h>
#include <graph/types.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


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

const char * ggml_backend_cannge_buffer_type_name(ggml_backend_buffer_type_t buft) {
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
    const double dbg_s0 = g_dbg_time.on ? g_dbg_time.now() : 0;
    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
    if (g_dbg_time.on) { g_dbg_time.cpy_sync += g_dbg_time.now() - dbg_s0; g_dbg_time.cpy_calls++; }
    if (getenv("GGML_CANNGE_DEBUG_IO")) {
        uint32_t a = 0, b = 0;
        const char * p = (const char *) data;
        for (size_t i = 0; i < size; i++) { a = (a + (uint8_t) p[i]) % 65535; b = (b + a) % 65535; }
        fprintf(stderr, "[DBG-IO] get %s ne=%lld,%lld,%lld,%lld nbytes=%zu sum=%08x\n", tensor->name,
                (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
                (long long) tensor->ne[3], size, (b << 16) | a);
    }
}


static bool ggml_backend_cannge_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                                 const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend_dst->context;
    ggml_backend_cannge_set_device(ctx->device);
    // layout-converting copy (strided src, dense split copy dst): gather via
    // host bounce, one H2D; the scheduler's split copies are dense so GE can
    // bind them directly without per-use staging
    if (!ggml_are_same_layout(src, dst)) {
        ggml_cannge_strided_copy(src, dst->data, false, ctx->stream);
        ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
        return true;
    }
    // cross-backend copies can involve host buffers (CPU fallback outputs),
    // ACL has no auto-direction memcpy: pick the kind from the buffer types
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        kind = ACL_MEMCPY_DEVICE_TO_HOST;
    }
    ACL_CHECK(aclrtMemcpyAsync(dst->data, ggml_nbytes(dst), src->data, ggml_nbytes(src), kind, ctx->stream));
    // synchronize both directions: the peer backend's threads proceed without
    // any stream ordering - on H2D the host src can be reused/overwritten by
    // later host work before a deferred pageable copy reads it (same pinned-
    // page lesson as set_tensor, F15b); on D2H the CPU consumer reads dst
    // immediately. a sync here is the only correct interlock.
    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
    if (getenv("GGML_CANNGE_DEBUG_IO")) {
        const size_t n = std::min(ggml_nbytes(src), (size_t) 4096);
        std::vector<char> sbuf(n), dbuf(n);
        const bool src_dev = !ggml_backend_buffer_is_host(src->buffer);
        const bool dst_dev = !ggml_backend_buffer_is_host(dst->buffer);
        if (src_dev) {
            aclrtMemcpy(sbuf.data(), n, src->data, n, ACL_MEMCPY_DEVICE_TO_HOST);
        } else {
            memcpy(sbuf.data(), src->data, n);
        }
        if (dst_dev) {
            aclrtMemcpy(dbuf.data(), n, dst->data, n, ACL_MEMCPY_DEVICE_TO_HOST);
        } else {
            memcpy(dbuf.data(), dst->data, n);
        }
        auto flet = [](const char * p, size_t n) {
            uint32_t a = 0, b = 0;
            for (size_t i = 0; i < n; i++) {
                a = (a + (uint8_t) p[i]) % 65535;
                b = (b + a) % 65535;
            }
            return (uint32_t)((b << 16) | a);
        };
        fprintf(stderr, "[DBG-IO] cpy %s -> %s kind=%d nbytes=%zu src=%08x dst=%08x %s", src->name, dst->name,
                (int) kind, ggml_nbytes(src), flet(sbuf.data(), n), flet(dbuf.data(), n),
                flet(sbuf.data(), n) == flet(dbuf.data(), n) ? "SAME" : "DIFF");
        // chunk fletchers to catch duplicated/stale halves beyond the first 4K
        {
            const size_t total = ggml_nbytes(src);
            for (size_t c = 4096; c + 4096 <= total && c <= 6 * 4096; c += 4096) {
                std::vector<char> sb2(4096), db2(4096);
                if (src_dev) {
                    aclrtMemcpy(sb2.data(), 4096, (const char *) src->data + c, 4096, ACL_MEMCPY_DEVICE_TO_HOST);
                } else {
                    memcpy(sb2.data(), (const char *) src->data + c, 4096);
                }
                if (dst_dev) {
                    aclrtMemcpy(db2.data(), 4096, (const char *) dst->data + c, 4096, ACL_MEMCPY_DEVICE_TO_HOST);
                } else {
                    memcpy(db2.data(), (const char *) dst->data + c, 4096);
                }
                fprintf(stderr, " +%zuk:%08x/%08x", c / 1024, flet(sb2.data(), 4096), flet(db2.data(), 4096));
            }
        }
        fprintf(stderr, "\n");
    }
    return true;
}

static void ggml_backend_cannge_synchronize(ggml_backend_t backend) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
}


// L0 gate of the fallback ladder: the whole-graph path is opt-in via env,
// read once; when off the backend reports no op support (default behavior)
bool ggml_backend_cannge_graph_env_enabled() {
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
    if (in_arena != nullptr) {
        aclrtFree(in_arena);
    }
    if (in_arena_host != nullptr) {
        aclrtFreeHost(in_arena_host);
    }
    if (fa_asl_dev != nullptr) {
        aclrtFree(fa_asl_dev);
    }
}

// 2D tiled repack: out[r*o_r + c*o_c] = in[r*i_r + c*i_c] for an [R x C]

static enum ggml_status ggml_backend_cannge_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;

    if (!ggml_backend_cannge_graph_env_enabled()) {
        GGML_LOG_ERROR("%s: compiled graph path disabled, set GGML_CANNGE_GRAPH=1 to enable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_cannge_set_device(ctx->device);
    const double dbg_t0 = g_dbg_time.on ? g_dbg_time.now() : 0;

    // static capability pre-check (L0 gate of the fallback ladder)
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_cannge_reg(), ctx->device);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }
        if (getenv("GGML_CANNGE_DEBUG_COMPILE") && node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr &&
            strstr(node->src[0]->name, "attn_output") != nullptr && node->src[1] != nullptr) {
            fprintf(stderr, "[DBG-WO-PLAN] cgraph %s: src1=%s ne=%lld,%lld,%lld,%lld\n", node->name,
                    node->src[1]->name, (long long) node->src[1]->ne[0], (long long) node->src[1]->ne[1],
                    (long long) node->src[1]->ne[2], (long long) node->src[1]->ne[3]);
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

    const double dbg_t1 = g_dbg_time.on ? g_dbg_time.now() : 0;
    const std::string signature = ggml_cannge_plan_signature(cgraph);
    if (getenv("GGML_CANNGE_DEBUG_SIG")) {
        fprintf(stderr, "[DBG-SIG] nodes=%d first=%s sig=%llu\n", cgraph->n_nodes, cgraph->nodes[0]->name,
                (unsigned long long) std::hash<std::string>{}(signature));
    }
    const double dbg_t1a = g_dbg_time.on ? g_dbg_time.now() : 0;

    ggml_cannge_plan * plan = nullptr;
    auto               it   = ctx->plans.find(signature);
    if (it != ctx->plans.end()) {
        plan = it->second;
    }

    // compile -> load once per signature, bind + execute on every call
    if (plan == nullptr) {
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COMPILE] start nodes=%d first=%s(%s) last=%s(%s) sig=%zu\n", cgraph->n_nodes,
                    cgraph->nodes[0]->name, ggml_op_name(cgraph->nodes[0]->op), cgraph->nodes[cgraph->n_nodes - 1]->name,
                    ggml_op_name(cgraph->nodes[cgraph->n_nodes - 1]->op), std::hash<std::string>{}(signature));
        }
        // build and compile first so a build failure is not cached; a
        // compile/load failure caches a FAILED plan below
        std::unique_ptr<ggml_cannge_plan> fresh(new ggml_cannge_plan);
        fresh->signature = signature;

        // IFA actual_seq_lengths buffer must exist before the build binds it
        if (io.fa_mask != nullptr && fresh->fa_asl_dev == nullptr &&
            aclrtMalloc(&fresh->fa_asl_dev, 8, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to allocate fa_asl buffer\n", __func__);
            fresh->fa_asl_dev = nullptr;
        }
        io.fa_asl_dev = fresh->fa_asl_dev;

        void * graph = nullptr;
        if (!ggml_cannge_graph_build(*fresh, io, cgraph, graph, err)) {
            GGML_LOG_ERROR("%s: graph build failed: %s\n", __func__, err.c_str());
            return GGML_STATUS_FAILED;
        }
        if (getenv("GGML_CANNGE_DUMP_GE")) {
            std::ofstream gs("/tmp/cannge_graph.txt", std::ios::app);
            gs << "===== graph_id " << fresh->graph_id << " sig " << std::hash<std::string>{}(signature) << " =====\n";
            (*(ge::Graph *) graph).Dump(ge::Graph::DumpFormat::kTxt, gs);
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
            if (g_dbg_time.on) g_dbg_time.evicts++;
        }
        plan->seq = ctx->next_plan_seq++;
        ctx->plans[signature] = plan;
    }

    const double dbg_t2 = g_dbg_time.on ? g_dbg_time.now() : 0;
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
        if (plan->input_staged_once.size() < io.inputs.size()) {
            plan->input_staged_once.resize(io.inputs.size(), 0);
        }
        bool staged_any      = false;
        bool staged_host_any = false; // dense host src: deferred pageable H2D needs the drain

        // pack the per-step dense host staged inputs into one pinned arena +
        // one async H2D (merged graphs stage ~150 tiny inputs per execute;
        // the per-copy launch cost dominated bind - batch them). weights
        // (stage-once) and view-aliased inputs keep their own paths
        std::vector<char>   in_arena_on(io.inputs.size(), 0);
        std::vector<size_t> in_arena_off(io.inputs.size(), 0);
        size_t              arena_need = 0;
        for (size_t i = 0; i < io.inputs.size(); i++) {
            ggml_tensor * t = io.inputs[i];
            if (!io.input_staged[i] || io.input_base[i] || plan->input_staged_once[i]) {
                continue;
            }
            const bool host = t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
            const bool wonce = !getenv("GGML_CANNGE_NO_WONCE") && t->op == GGML_OP_NONE &&
                               !(t->flags & GGML_TENSOR_FLAG_INPUT) && host && ggml_nbytes(t) > 64 * 1024;
            if (host && ggml_is_contiguous(t) && !wonce) {
                in_arena_on[i]  = 1;
                in_arena_off[i] = arena_need;
                arena_need += (ggml_nbytes(t) + 255) & ~(size_t) 255;
            }
        }
        if (arena_need > 0) {
            if (plan->in_arena_size < arena_need) {
                if (plan->in_arena != nullptr) {
                    ACL_CHECK(aclrtFree(plan->in_arena));
                }
                if (plan->in_arena_host != nullptr) {
                    ACL_CHECK(aclrtFreeHost(plan->in_arena_host));
                }
                if (aclrtMalloc(&plan->in_arena, arena_need, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
                    aclrtMallocHost(&plan->in_arena_host, arena_need) != ACL_SUCCESS) {
                    GGML_LOG_ERROR("%s: failed to allocate input arena (%zu bytes)\n", __func__, arena_need);
                    return GGML_STATUS_FAILED;
                }
                plan->in_arena_size = arena_need;
            }
            for (size_t i = 0; i < io.inputs.size(); i++) {
                if (in_arena_on[i]) {
                    memcpy((char *) plan->in_arena_host + in_arena_off[i], io.inputs[i]->data,
                           ggml_nbytes(io.inputs[i]));
                }
            }
            ACL_CHECK(aclrtMemcpyAsync(plan->in_arena, arena_need, plan->in_arena_host, arena_need,
                                       ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream));
            staged_any      = true;
            staged_host_any = true;
        }
        for (size_t i = 0; i < io.inputs.size(); i++) {
            ggml_tensor * t = io.inputs[i];
            // a view that aliases a previously registered input is bound with
            // its own declared shape (build_inputs used t) but the base's
            // buffer at the view's offset, so GE sees the same memory the
            // build's Slice/alias nodes operate on
            void * dev_ptr = io.input_base[i] ? (void *) ((char *) io.input_base[i]->data + io.input_base_offset[i])
                                              : t->data;
            if (in_arena_on[i]) {
                dev_ptr = (char *) plan->in_arena + in_arena_off[i];
            }
            if (io.input_staged[i] && !in_arena_on[i]) {
                if (plan->input_staging[i] == nullptr &&
                    aclrtMalloc(&plan->input_staging[i], ggml_nbytes(t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                    GGML_LOG_ERROR("%s: failed to allocate staging buffer for input %s (%zu bytes)\n", __func__, t->name,
                                   ggml_nbytes(t));
                    return GGML_STATUS_FAILED;
                }
                // big host-resident leaves that are not graph inputs are
                // weights: immutable, so stage once per plan instead of
                // re-uploading every execute. small tensors and the per-token
                // inputs (flagged INPUT) are re-staged every execute. The host
                // test is load-bearing: sched split copies are device-resident
                // op-NONE leaves too, but the scheduler refreshes them before
                // every compute - staging them once freezes every reuse of the
                // plan on the first execution's bytes (CONT_NC pass-2 kqv)
                const bool is_weight = !getenv("GGML_CANNGE_NO_WONCE") && t->op == GGML_OP_NONE &&
                                       !(t->flags & GGML_TENSOR_FLAG_INPUT) && t->buffer != nullptr &&
                                       ggml_backend_buffer_is_host(t->buffer) &&
                                       ggml_nbytes(t) > 64 * 1024;
                if (getenv("GGML_CANNGE_DEBUG_COMPILE") && ggml_nbytes(t) > 1024 * 1024 &&
                    (t->op != GGML_OP_NONE || (t->flags & GGML_TENSOR_FLAG_INPUT))) {
                    fprintf(stderr, "[DBG-WONCE] %s op=%d flags=%x not-a-weight-leaf\n", t->name, (int) t->op,
                            (unsigned) t->flags);
                }
                if (!is_weight || !plan->input_staged_once[i]) {
                    ggml_cannge_strided_copy(t, plan->input_staging[i], false, ctx->stream);
                    dev_ptr            = plan->input_staging[i];
                    staged_any         = true;
                    plan->input_staged_once[i] = is_weight ? 1 : 0;
                    // only a DENSE host source goes through the deferred
                    // pageable H2D; the bounce paths copy with blocking calls
                    if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer) &&
                        ggml_is_contiguous(t)) {
                        staged_host_any = true;
                    }
                } else {
                    dev_ptr = plan->input_staging[i];
                }
                if (g_dbg_time.on) {
                    g_dbg_time.stage_bytes += ggml_nbytes(t);
                    g_dbg_time.stage_names[t->name] += (double) ggml_nbytes(t);
                }
            }            gert::Tensor tensor;
            if (!ggml_cannge_bind_tensor(t, tensor, dev_ptr)) {
                return GGML_STATUS_FAILED;
            }
            if (getenv("GGML_CANNGE_DEBUG_IO")) {
                fprintf(stderr, "[DBG-IO] bind in[%zu] %s t->data=%p dev=%p staged=%d base=%s+%zu\n", i, t->name,
                        t->data, dev_ptr, (int) io.input_staged[i],
                        io.input_base[i] ? io.input_base[i]->name : "-", io.input_base_offset[i]);
            }
            inputs.push_back(std::move(tensor));
        }
        if (plan->fa_asl_dev != nullptr && io.fa_mask != nullptr) {
            // IFA actual_seq_lengths: count the visible prefix of the additive
            // mask (0 = attend, -inf = masked) and upload it; the mask buffer
            // was refreshed by the scheduler before this split ran
            ggml_tensor * m    = io.fa_mask;
            const size_t  es   = ggml_element_size(m);
            const int64_t n_kv = m->ne[0];
            std::vector<char> mbuf(n_kv * es);
            const bool m_host = m->buffer != nullptr && ggml_backend_buffer_is_host(m->buffer);
            if (m_host) {
                memcpy(mbuf.data(), m->data, mbuf.size());
            } else {
                ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
                ACL_CHECK(aclrtMemcpy(mbuf.data(), mbuf.size(), m->data, mbuf.size(), ACL_MEMCPY_DEVICE_TO_HOST));
            }
            int64_t asl = n_kv;
            if (es == 2) {
                for (int64_t i = 0; i < n_kv; i++) {
                    // signed-value compare: raw fp16 bits are not ordered
                    if (ggml_fp16_to_fp32(((const ggml_fp16_t *) mbuf.data())[i]) < -1.0e4f) {
                        asl = i;
                        break;
                    }
                }
            } else {
                for (int64_t i = 0; i < n_kv; i++) {
                    if (((const float *) mbuf.data())[i] < -1.0e4f) {
                        asl = i;
                        break;
                    }
                }
            }
            // blocking copy: an async H2D from a stack scalar would read
            // garbage once the block returns (F15b family)
            ACL_CHECK(aclrtMemcpy(plan->fa_asl_dev, 8, &asl, 8, ACL_MEMCPY_HOST_TO_DEVICE));
            if (getenv("GGML_CANNGE_DEBUG_IO")) {
                fprintf(stderr, "[DBG-FA] asl=%lld n_kv=%lld es=%zu host=%d mask0=%.1f\n", (long long) asl,
                        (long long) n_kv, es, (int) m_host,
                        es == 2 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) mbuf.data())[0])
                                : ((const float *) mbuf.data())[0]);
            }
            gert::Tensor at;
            at.MutableOriginShape().SetDimNum(1);
            at.MutableStorageShape().SetDimNum(1);
            at.MutableOriginShape().SetDim(0, 1);
            at.MutableStorageShape().SetDim(0, 1);
            at.SetOriginFormat(ge::FORMAT_ND);
            at.SetStorageFormat(ge::FORMAT_ND);
            at.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
            at.SetDataType(ge::DT_INT64);
            at.SetData(gert::TensorData((uint8_t *) plan->fa_asl_dev, nullptr, 8, gert::TensorPlacement::kOnDeviceHbm));
            inputs.push_back(std::move(at));
        }
        const double dbg_t2a = g_dbg_time.on ? g_dbg_time.now() : 0;
        if (staged_host_any) {
            // a dense staged host source may be reused by the CPU allocator as
            // soon as this call returns; the deferred pageable H2D could read
            // overwritten bytes (F15b pinned-page lesson). the blocking bounce
            // paths need no drain
            ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
        }
        const double dbg_t2b = g_dbg_time.on ? g_dbg_time.now() : 0;
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
            if (getenv("GGML_CANNGE_DEBUG_IO")) {
                fprintf(stderr, "[DBG-CPY] out_port %s t->data=%p\n", t->name, t->data);
            }
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
            if (getenv("GGML_CANNGE_DEBUG_IO")) {
                fprintf(stderr, "[DBG-CPY] out_check %s t->data=%p span=%zu overlaps=%d strided=%d ins=%zu\n", t->name,
                        t->data, ggml_nbytes(t), (int) overlaps, (int) strided, input_ranges.size());
            }
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

        if (getenv("GGML_CANNGE_DEBUG_IO")) {
            // checksum helper: cheap fletcher-32 over up to 4 KB read back from
            // the pointer's own side (device ptr -> D2H, host ptr -> direct)
            auto dbg_sum = [](const char * tag, ggml_tensor * t, bool dev) {
                const size_t n = std::min((size_t) ggml_nbytes(t), (size_t) 4096);
                std::vector<char> buf(n);
                if (dev) {
                    aclrtMemcpy(buf.data(), n, t->data, n, ACL_MEMCPY_DEVICE_TO_HOST);
                } else {
                    memcpy(buf.data(), t->data, n);
                }
                uint32_t a = 0, b = 0;
                for (size_t i = 0; i < n; i++) {
                    a = (a + (uint8_t) buf[i]) % 65535;
                    b = (b + a) % 65535;
                }
                fprintf(stderr, "[DBG-IO] %s %s ne=%lld,%lld,%lld,%lld sum=%08x", tag, t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                        (b << 16) | a);
                // chunk fletchers at 4 KB granularity across the whole span:
                // localizes strided/partial corruption to a head/token range
                {
                    const size_t total = ggml_nbytes(t);
                    const size_t csize = 4096;
                    for (size_t c = csize; c + csize <= total && c <= 8 * csize; c += csize) {
                        std::vector<char> cb(csize);
                        if (dev) {
                            aclrtMemcpy(cb.data(), csize, (const char *) t->data + c, csize, ACL_MEMCPY_DEVICE_TO_HOST);
                        } else {
                            memcpy(cb.data(), (const char *) t->data + c, csize);
                        }
                        uint32_t ca = 0, cb2 = 0;
                        for (size_t j = 0; j < csize; j++) {
                            ca = (ca + (uint8_t) cb[j]) % 65535;
                            cb2 = (cb2 + ca) % 65535;
                        }
                        fprintf(stderr, " +%zuk=%08x", c / 1024, (cb2 << 16) | ca);
                    }
                }
                // element 0,1,127 (head 0) and 128,129,255 (head 1): the head
                // boundary at index 128 catches partial writes per head
                const int idxs[6] = { 0, 1, 127, 128, 129, 255 };
                for (int e = 0; e < 6; e++) {
                    const size_t off = (size_t) idxs[e] * (t->type == GGML_TYPE_F16 ? 2 : 4);
                    if (off + (t->type == GGML_TYPE_F16 ? 2 : 4) > n) break;
                    float v;
                    if (t->type == GGML_TYPE_F16) {
                        uint16_t u;
                        memcpy(&u, buf.data() + off, 2);
                        v = ggml_fp16_to_fp32(u);
                    } else {
                        memcpy(&v, buf.data() + off, 4);
                    }
                    fprintf(stderr, " i%d=%.4g", idxs[e], v);
                }
                fprintf(stderr, "\n");
            };
            for (size_t i = 0; i < io.inputs.size(); i++) {
                const bool dev = !(io.inputs[i]->buffer && ggml_backend_buffer_is_host(io.inputs[i]->buffer));
                dbg_sum("in ", io.inputs[i], dev);
            }
        }
        const double dbg_t3 = g_dbg_time.on ? g_dbg_time.now() : 0;

        // SET_ROWS host snapshot. The scatter below runs after GE execute and
        // the staged-output copy-backs, but ggml-alloc already considers the
        // sched split-input copies dead after the last SET_ROWS node: later
        // tensors in the same graph reuse their bytes, and a copy-back
        // (e.g. kqv_out over the idx range, FA=on layer-35 crash) clobbers
        // them before the scatter reads them. Read both tensors into host
        // scratch at the only points where their content is final: idx
        // before execute (GE never writes it), data right after execute
        // (GE output materialization), both before any copy-back runs.
        std::vector<std::vector<int64_t>> sr_slots;
        std::vector<std::vector<char>>    sr_data;
        if (!io.set_rows.empty()) {
            sr_slots.resize(io.set_rows.size());
            for (size_t sri = 0; sri < io.set_rows.size(); sri++) {
                const ggml_cannge_set_rows_op & sr  = io.set_rows[sri];
                if (sr.in_graph) {
                    continue; // written by an in-graph GE ScatterUpdate
                }
                const int64_t                   nidx = sr.idx->ne[0];
                const size_t n_slots = (size_t) nidx * sr.idx->ne[1] * sr.idx->ne[2];
                const bool   idx_host = sr.idx->buffer != nullptr && ggml_backend_buffer_is_host(sr.idx->buffer);
                if (!idx_host) {
                    ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
                }
                if (sr.idx->type == GGML_TYPE_I64) {
                    sr_slots[sri].resize(n_slots);
                    if (idx_host) {
                        memcpy(sr_slots[sri].data(), sr.idx->data, n_slots * 8);
                    } else {
                        ACL_CHECK(aclrtMemcpy(sr_slots[sri].data(), n_slots * 8, sr.idx->data, ggml_nbytes(sr.idx),
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
                    sr_slots[sri].assign(tmp.begin(), tmp.end());
                }
            }
        }

        if (!ctx->ge->execute(plan->graph_id, ctx->stream, inputs, outputs, err)) {
            GGML_LOG_ERROR("%s: execute failed: %s\n", __func__, err.c_str());
            plan->state = ggml_cannge_plan::FAILED;
            return GGML_STATUS_FAILED;
        }
        const double dbg_t4 = g_dbg_time.on ? g_dbg_time.now() : 0;

        // data half of the SET_ROWS snapshot: GE has just materialized any
        // in-graph producer at t->data, and the sched has refreshed any
        // split-input copy before this split ran - but the copy-backs below
        // may overwrite the bytes before the scatter runs (see note above).
        // Staged outputs are the exception: their t->data is only filled by
        // the copy-back loop, so they are snapshotted after it.
        std::vector<bool> sr_data_staged(io.set_rows.size(), false);
        if (!io.set_rows.empty()) {
            sr_data.resize(io.set_rows.size());
            for (size_t sri = 0; sri < io.set_rows.size(); sri++) {
                const ggml_cannge_set_rows_op & sr = io.set_rows[sri];
                if (sr.in_graph) {
                    continue; // written by an in-graph GE ScatterUpdate
                }
                bool staged = false;
                for (const auto & sp : staged_ports) {
                    if (io.output_tensors[sp.first] == sr.data) {
                        staged = true;
                        break;
                    }
                }
                sr_data_staged[sri] = staged;
                if (staged) {
                    continue;
                }
                const bool data_host = sr.data->buffer != nullptr && ggml_backend_buffer_is_host(sr.data->buffer);
                sr_data[sri].resize(ggml_nbytes(sr.data));
                if (data_host) {
                    memcpy(sr_data[sri].data(), sr.data->data, sr_data[sri].size());
                } else {
                    // produced or refreshed on this stream just above: enqueue
                    // the D2H on the same stream, one shared drain after the
                    // copy-back loop covers all set_rows (was a per-row sync)
                    ACL_CHECK(aclrtMemcpyAsync(sr_data[sri].data(), sr_data[sri].size(), sr.data->data,
                                               sr_data[sri].size(), ACL_MEMCPY_DEVICE_TO_HOST, ctx->stream));
                }
            }
        }

        // staged output ports: host-aware copy from staging into the real ggml
        // buffer on the same stream, ordered after the graph; no internal sync,
        // ggml calls ggml_backend_synchronize
        for (const auto & sp : staged_ports) {
            ggml_tensor * t = io.output_tensors[sp.first];
            if (getenv("GGML_CANNGE_DEBUG_IO")) {
                fprintf(stderr, "[DBG-CPY] staged_port out %s t->data=%p staging=%p nb=%zu\n", t->name, t->data,
                        (void *) plan->staging[sp.first], ggml_nbytes(t));
            }
            if (sp.second) {
                ggml_cannge_strided_copy(t, plan->staging[sp.first], true, ctx->stream);
            } else {
                const bool host = t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
                const aclrtMemcpyKind kind = host ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE;
                ACL_CHECK(aclrtMemcpyAsync(t->data, ggml_nbytes(t), plan->staging[sp.first], ggml_nbytes(t),
                                           kind, ctx->stream));
            }
        }

        // SET_ROWS snapshot for staged outputs: t->data was filled by the
        // copy-back loop above. One shared drain covers those copy-backs and
        // every async D2H issued in the pre-loop (was a per-row sync)
        bool any_sr_data = false;
        for (size_t sri = 0; sri < io.set_rows.size(); sri++) {
            if (!io.set_rows[sri].in_graph && !sr_data_staged[sri]) {
                const ggml_tensor * d = io.set_rows[sri].data;
                if (d != nullptr && !(d->buffer != nullptr && ggml_backend_buffer_is_host(d->buffer))) {
                    any_sr_data = true;
                }
            }
            if (sr_data_staged[sri] && !io.set_rows[sri].in_graph) {
                any_sr_data = true;
            }
        }
        if (any_sr_data) {
            ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
        }
        for (size_t sri = 0; sri < io.set_rows.size(); sri++) {
            if (!sr_data_staged[sri]) {
                continue;
            }
            const ggml_cannge_set_rows_op & sr = io.set_rows[sri];
            if (sr.in_graph) {
                continue; // written by an in-graph GE ScatterUpdate
            }
            const bool data_host = sr.data->buffer != nullptr && ggml_backend_buffer_is_host(sr.data->buffer);
            sr_data[sri].resize(ggml_nbytes(sr.data));
            if (data_host) {
                memcpy(sr_data[sri].data(), sr.data->data, sr_data[sri].size());
            } else {
                ACL_CHECK(aclrtMemcpy(sr_data[sri].data(), sr_data[sri].size(), sr.data->data, sr_data[sri].size(),
                                      ACL_MEMCPY_DEVICE_TO_HOST));
            }
        }

        // side-effect writebacks (future SET support): host-aware copy back on
        // the same stream, ordered after the graph; no internal sync, ggml
        // calls ggml_backend_synchronize
        for (const ggml_cannge_writeback & wb : io.writebacks) {
            const bool src_host = wb.src->buffer != nullptr && ggml_backend_buffer_is_host(wb.src->buffer);
            const bool dst_host = wb.dst->buffer != nullptr && ggml_backend_buffer_is_host(wb.dst->buffer);
            if (src_host && dst_host) {
                memcpy(wb.dst->data, wb.src->data, ggml_nbytes(wb.dst));
            } else if (src_host && !dst_host) {
                ACL_CHECK(aclrtMemcpyAsync(wb.dst->data, ggml_nbytes(wb.dst), wb.src->data, ggml_nbytes(wb.src),
                                           ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream));
            } else if (!src_host && dst_host) {
                ACL_CHECK(aclrtMemcpyAsync(wb.dst->data, ggml_nbytes(wb.dst), wb.src->data, ggml_nbytes(wb.src),
                                           ACL_MEMCPY_DEVICE_TO_HOST, ctx->stream));
            } else {
                ACL_CHECK(aclrtMemcpyAsync(wb.dst->data, ggml_nbytes(wb.dst), wb.src->data, ggml_nbytes(wb.src),
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx->stream));
            }
        }

        // KV cache row scatter (SET_ROWS): dst[slot, i2, i3] = data[i, i2, i3]
        // with slot = idx[i, i2%ne12, i3%ne13], ggml-cpu ops.cpp set_rows_impl.
        // slots and row bytes come from the host snapshots taken around GE
        // execute: the ggml buffers may have been reused by later tensors in
        // this graph before this post-execute scatter runs (see snapshot note)
        for (size_t sri = 0; sri < io.set_rows.size(); sri++) {
            const ggml_cannge_set_rows_op & sr = io.set_rows[sri];
            if (sr.in_graph) {
                continue; // written by an in-graph GE ScatterUpdate
            }
            const int64_t nidx    = sr.idx->ne[0];
            const int64_t * slots = sr_slots[sri].data();
            const bool    dst_host  = sr.dst->buffer  != nullptr && ggml_backend_buffer_is_host(sr.dst->buffer);
            const size_t  es_d      = ggml_element_size(sr.data);
            const size_t  es_t      = ggml_element_size(sr.dst);
            // host-side cast scratch for the f32 rope output -> f16 cache case;
            // the snapshot is already host-resident, no per-row staging needed
            std::vector<char> cast_buf;
            if (es_d != es_t) {
                cast_buf.resize((size_t) sr.data->ne[0] * es_t);
            }
            for (int64_t i3 = 0; i3 < sr.data->ne[3]; i3++) {
                for (int64_t i2 = 0; i2 < sr.data->ne[2]; i2++) {
                    const int64_t i11 = i2 % sr.idx->ne[1];
                    const int64_t i12 = i3 % sr.idx->ne[2];
                    for (int64_t i = 0; i < nidx; i++) {
                        const int64_t sl =
                            slots[(size_t) i + (size_t) i11 * nidx + (size_t) i12 * nidx * sr.idx->ne[1]];
                        const char * src = sr_data[sri].data() + i * sr.data->nb[1] + i2 * sr.data->nb[2] +
                                           i3 * sr.data->nb[3];
                        if (getenv("GGML_CANNGE_DEBUG_IO") && i == 0 && i2 == 0 && i3 == 0) {
                            fprintf(stderr,
                                    "[DBG-CPY] set_rows %s(data=%p dst=%p dsth=%d sl0=%lld idx=%s(%p,raw snap=%lld nb=%zu)\n",
                                    sr.data->name, sr.data->data, sr.dst->data, (int) dst_host,
                                    (long long) slots[0], sr.idx->name, sr.idx->data,
                                    (long long) slots[0], ggml_nbytes(sr.idx));
                        }
                        char * dst_row = (char *) sr.dst->data + (size_t) sl * sr.dst->nb[1] + i2 * sr.dst->nb[2] +
                                         i3 * sr.dst->nb[3];
                        const size_t row_dst = (size_t) sr.data->ne[0] * es_t;
                        if (es_d == es_t) {
                            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                                fprintf(stderr, "[DBG-SR] row %s->%s src=%p dst=%p dsth=%d\n",
                                        sr.data->name, sr.dst->name, src, dst_row, (int) dst_host);
                            }
                            if (dst_host) {
                                memcpy(dst_row, src, row_dst);
                            } else {
                                ACL_CHECK(aclrtMemcpyAsync(dst_row, row_dst, src, row_dst,
                                                           ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream));
                            }
                        } else {
                            // cast in host memory, then copy to the destination side
                            if (sr.data->type == GGML_TYPE_F32 && sr.dst->type == GGML_TYPE_F16) {
                                for (int64_t e = 0; e < sr.data->ne[0]; e++) {
                                    ((ggml_fp16_t *) cast_buf.data())[e] =
                                        ggml_fp32_to_fp16(((const float *) src)[e]);
                                }
                            } else {
                                for (int64_t e = 0; e < sr.data->ne[0]; e++) {
                                    ((float *) cast_buf.data())[e] =
                                        ggml_fp16_to_fp32(((const ggml_fp16_t *) src)[e]);
                                }
                            }
                            if (dst_host) {
                                memcpy(dst_row, cast_buf.data(), row_dst);
                            } else {
                                ACL_CHECK(aclrtMemcpyAsync(dst_row, row_dst, cast_buf.data(), row_dst,
                                                           ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream));
                            }
                        }
                    }
                }
            }
        }
        if (getenv("GGML_CANNGE_DEBUG_IO")) {
            ACL_CHECK(aclrtSynchronizeStream(ctx->stream));
            for (size_t i = 0; i < io.output_tensors.size(); i++) {
                const bool dev = !(io.output_tensors[i]->buffer && ggml_backend_buffer_is_host(io.output_tensors[i]->buffer));
                const size_t n = std::min((size_t) ggml_nbytes(io.output_tensors[i]), (size_t) 4096);
                std::vector<char> buf(n);
                if (dev) {
                    aclrtMemcpy(buf.data(), n, io.output_tensors[i]->data, n, ACL_MEMCPY_DEVICE_TO_HOST);
                } else {
                    memcpy(buf.data(), io.output_tensors[i]->data, n);
                }
                uint32_t a = 0, b = 0;
                for (size_t j = 0; j < n; j++) {
                    a = (a + (uint8_t) buf[j]) % 65535;
                    b = (b + a) % 65535;
                }
                fprintf(stderr, "[DBG-IO] out %s%s ne=%lld,%lld,%lld,%lld nb=%lld,%lld,%lld,%lld sum=%08x",
                        ggml_is_contiguous(io.output_tensors[i]) ? "D " : "S ",
                        io.output_tensors[i]->name,
                        (long long) io.output_tensors[i]->ne[0], (long long) io.output_tensors[i]->ne[1],
                        (long long) io.output_tensors[i]->ne[2], (long long) io.output_tensors[i]->ne[3],
                        (long long) io.output_tensors[i]->nb[0], (long long) io.output_tensors[i]->nb[1],
                        (long long) io.output_tensors[i]->nb[2], (long long) io.output_tensors[i]->nb[3], (b << 16) | a);
                // chunked fletchers localize corruption to a head/token range
                {
                    const size_t total = ggml_nbytes(io.output_tensors[i]);
                    const size_t csize = 4096;
                    for (size_t c = csize; c + csize <= total && c <= 8 * csize; c += csize) {
                        std::vector<char> cb(csize);
                        if (dev) {
                            aclrtMemcpy(cb.data(), csize, (const char *) io.output_tensors[i]->data + c, csize,
                                        ACL_MEMCPY_DEVICE_TO_HOST);
                        } else {
                            memcpy(cb.data(), (const char *) io.output_tensors[i]->data + c, csize);
                        }
                        uint32_t ca = 0, cb2 = 0;
                        for (size_t j = 0; j < csize; j++) {
                            ca = (ca + (uint8_t) cb[j]) % 65535;
                            cb2 = (cb2 + ca) % 65535;
                        }
                        fprintf(stderr, " +%zuk=%08x", c / 1024, (cb2 << 16) | ca);
                    }
                }
                // sample indices 0,1,2,127,128,255 to catch per-head partial writes
                const ggml_tensor * ot = io.output_tensors[i];
                const int idxs[6] = { 0, 1, 2, 127, 128, 255 };
                for (int e = 0; e < 6; e++) {
                    const size_t off = (size_t) idxs[e] * (ot->type == GGML_TYPE_F16 ? 2 : 4);
                    if (off + (ot->type == GGML_TYPE_F16 ? 2 : 4) > n) break;
                    float v;
                    if (ot->type == GGML_TYPE_F16) {
                        uint16_t u;
                        memcpy(&u, buf.data() + off, 2);
                        v = ggml_fp16_to_fp32(u);
                    } else if (ot->type == GGML_TYPE_F32) {
                        memcpy(&v, buf.data() + off, 4);
                    } else {
                        break;
                    }
                    fprintf(stderr, " i%d=%.4g", idxs[e], v);
                }
                fprintf(stderr, "\n");
            }
        }
        if (g_dbg_time.on) {
            const double t = g_dbg_time.now();
            g_dbg_time.analyze += dbg_t1 - dbg_t0;
            g_dbg_time.lookup += dbg_t2 - dbg_t1;
            g_dbg_time.sig += dbg_t1a - dbg_t1;
            g_dbg_time.find += dbg_t2 - dbg_t1a;
            g_dbg_time.bind += dbg_t3 - dbg_t2;
            g_dbg_time.stage += dbg_t2a - dbg_t2;
            g_dbg_time.stage_sync += dbg_t2b - dbg_t2a;
            g_dbg_time.obind += dbg_t3 - dbg_t2b;
            g_dbg_time.exec += dbg_t4 - dbg_t3;
            g_dbg_time.post += t - dbg_t4;
            g_dbg_time.total += t - dbg_t0;
            if (++g_dbg_time.calls % 1000 == 0) g_dbg_time.dump();
        }
        return GGML_STATUS_SUCCESS;
    }

    return GGML_STATUS_FAILED;
}

// split-boundary events: let the scheduler use stream-level waits instead
// of a full host sync at every split boundary (the n_copies=1 fallback). The
// sched only reaches these when events exist (created per backend below and
// in ggml-backend.cpp for n_copies==1)
static ggml_backend_event_t ggml_backend_cannge_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    ggml_backend_cannge_set_device(dev_ctx->device);
    aclrtEvent ev = nullptr;
    if (aclrtCreateEvent(&ev) != ACL_SUCCESS || ev == nullptr) {
        return nullptr;
    }
    auto * event = new ggml_backend_event{ dev, (void *) ev };
    return event;
}

static void ggml_backend_cannge_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) event->device->context;
    ggml_backend_cannge_set_device(dev_ctx->device);
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));
    delete event;
}

static void ggml_backend_cannge_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    ggml_backend_cannge_set_device(dev_ctx->device);
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));
}

static void ggml_backend_cannge_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtRecordEvent((aclrtEvent) event->context, ctx->stream));
}

static void ggml_backend_cannge_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ACL_CHECK(aclrtStreamWaitEvent(ctx->stream, (aclrtEvent) event->context));
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
    /* .event_record            = */ ggml_backend_cannge_event_record,
    /* .event_wait              = */ ggml_backend_cannge_event_wait,
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
    /* .event_new            = */ ggml_backend_cannge_device_event_new,
    /* .event_free           = */ ggml_backend_cannge_device_event_free,
    /* .event_synchronize    = */ ggml_backend_cannge_device_event_synchronize,
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
