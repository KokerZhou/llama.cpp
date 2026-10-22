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

#include "graph.h"
#include "cpy.h"
#include "supports.h"
#include "ggml-cannge.h"
#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-cannge-impl.h"

#include <acl/acl.h>

#include <chrono>
#include <algorithm>
#include <map>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

void ggml_backend_cannge_set_device(int32_t device) {
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
    ggml_cannge_check(aclrtFree(ctx->dev_ptr), "aclrtFree", __func__);
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
    ggml_cannge_check(aclrtMemset((char *) tensor->data + offset, size, value, size), "aclrtMemset", __func__);
}

static void ggml_backend_cannge_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                  size_t offset, size_t size) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtMemcpy((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy", __func__);
}

static void ggml_backend_cannge_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                  size_t offset, size_t size) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtMemcpy(data, size, (const char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy", __func__);
}

static void ggml_backend_cannge_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cannge_buffer_context * ctx = (ggml_backend_cannge_buffer_context *) buffer->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size), "aclrtMemset", __func__);
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
    ggml_cannge_release_context(ctx);
    delete backend;
}

static void ggml_backend_cannge_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
                                                 size_t offset, size_t size) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtMemcpyAsync((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream), "aclrtMemcpyAsync", __func__);
    // synchronous: the caller reuses the host staging buffer right after; an
    // async copy would make the driver pin it until a later sync, ballooning
    // host memory during model load (seen: tens of GB pinned vs 64G cgroup)
    ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
}

static void ggml_backend_cannge_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                                 size_t offset, size_t size) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtMemcpyAsync(data, size, (const char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST, ctx->stream), "aclrtMemcpyAsync", __func__);
    const double dbg_s0 = g_dbg_time.on ? g_dbg_time.now() : 0;
    ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
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
        const bool copied = ggml_cannge_strided_copy(src, dst->data, false, ctx->stream);
        const bool drained = ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
        return copied && drained;
    }
    // cross-backend copies can involve host buffers (CPU fallback outputs),
    // ACL has no auto-direction memcpy: pick the kind from the buffer types
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        kind = ACL_MEMCPY_DEVICE_TO_HOST;
    }
    const bool copied = ggml_cannge_check(aclrtMemcpyAsync(dst->data, ggml_nbytes(dst), src->data, ggml_nbytes(src), kind, ctx->stream), "aclrtMemcpyAsync", __func__);
    // synchronize both directions: the peer backend's threads proceed without
    // any stream ordering - on H2D the host src can be reused/overwritten by
    // later host work before a deferred pageable copy reads it (same pinned-
    // page lesson as set_tensor, F15b); on D2H the CPU consumer reads dst
    // immediately. a sync here is the only correct interlock.
    const bool drained = ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
    if (!copied || !drained) {
        return false;
    }
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
    const double t0 = g_dbg_time.on ? g_dbg_time.now() : 0;
    ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
    if (g_dbg_time.on) {
        g_dbg_time.sync_calls++;
        g_dbg_time.sync_ms += g_dbg_time.now() - t0;
    }
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
    ggml_cannge_check(aclrtDestroyEvent((aclrtEvent) event->context), "aclrtDestroyEvent", __func__);
    delete event;
}

static void ggml_backend_cannge_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_cannge_device_context * dev_ctx = (ggml_backend_cannge_device_context *) dev->context;
    ggml_backend_cannge_set_device(dev_ctx->device);
    ggml_cannge_check(aclrtSynchronizeEvent((aclrtEvent) event->context), "aclrtSynchronizeEvent", __func__);
}

static void ggml_backend_cannge_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtRecordEvent((aclrtEvent) event->context, ctx->stream), "aclrtRecordEvent", __func__);
}

static void ggml_backend_cannge_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    ggml_backend_cannge_set_device(ctx->device);
    ggml_cannge_check(aclrtStreamWaitEvent(ctx->stream, (aclrtEvent) event->context), "aclrtStreamWaitEvent", __func__);
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
    ggml_cannge_check(aclrtGetMemInfo(ACL_HBM_MEM, free, total), "aclrtGetMemInfo", __func__);
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
            ggml_cannge_check(aclInit(nullptr), "aclInit", __func__);

            ggml_backend_cannge_reg_context * ctx = new ggml_backend_cannge_reg_context;

            uint32_t device_count = 0;
            ggml_cannge_check(aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount", __func__);
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
    if (!ctx->initialized) {
        GGML_LOG_ERROR("%s: failed to initialize device stream\n", __func__);
        ggml_cannge_release_context(ctx);
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
    ggml_cannge_check(aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount", __func__);
    return (int32_t) device_count;
}

void ggml_backend_cannge_get_device_description(int32_t device, char * description, size_t description_size) {
    ggml_backend_cannge_set_device(device);
    const char * soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name != nullptr ? soc_name : "unknown");
}

void ggml_backend_cannge_get_device_memory(int32_t device, size_t * free, size_t * total) {
    ggml_backend_cannge_set_device(device);
    ggml_cannge_check(aclrtGetMemInfo(ACL_HBM_MEM, free, total), "aclrtGetMemInfo", __func__);
}

GGML_BACKEND_DL_IMPL(ggml_backend_cannge_reg)
