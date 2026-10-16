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

#include "ggml-cannge-runtime.h"
#include "ggml-cannge.h"
#include "ggml-impl.h"
#include "graph-build.h"

#include <exe_graph/runtime/tensor.h>
#include <graph/graph.h>
#include <graph/types.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

ggml_backend_cannge_context::ggml_backend_cannge_context(int32_t device) : device(device), name(GGML_CANNGE_NAME + std::to_string(device)) {
    ggml_backend_cannge_set_device(device);
    ggml_cannge_check(aclrtCreateStream(&stream), "aclrtCreateStream", __func__);
}

ggml_backend_cannge_context::~ggml_backend_cannge_context() {
    ggml_backend_cannge_set_device(device);
    // GE teardown order: sync stream, drop session (via ge), then GEFinalize
    if (stream != nullptr) {
        ggml_cannge_check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", __func__);
    }
    ge.reset();
    plans.clear();
    if (stream != nullptr) {
        ggml_cannge_check(aclrtDestroyStream(stream), "aclrtDestroyStream", __func__);
    }
}

void ggml_backend_cannge_context::evict_oldest_plan() {
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
            ggml_cannge_check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", __func__);
        }
        if (ge != nullptr && oldest->second->graph_id != 0) {
            ge->remove(oldest->second->graph_id);
        }
        plans.erase(oldest);
    }
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

static ggml_status ggml_cannge_transfer_failed(ggml_backend_cannge_context * ctx, ggml_cannge_plan * plan) {
    std::fill(plan->input_staged_sources.begin(), plan->input_staged_sources.end(), nullptr);
    ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "drain failed transfer", __func__);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_cannge_finish_execution(
        ggml_backend_cannge_context * ctx, ggml_cannge_plan * plan, const ggml_cannge_plan_io & io,
        const std::vector<std::pair<size_t, bool>> & staged_ports,
        const std::vector<std::vector<int64_t>> & sr_slots) {
    std::vector<std::vector<char>> sr_data;
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
                if (!ggml_cannge_check(aclrtMemcpyAsync(sr_data[sri].data(), sr_data[sri].size(), sr.data->data,
                                           sr_data[sri].size(), ACL_MEMCPY_DEVICE_TO_HOST, ctx->stream), "aclrtMemcpyAsync", __func__)) {
                    return ggml_cannge_transfer_failed(ctx, plan);
                }
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
            if (!ggml_cannge_strided_copy(t, plan->staging[sp.first], true, ctx->stream)) {
                return ggml_cannge_transfer_failed(ctx, plan);
            }
        } else {
            const bool host = t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
            const aclrtMemcpyKind kind = host ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE;
            if (!ggml_cannge_check(aclrtMemcpyAsync(t->data, ggml_nbytes(t), plan->staging[sp.first], ggml_nbytes(t),
                                           kind, ctx->stream), "aclrtMemcpyAsync", __func__)) {
                return ggml_cannge_transfer_failed(ctx, plan);
            }
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
        if (!ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__)) {
            return ggml_cannge_transfer_failed(ctx, plan);
        }
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
            if (!ggml_cannge_check(aclrtMemcpy(sr_data[sri].data(), sr_data[sri].size(), sr.data->data, sr_data[sri].size(),
                                  ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy", __func__)) {
                return ggml_cannge_transfer_failed(ctx, plan);
            }
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
        } else {
            const aclrtMemcpyKind kind = src_host ? ACL_MEMCPY_HOST_TO_DEVICE :
                                        dst_host ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE;
            const aclError error = aclrtMemcpyAsync(wb.dst->data, ggml_nbytes(wb.dst), wb.src->data,
                                                   ggml_nbytes(wb.src), kind, ctx->stream);
            if (!ggml_cannge_check(error, "writeback", __func__)) {
                return ggml_cannge_transfer_failed(ctx, plan);
            }
        }
    }

    for (size_t sri = 0; sri < io.set_rows.size(); sri++) {
        const ggml_cannge_set_rows_op & sr = io.set_rows[sri];
        if (!sr.in_graph && !ggml_cannge_set_rows(sr, sr_slots[sri], sr_data[sri], ctx->stream)) {
            GGML_LOG_ERROR("%s: SET_ROWS copy failed for %s\n", __func__, sr.data->name);
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_cannge_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
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
    const std::string signature = ggml_cannge_plan_signature(cgraph, io);
    if (getenv("GGML_CANNGE_DEBUG_SIG")) {
        fprintf(stderr, "[DBG-SIG] nodes=%d first=%s sig=%llu\n", cgraph->n_nodes, cgraph->nodes[0]->name,
                (unsigned long long) std::hash<std::string>{}(signature));
    }
    const double dbg_t1a = g_dbg_time.on ? g_dbg_time.now() : 0;

    ggml_cannge_plan * plan = nullptr;
    auto               it   = ctx->plans.find(signature);
    if (it != ctx->plans.end()) {
        plan = it->second.get();
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
        std::unique_ptr<ge::Graph> graph_owner((ge::Graph *) graph);
        if (getenv("GGML_CANNGE_DUMP_GE")) {
            std::ofstream gs("/tmp/cannge_graph.txt", std::ios::app);
            gs << "===== graph_id " << fresh->graph_id << " sig " << std::hash<std::string>{}(signature) << " =====\n";
            graph_owner->Dump(ge::Graph::DumpFormat::kTxt, gs);
        }
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COMPILE] ES build done nodes=%d, GE compile...\n", cgraph->n_nodes);
        }

        if (ctx->ge == nullptr) {
            ctx->ge.reset(ggml_cannge_ge_context::create(ctx->device, err));
            if (ctx->ge == nullptr) {
                GGML_LOG_ERROR("%s: GE init failed: %s\n", __func__, err.c_str());
                return GGML_STATUS_FAILED;
            }
        }

        // the session owns an IR copy now, release the caller-side graph
        const bool compiled = ctx->ge->add_and_compile(graph_owner.get(), fresh->graph_id, err);
        graph_owner.reset();
        if (!compiled) {
            GGML_LOG_ERROR("%s: compile failed: %s\n", __func__, err.c_str());
            fresh->state = ggml_cannge_plan::FAILED;
        } else {
            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-COMPILE] done nodes=%d\n", cgraph->n_nodes);
            }
            fresh->state = ggml_cannge_plan::COMPILED;
        }

        if (ctx->plans.size() >= GGML_CANNGE_MAX_PLANS) {
            ctx->evict_oldest_plan();
            if (g_dbg_time.on) g_dbg_time.evicts++;
        }
        fresh->seq = ctx->next_plan_seq++;
        plan = fresh.get();
        ctx->plans.emplace(signature, std::move(fresh));
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
        if (plan->input_staged_sources.size() < io.inputs.size()) {
            plan->input_staged_sources.resize(io.inputs.size(), nullptr);
        }
        std::vector<const void *> staging_sources(io.inputs.size());
        for (size_t i = 0; i < io.inputs.size(); i++) {
            staging_sources[i] = ggml_cannge_staging_source(io.inputs[i]);
            if (staging_sources[i] != plan->input_staged_sources[i]) {
                plan->input_staged_sources[i] = nullptr;
            }
        }
        bool staged_any      = false;
        bool staged_host_any = false; // drain host transfers and validate reusable staging

        // pack the per-step dense host staged inputs into one pinned arena +
        // one async H2D (merged graphs stage ~150 tiny inputs per execute;
        // the per-copy launch cost dominated bind - batch them). weights
        // (stage-once) and view-aliased inputs keep their own paths
        std::vector<char>   in_arena_on(io.inputs.size(), 0);
        std::vector<size_t> in_arena_off(io.inputs.size(), 0);
        size_t              arena_need = 0;
        for (size_t i = 0; i < io.inputs.size(); i++) {
            ggml_tensor * t = io.inputs[i];
            if (!io.input_staged[i] || io.input_base[i] || plan->input_staged_sources[i]) {
                continue;
            }
            const bool host = t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
            if (host && ggml_is_contiguous(t) && staging_sources[i] == nullptr) {
                const size_t bytes = ggml_nbytes(t);
                const size_t max_size = std::numeric_limits<size_t>::max();
                if (bytes > max_size - 255 || ((bytes + 255) & ~(size_t) 255) > max_size - arena_need) {
                    GGML_LOG_ERROR("%s: input arena size overflow\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                in_arena_on[i]  = 1;
                in_arena_off[i] = arena_need;
                arena_need += (bytes + 255) & ~(size_t) 255;
            }
        }
        if (arena_need > 0) {
            if (!ggml_cannge_reserve_input_arena(plan->in_arena, plan->in_arena_host, plan->in_arena_size,
                                                arena_need, ctx->stream)) {
                GGML_LOG_ERROR("%s: failed to reserve input arena (%zu bytes)\n", __func__, arena_need);
                return GGML_STATUS_FAILED;
            }
            for (size_t i = 0; i < io.inputs.size(); i++) {
                if (in_arena_on[i]) {
                    memcpy((char *) plan->in_arena_host + in_arena_off[i], io.inputs[i]->data,
                           ggml_nbytes(io.inputs[i]));
                }
            }
            if (!ggml_cannge_check(aclrtMemcpyAsync(plan->in_arena, arena_need, plan->in_arena_host, arena_need,
                                           ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream), "aclrtMemcpyAsync", __func__)) {
                return ggml_cannge_transfer_failed(ctx, plan);
            }
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
                if (getenv("GGML_CANNGE_DEBUG_COMPILE") && ggml_nbytes(t) > 1024 * 1024 &&
                    (t->op != GGML_OP_NONE || (t->flags & GGML_TENSOR_FLAG_INPUT))) {
                    fprintf(stderr, "[DBG-WONCE] %s op=%d flags=%x not-a-weight-leaf\n", t->name, (int) t->op,
                            (unsigned) t->flags);
                }
                if (plan->input_staged_sources[i] == nullptr) {
                    if (!ggml_cannge_strided_copy(t, plan->input_staging[i], false, ctx->stream)) {
                        return ggml_cannge_transfer_failed(ctx, plan);
                    }
                    dev_ptr            = plan->input_staging[i];
                    staged_any         = true;
                    if (staging_sources[i] != nullptr) {
                        staged_host_any = true;
                    }
                    // Dense host sources must remain valid until the copy completes.
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
            }
            gert::Tensor tensor;
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
                ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
                ggml_cannge_check(aclrtMemcpy(mbuf.data(), mbuf.size(), m->data, mbuf.size(), ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy", __func__);
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
            ggml_cannge_check(aclrtMemcpy(plan->fa_asl_dev, 8, &asl, 8, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy", __func__);
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
            // overwritten bytes (F15b pinned-page lesson). Also validate new
            // reusable staging before publishing its source identity.
            if (!ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__)) {
                return ggml_cannge_transfer_failed(ctx, plan);
            }
        }
        plan->input_staged_sources = staging_sources;
        const double dbg_t2b = g_dbg_time.on ? g_dbg_time.now() : 0;
        // output ports whose device range overlaps an input range (aliased
        // views, e.g. PERMUTE of an input) must not be bound in place: GE
        // treats undeclared input/output overlap as unvalidated UB and may
        // silently skip the output write (GE source: runtime/v2/kernel/
        // common_kernel_impl/memory_copy.cc:574-579 TensorToOut, src==dst
        // case, @00ecb5c). The declared in-place channel
        // ge.exec.outputReuseInputMemIndexes requires exact address equality,
        // which offset views do not satisfy (GE source: docs/zh/design/features/
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
                    ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
                }
                if (sr.idx->type == GGML_TYPE_I64) {
                    sr_slots[sri].resize(n_slots);
                    if (idx_host) {
                        memcpy(sr_slots[sri].data(), sr.idx->data, n_slots * 8);
                    } else {
                        ggml_cannge_check(aclrtMemcpy(sr_slots[sri].data(), n_slots * 8, sr.idx->data, ggml_nbytes(sr.idx),
                                              ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy", __func__);
                    }
                } else {
                    std::vector<int32_t> tmp(n_slots);
                    if (idx_host) {
                        memcpy(tmp.data(), sr.idx->data, n_slots * 4);
                    } else {
                        ggml_cannge_check(aclrtMemcpy(tmp.data(), n_slots * 4, sr.idx->data, ggml_nbytes(sr.idx),
                                              ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy", __func__);
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

        if (ggml_cannge_finish_execution(ctx, plan, io, staged_ports, sr_slots) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
        if (getenv("GGML_CANNGE_DEBUG_IO")) {
            ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__);
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
