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

#include "graph.h"
#include "cpy.h"
#include "debug.h"
#include "supports.h"
#include "ggml-cannge-impl.h"
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

static ggml_status ggml_cannge_transfer_failed(ggml_backend_cannge_context * ctx, ggml_cannge_plan * plan) {
    std::fill(plan->input_staged_sources.begin(), plan->input_staged_sources.end(), nullptr);
    if (ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "drain failed transfer", __func__)) {
        plan->in_arena_pending = false;
    }
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
            if (!ggml_cannge_read_snapshot(sr.data, ggml_nbytes(sr.data), ctx->stream, sr_data[sri])) {
                return ggml_cannge_transfer_failed(ctx, plan);
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

    // Staged snapshots must observe copy-back completion, including copies into host tensors.
    if (std::find(sr_data_staged.begin(), sr_data_staged.end(), true) != sr_data_staged.end()) {
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
        if (!ggml_cannge_read_snapshot(sr.data, ggml_nbytes(sr.data), ctx->stream, sr_data[sri])) {
            return ggml_cannge_transfer_failed(ctx, plan);
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

static ggml_cannge_plan * ggml_cannge_prepare_plan(ggml_backend_cannge_context * ctx, ggml_cgraph * cgraph,
        ggml_cannge_plan_io & io, const std::string & signature, std::string & err) {
    ggml_cannge_plan * plan = nullptr;
    auto               it   = ctx->plans.find(signature);
    if (it != ctx->plans.end()) {
        plan = it->second.get();
    }

    // compile -> load once per signature, bind + execute on every call
    if (plan == nullptr) {
        if (ctx->plans.size() >= GGML_CANNGE_MAX_PLANS) {
            if (!ctx->evict_oldest_plan()) {
                err = "failed to evict cached graph";
                GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
                return nullptr;
            }
            if (g_dbg_time.on) g_dbg_time.evicts++;
        }
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COMPILE] start nodes=%d first=%s(%s) last=%s(%s) sig=%zu t=%.1f\n", cgraph->n_nodes,
                    cgraph->nodes[0]->name, ggml_op_name(cgraph->nodes[0]->op), cgraph->nodes[cgraph->n_nodes - 1]->name,
                    ggml_op_name(cgraph->nodes[cgraph->n_nodes - 1]->op), std::hash<std::string>{}(signature),
                    g_dbg_time.now() / 1000.0);
        }
        // build and compile first so a build failure is not cached; a
        // compile/load failure caches a FAILED plan below
        std::unique_ptr<ggml_cannge_plan> fresh(new ggml_cannge_plan);
        fresh->signature = signature;

        // IFA actual_seq_lengths buffer must exist before the build binds it
        if (io.fa_mask != nullptr && fresh->fa_asl_dev == nullptr &&
            aclrtMalloc(&fresh->fa_asl_dev, 8, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to allocate fa_asl buffer\n", __func__);
            return nullptr;
        }
        io.fa_asl_dev = fresh->fa_asl_dev;

        void * graph = nullptr;
        if (!ggml_cannge_graph_build(*fresh, io, cgraph, graph, err)) {
            GGML_LOG_ERROR("%s: graph build failed: %s\n", __func__, err.c_str());
            return nullptr;
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
                return nullptr;
            }
        }

        // the session owns an IR copy now, release the caller-side graph
        // cache key: the plan signature covers op sequence, shapes, strides,
        // port order and options; the emit version covers the ES lowering
        // (bump GGML_CANNGE_EMIT_VERSION when the build emits different ops
        // for the same ggml graph, else a stale disk entry is restored)
        char graph_key[48];
        snprintf(graph_key, sizeof(graph_key), "cannge_v%d_%016zx", GGML_CANNGE_EMIT_VERSION,
                 std::hash<std::string>{}(signature));
        const bool compiled = ctx->ge->add_and_compile(graph_owner.get(), fresh->graph_id, err, graph_key);
        graph_owner.reset();
        if (!compiled) {
            GGML_LOG_ERROR("%s: compile failed: %s\n", __func__, err.c_str());
            fresh->state = ggml_cannge_plan::FAILED;
        } else {
            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-COMPILE] done nodes=%d t=%.1f\n", cgraph->n_nodes, g_dbg_time.now() / 1000.0);
            }
            fresh->state = ggml_cannge_plan::COMPILED;
        }

        fresh->seq = ctx->next_plan_seq++;
        plan = fresh.get();
        ctx->plans.emplace(signature, std::move(fresh));
    }

    if (plan->state == ggml_cannge_plan::FAILED) {
        return nullptr;
    }

    if (plan->state == ggml_cannge_plan::COMPILED) {
        if (!ctx->ge->load(plan->graph_id, ctx->stream, err)) {
            GGML_LOG_ERROR("%s: load failed: %s\n", __func__, err.c_str());
            plan->state = ggml_cannge_plan::FAILED;
            return nullptr;
        }
        plan->state = ggml_cannge_plan::LOADED;
    }

    return plan->state == ggml_cannge_plan::LOADED ? plan : nullptr;
}

static ggml_status ggml_cannge_bind_inputs(ggml_backend_cannge_context * ctx, ggml_cannge_plan * plan,
        const ggml_cannge_plan_io & io, std::vector<gert::Tensor> & inputs, double & staged_at, double & ready_at) {
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
                return ggml_cannge_transfer_failed(ctx, plan);
            }
            in_arena_on[i]  = 1;
            in_arena_off[i] = arena_need;
            arena_need += (bytes + 255) & ~(size_t) 255;
        }
    }
    if (arena_need > 0) {
        if (!ggml_cannge_reserve_input_arena(plan->in_arena, plan->in_arena_host, plan->in_arena_size,
                                            arena_need, ctx->stream, plan->in_arena_pending)) {
            GGML_LOG_ERROR("%s: failed to reserve input arena (%zu bytes)\n", __func__, arena_need);
            return ggml_cannge_transfer_failed(ctx, plan);
        }
        plan->in_arena_pending = false;
        for (size_t i = 0; i < io.inputs.size(); i++) {
            if (in_arena_on[i]) {
                memcpy((char *) plan->in_arena_host + in_arena_off[i], io.inputs[i]->data,
                       ggml_nbytes(io.inputs[i]));
            }
        }
        // A failed enqueue is conservatively treated as pending until a successful drain.
        plan->in_arena_pending = true;
        if (!ggml_cannge_check(aclrtMemcpyAsync(plan->in_arena, arena_need, plan->in_arena_host, arena_need,
                                       ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream), "aclrtMemcpyAsync", __func__)) {
            return ggml_cannge_transfer_failed(ctx, plan);
        }
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
                return ggml_cannge_transfer_failed(ctx, plan);
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
            return ggml_cannge_transfer_failed(ctx, plan);
        }
        if (getenv("GGML_CANNGE_DEBUG_IO")) {
            fprintf(stderr, "[DBG-IO] bind in[%zu] %s t->data=%p dev=%p staged=%d base=%s+%zu\n", i, t->name,
                    t->data, dev_ptr, (int) io.input_staged[i],
                    io.input_base[i] ? io.input_base[i]->name : "-", io.input_base_offset[i]);
        }
        inputs.push_back(std::move(tensor));
    }
    if (plan->fa_asl_dev != nullptr && io.fa_mask != nullptr) {
        if (!ggml_cannge_upload_attention_length(io.fa_mask, plan->fa_asl_dev, ctx->stream)) {
            GGML_LOG_ERROR("%s: attention mask transfer failed for %s\n", __func__, io.fa_mask->name);
            return ggml_cannge_transfer_failed(ctx, plan);
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
    staged_at = g_dbg_time.on ? g_dbg_time.now() : 0;
    if (staged_host_any) {
        // a dense staged host source may be reused by the CPU allocator as
        // soon as this call returns; the deferred pageable H2D could read
        // overwritten bytes (F15b pinned-page lesson). Also validate new
        // reusable staging before publishing its source identity.
        if (!ggml_cannge_check(aclrtSynchronizeStream(ctx->stream), "aclrtSynchronizeStream", __func__)) {
            return ggml_cannge_transfer_failed(ctx, plan);
        }
        plan->in_arena_pending = false;
    }
    plan->input_staged_sources = staging_sources;
    ready_at = g_dbg_time.on ? g_dbg_time.now() : 0;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_cannge_bind_outputs(ggml_backend_cannge_context * ctx, ggml_cannge_plan * plan,
        const ggml_cannge_plan_io & io, std::vector<gert::Tensor> & outputs,
        std::vector<std::pair<size_t, bool>> & staged_ports) {
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
                return ggml_cannge_transfer_failed(ctx, plan);
            }
            dev_ptr = plan->staging[i];
            staged_ports.push_back({ i, strided });
        }
        gert::Tensor tensor;
        if (!ggml_cannge_bind_tensor(t, tensor, dev_ptr)) {
            return ggml_cannge_transfer_failed(ctx, plan);
        }
        outputs.push_back(std::move(tensor));
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_cannge_snapshot_indices(ggml_backend_cannge_context * ctx, ggml_cannge_plan * plan,
        const ggml_cannge_plan_io & io, std::vector<std::vector<int64_t>> & sr_slots) {
    // Capture indices before GE can reuse their buffers for later nodes.
    sr_slots.resize(io.set_rows.size());
    for (size_t i = 0; i < io.set_rows.size(); i++) {
        const ggml_cannge_set_rows_op & sr = io.set_rows[i];
        if (sr.in_graph) {
            continue;
        }
        if (!ggml_cannge_read_indices(sr.idx, ctx->stream, sr_slots[i])) {
            GGML_LOG_ERROR("%s: SET_ROWS index snapshot failed\n", __func__);
            return ggml_cannge_transfer_failed(ctx, plan);
        }
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_cannge_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cannge_context * ctx = (ggml_backend_cannge_context *) backend->context;
    if (!ctx->initialized) {
        return GGML_STATUS_FAILED;
    }

    if (!ggml_backend_cannge_graph_env_enabled()) {
        GGML_LOG_ERROR("%s: compiled graph path disabled, set GGML_CANNGE_GRAPH=1 to enable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (!ggml_cannge_check(aclrtSetDevice(ctx->device), "select execution device", __func__)) {
        return GGML_STATUS_FAILED;
    }
    const double dbg_t0 = g_dbg_time.on ? g_dbg_time.now() : 0;

    // static capability pre-check (L0 gate of the fallback ladder)
    const ggml_cannge_options options = ggml_cannge_options::from_env();
    bool needs_compute = false;
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
        const char * reason = nullptr;
        if (!ggml_cannge_supports_op(node, options, &reason)) {
            GGML_LOG_ERROR("%s: %s (%s) rejected: %s\n", __func__, node->name, ggml_op_name(node->op), reason);
            return GGML_STATUS_FAILED;
        }
        if (node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
            node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE) {
            needs_compute = true;
        }
    }
    if (!needs_compute) {
        return GGML_STATUS_SUCCESS;
    }

    std::string err;

    // per-call IO analysis with fresh pointers: the compute context (and its
    // tensors) may be freed between graph_compute calls, so nothing that
    // survives in the plan cache may reference ggml tensors
    ggml_cannge_plan_io io;
    if (!ggml_cannge_plan_analyze(cgraph, io, err, options)) {
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

    ggml_cannge_plan * plan = ggml_cannge_prepare_plan(ctx, cgraph, io, signature, err);
    if (plan == nullptr) {
        return GGML_STATUS_FAILED;
    }
    const double dbg_t2 = g_dbg_time.on ? g_dbg_time.now() : 0;
    // bind all graph IO to the ggml device buffers of this call
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
    inputs.reserve(io.inputs.size());
    outputs.reserve(io.output_tensors.size());
    double dbg_t2a = 0, dbg_t2b = 0;
    if (ggml_cannge_bind_inputs(ctx, plan, io, inputs, dbg_t2a, dbg_t2b) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    std::vector<std::pair<size_t, bool>> staged_ports;
    if (ggml_cannge_bind_outputs(ctx, plan, io, outputs, staged_ports) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    if (getenv("GGML_CANNGE_DEBUG_IO")) {
        for (auto * tensor : io.inputs) {
            if (!ggml_cannge_debug_tensor("in", tensor, ctx->stream)) {
                return GGML_STATUS_FAILED;
            }
        }
    }
    const double dbg_t3 = g_dbg_time.on ? g_dbg_time.now() : 0;

    std::vector<std::vector<int64_t>> sr_slots;
    if (ggml_cannge_snapshot_indices(ctx, plan, io, sr_slots) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
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
        for (auto * tensor : io.output_tensors) {
            if (!ggml_cannge_debug_tensor("out", tensor, ctx->stream)) {
                return GGML_STATUS_FAILED;
            }
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
