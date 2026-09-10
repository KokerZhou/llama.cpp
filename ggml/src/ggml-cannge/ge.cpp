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

#include "ge.h"

#include "ggml-impl.h"

#include <acl/acl.h>
#include <map>
#include <mutex>
#include <set>

// GE header layout: new (<cann>/include/ge, graph, exe_graph) first, old flat
// layout as fallback; values follow the JittorInfer reference config
#if defined(__has_include)
#if __has_include(<ge/ge_api.h>)
#include <ge/ge_api.h>
#include <ge/ge_api_types.h>
#include <ge/ge_error_codes.h>
#else
#include <ge_api.h>
#include <ge_api_types.h>
#include <ge_error_codes.h>
#endif
#else
#include <ge/ge_api.h>
#include <ge/ge_api_types.h>
#include <ge/ge_error_codes.h>
#endif

// process-wide GE init reference count
static std::mutex g_ge_mutex;
static int        g_ge_ref_count = 0;

static bool ge_global_init(int32_t device, std::string & err) {
    std::lock_guard<std::mutex> lock(g_ge_mutex);
    if (g_ge_ref_count > 0) {
        g_ge_ref_count++;
        return true;
    }

    std::map<ge::AscendString, ge::AscendString> config = {
        // TODO(M7): ge.exec.deviceId is global and pins the first device;
        // multi-device support needs per-device GE sessions with a refactor
        { "ge.exec.deviceId", std::to_string(device).c_str() },
        { "ge.graphRunMode", "0" },
        { "ge.tiling_schedule_optimize", "1" },
        { "ge.exec.precision_mode", "allow_fp32_to_fp16" },
        { "ge.exec.reuseZeroCopyMemory", "1" }, // re-tune on 310 measurements
    };
    if (ge::GEInitialize(config) != ge::SUCCESS) {
        err = "GEInitialize failed";
        return false;
    }
    g_ge_ref_count = 1;
    return true;
}

static void ge_global_finalize() {
    std::lock_guard<std::mutex> lock(g_ge_mutex);
    if (g_ge_ref_count > 0 && --g_ge_ref_count == 0) {
        ge::GEFinalize();
    }
}

struct ggml_cannge_ge_context::impl {
    int32_t            device       = -1;
    ge::Session *      session      = nullptr;
    uint32_t           next_graph_id = 1; // graph ids are user assigned, keep them unique
    std::set<uint32_t> graphs;            // added graph ids, for rollback and eviction
};

ggml_cannge_ge_context * ggml_cannge_ge_context::create(int32_t device, std::string & err) {
    std::unique_ptr<ggml_cannge_ge_context> ctx(new ggml_cannge_ge_context);
    ctx->pimpl.reset(new impl);
    ctx->pimpl->device = device;

    if (!ge_global_init(device, err)) {
        return nullptr;
    }

    std::map<ge::AscendString, ge::AscendString> options = {
        { "ge.session_device_id", std::to_string(device).c_str() },
    };
    try {
        ctx->pimpl->session = new ge::Session(options);
    } catch (const std::exception & e) {
        err = std::string("failed to create GE session: ") + e.what();
        ge_global_finalize();
        return nullptr;
    }
    return ctx.release();
}

ggml_cannge_ge_context::~ggml_cannge_ge_context() {
    // order matters: session is deleted before GEFinalize, see docs/cannge D6
    if (pimpl && pimpl->session != nullptr) {
        delete pimpl->session;
        pimpl->session = nullptr;
    }
    ge_global_finalize();
}

bool ggml_cannge_ge_context::add_and_compile(void * graph, uint32_t & graph_id, std::string & err) {
    if (graph == nullptr) {
        err = "null graph";
        return false;
    }

    graph_id = pimpl->next_graph_id++;

    // AddGraphWithCopy keeps the session clear of the caller's ge::Graph, the
    // caller releases the IR once this returns (D6: AddGraph mutates the input)
    ge::Status ret = pimpl->session->AddGraphWithCopy(graph_id, *(ge::Graph *) graph);
    if (ret != ge::SUCCESS) {
        err = "AddGraphWithCopy failed: " + ge::GEGetErrorMsg();
        return false;
    }

    ret = pimpl->session->CompileGraph(graph_id);
    if (ret != ge::SUCCESS) {
        // roll the graph back so no half-added graph stays in the session
        pimpl->session->RemoveGraph(graph_id);
        err = "CompileGraph failed: " + ge::GEGetErrorMsg();
        return false;
    }

    pimpl->graphs.insert(graph_id);
    return true;
}

bool ggml_cannge_ge_context::load(uint32_t graph_id, void * stream, std::string & err) {
    std::map<ge::AscendString, ge::AscendString> options;
    ge::Status ret = pimpl->session->LoadGraph(graph_id, options, stream);
    if (ret != ge::SUCCESS) {
        err = "LoadGraph failed: " + ge::GEGetErrorMsg();
        return false;
    }
    return true;
}

bool ggml_cannge_ge_context::remove(uint32_t graph_id) {
    if (!pimpl->graphs.count(graph_id)) {
        return false;
    }
    ge::Status ret = pimpl->session->RemoveGraph(graph_id);
    if (ret != ge::SUCCESS) {
        GGML_LOG_ERROR("%s: RemoveGraph(%u) failed: %s\n", __func__, graph_id, ge::GEGetErrorMsg().c_str());
        return false;
    }
    pimpl->graphs.erase(graph_id);
    return true;
}

bool ggml_cannge_ge_context::execute(uint32_t graph_id, void * stream, const std::vector<gert::Tensor> & inputs,
                                     std::vector<gert::Tensor> & outputs, std::string & err) {
    ge::Status ret = pimpl->session->ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    if (ret != ge::SUCCESS) {
        err = "ExecuteGraphWithStreamAsync failed: " + ge::GEGetErrorMsg();
        return false;
    }
    return true;
}
