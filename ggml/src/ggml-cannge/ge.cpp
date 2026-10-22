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
#include <filesystem>
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
static std::string g_ge_finalize_error;

static bool ge_global_init(int32_t device, std::string & err) {
    std::lock_guard<std::mutex> lock(g_ge_mutex);
    if (!g_ge_finalize_error.empty()) {
        err = g_ge_finalize_error;
        return false;
    }
    if (g_ge_ref_count > 0) {
        g_ge_ref_count++;
        return true;
    }

    std::map<ge::AscendString, ge::AscendString> config = {
        // Global default; each session selects its device with ge.session_device_id.
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
        const ge::Status status = ge::GEFinalize();
        if (status != ge::SUCCESS) {
            // GE may have partially torn down its global state before returning failure.
            g_ge_finalize_error = "GEFinalize failed (status " + std::to_string(status) + "): " +
                                  ge::GEGetErrorMsg() + "; restart the process before creating another CANNGE session";
            GGML_LOG_ERROR("%s: %s\n", __func__, g_ge_finalize_error.c_str());
        }
    }
}

struct ggml_cannge_ge_context::impl {
    int32_t            device       = -1;
    bool               ge_initialized = false;
    std::unique_ptr<ge::Session> session;
    uint32_t           next_graph_id = 1; // graph ids are user assigned, keep them unique
    std::set<uint32_t> graphs;            // added graph ids, for rollback and eviction
    std::string        cache_dir;         // empty: graph compile cache disabled
    std::map<uint32_t, std::string> graph_keys; // graph_id -> ge.graph_key for load
};

ggml_cannge_ge_context * ggml_cannge_ge_context::create(int32_t device, std::string & err) {
    std::unique_ptr<ggml_cannge_ge_context> ctx(new ggml_cannge_ge_context);
    ctx->pimpl.reset(new impl);
    ctx->pimpl->device = device;

    if (!ge_global_init(device, err)) {
        return nullptr;
    }
    ctx->pimpl->ge_initialized = true;

    std::map<ge::AscendString, ge::AscendString> options = {
        { "ge.session_device_id", std::to_string(device).c_str() },
    };
    // Disk caching is opt-in until cross-run correctness is validated.
    const char * cache_env = getenv("GGML_CANNGE_GE_CACHE_DIR");
    std::string  cache_dir;
    if (cache_env != nullptr) {
        cache_dir = cache_env;
    }
    if (!cache_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        if (ec) {
            GGML_LOG_WARN("%s: GE cache dir %s unusable (%s), compile cache off\n", __func__, cache_dir.c_str(),
                          ec.message().c_str());
        } else {
            options["ge.graph_compiler_cache_dir"] = cache_dir.c_str();
            ctx->pimpl->cache_dir = cache_dir;
        }
    }
    try {
        ctx->pimpl->session.reset(new ge::Session(options));
    } catch (const std::exception & e) {
        err = std::string("failed to create GE session: ") + e.what();
        return nullptr;
    }
    return ctx.release();
}

ggml_cannge_ge_context::~ggml_cannge_ge_context() {
    if (pimpl) {
        // Release the session before its reference to the GE runtime.
        pimpl->session.reset();
        if (pimpl->ge_initialized) {
            ge_global_finalize();
        }
    }
}

bool ggml_cannge_ge_context::add_and_compile(void * graph, uint32_t & graph_id, std::string & err,
                                             const std::string & graph_key) {
    if (graph == nullptr) {
        err = "null graph";
        return false;
    }

    graph_id = pimpl->next_graph_id++;

    std::map<ge::AscendString, ge::AscendString> options;
    if (!pimpl->cache_dir.empty() && !graph_key.empty()) {
        options["ge.graph_key"] = graph_key.c_str();
    }

    // AddGraphWithCopy keeps the session clear of the caller's ge::Graph, the
    // caller releases the IR once this returns (D6: AddGraph mutates the input)
    ge::Status ret = pimpl->session->AddGraphWithCopy(graph_id, *(ge::Graph *) graph, options);
    if (ret != ge::SUCCESS) {
        err = "AddGraphWithCopy failed: " + ge::GEGetErrorMsg();
        return false;
    }

    pimpl->graphs.insert(graph_id);
    pimpl->graph_keys[graph_id] = graph_key;
    ret = pimpl->session->CompileGraph(graph_id);
    if (ret != ge::SUCCESS) {
        err = "CompileGraph failed: " + ge::GEGetErrorMsg();
        // Keep failed removals tracked so eviction can retry them.
        remove(graph_id);
        return false;
    }

    return true;
}

bool ggml_cannge_ge_context::load(uint32_t graph_id, void * stream, std::string & err) {
    std::map<ge::AscendString, ge::AscendString> options;
    const auto key = pimpl->graph_keys.find(graph_id);
    if (!pimpl->cache_dir.empty() && key != pimpl->graph_keys.end() && !key->second.empty()) {
        // the runner reads user_graph_key at load to restore the cached execution order
        options["ge.graph_key"] = key->second.c_str();
    }
    ge::Status ret = pimpl->session->LoadGraph(graph_id, options, stream);
    if (ret != ge::SUCCESS) {
        err = "LoadGraph failed: " + ge::GEGetErrorMsg();
        return false;
    }
    return true;
}

bool ggml_cannge_ge_context::remove(uint32_t graph_id) {
    if (!pimpl->graphs.count(graph_id)) {
        return true;
    }
    ge::Status ret = pimpl->session->RemoveGraph(graph_id);
    if (ret != ge::SUCCESS) {
        GGML_LOG_ERROR("%s: RemoveGraph(%u) failed: %s\n", __func__, graph_id, ge::GEGetErrorMsg().c_str());
        return false;
    }
    pimpl->graphs.erase(graph_id);
    pimpl->graph_keys.erase(graph_id);
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
