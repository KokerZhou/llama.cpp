// Shared CANNGE types and diagnostics. Backend context is in ggml-cannge-impl.h.

#pragma once

#include "ggml-backend-impl.h"

#include <acl/acl.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

constexpr const char * GGML_CANNGE_NAME = "CANNGE";

// plan cache bound; HBM budget based eviction per D6 comes later
constexpr size_t GGML_CANNGE_MAX_PLANS = 256;

bool ggml_cannge_check(aclError error, const char * operation, const char * function);

// debug timing accumulators (GGML_CANNGE_DEBUG_TIMING): ms per phase, printed
// every 1000 graph_compute calls; locates where token time goes (note-37)
struct ggml_cannge_dbg_timing {
    bool      on = getenv("GGML_CANNGE_DEBUG_TIMING") != nullptr;
    double    analyze = 0, lookup = 0, sig = 0, find = 0, bind = 0, stage = 0, stage_sync = 0, obind = 0, exec = 0, post = 0, total = 0, cpy_sync = 0;
    double    stage_bytes = 0;
    std::map<std::string, double> stage_names;
    long long calls = 0, cpy_calls = 0, evicts = 0;
    static double now() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    ~ggml_cannge_dbg_timing();
    void dump();
};

extern ggml_cannge_dbg_timing g_dbg_time;

bool ggml_backend_cannge_graph_env_enabled();

// ggml-cannge.cpp
const char * ggml_backend_cannge_buffer_type_name(ggml_backend_buffer_type_t buft);
