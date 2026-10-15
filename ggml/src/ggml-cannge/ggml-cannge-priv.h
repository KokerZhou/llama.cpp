// CANNGE backend private cross-TU declarations (not installed).
// Split out of ggml-cannge.cpp so the backend interface, the op gates and
// the IO copies compile as separate units (note-48 batch G).

#pragma once

#include "ggml-backend-impl.h"

#include <acl/acl.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#define GGML_CANNGE_NAME "CANNGE"

// plan cache bound; HBM budget based eviction per D6 comes later
#define GGML_CANNGE_MAX_PLANS 256

#define ACL_CHECK(expr)                                                                                                          \
    do {                                                                                                                         \
        aclError err = (expr);                                                                                                   \
        if (err != ACL_SUCCESS) {                                                                                                \
            GGML_LOG_ERROR("%s: ACL error %d (%s) at %s:%d\n", __func__, err, aclGetRecentErrMsg(), __FILE__, __LINE__);         \
        }                                                                                                                        \
    } while (0)

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

// ggml-cannge-io.cpp
void ggml_cannge_strided_copy(const ggml_tensor * t, void * dense, bool dense_to_strided, aclrtStream stream);
bool ggml_cannge_reserve_input_arena(void *& device, void *& host, size_t & capacity, size_t size, aclrtStream stream);
struct ggml_cannge_set_rows_op;
bool ggml_cannge_set_rows(const ggml_cannge_set_rows_op & op, const std::vector<int64_t> & slots,
                          const std::vector<char> & data, aclrtStream stream);

// ggml-cannge-gates.cpp
bool ggml_cannge_float_dtype(enum ggml_type type);
bool ggml_backend_cannge_graph_env_enabled();
bool ggml_backend_cannge_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
bool ggml_backend_cannge_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

// ggml-cannge.cpp
const char * ggml_backend_cannge_buffer_type_name(ggml_backend_buffer_type_t buft);
