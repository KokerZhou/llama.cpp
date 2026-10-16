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

#include "common.h"
#include "ggml-impl.h"

#include <algorithm>

ggml_cannge_dbg_timing g_dbg_time;

bool ggml_cannge_check(aclError error, const char * operation, const char * function) {
    if (error != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: %s failed with ACL error %d: %s\n", function, operation, error, aclGetRecentErrMsg());
        return false;
    }
    return true;
}

ggml_cannge_dbg_timing::~ggml_cannge_dbg_timing() {
    dump();
}

void ggml_cannge_dbg_timing::dump() {
    if (!on || calls == 0) return;
    fprintf(stderr, "[DBG-TIME] calls=%lld analyze=%.1f sig=%.1f find=%.1f bind=%.1f(stage=%.1f+sync=%.1f+obind=%.1f) exec=%.1f post=%.1f total=%.1f | cpy n=%lld sync=%.1f ms | staged=%.0f MB\n",
            calls, analyze, sig, find, bind, stage, stage_sync, obind, exec, post, total, cpy_calls, cpy_sync,
            stage_bytes / 1e6);
    std::vector<std::pair<double, std::string>> top;
    top.reserve(stage_names.size());
    for (const auto & kv : stage_names) {
        top.emplace_back(kv.second, kv.first);
    }
    std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    for (size_t i = 0; i < top.size() && i < 8; i++) {
        fprintf(stderr, "[DBG-TIME]   staged %-28s %.1f MB\n", top[i].second.c_str(), top[i].first / 1e6);
    }
    fprintf(stderr, "[DBG-TIME] evicts=%lld\n", evicts);
}
